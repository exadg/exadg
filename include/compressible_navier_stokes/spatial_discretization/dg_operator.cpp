/*
 * dg_operator.cpp
 *
 *  Created on: May 3, 2019
 *      Author: fehn
 */

#include "dg_operator.h"

namespace CompNS
{
template<int dim, typename Number>
DGOperator<dim, Number>::DGOperator(parallel::TriangulationBase<dim> const & triangulation,
                                    InputParameters const &              param_in,
                                    std::shared_ptr<Postprocessor>       postprocessor_in)
  : dealii::Subscriptor(),
    param(param_in),
    fe(new FESystem<dim>(FE_DGQ<dim>(param_in.degree), dim + 2)),
    fe_vector(new FESystem<dim>(FE_DGQ<dim>(param_in.degree), dim)),
    fe_scalar(param_in.degree),
    mapping_degree(1),
    n_q_points_conv(param.degree + 1),
    n_q_points_visc(param.degree + 1),
    dof_handler(triangulation),
    dof_handler_vector(triangulation),
    dof_handler_scalar(triangulation),
    postprocessor(postprocessor_in),
    wall_time_operator_evaluation(0.0)
{
  // Mapping
  if(param.mapping == MappingType::Affine)
  {
    mapping_degree = 1;
  }
  else if(param.mapping == MappingType::Isoparametric)
  {
    mapping_degree = param.degree;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented"));
  }

  mapping.reset(new MappingQGeneric<dim>(mapping_degree));

  // Quadrature rule
  if(param.n_q_points_convective == QuadratureRule::Standard)
    n_q_points_conv = param.degree + 1;
  else if(param.n_q_points_convective == QuadratureRule::Overintegration32k)
    n_q_points_conv = param.degree + (param.degree + 2) / 2;
  else if(param.n_q_points_convective == QuadratureRule::Overintegration2k)
    n_q_points_conv = 2 * param.degree + 1;
  else
    AssertThrow(false, ExcMessage("Specified quadrature rule is not implemented."));

  if(param.n_q_points_viscous == QuadratureRule::Standard)
    n_q_points_visc = param.degree + 1;
  else if(param.n_q_points_viscous == QuadratureRule::Overintegration32k)
    n_q_points_visc = param.degree + (param.degree + 2) / 2;
  else if(param.n_q_points_viscous == QuadratureRule::Overintegration2k)
    n_q_points_visc = 2 * param.degree + 1;
  else
    AssertThrow(false, ExcMessage("Specified quadrature rule is not implemented."));
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::setup(
  std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_density_in,
  std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_velocity_in,
  std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_pressure_in,
  std::shared_ptr<BoundaryDescriptorEnergy<dim>> boundary_descriptor_energy_in,
  std::shared_ptr<FieldFunctions<dim>>           field_functions_in)
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Setup compressible Navier-Stokes DG operator ..." << std::endl;

  boundary_descriptor_density  = boundary_descriptor_density_in;
  boundary_descriptor_velocity = boundary_descriptor_velocity_in;
  boundary_descriptor_pressure = boundary_descriptor_pressure_in;
  boundary_descriptor_energy   = boundary_descriptor_energy_in;
  field_functions              = field_functions_in;

  create_dofs();

  initialize_matrix_free();

  setup_operators();

  setup_postprocessor();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
types::global_dof_index
DGOperator<dim, Number>::get_number_of_dofs() const
{
  return dof_handler.n_dofs();
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::initialize_dof_vector(VectorType & src) const
{
  matrix_free.initialize_dof_vector(src, dof_index_all);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::initialize_dof_vector_scalar(VectorType & src) const
{
  matrix_free.initialize_dof_vector(src, dof_index_scalar);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::initialize_dof_vector_dim_components(VectorType & src) const
{
  matrix_free.initialize_dof_vector(src, dof_index_vector);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::prescribe_initial_conditions(VectorType & src, double const time) const
{
  this->field_functions->initial_solution->set_time(time);

  // This is necessary if Number == float
  typedef LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

  VectorTypeDouble src_double;
  src_double = src;

  VectorTools::interpolate(*mapping,
                           dof_handler,
                           *(this->field_functions->initial_solution),
                           src_double);

  src = src_double;
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::evaluate(VectorType & dst, VectorType const & src, Number const time) const
{
  Timer timer;
  timer.restart();

  evaluate_convective_and_viscous(dst, src, time);

  // shift viscous and convective terms to the right-hand side of the equation
  dst *= -1.0;

  // body force term
  if(param.right_hand_side == true)
  {
    body_force_operator.evaluate_add(dst, src, time);
  }

  // apply inverse mass matrix
  inverse_mass_all.apply(dst, dst);

  wall_time_operator_evaluation += timer.wall_time();
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::evaluate_convective(VectorType &       dst,
                                             VectorType const & src,
                                             Number const       time) const
{
  if(param.equation_type == EquationType::Euler ||
     param.equation_type == EquationType::NavierStokes)
  {
    convective_operator.evaluate(dst, src, time);
  }
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::evaluate_viscous(VectorType &       dst,
                                          VectorType const & src,
                                          Number const       time) const
{
  if(param.equation_type == EquationType::NavierStokes)
  {
    viscous_operator.evaluate(dst, src, time);
  }
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::evaluate_convective_and_viscous(VectorType &       dst,
                                                         VectorType const & src,
                                                         Number const       time) const
{
  if(param.use_combined_operator == true)
  {
    // viscous and convective terms
    combined_operator.evaluate(dst, src, time);
  }
  else // apply operators separately
  {
    // set dst to zero
    dst = 0.0;

    // viscous operator
    if(param.equation_type == EquationType::NavierStokes)
    {
      viscous_operator.evaluate_add(dst, src, time);
    }

    // convective operator
    if(param.equation_type == EquationType::Euler ||
       param.equation_type == EquationType::NavierStokes)
    {
      convective_operator.evaluate_add(dst, src, time);
    }
  }
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::apply_inverse_mass(VectorType & dst, VectorType const & src) const
{
  // apply inverse mass matrix
  inverse_mass_all.apply(dst, src);
}

template<int dim, typename Number>
MatrixFree<dim, Number> const &
DGOperator<dim, Number>::get_matrix_free() const
{
  return matrix_free;
}

template<int dim, typename Number>
Mapping<dim> const &
DGOperator<dim, Number>::get_mapping() const
{
  return *mapping;
}

template<int dim, typename Number>
FESystem<dim> const &
DGOperator<dim, Number>::get_fe() const
{
  return *fe;
}

template<int dim, typename Number>
DoFHandler<dim> const &
DGOperator<dim, Number>::get_dof_handler() const
{
  return dof_handler;
}

template<int dim, typename Number>
DoFHandler<dim> const &
DGOperator<dim, Number>::get_dof_handler_scalar() const
{
  return dof_handler_scalar;
}

template<int dim, typename Number>
DoFHandler<dim> const &
DGOperator<dim, Number>::get_dof_handler_vector() const
{
  return dof_handler_vector;
}

template<int dim, typename Number>
unsigned int
DGOperator<dim, Number>::get_dof_index_vector() const
{
  return dof_index_vector;
}

template<int dim, typename Number>
unsigned int
DGOperator<dim, Number>::get_dof_index_scalar() const
{
  return dof_index_scalar;
}

template<int dim, typename Number>
unsigned int
DGOperator<dim, Number>::get_quad_index_standard() const
{
  return quad_index_standard;
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::compute_pressure(VectorType & dst, VectorType const & src) const
{
  p_u_T_calculator.compute_pressure(dst, src);
  inverse_mass_scalar.apply(dst, dst);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::compute_velocity(VectorType & dst, VectorType const & src) const
{
  p_u_T_calculator.compute_velocity(dst, src);
  inverse_mass_vector.apply(dst, dst);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::compute_temperature(VectorType & dst, VectorType const & src) const
{
  p_u_T_calculator.compute_temperature(dst, src);
  inverse_mass_scalar.apply(dst, dst);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::compute_vorticity(VectorType & dst, VectorType const & src) const
{
  vorticity_calculator.compute_vorticity(dst, src);
  inverse_mass_vector.apply(dst, dst);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::compute_divergence(VectorType & dst, VectorType const & src) const
{
  divergence_calculator.compute_divergence(dst, src);
  inverse_mass_scalar.apply(dst, dst);
}

template<int dim, typename Number>
double
DGOperator<dim, Number>::get_wall_time_operator_evaluation() const
{
  return wall_time_operator_evaluation;
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::do_postprocessing(VectorType const & solution,
                                           double const       time,
                                           int const          time_step_number) const
{
  postprocessor->do_postprocessing(solution, time, time_step_number);
}

template<int dim, typename Number>
double
DGOperator<dim, Number>::calculate_minimum_element_length() const
{
  return calculate_minimum_vertex_distance(dof_handler.get_triangulation());
}

template<int dim, typename Number>
unsigned int
DGOperator<dim, Number>::get_polynomial_degree() const
{
  return param.degree;
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::create_dofs()
{
  // enumerate degrees of freedom
  dof_handler.distribute_dofs(*fe);
  dof_handler_vector.distribute_dofs(*fe_vector);
  dof_handler_scalar.distribute_dofs(fe_scalar);

  unsigned int ndofs_per_cell = Utilities::pow(param.degree + 1, dim) * (dim + 2);

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  pcout << std::endl
        << "Discontinuous Galerkin finite element discretization:" << std::endl
        << std::endl;

  print_parameter(pcout, "degree of 1D polynomials", param.degree);
  print_parameter(pcout, "number of dofs per cell", ndofs_per_cell);
  print_parameter(pcout, "number of dofs (total)", dof_handler.n_dofs());
  print_parameter(pcout, "number of 1D q-points (std)", param.degree + 1);
  print_parameter(pcout, "number of 1D q-points (over-conv)", n_q_points_conv);
  print_parameter(pcout, "number of 1D q-points (over-vis)", n_q_points_visc);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::initialize_matrix_free()
{
  // quadratures used to perform integrals
  std::vector<Quadrature<1>> quadratures;
  quadratures.resize(static_cast<typename std::underlying_type<QuadratureSelector>::type>(
    QuadratureSelector::n_variants));
  quadratures[static_cast<typename std::underlying_type<QuadratureSelector>::type>(
    QuadratureSelector::standard)]             = QGauss<1>(param.degree + 1);
  quadratures[static_cast<typename std::underlying_type<QuadratureSelector>::type>(
    QuadratureSelector::overintegration_conv)] = QGauss<1>(n_q_points_conv);
  quadratures[static_cast<typename std::underlying_type<QuadratureSelector>::type>(
    QuadratureSelector::overintegration_vis)]  = QGauss<1>(n_q_points_visc);

  // dof handler
  std::vector<const DoFHandler<dim> *> dof_handler_vec;
  dof_handler_vec.resize(static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
    DofHandlerSelector::n_variants));
  dof_handler_vec[dof_index_all]    = &dof_handler;
  dof_handler_vec[dof_index_vector] = &dof_handler_vector;
  dof_handler_vec[dof_index_scalar] = &dof_handler_scalar;

  // initialize matrix_free_data
  typename MatrixFree<dim, Number>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
    MatrixFree<dim, Number>::AdditionalData::partition_partition;

  additional_data.mapping_update_flags =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
     update_values);
  additional_data.mapping_update_flags_boundary_faces |= update_quadrature_points;
  additional_data.mapping_update_flags_inner_faces |= update_quadrature_points;

  // constraints
  std::vector<const AffineConstraints<double> *> constraint_matrix_vec;
  constraint_matrix_vec.resize(static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
    DofHandlerSelector::n_variants));
  AffineConstraints<double> constraint;
  constraint.close();
  constraint_matrix_vec[dof_index_all]    = &constraint;
  constraint_matrix_vec[dof_index_vector] = &constraint;
  constraint_matrix_vec[dof_index_scalar] = &constraint;

  matrix_free.reinit(
    *mapping, dof_handler_vec, constraint_matrix_vec, quadratures, additional_data);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::setup_operators()
{
  // mass matrix operator
  MassMatrixOperatorData mass_matrix_operator_data;
  mass_matrix_operator_data.dof_index  = dof_index_all;
  mass_matrix_operator_data.quad_index = quad_index_standard;
  mass_matrix_operator.initialize(matrix_free, mass_matrix_operator_data);

  // inverse mass matrix operator
  inverse_mass_all.initialize(matrix_free, param.degree, dof_index_all, quad_index_standard);
  inverse_mass_vector.initialize(matrix_free, param.degree, dof_index_vector, quad_index_standard);
  inverse_mass_scalar.initialize(matrix_free, param.degree, dof_index_scalar, quad_index_standard);

  // body force operator
  BodyForceOperatorData<dim> body_force_operator_data;
  body_force_operator_data.dof_index  = dof_index_all;
  body_force_operator_data.quad_index = quad_index_standard;
  body_force_operator_data.rhs_rho    = field_functions->right_hand_side_density;
  body_force_operator_data.rhs_u      = field_functions->right_hand_side_velocity;
  body_force_operator_data.rhs_E      = field_functions->right_hand_side_energy;
  body_force_operator.initialize(matrix_free, body_force_operator_data);

  // convective operator
  ConvectiveOperatorData<dim> convective_operator_data;
  convective_operator_data.dof_index             = dof_index_all;
  convective_operator_data.quad_index            = quad_index_overintegration_conv;
  convective_operator_data.bc_rho                = boundary_descriptor_density;
  convective_operator_data.bc_u                  = boundary_descriptor_velocity;
  convective_operator_data.bc_p                  = boundary_descriptor_pressure;
  convective_operator_data.bc_E                  = boundary_descriptor_energy;
  convective_operator_data.heat_capacity_ratio   = param.heat_capacity_ratio;
  convective_operator_data.specific_gas_constant = param.specific_gas_constant;
  convective_operator.initialize(matrix_free, convective_operator_data);

  // viscous operator
  ViscousOperatorData<dim> viscous_operator_data;
  viscous_operator_data.dof_index             = dof_index_all;
  viscous_operator_data.quad_index            = quad_index_overintegration_vis;
  viscous_operator_data.degree                = param.degree;
  viscous_operator_data.IP_factor             = param.IP_factor;
  viscous_operator_data.dynamic_viscosity     = param.dynamic_viscosity;
  viscous_operator_data.reference_density     = param.reference_density;
  viscous_operator_data.thermal_conductivity  = param.thermal_conductivity;
  viscous_operator_data.heat_capacity_ratio   = param.heat_capacity_ratio;
  viscous_operator_data.specific_gas_constant = param.specific_gas_constant;
  viscous_operator_data.bc_rho                = boundary_descriptor_density;
  viscous_operator_data.bc_u                  = boundary_descriptor_velocity;
  viscous_operator_data.bc_E                  = boundary_descriptor_energy;
  viscous_operator.initialize(*mapping, matrix_free, viscous_operator_data);

  if(param.use_combined_operator == true)
  {
    AssertThrow(param.n_q_points_convective == param.n_q_points_viscous,
                ExcMessage("Use the same number of quadrature points for convective term "
                           "and viscous term in case of combined operator."));

    CombinedOperatorData<dim> combined_operator_data;
    combined_operator_data.dof_index  = dof_index_all;
    combined_operator_data.quad_index = quad_index_overintegration_vis;
    combined_operator_data.bc_rho     = boundary_descriptor_density;
    combined_operator_data.bc_u       = boundary_descriptor_velocity;
    combined_operator_data.bc_p       = boundary_descriptor_pressure;
    combined_operator_data.bc_E       = boundary_descriptor_energy;

    combined_operator.initialize(matrix_free,
                                 combined_operator_data,
                                 convective_operator,
                                 viscous_operator);
  }

  // calculators
  p_u_T_calculator.initialize(matrix_free,
                              dof_index_all,
                              dof_index_vector,
                              dof_index_scalar,
                              quad_index_l2_projections,
                              param.heat_capacity_ratio,
                              param.specific_gas_constant);
  vorticity_calculator.initialize(matrix_free, dof_index_vector, quad_index_standard);
  divergence_calculator.initialize(matrix_free,
                                   dof_index_vector,
                                   dof_index_scalar,
                                   quad_index_standard);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::setup_postprocessor()
{
  postprocessor->setup(*this);
}

template class DGOperator<2, float>;
template class DGOperator<2, double>;

template class DGOperator<3, float>;
template class DGOperator<3, double>;

} // namespace CompNS
