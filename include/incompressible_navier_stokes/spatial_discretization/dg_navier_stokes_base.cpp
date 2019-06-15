/*
 * dg_navier_stokes_base.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include "dg_navier_stokes_base.h"

#include "../../poisson/preconditioner/multigrid_preconditioner.h"

namespace IncNS
{
template<int dim, typename Number>
DGNavierStokesBase<dim, Number>::DGNavierStokesBase(
  parallel::Triangulation<dim> const & triangulation,
  InputParameters const &              parameters_in,
  std::shared_ptr<Postprocessor>       postprocessor_in)
  : param(parameters_in),
    fe_u(new FESystem<dim>(FE_DGQ<dim>(param.degree_u), dim)),
    fe_p(param.get_degree_p()),
    fe_u_scalar(param.degree_u),
    mapping_degree(1),
    dof_handler_u(triangulation),
    dof_handler_p(triangulation),
    dof_handler_u_scalar(triangulation),
    dof_index_first_point(0),
    postprocessor(postprocessor_in),
    pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
{
  if(param.mapping == MappingType::Affine)
  {
    mapping_degree = 1;
  }
  else if(param.mapping == MappingType::Isoparametric)
  {
    mapping_degree = param.degree_u;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented"));
  }

  mapping.reset(new MappingQGeneric<dim>(mapping_degree));
}

template<int dim, typename Number>
DGNavierStokesBase<dim, Number>::~DGNavierStokesBase()
{
  matrix_free.clear();
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::setup(
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> const
                                                  periodic_face_pairs_in,
  std::shared_ptr<BoundaryDescriptorU<dim>> const boundary_descriptor_velocity_in,
  std::shared_ptr<BoundaryDescriptorP<dim>> const boundary_descriptor_pressure_in,
  std::shared_ptr<FieldFunctions<dim>> const      field_functions_in)
{
  pcout << std::endl << "Setup Navier-Stokes operator ..." << std::endl << std::flush;

  periodic_face_pairs          = periodic_face_pairs_in;
  boundary_descriptor_velocity = boundary_descriptor_velocity_in;
  boundary_descriptor_pressure = boundary_descriptor_pressure_in;
  field_functions              = field_functions_in;

  initialize_boundary_descriptor_laplace();

  initialize_dof_handler();

  // depending on DoFHandler
  initialize_matrix_free();

  // depending on MatrixFree
  initialize_operators();

  // depending on MatrixFree
  setup_projection_operator();

  // turbulence model
  if(param.use_turbulence_model == true)
  {
    // Depending on MatrixFree, Mapping, ViscousOperator
    initialize_turbulence_model();
  }

  // depending on MatrixFree
  initialize_calculators_for_derived_quantities();

  if(param.pure_dirichlet_bc == true &&
     param.adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalSolutionInPoint)
  {
    initialization_pure_dirichlet_bc();
  }

  // depending on DoFHandler, Mapping, MatrixFree
  initialize_postprocessor();

  pcout << std::endl << "... done!" << std::endl << std::flush;
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::initialize_boundary_descriptor_laplace()
{
  boundary_descriptor_laplace.reset(new Poisson::BoundaryDescriptor<dim>());

  // Dirichlet BCs for pressure
  boundary_descriptor_laplace->dirichlet_bc = boundary_descriptor_pressure->dirichlet_bc;

  // Neumann BCs for pressure
  // Note: for the dual splitting scheme, neumann_bc contains functions corresponding
  //       to dudt term required in pressure Neumann boundary condition.
  // Here: set this functions explicitly to ZeroFunction when filling the boundary
  //       descriptor for the Laplace operator because these inhomogeneous
  //       boundary conditions have to be implemented separately
  //       and can not be applied by the Laplace operator.
  for(typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::const_iterator it =
        boundary_descriptor_pressure->neumann_bc.begin();
      it != boundary_descriptor_pressure->neumann_bc.end();
      ++it)
  {
    std::shared_ptr<Function<dim>> zero_function;
    zero_function.reset(new Functions::ZeroFunction<dim>(1));
    boundary_descriptor_laplace->neumann_bc.insert(
      std::pair<types::boundary_id, std::shared_ptr<Function<dim>>>(it->first, zero_function));
  }
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::initialize_dof_handler()
{
  // enumerate degrees of freedom
  dof_handler_u.distribute_dofs(*fe_u);
  dof_handler_u.distribute_mg_dofs();
  dof_handler_p.distribute_dofs(fe_p);
  dof_handler_p.distribute_mg_dofs();
  dof_handler_u_scalar.distribute_dofs(fe_u_scalar);
  dof_handler_u_scalar.distribute_mg_dofs(); // probably, we don't need this

  unsigned int const ndofs_per_cell_velocity = Utilities::pow(param.degree_u + 1, dim) * dim;
  unsigned int const ndofs_per_cell_pressure = Utilities::pow(param.get_degree_p() + 1, dim);

  pcout << std::endl
        << "Discontinuous Galerkin finite element discretization:" << std::endl
        << std::endl
        << std::flush;

  pcout << "Velocity:" << std::endl;
  print_parameter(pcout, "degree of 1D polynomials", param.degree_u);
  print_parameter(pcout, "number of dofs per cell", ndofs_per_cell_velocity);
  print_parameter(pcout, "number of dofs (total)", dof_handler_u.n_dofs());

  pcout << "Pressure:" << std::endl;
  print_parameter(pcout, "degree of 1D polynomials", param.get_degree_p());
  print_parameter(pcout, "number of dofs per cell", ndofs_per_cell_pressure);
  print_parameter(pcout, "number of dofs (total)", dof_handler_p.n_dofs());

  pcout << "Velocity and pressure:" << std::endl;
  print_parameter(pcout,
                  "number of dofs per cell",
                  ndofs_per_cell_velocity + ndofs_per_cell_pressure);
  print_parameter(pcout, "number of dofs (total)", dof_handler_u.n_dofs() + dof_handler_p.n_dofs());

  pcout << std::flush;
}

template<int dim, typename Number>
types::global_dof_index
DGNavierStokesBase<dim, Number>::get_number_of_dofs() const
{
  return dof_handler_u.n_dofs() + dof_handler_p.n_dofs();
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::initialize_matrix_free()
{
  // initialize matrix_free_data
  typename MatrixFree<dim, Number>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
    MatrixFree<dim, Number>::AdditionalData::partition_partition;
  additional_data.mapping_update_flags =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
     update_values);

  additional_data.mapping_update_flags_inner_faces =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
     update_values);

  additional_data.mapping_update_flags_boundary_faces =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
     update_values);

  if(param.use_cell_based_face_loops)
  {
    auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> const *>(
      &dof_handler_u.get_triangulation());
    Categorization::do_cell_based_loops(*tria, additional_data);
  }

  // dofhandler
  std::vector<const DoFHandler<dim> *> dof_handler_vec;

  dof_handler_vec.resize(static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
    DofHandlerSelector::n_variants));
  dof_handler_vec[dof_index_u]        = &dof_handler_u;
  dof_handler_vec[dof_index_p]        = &dof_handler_p;
  dof_handler_vec[dof_index_u_scalar] = &dof_handler_u_scalar;

  // constraint
  std::vector<const AffineConstraints<double> *> constraint_matrix_vec;
  constraint_matrix_vec.resize(static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
    DofHandlerSelector::n_variants));
  AffineConstraints<double> constraint_u, constraint_p, constraint_u_scalar;
  constraint_u.close();
  constraint_p.close();
  constraint_u_scalar.close();
  constraint_matrix_vec[dof_index_u]        = &constraint_u;
  constraint_matrix_vec[dof_index_p]        = &constraint_p;
  constraint_matrix_vec[dof_index_u_scalar] = &constraint_u_scalar;

  // quadrature
  std::vector<Quadrature<1>> quadratures;

  // resize quadratures
  quadratures.resize(static_cast<typename std::underlying_type<QuadratureSelector>::type>(
    QuadratureSelector::n_variants));
  // velocity
  quadratures[quad_index_u] = QGauss<1>(param.degree_u + 1);
  // pressure
  quadratures[quad_index_p] = QGauss<1>(param.get_degree_p() + 1);
  // exact integration of nonlinear convective term
  quadratures[quad_index_u_nonlinear] = QGauss<1>(param.degree_u + (param.degree_u + 2) / 2);

  // reinit
  matrix_free.reinit(
    *mapping, dof_handler_vec, constraint_matrix_vec, quadratures, additional_data);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::initialize_operators()
{
  // mass matrix operator
  mass_matrix_operator_data.dof_index  = dof_index_u;
  mass_matrix_operator_data.quad_index = quad_index_u;
  mass_matrix_operator.reinit(matrix_free, mass_matrix_operator_data);

  // inverse mass matrix operator
  inverse_mass_velocity.initialize(matrix_free, param.degree_u, dof_index_u, quad_index_u);

  // inverse mass matrix operator velocity scalar
  inverse_mass_velocity_scalar.initialize(matrix_free,
                                          param.degree_u,
                                          dof_index_u_scalar,
                                          quad_index_u);

  // body force operator
  RHSOperatorData<dim> rhs_data;
  rhs_data.dof_index     = dof_index_u;
  rhs_data.quad_index    = quad_index_u;
  rhs_data.kernel_data.f = field_functions->right_hand_side;
  rhs_operator.reinit(matrix_free, rhs_data);

  // gradient operator
  gradient_operator_data.dof_index_velocity   = dof_index_u;
  gradient_operator_data.dof_index_pressure   = dof_index_p;
  gradient_operator_data.quad_index           = quad_index_u;
  gradient_operator_data.integration_by_parts = param.gradp_integrated_by_parts;
  gradient_operator_data.use_boundary_data    = param.gradp_use_boundary_data;
  gradient_operator_data.bc                   = boundary_descriptor_pressure;
  gradient_operator.reinit(matrix_free, gradient_operator_data);

  // divergence operator
  divergence_operator_data.dof_index_velocity   = dof_index_u;
  divergence_operator_data.dof_index_pressure   = dof_index_p;
  divergence_operator_data.quad_index           = quad_index_u;
  divergence_operator_data.integration_by_parts = param.divu_integrated_by_parts;
  divergence_operator_data.use_boundary_data    = param.divu_use_boundary_data;
  divergence_operator_data.bc                   = boundary_descriptor_velocity;
  divergence_operator.reinit(matrix_free, divergence_operator_data);

  // convective operator
  convective_operator_data.formulation          = param.formulation_convective_term;
  convective_operator_data.dof_index            = dof_index_u;
  convective_operator_data.quad_index           = quad_index_u_nonlinear;
  convective_operator_data.upwind_factor        = param.upwind_factor;
  convective_operator_data.bc                   = boundary_descriptor_velocity;
  convective_operator_data.use_outflow_bc       = param.use_outflow_bc_convective_term;
  convective_operator_data.type_dirichlet_bc    = param.type_dirichlet_bc_convective;
  convective_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
  convective_operator.initialize(matrix_free, convective_operator_data);

  // viscous operator
  viscous_operator_data.kernel_data.degree                   = param.degree_u;
  viscous_operator_data.kernel_data.degree_mapping           = mapping_degree;
  viscous_operator_data.kernel_data.IP_factor                = param.IP_factor_viscous;
  viscous_operator_data.kernel_data.viscosity                = param.viscosity;
  viscous_operator_data.kernel_data.formulation_viscous_term = param.formulation_viscous_term;
  viscous_operator_data.kernel_data.penalty_term_div_formulation =
    param.penalty_term_div_formulation;
  viscous_operator_data.kernel_data.IP_formulation        = param.IP_formulation_viscous;
  viscous_operator_data.kernel_data.viscosity_is_variable = param.use_turbulence_model;
  viscous_operator_data.bc                                = boundary_descriptor_velocity;
  viscous_operator_data.dof_index                         = dof_index_u;
  viscous_operator_data.quad_index                        = quad_index_u;
  viscous_operator_data.use_cell_based_loops              = param.use_cell_based_face_loops;
  AffineConstraints<double> constraint_dummy;
  constraint_dummy.close();
  viscous_operator.reinit(matrix_free, constraint_dummy, viscous_operator_data);
}


template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::initialize_turbulence_model()
{
  viscosity_coefficients.reset(new VariableCoefficients<dim, Number>());
  // allocate vectors for variable coefficients and initialize with constant viscosity
  viscosity_coefficients->initialize(matrix_free, param.degree_u, param.viscosity);

  // initialize turbulence model
  TurbulenceModelData model_data;
  model_data.turbulence_model    = param.turbulence_model;
  model_data.constant            = param.turbulence_model_constant;
  model_data.kinematic_viscosity = param.viscosity;
  model_data.dof_index           = dof_index_u;
  model_data.quad_index          = quad_index_u;
  model_data.degree              = param.degree_u;
  turbulence_model.initialize(matrix_free, *mapping, viscosity_coefficients, model_data);

  // set pointer so that viscous operator can access the coefficients
  viscous_operator.set_viscosity_coefficients_ptr(viscosity_coefficients);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::initialize_calculators_for_derived_quantities()
{
  vorticity_calculator.initialize(matrix_free, dof_index_u, quad_index_u);
  divergence_calculator.initialize(matrix_free, dof_index_u, dof_index_u_scalar, quad_index_u);
  velocity_magnitude_calculator.initialize(matrix_free,
                                           dof_index_u,
                                           dof_index_u_scalar,
                                           quad_index_u);
  q_criterion_calculator.initialize(matrix_free, dof_index_u, dof_index_u_scalar, quad_index_u);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::initialization_pure_dirichlet_bc()
{
  dof_index_first_point = 0;
  for(unsigned int d = 0; d < dim; ++d)
    first_point[d] = 0.0;

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    typename DoFHandler<dim>::active_cell_iterator first_cell;
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_p.begin_active(),
                                                   endc = dof_handler_p.end();

    bool processor_has_active_cells = false;
    for(; cell != endc; ++cell)
    {
      if(cell->is_locally_owned())
      {
        first_cell = cell;

        processor_has_active_cells = true;
        break;
      }
    }

    AssertThrow(processor_has_active_cells == true,
                ExcMessage("No active cells on Processor with ID=0"));

    FEValues<dim> fe_values(dof_handler_p.get_fe(),
                            Quadrature<dim>(dof_handler_p.get_fe().get_unit_support_points()),
                            update_quadrature_points);

    fe_values.reinit(first_cell);

    first_point = fe_values.quadrature_point(0);
    std::vector<types::global_dof_index> dof_indices(dof_handler_p.get_fe().dofs_per_cell);
    first_cell->get_dof_indices(dof_indices);
    dof_index_first_point = dof_indices[0];
  }
  dof_index_first_point = Utilities::MPI::sum(dof_index_first_point, MPI_COMM_WORLD);
  for(unsigned int d = 0; d < dim; ++d)
  {
    first_point[d] = Utilities::MPI::sum(first_point[d], MPI_COMM_WORLD);
  }
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::initialize_postprocessor()
{
  postprocessor->setup(*this);
}

template<int dim, typename Number>
MatrixFree<dim, Number> const &
DGNavierStokesBase<dim, Number>::get_matrix_free() const
{
  return matrix_free;
}

template<int dim, typename Number>
unsigned int
DGNavierStokesBase<dim, Number>::get_dof_index_velocity() const
{
  return dof_index_u;
}

template<int dim, typename Number>
unsigned int
DGNavierStokesBase<dim, Number>::get_dof_index_velocity_scalar() const
{
  return dof_index_u_scalar;
}

template<int dim, typename Number>
unsigned int
DGNavierStokesBase<dim, Number>::get_quad_index_velocity_linear() const
{
  return quad_index_u;
}

template<int dim, typename Number>
unsigned int
DGNavierStokesBase<dim, Number>::get_quad_index_velocity_nonlinear() const
{
  return quad_index_u_nonlinear;
}

template<int dim, typename Number>
unsigned int
DGNavierStokesBase<dim, Number>::get_dof_index_pressure() const
{
  return dof_index_p;
}

template<int dim, typename Number>
unsigned int
DGNavierStokesBase<dim, Number>::get_quad_index_pressure() const
{
  return quad_index_p;
}

template<int dim, typename Number>
Mapping<dim> const &
DGNavierStokesBase<dim, Number>::get_mapping() const
{
  return *mapping;
}

template<int dim, typename Number>
FESystem<dim> const &
DGNavierStokesBase<dim, Number>::get_fe_u() const
{
  return *fe_u;
}

template<int dim, typename Number>
FE_DGQ<dim> const &
DGNavierStokesBase<dim, Number>::get_fe_p() const
{
  return fe_p;
}

template<int dim, typename Number>
DoFHandler<dim> const &
DGNavierStokesBase<dim, Number>::get_dof_handler_u() const
{
  return dof_handler_u;
}

template<int dim, typename Number>
DoFHandler<dim> const &
DGNavierStokesBase<dim, Number>::get_dof_handler_u_scalar() const
{
  return dof_handler_u_scalar;
}

template<int dim, typename Number>
DoFHandler<dim> const &
DGNavierStokesBase<dim, Number>::get_dof_handler_p() const
{
  return dof_handler_p;
}

template<int dim, typename Number>
double
DGNavierStokesBase<dim, Number>::get_viscosity() const
{
  return param.viscosity;
}

template<int dim, typename Number>
VectorizedArray<Number>
DGNavierStokesBase<dim, Number>::get_viscosity_boundary_face(unsigned int const face,
                                                             unsigned int const q) const
{
  VectorizedArray<Number> viscosity = make_vectorized_array<Number>(get_viscosity());

  bool const viscosity_is_variable = param.use_turbulence_model;
  if(viscosity_is_variable)
    viscosity_coefficients->get_coefficient_face(face, q);

  return viscosity;
}

template<int dim, typename Number>
MassMatrixOperatorData const &
DGNavierStokesBase<dim, Number>::get_mass_matrix_operator_data() const
{
  return mass_matrix_operator_data;
}

template<int dim, typename Number>
ViscousOperatorData<dim> const &
DGNavierStokesBase<dim, Number>::get_viscous_operator_data() const
{
  return viscous_operator_data;
}

template<int dim, typename Number>
ConvectiveOperatorData<dim> const &
DGNavierStokesBase<dim, Number>::get_convective_operator_data() const
{
  return convective_operator_data;
}

template<int dim, typename Number>
GradientOperatorData<dim> const &
DGNavierStokesBase<dim, Number>::get_gradient_operator_data() const
{
  return gradient_operator_data;
}

template<int dim, typename Number>
DivergenceOperatorData<dim> const &
DGNavierStokesBase<dim, Number>::get_divergence_operator_data() const
{
  return divergence_operator_data;
}

template<int dim, typename Number>
std::shared_ptr<FieldFunctions<dim>> const
DGNavierStokesBase<dim, Number>::get_field_functions() const
{
  return field_functions;
}

// Polynomial degree required for CFL condition, e.g., CFL_k = CFL / k^{exp}.
template<int dim, typename Number>
unsigned int
DGNavierStokesBase<dim, Number>::get_polynomial_degree() const
{
  return param.degree_u;
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::initialize_vector_velocity(VectorType & src) const
{
  matrix_free.initialize_dof_vector(src, dof_index_u);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::initialize_vector_velocity_scalar(VectorType & src) const
{
  matrix_free.initialize_dof_vector(src, dof_index_u_scalar);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::initialize_vector_pressure(VectorType & src) const
{
  matrix_free.initialize_dof_vector(src, dof_index_p);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::prescribe_initial_conditions(VectorType & velocity,
                                                              VectorType & pressure,
                                                              double const evaluation_time) const
{
  field_functions->initial_solution_velocity->set_time(evaluation_time);
  field_functions->initial_solution_pressure->set_time(evaluation_time);

  // This is necessary if Number == float
  typedef LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

  VectorTypeDouble velocity_double;
  VectorTypeDouble pressure_double;
  velocity_double = velocity;
  pressure_double = pressure;

  VectorTools::interpolate(*mapping,
                           dof_handler_u,
                           *(field_functions->initial_solution_velocity),
                           velocity_double);

  VectorTools::interpolate(*mapping,
                           dof_handler_p,
                           *(field_functions->initial_solution_pressure),
                           pressure_double);

  velocity = velocity_double;
  pressure = pressure_double;
}

template<int dim, typename Number>
double
DGNavierStokesBase<dim, Number>::calculate_minimum_element_length() const
{
  return calculate_minimum_vertex_distance(dof_handler_u.get_triangulation());
}

template<int dim, typename Number>
double
DGNavierStokesBase<dim, Number>::calculate_time_step_cfl(VectorType const & velocity,
                                                         double const       cfl,
                                                         double const       exponent_degree) const
{
  return calculate_time_step_cfl_local<dim, Number>(matrix_free,
                                                    dof_index_u,
                                                    quad_index_u,
                                                    velocity,
                                                    cfl,
                                                    param.degree_u,
                                                    exponent_degree,
                                                    param.adaptive_time_stepping_cfl_type);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::apply_mass_matrix(VectorType & dst, VectorType const & src) const
{
  mass_matrix_operator.apply(dst, src);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::apply_mass_matrix_add(VectorType &       dst,
                                                       VectorType const & src) const
{
  mass_matrix_operator.apply_add(dst, src);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::shift_pressure(VectorType &   pressure,
                                                double const & eval_time) const
{
  VectorType vec1(pressure);
  for(unsigned int i = 0; i < vec1.local_size(); ++i)
    vec1.local_element(i) = 1.;
  field_functions->analytical_solution_pressure->set_time(eval_time);
  double const exact   = field_functions->analytical_solution_pressure->value(first_point);
  double       current = 0.;
  if(pressure.locally_owned_elements().is_element(dof_index_first_point))
    current = pressure(dof_index_first_point);
  current = Utilities::MPI::sum(current, MPI_COMM_WORLD);
  pressure.add(exact - current, vec1);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::shift_pressure_mean_value(VectorType &   pressure,
                                                           double const & eval_time) const
{
  // one cannot use Number as template here since Number might be float
  // while analytical_solution_pressure is of type Function<dim,double>
  typedef LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

  VectorTypeDouble vec_double;
  vec_double = pressure; // initialize

  field_functions->analytical_solution_pressure->set_time(eval_time);
  VectorTools::interpolate(*mapping,
                           dof_handler_p,
                           *(field_functions->analytical_solution_pressure),
                           vec_double);

  double const exact   = vec_double.mean_value();
  double const current = pressure.mean_value();

  VectorType vec_temp2(pressure);
  for(unsigned int i = 0; i < vec_temp2.local_size(); ++i)
    vec_temp2.local_element(i) = 1.;

  pressure.add(exact - current, vec_temp2);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::compute_vorticity(VectorType & dst, VectorType const & src) const
{
  vorticity_calculator.compute_vorticity(dst, src);

  inverse_mass_velocity.apply(dst, dst);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::compute_divergence(VectorType & dst, VectorType const & src) const
{
  divergence_calculator.compute_divergence(dst, src);

  inverse_mass_velocity_scalar.apply(dst, dst);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::compute_velocity_magnitude(VectorType &       dst,
                                                            VectorType const & src) const
{
  velocity_magnitude_calculator.compute(dst, src);

  inverse_mass_velocity_scalar.apply(dst, dst);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::compute_vorticity_magnitude(VectorType &       dst,
                                                             VectorType const & src) const
{
  velocity_magnitude_calculator.compute(dst, src);

  inverse_mass_velocity_scalar.apply(dst, dst);
}

/*
 *  Streamfunction psi (2D only): defined as u1 = d(psi)/dx2, u2 = - d(psi)/dx1
 *
 *  Vorticity: omega = du2/dx1 - du1/dx2
 *
 *  --> laplace(psi) = (d²/dx1²+d²/dx2²)(psi)
 *                   = d(d(psi)/dx1)/dx1 + d(d(psi)/dx2)/dx2
 *                   = d(-u2)/dx1 + d(u1)/dx2 = - omega
 *
 *  or
 *      - laplace(psi) = omega
 *
 *  with homogeneous Dirichlet BC's (assumption: whole boundary == streamline)
 */
template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::compute_streamfunction(VectorType &       dst,
                                                        VectorType const & src) const
{
  AssertThrow(dim == 2, ExcMessage("Calculation of streamfunction can only be used for dim==2."));

  // compute rhs vector
  StreamfunctionCalculatorRHSOperator<dim, Number> rhs_operator;
  rhs_operator.initialize(matrix_free, dof_index_u, dof_index_u_scalar, quad_index_u);
  VectorType rhs;
  initialize_vector_velocity_scalar(rhs);
  rhs_operator.apply(rhs, src);

  // setup Laplace operator for scalar velocity vector
  Poisson::LaplaceOperatorData<dim> laplace_operator_data;
  laplace_operator_data.dof_index  = get_dof_index_velocity_scalar();
  laplace_operator_data.quad_index = get_quad_index_velocity_linear();

  std::shared_ptr<Poisson::BoundaryDescriptor<dim>> boundary_descriptor_streamfunction;
  boundary_descriptor_streamfunction.reset(new Poisson::BoundaryDescriptor<dim>());

  // fill boundary descriptor: Assumption: only Dirichlet BC's
  boundary_descriptor_streamfunction->dirichlet_bc = boundary_descriptor_velocity->dirichlet_bc;

  AssertThrow(boundary_descriptor_velocity->neumann_bc.empty() == true,
              ExcMessage("Assumption is not fulfilled. Streamfunction calculator is "
                         "not implemented for this type of boundary conditions."));
  AssertThrow(boundary_descriptor_velocity->symmetry_bc.empty() == true,
              ExcMessage("Assumption is not fulfilled. Streamfunction calculator is "
                         "not implemented for this type of boundary conditions."));

  laplace_operator_data.bc = boundary_descriptor_streamfunction;

  laplace_operator_data.kernel_data.IP_factor      = 1.0;
  laplace_operator_data.kernel_data.degree         = this->param.degree_u;
  laplace_operator_data.kernel_data.degree_mapping = this->mapping_degree;

  typedef Poisson::LaplaceOperator<dim, Number> Laplace;
  Laplace                                       laplace_operator;
  AffineConstraints<double>                     constraint_dummy;
  laplace_operator.reinit(matrix_free, constraint_dummy, laplace_operator_data);

  // setup preconditioner
  std::shared_ptr<PreconditionerBase<Number>> preconditioner;

  // use multigrid preconditioner with Chebyshev smoother
  MultigridData mg_data;

  typedef Poisson::MultigridPreconditioner<dim, Number, MultigridNumber> MULTIGRID;

  preconditioner.reset(new MULTIGRID());

  std::shared_ptr<MULTIGRID> mg_preconditioner =
    std::dynamic_pointer_cast<MULTIGRID>(preconditioner);

  // explicit copy needed since function is called on const
  auto periodic_face_pairs = this->periodic_face_pairs;

  parallel::Triangulation<dim> const * tria =
    dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler_u_scalar.get_triangulation());
  const FiniteElement<dim> & fe = dof_handler_u_scalar.get_fe();

  mg_preconditioner->initialize(mg_data,
                                tria,
                                fe,
                                *mapping,
                                laplace_operator.get_operator_data(),
                                &laplace_operator.get_operator_data().bc->dirichlet_bc,
                                &periodic_face_pairs);

  // setup solver
  CGSolverData solver_data;
  solver_data.solver_tolerance_rel = 1.e-10;
  solver_data.use_preconditioner   = true;

  CGSolver<Laplace, PreconditionerBase<Number>, VectorType> poisson_solver(laplace_operator,
                                                                           *preconditioner,
                                                                           solver_data);

  // solve Poisson problem
  poisson_solver.solve(dst, rhs, /* update preconditioner = */ false);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::compute_q_criterion(VectorType & dst, VectorType const & src) const
{
  q_criterion_calculator.compute(dst, src);

  inverse_mass_velocity_scalar.apply(dst, dst);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::apply_inverse_mass_matrix(VectorType &       dst,
                                                           VectorType const & src) const
{
  inverse_mass_velocity.apply(dst, src);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::evaluate_convective_term(VectorType &       dst,
                                                          VectorType const & src,
                                                          Number const       evaluation_time) const
{
  convective_operator.evaluate(dst, src, evaluation_time);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::evaluate_pressure_gradient_term(VectorType &       dst,
                                                                 VectorType const & src,
                                                                 double const evaluation_time) const
{
  gradient_operator.evaluate(dst, src, evaluation_time);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::evaluate_velocity_divergence_term(
  VectorType &       dst,
  VectorType const & src,
  double const       evaluation_time) const
{
  divergence_operator.evaluate(dst, src, evaluation_time);
}

// OIF splitting
template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::evaluate_negative_convective_term_and_apply_inverse_mass_matrix(
  VectorType &       dst,
  VectorType const & src,
  Number const       evaluation_time) const
{
  convective_operator.evaluate(dst, src, evaluation_time);

  // shift convective term to the rhs of the equation
  dst *= -1.0;

  inverse_mass_velocity.apply(dst, dst);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::evaluate_negative_convective_term_and_apply_inverse_mass_matrix(
  VectorType &       dst,
  VectorType const & src,
  Number const       evaluation_time,
  VectorType const & velocity_transport) const
{
  convective_operator.evaluate_linear_transport(dst, src, evaluation_time, velocity_transport);

  // shift convective term to the rhs of the equation
  dst *= -1.0;

  inverse_mass_velocity.apply(dst, dst);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::update_turbulence_model(VectorType const & velocity)
{
  // calculate turbulent viscosity locally in each cell and face quadrature point
  turbulence_model.calculate_turbulent_viscosity(velocity);
}

template<int dim, typename Number>
double
DGNavierStokesBase<dim, Number>::calculate_dissipation_convective_term(VectorType const & velocity,
                                                                       double const time) const
{
  VectorType dst;
  dst.reinit(velocity, false);
  convective_operator.evaluate(dst, velocity, time);
  return velocity * dst;
}

template<int dim, typename Number>
double
DGNavierStokesBase<dim, Number>::calculate_dissipation_viscous_term(
  VectorType const & velocity) const
{
  VectorType dst;
  dst.reinit(velocity, false);
  viscous_operator.apply(dst, velocity);
  return velocity * dst;
}

template<int dim, typename Number>
double
DGNavierStokesBase<dim, Number>::calculate_dissipation_divergence_term(
  VectorType const & velocity) const
{
  if(param.use_divergence_penalty == true)
  {
    VectorType dst;
    dst.reinit(velocity, false);
    projection_operator->apply_div_penalty(dst, velocity);
    return velocity * dst;
  }
  else
  {
    return 0.0;
  }
}

template<int dim, typename Number>
double
DGNavierStokesBase<dim, Number>::calculate_dissipation_continuity_term(
  VectorType const & velocity) const
{
  if(param.use_continuity_penalty == true)
  {
    VectorType dst;
    dst.reinit(velocity, false);
    projection_operator->apply_conti_penalty(dst, velocity);
    return velocity * dst;
  }
  else
  {
    return 0.0;
  }
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::setup_projection_operator()
{
  // setup projection operator
  ProjectionOperatorData proj_op_data;
  proj_op_data.type_penalty_parameter = param.type_penalty_parameter;
  proj_op_data.viscosity              = param.viscosity;
  proj_op_data.use_divergence_penalty = param.use_divergence_penalty;
  proj_op_data.use_continuity_penalty = param.use_continuity_penalty;
  proj_op_data.degree                 = param.degree_u;
  proj_op_data.penalty_factor_div     = param.divergence_penalty_factor;
  proj_op_data.penalty_factor_conti   = param.continuity_penalty_factor;
  proj_op_data.which_components       = param.continuity_penalty_components;
  proj_op_data.use_cell_based_loops   = param.use_cell_based_face_loops;
  proj_op_data.implement_block_diagonal_preconditioner_matrix_free =
    param.implement_block_diagonal_preconditioner_matrix_free;
  proj_op_data.preconditioner_block_jacobi = param.preconditioner_block_diagonal_projection;
  proj_op_data.block_jacobi_solver_data    = param.solver_data_block_diagonal_projection;

  projection_operator.reset(new PROJ_OPERATOR(
    matrix_free, get_dof_index_velocity(), get_quad_index_velocity_linear(), proj_op_data));
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::setup_projection_solver()
{
  // setup projection solver

  // divergence penalty only
  if(param.use_divergence_penalty == true && param.use_continuity_penalty == false)
  {
    // use direct solver
    if(param.solver_projection == SolverProjection::LU)
    {
      // projection operator
      typedef DirectProjectionSolverDivergencePenalty<dim, Number, PROJ_OPERATOR> PROJ_SOLVER;

      projection_solver.reset(new PROJ_SOLVER(*projection_operator));
    }
    // use iterative solver (PCG)
    else if(param.solver_projection == SolverProjection::CG)
    {
      // projection operator
      elementwise_projection_operator.reset(new ELEMENTWISE_PROJ_OPERATOR(*projection_operator));

      // preconditioner
      typedef Elementwise::PreconditionerBase<VectorizedArray<Number>> PROJ_PRECONDITIONER;

      if(param.preconditioner_projection == PreconditionerProjection::None)
      {
        typedef Elementwise::PreconditionerIdentity<VectorizedArray<Number>> IDENTITY;

        elementwise_preconditioner_projection.reset(
          new IDENTITY(elementwise_projection_operator->get_problem_size()));
      }
      else if(param.preconditioner_projection == PreconditionerProjection::InverseMassMatrix)
      {
        typedef Elementwise::InverseMassMatrixPreconditioner<dim, dim, Number> INVERSE_MASS;

        elementwise_preconditioner_projection.reset(
          new INVERSE_MASS(projection_operator->get_matrix_free(),
                           projection_operator->get_dof_index(),
                           projection_operator->get_quad_index()));
      }
      else
      {
        AssertThrow(false, ExcMessage("The specified preconditioner is not implemented."));
      }

      // solver
      Elementwise::IterativeSolverData projection_solver_data;
      projection_solver_data.solver_type         = Elementwise::Solver::CG;
      projection_solver_data.solver_data.abs_tol = param.solver_data_projection.abs_tol;
      projection_solver_data.solver_data.rel_tol = param.solver_data_projection.rel_tol;

      typedef Elementwise::
        IterativeSolver<dim, dim, Number, ELEMENTWISE_PROJ_OPERATOR, PROJ_PRECONDITIONER>
          PROJ_SOLVER;

      projection_solver.reset(new PROJ_SOLVER(
        *std::dynamic_pointer_cast<ELEMENTWISE_PROJ_OPERATOR>(elementwise_projection_operator),
        *std::dynamic_pointer_cast<PROJ_PRECONDITIONER>(elementwise_preconditioner_projection),
        projection_solver_data));
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified projection solver not implemented."));
    }
  }
  // both divergence and continuity penalty terms
  else if(param.use_divergence_penalty == true && param.use_continuity_penalty == true)
  {
    // preconditioner
    if(param.preconditioner_projection == PreconditionerProjection::InverseMassMatrix)
    {
      preconditioner_projection.reset(new InverseMassMatrixPreconditioner<dim, dim, Number>(
        matrix_free, param.degree_u, get_dof_index_velocity(), get_quad_index_velocity_linear()));
    }
    else if(param.preconditioner_projection == PreconditionerProjection::PointJacobi)
    {
      // Note that at this point (when initializing the Jacobi preconditioner and calculating the
      // diagonal) the penalty parameter of the projection operator has not been calculated and the
      // time step size has not been set. Hence, update_preconditioner = true should be used for the
      // Jacobi preconditioner in order to use to correct diagonal for preconditioning.
      preconditioner_projection.reset(new JacobiPreconditioner<PROJ_OPERATOR>(
        *std::dynamic_pointer_cast<PROJ_OPERATOR>(projection_operator)));
    }
    else if(param.preconditioner_projection == PreconditionerProjection::BlockJacobi)
    {
      // Note that at this point (when initializing the Jacobi preconditioner)
      // the penalty parameter of the projection operator has not been calculated and the time step
      // size has not been set. Hence, update_preconditioner = true should be used for the Jacobi
      // preconditioner in order to use to correct diagonal blocks for preconditioning.
      preconditioner_projection.reset(new BlockJacobiPreconditioner<PROJ_OPERATOR>(
        *std::dynamic_pointer_cast<PROJ_OPERATOR>(projection_operator)));
    }
    else
    {
      AssertThrow(param.preconditioner_projection == PreconditionerProjection::None ||
                    param.preconditioner_projection ==
                      PreconditionerProjection::InverseMassMatrix ||
                    param.preconditioner_projection == PreconditionerProjection::PointJacobi ||
                    param.preconditioner_projection == PreconditionerProjection::BlockJacobi,
                  ExcMessage("Specified preconditioner of projection solver not implemented."));
    }

    // solver
    if(param.solver_projection == SolverProjection::CG)
    {
      // setup solver data
      CGSolverData projection_solver_data;
      projection_solver_data.max_iter             = param.solver_data_projection.max_iter;
      projection_solver_data.solver_tolerance_abs = param.solver_data_projection.abs_tol;
      projection_solver_data.solver_tolerance_rel = param.solver_data_projection.rel_tol;
      // default value of use_preconditioner = false
      if(param.preconditioner_projection == PreconditionerProjection::InverseMassMatrix ||
         param.preconditioner_projection == PreconditionerProjection::PointJacobi ||
         param.preconditioner_projection == PreconditionerProjection::BlockJacobi)
      {
        projection_solver_data.use_preconditioner = true;
      }
      else
      {
        AssertThrow(param.preconditioner_projection == PreconditionerProjection::None ||
                      param.preconditioner_projection ==
                        PreconditionerProjection::InverseMassMatrix ||
                      param.preconditioner_projection == PreconditionerProjection::PointJacobi ||
                      param.preconditioner_projection == PreconditionerProjection::BlockJacobi,
                    ExcMessage("Specified preconditioner of projection solver not implemented."));
      }

      // setup solver
      projection_solver.reset(new CGSolver<PROJ_OPERATOR, PreconditionerBase<Number>, VectorType>(
        *std::dynamic_pointer_cast<PROJ_OPERATOR>(projection_operator),
        *preconditioner_projection,
        projection_solver_data));
    }
    else
    {
      AssertThrow(param.solver_projection == SolverProjection::CG,
                  ExcMessage("Specified projection solver not implemented."));
    }
  }
  else
  {
    AssertThrow(
      param.use_divergence_penalty == false && param.use_continuity_penalty == false,
      ExcMessage(
        "Specified combination of divergence and continuity penalty operators not implemented."));
  }
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::update_projection_operator(VectorType const & velocity,
                                                            double const       time_step_size) const
{
  AssertThrow(projection_operator.get() != 0,
              ExcMessage("Projection operator is not initialized."));

  // Update projection operator, i.e., the penalty parameters that depend on the velocity field
  projection_operator->calculate_penalty_parameter(velocity);

  // Set the correct time step size.
  projection_operator->set_time_step_size(time_step_size);
}

template<int dim, typename Number>
unsigned int
DGNavierStokesBase<dim, Number>::solve_projection(VectorType &       dst,
                                                  VectorType const & src,
                                                  bool const &       update_preconditioner) const
{
  Assert(projection_solver.get() != 0, ExcMessage("Projection solver has not been initialized."));

  unsigned int n_iter = projection_solver->solve(dst, src, update_preconditioner);

  return n_iter;
}

template class DGNavierStokesBase<2, float>;
template class DGNavierStokesBase<3, float>;

template class DGNavierStokesBase<2, double>;
template class DGNavierStokesBase<3, double>;

} // namespace IncNS
