/*
 * dg_comp_navier_stokes.h
 *
 *  Created on: 2018
 *      Author: fehn
 */

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_COMP_NAVIER_STOKES_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_COMP_NAVIER_STOKES_H_

// deal.II
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/vector_tools.h>

// user interface
#include "../../compressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../../compressible_navier_stokes/user_interface/field_functions.h"
#include "../../compressible_navier_stokes/user_interface/input_parameters.h"

// operators
#include "../../compressible_navier_stokes/spatial_discretization/comp_navier_stokes_operators.h"
#include "comp_navier_stokes_calculators.h"
#include "operators/inverse_mass_matrix.h"

// interface space-time
#include "../interface_space_time/operator.h"

// time integration
#include "time_integration/time_step_calculation.h"

namespace CompNS
{
template<int dim, int degree, int n_q_points_conv, int n_q_points_vis, typename Number>
class DGOperator : public dealii::Subscriptor, public Interface::Operator<Number>
{
public:
  enum class DofHandlerSelector
  {
    all_components = 0,
    vector         = 1,
    scalar         = 2,
    n_variants     = scalar + 1
  };

  enum class QuadratureSelector
  {
    overintegration_conv = 0,
    overintegration_vis  = 1,
    standard             = 2,
    n_variants           = standard + 1
  };

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef PostProcessor<dim, degree, n_q_points_conv, n_q_points_vis, Number> Postprocessor;

  static const unsigned int dof_index_all =
    static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
      DofHandlerSelector::all_components);
  static const unsigned int dof_index_vector =
    static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
      DofHandlerSelector::vector);
  static const unsigned int dof_index_scalar =
    static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
      DofHandlerSelector::scalar);

  static const unsigned int quad_index_standard =
    static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::standard);
  static const unsigned int quad_index_overintegration_conv =
    static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::overintegration_conv);
  static const unsigned int quad_index_overintegration_vis =
    static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::overintegration_vis);

  // TODO: which quadrature rule should we use for p, u, T calculation (L2 projections used to
  // calculate derived quantities)
  static const unsigned int quad_index_l2_projections =
    quad_index_standard; // quad_index_overintegration_conv;
  static const unsigned int n_q_points_l2_projections =
    (quad_index_l2_projections == quad_index_standard) ? degree + 1 : n_q_points_conv;

  DGOperator(parallel::Triangulation<dim> const & triangulation,
             InputParameters<dim> const &         param_in,
             std::shared_ptr<Postprocessor>       postprocessor_in)
    : dealii::Subscriptor(),
      fe(new FESystem<dim>(FE_DGQ<dim>(degree), dim + 2)),
      fe_vector(new FESystem<dim>(FE_DGQ<dim>(degree), dim)),
      fe_scalar(degree),
      mapping(param_in.degree_mapping),
      dof_handler(triangulation),
      dof_handler_vector(triangulation),
      dof_handler_scalar(triangulation),
      param(param_in),
      postprocessor(postprocessor_in),
      wall_time_operator_evaluation(0.0)
  {
  }

  void
  setup(std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_density_in,
        std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_velocity_in,
        std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_pressure_in,
        std::shared_ptr<BoundaryDescriptorEnergy<dim>> boundary_descriptor_energy_in,
        std::shared_ptr<FieldFunctions<dim>>           field_functions_in,
        std::shared_ptr<AnalyticalSolution<dim>>       analytical_solution_in)
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

    setup_postprocessor(analytical_solution_in);

    pcout << std::endl << "... done!" << std::endl;
  }

  unsigned int
  get_number_of_dofs() const
  {
    return dof_handler.n_dofs();
  }

  // initialization of DoF vectors
  void
  initialize_dof_vector(VectorType & src) const
  {
    data.initialize_dof_vector(src, dof_index_all);
  }

  void
  initialize_dof_vector_scalar(VectorType & src) const
  {
    data.initialize_dof_vector(src, dof_index_scalar);
  }

  void
  initialize_dof_vector_dim_components(VectorType & src) const
  {
    data.initialize_dof_vector(src, dof_index_vector);
  }

  // set initial conditions
  void
  prescribe_initial_conditions(VectorType & src, double const evaluation_time) const
  {
    this->field_functions->initial_solution->set_time(evaluation_time);

    VectorTools::interpolate(mapping, dof_handler, *(this->field_functions->initial_solution), src);
  }

  /*
   *  This function is used in case of explicit time integration:
   *  This function evaluates the right-hand side operator, the
   *  convective and diffusive term (subsequently multiplied by -1.0 in order
   *  to shift these terms to the right-hand side of the equations)
   *  and finally applies the inverse mass matrix operator.
   */
  void
  evaluate(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    Timer timer;
    timer.restart();

    // set dst to zero
    dst = 0.0;

    if(param.use_combined_operator == true)
    {
      // viscous and convective terms
      combined_operator.evaluate_add(dst, src, evaluation_time);
    }
    else // apply operators separately
    {
      // viscous operator
      if(param.equation_type == EquationType::NavierStokes)
      {
        viscous_operator.evaluate_add(dst, src, evaluation_time);
      }

      // convective operator
      if(param.equation_type == EquationType::Euler ||
         param.equation_type == EquationType::NavierStokes)
      {
        convective_operator.evaluate_add(dst, src, evaluation_time);
      }
    }

    // shift diffusive and convective term to the rhs of the equation
    dst *= -1.0;

    // body force term
    if(param.right_hand_side == true)
    {
      body_force_operator.evaluate_add(dst, src, evaluation_time);
    }

    // apply inverse mass matrix
    inverse_mass_matrix_operator->apply(dst, dst);

    wall_time_operator_evaluation += timer.wall_time();
  }

  void
  evaluate_convective(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    // set dst to zero
    dst = 0.0;

    // convective operator
    if(param.equation_type == EquationType::Euler ||
       param.equation_type == EquationType::NavierStokes)
    {
      convective_operator.evaluate_add(dst, src, evaluation_time);
    }
  }

  void
  evaluate_viscous(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    // set dst to zero
    dst = 0.0;

    // viscous operator
    if(param.equation_type == EquationType::NavierStokes)
    {
      viscous_operator.evaluate_add(dst, src, evaluation_time);
    }
  }

  void
  evaluate_convective_and_viscous(VectorType &       dst,
                                  VectorType const & src,
                                  Number const       evaluation_time) const
  {
    // set dst to zero
    dst = 0.0;

    if(param.use_combined_operator == true)
    {
      // viscous and convective terms
      combined_operator.evaluate_add(dst, src, evaluation_time);
    }
    else // apply operators separately
    {
      // viscous operator
      if(param.equation_type == EquationType::NavierStokes)
      {
        viscous_operator.evaluate_add(dst, src, evaluation_time);
      }

      // convective operator
      if(param.equation_type == EquationType::Euler ||
         param.equation_type == EquationType::NavierStokes)
      {
        convective_operator.evaluate_add(dst, src, evaluation_time);
      }
    }
  }

  void
  apply_inverse_mass(VectorType & dst, VectorType const & src) const
  {
    // apply inverse mass matrix
    inverse_mass_matrix_operator->apply(dst, src);
  }

  // getters
  MatrixFree<dim, Number> const &
  get_data() const
  {
    return data;
  }

  Mapping<dim> const &
  get_mapping() const
  {
    return mapping;
  }

  FESystem<dim> const &
  get_fe() const
  {
    return *fe;
  }

  DoFHandler<dim> const &
  get_dof_handler() const
  {
    return dof_handler;
  }

  DoFHandler<dim> const &
  get_dof_handler_scalar() const
  {
    return dof_handler_scalar;
  }

  DoFHandler<dim> const &
  get_dof_handler_vector() const
  {
    return dof_handler_vector;
  }

  unsigned int
  get_dof_index_vector() const
  {
    return dof_index_vector;
  }

  unsigned int
  get_dof_index_scalar() const
  {
    return dof_index_scalar;
  }

  unsigned int
  get_quad_index_standard() const
  {
    return quad_index_standard;
  }

  // pressure
  void
  compute_pressure(VectorType & dst, VectorType const & src) const
  {
    p_u_T_calculator.compute_pressure(dst, src);
    inverse_mass_matrix_operator_scalar->apply(dst, dst);
  }

  // velocity
  void
  compute_velocity(VectorType & dst, VectorType const & src) const
  {
    p_u_T_calculator.compute_velocity(dst, src);
    inverse_mass_matrix_operator_vector->apply(dst, dst);
  }

  // temperature
  void
  compute_temperature(VectorType & dst, VectorType const & src) const
  {
    p_u_T_calculator.compute_temperature(dst, src);
    inverse_mass_matrix_operator_scalar->apply(dst, dst);
  }

  // vorticity
  void
  compute_vorticity(VectorType & dst, VectorType const & src) const
  {
    vorticity_calculator.compute_vorticity(dst, src);
    inverse_mass_matrix_operator_vector->apply(dst, dst);
  }

  // divergence
  void
  compute_divergence(VectorType & dst, VectorType const & src) const
  {
    divergence_calculator.compute_divergence(dst, src);
    inverse_mass_matrix_operator_scalar->apply(dst, dst);
  }

  double
  get_wall_time_operator_evaluation() const
  {
    return wall_time_operator_evaluation;
  }

  void
  do_postprocessing(VectorType const & solution,
                    double const       time,
                    int const          time_step_number) const
  {
    postprocessor->do_postprocessing(solution, time, time_step_number);
  }

  double
  calculate_minimum_element_length() const
  {
    return calculate_minimum_vertex_distance(dof_handler.get_triangulation());
  }

  unsigned int
  get_polynomial_degree() const
  {
    return degree;
  }

private:
  void
  create_dofs()
  {
    // enumerate degrees of freedom
    dof_handler.distribute_dofs(*fe);
    dof_handler_vector.distribute_dofs(*fe_vector);
    dof_handler_scalar.distribute_dofs(fe_scalar);

    constexpr int ndofs_per_cell = Utilities::pow(degree + 1, dim) * (dim + 2);

    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    pcout << std::endl
          << "Discontinuous Galerkin finite element discretization:" << std::endl
          << std::endl;

    print_parameter(pcout, "degree of 1D polynomials", degree);
    print_parameter(pcout, "number of dofs per cell", ndofs_per_cell);
    print_parameter(pcout, "number of dofs (total)", dof_handler.n_dofs());
    print_parameter(pcout, "number of 1D q-points (std)", degree + 1);
    print_parameter(pcout, "number of 1D q-points (over-conv)", n_q_points_conv);
    print_parameter(pcout, "number of 1D q-points (over-vis)", n_q_points_vis);
  }

  void
  initialize_matrix_free()
  {
    // quadratures used to perform integrals
    std::vector<Quadrature<1>> quadratures;
    quadratures.resize(static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::n_variants));
    quadratures[static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::standard)]             = QGauss<1>(degree + 1);
    quadratures[static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::overintegration_conv)] = QGauss<1>(n_q_points_conv);
    quadratures[static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::overintegration_vis)]  = QGauss<1>(n_q_points_vis);

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
    constraint_matrix_vec.resize(
      static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
        DofHandlerSelector::n_variants));
    AffineConstraints<double> constraint;
    constraint.close();
    constraint_matrix_vec[dof_index_all]    = &constraint;
    constraint_matrix_vec[dof_index_vector] = &constraint;
    constraint_matrix_vec[dof_index_scalar] = &constraint;

    data.reinit(mapping, dof_handler_vec, constraint_matrix_vec, quadratures, additional_data);
  }

  void
  setup_operators()
  {
    // mass matrix operator
    mass_matrix_operator_data.dof_index  = dof_index_all;
    mass_matrix_operator_data.quad_index = quad_index_standard;
    mass_matrix_operator.initialize(data, mass_matrix_operator_data);

    // inverse mass matrix operator
    inverse_mass_matrix_operator.reset(
      new InverseMassMatrixOperator<dim, degree, Number, dim + 2>());
    inverse_mass_matrix_operator->initialize(data, dof_index_all, quad_index_standard);

    inverse_mass_matrix_operator_vector.reset(
      new InverseMassMatrixOperator<dim, degree, Number, dim>());
    inverse_mass_matrix_operator_vector->initialize(data, dof_index_vector, quad_index_standard);

    inverse_mass_matrix_operator_scalar.reset(
      new InverseMassMatrixOperator<dim, degree, Number, 1>());
    inverse_mass_matrix_operator_scalar->initialize(data, dof_index_scalar, quad_index_standard);

    // body force operator
    BodyForceOperatorData<dim> body_force_operator_data;
    body_force_operator_data.dof_index  = dof_index_all;
    body_force_operator_data.quad_index = quad_index_standard;
    body_force_operator_data.rhs_rho    = field_functions->right_hand_side_density;
    body_force_operator_data.rhs_u      = field_functions->right_hand_side_velocity;
    body_force_operator_data.rhs_E      = field_functions->right_hand_side_energy;
    body_force_operator.initialize(data, body_force_operator_data);

    // convective operator
    convective_operator_data.dof_index             = dof_index_all;
    convective_operator_data.quad_index            = quad_index_overintegration_conv;
    convective_operator_data.bc_rho                = boundary_descriptor_density;
    convective_operator_data.bc_u                  = boundary_descriptor_velocity;
    convective_operator_data.bc_p                  = boundary_descriptor_pressure;
    convective_operator_data.bc_E                  = boundary_descriptor_energy;
    convective_operator_data.heat_capacity_ratio   = param.heat_capacity_ratio;
    convective_operator_data.specific_gas_constant = param.specific_gas_constant;
    convective_operator.initialize(data, convective_operator_data);

    // viscous operator
    viscous_operator_data.dof_index             = dof_index_all;
    viscous_operator_data.quad_index            = quad_index_overintegration_vis;
    viscous_operator_data.IP_factor             = param.IP_factor;
    viscous_operator_data.dynamic_viscosity     = param.dynamic_viscosity;
    viscous_operator_data.reference_density     = param.reference_density;
    viscous_operator_data.thermal_conductivity  = param.thermal_conductivity;
    viscous_operator_data.heat_capacity_ratio   = param.heat_capacity_ratio;
    viscous_operator_data.specific_gas_constant = param.specific_gas_constant;
    viscous_operator_data.bc_rho                = boundary_descriptor_density;
    viscous_operator_data.bc_u                  = boundary_descriptor_velocity;
    viscous_operator_data.bc_E                  = boundary_descriptor_energy;
    viscous_operator.initialize(mapping, data, viscous_operator_data);

    if(param.use_combined_operator == true)
    {
      AssertThrow(n_q_points_conv == n_q_points_vis,
                  ExcMessage("Use the same number of quadrature points for convective term "
                             "and viscous term in case of combined operator."));

      combined_operator_data.dof_index             = dof_index_all;
      combined_operator_data.quad_index            = quad_index_overintegration_vis;
      combined_operator_data.IP_factor             = param.IP_factor;
      combined_operator_data.dynamic_viscosity     = param.dynamic_viscosity;
      combined_operator_data.reference_density     = param.reference_density;
      combined_operator_data.thermal_conductivity  = param.thermal_conductivity;
      combined_operator_data.heat_capacity_ratio   = param.heat_capacity_ratio;
      combined_operator_data.specific_gas_constant = param.specific_gas_constant;
      combined_operator_data.bc_rho                = boundary_descriptor_density;
      combined_operator_data.bc_u                  = boundary_descriptor_velocity;
      combined_operator_data.bc_p                  = boundary_descriptor_pressure;
      combined_operator_data.bc_E                  = boundary_descriptor_energy;
      combined_operator.initialize(mapping, data, combined_operator_data);
    }

    // calculators
    p_u_T_calculator.initialize(data,
                                dof_index_all,
                                dof_index_vector,
                                dof_index_scalar,
                                quad_index_l2_projections,
                                param.heat_capacity_ratio,
                                param.specific_gas_constant);
    vorticity_calculator.initialize(data, dof_index_vector, quad_index_standard);
    divergence_calculator.initialize(data, dof_index_vector, dof_index_scalar, quad_index_standard);
  }

  void
  setup_postprocessor(std::shared_ptr<AnalyticalSolution<dim>> analytical_solution_in)
  {
    DofQuadIndexData dof_quad_index_data;
    dof_quad_index_data.dof_index_velocity  = dof_index_vector;
    dof_quad_index_data.dof_index_pressure  = dof_index_scalar;
    dof_quad_index_data.quad_index_velocity = quad_index_standard;

    postprocessor->setup(*this,
                         dof_handler,
                         dof_handler_vector,
                         dof_handler_scalar,
                         mapping,
                         data,
                         dof_quad_index_data,
                         analytical_solution_in);
  }

  // fe
  std::shared_ptr<FESystem<dim>> fe;        // all (dim+2) components: (rho, rho u, rho E)
  std::shared_ptr<FESystem<dim>> fe_vector; // e.g. velocity
  FE_DGQ<dim>                    fe_scalar; // scalar quantity, e.g, pressure

  // mapping
  MappingQGeneric<dim> mapping;

  // DoFHandler for all (dim+2) components: (rho, rho u, rho E)
  DoFHandler<dim> dof_handler;
  // DoFHandler for vectorial quantities such as the velocity
  DoFHandler<dim> dof_handler_vector;
  // DoFHandler for scalar quantities such as pressure, temperature
  DoFHandler<dim> dof_handler_scalar;

  MatrixFree<dim, Number> data;

  InputParameters<dim> const & param;

  std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_density;
  std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_velocity;
  std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_pressure;
  std::shared_ptr<BoundaryDescriptorEnergy<dim>> boundary_descriptor_energy;
  std::shared_ptr<FieldFunctions<dim>>           field_functions;

  // DG operators

  // use standard quadrature for mass matrix operator
  MassMatrixOperatorData                              mass_matrix_operator_data;
  MassMatrixOperator<dim, degree, degree + 1, Number> mass_matrix_operator;

  std::shared_ptr<InverseMassMatrixOperator<dim, degree, Number, dim + 2>>
    inverse_mass_matrix_operator;
  std::shared_ptr<InverseMassMatrixOperator<dim, degree, Number, dim>>
    inverse_mass_matrix_operator_vector;
  std::shared_ptr<InverseMassMatrixOperator<dim, degree, Number, 1>>
    inverse_mass_matrix_operator_scalar;

  // use standard quadrature for body force operator
  BodyForceOperatorData<dim>                         body_force_operator_data;
  BodyForceOperator<dim, degree, degree + 1, Number> body_force_operator;

  ConvectiveOperatorData<dim>                              convective_operator_data;
  ConvectiveOperator<dim, degree, n_q_points_conv, Number> convective_operator;

  ViscousOperatorData<dim>                             viscous_operator_data;
  ViscousOperator<dim, degree, n_q_points_vis, Number> viscous_operator;

  // convective and viscous terms combined to one operator
  CombinedOperatorData<dim>                             combined_operator_data;
  CombinedOperator<dim, degree, n_q_points_vis, Number> combined_operator;

  // L2 projections to calculate derived quantities
  p_u_T_Calculator<dim, degree, n_q_points_l2_projections, Number> p_u_T_calculator;
  VorticityCalculator<dim, degree, Number>                         vorticity_calculator;
  DivergenceCalculator<dim, degree, Number>                        divergence_calculator;

  // postprocessor
  std::shared_ptr<Postprocessor> postprocessor;

  // wall time for operator evaluation
  mutable double wall_time_operator_evaluation;
};

} // namespace CompNS

#endif /* INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_ */
