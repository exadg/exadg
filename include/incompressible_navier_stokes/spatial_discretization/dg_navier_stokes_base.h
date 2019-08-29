/*
 * dg_navier_stokes_base.h
 *
 *  Created on: Jun 27, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_BASE_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_BASE_H_

// deal.II
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/matrix_free/operators.h>

// user interface
#include "../../incompressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../../incompressible_navier_stokes/user_interface/field_functions.h"
#include "../../incompressible_navier_stokes/user_interface/input_parameters.h"

// calculators
#include "../../incompressible_navier_stokes/spatial_discretization/calculators/divergence_calculator.h"
#include "../../incompressible_navier_stokes/spatial_discretization/calculators/q_criterion_calculator.h"
#include "../../incompressible_navier_stokes/spatial_discretization/calculators/streamfunction_calculator_rhs_operator.h"
#include "../../incompressible_navier_stokes/spatial_discretization/calculators/velocity_magnitude_calculator.h"
#include "../../incompressible_navier_stokes/spatial_discretization/calculators/vorticity_calculator.h"

// operators
#include "../../operators/inverse_mass_matrix.h"
#include "../../poisson/spatial_discretization/laplace_operator.h"
#include "operators/convective_operator.h"
#include "operators/divergence_operator.h"
#include "operators/gradient_operator.h"
#include "operators/mass_matrix_operator.h"
#include "operators/momentum_operator.h"
#include "operators/projection_operator.h"
#include "operators/rhs_operator.h"
#include "operators/viscous_operator.h"

// LES turbulence model
#include "turbulence_model.h"

// interface space-time
#include "interface.h"

// preconditioners and solvers
#include "../../solvers_and_preconditioners/newton/newton_solver.h"
#include "../../solvers_and_preconditioners/preconditioner/inverse_mass_matrix_preconditioner.h"
#include "../../solvers_and_preconditioners/preconditioner/jacobi_preconditioner.h"
#include "../../solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h"
#include "../preconditioners/multigrid_preconditioner_momentum.h"

// time integration
#include "time_integration/time_step_calculation.h"

// postprocessor
#include "../postprocessor/postprocessor_base.h"

using namespace dealii;

namespace IncNS
{
template<int dim, typename Number>
class DGNavierStokesBase : public dealii::Subscriptor, public Interface::OperatorBase<Number>
{
protected:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef PostProcessorBase<dim, Number> Postprocessor;

  typedef float MultigridNumber;

  enum class DofHandlerSelector
  {
    velocity        = 0,
    pressure        = 1,
    velocity_scalar = 2,
    n_variants      = velocity_scalar + 1
  };

  enum class QuadratureSelector
  {
    velocity           = 0,
    pressure           = 1,
    velocity_nonlinear = 2,
    n_variants         = velocity_nonlinear + 1
  };

  static const unsigned int number_vorticity_components = (dim == 2) ? 1 : dim;

  static const unsigned int dof_index_u =
    static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
      DofHandlerSelector::velocity);
  static const unsigned int dof_index_p =
    static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
      DofHandlerSelector::pressure);
  static const unsigned int dof_index_u_scalar =
    static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
      DofHandlerSelector::velocity_scalar);

  static const unsigned int quad_index_u =
    static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::velocity);
  static const unsigned int quad_index_p =
    static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::pressure);
  static const unsigned int quad_index_u_nonlinear =
    static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::velocity_nonlinear);

public:
  /*
   * Constructor.
   */
  DGNavierStokesBase(parallel::TriangulationBase<dim> const & triangulation,
                     InputParameters const &                  parameters,
                     std::shared_ptr<Postprocessor>           postprocessor);

  /*
   * Destructor.
   */
  virtual ~DGNavierStokesBase();

  /*
   * Setup function. Initializes basic finite element components, matrix-free object, and basic
   * operators. This function does not perform the setup related to the solution of linear systems
   * of equations.
   */
  virtual void
  setup(std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> const
                                                        periodic_face_pairs,
        std::shared_ptr<BoundaryDescriptorU<dim>> const boundary_descriptor_velocity,
        std::shared_ptr<BoundaryDescriptorP<dim>> const boundary_descriptor_pressure,
        std::shared_ptr<FieldFunctions<dim>> const      field_functions);

  /*
   * This function initializes operators, preconditioners, and solvers related to the solution of
   * (non-)linear systems of equation required for implicit formulations. It has to be extended
   * by derived classes if necessary.
   */
  virtual void
  setup_solvers(double const & scaling_factor_time_derivative_term, VectorType const & velocity);

  /*
   * Getters and setters.
   */
  MatrixFree<dim, Number> const &
  get_matrix_free() const;

  unsigned int
  get_dof_index_velocity() const;

  unsigned int
  get_dof_index_velocity_scalar() const;

  unsigned int
  get_quad_index_velocity_linear() const;

  unsigned int
  get_quad_index_velocity_nonlinear() const;

  unsigned int
  get_quad_index_velocity_linearized() const;

  unsigned int
  get_dof_index_pressure() const;

  unsigned int
  get_quad_index_pressure() const;

  unsigned int
  get_degree_p() const;

  Mapping<dim> const &
  get_mapping() const;

  FESystem<dim> const &
  get_fe_u() const;

  FE_DGQ<dim> const &
  get_fe_p() const;

  DoFHandler<dim> const &
  get_dof_handler_u() const;

  DoFHandler<dim> const &
  get_dof_handler_u_scalar() const;

  DoFHandler<dim> const &
  get_dof_handler_p() const;

  types::global_dof_index
  get_number_of_dofs() const;

  double
  get_viscosity() const;

  VectorizedArray<Number>
  get_viscosity_boundary_face(unsigned int const face, unsigned int const q) const;

  // Polynomial degree required, e.g., for CFL condition (CFL_k = CFL / k^{exp}).
  unsigned int
  get_polynomial_degree() const;

  void
  set_velocity_ptr(VectorType const & velocity) const;

  /*
   * Initialization of vectors.
   */
  void
  initialize_vector_velocity(VectorType & src) const;

  void
  initialize_vector_velocity_scalar(VectorType & src) const;

  void
  initialize_vector_pressure(VectorType & src) const;

  /*
   * Prescribe initial conditions using a specified analytical/initial solution function.
   */
  void
  prescribe_initial_conditions(VectorType & velocity,
                               VectorType & pressure,
                               double const time) const;

  /*
   * Time step calculation.
   */

  // Minimum element length h_min required for global CFL condition.
  double
  calculate_minimum_element_length() const;

  // Calculate time step size according to local CFL criterion
  double
  calculate_time_step_cfl(VectorType const & velocity,
                          double const       cfl,
                          double const       exponent_degree) const;

  /*
   * Special case: pure Dirichlet boundary conditions. For incompressible flows with pure Dirichlet
   * boundary conditions for the velocity (or more precisely with no Dirichlet boundary conditions
   * for the pressure), the pressure field is only defined up to an additive constant (since only
   * the pressure gradient appears in the incompressible Navier-Stokes equations. Different options
   * are available to fix the pressure level as described below.
   */

  // If an analytical solution is available: shift pressure so that the numerical pressure solution
  // coincides with the analytical pressure solution in an arbitrary point. Note that the parameter
  // 'time' is only needed for unsteady problems.
  void
  shift_pressure(VectorType & pressure, double const & time = 0.0) const;

  // If an analytical solution is available: shift pressure so that the numerical pressure solution
  // has a mean value identical to the "exact pressure solution" obtained by interpolation of
  // analytical solution. Note that the parameter 'time' is only needed for unsteady problems.
  void
  shift_pressure_mean_value(VectorType & pressure, double const & time = 0.0) const;

  /*
   * Computation of derived quantities which is needed for postprocessing but some of them are also
   * needed, e.g., for special splitting-type time integration schemes.
   */

  // vorticity
  void
  compute_vorticity(VectorType & dst, VectorType const & src) const;

  // divergence
  void
  compute_divergence(VectorType & dst, VectorType const & src) const;

  // velocity_magnitude
  void
  compute_velocity_magnitude(VectorType & dst, VectorType const & src) const;

  // vorticity_magnitude
  void
  compute_vorticity_magnitude(VectorType & dst, VectorType const & src) const;

  // streamfunction
  void
  compute_streamfunction(VectorType & dst, VectorType const & src) const;

  // Q criterion
  void
  compute_q_criterion(VectorType & dst, VectorType const & src) const;

  /*
   * Operators.
   */

  // mass matrix
  void
  apply_mass_matrix(VectorType & dst, VectorType const & src) const;

  void
  apply_mass_matrix_add(VectorType & dst, VectorType const & src) const;

  // body force term
  void
  evaluate_add_body_force_term(VectorType & dst, double const time) const;

  // convective term
  void
  evaluate_convective_term(VectorType & dst, VectorType const & src, Number const time) const;

  // pressure gradient term
  void
  evaluate_pressure_gradient_term(VectorType &       dst,
                                  VectorType const & src,
                                  double const       time) const;

  // velocity divergence
  void
  evaluate_velocity_divergence_term(VectorType &       dst,
                                    VectorType const & src,
                                    double const       time) const;

  // OIF splitting
  void
  evaluate_negative_convective_term_and_apply_inverse_mass_matrix(VectorType &       dst,
                                                                  VectorType const & src,
                                                                  Number const       time) const;

  // OIF splitting: interpolated velocity solution
  void
  evaluate_negative_convective_term_and_apply_inverse_mass_matrix(
    VectorType &       dst,
    VectorType const & src,
    Number const       time,
    VectorType const & solution_interpolated) const;

  // inverse velocity mass matrix
  void
  apply_inverse_mass_matrix(VectorType & dst, VectorType const & src) const;

  /*
   *  Update turbulence model, i.e., calculate turbulent viscosity.
   */
  void
  update_turbulence_model(VectorType const & velocity);

  /*
   * Projection step.
   */
  void
  update_projection_operator(VectorType const & velocity, double const time_step_size) const;

  unsigned int
  solve_projection(VectorType &       dst,
                   VectorType const & src,
                   bool const &       update_preconditioner) const;

  /*
   * Postprocessing.
   */
  double
  calculate_dissipation_convective_term(VectorType const & velocity, double const time) const;

  double
  calculate_dissipation_viscous_term(VectorType const & velocity) const;

  double
  calculate_dissipation_divergence_term(VectorType const & velocity) const;

  double
  calculate_dissipation_continuity_term(VectorType const & velocity) const;

protected:
  /*
   * Projection step.
   */
  void
  setup_projection_solver();

  bool
  unsteady_problem_has_to_be_solved() const;

  /*
   * List of input parameters.
   */
  InputParameters const & param;

  /*
   * Basic finite element ingredients.
   */
  std::shared_ptr<FESystem<dim>> fe_u;
  FE_DGQ<dim>                    fe_p;
  FE_DGQ<dim>                    fe_u_scalar;

  unsigned int                          mapping_degree;
  std::shared_ptr<MappingQGeneric<dim>> mapping;

  DoFHandler<dim> dof_handler_u;
  DoFHandler<dim> dof_handler_p;
  DoFHandler<dim> dof_handler_u_scalar;

  MatrixFree<dim, Number> matrix_free;

  AffineConstraints<double> constraint_u, constraint_p, constraint_u_scalar;

  /*
   * Special case: pure Dirichlet boundary conditions.
   */
  Point<dim>              first_point;
  types::global_dof_index dof_index_first_point;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_face_pairs;

  /*
   * User interface: Boundary conditions and field functions.
   */
  std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity;
  std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure;
  std::shared_ptr<FieldFunctions<dim>>      field_functions;

  /*
   * In case of projection type incompressible Navier-Stokes solvers this variable is needed to
   * solve the pressure Poisson equation. However, this variable is also needed in case of a
   * coupled solution approach. In that case, it is used for the block preconditioner (or more
   * precisely for the Schur-complement preconditioner and the preconditioner used to approximately
   * invert the Laplace operator).
   *
   * While the functions specified in BoundaryDescriptorLaplace are relevant for projection-type
   * solvers (pressure Poisson equation has to be solved), the function specified in
   * BoundaryDescriptorLaplace are irrelevant for a coupled solution approach (since the pressure
   * Laplace operator is only needed for preconditioning, and hence, only the homogeneous part of
   * the operator has to be evaluated).
   *
   */
  std::shared_ptr<Poisson::BoundaryDescriptor<dim>> boundary_descriptor_laplace;

  /*
   * Operator kernels.
   */
  std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> convective_kernel;
  std::shared_ptr<Operators::ViscousKernel<dim, Number>>    viscous_kernel;

  std::shared_ptr<Operators::DivergencePenaltyKernel<dim, Number>> div_penalty_kernel;
  std::shared_ptr<Operators::ContinuityPenaltyKernel<dim, Number>> conti_penalty_kernel;

  /*
   * Basic operators.
   */
  MassMatrixOperator<dim, dim, Number> mass_matrix_operator;
  ConvectiveOperator<dim, Number>      convective_operator;
  ViscousOperator<dim, Number>         viscous_operator;
  RHSOperator<dim, Number>             rhs_operator;
  GradientOperator<dim, Number>        gradient_operator;
  DivergenceOperator<dim, Number>      divergence_operator;

  DivergencePenaltyOperator<dim, Number> div_penalty_operator;
  ContinuityPenaltyOperator<dim, Number> conti_penalty_operator;

  /*
   * Linear(ized) momentum operator.
   */
  mutable MomentumOperator<dim, Number> momentum_operator;

  /*
   * Inverse mass matrix operator.
   */
  InverseMassMatrixOperator<dim, dim, Number> inverse_mass_velocity;
  InverseMassMatrixOperator<dim, 1, Number>   inverse_mass_velocity_scalar;

  /*
   * Projection operator.
   */
  typedef ProjectionOperator<dim, Number> PROJ_OPERATOR;
  std::shared_ptr<PROJ_OPERATOR>          projection_operator;

  /*
   * Projection solver.
   */

  // elementwise solver/preconditioner
  typedef Elementwise::OperatorBase<dim, Number, PROJ_OPERATOR> ELEMENTWISE_PROJ_OPERATOR;
  std::shared_ptr<ELEMENTWISE_PROJ_OPERATOR>                    elementwise_projection_operator;

  typedef Elementwise::PreconditionerBase<VectorizedArray<Number>> ELEMENTWISE_PRECONDITIONER;
  std::shared_ptr<ELEMENTWISE_PRECONDITIONER> elementwise_preconditioner_projection;

  // projection solver
  std::shared_ptr<IterativeSolverBase<VectorType>> projection_solver;
  std::shared_ptr<PreconditionerBase<Number>>      preconditioner_projection;

  /*
   * Calculators used to obtain derived quantities.
   */
  VorticityCalculator<dim, Number>         vorticity_calculator;
  DivergenceCalculator<dim, Number>        divergence_calculator;
  VelocityMagnitudeCalculator<dim, Number> velocity_magnitude_calculator;
  QCriterionCalculator<dim, Number>        q_criterion_calculator;

  /*
   * Postprocessor.
   */
  std::shared_ptr<Postprocessor> postprocessor;

  ConditionalOStream pcout;

private:
  /*
   * Initialization functions called during setup phase.
   */
  void
  initialize_boundary_descriptor_laplace();

  void
  initialize_dof_handler();

  void
  initialize_matrix_free();

  void
  initialize_operators();

  void
  initialize_momentum_operator(double const &     scaling_factor_time_derivative_term,
                               VectorType const & velocity);

  void
  initialize_turbulence_model();

  void
  initialize_calculators_for_derived_quantities();

  void
  initialization_pure_dirichlet_bc();

  void
  initialize_postprocessor();

  /*
   * LES turbulence modeling.
   */
  TurbulenceModel<dim, Number> turbulence_model;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_BASE_H_ */
