/*
 * dg_convection_diffusion_operation.h
 *
 *  Created on: Aug 2, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_
#define INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/vector_tools.h>

// user interface
#include "../user_interface/boundary_descriptor.h"
#include "../user_interface/field_functions.h"
#include "../user_interface/input_parameters.h"

// operators
#include "../../operators/inverse_mass_matrix.h"
#include "operators/convection_diffusion_operator.h"
#include "operators/convection_diffusion_operator_efficiency.h"

// preconditioners
#include "../../solvers_and_preconditioners/preconditioner/inverse_mass_matrix_preconditioner.h"
#include "../../solvers_and_preconditioners/preconditioner/jacobi_preconditioner.h"
#include "../preconditioners/multigrid_preconditioner.h"

// solvers
#include "../../solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h"

// interface space-time
#include "../interface_space_time/operator.h"

// time integration
#include "time_integration/interpolate.h"
#include "time_integration/time_step_calculation.h"

// postprocessor
#include "../postprocessor/postprocessor.h"

using namespace dealii;

namespace ConvDiff
{
template<int dim, int degree, typename Number>
class DGOperation : public dealii::Subscriptor, public Interface::Operator<Number>
{
private:
  typedef float MultigridNumber;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    PeriodicFaces;

public:
  /*
   * Constructor.
   */
  DGOperation(parallel::Triangulation<dim> const &        triangulation,
              InputParameters const &                     param_in,
              std::shared_ptr<PostProcessor<dim, degree>> postprocessor_in);

  /*
   * Setup function. Initializes basic finite element components, matrix-free object, and basic
   * operators. This function does not perform the setup related to the solution of linear systems
   * of equations.
   */
  void
  setup(PeriodicFaces const                            periodic_face_pairs_in,
        std::shared_ptr<BoundaryDescriptor<dim>> const boundary_descriptor_in,
        std::shared_ptr<FieldFunctions<dim>> const     field_functions_in,
        std::shared_ptr<AnalyticalSolution<dim>> const analytical_solution_in);

  /*
   * This function initializes operators, preconditioners, and solvers related to the solution of
   * linear systems of equation required for implicit formulations.
   */
  void
  setup_solver(double const scaling_factor_time_derivative_term_in = -1.0);

  /*
   * Initialization of dof-vector.
   */
  void
  initialize_dof_vector(VectorType & src) const;

  /*
   * Prescribe initial conditions using a specified analytical/initial solution function.
   */
  void
  prescribe_initial_conditions(VectorType & src, double const evaluation_time) const;

  /*
   * This function is used in case of explicit time integration:
   *
   * It evaluates the right-hand side operator, the convective and diffusive terms (subsequently
   * multiplied by -1.0 in order to shift these terms to the right-hand side of the equations) and
   * finally applies the inverse mass matrix operator.
   */
  void
  evaluate(VectorType & dst, VectorType const & src, double const evaluation_time) const;

  /*
   * This function evaluates the convective term which is needed when using an explicit formulation
   * for the convective term.
   */
  void
  evaluate_convective_term(VectorType &       dst,
                           VectorType const & src,
                           double const       evaluation_time) const;

  /*
   * This function is called by OIF sub-stepping algorithm. It evaluates the convective term,
   * multiplies the result by -1.0 and applies the inverse mass matrix operator.
   */
  void
  evaluate_negative_convective_term_and_apply_inverse_mass_matrix(
    VectorType &       dst,
    VectorType const & src,
    double const       evaluation_time) const;

  /*
   * This function calculates the inhomogeneous parts of all operators arising e.g. from
   * inhomogeneous boundary conditions or the solution at previous instants of time occuring in the
   * discrete time derivate term.
   *
   * Note that the convective operator only has a contribution to the right-hand side if it is
   * formulated implicitly in time. In case of an explicit treatment the whole convective operator
   * (call function evaluate() instead of rhs()) has to be added to the right-hand side of the
   * equations.
   */
  void
  rhs(VectorType & dst, double const evaluation_time = 0.0) const;

  /*
   * This function applies the mass matrix operator to the src-vector and adds the result to the
   * dst-vector.
   */
  void
  apply_mass_matrix_add(VectorType & dst, VectorType const & src) const;

  /*
   * This function solves the linear system of equations in case of implicit time integration for
   * the diffusive term (and the convective term).
   */
  unsigned int
  solve(VectorType &       sol,
        VectorType const & rhs,
        bool const         update_preconditioner,
        double const       scaling_factor = -1.0,
        double const       time           = -1.0);

  /*
   * Calculate time step size according to local CFL criterion
   */

  // use numerical velocity field
  double
  calculate_time_step_cfl_numerical_velocity(double const cfl, double const exponent_degree) const;

  // use analytical velocity field
  double
  calculate_time_step_cfl_analytical_velocity(double const time,
                                              double const cfl,
                                              double const exponent_degree) const;

  // Calculate maximum velocity (required for global CFL criterion).
  double
  calculate_maximum_velocity(double const time) const;

  // Calculate minimum element length (required for global CFL criterion).
  double
  calculate_minimum_element_length() const;

  /*
   * Setters and getters.
   */
  MatrixFree<dim, Number> const &
  get_data() const;

  Mapping<dim> const &
  get_mapping() const;

  DoFHandler<dim> const &
  get_dof_handler() const;

  unsigned int
  get_polynomial_degree() const;

  unsigned int
  get_number_of_dofs() const;

  /*
   * Numerical velocity field.
   */
  void
  set_velocity(VectorType const & velocity) const;

  void
  set_velocities_and_times(std::vector<VectorType const *> & velocities_in,
                           std::vector<double> &             times_in) const;

  /*
   * Perform postprocessing for a given solution vector.
   */
  void
  do_postprocessing(VectorType const & solution,
                    double const       time             = 0.0,
                    int const          time_step_number = -1) const;

private:
  /*
   * Initializes DoFHandlers.
   */
  void
  create_dofs();

  /*
   * Initializes MatrixFree-object.
   */
  void
  initialize_matrix_free();

  /*
   * Initializes individual operators (mass, convective, viscous terms, rhs).
   */
  void
  setup_operators();

  /*
   * Initializes convection-diffusion operator which is the operator required for the solution of
   * linear systems of equations.
   */
  void
  initialize_convection_diffusion_operator(
    double const scaling_factor_time_derivative_term_in = -1.0);

  /*
   * Initializes the preconditioner.
   */
  void
  initialize_preconditioner();

  /*
   * Initializes the solver.
   */
  void
  initialize_solver();

  /*
   * Initializes the postprocessor.
   */
  void
  setup_postprocessor(std::shared_ptr<AnalyticalSolution<dim>> analytical_solution);

  /*
   * Basic finite element ingredients.
   */
  FE_DGQ<dim>          fe;
  MappingQGeneric<dim> mapping;
  DoFHandler<dim>      dof_handler;

  AffineConstraints<double> constraint_matrix;


  MatrixFree<dim, Number> data;

  /*
   * List of input parameters.
   */
  InputParameters const & param;

  /*
   * User interface: Boundary conditions and field functions.
   */
  std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor;
  std::shared_ptr<FieldFunctions<dim>>     field_functions;

  /*
   * Output to screen.
   */
  ConditionalOStream pcout;

  /*
   * Basic operators.
   */
  MassMatrixOperator<dim, degree, Number>           mass_matrix_operator;
  InverseMassMatrixOperator<dim, degree, Number, 1> inverse_mass_matrix_operator;
  ConvectiveOperator<dim, degree, degree, Number>   convective_operator;
  DiffusiveOperator<dim, degree, Number>            diffusive_operator;
  RHSOperator<dim, degree, Number>                  rhs_operator;

  /*
   * Numerical velocity field.
   */
  std::shared_ptr<FESystem<dim>>   fe_velocity;
  std::shared_ptr<DoFHandler<dim>> dof_handler_velocity;

  mutable std::vector<VectorType const *> velocities;
  mutable std::vector<double>             times;
  mutable VectorType                      velocity;

  /*
   * Solution of linear systems of equations
   */
  ConvectionDiffusionOperator<dim, degree, Number> conv_diff_operator;

  std::shared_ptr<PreconditionerBase<Number>> preconditioner;

  std::shared_ptr<IterativeSolverBase<VectorType>> iterative_solver;


  // TODO This variable is only needed when using a multigrid preconditioner
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_face_pairs;

  /*
   * Convection-diffusion operator for runtime optimization (merged operators including
   * rhs-operator). This operator can only be used for explicit time integration.
   */
  ConvectionDiffusionOperatorEfficiency<dim, degree, Number>
    convection_diffusion_operator_efficiency;

  /*
   * Postprocessor.
   */
  std::shared_ptr<PostProcessor<dim, degree>> postprocessor;
};

} // namespace ConvDiff

#endif /* INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_ */
