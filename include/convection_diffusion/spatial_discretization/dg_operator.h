/*
 * dg_convection_diffusion_operation.h
 *
 *  Created on: Aug 2, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_
#define INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_

// deal.II
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
#include "operators/mass_matrix_operator.h"
#include "operators/rhs_operator.h"

// solvers and preconditioners
#include "../../solvers_and_preconditioners/preconditioner/inverse_mass_matrix_preconditioner.h"
#include "../../solvers_and_preconditioners/preconditioner/jacobi_preconditioner.h"
#include "../../solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h"
#include "../preconditioners/multigrid_preconditioner.h"

// time integration and interface
#include "interface.h"

// postprocessor
#include "../postprocessor/postprocessor_base.h"
#include "operators/combined_operator.h"

using namespace dealii;

namespace ConvDiff
{
template<int dim, typename Number>
class DGOperator : public dealii::Subscriptor, public Interface::Operator<Number>
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
  DGOperator(parallel::TriangulationBase<dim> const &            triangulation,
             InputParameters const &                         param_in,
             std::shared_ptr<PostProcessorBase<dim, Number>> postprocessor_in);

  /*
   * Setup function. Initializes basic finite element components, matrix-free object, and basic
   * operators. This function does not perform the setup related to the solution of linear systems
   * of equations.
   */
  void
  setup(PeriodicFaces const                            periodic_face_pairs_in,
        std::shared_ptr<BoundaryDescriptor<dim>> const boundary_descriptor_in,
        std::shared_ptr<FieldFunctions<dim>> const     field_functions_in);

  /*
   * This function initializes operators, preconditioners, and solvers related to the solution of
   * linear systems of equation required for implicit formulations.
   */
  void
  setup_operators_and_solver(double const       scaling_factor_mass_matrix = -1.0,
                             VectorType const * velocity                   = nullptr);

  /*
   * Initialization of dof-vector.
   */
  void
  initialize_dof_vector(VectorType & src) const;

  /*
   * Initialization of velocity dof-vector (in case of numerical velocity field).
   */
  void
  initialize_dof_vector_velocity(VectorType & src) const;

  /*
   * Obtain velocity dof-vector by interpolation of specified analytical velocity field.
   */
  void
  interpolate_velocity(VectorType & velocity, double const time) const;

  /*
   * Obtain velocity dof-vector by L2-projection using the specified analytical velocity field.
   */
  void
  project_velocity(VectorType & velocity, double const time) const;

  /*
   * Prescribe initial conditions using a specified analytical function.
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
  evaluate_explicit_time_int(VectorType &       dst,
                             VectorType const & src,
                             double const       evaluation_time,
                             VectorType const * velocity = nullptr) const;

  /*
   * This function evaluates the convective term which is needed when using an explicit formulation
   * for the convective term.
   */
  void
  evaluate_convective_term(VectorType &       dst,
                           VectorType const & src,
                           double const       evaluation_time,
                           VectorType const * velocity = nullptr) const;

  /*
   * This function is called by OIF sub-stepping algorithm. It evaluates the convective term,
   * multiplies the result by -1.0 and applies the inverse mass matrix operator.
   */
  void
  evaluate_oif(VectorType &       dst,
               VectorType const & src,
               double const       evaluation_time,
               VectorType const * velocity = nullptr) const;

  /*
   * This function calculates the inhomogeneous parts of all operators arising e.g. from
   * inhomogeneous boundary conditions or the solution at previous instants of time occurring in the
   * discrete time derivative term.
   *
   * Note that the convective operator only has a contribution to the right-hand side if it is
   * formulated implicitly in time. In case of an explicit treatment the whole convective operator
   * (call function evaluate_convective_term() instead of rhs()) has to be added to the right-hand
   * side of the equations.
   */
  void
  rhs(VectorType &       dst,
      double const       evaluation_time = 0.0,
      VectorType const * velocity        = nullptr) const;

  /*
   * This function applies the mass matrix operator to the src-vector and writes the result to the
   * dst-vector.
   */
  void
  apply_mass_matrix(VectorType & dst, VectorType const & src) const;

  /*
   * This function applies the mass matrix operator to the src-vector and adds the result to the
   * dst-vector.
   */
  void
  apply_mass_matrix_add(VectorType & dst, VectorType const & src) const;

  /*
   * This function applies the convective operator to the src-vector and writes the result to the
   * dst-vector. It is needed for throughput measurements of the matrix-free implementation.
   */
  void
  apply_convective_term(VectorType & dst, VectorType const & src) const;

  void
  update_convective_term(double const evaluation_time, VectorType const * velocity = nullptr) const;

  /*
   * This function applies the diffusive operator to the src-vector and writes the result to the
   * dst-vector. It is needed for throughput measurements of the matrix-free implementation.
   */
  void
  apply_diffusive_term(VectorType & dst, VectorType const & src) const;

  /*
   * This function applies the combined mass-convection-diffusion operator to the src-vector
   * and writes the result to the dst-vector. It is needed for throughput measurements of the
   * matrix-free implementation.
   */
  void
  apply_conv_diff_operator(VectorType & dst, VectorType const & src) const;

  void
  update_conv_diff_operator(double const       evaluation_time,
                            double const       scaling_factor,
                            VectorType const * velocity = nullptr);

  /*
   * This function solves the linear system of equations in case of implicit time integration or
   * steady-state problems (potentially involving the mass matrix, convective, and diffusive
   * operators).
   */
  unsigned int
  solve(VectorType &       sol,
        VectorType const & rhs,
        bool const         update_preconditioner,
        double const       scaling_factor = -1.0,
        double const       time           = -1.0,
        VectorType const * velocity       = nullptr);

  /*
   * Calculate time step size according to local CFL criterion
   */

  // use numerical velocity field
  double
  calculate_time_step_cfl_numerical_velocity(VectorType const & velocity,
                                             double const       cfl,
                                             double const       exponent_degree) const;

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
  Mapping<dim> const &
  get_mapping() const;

  DoFHandler<dim> const &
  get_dof_handler() const;

  unsigned int
  get_polynomial_degree() const;

  types::global_dof_index
  get_number_of_dofs() const;

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
   * Dof index for velocity (in case of numerical velocity field)
   */
  int
  get_dof_index_velocity() const;

  /*
   * Initializes individual operators (mass, convective, viscous terms, rhs).
   */
  void
  setup_operators(double const scaling_factor_mass_matrix, VectorType const * velocity = nullptr);

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
  setup_postprocessor();

  /*
   * List of input parameters.
   */
  InputParameters const & param;

  /*
   * Basic finite element ingredients.
   */
  FE_DGQ<dim>                           fe;
  unsigned int                          mapping_degree;
  std::shared_ptr<MappingQGeneric<dim>> mapping;
  DoFHandler<dim>                       dof_handler;

  /*
   * Numerical velocity field.
   */
  std::shared_ptr<FESystem<dim>>   fe_velocity;
  std::shared_ptr<DoFHandler<dim>> dof_handler_velocity;

  /*
   * Constraints
   */
  AffineConstraints<double> constraint_matrix;

  /*
   * Matrix-free operator evaluation
   */
  MatrixFree<dim, Number> matrix_free;

  /*
   * Periodic face pairs: This variable is only needed when using a multigrid preconditioner
   */
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_face_pairs;

  /*
   * User interface: Boundary conditions and field functions.
   */
  std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor;
  std::shared_ptr<FieldFunctions<dim>>     field_functions;

  /*
   * Basic operators.
   */
  MassMatrixOperator<dim, Number>           mass_matrix_operator;
  InverseMassMatrixOperator<dim, 1, Number> inverse_mass_matrix_operator;
  ConvectiveOperator<dim, Number>           convective_operator;
  DiffusiveOperator<dim, Number>            diffusive_operator;
  RHSOperator<dim, Number>                  rhs_operator;

  /*
   * Merged operators
   */
  Operator<dim, Number> combined_operator;

  std::shared_ptr<PreconditionerBase<Number>> preconditioner;

  std::shared_ptr<IterativeSolverBase<VectorType>> iterative_solver;

  /*
   * Output to screen.
   */
  ConditionalOStream pcout;

  /*
   * Postprocessor.
   */
  std::shared_ptr<PostProcessorBase<dim, Number>> postprocessor;
};

} // namespace ConvDiff

#endif /* INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_ */
