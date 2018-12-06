/*
 * dg_navier_stokes_coupled_solver.h
 *
 *  Created on: Jun 27, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_COUPLED_SOLVER_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_COUPLED_SOLVER_H_

#include "../../incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_base.h"
#include "solvers_and_preconditioners/newton/newton_solver.h"

#include "../interface_space_time/operator.h"
#include "../preconditioners/block_preconditioner.h"

namespace IncNS
{
template<int dim, int degree_u, int degree_p = degree_u - 1, typename Number = double>
class DGNavierStokesCoupled : public DGNavierStokesBase<dim, degree_u, degree_p, Number>,
                              public Interface::OperatorCoupled<Number>
{
private:
  typedef DGNavierStokesBase<dim, degree_u, degree_p, Number> BASE;

  typedef typename BASE::Postprocessor Postprocessor;

  typedef typename BASE::VectorType VectorType;

  typedef LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef DGNavierStokesCoupled<dim, degree_u, degree_p, Number> THIS;

public:
  /*
   * Constructor.
   */
  DGNavierStokesCoupled(parallel::distributed::Triangulation<dim> const & triangulation,
                        InputParameters<dim> const &                      parameters_in,
                        std::shared_ptr<Postprocessor>                    postprocessor_in);

  /*
   * Destructor.
   */
  virtual ~DGNavierStokesCoupled();

  void
  setup(std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> const
                                                        periodic_face_pairs,
        std::shared_ptr<BoundaryDescriptorU<dim>> const boundary_descriptor_velocity,
        std::shared_ptr<BoundaryDescriptorP<dim>> const boundary_descriptor_pressure,
        std::shared_ptr<FieldFunctions<dim>> const      field_functions,
        std::shared_ptr<AnalyticalSolution<dim>> const  analytical_solution);

  void
  setup_solvers(double const & scaling_factor_time_derivative_term = 1.0);

  /*
   *  Update divergence penalty operator by recalculating the penalty parameter
   *  which depends on the current velocity field
   */
  void
  update_divergence_penalty_operator(VectorType const & velocity) const;

  /*
   *  Update continuity penalty operator by recalculating the penalty parameter
   *  which depends on the current velocity field
   */
  void
  update_continuity_penalty_operator(VectorType const & velocity) const;


  /*
   * Initialization of vectors.
   */
  void
  initialize_block_vector_velocity_pressure(BlockVectorType & src) const;

  /*
   * The implementation of the Newton solver requires that the underlying operator implements a
   * function called "initialize_vector_for_newton_solver".
   */
  void
  initialize_vector_for_newton_solver(BlockVectorType & src) const;

  /*
   * Returns whether a non-linear problem has to be solved.
   */
  bool
  nonlinear_problem_has_to_be_solved() const;

  /*
   * Return whether an unsteady problem has to be solved.
   */
  bool
  unsteady_problem_has_to_be_solved() const;

  /*
   * Setters and getters.
   */

  /*
   *  This function sets the variable scaling_factor_continuity, and also the related scaling factor
   * for the pressure gradient operator.
   */
  void
  set_scaling_factor_continuity(double const scaling_factor);

  void
  set_sum_alphai_ui(VectorType const * vector = nullptr);

  void
  set_solution_linearization(BlockVectorType const & solution_linearization);

  LinearAlgebra::distributed::Vector<Number> const &
  get_velocity_linearization() const;

  CompatibleLaplaceOperatorData<dim> const
  get_compatible_laplace_operator_data() const;

  /*
   * Stokes equations or convective term treated explicitly: solve linear system of equations
   */

  /*
   *  This function solves the linear Stokes problem (steady/unsteady Stokes or unsteady
   * Navier-Stokes with explicit treatment of convective term). The parameter
   * scaling_factor_mass_matrix_term has to be specified for unsteady problem and can be omitted for
   * steady problems.
   */
  unsigned int
  solve_linear_stokes_problem(BlockVectorType &       dst,
                              BlockVectorType const & src,
                              double const &          scaling_factor_mass_matrix_term = 1.0);


  /*
   *  For the linear solver, the operator of the linear(ized) problem has to implement a function
   * called vmult() which calculates the matrix-vector product for the linear(ized) problem.
   */
  void
  vmult(BlockVectorType & dst, BlockVectorType const & src) const;

  /*
   * This function calculates the right-hand side of the steady Stokes problem, or unsteady Stokes
   * problem, or unsteady Navier-Stokes problem with explicit treatment of the convective term. The
   * parameters 'src' and 'eval_time' have to be specified for unsteady problems. For steady
   * problems these parameters are omitted.
   */
  void
  rhs_stokes_problem(BlockVectorType & dst, double const & eval_time = 0.0) const;


  /*
   * Convective term treated implicitly: solve non-linear system of equations
   */

  /*
   * This function solves the nonlinear problem for steady problems.
   */
  void
  solve_nonlinear_steady_problem(BlockVectorType & dst,
                                 unsigned int &    newton_iterations,
                                 unsigned int &    linear_iterations);

  /*
   * This function solves the nonlinear problem for unsteady problems.
   */
  void
  solve_nonlinear_problem(BlockVectorType &  dst,
                          VectorType const & sum_alphai_ui,
                          double const &     evaluation_time,
                          double const &     scaling_factor_mass_matrix_term,
                          unsigned int &     newton_iterations,
                          unsigned int &     linear_iterations);


  /*
   * This function evaluates the nonlinear residual.
   */
  void
  evaluate_nonlinear_residual(BlockVectorType & dst, BlockVectorType const & src) const;

  /*
   *  This function evaluates the nonlinear residual of the steady Navier-Stokes equations.
   *  This function has to be implemented seperately (for example, the convective term will be
   *  evaluated in case of the Navier-Stokes equations and the time-derivative term is never
   * evaluated).
   */
  void
  evaluate_nonlinear_residual_steady(BlockVectorType & dst, BlockVectorType const & src) const;

  /*
   * Postprocessing.
   */
  void
  do_postprocessing(VectorType const & velocity,
                    VectorType const & pressure,
                    double const       time,
                    unsigned int const time_step_number) const;

  void
  do_postprocessing_steady_problem(VectorType const & velocity, VectorType const & pressure) const;

private:
  void
  initialize_momentum_operator(double const & scaling_factor_time_derivative_term = 1.0);

  void
  initialize_preconditioner_coupled();

  void
  initialize_solver_coupled();

  /*
   * Coupled system of equations (operator, preconditioner, solver).
   */
  typedef BlockPreconditioner<dim, degree_u, degree_p, Number> Preconditioner;
  friend class BlockPreconditioner<dim, degree_u, degree_p, Number>;

  MomentumOperator<dim, degree_u, Number> momentum_operator;

  VectorType mutable temp_vector;
  VectorType const *      sum_alphai_ui;
  BlockVectorType const * vector_linearization;

  std::shared_ptr<Preconditioner> preconditioner;

  std::shared_ptr<IterativeSolverBase<BlockVectorType>> linear_solver;

  std::shared_ptr<NewtonSolver<BlockVectorType, THIS, THIS, IterativeSolverBase<BlockVectorType>>>
    newton_solver;

  double evaluation_time;
  double scaling_factor_time_derivative_term;

  double scaling_factor_continuity;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_COUPLED_SOLVER_H_ \
        */
