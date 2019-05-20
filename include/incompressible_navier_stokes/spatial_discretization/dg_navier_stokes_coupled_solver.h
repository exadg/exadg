/*
 * dg_navier_stokes_coupled_solver.h
 *
 *  Created on: Jun 27, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_COUPLED_SOLVER_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_COUPLED_SOLVER_H_

#include "dg_navier_stokes_base.h"
#include "interface.h"

#include "../../poisson/preconditioner/multigrid_preconditioner.h"
#include "../../poisson/spatial_discretization/laplace_operator.h"
#include "../../solvers_and_preconditioners/newton/newton_solver.h"
#include "../../solvers_and_preconditioners/util/check_multigrid.h"
#include "../preconditioners/compatible_laplace_multigrid_preconditioner.h"
#include "../preconditioners/compatible_laplace_operator.h"
#include "../preconditioners/multigrid_preconditioner.h"
#include "../preconditioners/pressure_convection_diffusion_operator.h"
#include "momentum_operator.h"

namespace IncNS
{
// forward declaration
template<int dim, typename Number>
class DGNavierStokesCoupled;

template<int dim, typename Number>
class BlockPreconditioner
{
private:
  typedef LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef DGNavierStokesCoupled<dim, Number> PDEOperator;

public:
  BlockPreconditioner() : pde_operator(nullptr)
  {
  }

  void
  initialize(PDEOperator * pde_operator_in)
  {
    pde_operator = pde_operator_in;
  }

  void
  update(PDEOperator const * op)
  {
    pde_operator->update_block_preconditioner(op);
  }

  void
  vmult(BlockVectorType & dst, BlockVectorType const & src) const
  {
    pde_operator->apply_block_preconditioner(dst, src);
  }

  PDEOperator * pde_operator;
};

template<int dim, typename Number = double>
class DGNavierStokesCoupled : public DGNavierStokesBase<dim, Number>,
                              public Interface::OperatorCoupled<Number>
{
private:
  typedef DGNavierStokesBase<dim, Number> BASE;

  typedef typename BASE::MultigridNumber MultigridNumber;

  typedef typename BASE::Postprocessor Postprocessor;

  typedef typename BASE::VectorType VectorType;

  typedef LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef DGNavierStokesCoupled<dim, Number> THIS;

public:
  /*
   * Constructor.
   */
  DGNavierStokesCoupled(parallel::Triangulation<dim> const & triangulation,
                        InputParameters const &              parameters_in,
                        std::shared_ptr<Postprocessor>       postprocessor_in);

  /*
   * Destructor.
   */
  virtual ~DGNavierStokesCoupled();

  void
  setup(std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> const
                                                        periodic_face_pairs,
        std::shared_ptr<BoundaryDescriptorU<dim>> const boundary_descriptor_velocity,
        std::shared_ptr<BoundaryDescriptorP<dim>> const boundary_descriptor_pressure,
        std::shared_ptr<FieldFunctions<dim>> const      field_functions);

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
                              bool const &            update_preconditioner,
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
                                 bool const &      update_preconditioner,
                                 unsigned int &    newton_iterations,
                                 unsigned int &    linear_iterations);

  /*
   * This function solves the nonlinear problem for unsteady problems.
   */
  void
  solve_nonlinear_problem(BlockVectorType &  dst,
                          VectorType const & sum_alphai_ui,
                          double const &     evaluation_time,
                          bool const &       update_preconditioner,
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

  /*
   * Block preconditioner
   */
  void
  update_block_preconditioner(THIS const * /*operator*/);

  void
  apply_block_preconditioner(BlockVectorType & dst, BlockVectorType const & src) const;

private:
  void
  initialize_momentum_operator(double const & scaling_factor_time_derivative_term = 1.0);

  void
  initialize_solver_coupled();

  /*
   * Block preconditioner
   */
  void
  initialize_block_preconditioner();

  void
  initialize_vectors();

  void
  initialize_preconditioner_velocity_block();

  void
  setup_multigrid_preconditioner_momentum();

  void
  setup_iterative_solver_momentum();

  void
  initialize_preconditioner_pressure_block();

  CompatibleLaplaceOperatorData<dim> const
  get_compatible_laplace_operator_data() const;

  void
  setup_multigrid_preconditioner_schur_complement();

  void
  setup_iterative_solver_schur_complement();

  void
  setup_pressure_convection_diffusion_operator();

  void
  apply_preconditioner_velocity_block(VectorType & dst, VectorType const & src) const;

  void
  apply_preconditioner_pressure_block(VectorType & dst, VectorType const & src) const;

  void
  apply_inverse_negative_laplace_operator(VectorType & dst, VectorType const & src) const;

  /*
   * Newton-Krylov solver for (non-)linear problem
   */

  // Linear(ized) momentum operator
  MomentumOperator<dim, Number> momentum_operator;

  // temporary vector needed to evaluate both the nonlinear residual and the linearized operator
  VectorType mutable temp_vector;

  // vector needed to evaluate the nonlinear residual (which stays constant over all Newton
  // iterations)
  VectorType const * sum_alphai_ui;

  // linear solver
  std::shared_ptr<IterativeSolverBase<BlockVectorType>> linear_solver;

  // Newton solver
  std::shared_ptr<NewtonSolver<BlockVectorType, THIS, THIS, IterativeSolverBase<BlockVectorType>>>
    newton_solver;

  // time at which the linear/nonlinear operators are to be evaluated
  double evaluation_time;

  // scaling factor in front of the mass matrix operator (gamma0/dt)
  double scaling_factor_time_derivative_term;

  double scaling_factor_continuity;

  /*
   * Block preconditioner for linear(ized) problem
   */
  typedef BlockPreconditioner<dim, Number> Preconditioner;
  Preconditioner                           block_preconditioner;

  // preconditioner velocity/momentum block
  std::shared_ptr<PreconditionerBase<Number>> preconditioner_momentum;

  std::shared_ptr<IterativeSolverBase<VectorType>> solver_velocity_block;

  // preconditioner pressure/Schur-complement block
  std::shared_ptr<PreconditionerBase<Number>> multigrid_preconditioner_schur_complement;
  std::shared_ptr<PreconditionerBase<Number>> inv_mass_matrix_preconditioner_schur_complement;

  std::shared_ptr<PressureConvectionDiffusionOperator<dim, Number>>
    pressure_convection_diffusion_operator;

  std::shared_ptr<Poisson::LaplaceOperator<dim, Number>> laplace_operator_classical;

  std::shared_ptr<CompatibleLaplaceOperator<dim, Number>> laplace_operator_compatible;

  std::shared_ptr<IterativeSolverBase<VectorType>> solver_pressure_block;

  // temporary vectors that are necessary when using preconditioners of block-triangular type
  VectorType mutable vec_tmp_pressure;
  VectorType mutable vec_tmp_velocity, vec_tmp_velocity_2;

  // temporary vectors that are necessary when applying the Schur-complement preconditioner (scp)
  VectorType mutable tmp_scp_pressure;
  VectorType mutable tmp_scp_velocity, tmp_scp_velocity_2;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_COUPLED_SOLVER_H_ \
        */
