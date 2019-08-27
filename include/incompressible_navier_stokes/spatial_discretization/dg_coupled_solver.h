/*
 * dg_coupled_solver.h
 *
 *  Created on: Jun 27, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_COUPLED_SOLVER_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_COUPLED_SOLVER_H_

#include "dg_navier_stokes_base.h"

#include "../../convection_diffusion/spatial_discretization/operators/combined_operator.h"

#include "../../poisson/preconditioner/multigrid_preconditioner.h"
#include "../../poisson/spatial_discretization/laplace_operator.h"
#include "../../solvers_and_preconditioners/newton/newton_solver.h"
#include "../../solvers_and_preconditioners/util/check_multigrid.h"
#include "../preconditioners/compatible_laplace_multigrid_preconditioner.h"
#include "../preconditioners/compatible_laplace_operator.h"
#include "../preconditioners/multigrid_preconditioner_momentum.h"

namespace IncNS
{
// forward declaration
template<int dim, typename Number>
class DGNavierStokesCoupled;

template<int dim, typename Number>
class NonlinearOperatorCoupled
{
private:
  typedef LinearAlgebra::distributed::Vector<Number>      VectorType;
  typedef LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef DGNavierStokesCoupled<dim, Number> PDEOperator;

public:
  NonlinearOperatorCoupled()
    : pde_operator(nullptr), rhs_vector(nullptr), time(0.0), scaling_factor_mass_matrix(1.0)
  {
  }

  void
  initialize(PDEOperator const & pde_operator)
  {
    this->pde_operator = &pde_operator;
  }

  void
  update(VectorType const & rhs_vector, double const & time, double const & scaling_factor)
  {
    this->rhs_vector                 = &rhs_vector;
    this->time                       = time;
    this->scaling_factor_mass_matrix = scaling_factor;
  }

  /*
   * The implementation of the Newton solver requires a function called
   * 'initialize_vector_for_newton_solver'.
   */
  void
  initialize_vector_for_newton_solver(BlockVectorType & src) const
  {
    pde_operator->initialize_block_vector_velocity_pressure(src);
  }

  /*
   * The implementation of the Newton solver requires a function called
   * 'evaluate_nonlinear_residual'.
   */
  void
  evaluate_nonlinear_residual(BlockVectorType & dst, BlockVectorType const & src) const
  {
    pde_operator->evaluate_nonlinear_residual(
      dst, src, rhs_vector, time, scaling_factor_mass_matrix);
  }

private:
  PDEOperator const * pde_operator;

  VectorType const * rhs_vector;
  double             time;
  double             scaling_factor_mass_matrix;
};

template<int dim, typename Number>
class LinearOperatorCoupled : public dealii::Subscriptor
{
private:
  typedef LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef DGNavierStokesCoupled<dim, Number> PDEOperator;

public:
  LinearOperatorCoupled()
    : dealii::Subscriptor(), pde_operator(nullptr), time(0.0), scaling_factor_mass_matrix(1.0)
  {
  }

  void
  initialize(PDEOperator const & pde_operator)
  {
    this->pde_operator = &pde_operator;
  }

  /*
   * The implementation of the Newton solver requires a function called
   * 'set_solution_linearization'.
   */
  void
  set_solution_linearization(BlockVectorType const & solution_linearization) const
  {
    pde_operator->set_velocity_ptr(solution_linearization.block(0));
  }

  void
  update(double const & time, double const & scaling_factor)
  {
    this->time                       = time;
    this->scaling_factor_mass_matrix = scaling_factor;
  }

  /*
   * The implementation of linear solvers in deal.ii requires that a function called 'vmult' is
   * provided.
   */
  void
  vmult(BlockVectorType & dst, BlockVectorType const & src) const
  {
    pde_operator->apply_linearized_problem(dst, src, time, scaling_factor_mass_matrix);
  }

private:
  PDEOperator const * pde_operator;

  double time;
  double scaling_factor_mass_matrix;
};

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
  update()
  {
    pde_operator->update_block_preconditioner();
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
  typedef DGNavierStokesBase<dim, Number>    Base;
  typedef DGNavierStokesCoupled<dim, Number> This;

  typedef typename Base::MultigridNumber MultigridNumber;
  typedef typename Base::Postprocessor   Postprocessor;
  typedef typename Base::VectorType      VectorType;

  typedef LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

public:
  /*
   * Constructor.
   */
  DGNavierStokesCoupled(parallel::TriangulationBase<dim> const & triangulation,
                        InputParameters const &              parameters,
                        std::shared_ptr<Postprocessor>       postprocessor);

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
  setup_solvers(double const & scaling_factor_time_derivative_term, VectorType const & velocity);

  /*
   *  Update divergence penalty operator by recalculating the penalty parameter
   *  which depends on the current velocity field
   */
  void
  update_divergence_penalty_operator(VectorType const & velocity);

  /*
   *  Update continuity penalty operator by recalculating the penalty parameter
   *  which depends on the current velocity field
   */
  void
  update_continuity_penalty_operator(VectorType const & velocity);


  /*
   * Initialization of vectors.
   */
  void
  initialize_block_vector_velocity_pressure(BlockVectorType & src) const;

  /*
   * Setters and getters.
   */

  /*
   *  This function sets the variable scaling_factor_continuity, and also the related scaling factor
   * for the pressure gradient operator.
   */
  void
  set_scaling_factor_continuity(double const scaling_factor);

  /*
   * Stokes equations or convective term treated explicitly: solve linear system of equations
   */

  /*
   * This function solves the linear Stokes problem (Stokes equations or Navier-Stokes
   * equations with an explicit treatment of the convective term). The parameter
   * scaling_factor_mass_matrix_term has to be specified for unsteady problem and
   * can be omitted for steady problems.
   */
  unsigned int
  solve_linear_stokes_problem(BlockVectorType &       dst,
                              BlockVectorType const & src,
                              bool const &            update_preconditioner,
                              double const &          time                            = 0.0,
                              double const &          scaling_factor_mass_matrix_term = 1.0);

  /*
   * Convective term treated implicitly: solve non-linear system of equations
   */

  /*
   * This function solves the nonlinear problem for steady problems.
   */
  void
  solve_nonlinear_steady_problem(BlockVectorType &  dst,
                                 VectorType const & rhs_vector,
                                 bool const &       update_preconditioner,
                                 unsigned int &     newton_iterations,
                                 unsigned int &     linear_iterations);

  /*
   * This function solves the nonlinear problem for unsteady problems.
   */
  void
  solve_nonlinear_problem(BlockVectorType &  dst,
                          VectorType const & rhs_vector,
                          double const &     time,
                          bool const &       update_preconditioner,
                          double const &     scaling_factor_mass_matrix_term,
                          unsigned int &     newton_iterations,
                          unsigned int &     linear_iterations);


  /*
   * This function evaluates the nonlinear residual.
   */
  void
  evaluate_nonlinear_residual(BlockVectorType &       dst,
                              BlockVectorType const & src,
                              VectorType const *      rhs_vector,
                              double const &          time,
                              double const &          scaling_factor_mass_matrix) const;

  /*
   * This function evaluates the nonlinear residual of the steady Navier-Stokes equations.
   * This function has to be implemented separately (for example, the convective term will be
   * evaluated in case of the Navier-Stokes equations and the time-derivative term is never
   * evaluated).
   */
  void
  evaluate_nonlinear_residual_steady(BlockVectorType &       dst,
                                     BlockVectorType const & src,
                                     double const &          time) const;

  /*
   * This function calculates the matrix-vector product for the linear(ized) problem.
   */
  void
  apply_linearized_problem(BlockVectorType &       dst,
                           BlockVectorType const & src,
                           double const &          time,
                           double const &          scaling_factor_mass_matrix) const;

  /*
   * This function calculates the right-hand side of the steady Stokes problem, or unsteady Stokes
   * problem, or unsteady Navier-Stokes problem with explicit treatment of the convective term. The
   * parameters 'src' and 'time' have to be specified for unsteady problems. For steady
   * problems these parameters are omitted.
   */
  void
  rhs_stokes_problem(BlockVectorType & dst, double const & time = 0.0) const;

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
  update_block_preconditioner();

  void
  apply_block_preconditioner(BlockVectorType & dst, BlockVectorType const & src) const;

private:
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

  // temporary vector needed to evaluate both the nonlinear residual and the linearized operator
  VectorType mutable temp_vector;

  double scaling_factor_continuity;

  // Nonlinear operator
  NonlinearOperatorCoupled<dim, Number> nonlinear_operator;

  // Newton solver
  std::shared_ptr<NewtonSolver<BlockVectorType,
                               NonlinearOperatorCoupled<dim, Number>,
                               LinearOperatorCoupled<dim, Number>,
                               IterativeSolverBase<BlockVectorType>>>
    newton_solver;

  // Linear operator
  LinearOperatorCoupled<dim, Number> linear_operator;

  // Linear solver
  std::shared_ptr<IterativeSolverBase<BlockVectorType>> linear_solver;

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

  std::shared_ptr<ConvDiff::Operator<dim, Number>> pressure_conv_diff_operator;

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

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_COUPLED_SOLVER_H_ \
        */
