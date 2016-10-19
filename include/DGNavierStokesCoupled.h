/*
 * DGNavierStokesCoupled.h
 *
 *  Created on: Jun 27, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_DGNAVIERSTOKESCOUPLED_H_
#define INCLUDE_DGNAVIERSTOKESCOUPLED_H_

using namespace dealii;

#include "PreconditionerNavierStokes.h"
#include "NewtonSolver.h"
#include "DGNavierStokesBase.h"

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
class DGNavierStokesCoupled;

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
class DGNavierStokesCoupled : public DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>
{
public:
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::value_type value_type;

  DGNavierStokesCoupled(parallel::distributed::Triangulation<dim> const &triangulation,
                        InputParametersNavierStokes<dim> const          &parameter)
    :
    DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>(triangulation,parameter),
    sum_alphai_ui(nullptr),
    vector_linearization(nullptr)
  {}

  void setup_solvers ();

  // initialization of vectors
  void initialize_block_vector_velocity_pressure(parallel::distributed::BlockVector<value_type> &src) const
  {
    // velocity(1 block) + pressure(1 block)
    src.reinit(2);

    this->data.initialize_dof_vector(src.block(0),
        static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector>::type >
    (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector::velocity));

    this->data.initialize_dof_vector(src.block(1),
        static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector>::type >
    (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector::pressure));

    src.collect_sizes();
  }

  void initialize_vector_for_newton_solver(parallel::distributed::BlockVector<value_type> &src) const
  {
    initialize_block_vector_velocity_pressure(src);
  }

  bool nonlinear_problem_has_to_be_solved() const
  {
      return ( this->param.equation_type == EquationType::NavierStokes &&
               (this->param.problem_type == ProblemType::Steady ||
                this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit) );
  }

  bool unsteady_problem_has_to_be_solved() const
  {
    return ( this->param.problem_type == ProblemType::Unsteady );
  }

  unsigned int solve_linear_stokes_problem (parallel::distributed::BlockVector<value_type>       &dst,
                                            parallel::distributed::BlockVector<value_type> const &src);

  void solve_nonlinear_problem (parallel::distributed::BlockVector<value_type>  &dst,
                                parallel::distributed::Vector<value_type> const &sum_alphai_ui,
                                unsigned int                                    &newton_iterations,
                                double                                          &average_linear_iterations);

  void solve_nonlinear_steady_problem (parallel::distributed::BlockVector<value_type>  &dst,
                                       unsigned int                                    &newton_iterations,
                                       double                                          &average_linear_iterations);

  void apply_linearized_problem (parallel::distributed::BlockVector<value_type> &dst,
                                 parallel::distributed::BlockVector<value_type> const &src) const;

  void vmult (parallel::distributed::BlockVector<value_type> &dst,
              parallel::distributed::BlockVector<value_type> const &src) const;

  void rhs_stokes_problem (parallel::distributed::BlockVector<value_type>  &dst,
                           parallel::distributed::Vector<value_type> const *src = nullptr) const;

  void evaluate_nonlinear_residual (parallel::distributed::BlockVector<value_type>       &dst,
                                    parallel::distributed::BlockVector<value_type> const &src);

  void set_solution_linearization(parallel::distributed::BlockVector<value_type> const *solution_linearization)
  {
    vector_linearization = solution_linearization;
  }

  parallel::distributed::Vector<value_type> const * get_velocity_linearization() const
  {
    if(vector_linearization != nullptr)
    {
      return &vector_linearization->block(0);
    }
    else
    {
      return nullptr;
    }
  }

private:
  friend class BlockPreconditionerNavierStokes<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>;

  parallel::distributed::Vector<value_type> mutable temp;
  parallel::distributed::Vector<value_type> const *sum_alphai_ui;
  parallel::distributed::BlockVector<value_type> const *vector_linearization;

  std_cxx11::shared_ptr<PreconditionerNavierStokesBase<value_type> > preconditioner;
  std_cxx11::shared_ptr<IterativeSolverBase<parallel::distributed::BlockVector<value_type> > > linear_solver;

  std_cxx11::shared_ptr<NewtonSolver<parallel::distributed::BlockVector<value_type>,
                                     DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>,
                                     IterativeSolverBase<parallel::distributed::BlockVector<value_type> > > >
    newton_solver;
};



template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
setup_solvers ()
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Setup solvers ..." << std::endl;

  // temp has to be initialized whenever an unsteady problem has to be solved
  if(unsteady_problem_has_to_be_solved())
  {
    this->initialize_vector_velocity(temp);
  }

  if(this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockDiagonal ||
     this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockTriangular ||
     this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockTriangularFactorization)
  {
    BlockPreconditionerData preconditioner_data;
    preconditioner_data.preconditioner_type = this->param.preconditioner_linearized_navier_stokes;
    preconditioner_data.momentum_preconditioner = this->param.momentum_preconditioner;
    preconditioner_data.solver_momentum_preconditioner = this->param.solver_momentum_preconditioner;
    preconditioner_data.multigrid_data_momentum_preconditioner = this->param.multigrid_data_momentum_preconditioner;
    preconditioner_data.rel_tol_solver_momentum_preconditioner = this->param.rel_tol_solver_momentum_preconditioner;
    preconditioner_data.schur_complement_preconditioner = this->param.schur_complement_preconditioner;
    preconditioner_data.discretization_of_laplacian = this->param.discretization_of_laplacian;
    preconditioner_data.solver_schur_complement_preconditioner = this->param.solver_schur_complement_preconditioner;
    preconditioner_data.multigrid_data_schur_complement_preconditioner = this->param.multigrid_data_schur_complement_preconditioner;
    preconditioner_data.rel_tol_solver_schur_complement_preconditioner = this->param.rel_tol_solver_schur_complement_preconditioner;

    preconditioner.reset(new BlockPreconditionerNavierStokes<dim, fe_degree, fe_degree_p,
                             fe_degree_xwall, n_q_points_1d_xwall, value_type>(this,preconditioner_data));
  }


  if(this->param.solver_linearized_navier_stokes == SolverLinearizedNavierStokes::GMRES)
  {
    GMRESSolverData solver_data;
    solver_data.max_iter = this->param.max_iter_linear;
    solver_data.solver_tolerance_abs = this->param.abs_tol_linear;
    solver_data.solver_tolerance_rel = this->param.rel_tol_linear;
    solver_data.right_preconditioning = this->param.use_right_preconditioning;
    solver_data.max_n_tmp_vectors = this->param.max_n_tmp_vectors;
    solver_data.compute_eigenvalues = false;

    if(this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockDiagonal ||
       this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockTriangular ||
       this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockTriangularFactorization)
    {
      solver_data.use_preconditioner = true;
    }

    linear_solver.reset(new GMRESSolverNavierStokes<DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>,
                                        PreconditionerNavierStokesBase<value_type>,
                                        parallel::distributed::BlockVector<value_type> >
        (*this,*preconditioner,solver_data));
  }
  else if(this->param.solver_linearized_navier_stokes == SolverLinearizedNavierStokes::FGMRES)
  {
    FGMRESSolverData solver_data;
    solver_data.max_iter = this->param.max_iter_linear;
    solver_data.solver_tolerance_abs = this->param.abs_tol_linear;
    solver_data.solver_tolerance_rel = this->param.rel_tol_linear;
    solver_data.max_n_tmp_vectors = this->param.max_n_tmp_vectors;

    if(this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockDiagonal ||
       this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockTriangular ||
       this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockTriangularFactorization)
    {
      solver_data.use_preconditioner = true;
    }

    linear_solver.reset(new FGMRESSolverNavierStokes<DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>,
                                         PreconditionerNavierStokesBase<value_type>,
                                         parallel::distributed::BlockVector<value_type> >
        (*this,*preconditioner,solver_data));
  }
  else
  {
    AssertThrow(this->param.solver_linearized_navier_stokes == SolverLinearizedNavierStokes::GMRES ||
                this->param.solver_linearized_navier_stokes == SolverLinearizedNavierStokes::FGMRES,
                ExcMessage("Specified solver for linearized Navier-Stokes problem not available."));
  }

  // Newton solver
  if(nonlinear_problem_has_to_be_solved())
  {
    NewtonSolverData newton_solver_data;
    newton_solver_data.abs_tol = this->param.abs_tol_newton;
    newton_solver_data.rel_tol = this->param.rel_tol_newton;
    newton_solver_data.max_iter = this->param.max_iter_newton;

    newton_solver.reset(new NewtonSolver<parallel::distributed::BlockVector<value_type>,
                                         DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>,
                                         IterativeSolverBase<parallel::distributed::BlockVector<value_type> > >
       (newton_solver_data,*this,*linear_solver));
  }

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
vmult (parallel::distributed::BlockVector<value_type>       &dst,
       parallel::distributed::BlockVector<value_type> const &src) const
{
  apply_linearized_problem(dst,src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
apply_linearized_problem (parallel::distributed::BlockVector<value_type>       &dst,
                          parallel::distributed::BlockVector<value_type> const &src) const
{
  // (1,1) block of saddle point matrix
  this->viscous_operator.apply(dst.block(0),src.block(0));

  if(unsteady_problem_has_to_be_solved())
  {
    temp.equ(this->scaling_factor_time_derivative_term,src.block(0));
    this->mass_matrix_operator.apply_add(dst.block(0),temp);
  }

  if(nonlinear_problem_has_to_be_solved())
    this->convective_operator.apply_linearized_add(dst.block(0),
                                                   src.block(0),
                                                   &vector_linearization->block(0),
                                                   this->evaluation_time);

  // (1,2) block of saddle point matrix
  // gradient operator: dst = velocity, src = pressure
  this->gradient_operator.apply_add(dst.block(0),src.block(1));

  // (2,1) block of saddle point matrix
  // divergence operator: dst = pressure, src = velocity
  this->divergence_operator.apply(dst.block(1),src.block(0));
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  dst.block(1) *= -1.0;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
rhs_stokes_problem (parallel::distributed::BlockVector<value_type>  &dst,
                    parallel::distributed::Vector<value_type> const *src) const
{
  // velocity-block
  this->viscous_operator.rhs(dst.block(0),this->evaluation_time);
  this->gradient_operator.rhs_add(dst.block(0),this->evaluation_time);

  if(unsteady_problem_has_to_be_solved())
    this->mass_matrix_operator.apply_add(dst.block(0),*src);

  if(this->param.right_hand_side == true)
    this->body_force_operator.evaluate_add(dst.block(0),this->evaluation_time);

  // pressure-block
  this->divergence_operator.rhs(dst.block(1),this->evaluation_time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  dst.block(1) *= -1.0;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
evaluate_nonlinear_residual (parallel::distributed::BlockVector<value_type>       &dst,
                             parallel::distributed::BlockVector<value_type> const &src)
{
  // velocity-block

  if(this->param.right_hand_side == true)
  {
    this->body_force_operator.evaluate(dst.block(0),this->evaluation_time);
    // shift body force term to the left-hand side of the equation
    dst.block(0) *= -1.0;
  }
  else // right_hand_side == false
  {
    // set dst.block(0) to zero. This is necessary since the subsequent operators
    // call functions of type ..._add
    dst.block(0) = 0.0;
  }

  if(unsteady_problem_has_to_be_solved())
  {
    temp.equ(this->scaling_factor_time_derivative_term,src.block(0));
    temp.add(-1.0,*sum_alphai_ui);
    this->mass_matrix_operator.apply_add(dst.block(0),temp);
  }

  this->convective_operator.evaluate_add(dst.block(0),src.block(0),this->evaluation_time);
  this->viscous_operator.evaluate_add(dst.block(0),src.block(0),this->evaluation_time);
  this->gradient_operator.evaluate_add(dst.block(0),src.block(1),this->evaluation_time);

  // pressure-block

  this->divergence_operator.evaluate(dst.block(1),src.block(0),this->evaluation_time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  dst.block(1) *= -1.0;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
unsigned int DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
solve_linear_stokes_problem (parallel::distributed::BlockVector<value_type>       &dst,
                             parallel::distributed::BlockVector<value_type> const &src)
{
  return linear_solver->solve(dst,src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
solve_nonlinear_steady_problem (parallel::distributed::BlockVector<value_type>  &dst,
                                unsigned int                                    &newton_iterations,
                                double                                          &average_linear_iterations)
{
  newton_solver->solve(dst,newton_iterations,average_linear_iterations);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
solve_nonlinear_problem (parallel::distributed::BlockVector<value_type>  &dst,
                         parallel::distributed::Vector<value_type> const &sum_alphai_ui,
                         unsigned int                                    &newton_iterations,
                         double                                          &average_linear_iterations)
{
  this->sum_alphai_ui = &sum_alphai_ui;
  newton_solver->solve(dst,newton_iterations,average_linear_iterations);
  this->sum_alphai_ui = nullptr;
}

#endif /* INCLUDE_DGNAVIERSTOKESCOUPLED_H_ */
