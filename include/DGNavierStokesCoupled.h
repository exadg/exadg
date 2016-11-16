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

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
class DGNavierStokesCoupled;

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
class DGNavierStokesCoupled : public DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>
{
public:
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::value_type value_type;

  DGNavierStokesCoupled(parallel::distributed::Triangulation<dim> const &triangulation,
                        InputParametersNavierStokes<dim> const          &parameter)
    :
    DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>(triangulation,parameter),
    sum_alphai_ui(nullptr),
    vector_linearization(nullptr)
  {}

  virtual ~DGNavierStokesCoupled(){};

  void setup_solvers ();

  // initialization of vectors
  void initialize_block_vector_velocity_pressure(parallel::distributed::BlockVector<value_type> &src) const
  {
    // velocity(1 block) + pressure(1 block)
    src.reinit(2);

    this->data.initialize_dof_vector(src.block(0),
        static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector>::type >
    (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector::velocity));

    this->data.initialize_dof_vector(src.block(1),
        static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector>::type >
    (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector::pressure));

    src.collect_sizes();
  }

  // TODO: remove this function from DGNavierStokesCoupled
  virtual void set_evaluation_time(double const eval_time)
  {
    this->evaluation_time = eval_time;

    velocity_conv_diff_operator.set_evaluation_time(eval_time);
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

  void set_solution_linearization(parallel::distributed::BlockVector<value_type> const &solution_linearization)
  {
    velocity_conv_diff_operator.set_solution_linearization(solution_linearization.block(0));
  }

  parallel::distributed::Vector<value_type> const &get_velocity_linearization() const
  {
    return velocity_conv_diff_operator.get_solution_linearization();
  }

private:
  friend class BlockPreconditionerNavierStokes<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type,
    DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule> >;

  VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type> velocity_conv_diff_operator;

  parallel::distributed::Vector<value_type> mutable temp;
  parallel::distributed::Vector<value_type> const *sum_alphai_ui;
  parallel::distributed::BlockVector<value_type> const *vector_linearization;

  std_cxx11::shared_ptr<PreconditionerNavierStokesBase<value_type,
    DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule> > > preconditioner;

  std_cxx11::shared_ptr<IterativeSolverBase<parallel::distributed::BlockVector<value_type> > > linear_solver;

  std_cxx11::shared_ptr<NewtonSolver<parallel::distributed::BlockVector<value_type>,
                                     DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>,
                                     DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>,
                                     IterativeSolverBase<parallel::distributed::BlockVector<value_type> > > >
    newton_solver;
};



template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
setup_solvers ()
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Setup solvers ..." << std::endl;

  // setup velocity convection-diffusion operator.
  // This is done in function setup_solvers() since velocity convection-diffusion
  // operator data needs scaling_factor_time_derivative_term as input parameter.
  VelocityConvDiffOperatorData<dim> vel_conv_diff_operator_data;

  // unsteady problem
  if(unsteady_problem_has_to_be_solved())
  {
    vel_conv_diff_operator_data.unsteady_problem = true;
    vel_conv_diff_operator_data.scaling_factor_time_derivative_term = this->scaling_factor_time_derivative_term;
  }
  else
  {
    vel_conv_diff_operator_data.unsteady_problem = false;
  }

  // convective problem
  if(nonlinear_problem_has_to_be_solved())
  {
    vel_conv_diff_operator_data.convective_problem = true;
  }
  else
  {
    vel_conv_diff_operator_data.convective_problem = false;
  }

  vel_conv_diff_operator_data.mass_matrix_operator_data = this->get_mass_matrix_operator_data();
  vel_conv_diff_operator_data.viscous_operator_data = this->get_viscous_operator_data();
  vel_conv_diff_operator_data.convective_operator_data = this->get_convective_operator_data();

  vel_conv_diff_operator_data.dof_index = this->get_dof_index_velocity();
  vel_conv_diff_operator_data.periodic_face_pairs_level0 = this->periodic_face_pairs;

  velocity_conv_diff_operator.initialize(
      this->get_data(),
      vel_conv_diff_operator_data,
      this->mass_matrix_operator,
      this->viscous_operator,
      this->convective_operator);



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
    preconditioner_data.max_n_tmp_vectors_solver_momentum_preconditioner = this->param.max_n_tmp_vectors_solver_momentum_preconditioner;
    preconditioner_data.schur_complement_preconditioner = this->param.schur_complement_preconditioner;
    preconditioner_data.discretization_of_laplacian = this->param.discretization_of_laplacian;
    preconditioner_data.solver_schur_complement_preconditioner = this->param.solver_schur_complement_preconditioner;
    preconditioner_data.multigrid_data_schur_complement_preconditioner = this->param.multigrid_data_schur_complement_preconditioner;
    preconditioner_data.rel_tol_solver_schur_complement_preconditioner = this->param.rel_tol_solver_schur_complement_preconditioner;

    preconditioner.reset(new BlockPreconditionerNavierStokes<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type,
                               DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule> >
        (this,preconditioner_data));
  }


  if(this->param.solver_linearized_navier_stokes == SolverLinearizedNavierStokes::GMRES)
  {
    GMRESSolverData solver_data;
    solver_data.max_iter = this->param.max_iter_linear;
    solver_data.solver_tolerance_abs = this->param.abs_tol_linear;
    solver_data.solver_tolerance_rel = this->param.rel_tol_linear;
    solver_data.right_preconditioning = this->param.use_right_preconditioning;
    solver_data.update_preconditioner = this->param.update_preconditioner;
    solver_data.max_n_tmp_vectors = this->param.max_n_tmp_vectors;
    solver_data.compute_eigenvalues = false;

    if(this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockDiagonal ||
       this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockTriangular ||
       this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockTriangularFactorization)
    {
      solver_data.use_preconditioner = true;
    }

    linear_solver.reset(new GMRESSolver<DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>,
                                        PreconditionerNavierStokesBase<value_type, DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule> >,
                                        parallel::distributed::BlockVector<value_type> >
        (*this,*preconditioner,solver_data));
  }
  else if(this->param.solver_linearized_navier_stokes == SolverLinearizedNavierStokes::FGMRES)
  {
    FGMRESSolverData solver_data;
    solver_data.max_iter = this->param.max_iter_linear;
    solver_data.solver_tolerance_abs = this->param.abs_tol_linear;
    solver_data.solver_tolerance_rel = this->param.rel_tol_linear;
    solver_data.update_preconditioner = this->param.update_preconditioner;
    solver_data.max_n_tmp_vectors = this->param.max_n_tmp_vectors;

    if(this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockDiagonal ||
       this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockTriangular ||
       this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockTriangularFactorization)
    {
      solver_data.use_preconditioner = true;
    }

    linear_solver.reset(new FGMRESSolver<DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>,
                                         PreconditionerNavierStokesBase<value_type, DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>>,
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
    newton_solver.reset(new NewtonSolver<parallel::distributed::BlockVector<value_type>,
                                         DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>,
                                         DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>,
                                         IterativeSolverBase<parallel::distributed::BlockVector<value_type> > >
       (this->param.newton_solver_data_coupled,*this,*this,*linear_solver));
  }

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
vmult (parallel::distributed::BlockVector<value_type>       &dst,
       parallel::distributed::BlockVector<value_type> const &src) const
{
  apply_linearized_problem(dst,src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
apply_linearized_problem (parallel::distributed::BlockVector<value_type>       &dst,
                          parallel::distributed::BlockVector<value_type> const &src) const
{
  // (1,1) block of saddle point matrix

//  this->viscous_operator.apply(dst.block(0),src.block(0));
//
//  if(unsteady_problem_has_to_be_solved())
//  {
//    temp.equ(this->scaling_factor_time_derivative_term,src.block(0));
//    this->mass_matrix_operator.apply_add(dst.block(0),temp);
//  }
//
//  if(nonlinear_problem_has_to_be_solved())
//    this->convective_operator.apply_linearized_add(dst.block(0),
//                                                   src.block(0),
//                                                   &vector_linearization->block(0),
//                                                   this->evaluation_time);

  velocity_conv_diff_operator.vmult(dst.block(0),src.block(0));

  // (1,2) block of saddle point matrix
  // gradient operator: dst = velocity, src = pressure
  this->gradient_operator.apply_add(dst.block(0),src.block(1));

  // (2,1) block of saddle point matrix
  // divergence operator: dst = pressure, src = velocity
  this->divergence_operator.apply(dst.block(1),src.block(0));
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  dst.block(1) *= -1.0;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
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
  // with respect to pressure gradient term and velocity divergence term
  dst.block(1) *= -1.0;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
evaluate_nonlinear_residual (parallel::distributed::BlockVector<value_type>       &dst,
                             parallel::distributed::BlockVector<value_type> const &src)
{
  // velocity-block

  // set dst.block(0) to zero. This is necessary since subsequent operators
  // call functions of type ..._add
  dst.block(0) = 0.0;

  if(this->param.right_hand_side == true)
  {
    this->body_force_operator.evaluate(dst.block(0),this->evaluation_time);
    // Shift body force term to the left-hand side of the equation.
    // This works since body_force_operator is the first operator
    // that is evaluated.
    dst.block(0) *= -1.0;
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
  // with respect to pressure gradient term and velocity divergence term
  dst.block(1) *= -1.0;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
unsigned int DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
solve_linear_stokes_problem (parallel::distributed::BlockVector<value_type>       &dst,
                             parallel::distributed::BlockVector<value_type> const &src)
{
  return linear_solver->solve(dst,src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
solve_nonlinear_steady_problem (parallel::distributed::BlockVector<value_type>  &dst,
                                unsigned int                                    &newton_iterations,
                                double                                          &average_linear_iterations)
{
  // solve nonlinear problem
  newton_solver->solve(dst,newton_iterations,average_linear_iterations);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
solve_nonlinear_problem (parallel::distributed::BlockVector<value_type>  &dst,
                         parallel::distributed::Vector<value_type> const &sum_alphai_ui,
                         unsigned int                                    &newton_iterations,
                         double                                          &average_linear_iterations)
{
  // Set sum_alphai_ui, this variable is used when evaluating the nonlinear residual
  this->sum_alphai_ui = &sum_alphai_ui;

  // Solve nonlinear problem
  newton_solver->solve(dst,newton_iterations,average_linear_iterations);

  // Reset sum_alphai_ui
  this->sum_alphai_ui = nullptr;
}

#endif /* INCLUDE_DGNAVIERSTOKESCOUPLED_H_ */
