/*
 * DGNavierStokesCoupled.h
 *
 *  Created on: Jun 27, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_DGNAVIERSTOKESCOUPLED_H_
#define INCLUDE_DGNAVIERSTOKESCOUPLED_H_

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
    vector_linearization(nullptr),
    evaluation_time(0.0),
    scaling_factor_time_derivative_term(1.0)
  {}

  virtual ~DGNavierStokesCoupled(){};

  void setup_solvers(double const &scaling_factor_time_derivative_term = 1.0);

  // initialization of vectors
  void initialize_block_vector_velocity_pressure(parallel::distributed::BlockVector<value_type> &src) const
  {
    // velocity(1 block) + pressure(1 block)
    src.reinit(2);

    typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector DofHandlerSelector;

    this->data.initialize_dof_vector(src.block(0), static_cast<typename std::underlying_type<DofHandlerSelector>::type>(DofHandlerSelector::velocity));
    this->data.initialize_dof_vector(src.block(1),static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure));

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

  /*
   *  This function solves the linear Stokes problem (steady/unsteady Stokes or unsteady
   *  Navier-Stokes with explicit treatment of convective term).
   *  The parameter scaling_factor_mass_matrix_term has to be specified for unsteady problem.
   *  For steady problems this parameter is omitted.
   */
  unsigned int solve_linear_stokes_problem (parallel::distributed::BlockVector<value_type>       &dst,
                                            parallel::distributed::BlockVector<value_type> const &src,
                                            double const                                         &scaling_factor_mass_matrix_term = 1.0);


  /*
   *  For the linear solver, the operator of the linear(ized) problem has to
   *  implement a function called vmult().
   */
  void vmult (parallel::distributed::BlockVector<value_type> &dst,
              parallel::distributed::BlockVector<value_type> const &src) const;

  /*
   *  This function calculates the matrix vector product for the linear(ized) problem.
   */
  void apply_linearized_problem (parallel::distributed::BlockVector<value_type> &dst,
                                 parallel::distributed::BlockVector<value_type> const &src) const;

  /*
   *  This function calculates the rhs of the steady Stokes problem, or unsteady Stokes problem,
   *  or unsteady Navier-Stokes problem with explicit treatment of the convective term.
   *  The parameters 'src' and 'eval_time' have to be specified for unsteady problems.
   *  For steady problems these parameters are omitted.
   */
  void rhs_stokes_problem (parallel::distributed::BlockVector<value_type>  &dst,
                           parallel::distributed::Vector<value_type> const *src = nullptr,
                           double const                                    &eval_time = 0.0) const;


  /*
   *  This function solves the nonlinear problem for steady problems.
   */
  void solve_nonlinear_steady_problem (parallel::distributed::BlockVector<value_type>  &dst,
                                       unsigned int                                    &newton_iterations,
                                       double                                          &average_linear_iterations);

  /*
   *  This function solves the nonlinear problem for unsteady problems.
   */
  void solve_nonlinear_problem (parallel::distributed::BlockVector<value_type>  &dst,
                                parallel::distributed::Vector<value_type> const &sum_alphai_ui,
                                double const                                    &eval_time,
                                double const                                    &scaling_factor_mass_matrix_term,
                                unsigned int                                    &newton_iterations,
                                double                                          &average_linear_iterations);

  /*
   *  This function evaluates the nonlinear residual.
   */
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

  CompatibleLaplaceOperatorData<dim> const get_compatible_laplace_operator_data() const
  {
    CompatibleLaplaceOperatorData<dim> comp_laplace_operator_data;
    comp_laplace_operator_data.dof_index_velocity = this->get_dof_index_velocity();
    comp_laplace_operator_data.dof_index_pressure = this->get_dof_index_pressure();
    return comp_laplace_operator_data;
  }

private:
  friend class BlockPreconditionerNavierStokes<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type,
    DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule> >;

  VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type> velocity_conv_diff_operator;

  parallel::distributed::Vector<value_type> mutable temp_vector;
  parallel::distributed::Vector<value_type> const *sum_alphai_ui;
  parallel::distributed::BlockVector<value_type> const *vector_linearization;

  std_cxx11::shared_ptr<PreconditionerNavierStokesBase<value_type,
    DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule> > > preconditioner;

  std_cxx11::shared_ptr<IterativeSolverBase<parallel::distributed::BlockVector<value_type> > > linear_solver;

  std_cxx11::shared_ptr<NewtonSolver<parallel::distributed::BlockVector<value_type>,
                                     DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>,
                                     DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>,
                                     IterativeSolverBase<parallel::distributed::BlockVector<value_type> > > > newton_solver;

  double evaluation_time;
  double scaling_factor_time_derivative_term;
};



template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
setup_solvers(double const &scaling_factor_time_derivative_term)
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Setup solvers ..." << std::endl;

  // Setup velocity convection-diffusion operator.
  // This is done in function setup_solvers() since velocity convection-diffusion
  // operator data needs scaling_factor_time_derivative_term as input parameter.

  // Note that the velocity_conv_diff_operator has to be initialized
  // before calling the setup of the BlockPreconditioner!
  VelocityConvDiffOperatorData<dim> vel_conv_diff_operator_data;

  // unsteady problem
  if(unsteady_problem_has_to_be_solved())
    vel_conv_diff_operator_data.unsteady_problem = true;
  else
    vel_conv_diff_operator_data.unsteady_problem = false;

  // convective problem
  if(nonlinear_problem_has_to_be_solved())
    vel_conv_diff_operator_data.convective_problem = true;
  else
    vel_conv_diff_operator_data.convective_problem = false;

  vel_conv_diff_operator_data.dof_index = this->get_dof_index_velocity();

  velocity_conv_diff_operator.initialize(
      this->get_data(),
      vel_conv_diff_operator_data,
      this->mass_matrix_operator,
      this->viscous_operator,
      this->convective_operator);

  velocity_conv_diff_operator.set_scaling_factor_time_derivative_term(scaling_factor_time_derivative_term);


  // temp has to be initialized whenever an unsteady problem has to be solved
  if(unsteady_problem_has_to_be_solved())
  {
    this->initialize_vector_velocity(temp_vector);
  }

  // setup preconditioner
  if(this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockDiagonal ||
     this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockTriangular ||
     this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockTriangularFactorization)
  {
    BlockPreconditionerData preconditioner_data;
    preconditioner_data.preconditioner_type = this->param.preconditioner_linearized_navier_stokes;
    preconditioner_data.momentum_preconditioner = this->param.momentum_preconditioner;
    preconditioner_data.exact_inversion_of_momentum_block = this->param.exact_inversion_of_momentum_block;
    preconditioner_data.multigrid_data_momentum_preconditioner = this->param.multigrid_data_momentum_preconditioner;
    preconditioner_data.rel_tol_solver_momentum_preconditioner = this->param.rel_tol_solver_momentum_preconditioner;
    preconditioner_data.max_n_tmp_vectors_solver_momentum_preconditioner = this->param.max_n_tmp_vectors_solver_momentum_preconditioner;
    preconditioner_data.schur_complement_preconditioner = this->param.schur_complement_preconditioner;
    preconditioner_data.discretization_of_laplacian = this->param.discretization_of_laplacian;
    preconditioner_data.exact_inversion_of_laplace_operator = this->param.exact_inversion_of_laplace_operator;
    preconditioner_data.multigrid_data_schur_complement_preconditioner = this->param.multigrid_data_schur_complement_preconditioner;
    preconditioner_data.rel_tol_solver_schur_complement_preconditioner = this->param.rel_tol_solver_schur_complement_preconditioner;

    preconditioner.reset(new BlockPreconditionerNavierStokes<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type,
                               DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule> >
        (this,preconditioner_data));
  }

  // setup linear solver
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

  // setup Newton solver
  if(nonlinear_problem_has_to_be_solved())
  {
    newton_solver.reset(new NewtonSolver<parallel::distributed::BlockVector<value_type>,
                                         DGNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>,
                                         DGNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>,
                                         IterativeSolverBase<parallel::distributed::BlockVector<value_type> > >
       (this->param.newton_solver_data_coupled,*this,*this,*linear_solver));
  }

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
unsigned int DGNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
solve_linear_stokes_problem (parallel::distributed::BlockVector<value_type>       &dst,
                             parallel::distributed::BlockVector<value_type> const &src,
                             double const                                         &scaling_factor_mass_matrix_term)
{
  // Set scaling_factor_time_derivative_term for linear operator (velocity_conv_diff_operator).
  velocity_conv_diff_operator.set_scaling_factor_time_derivative_term(scaling_factor_mass_matrix_term);

  // Note that there is no need to set the evaluation time for the velocity_conv_diff_operator
  // in this because because this function is only called if the convective term is not considered
  // in the velocity_conv_diff_operator (Stokes eq. or explicit treatment of convective term).

  return linear_solver->solve(dst,src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
rhs_stokes_problem (parallel::distributed::BlockVector<value_type>  &dst,
                    parallel::distributed::Vector<value_type> const *src,
                    double const                                    &eval_time) const
{
  // velocity-block
  this->viscous_operator.rhs(dst.block(0),eval_time);
  this->gradient_operator.rhs_add(dst.block(0),eval_time);

  if(unsteady_problem_has_to_be_solved())
    this->mass_matrix_operator.apply_add(dst.block(0),*src);

  if(this->param.right_hand_side == true)
    this->body_force_operator.evaluate_add(dst.block(0),eval_time);

  // pressure-block
  this->divergence_operator.rhs(dst.block(1),eval_time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  dst.block(1) *= -1.0;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
vmult (parallel::distributed::BlockVector<value_type>       &dst,
       parallel::distributed::BlockVector<value_type> const &src) const
{
  apply_linearized_problem(dst,src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
apply_linearized_problem (parallel::distributed::BlockVector<value_type>       &dst,
                          parallel::distributed::BlockVector<value_type> const &src) const
{
  // (1,1) block of saddle point matrix
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
                         double const                                    &eval_time,
                         double const                                    &scaling_factor_mass_matrix_term,
                         unsigned int                                    &newton_iterations,
                         double                                          &average_linear_iterations)
{
  // Set sum_alphai_ui (this variable is used when evaluating the nonlinear residual).
  this->sum_alphai_ui = &sum_alphai_ui;

  // Set evaluation_time for nonlinear operator (=DGNavierStokesCoupled)
  evaluation_time = eval_time;
  // Set scaling_factor_time_derivative_term for nonlinear operator (=DGNavierStokesCoupled)
  scaling_factor_time_derivative_term = scaling_factor_mass_matrix_term;

  // Set correct evaluation time for linear operator (velocity_conv_diff_operator).
  velocity_conv_diff_operator.set_evaluation_time(eval_time);
  // Set scaling_factor_time_derivative_term for linear operator (velocity_conv_diff_operator).
  velocity_conv_diff_operator.set_scaling_factor_time_derivative_term(scaling_factor_mass_matrix_term);

  // Solve nonlinear problem
  newton_solver->solve(dst,newton_iterations,average_linear_iterations);

  // Reset sum_alphai_ui
  this->sum_alphai_ui = nullptr;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
evaluate_nonlinear_residual (parallel::distributed::BlockVector<value_type>       &dst,
                             parallel::distributed::BlockVector<value_type> const &src)
{
  // velocity-block

  // set dst.block(0) to zero. This is necessary since subsequent operators
  // call functions of type ..._add
  dst.block(0) = 0.0;

  if(this->param.right_hand_side == true)
  {
    this->body_force_operator.evaluate(dst.block(0),evaluation_time);
    // Shift body force term to the left-hand side of the equation.
    // This works since body_force_operator is the first operator
    // that is evaluated.
    dst.block(0) *= -1.0;
  }

  if(unsteady_problem_has_to_be_solved())
  {
    temp_vector.equ(scaling_factor_time_derivative_term,src.block(0));
    temp_vector.add(-1.0,*sum_alphai_ui);
    this->mass_matrix_operator.apply_add(dst.block(0),temp_vector);
  }

  this->convective_operator.evaluate_add(dst.block(0),src.block(0),evaluation_time);
  this->viscous_operator.evaluate_add(dst.block(0),src.block(0),evaluation_time);
  this->gradient_operator.evaluate_add(dst.block(0),src.block(1),evaluation_time);

  // pressure-block

  this->divergence_operator.evaluate(dst.block(1),src.block(0),evaluation_time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  dst.block(1) *= -1.0;
}

#endif /* INCLUDE_DGNAVIERSTOKESCOUPLED_H_ */
