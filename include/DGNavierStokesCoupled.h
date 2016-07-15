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

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
class DGNavierStokesCoupled;

template<class NUMBER>
void output_eigenvalues(const std::vector<NUMBER> &eigenvalues,const std::string &text)
{
//    deallog << text << std::endl;
//    for (unsigned int j = 0; j < eigenvalues.size(); ++j)
//      {
//        deallog << ' ' << eigenvalues.at(j) << std::endl;
//      }
//    deallog << std::endl;

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << text << std::endl;
    for (unsigned int j = 0; j < eigenvalues.size(); ++j)
    {
      std::cout << ' ' << eigenvalues.at(j) << std::endl;
    }
    std::cout << std::endl;
  }
}

struct LinearSolverData
{
  double abs_tol;
  double rel_tol;
  unsigned int max_iter;
  PreconditionerDataLinearSolver preconditioner_data;
};

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall,typename value_type>
class SolverLinearizedProblem
{
public:
  void initialize(LinearSolverData                                     solver_data_in,
                  DGNavierStokesCoupled<dim, fe_degree,
                    fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> *underlying_operator_in)
  {
    solver_data = solver_data_in;
    underlying_operator = underlying_operator_in;

    if(solver_data.preconditioner_data.preconditioner_type == PreconditionerLinearizedNavierStokes::BlockDiagonal ||
       solver_data.preconditioner_data.preconditioner_type == PreconditionerLinearizedNavierStokes::BlockTriangular ||
       solver_data.preconditioner_data.preconditioner_type == PreconditionerLinearizedNavierStokes::BlockTriangularFactorization)
      preconditioner.reset(new BlockPreconditionerNavierStokes<dim, fe_degree,
          fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(underlying_operator,solver_data.preconditioner_data));

  }

  unsigned int solve(parallel::distributed::BlockVector<value_type>       &dst,
                     parallel::distributed::BlockVector<value_type> const &src,
                     parallel::distributed::BlockVector<value_type> const *solution_linearization = nullptr)
  {
    if(solution_linearization != nullptr)
      underlying_operator->set_solution_linearization(solution_linearization);

    ReductionControl solver_control (solver_data.max_iter, solver_data.abs_tol, solver_data.rel_tol);
    typename SolverGMRES<parallel::distributed::BlockVector<value_type> >::AdditionalData additional_data;
    additional_data.max_n_tmp_vectors = 60;
    // use right preconditioning A*P^{-1}
    additional_data.right_preconditioning = true;

    SolverGMRES<parallel::distributed::BlockVector<value_type> > solver (solver_control, additional_data);

    if(false)
    {
      solver.connect_eigenvalues_slot(std_cxx11::bind(output_eigenvalues<std::complex<double> >,std_cxx11::_1,"Eigenvalues: "),true);
    }

//    underlying_operator->vmult(dst,src);
//    double l2_norm = dst.l2_norm();
//    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
//      std::cout<<"L2 norm of Matrix Vector Product = "<<std::setprecision(14)<<l2_norm<<std::endl;

    if(solver_data.preconditioner_data.preconditioner_type == PreconditionerLinearizedNavierStokes::None)
    {
      solver.solve (*underlying_operator, dst, src, PreconditionIdentity());
    }
    else
    {
      solver.solve (*underlying_operator, dst, src, *preconditioner);
    }

    if(solution_linearization != nullptr)
      underlying_operator->set_solution_linearization(nullptr);

    return solver_control.last_step();
  }

private:
  LinearSolverData solver_data;
  DGNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> *underlying_operator;
  std_cxx11::shared_ptr<PreconditionerNavierStokesBase<value_type> > preconditioner;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
class DGNavierStokesCoupled : public DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>
{
public:
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::value_type value_type;

  DGNavierStokesCoupled(parallel::distributed::Triangulation<dim> const &triangulation,
                        InputParameters const                           &parameter)
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
        static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity));

    this->data.initialize_dof_vector(src.block(1),
        static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure));

    src.collect_sizes();
  }

  void initialize_vector_for_newton_solver(parallel::distributed::BlockVector<value_type> &src) const
  {
    initialize_block_vector_velocity_pressure(src);
  }

  bool nonlinear_problem_has_to_be_solved() const
  {
      return (this->param.equation_type == EquationType::NavierStokes &&
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
                                unsigned int                                    &newton_iterations,
                                double                                          &average_linear_iterations,
                                parallel::distributed::Vector<value_type> const *sum_alphai_ui = nullptr);

  void apply_linearized_problem (parallel::distributed::BlockVector<value_type> &dst,
                                 parallel::distributed::BlockVector<value_type> const &src) const;

  void vmult (parallel::distributed::BlockVector<value_type> &dst,
              parallel::distributed::BlockVector<value_type> const &src) const;

  void rhs_stokes_problem (parallel::distributed::BlockVector<value_type>  &dst,
                           parallel::distributed::Vector<value_type> const *src = nullptr) const;

  void evaluate_convective_term (parallel::distributed::Vector<value_type>       &dst,
                                 parallel::distributed::Vector<value_type> const &src,
                                 value_type const                                evaluation_time) const;

  void evaluate_nonlinear_residual (parallel::distributed::BlockVector<value_type>       &dst,
                                    parallel::distributed::BlockVector<value_type> const &src);

  void set_solution_linearization(parallel::distributed::BlockVector<value_type> const *solution_linearization)
  {
    vector_linearization = solution_linearization;
  }

private:
  friend class BlockPreconditionerNavierStokes<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>;

  parallel::distributed::Vector<value_type> mutable temp;
  parallel::distributed::Vector<value_type> const *sum_alphai_ui;
  parallel::distributed::BlockVector<value_type> const *vector_linearization;

  SolverLinearizedProblem<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type> linear_solver;
  NewtonSolver<parallel::distributed::BlockVector<value_type>,
               DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>,
               SolverLinearizedProblem<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type> >
    newton_solver;
};



template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
setup_solvers ()
{
  // temp has to be initialized whenever an unsteady problem has to be solved
  if(unsteady_problem_has_to_be_solved())
  {
    this->initialize_vector_velocity(temp);
  }

  // linear solver that is used to solve the linear Stokes problem and the linearized Navier-Stokes problem
  LinearSolverData linear_solver_data;
  linear_solver_data.abs_tol = this->param.abs_tol_linear;
  linear_solver_data.rel_tol = this->param.rel_tol_linear;
  linear_solver_data.max_iter = this->param.max_iter_linear;

  PreconditionerDataLinearSolver preconditioner_data;
  preconditioner_data.preconditioner_type = this->param.preconditioner_linearized_navier_stokes;
  preconditioner_data.preconditioner_momentum = this->param.preconditioner_momentum;
  preconditioner_data.preconditioner_schur_complement = this->param.preconditioner_schur_complement;

  linear_solver_data.preconditioner_data = preconditioner_data;

  linear_solver.initialize(linear_solver_data,this);

  // Newton solver
  if(nonlinear_problem_has_to_be_solved())
  {
    NewtonSolverData newton_solver_data;
    newton_solver_data.abs_tol = this->param.abs_tol_newton;
    newton_solver_data.rel_tol = this->param.rel_tol_newton;
    newton_solver_data.max_iter = this->param.max_iter_newton;

    newton_solver.initialize(newton_solver_data,this,&linear_solver);
  }
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
    this->convective_operator.apply_linearized_add(dst.block(0),src.block(0),&vector_linearization->block(0),this->time+this->time_step);

  // (1,2) block of saddle point matrix
  // gradient operator: dst = velocity, src = pressure
  this->gradient_operator.apply_add(dst.block(0),src.block(1));

  // (2,1) block of saddle point matrix
  // divergence operator: dst = pressure, src = velocity
  this->divergence_operator.apply(dst.block(1),src.block(0));
  if(this->param.use_symmetric_saddle_point_matrix == true)
    dst.block(1) *= -1.0;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
rhs_stokes_problem (parallel::distributed::BlockVector<value_type>  &dst,
                    parallel::distributed::Vector<value_type> const *src) const
{
  // velocity-block
  this->body_force_operator.evaluate(dst.block(0),this->time+this->time_step);
  this->viscous_operator.rhs_add(dst.block(0),this->time+this->time_step);
  this->gradient_operator.rhs_add(dst.block(0),this->time+this->time_step);

  if(unsteady_problem_has_to_be_solved())
    this->mass_matrix_operator.apply_add(dst.block(0),*src);

  // pressure-block
  this->divergence_operator.rhs(dst.block(1),this->time+this->time_step);
  if(this->param.use_symmetric_saddle_point_matrix == true)
    dst.block(1) *= -1.0;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
evaluate_convective_term (parallel::distributed::Vector<value_type>       &dst,
                          parallel::distributed::Vector<value_type> const &src,
                          value_type const                                evaluation_time) const
{
  this->convective_operator.evaluate(dst,src,evaluation_time);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
evaluate_nonlinear_residual (parallel::distributed::BlockVector<value_type>       &dst,
                             parallel::distributed::BlockVector<value_type> const &src)
{
  // velocity-block
  this->body_force_operator.evaluate(dst.block(0),this->time+this->time_step);
  // shift body force term to the left-hand side of the equation
  dst.block(0) *= -1.0;

  if(unsteady_problem_has_to_be_solved())
  {
    temp.equ(this->scaling_factor_time_derivative_term,src.block(0));
    temp.add(-1.0,*sum_alphai_ui);
    this->mass_matrix_operator.apply_add(dst.block(0),temp);
  }

  this->convective_operator.evaluate_add(dst.block(0),src.block(0),this->time+this->time_step);
  this->viscous_operator.evaluate_add(dst.block(0),src.block(0),this->time+this->time_step);
  this->gradient_operator.evaluate_add(dst.block(0),src.block(1),this->time+this->time_step);

  // pressure-block
  this->divergence_operator.evaluate(dst.block(1),src.block(0),this->time+this->time_step);
  if(this->param.use_symmetric_saddle_point_matrix == true)
    dst.block(1) *= -1.0;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
unsigned int DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
solve_linear_stokes_problem (parallel::distributed::BlockVector<value_type>       &dst,
                             parallel::distributed::BlockVector<value_type> const &src)
{
  return linear_solver.solve(dst,src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
solve_nonlinear_problem (parallel::distributed::BlockVector<value_type>  &dst,
                         unsigned int                                    &newton_iterations,
                         double                                          &average_linear_iterations,
                         parallel::distributed::Vector<value_type> const *sum_alphai_ui)
{
  this->sum_alphai_ui = sum_alphai_ui;
  newton_solver.solve(dst,newton_iterations,average_linear_iterations);
}

#endif /* INCLUDE_DGNAVIERSTOKESCOUPLED_H_ */
