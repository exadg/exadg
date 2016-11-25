/*
 * DGNavierStokesPressureCorrection.h
 *
 *  Created on: Oct 26, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_DGNAVIERSTOKESPRESSURECORRECTION_H_
#define INCLUDE_DGNAVIERSTOKESPRESSURECORRECTION_H_

#include "DGNavierStokesProjectionMethods.h"

#include "HelmholtzOperator.h"
#include "VelocityConvDiffOperator.h"

#include "NewtonSolver.h"
#include "IterativeSolvers.h"

#include "InverseMassMatrixPreconditioner.h"
#include "../include/JacobiPreconditioner.h"
#include "../include/MultigridPreconditionerNavierStokes.h"

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
class DGNavierStokesPressureCorrection : public DGNavierStokesProjectionMethods<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>
{
public:
  typedef DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule> BaseClass;
  typedef typename BaseClass::value_type value_type;

  typedef FEFaceEvaluationWrapperPressure<dim,fe_degree_p,fe_degree_xwall,BaseClass::n_actual_q_points_vel_linear,1,
                                          value_type,BaseClass::is_xwall> FEFaceEval_Pressure_Velocity_linear;

  DGNavierStokesPressureCorrection(parallel::distributed::Triangulation<dim> const &triangulation,
                                   InputParametersNavierStokes<dim> const          &parameter)
    :
    DGNavierStokesProjectionMethods<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>(triangulation,parameter),
    rhs_vector(nullptr),
    evaluation_time(0.0),
    scaling_factor_time_derivative_term(1.0)
  {}

  virtual ~DGNavierStokesPressureCorrection()
  {}

  void setup_solvers(double const &time_step_size,
                     double const &scaling_factor_time_derivative_term);

  // momentum step: linear system of equations (Stokes or convective term treated explicitly)
  void solve_linear_momentum_equation (parallel::distributed::Vector<value_type>       &solution,
                                       parallel::distributed::Vector<value_type> const &rhs,
                                       double const                                    &scaling_factor_mass_matrix_term,
                                       unsigned int                                    &linear_iterations);

  // momentum step: nonlinear system of equations (convective term treated implicitly)
  void solve_nonlinear_momentum_equation (parallel::distributed::Vector<value_type>       &dst,
                                          parallel::distributed::Vector<value_type> const &rhs_vector,
                                          double const                                    &eval_time,
                                          double const                                    &scaling_factor_mass_matrix_term,
                                          unsigned int                                    &newton_iterations,
                                          double                                          &average_linear_iterations);


  /*
   * The implementation of the Newton solver requires that the underlying operator
   * implements a function called "evaluate_nonlinear_residual"
   */
  void evaluate_nonlinear_residual (parallel::distributed::Vector<value_type>       &dst,
                                    parallel::distributed::Vector<value_type> const &src);

  /*
   * The implementation of the Newton solver requires that the underlying operator
   * implements a function called "initialize_vector_for_newton_solver"
   */
  void initialize_vector_for_newton_solver(parallel::distributed::Vector<value_type> &src) const
  {
    this->initialize_vector_velocity(src);
  }

  // rhs pressure gradient
  void rhs_pressure_gradient_term(parallel::distributed::Vector<value_type> &dst,
                                  double const                              evaluation_time) const;

  // body forces
  void  evaluate_add_body_force_term(parallel::distributed::Vector<value_type> &dst,
                                     double const                              evaluation_time) const;


  // apply inverse pressure mass matrix
  void apply_inverse_pressure_mass_matrix(parallel::distributed::Vector<value_type>        &dst,
                                          const parallel::distributed::Vector<value_type>  &src) const;

private:
  // momentum equation
  VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type> velocity_conv_diff_operator;
  std_cxx11::shared_ptr<PreconditionerBase<value_type> > momentum_preconditioner;
  std_cxx11::shared_ptr<IterativeSolverBase<parallel::distributed::Vector<value_type> > > momentum_linear_solver;

  std_cxx11::shared_ptr<NewtonSolver<parallel::distributed::Vector<value_type>,
                                     DGNavierStokesPressureCorrection<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>,
                                     VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type>,
                                     IterativeSolverBase<parallel::distributed::Vector<value_type> > > >
    momentum_newton_solver;

  InverseMassMatrixOperator<dim,fe_degree_p,value_type> inverse_mass_matrix_operator_pressure;

  parallel::distributed::Vector<value_type> temp_vector;
  parallel::distributed::Vector<value_type> const *rhs_vector;

  double evaluation_time;
  double scaling_factor_time_derivative_term;

  // setup of solvers
  void setup_momentum_solver(double const &scaling_factor_time_derivative_term);

  void setup_inverse_mass_matrix_operator_pressure();
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesPressureCorrection<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
setup_solvers(double const &time_step_size,
              double const &scaling_factor_time_derivative_term)
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Setup solvers ..." << std::endl;

  setup_momentum_solver(scaling_factor_time_derivative_term);

  this->setup_pressure_poisson_solver(time_step_size);

  this->setup_projection_solver();

  setup_inverse_mass_matrix_operator_pressure();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesPressureCorrection<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
setup_momentum_solver(double const &scaling_factor_time_derivative_term)
{
  // setup velocity convection-diffusion operator
  VelocityConvDiffOperatorData<dim> vel_conv_diff_operator_data;

  // unsteady problem
  vel_conv_diff_operator_data.unsteady_problem = true;

  // convective problem
  if(this->param.equation_type == EquationType::NavierStokes &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
  {
    vel_conv_diff_operator_data.convective_problem = true;
  }
  else
  {
    vel_conv_diff_operator_data.convective_problem = false;
  }

  vel_conv_diff_operator_data.dof_index = this->get_dof_index_velocity();

  velocity_conv_diff_operator.initialize(
      this->get_data(),
      vel_conv_diff_operator_data,
      this->mass_matrix_operator,
      this->viscous_operator,
      this->convective_operator);

  velocity_conv_diff_operator.set_scaling_factor_time_derivative_term(scaling_factor_time_derivative_term);


  // setup preconditioner for momentum equation
  if(this->param.preconditioner_momentum == PreconditionerMomentum::InverseMassMatrix)
  {
    typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector DofHandlerSelector;
    typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::QuadratureSelector QuadratureSelector;

    momentum_preconditioner.reset(new InverseMassMatrixPreconditioner<dim,fe_degree,value_type>(
        this->data,
        static_cast<typename std::underlying_type<DofHandlerSelector>::type>(DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type<QuadratureSelector>::type>(QuadratureSelector::velocity)));
  }
  else if(this->param.preconditioner_momentum == PreconditionerMomentum::Jacobi)
  {
    momentum_preconditioner.reset(new JacobiPreconditioner<value_type,
        VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type> >
      (velocity_conv_diff_operator));
  }
  else if(this->param.preconditioner_momentum == PreconditionerMomentum::VelocityDiffusion)
  {
    typedef float Number;

    typedef MyMultigridPreconditionerVelocityDiffusion<dim,value_type,
        HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>,
        VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type> > MULTIGRID;

    momentum_preconditioner.reset(new MULTIGRID());

    std_cxx11::shared_ptr<MULTIGRID> mg_preconditioner = std::dynamic_pointer_cast<MULTIGRID>(momentum_preconditioner);

    mg_preconditioner->initialize(this->param.multigrid_data_momentum,
                                  this->dof_handler_u,
                                  this->mapping,
                                  velocity_conv_diff_operator,
                                  this->periodic_face_pairs);
  }
  else if(this->param.preconditioner_momentum == PreconditionerMomentum::VelocityConvectionDiffusion)
  {
    typedef float Number;

    typedef MyMultigridPreconditionerVelocityConvectionDiffusion<dim,value_type,
        VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>,
        VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type> > MULTIGRID;

    momentum_preconditioner.reset(new MULTIGRID());

    std_cxx11::shared_ptr<MULTIGRID> mg_preconditioner = std::dynamic_pointer_cast<MULTIGRID>(momentum_preconditioner);

    mg_preconditioner->initialize(this->param.multigrid_data_momentum,
                                  this->get_dof_handler_u(),
                                  this->get_mapping(),
                                  velocity_conv_diff_operator,
                                  this->periodic_face_pairs);
  }

  // setup linear solver for momentum equation
  if(this->param.solver_momentum == SolverMomentum::PCG)
  {
    // setup solver data
    CGSolverData solver_data;
    solver_data.solver_tolerance_abs = this->param.abs_tol_momentum_linear;
    solver_data.solver_tolerance_rel = this->param.rel_tol_momentum_linear;
    solver_data.max_iter = this->param.max_iter_momentum_linear;

    if(this->param.preconditioner_momentum == PreconditionerMomentum::Jacobi ||
       this->param.preconditioner_momentum == PreconditionerMomentum::InverseMassMatrix ||
       this->param.preconditioner_momentum == PreconditionerMomentum::VelocityDiffusion ||
       this->param.preconditioner_momentum == PreconditionerMomentum::VelocityConvectionDiffusion)
    {
      solver_data.use_preconditioner = true;
    }
    solver_data.update_preconditioner = this->param.update_preconditioner_momentum;

    // setup solver
    momentum_linear_solver.reset(new CGSolver<VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type>,
                                              PreconditionerBase<value_type>,
                                              parallel::distributed::Vector<value_type> >(
        velocity_conv_diff_operator,
        *momentum_preconditioner,
        solver_data));
  }
  else if(this->param.solver_momentum == SolverMomentum::GMRES)
  {
    // setup solver data
    GMRESSolverData solver_data;
    solver_data.solver_tolerance_abs = this->param.abs_tol_momentum_linear;
    solver_data.solver_tolerance_rel = this->param.rel_tol_momentum_linear;
    solver_data.max_iter = this->param.max_iter_momentum_linear;
    solver_data.right_preconditioning = this->param.use_right_preconditioning_momentum;
    solver_data.max_n_tmp_vectors = this->param.max_n_tmp_vectors_momentum;
    solver_data.compute_eigenvalues = false;

    if(this->param.preconditioner_momentum == PreconditionerMomentum::Jacobi ||
       this->param.preconditioner_momentum == PreconditionerMomentum::InverseMassMatrix ||
       this->param.preconditioner_momentum == PreconditionerMomentum::VelocityDiffusion ||
       this->param.preconditioner_momentum == PreconditionerMomentum::VelocityConvectionDiffusion)
    {
      solver_data.use_preconditioner = true;
    }
    solver_data.update_preconditioner = this->param.update_preconditioner_momentum;

    // setup solver
    momentum_linear_solver.reset(new GMRESSolver<VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type>,
                                                 PreconditionerBase<value_type>,
                                                 parallel::distributed::Vector<value_type> >(
        velocity_conv_diff_operator,
        *momentum_preconditioner,
        solver_data));
  }
  else
  {
    AssertThrow(this->param.solver_momentum == SolverMomentum::PCG ||
                this->param.solver_momentum == SolverMomentum::GMRES,
                ExcMessage("Specified Viscous Solver not implemented - possibilities are PCG and GMRES"));
  }


  // Navier-Stokes equations with an implicit treatment of the convective term
  if(this->param.equation_type == EquationType::NavierStokes &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
  {
    // initialize temp vector
    this->initialize_vector_velocity(temp_vector);

    // setup Newton solver
    momentum_newton_solver.reset(new NewtonSolver<parallel::distributed::Vector<value_type>,
                                                  DGNavierStokesPressureCorrection<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>,
                                                  VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type>,
                                                  IterativeSolverBase<parallel::distributed::Vector<value_type> > >(
        this->param.newton_solver_data_momentum,
        *this,
        velocity_conv_diff_operator,
        *momentum_linear_solver));
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesPressureCorrection<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
setup_inverse_mass_matrix_operator_pressure ()
{
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector DofHandlerSelector;
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::QuadratureSelector QuadratureSelector;

  // inverse mass matrix operator pressure (needed for pressure update in case of rotational formulation)
  inverse_mass_matrix_operator_pressure.initialize(
      this->data,
      static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure),
      static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::pressure));
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesPressureCorrection<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
solve_linear_momentum_equation(parallel::distributed::Vector<value_type>       &solution,
                               parallel::distributed::Vector<value_type> const &rhs,
                               double const                                    &scaling_factor_mass_matrix_term,
                               unsigned int                                    &linear_iterations)
{
  // Set scaling_factor_time_derivative_term for linear operator (=velocity_conv_diff_operator).
  velocity_conv_diff_operator.set_scaling_factor_time_derivative_term(scaling_factor_mass_matrix_term);

  // Note that there is no need to set the evaluation time for the velocity_conv_diff_operator
  // in this because because this function is only called if the convective term is not considered
  // in the velocity_conv_diff_operator (Stokes eq. or explicit treatment of convective term).

  linear_iterations = momentum_linear_solver->solve(solution,rhs);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesPressureCorrection<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
evaluate_add_body_force_term(parallel::distributed::Vector<value_type> &dst,
                             double const                              evaluation_time) const
{
  this->body_force_operator.evaluate_add(dst,evaluation_time);
}


template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesPressureCorrection<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
solve_nonlinear_momentum_equation(parallel::distributed::Vector<value_type>       &dst,
                                  parallel::distributed::Vector<value_type> const &rhs_vector,
                                  double const                                    &eval_time,
                                  double const                                    &scaling_factor_mass_matrix_term,
                                  unsigned int                                    &newton_iterations,
                                  double                                          &average_linear_iterations)
{
  // Set rhs_vector, this variable is used when evaluating the nonlinear residual
  this->rhs_vector = &rhs_vector;

  // Set evaluation_time for nonlinear operator (=DGNavierStokesPressureCorrection)
  evaluation_time = eval_time;
  // Set scaling_time_derivative_term for nonlinear operator (=DGNavierStokesPressureCorrection)
  scaling_factor_time_derivative_term = scaling_factor_mass_matrix_term;

  // Set correct evaluation time for linear operator (=velocity_conv_diff_operator).
  velocity_conv_diff_operator.set_evaluation_time(eval_time);
  // Set scaling_factor_time_derivative_term for linear operator (=velocity_conv_diff_operator).
  velocity_conv_diff_operator.set_scaling_factor_time_derivative_term(scaling_factor_mass_matrix_term);

  // Solve nonlinear problem
  momentum_newton_solver->solve(dst,newton_iterations,average_linear_iterations);

  // Reset rhs_vector
  this->rhs_vector = nullptr;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesPressureCorrection<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
evaluate_nonlinear_residual (parallel::distributed::Vector<value_type>             &dst,
                             const parallel::distributed::Vector<value_type>       &src)
{
  // set dst to zero
  dst = 0.0;

  // mass matrix term
  temp_vector.equ(scaling_factor_time_derivative_term,src);
  this->mass_matrix_operator.apply_add(dst,temp_vector);

  // always evaluate convective term since this function is only called
  // if a nonlinear problem has to be solved, i.e., if the convective operator
  // has to be considered
  this->convective_operator.evaluate_add(dst,src,evaluation_time);

  // viscous term
  this->viscous_operator.evaluate_add(dst,src,evaluation_time);

  // rhs vector
  dst.add(-1.0,*rhs_vector);
}


template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesPressureCorrection<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
rhs_pressure_gradient_term(parallel::distributed::Vector<value_type> &dst,
                           double const                              evaluation_time) const
{
  this->gradient_operator.rhs(dst,evaluation_time);
}



template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesPressureCorrection<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
apply_inverse_pressure_mass_matrix(parallel::distributed::Vector<value_type>        &dst,
                                   const parallel::distributed::Vector<value_type>  &src) const
{
  inverse_mass_matrix_operator_pressure.apply(dst,src);
}


#endif /* INCLUDE_DGNAVIERSTOKESPRESSURECORRECTION_H_ */
