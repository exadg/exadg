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

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
class DGNavierStokesPressureCorrection : public DGNavierStokesProjectionMethods<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>
{
public:
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::value_type value_type;

  typedef FEFaceEvaluationWrapperPressure<dim,fe_degree_p,fe_degree_xwall,DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::n_actual_q_points_vel_linear,1,value_type,
      DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::is_xwall> FEFaceEval_Pressure_Velocity_linear;

  DGNavierStokesPressureCorrection(parallel::distributed::Triangulation<dim> const &triangulation,
                                   InputParametersNavierStokes<dim> const          &parameter)
    :
    DGNavierStokesProjectionMethods<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>(triangulation,parameter),
    rhs_vector(nullptr)
  {}

  virtual ~DGNavierStokesPressureCorrection()
  {}

  void setup_solvers(double const time_step_size);

  // momentum step: linear system of equations (Stokes or convective term treated explicitly)
  void solve_linear_momentum_equation (parallel::distributed::Vector<value_type>       &solution,
                                       parallel::distributed::Vector<value_type> const &rhs,
                                       unsigned int                                    &linear_iterations);

  // momentum step: nonlinear system of equations (convective term treated implicitly)
  void solve_nonlinear_momentum_equation (parallel::distributed::Vector<value_type>       &dst,
                                          parallel::distributed::Vector<value_type> const &rhs_vector,
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
  void rhs_pressure_gradient_term(parallel::distributed::Vector<value_type>       &dst,
                                  value_type const                                evaluation_time) const;

  // body forces
  void  evaluate_add_body_force_term(parallel::distributed::Vector<value_type>  &dst,
                                     const value_type                           evaluation_time) const;


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

  // setup of solvers
  void setup_momentum_solver();
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesPressureCorrection<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
setup_solvers(double const time_step_size)
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Setup solvers ..." << std::endl;

  setup_momentum_solver();

  this->setup_pressure_poisson_solver(time_step_size);

  this->setup_projection_solver();

  // inverse mass matrix operator pressure (needed for pressure update in case of rotational formulation)
  inverse_mass_matrix_operator_pressure.initialize(this->data,
      static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector>::type >
        (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector::pressure),
      static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::QuadratureSelector>::type >
        (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::QuadratureSelector::pressure));

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesPressureCorrection<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
setup_momentum_solver ()
{
  // setup velocity convection-diffusion operator
  VelocityConvDiffOperatorData<dim> vel_conv_diff_operator_data;

  // unsteady problem
  vel_conv_diff_operator_data.unsteady_problem = true;
  vel_conv_diff_operator_data.scaling_factor_time_derivative_term = this->scaling_factor_time_derivative_term;

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

  vel_conv_diff_operator_data.mass_matrix_operator_data = this->get_mass_matrix_operator_data();
  // TODO: Velocity conv diff operator is initialized with constant viscosity, in case of varying viscosities
  // the vel conv diff operator (the viscous operator of the conv diff operator) has to be updated before applying this
  // preconditioner
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


  // setup preconditioner for momentum equation
  if(this->param.preconditioner_momentum == PreconditionerMomentum::InverseMassMatrix)
  {
    momentum_preconditioner.reset(new InverseMassMatrixPreconditioner<dim,fe_degree,value_type>(
        this->data,
        static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector>::type >
            (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::QuadratureSelector>::type >
            (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::QuadratureSelector::velocity)));
  }
  else if(this->param.preconditioner_momentum == PreconditionerMomentum::Jacobi)
  {
    momentum_preconditioner.reset(new JacobiPreconditioner<value_type,
        VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type> >
      (velocity_conv_diff_operator));
  }
  else if(this->param.preconditioner_momentum == PreconditionerMomentum::VelocityDiffusion)
  {
    // Geometric multigrid V-cycle performed on Helmholtz operator
    HelmholtzOperatorData<dim> helmholtz_operator_data;

    helmholtz_operator_data.mass_matrix_operator_data = this->mass_matrix_operator_data;
    // TODO: this Helmholtz operator is initialized with constant viscosity, in case of varying viscosities
    // the helmholtz operator (the viscous operator of the helmholtz operator) has to be updated before applying this
    // preconditioner
    helmholtz_operator_data.viscous_operator_data = this->viscous_operator_data;

    helmholtz_operator_data.dof_index = this->get_dof_index_velocity();

    // always unsteady problem
    helmholtz_operator_data.unsteady_problem = true;
    // TODO: this Helmholtz operator is initialized with constant scaling_factor_time_derivative term,
    // in case of a varying scaling_factor_time_derivate_term (adaptive time stepping)
    // the helmholtz operator has to be updated before applying this preconditioner
    helmholtz_operator_data.scaling_factor_time_derivative_term = this->scaling_factor_time_derivative_term;

    helmholtz_operator_data.periodic_face_pairs_level0 = this->periodic_face_pairs;

    typedef float Number;
    momentum_preconditioner.reset(new MyMultigridPreconditioner<dim,value_type,
                                        HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>,
                                        HelmholtzOperatorData<dim> >());

    std_cxx11::shared_ptr<MyMultigridPreconditioner<dim,value_type,
                            HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>,
                            HelmholtzOperatorData<dim> > >
      mg_preconditioner = std::dynamic_pointer_cast<MyMultigridPreconditioner<dim,value_type,
                                                      HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>,
                                                      HelmholtzOperatorData<dim> > >(momentum_preconditioner);

    mg_preconditioner->initialize(this->param.multigrid_data_momentum,
                                  this->get_dof_handler_u(),
                                  this->get_mapping(),
                                  helmholtz_operator_data,
                                  this->boundary_descriptor_velocity->dirichlet_bc);
  }
  else if(this->param.preconditioner_momentum == PreconditionerMomentum::VelocityConvectionDiffusion)
  {
    typedef float Number;

    bool use_chebyshev_smoother = true;
    if(use_chebyshev_smoother) // TODO
    {
      momentum_preconditioner.reset(new MyMultigridPreconditionerVelocityConvectionDiffusion<dim,value_type,
                                          VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>,
                                          VelocityConvDiffOperatorData<dim>,
                                          VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type> >());

      std_cxx11::shared_ptr<MyMultigridPreconditionerVelocityConvectionDiffusion<dim,value_type,
                              VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>,
                              VelocityConvDiffOperatorData<dim>,
                              VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type> > >
        mg_preconditioner = std::dynamic_pointer_cast<MyMultigridPreconditionerVelocityConvectionDiffusion<dim,value_type,
                                                        VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>,
                                                        VelocityConvDiffOperatorData<dim>,
                                                        VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type> > >
            (momentum_preconditioner);

      mg_preconditioner->initialize(this->param.multigrid_data_momentum,
                                    this->get_dof_handler_u(),
                                    this->get_mapping(),
                                    vel_conv_diff_operator_data,
                                    this->boundary_descriptor_velocity->dirichlet_bc);
    }
    else
    {
      momentum_preconditioner.reset(new MyMultigridPreconditionerGMRESSmoother<dim,value_type,
                                          VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>,
                                          VelocityConvDiffOperatorData<dim>,
                                          VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type> >());

      std_cxx11::shared_ptr<MyMultigridPreconditionerGMRESSmoother<dim,value_type,
                              VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>,
                              VelocityConvDiffOperatorData<dim>,
                              VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type> > >
        mg_preconditioner = std::dynamic_pointer_cast<MyMultigridPreconditionerGMRESSmoother<dim,value_type,
                                                        VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>,
                                                        VelocityConvDiffOperatorData<dim>,
                                                        VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type> > >
          (momentum_preconditioner);

      mg_preconditioner->initialize(this->param.multigrid_data_momentum,
                                    this->get_dof_handler_u(),
                                    this->get_mapping(),
                                    vel_conv_diff_operator_data,
                                    this->boundary_descriptor_velocity->dirichlet_bc);
    }
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
solve_linear_momentum_equation(parallel::distributed::Vector<value_type>       &solution,
                               parallel::distributed::Vector<value_type> const &rhs,
                               unsigned int                                    &linear_iterations)
{
  linear_iterations = momentum_linear_solver->solve(solution,rhs);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesPressureCorrection<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
evaluate_add_body_force_term(parallel::distributed::Vector<value_type>  &dst,
                             const value_type                           evaluation_time) const
{
  this->body_force_operator.evaluate_add(dst,evaluation_time);
}


template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesPressureCorrection<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
solve_nonlinear_momentum_equation(parallel::distributed::Vector<value_type>       &dst,
                                  parallel::distributed::Vector<value_type> const &rhs_vector,
                                  unsigned int                                    &newton_iterations,
                                  double                                          &average_linear_iterations)
{
  // Set rhs_vector, this variable is used when evaluating the nonlinear residual
  this->rhs_vector = &rhs_vector;

  // Set correct evaluation time for linear operator (=velocity_conv_diff_operator).
  velocity_conv_diff_operator.set_evaluation_time(this->evaluation_time);

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
  temp_vector.equ(this->scaling_factor_time_derivative_term,src);
  this->mass_matrix_operator.apply_add(dst,temp_vector);

  // always evaluate convective term since this function is only called
  // if a nonlinear problem has to be solved, i.e., if the convective operator
  // has to be considered
  this->convective_operator.evaluate_add(dst,src,this->evaluation_time);

  // viscous term
  this->viscous_operator.evaluate_add(dst,src,this->evaluation_time);

  // rhs vector
  dst.add(-1.0,*rhs_vector);
}


template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesPressureCorrection<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
rhs_pressure_gradient_term(parallel::distributed::Vector<value_type>       &dst,
                           value_type const                                evaluation_time) const
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
