/*
 * DGNavierStokesDualSplitting.h
 *
 *  Created on: Jun 27, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_DGNAVIERSTOKESDUALSPLITTING_H_
#define INCLUDE_DGNAVIERSTOKESDUALSPLITTING_H_

#include "DGNavierStokesProjectionMethods.h"

#include "CurlCompute.h"
#include "HelmholtzOperator.h"

#include "NewtonSolver.h"
#include "IterativeSolvers.h"

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
class DGNavierStokesDualSplitting : public DGNavierStokesProjectionMethods<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>
{
public:
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::value_type value_type;
  static const unsigned int n_actual_q_points_vel_nonlinear = (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::is_xwall) ? xwall_quad_rule : fe_degree+(fe_degree+2)/2;

  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_nonlinear,dim,value_type,DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::is_xwall> FEFaceEval_Velocity_Velocity_nonlinear;
  typedef FEFaceEvaluationWrapperPressure<dim,fe_degree_p,fe_degree_xwall,DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::n_actual_q_points_vel_linear,1,value_type,
      DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::is_xwall> FEFaceEval_Pressure_Velocity_linear;
  typedef FEFaceEvaluationWrapperPressure<dim,fe_degree_p,fe_degree_xwall,n_actual_q_points_vel_nonlinear,1,value_type,
      DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::is_xwall> FEFaceEval_Pressure_Velocity_nonlinear;

  DGNavierStokesDualSplitting(parallel::distributed::Triangulation<dim> const &triangulation,
                              InputParametersNavierStokes<dim> const          &parameter)
    :
    DGNavierStokesProjectionMethods<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>(triangulation,parameter),
    fe_param(parameter),
    velocity_linear(nullptr)
  {}

  virtual ~DGNavierStokesDualSplitting()
  {}

  void setup_solvers ();

  // implicit solution of convective step
  void solve_nonlinear_convective_problem (parallel::distributed::Vector<value_type>       &dst,
                                           parallel::distributed::Vector<value_type> const &sum_alphai_ui,
                                           unsigned int                                    &newton_iterations,
                                           double                                          &average_linear_iterations);

  /*
   *  The implementation of the Newton solver requires that the underlying operator
   *  implements a function called "evaluate_nonlinear_residual"
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

  /*
   * The implementation of the Newton solver requires that the underlying operator
   * implements a function called "set_solution_linearization"
   */
  void set_solution_linearization(parallel::distributed::Vector<value_type> const *solution_linearization)
  {
    velocity_linear = solution_linearization;
  }

  /*
   *  To solve the linearized convective problem, the underlying operator
   *  has to implement a function called "vmult"
   */
  void vmult (parallel::distributed::Vector<value_type>       &dst,
              parallel::distributed::Vector<value_type> const &src) const;

  void apply_linearized_convective_problem (parallel::distributed::Vector<value_type>       &dst,
                                            parallel::distributed::Vector<value_type> const &src) const;

  // convective term
  void evaluate_convective_term_and_apply_inverse_mass_matrix(parallel::distributed::Vector<value_type>       &dst,
                                                              parallel::distributed::Vector<value_type> const &src,
                                                              value_type const                                evaluation_time) const;

  // body forces
  void  evaluate_body_force_and_apply_inverse_mass_matrix (parallel::distributed::Vector<value_type>  &dst,
                                                           const value_type                           evaluation_time) const;

  // rhs pressure
  void rhs_pressure_BC_term (parallel::distributed::Vector<value_type>       &dst,
                             const parallel::distributed::Vector<value_type> &src) const;

  // rhs pressure
  void rhs_ppe_nbc_add (parallel::distributed::Vector<value_type> &dst,
                        double const                              &evaluation_time);

  void rhs_ppe_convective_add (parallel::distributed::Vector<value_type>       &dst,
                               const parallel::distributed::Vector<value_type> &src) const;

  void rhs_ppe_viscous_add(parallel::distributed::Vector<value_type>       &dst,
                           const parallel::distributed::Vector<value_type> &src) const;

  // viscous step
  unsigned int solve_viscous (parallel::distributed::Vector<value_type>       &dst,
                              const parallel::distributed::Vector<value_type> &src);

  FEParameters<dim> const & get_fe_parameters() const
  {
    return this->fe_param;
  }

protected:
  FEParameters<dim> fe_param;
  HelmholtzOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,value_type> helmholtz_operator;
  std_cxx11::shared_ptr<PreconditionerBase<value_type> > helmholtz_preconditioner;

private:
  std_cxx11::shared_ptr<IterativeSolverBase<parallel::distributed::Vector<value_type> > > helmholtz_solver;

  parallel::distributed::Vector<value_type> const *velocity_linear;
  parallel::distributed::Vector<value_type> temp;
  parallel::distributed::Vector<value_type> sum_alphai_ui;

  // implicit solution of convective step
  std_cxx11::shared_ptr<InverseMassMatrixPreconditioner<dim,fe_degree,value_type> >
      preconditioner_convective_problem;
  std_cxx11::shared_ptr<IterativeSolverBase<parallel::distributed::Vector<value_type> > > linear_solver;
  std_cxx11::shared_ptr<NewtonSolver<parallel::distributed::Vector<value_type>,
                                     DGNavierStokesDualSplitting<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>,
                                     IterativeSolverBase<parallel::distributed::Vector<value_type> > > >
    newton_solver;

  // setup of solvers
  void setup_convective_solver();
  void setup_helmholtz_solver();
  virtual void setup_helmholtz_preconditioner(HelmholtzOperatorData<dim> &helmholtz_operator_data);

  // rhs pressure: BC term
  void local_rhs_pressure_BC_term (const MatrixFree<dim,value_type>                &data,
                                   parallel::distributed::Vector<value_type>       &dst,
                                   const parallel::distributed::Vector<value_type> &src,
                                   const std::pair<unsigned int,unsigned int>      &cell_range) const;

  void local_rhs_pressure_BC_term_face (const MatrixFree<dim,value_type>                 &data,
                                        parallel::distributed::Vector<value_type>        &dst,
                                        const parallel::distributed::Vector<value_type>  &src,
                                        const std::pair<unsigned int,unsigned int>       &face_range) const;

  void local_rhs_pressure_BC_term_boundary_face(const MatrixFree<dim,value_type>                 &data,
                                                parallel::distributed::Vector<value_type>        &dst,
                                                const parallel::distributed::Vector<value_type>  &src,
                                                const std::pair<unsigned int,unsigned int>       &face_range) const;

  // rhs pressure: NBC term
  void local_rhs_ppe_nbc_add (const MatrixFree<dim,value_type>                &data,
                              parallel::distributed::Vector<value_type>       &dst,
                              const parallel::distributed::Vector<value_type> &src,
                              const std::pair<unsigned int,unsigned int>      &cell_range) const;

  void local_rhs_ppe_nbc_add_face (const MatrixFree<dim,value_type>                 &data,
                                   parallel::distributed::Vector<value_type>        &dst,
                                   const parallel::distributed::Vector<value_type>  &src,
                                   const std::pair<unsigned int,unsigned int>       &face_range) const;

  void local_rhs_ppe_nbc_add_boundary_face(const MatrixFree<dim,value_type>                 &data,
                                           parallel::distributed::Vector<value_type>        &dst,
                                           const parallel::distributed::Vector<value_type>  &src,
                                           const std::pair<unsigned int,unsigned int>       &face_range) const;

  // rhs pressure: convective term
  void local_rhs_ppe_convective_add (const MatrixFree<dim,value_type>                &data,
                                     parallel::distributed::Vector<value_type>       &dst,
                                     const parallel::distributed::Vector<value_type> &src,
                                     const std::pair<unsigned int,unsigned int>      &cell_range) const;

  void local_rhs_ppe_convective_add_face (const MatrixFree<dim,value_type>                 &data,
                                          parallel::distributed::Vector<value_type>        &dst,
                                          const parallel::distributed::Vector<value_type>  &src,
                                          const std::pair<unsigned int,unsigned int>       &face_range) const;

  void local_rhs_ppe_convective_add_boundary_face(const MatrixFree<dim,value_type>                 &data,
                                                  parallel::distributed::Vector<value_type>        &dst,
                                                  const parallel::distributed::Vector<value_type>  &src,
                                                  const std::pair<unsigned int,unsigned int>       &face_range) const;

  // rhs pressure: viscous term
  void local_rhs_ppe_viscous_add (const MatrixFree<dim,value_type>                &data,
                                  parallel::distributed::Vector<value_type>       &dst,
                                  const parallel::distributed::Vector<value_type> &src,
                                  const std::pair<unsigned int,unsigned int>      &cell_range) const;

  void local_rhs_ppe_viscous_add_face (const MatrixFree<dim,value_type>                 &data,
                                       parallel::distributed::Vector<value_type>        &dst,
                                       const parallel::distributed::Vector<value_type>  &src,
                                       const std::pair<unsigned int,unsigned int>       &face_range) const;

  void local_rhs_ppe_viscous_add_boundary_face(const MatrixFree<dim,value_type>                 &data,
                                               parallel::distributed::Vector<value_type>        &dst,
                                               const parallel::distributed::Vector<value_type>  &src,
                                               const std::pair<unsigned int,unsigned int>       &face_range) const;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
setup_solvers ()
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Setup solvers ..." << std::endl;

  // initialize vectors that are needed by the nonlinear solver
  if(this->param.equation_type == EquationType::NavierStokes &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
  {
    setup_convective_solver();
  }

  this->setup_pressure_poisson_solver();

  this->setup_projection_solver();

  setup_helmholtz_solver();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
setup_convective_solver ()
{
  this->initialize_vector_velocity(temp);
  this->initialize_vector_velocity(sum_alphai_ui);

  // preconditioner implicit convective step
  preconditioner_convective_problem.reset(new InverseMassMatrixPreconditioner<dim,fe_degree,value_type>
     (this->data,
      static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector>::type >
          (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector::velocity),
      static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::QuadratureSelector>::type >
          (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::QuadratureSelector::velocity)));

  // linear solver (GMRES)
  GMRESSolverData solver_data;
  solver_data.max_iter = this->param.max_iter_linear_convective;
  solver_data.solver_tolerance_abs = this->param.abs_tol_linear_convective;
  solver_data.solver_tolerance_rel = this->param.rel_tol_linear_convective;
  solver_data.right_preconditioning = this->param.use_right_preconditioning_convective;
  solver_data.max_n_tmp_vectors = this->param.max_n_tmp_vectors_convective;

  // always use inverse mass matrix preconditioner
  solver_data.use_preconditioner = true;

  linear_solver.reset(new GMRESSolver<DGNavierStokesDualSplitting<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>,
                                      InverseMassMatrixPreconditioner<dim,fe_degree,value_type>,
                                      parallel::distributed::Vector<value_type> >
      (*this,*preconditioner_convective_problem,solver_data));

  newton_solver.reset(new NewtonSolver<parallel::distributed::Vector<value_type>,
                                       DGNavierStokesDualSplitting<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>,
                                       IterativeSolverBase<parallel::distributed::Vector<value_type> > >
     (this->param.newton_solver_data_convective,*this,*linear_solver));
}


template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
setup_helmholtz_solver ()
{
  // setup helmholtz operator
  HelmholtzOperatorData<dim> helmholtz_operator_data;

  helmholtz_operator_data.mass_matrix_operator_data = this->mass_matrix_operator_data;
  helmholtz_operator_data.viscous_operator_data = this->viscous_operator_data;

  helmholtz_operator_data.dof_index = static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector>::type >
                                        (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector::velocity);
  helmholtz_operator_data.scaling_factor_time_derivative_term = this->scaling_factor_time_derivative_term;
  helmholtz_operator_data.periodic_face_pairs_level0 = this->periodic_face_pairs;

  helmholtz_operator.initialize(this->data,helmholtz_operator_data,this->mass_matrix_operator,this->viscous_operator);

  // setup helmholtz preconditioner
  setup_helmholtz_preconditioner(helmholtz_operator_data);

  if(this->param.solver_viscous == SolverViscous::PCG)
  {
    // setup solver data
    CGSolverData solver_data;
    // use default value of max_iter
    solver_data.solver_tolerance_abs = this->param.abs_tol_viscous;
    solver_data.solver_tolerance_rel = this->param.rel_tol_viscous;
    // default value of use_preconditioner = false
    if(this->param.preconditioner_viscous == PreconditionerViscous::Jacobi ||
       this->param.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix ||
       this->param.preconditioner_viscous == PreconditionerViscous::GeometricMultigrid)
    {
      solver_data.use_preconditioner = true;
    }

    // setup helmholtz solver
    helmholtz_solver.reset(new CGSolver<HelmholtzOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,value_type>,
                                        PreconditionerBase<value_type>,
                                        parallel::distributed::Vector<value_type> >
       (helmholtz_operator,
        *helmholtz_preconditioner,
        solver_data));
  }
  else if(this->param.solver_viscous == SolverViscous::GMRES)
  {
    // setup solver data
    GMRESSolverData solver_data;
    // use default value of max_iter
    solver_data.solver_tolerance_abs = this->param.abs_tol_viscous;
    solver_data.solver_tolerance_rel = this->param.rel_tol_viscous;
    // use default value of right_preconditioning
    // use default value of max_n_tmp_vectors
    // use default value of compute_eigenvalues

    // default value of use_preconditioner = false
    if(this->param.preconditioner_viscous == PreconditionerViscous::Jacobi ||
       this->param.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix ||
       this->param.preconditioner_viscous == PreconditionerViscous::GeometricMultigrid)
    {
      solver_data.use_preconditioner = true;
    }

    // setup helmholtz solver
    helmholtz_solver.reset(new GMRESSolver<HelmholtzOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,value_type>,
                                           PreconditionerBase<value_type>,
                                           parallel::distributed::Vector<value_type> >
       (helmholtz_operator,
        *helmholtz_preconditioner,
        solver_data));
  }
  else
  {
    AssertThrow(this->param.solver_viscous == SolverViscous::PCG ||
                this->param.solver_viscous == SolverViscous::GMRES,
                ExcMessage("Specified Viscous Solver not implemented - possibilities are PCG and GMRES"));
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
setup_helmholtz_preconditioner (HelmholtzOperatorData<dim> &helmholtz_operator_data)
{
  if(this->param.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix)
  {
    helmholtz_preconditioner.reset(new InverseMassMatrixPreconditioner<dim,fe_degree,value_type>
       (this->data,
        static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector>::type >
          (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::QuadratureSelector>::type >
          (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::QuadratureSelector::velocity)));
  }
  else if(this->param.preconditioner_viscous == PreconditionerViscous::Jacobi)
  {
    helmholtz_preconditioner.reset(new JacobiPreconditioner<value_type,
        HelmholtzOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule, value_type> >(helmholtz_operator));
  }
  else if(this->param.preconditioner_viscous == PreconditionerViscous::GeometricMultigrid)
  {
    MultigridData mg_data;
    mg_data = this->param.multigrid_data_viscous;

    // use single precision for multigrid
    typedef float Number;

    helmholtz_preconditioner.reset(new MyMultigridPreconditioner<dim,value_type,
                                         HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>,
                                         HelmholtzOperatorData<dim> > ());

    std_cxx11::shared_ptr<MyMultigridPreconditioner<dim,value_type,
                            HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>,
                            HelmholtzOperatorData<dim> > >
      mg_preconditioner = std::dynamic_pointer_cast<MyMultigridPreconditioner<dim,value_type,
                                                      HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>,
                                                      HelmholtzOperatorData<dim> > >(helmholtz_preconditioner);

    mg_preconditioner->initialize(mg_data,
                                  this->dof_handler_u,
                                  this->mapping,
                                  helmholtz_operator_data,
                                  this->boundary_descriptor_velocity->dirichlet_bc);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
vmult (parallel::distributed::Vector<value_type>       &dst,
       parallel::distributed::Vector<value_type> const &src) const
{
  apply_linearized_convective_problem(dst,src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
apply_linearized_convective_problem (parallel::distributed::Vector<value_type>       &dst,
                                     parallel::distributed::Vector<value_type> const &src) const
{
  this->mass_matrix_operator.apply(dst,src);
  // dst-vector only contains velocity (and not the pressure)
  dst *= this->scaling_factor_time_derivative_term;

  this->convective_operator.apply_linearized_add(dst,
                                                 src,
                                                 velocity_linear,
                                                 this->evaluation_time);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
solve_nonlinear_convective_problem (parallel::distributed::Vector<value_type>       &dst,
                                    parallel::distributed::Vector<value_type> const &sum_alphai_ui,
                                    unsigned int                                    &newton_iterations,
                                    double                                          &average_linear_iterations)
{
  this->sum_alphai_ui = sum_alphai_ui;

  // update the linear operator so that it uses the correct
  // linearized velocity field
  set_solution_linearization(&dst);

  // solve nonlinear problem
  newton_solver->solve(dst,newton_iterations,average_linear_iterations);

  // reset solution linearization
  set_solution_linearization(nullptr);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
evaluate_nonlinear_residual(parallel::distributed::Vector<value_type>       &dst,
                            const parallel::distributed::Vector<value_type> &src)
{
  if(this->param.right_hand_side == true)
  {
    this->body_force_operator.evaluate(dst,this->evaluation_time);
    // shift body force term to the left-hand side of the equation
    dst *= -1.0;
  }
  else // right_hand_side == false
  {
    // set dst to zero. This is necessary since the subsequent operators
    // call functions of type ..._add
    dst = 0.0;
  }

  // temp, src, sum_alphai_ui have the same number of blocks
  temp.equ(this->scaling_factor_time_derivative_term,src);
  temp.add(-1.0,sum_alphai_ui);

  this->mass_matrix_operator.apply_add(dst,temp);

  this->convective_operator.evaluate_add(dst,src,this->evaluation_time);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
evaluate_convective_term_and_apply_inverse_mass_matrix (parallel::distributed::Vector<value_type>       &dst,
                                                        parallel::distributed::Vector<value_type> const &src,
                                                        value_type const                                evaluation_time) const
{
  this->convective_operator.evaluate(dst,src,evaluation_time);
  this->inverse_mass_matrix_operator->apply(dst,dst);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
evaluate_body_force_and_apply_inverse_mass_matrix(parallel::distributed::Vector<value_type>  &dst,
                                                  const value_type                           evaluation_time) const
{
  this->body_force_operator.evaluate(dst,evaluation_time);

  this->inverse_mass_matrix_operator->apply(dst,dst);
}

// will be removed
template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
rhs_pressure_BC_term (parallel::distributed::Vector<value_type>       &dst,
                      const parallel::distributed::Vector<value_type> &src) const
{
  this->data.loop (&DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::local_rhs_pressure_BC_term,
                   &DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::local_rhs_pressure_BC_term_face,
                   &DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::local_rhs_pressure_BC_term_boundary_face,
                   this, dst, src);
}

// will be removed
template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
local_rhs_pressure_BC_term (const MatrixFree<dim,value_type>                 &,
                            parallel::distributed::Vector<value_type>        &,
                            const parallel::distributed::Vector<value_type>  &,
                            const std::pair<unsigned int,unsigned int>       &) const
{

}

// will be removed
template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
local_rhs_pressure_BC_term_face (const MatrixFree<dim,value_type>                &,
                                 parallel::distributed::Vector<value_type>       &,
                                 const parallel::distributed::Vector<value_type> &,
                                 const std::pair<unsigned int,unsigned int>      &) const
{

}

// will be removed
template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
local_rhs_pressure_BC_term_boundary_face (const MatrixFree<dim,value_type>                 &data,
                                          parallel::distributed::Vector<value_type>        &dst,
                                          const parallel::distributed::Vector<value_type>  &,
                                          const std::pair<unsigned int,unsigned int>       &face_range) const
{
  FEFaceEval_Pressure_Velocity_nonlinear fe_eval_pressure(data,&this->fe_param,true,
      static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector>::type >
        (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector::pressure));

  //TODO: quadrature formula
//    FEFaceEval_Pressure_Velocity_linear fe_eval_pressure(data,this->fe_param,true,
//        static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure));

  // set the correct time for the evaluation of the right_hand_side - function
  if(this->param.right_hand_side == true)
    this->field_functions->right_hand_side->set_time(this->evaluation_time);

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval_pressure.reinit (face);

    double factor = this->laplace_operator.get_penalty_factor();
    VectorizedArray<value_type> tau_IP = fe_eval_pressure.read_cell_data(this->laplace_operator.get_array_penalty_parameter()) * (value_type)factor;

    typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
    types::boundary_id boundary_id = data.get_boundary_indicator(face);

    for(unsigned int q=0;q<fe_eval_pressure.n_q_points;++q)
    {
      it = this->boundary_descriptor_pressure->dirichlet_bc.find(boundary_id);
      if(it != this->boundary_descriptor_pressure->dirichlet_bc.end())
      {
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval_pressure.quadrature_point(q);

        // evaluate right-hand side
        Tensor<1,dim,VectorizedArray<value_type> > rhs_np;

        if(this->param.right_hand_side == true)
        {
          for(unsigned int d=0;d<dim;++d)
          {
            value_type array_rhs [VectorizedArray<value_type>::n_array_elements];
            for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
            {
              Point<dim> q_point;
              for (unsigned int d=0; d<dim; ++d)
              q_point[d] = q_points[d][n];
              array_rhs[n] = this->field_functions->right_hand_side->value(q_point,d);
            }
            rhs_np[d].load(&array_rhs[0]);
          }
        }

        // evaluate boundary condition
        Tensor<1,dim,VectorizedArray<value_type> > dudt_np;
        // set time for the correct evaluation of boundary conditions
        it->second->set_time(this->evaluation_time);

        for(unsigned int d=0;d<dim;++d)
        {
          value_type array_dudt [VectorizedArray<value_type>::n_array_elements];
          for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for (unsigned int d=0; d<dim; ++d)
              q_point[d] = q_points[d][n];
            array_dudt[n] = it->second->value(q_point,d);
          }
          dudt_np[d].load(&array_dudt[0]);
        }

        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);
        VectorizedArray<value_type> h;

        h = - normal * (dudt_np - rhs_np);

        fe_eval_pressure.submit_normal_gradient(make_vectorized_array<value_type>(0.0),q);
        fe_eval_pressure.submit_value(h,q);
      }

      it = this->boundary_descriptor_pressure->neumann_bc.find(boundary_id);
      if (it != this->boundary_descriptor_pressure->neumann_bc.end())
      {
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval_pressure.quadrature_point(q);
        VectorizedArray<value_type> g;

        // set time for the correct evaluation of boundary conditions
        it->second->set_time(this->evaluation_time);

        value_type array [VectorizedArray<value_type>::n_array_elements];
        for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
        {
          Point<dim> q_point;
          for (unsigned int d=0; d<dim; ++d)
            q_point[d] = q_points[d][n];
          array[n] = it->second->value(q_point);
        }
        g.load(&array[0]);

        fe_eval_pressure.submit_normal_gradient(-g,q);
        fe_eval_pressure.submit_value(2.0 * tau_IP * g,q);
      }
    }
    fe_eval_pressure.integrate(true,true);
    fe_eval_pressure.distribute_local_to_global(dst);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
rhs_ppe_nbc_add (parallel::distributed::Vector<value_type> &dst,
                 double const                              &eval_time)
{
  this->evaluation_time = eval_time;

  parallel::distributed::Vector<value_type> src_dummy;
  this->data.loop (&DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::local_rhs_ppe_nbc_add,
                   &DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::local_rhs_ppe_nbc_add_face,
                   &DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::local_rhs_ppe_nbc_add_boundary_face,
                   this, dst, src_dummy);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
local_rhs_ppe_nbc_add (const MatrixFree<dim,value_type>                 &,
                       parallel::distributed::Vector<value_type>        &,
                       const parallel::distributed::Vector<value_type>  &,
                       const std::pair<unsigned int,unsigned int>       &) const
{

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
local_rhs_ppe_nbc_add_face (const MatrixFree<dim,value_type>                &,
                            parallel::distributed::Vector<value_type>       &,
                            const parallel::distributed::Vector<value_type> &,
                            const std::pair<unsigned int,unsigned int>      &) const
{

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
local_rhs_ppe_nbc_add_boundary_face (const MatrixFree<dim,value_type>                 &data,
                                     parallel::distributed::Vector<value_type>        &dst,
                                     const parallel::distributed::Vector<value_type>  &,
                                     const std::pair<unsigned int,unsigned int>       &face_range) const
{
  unsigned int dof_index_pressure = static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector>::type >
                                      (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector::pressure);
  unsigned int quad_index_pressure = static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::QuadratureSelector>::type >
                                      (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::QuadratureSelector::pressure);

  FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> fe_eval(data, true,dof_index_pressure,quad_index_pressure);

  // set the correct time for the evaluation of the right_hand_side - function
  if(this->param.right_hand_side == true)
    this->field_functions->right_hand_side->set_time(this->evaluation_time);

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval.reinit (face);

    typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
    types::boundary_id boundary_id = data.get_boundary_indicator(face);

    for(unsigned int q=0;q<fe_eval.n_q_points;++q)
    {
      it = this->boundary_descriptor_pressure->dirichlet_bc.find(boundary_id);
      if(it != this->boundary_descriptor_pressure->dirichlet_bc.end())
      {
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);

        // evaluate right-hand side
        Tensor<1,dim,VectorizedArray<value_type> > rhs;

        if(this->param.right_hand_side == true)
        {
          for(unsigned int d=0;d<dim;++d)
          {
            value_type array_rhs [VectorizedArray<value_type>::n_array_elements];
            for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
            {
              Point<dim> q_point;
              for (unsigned int d=0; d<dim; ++d)
              q_point[d] = q_points[d][n];
              array_rhs[n] = this->field_functions->right_hand_side->value(q_point,d);
            }
            rhs[d].load(&array_rhs[0]);
          }
        }

        // evaluate boundary condition
        Tensor<1,dim,VectorizedArray<value_type> > dudt;
        // set time for the correct evaluation of boundary conditions
        it->second->set_time(this->evaluation_time);

        for(unsigned int d=0;d<dim;++d)
        {
          value_type array_dudt [VectorizedArray<value_type>::n_array_elements];
          for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for (unsigned int d=0; d<dim; ++d)
              q_point[d] = q_points[d][n];
            array_dudt[n] = it->second->value(q_point,d);
          }
          dudt[d].load(&array_dudt[0]);
        }

        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
        VectorizedArray<value_type> h;

        h = - normal * (dudt - rhs);

        fe_eval.submit_normal_gradient(make_vectorized_array<value_type>(0.0),q);
        fe_eval.submit_value(h,q);
      }

      it = this->boundary_descriptor_pressure->neumann_bc.find(boundary_id);
      if (it != this->boundary_descriptor_pressure->neumann_bc.end())
      {
        VectorizedArray<value_type> zero = make_vectorized_array<value_type>(0.0);
        fe_eval.submit_normal_gradient(zero,q);
        fe_eval.submit_value(zero,q);
      }
    }
    fe_eval.integrate(true,true);
    fe_eval.distribute_local_to_global(dst);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
rhs_ppe_convective_add (parallel::distributed::Vector<value_type>       &dst,
                        const parallel::distributed::Vector<value_type> &src) const
{
  this->data.loop (&DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::local_rhs_ppe_convective_add,
                   &DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::local_rhs_ppe_convective_add_face,
                   &DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::local_rhs_ppe_convective_add_boundary_face,
                   this, dst, src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
local_rhs_ppe_convective_add (const MatrixFree<dim,value_type>                &,
                              parallel::distributed::Vector<value_type>       &,
                              const parallel::distributed::Vector<value_type> &,
                              const std::pair<unsigned int,unsigned int>      &) const
{

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
local_rhs_ppe_convective_add_face (const MatrixFree<dim,value_type>                &,
                                   parallel::distributed::Vector<value_type>       &,
                                   const parallel::distributed::Vector<value_type> &,
                                   const std::pair<unsigned int,unsigned int>      &) const
{

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
local_rhs_ppe_convective_add_boundary_face (const MatrixFree<dim,value_type>                 &data,
                                            parallel::distributed::Vector<value_type>        &dst,
                                            const parallel::distributed::Vector<value_type>  &src,
                                            const std::pair<unsigned int,unsigned int>       &face_range) const
{

  FEFaceEval_Velocity_Velocity_nonlinear fe_eval_velocity(data,&this->fe_param,true,
      static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector>::type >
        (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector::velocity));

  FEFaceEval_Pressure_Velocity_nonlinear fe_eval_pressure(data,&this->fe_param,true,
      static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector>::type >
        (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector::pressure));

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval_velocity.reinit(face);
    fe_eval_velocity.read_dof_values(src);
    fe_eval_velocity.evaluate (true,true);

    fe_eval_pressure.reinit (face);

    typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
    types::boundary_id boundary_id = data.get_boundary_indicator(face);

    for(unsigned int q=0;q<fe_eval_pressure.n_q_points;++q)
    {
      it = this->boundary_descriptor_pressure->dirichlet_bc.find(boundary_id);
      if(it != this->boundary_descriptor_pressure->dirichlet_bc.end())
      {
        VectorizedArray<value_type> h;
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);

        Tensor<1,dim,VectorizedArray<value_type> > u = fe_eval_velocity.get_value(q);
        Tensor<2,dim,VectorizedArray<value_type> > grad_u = fe_eval_velocity.get_gradient(q);
        Tensor<1,dim,VectorizedArray<value_type> > convective_term = grad_u * u + fe_eval_velocity.get_divergence(q) * u;

        h = - normal * convective_term;

        fe_eval_pressure.submit_value(h,q);
      }

      it = this->boundary_descriptor_pressure->neumann_bc.find(boundary_id);
      if (it != this->boundary_descriptor_pressure->neumann_bc.end())
      {
        fe_eval_pressure.submit_value(make_vectorized_array<value_type>(0.0),q);
      }
    }
    fe_eval_pressure.integrate(true,false);
    fe_eval_pressure.distribute_local_to_global(dst);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
rhs_ppe_viscous_add(parallel::distributed::Vector<value_type>       &dst,
                    const parallel::distributed::Vector<value_type> &src) const
{
  this->data.loop (&DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::local_rhs_ppe_viscous_add,
                   &DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::local_rhs_ppe_viscous_add_face,
                   &DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::local_rhs_ppe_viscous_add_boundary_face,
                   this, dst, src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
local_rhs_ppe_viscous_add(const MatrixFree<dim,value_type>                &,
                          parallel::distributed::Vector<value_type>       &,
                          const parallel::distributed::Vector<value_type> &,
                          const std::pair<unsigned int,unsigned int>      &) const
{

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
local_rhs_ppe_viscous_add_face (const MatrixFree<dim,value_type>                 &,
                                parallel::distributed::Vector<value_type>        &,
                                const parallel::distributed::Vector<value_type>  &,
                                const std::pair<unsigned int,unsigned int>       &) const
{

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
local_rhs_ppe_viscous_add_boundary_face(const MatrixFree<dim,value_type>                &data,
                                        parallel::distributed::Vector<value_type>       &dst,
                                        const parallel::distributed::Vector<value_type> &src,
                                        const std::pair<unsigned int,unsigned int>      &face_range) const
{
  FEFaceEval_Velocity_Velocity_nonlinear fe_eval_omega(data,&this->fe_param,true,
      static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector>::type >
        (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector::velocity));

  FEFaceEval_Pressure_Velocity_nonlinear fe_eval_pressure(data,&this->fe_param,true,
      static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector>::type >
        (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector::pressure));

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval_pressure.reinit (face);

    fe_eval_omega.reinit (face);
    fe_eval_omega.read_dof_values(src);
    fe_eval_omega.evaluate (false,true);

    typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
    types::boundary_id boundary_id = data.get_boundary_indicator(face);

    for(unsigned int q=0;q<fe_eval_pressure.n_q_points;++q)
    {
      VectorizedArray<value_type> viscosity;
      if(this->viscous_operator.viscosity_is_variable())
        viscosity = this->viscous_operator.get_viscous_coefficient_face()[face][q];
      else
        viscosity = make_vectorized_array<value_type>(this->viscous_operator.get_const_viscosity());

      it = this->boundary_descriptor_pressure->dirichlet_bc.find(boundary_id);
      if(it != this->boundary_descriptor_pressure->dirichlet_bc.end())
      {
        VectorizedArray<value_type> h;
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);

        Tensor<1,dim,VectorizedArray<value_type> > curl_omega = CurlCompute<dim,FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_nonlinear,dim,value_type,DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::is_xwall> >::compute(fe_eval_omega,q);
        h = - normal * (viscosity*curl_omega);

        fe_eval_pressure.submit_value(h,q);
      }

      it = this->boundary_descriptor_pressure->neumann_bc.find(boundary_id);
      if (it != this->boundary_descriptor_pressure->neumann_bc.end())
      {
        fe_eval_pressure.submit_value(make_vectorized_array<value_type>(0.0),q);
      }
    }
    fe_eval_pressure.integrate(true,false);
    fe_eval_pressure.distribute_local_to_global(dst);
  }
}


template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
unsigned int DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
solve_viscous (parallel::distributed::Vector<value_type>       &dst,
               const parallel::distributed::Vector<value_type> &src)
{
  // update helmholtz_operator
  helmholtz_operator.set_scaling_factor_time_derivative_term(this->scaling_factor_time_derivative_term);
  // viscous_operator.set_constant_viscosity(viscosity);
  // viscous_operator.set_variable_viscosity(viscosity);

  // update preconditioner
  if(this->param.preconditioner_viscous == PreconditionerViscous::Jacobi)
  {
    // TODO: recalculate diagonal (say every 10, 100 time steps) in case of varying parameters
    // of mass matrix term or viscous term, e.g. strongly varying time step sizes (adaptive time step control)
    // or strongly varying viscosity (turbulence)
    /*
    std_cxx11::shared_ptr<JacobiPreconditioner<value_type,HelmholtzOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule> > >
      jacobi_preconditioner = std::dynamic_pointer_cast<JacobiPreconditioner<value_type,HelmholtzOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule> > >(helmholtz_preconditioner);
    jacobi_preconditioner->recalculate_diagonal(helmholtz_operator);
    */
  }
  else if(this->param.preconditioner_viscous == PreconditionerViscous::GeometricMultigrid)
  {
    // TODO: update multigrid preconditioner (diagonals) in case of varying parameters
  }

  unsigned int n_iter = helmholtz_solver->solve(dst,src);

  return n_iter;
}

#endif /* INCLUDE_DGNAVIERSTOKESDUALSPLITTING_H_ */
