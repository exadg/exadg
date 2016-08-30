/*
 * DGNavierStokesDualSplitting.h
 *
 *  Created on: Jun 27, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_DGNAVIERSTOKESDUALSPLITTING_H_
#define INCLUDE_DGNAVIERSTOKESDUALSPLITTING_H_

#include "DGNavierStokesBase.h"

#include "ProjectionSolver.h"
#include "poisson_solver.h"
#include "CurlCompute.h"
#include "HelmholtzOperator.h"

#include "NewtonSolver.h"
#include "IterativeSolvers.h"

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
class DGNavierStokesDualSplitting : public DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>
{
public:
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::value_type value_type;
  static const unsigned int n_actual_q_points_vel_nonlinear = (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::is_xwall) ? n_q_points_1d_xwall : fe_degree+(fe_degree+2)/2;

  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_nonlinear,dim,value_type,DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::is_xwall> FEFaceEval_Velocity_Velocity_nonlinear;
  typedef FEFaceEvaluationWrapperPressure<dim,fe_degree_p,fe_degree_xwall,DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::n_actual_q_points_vel_linear,1,value_type,
      DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::is_xwall> FEFaceEval_Pressure_Velocity_linear;
  typedef FEFaceEvaluationWrapperPressure<dim,fe_degree_p,fe_degree_xwall,n_actual_q_points_vel_nonlinear,1,value_type,
      DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::is_xwall> FEFaceEval_Pressure_Velocity_nonlinear;

  DGNavierStokesDualSplitting(parallel::distributed::Triangulation<dim> const &triangulation,
                              InputParametersNavierStokes const               &parameter)
    :
    DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>(triangulation,parameter),
    projection_operator(nullptr),
    velocity_linear(nullptr)
  {}

  virtual ~DGNavierStokesDualSplitting()
  {
    delete projection_operator;
    projection_operator = nullptr;
  }

  void setup_solvers ();

  // implicit solution of convective step
  void solve_nonlinear_convective_problem (parallel::distributed::Vector<value_type>       &dst,
                                           parallel::distributed::Vector<value_type> const &sum_alphai_ui,
                                           unsigned int                                    &newton_iterations,
                                           double                                          &average_linear_iterations);

  void evaluate_nonlinear_residual (parallel::distributed::Vector<value_type>       &dst,
                                    parallel::distributed::Vector<value_type> const &src);

  void vmult (parallel::distributed::Vector<value_type>       &dst,
              parallel::distributed::Vector<value_type> const &src) const;

  void apply_linearized_convective_problem (parallel::distributed::Vector<value_type>       &dst,
                                            parallel::distributed::Vector<value_type> const &src) const;

  // inverse mass matrix
  void apply_inverse_mass_matrix (parallel::distributed::Vector<value_type>       &dst,
                                  parallel::distributed::Vector<value_type> const &src) const;

  // body forces
  void  calculate_body_force (parallel::distributed::Vector<value_type>  &dst,
                              const value_type                           evaluation_time) const;

  // rhs pressure
  void rhs_pressure_divergence_term (parallel::distributed::Vector<value_type>        &dst,
                                     const parallel::distributed::Vector<value_type>  &src,
                                     const value_type                                 evaluation_time) const;

  void rhs_pressure_BC_term (parallel::distributed::Vector<value_type>       &dst,
                             const parallel::distributed::Vector<value_type> &src) const;

  void rhs_pressure_convective_term (parallel::distributed::Vector<value_type>       &dst,
                                     const parallel::distributed::Vector<value_type> &src) const;

  void rhs_pressure_viscous_term (parallel::distributed::Vector<value_type>       &dst,
                                  const parallel::distributed::Vector<value_type> &src) const;

  // solve pressure step
  unsigned int solve_pressure (parallel::distributed::Vector<value_type>        &dst,
                               const parallel::distributed::Vector<value_type>  &src) const;

  // projection step
  void rhs_projection (parallel::distributed::Vector<value_type>       &dst,
                       const parallel::distributed::Vector<value_type> &src_velocity,
                       const parallel::distributed::Vector<value_type> &src_pressure) const;

  unsigned int solve_projection (parallel::distributed::Vector<value_type>       &dst,
                                 const parallel::distributed::Vector<value_type> &src,
                                 const parallel::distributed::Vector<value_type> &velocity_n,
                                 double const                                    cfl) const;

  // viscous step
  void  rhs_viscous (parallel::distributed::Vector<value_type>       &dst,
                     const parallel::distributed::Vector<value_type> &src);

  unsigned int solve_viscous (parallel::distributed::Vector<value_type>       &dst,
                              const parallel::distributed::Vector<value_type> &src);


  // initialization of vectors
  void initialize_vector_for_newton_solver(parallel::distributed::Vector<value_type> &src) const
  {
    this->initialize_vector_velocity(src);
  }

  void set_solution_linearization(parallel::distributed::Vector<value_type> const *solution_linearization)
  {
    velocity_linear = solution_linearization;
  }

protected:
  std_cxx11::shared_ptr<PreconditionerBase<value_type> > helmholtz_preconditioner;
  ProjectionOperatorBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type> * projection_operator;
  std_cxx11::shared_ptr<ProjectionSolverBase<value_type> > projection_solver;

private:
  LaplaceOperator<dim,value_type> laplace_operator;

  std_cxx11::shared_ptr<PreconditionerBase<value_type> > preconditioner_pressure_poisson;
  std_cxx11::shared_ptr<CGSolver<LaplaceOperator<dim,value_type>, PreconditionerBase<value_type>,parallel::distributed::Vector<value_type> > > pressure_poisson_solver;

  HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,value_type> helmholtz_operator;
  std_cxx11::shared_ptr<IterativeSolverBase<parallel::distributed::Vector<value_type> > > helmholtz_solver;

  parallel::distributed::Vector<value_type> const *velocity_linear;
  parallel::distributed::Vector<value_type> temp;
  parallel::distributed::Vector<value_type> sum_alphai_ui;

  // implicit solution of convective step
  std_cxx11::shared_ptr<PreconditionerBase<value_type> > preconditioner_convective_problem;
  std_cxx11::shared_ptr<IterativeSolverBase<parallel::distributed::Vector<value_type> > > linear_solver;
  std_cxx11::shared_ptr<NewtonSolver<parallel::distributed::Vector<value_type>,
                                     DGNavierStokesDualSplitting<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>,
                                     IterativeSolverBase<parallel::distributed::Vector<value_type> > > >
    newton_solver;

  // setup of solvers
  void setup_convective_solver();

  void setup_pressure_poisson_solver();

  virtual void setup_projection_solver();

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

  // rhs pressure: convective term
  void local_rhs_pressure_convective_term (const MatrixFree<dim,value_type>                &data,
                                           parallel::distributed::Vector<value_type>       &dst,
                                           const parallel::distributed::Vector<value_type> &src,
                                           const std::pair<unsigned int,unsigned int>      &cell_range) const;

  void local_rhs_pressure_convective_term_face (const MatrixFree<dim,value_type>                 &data,
                                                parallel::distributed::Vector<value_type>        &dst,
                                                const parallel::distributed::Vector<value_type>  &src,
                                                const std::pair<unsigned int,unsigned int>       &face_range) const;

  void local_rhs_pressure_convective_term_boundary_face(const MatrixFree<dim,value_type>                 &data,
                                                        parallel::distributed::Vector<value_type>        &dst,
                                                        const parallel::distributed::Vector<value_type>  &src,
                                                        const std::pair<unsigned int,unsigned int>       &face_range) const;

  // rhs pressure: viscous term
  void local_rhs_pressure_viscous_term (const MatrixFree<dim,value_type>                &data,
                                        parallel::distributed::Vector<value_type>       &dst,
                                        const parallel::distributed::Vector<value_type> &src,
                                        const std::pair<unsigned int,unsigned int>      &cell_range) const;

  void local_rhs_pressure_viscous_term_face (const MatrixFree<dim,value_type>                 &data,
                                             parallel::distributed::Vector<value_type>        &dst,
                                             const parallel::distributed::Vector<value_type>  &src,
                                             const std::pair<unsigned int,unsigned int>       &face_range) const;

  void local_rhs_pressure_viscous_term_boundary_face(const MatrixFree<dim,value_type>                 &data,
                                                     parallel::distributed::Vector<value_type>        &dst,
                                                     const parallel::distributed::Vector<value_type>  &src,
                                                     const std::pair<unsigned int,unsigned int>       &face_range) const;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
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

  setup_pressure_poisson_solver();

  setup_projection_solver();

  setup_helmholtz_solver();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
setup_convective_solver ()
{
//  this->initialize_vector_velocity(velocity_linear);
  this->initialize_vector_velocity(temp);
  this->initialize_vector_velocity(sum_alphai_ui);

  // preconditioner implicit convective step
  preconditioner_convective_problem.reset(new InverseMassMatrixPreconditioner<dim,fe_degree,value_type>(
      this->data,
      static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector>::type >
          (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector::velocity),
      static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::QuadratureSelector>::type >
          (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::QuadratureSelector::velocity)));

  // linear solver (GMRES)
  GMRESSolverData solver_data;
  solver_data.max_iter = this->param.max_iter_linear;
  solver_data.solver_tolerance_abs = this->param.abs_tol_linear;
  solver_data.solver_tolerance_rel = this->param.rel_tol_linear;
  // use default value of right_preconditioning
  // use default value of max_n_tmp_vectors
  // use default value of compute_eigenvalues
  solver_data.use_preconditioner = true;

  linear_solver.reset(new GMRESSolver<DGNavierStokesDualSplitting<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>,
                                      PreconditionerBase<value_type>,
                                      parallel::distributed::Vector<value_type> >
      (*this,*preconditioner_convective_problem,solver_data));

  // Newton solver for nonlinear problem
  NewtonSolverData newton_solver_data;
  newton_solver_data.abs_tol = this->param.abs_tol_newton;
  newton_solver_data.rel_tol = this->param.rel_tol_newton;
  newton_solver_data.max_iter = this->param.max_iter_newton;

  newton_solver.reset(new NewtonSolver<parallel::distributed::Vector<value_type>,
                                       DGNavierStokesDualSplitting<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>,
                                       IterativeSolverBase<parallel::distributed::Vector<value_type> > >
     (newton_solver_data,*this,*linear_solver));
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
setup_pressure_poisson_solver ()
{
  // setup Laplace operator
  LaplaceOperatorData<dim> laplace_operator_data;
  laplace_operator_data.laplace_dof_index = static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector>::type >
    (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector::pressure);
  laplace_operator_data.laplace_quad_index = static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::QuadratureSelector>::type >
    (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::QuadratureSelector::pressure);
  laplace_operator_data.penalty_factor = this->param.IP_factor_pressure;

  if(this->param.use_approach_of_ferrer == true)
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << "Approach of Ferrer et al. is applied: IP_factor_pressure is scaled by time_step_size/time_step_size_ref!"
          << std::endl;

    laplace_operator_data.penalty_factor = this->param.IP_factor_pressure/this->time_step*this->param.deltat_ref;
  }

  laplace_operator_data.dirichlet_boundaries = this->neumann_boundary;
  laplace_operator_data.neumann_boundaries = this->dirichlet_boundary;
  laplace_operator_data.periodic_face_pairs_level0 = this->periodic_face_pairs;
  laplace_operator.reinit(this->data,this->mapping,laplace_operator_data);

  // setup preconditioner
  if(this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::Jacobi)
  {
    preconditioner_pressure_poisson.reset(new JacobiPreconditioner<value_type, LaplaceOperator<dim,value_type> >(laplace_operator));
  }
  else if(this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::GeometricMultigrid)
  {
    MultigridData mg_data;
    mg_data = this->param.multigrid_data_pressure_poisson;

    // use single precision for multigrid
    typedef float Number;

    preconditioner_pressure_poisson.reset(new MyMultigridPreconditioner<dim,value_type,LaplaceOperator<dim,Number>, LaplaceOperatorData<dim> >(
        mg_data,
        this->dof_handler_p,
        this->mapping,
        laplace_operator_data,
        laplace_operator_data.dirichlet_boundaries));
  }
  else
  {
    AssertThrow(this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::None ||
                this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::Jacobi ||
                this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::GeometricMultigrid,
                ExcMessage("Specified preconditioner for pressure Poisson equation not implemented"));
  }

  // setup solver data
  CGSolverData solver_data;
  // use default value of max_iter
  solver_data.solver_tolerance_abs = this->param.abs_tol_pressure;
  solver_data.solver_tolerance_rel = this->param.rel_tol_pressure;
  // default value of use_preconditioner = false
  if(this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::Jacobi ||
     this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::GeometricMultigrid)
  {
    solver_data.use_preconditioner = true;
  }

  // setup solver
  pressure_poisson_solver.reset(new CGSolver<LaplaceOperator<dim,value_type>, PreconditionerBase<value_type>, parallel::distributed::Vector<value_type> >(
      laplace_operator,
      *preconditioner_pressure_poisson,
      solver_data));
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
setup_projection_solver ()
{
  // initialize projection solver
  ProjectionOperatorData projection_operator_data;
  projection_operator_data.penalty_parameter_divergence = this->param.penalty_factor_divergence;
  projection_operator_data.penalty_parameter_continuity = this->param.penalty_factor_continuity;
  projection_operator_data.solve_stokes_equations = (this->param.equation_type == EquationType::Stokes);

  if(this->param.projection_type == ProjectionType::NoPenalty)
  {
    projection_solver.reset(new ProjectionSolverNoPenalty<dim, fe_degree, value_type>(
        this->data,
        static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector>::type >
          (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::QuadratureSelector>::type >
          (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::QuadratureSelector::velocity)));
  }
  else if(this->param.projection_type == ProjectionType::DivergencePenalty &&
          this->param.solver_projection == SolverProjection::LU)
  {
    if(projection_operator != nullptr)
    {
      delete projection_operator;
      projection_operator = nullptr;
    }

    projection_operator = new ProjectionOperatorBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(
        this->data,
        this->fe_param,
        static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector>::type >
          (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::QuadratureSelector>::type >
          (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::QuadratureSelector::velocity),
        projection_operator_data);

    projection_solver.reset(new DirectProjectionSolverDivergencePenalty
        <dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(projection_operator));
  }
  else if(this->param.projection_type == ProjectionType::DivergencePenalty &&
          this->param.solver_projection == SolverProjection::PCG)
  {
    if(projection_operator != nullptr)
    {
      delete projection_operator;
      projection_operator = nullptr;
    }

    projection_operator = new ProjectionOperatorDivergencePenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(
        this->data,
        this->fe_param,
        static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector>::type >
          (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::QuadratureSelector>::type >
          (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::QuadratureSelector::velocity),
        projection_operator_data);

    ProjectionSolverData projection_solver_data;
    projection_solver_data.solver_tolerance_abs = this->param.abs_tol_projection;
    projection_solver_data.solver_tolerance_rel = this->param.rel_tol_projection;
    projection_solver_data.solver_projection = this->param.solver_projection;
    projection_solver_data.preconditioner_projection = this->param.preconditioner_projection;

    projection_solver.reset(new IterativeProjectionSolverDivergencePenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(
        projection_operator,
        projection_solver_data));
  }
  else if(this->param.projection_type == ProjectionType::DivergenceAndContinuityPenalty)
  {
    if(projection_operator != nullptr)
    {
      delete projection_operator;
      projection_operator = nullptr;
    }

    projection_operator = new ProjectionOperatorDivergenceAndContinuityPenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(
        this->data,
        this->fe_param,
        static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector>::type >
          (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::QuadratureSelector>::type >
          (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::QuadratureSelector::velocity),
        projection_operator_data);

    ProjectionSolverData projection_solver_data;
    projection_solver_data.solver_tolerance_abs = this->param.abs_tol_projection;
    projection_solver_data.solver_tolerance_rel = this->param.rel_tol_projection;
    projection_solver_data.solver_projection = this->param.solver_projection;
    projection_solver_data.preconditioner_projection = this->param.preconditioner_projection;

    projection_solver.reset(new IterativeProjectionSolverDivergenceAndContinuityPenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(
                              projection_operator,
                              projection_solver_data));
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
setup_helmholtz_solver ()
{
  // setup helmholtz operator
  HelmholtzOperatorData<dim> helmholtz_operator_data;

  helmholtz_operator_data.mass_matrix_operator_data = this->mass_matrix_operator_data;
  helmholtz_operator_data.viscous_operator_data = this->viscous_operator_data;

  helmholtz_operator_data.dof_index = static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector>::type >
                                        (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector::velocity);
  helmholtz_operator_data.mass_matrix_coefficient = this->scaling_factor_time_derivative_term;
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
    helmholtz_solver.reset(new CGSolver<HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,value_type>, PreconditionerBase<value_type>, parallel::distributed::Vector<value_type> >(
        helmholtz_operator,
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
    helmholtz_solver.reset(new GMRESSolver<HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,value_type>, PreconditionerBase<value_type>, parallel::distributed::Vector<value_type> >(
        helmholtz_operator,
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

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
setup_helmholtz_preconditioner (HelmholtzOperatorData<dim> &helmholtz_operator_data)
{
  if(this->param.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix)
  {
    helmholtz_preconditioner.reset(new InverseMassMatrixPreconditioner<dim,fe_degree,value_type>(
        this->data,
        static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector>::type >
          (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::QuadratureSelector>::type >
          (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::QuadratureSelector::velocity)));
  }
  else if(this->param.preconditioner_viscous == PreconditionerViscous::Jacobi)
  {
    helmholtz_preconditioner.reset(new JacobiPreconditioner<value_type,
        HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall, value_type> >(helmholtz_operator));
  }
  else if(this->param.preconditioner_viscous == PreconditionerViscous::GeometricMultigrid)
  {
    MultigridData mg_data;
    mg_data = this->param.multigrid_data_viscous;

    // use single precision for multigrid
    typedef float Number;

    helmholtz_preconditioner.reset(new MyMultigridPreconditioner<dim,value_type,HelmholtzOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>, HelmholtzOperatorData<dim> >(
        mg_data,
        this->dof_handler_u,
        this->mapping,
        helmholtz_operator_data,
        this->dirichlet_boundary,
        this->fe_param));
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
vmult (parallel::distributed::Vector<value_type>       &dst,
       parallel::distributed::Vector<value_type> const &src) const
{
  apply_linearized_convective_problem(dst,src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
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

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
solve_nonlinear_convective_problem (parallel::distributed::Vector<value_type>       &dst,
                                    parallel::distributed::Vector<value_type> const &sum_alphai_ui,
                                    unsigned int                                    &newton_iterations,
                                    double                                          &average_linear_iterations)
{
  this->sum_alphai_ui = sum_alphai_ui;
  newton_solver->solve(dst,newton_iterations,average_linear_iterations);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
evaluate_nonlinear_residual (parallel::distributed::Vector<value_type>             &dst,
                             const parallel::distributed::Vector<value_type>       &src)
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

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
apply_inverse_mass_matrix (parallel::distributed::Vector<value_type>       &dst,
                           parallel::distributed::Vector<value_type> const &src) const
{
  this->inverse_mass_matrix_operator->apply_inverse_mass_matrix(dst,src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
calculate_body_force (parallel::distributed::Vector<value_type>  &dst,
                      const value_type                           evaluation_time) const
{
  this->body_force_operator.evaluate(dst,evaluation_time);

  this->inverse_mass_matrix_operator->apply_inverse_mass_matrix(dst,dst);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
unsigned int DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
solve_pressure (parallel::distributed::Vector<value_type>        &dst,
                const parallel::distributed::Vector<value_type>  &src) const
{
  unsigned int n_iter = pressure_poisson_solver->solve(dst,src);

  return n_iter;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
rhs_pressure_divergence_term (parallel::distributed::Vector<value_type>        &dst,
                              const parallel::distributed::Vector<value_type>  &src,
                              const value_type                                 evaluation_time) const
{
  this->divergence_operator.evaluate(dst,src,evaluation_time);

  if(this->param.small_time_steps_stability == true)
    dst *= -1.0;
  else
    dst *= -this->scaling_factor_time_derivative_term;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
rhs_pressure_BC_term (parallel::distributed::Vector<value_type>       &dst,
                      const parallel::distributed::Vector<value_type> &src) const
{
  this->data.loop (&DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_BC_term,
                   &DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_BC_term_face,
                   &DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_BC_term_boundary_face,
                   this, dst, src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_pressure_BC_term (const MatrixFree<dim,value_type>                 &,
                            parallel::distributed::Vector<value_type>        &,
                            const parallel::distributed::Vector<value_type>  &,
                            const std::pair<unsigned int,unsigned int>       &) const
{

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_pressure_BC_term_face (const MatrixFree<dim,value_type>                &,
                                 parallel::distributed::Vector<value_type>       &,
                                 const parallel::distributed::Vector<value_type> &,
                                 const std::pair<unsigned int,unsigned int>      &) const
{

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_pressure_BC_term_boundary_face (const MatrixFree<dim,value_type>                 &data,
                                          parallel::distributed::Vector<value_type>        &dst,
                                          const parallel::distributed::Vector<value_type>  &,
                                          const std::pair<unsigned int,unsigned int>       &face_range) const
{
  FEFaceEval_Pressure_Velocity_nonlinear fe_eval_pressure(data,this->fe_param,true,
      static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector>::type >
        (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector::pressure));

  //TODO: quadrature formula
//    FEFaceEval_Pressure_Velocity_linear fe_eval_pressure(data,this->fe_param,true,
//        static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure));

  // set the correct time for the evaluation of the right_hand_side - function
  if(this->param.right_hand_side == true)
    this->field_functions->right_hand_side->set_time(this->evaluation_time);

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval_pressure.reinit (face);

    double factor = laplace_operator.get_penalty_factor();
    VectorizedArray<value_type> tau_IP = fe_eval_pressure.read_cell_data(laplace_operator.get_array_penalty_parameter()) * (value_type)factor;

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

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
rhs_pressure_convective_term (parallel::distributed::Vector<value_type>       &dst,
                              const parallel::distributed::Vector<value_type> &src) const
{
  this->data.loop (&DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_convective_term,
                   &DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_convective_term_face,
                   &DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_convective_term_boundary_face,
                   this, dst, src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_pressure_convective_term (const MatrixFree<dim,value_type>                &,
                                    parallel::distributed::Vector<value_type>       &,
                                    const parallel::distributed::Vector<value_type> &,
                                    const std::pair<unsigned int,unsigned int>      &) const
{

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_pressure_convective_term_face (const MatrixFree<dim,value_type>                &,
                                         parallel::distributed::Vector<value_type>       &,
                                         const parallel::distributed::Vector<value_type> &,
                                         const std::pair<unsigned int,unsigned int>      &) const
{

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_pressure_convective_term_boundary_face (const MatrixFree<dim,value_type>                 &data,
                                                  parallel::distributed::Vector<value_type>        &dst,
                                                  const parallel::distributed::Vector<value_type>  &src,
                                                  const std::pair<unsigned int,unsigned int>       &face_range) const
{

  FEFaceEval_Velocity_Velocity_nonlinear fe_eval_velocity(data,this->fe_param,true,
      static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector>::type >
        (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector::velocity));

  FEFaceEval_Pressure_Velocity_nonlinear fe_eval_pressure(data,this->fe_param,true,
      static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector>::type >
        (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector::pressure));

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

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
rhs_pressure_viscous_term (parallel::distributed::Vector<value_type>       &dst,
                           const parallel::distributed::Vector<value_type> &src) const
{
  this->data.loop (&DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_viscous_term,
                   &DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_viscous_term_face,
                   &DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_viscous_term_boundary_face,
                   this, dst, src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_pressure_viscous_term (const MatrixFree<dim,value_type>                &,
                                 parallel::distributed::Vector<value_type>       &,
                                 const parallel::distributed::Vector<value_type> &,
                                 const std::pair<unsigned int,unsigned int>      &) const
{

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_pressure_viscous_term_face (const MatrixFree<dim,value_type>                 &,
                                      parallel::distributed::Vector<value_type>        &,
                                      const parallel::distributed::Vector<value_type>  &,
                                      const std::pair<unsigned int,unsigned int>       &) const
{

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_pressure_viscous_term_boundary_face (const MatrixFree<dim,value_type>                &data,
                                               parallel::distributed::Vector<value_type>       &dst,
                                               const parallel::distributed::Vector<value_type> &src,
                                               const std::pair<unsigned int,unsigned int>      &face_range) const
{
  FEFaceEval_Velocity_Velocity_nonlinear fe_eval_omega(data,this->fe_param,true,
      static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector>::type >
        (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector::velocity));

  FEFaceEval_Pressure_Velocity_nonlinear fe_eval_pressure(data,this->fe_param,true,
      static_cast<typename std::underlying_type<typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector>::type >
        (DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector::pressure));

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

        Tensor<1,dim,VectorizedArray<value_type> > curl_omega = CurlCompute<dim,FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_nonlinear,dim,value_type,DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::is_xwall> >::compute(fe_eval_omega,q);
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

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
unsigned int DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
solve_projection (parallel::distributed::Vector<value_type>       &dst,
                  const parallel::distributed::Vector<value_type> &src,
                  const parallel::distributed::Vector<value_type> &velocity_n,
                  double const                                    cfl) const
{
  if(this->param.projection_type != ProjectionType::NoPenalty)
    projection_operator->calculate_array_penalty_parameter(velocity_n,cfl,this->time_step);

  unsigned int n_iter = projection_solver->solve(dst,src);

  return n_iter;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
rhs_projection (parallel::distributed::Vector<value_type>        &dst,
                const parallel::distributed::Vector<value_type>  &src_velocity,
                const parallel::distributed::Vector<value_type>  &src_pressure) const
{
  this->gradient_operator.evaluate(dst,src_pressure,this->evaluation_time);

  dst *= - 1.0/this->scaling_factor_time_derivative_term;

  this->mass_matrix_operator.apply_add(dst,src_velocity);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
unsigned int DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
solve_viscous (parallel::distributed::Vector<value_type>       &dst,
               const parallel::distributed::Vector<value_type> &src)
{
  // update helmholtz_operator
  helmholtz_operator.set_mass_matrix_coefficient(this->scaling_factor_time_derivative_term);
  // viscous_operator.set_constant_viscosity(viscosity);
  // viscous_operator.set_variable_viscosity(viscosity);

  // update preconditioner
  if(this->param.preconditioner_viscous == PreconditionerViscous::Jacobi)
  {
    // TODO: recalculate diagonal (say every 10, 100 time steps) in case of varying parameters
    // of mass matrix term or viscous term, e.g. strongly varying time step sizes (adaptive time step control)
    // or strongly varying viscosity (turbulence)
    /*
    std_cxx11::shared_ptr<JacobiPreconditioner<value_type,HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall> > >
      jacobi_preconditioner = std::dynamic_pointer_cast<JacobiPreconditioner<value_type,HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall> > >(helmholtz_preconditioner);
    jacobi_preconditioner->recalculate_diagonal(helmholtz_operator);
    */
  }
  else if(this->param.preconditioner_viscous == PreconditionerViscous::GeometricMultigrid)
  {
    // TODO: update multigrid preconditioner (diagonals) in case of varying parameters

    // check multigrid smoothing
    /*
    typedef float Number;

    parallel::distributed::Vector<value_type> check1;
    helmholtz_operator.initialize_dof_vector(check1);
    parallel::distributed::Vector<value_type> check2(check1), tmp(check1);
    parallel::distributed::Vector<Number> check3;
    check3 = check1;
    for (unsigned int i=0; i<check1.size(); ++i)
      check1(i) = (double)rand()/RAND_MAX;
    helmholtz_operator.vmult(tmp, check1);
    tmp *= -1.0;
    helmholtz_preconditioner->vmult(check2, tmp);
    check2 += check1;

    parallel::distributed::Vector<Number> tmp_float, check1_float;
    tmp_float = tmp;
    check1_float = check1;
    std_cxx11::shared_ptr<MyMultigridPreconditioner<dim,value_type,HelmholtzOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall,Number>, HelmholtzOperatorData<dim> > >
      my_preconditioner = std::dynamic_pointer_cast<MyMultigridPreconditioner<dim,value_type,HelmholtzOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall,Number>, HelmholtzOperatorData<dim> > >(helmholtz_preconditioner);
    // mg_smoother is private in MyMultigridPreconditioner -> make mg_smoother public when using this function to test the multigrid method
    my_preconditioner->mg_smoother[my_preconditioner->mg_smoother.max_level()].vmult(check3,tmp_float);
    check3 += check1_float;

    //my_preconditioner->mg_matrices[my_preconditioner->mg_matrices.max_level()].vmult(tmp_float,check1_float);
    //check1_float = tmp;
    //tmp_float *= -1.0;
    //std::cout<<"L2 norm tmp = "<<tmp_float.l2_norm()<<std::endl;
    //std::cout<<"L2 norm check = "<<check1_float.l2_norm()<<std::endl;

    DataOut<dim> data_out;
    data_out.attach_dof_handler (helmholtz_operator.get_data().get_dof_handler(helmholtz_operator.get_operator_data().dof_index));

    std::vector<std::string> initial (dim, "initial");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      initial_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector (helmholtz_operator.get_data().get_dof_handler(helmholtz_operator.get_operator_data().dof_index),check1, initial, initial_component_interpretation);

    std::vector<std::string> mg_cycle (dim, "mg_cycle");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      mg_cylce_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector (helmholtz_operator.get_data().get_dof_handler(helmholtz_operator.get_operator_data().dof_index),check2, mg_cycle, mg_cylce_component_interpretation);

    std::vector<std::string> smoother (dim, "smoother");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      smoother_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector (helmholtz_operator.get_data().get_dof_handler(helmholtz_operator.get_operator_data().dof_index),check3, smoother, smoother_component_interpretation);

    data_out.build_patches (helmholtz_operator.get_data().get_dof_handler(helmholtz_operator.get_operator_data().dof_index).get_fe().degree*3);
    std::ostringstream filename;
    filename << "smoothing.vtk";

    std::ofstream output (filename.str().c_str());
    data_out.write_vtk(output);
    std::abort();
    */
  }

  unsigned int n_iter = helmholtz_solver->solve(dst,src);

  return n_iter;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
rhs_viscous (parallel::distributed::Vector<value_type>       &dst,
             const parallel::distributed::Vector<value_type> &src)
{
  this->mass_matrix_operator.apply(dst,src);
  dst *= this->scaling_factor_time_derivative_term;

  this->viscous_operator.rhs_add(dst,this->evaluation_time);
}


#endif /* INCLUDE_DGNAVIERSTOKESDUALSPLITTING_H_ */
