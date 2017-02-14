/*
 * DGNavierStokesProjectionMethods.h
 *
 *  Created on: Nov 7, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_DGNAVIERSTOKESPROJECTIONMETHODS_H_
#define INCLUDE_DGNAVIERSTOKESPROJECTIONMETHODS_H_

#include "DGNavierStokesBase.h"
#include "ProjectionSolver.h"
#include "poisson_solver.h"

#include "IterativeSolvers.h"

#include "MultigridPreconditionerLaplace.h"

/*
 *  Base class for projection type splitting methods such as
 *  the high-order dual splitting scheme (velocity-correction) or
 *  pressure correction schemes
 */
template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
class DGNavierStokesProjectionMethods : public DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>
{
public:
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::value_type value_type;

  DGNavierStokesProjectionMethods(parallel::distributed::Triangulation<dim> const &triangulation,
                                  InputParametersNavierStokes<dim> const          &parameter)
    :
    DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>(triangulation,parameter),
    projection_operator(nullptr)
  {}

  virtual ~DGNavierStokesProjectionMethods()
  {
    delete projection_operator;
    projection_operator = nullptr;
  }

  virtual void setup_solvers(double const &time_step_size,
                             double const &scaling_factor_time_derivative_term) = 0;

  // velocity divergence
  void evaluate_velocity_divergence_term(parallel::distributed::Vector<value_type>        &dst,
                                         const parallel::distributed::Vector<value_type>  &src,
                                         const double                                     evaluation_time) const;

  // mass_matrix
  void apply_mass_matrix(parallel::distributed::Vector<value_type>       &dst,
                         parallel::distributed::Vector<value_type> const &src) const;

  // pressure gradient term
  void evaluate_pressure_gradient_term(parallel::distributed::Vector<value_type>       &dst,
                                       parallel::distributed::Vector<value_type> const &src,
                                       value_type const                                evaluation_time) const;

  // rhs viscous term (add)
  void rhs_add_viscous_term(parallel::distributed::Vector<value_type> &dst,
                            const value_type                          evaluation_time) const;

  // rhs pressure Poisson equation: inhomogeneous parts of boundary face
  // integrals of negative Laplace operator
  void rhs_ppe_laplace_add(parallel::distributed::Vector<value_type> &dst,
                           double const                              &evaluation_time) const;

  // solve pressure step
  unsigned int solve_pressure (parallel::distributed::Vector<value_type>        &dst,
                               const parallel::distributed::Vector<value_type>  &src) const;

  // solve projection step
  unsigned int solve_projection (parallel::distributed::Vector<value_type>       &dst,
                                 const parallel::distributed::Vector<value_type> &src,
                                 const parallel::distributed::Vector<value_type> &velocity_n,
                                 double const                                    cfl,
                                 double const                                    time_step_size) const;

protected:
  virtual void setup_pressure_poisson_solver(double const time_step_size);
  void setup_projection_solver();

  // Pressure Poisson equation
  LaplaceOperator<dim,value_type> laplace_operator;
  std_cxx11::shared_ptr<PreconditionerBase<value_type> > preconditioner_pressure_poisson;
  std_cxx11::shared_ptr<IterativeSolverBase<parallel::distributed::Vector<value_type> > > pressure_poisson_solver;

  // Projection method
  ProjectionOperatorBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> * projection_operator;
  std_cxx11::shared_ptr<ProjectionSolverBase<value_type> > projection_solver;

private:
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesProjectionMethods<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
setup_pressure_poisson_solver (double const time_step_size)
{
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector DofHandlerSelector;
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::QuadratureSelector QuadratureSelector;

  // setup Laplace operator
  LaplaceOperatorData<dim> laplace_operator_data;
  laplace_operator_data.laplace_dof_index = static_cast<typename std::underlying_type<DofHandlerSelector>::type>(DofHandlerSelector::pressure);
  laplace_operator_data.laplace_quad_index = static_cast<typename std::underlying_type<QuadratureSelector>::type>(QuadratureSelector::pressure);
  laplace_operator_data.penalty_factor = this->param.IP_factor_pressure;

  // TODO: do this in derived classes
  if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    laplace_operator_data.needs_mean_value_constraint = this->param.pure_dirichlet_bc;
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    // One can show that the linear system of equations of the PPE is consistent
    // in case of the pressure-correction scheme if the velocity Dirichlet BC is consistent.
    // So there should be no need to solve a tranformed linear system of equations.
//    laplace_operator_data.needs_mean_value_constraint = false;

    // In principle, it works (since the linear system of equations is consistent)
    // but we detected no convergence for some test cases and specific parameters.
    // Hence, for reasons of robustness we also solve a transformed linear system of equations
    // in case of the pressure-correction scheme.
    laplace_operator_data.needs_mean_value_constraint = this->param.pure_dirichlet_bc;
  }

  if(this->param.use_approach_of_ferrer == true)
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << "Approach of Ferrer et al. is applied: IP_factor_pressure is scaled by time_step_size/time_step_size_ref!"
          << std::endl;

    // only makes sense in case of constant time step sizes
    laplace_operator_data.penalty_factor = this->param.IP_factor_pressure/time_step_size*this->param.deltat_ref;
  }

  laplace_operator_data.bc = this->boundary_descriptor_laplace;

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

    typedef MyMultigridPreconditionerLaplace<dim, value_type, LaplaceOperator<dim,Number>, LaplaceOperatorData<dim> > MULTIGRID;

    preconditioner_pressure_poisson.reset(new MULTIGRID());

    std_cxx11::shared_ptr<MULTIGRID> mg_preconditioner = std::dynamic_pointer_cast<MULTIGRID>(preconditioner_pressure_poisson);

    mg_preconditioner->initialize(mg_data,
                                  this->dof_handler_p,
                                  this->mapping,
                                  laplace_operator_data,
                                  laplace_operator_data.bc->dirichlet);
  }
  else
  {
    AssertThrow(this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::None ||
                this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::Jacobi ||
                this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::GeometricMultigrid,
                ExcMessage("Specified preconditioner for pressure Poisson equation not implemented"));
  }

  if(this->param.solver_pressure_poisson == SolverPressurePoisson::PCG)
  {
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
    pressure_poisson_solver.reset(new CGSolver<LaplaceOperator<dim,value_type>,
                                               PreconditionerBase<value_type>,
                                               parallel::distributed::Vector<value_type> >
       (laplace_operator,
        *preconditioner_pressure_poisson,
        solver_data));
  }
  else if(this->param.solver_pressure_poisson == SolverPressurePoisson::FGMRES)
  {
    FGMRESSolverData solver_data;
    // use default value of max_iter
    solver_data.solver_tolerance_abs = this->param.abs_tol_pressure;
    solver_data.solver_tolerance_rel = this->param.rel_tol_pressure;
    solver_data.max_n_tmp_vectors = this->param.max_n_tmp_vectors_pressure_poisson;
    // use default value of update_preconditioner (=false)

    if(this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::Jacobi ||
       this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::GeometricMultigrid)
    {
      solver_data.use_preconditioner = true;
    }

    pressure_poisson_solver.reset(new FGMRESSolver<LaplaceOperator<dim,value_type>,
                                                   PreconditionerBase<value_type>,
                                                   parallel::distributed::Vector<value_type> >
        (laplace_operator,
         *preconditioner_pressure_poisson,
         solver_data));
  }
  else
  {
    AssertThrow(this->param.solver_viscous == SolverViscous::PCG ||
                this->param.solver_viscous == SolverViscous::FGMRES,
                ExcMessage("Specified  solver for pressure Poisson equation not implemented - possibilities are PCG and FGMRES"));
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesProjectionMethods<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
setup_projection_solver ()
{
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector DofHandlerSelector;
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::QuadratureSelector QuadratureSelector;

  // initialize projection solver
  ProjectionOperatorData projection_operator_data;
  projection_operator_data.penalty_parameter_divergence = this->param.penalty_factor_divergence;
  projection_operator_data.penalty_parameter_continuity = this->param.penalty_factor_continuity;
  projection_operator_data.solve_stokes_equations = (this->param.equation_type == EquationType::Stokes);

  if(this->param.projection_type == ProjectionType::NoPenalty)
  {
    projection_solver.reset(new ProjectionSolverNoPenalty<dim, fe_degree, value_type>(
        this->data,
        static_cast<typename std::underlying_type<DofHandlerSelector>::type>(DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type<QuadratureSelector>::type>(QuadratureSelector::velocity)));
  }
  else if(this->param.projection_type == ProjectionType::DivergencePenalty &&
          this->param.solver_projection == SolverProjection::LU)
  {
    if(projection_operator != nullptr)
    {
      delete projection_operator;
      projection_operator = nullptr;
    }

    projection_operator = new ProjectionOperatorBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>(
        this->data,
        static_cast<typename std::underlying_type<DofHandlerSelector>::type>(DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type<QuadratureSelector>::type>(QuadratureSelector::velocity),
        projection_operator_data);

    projection_solver.reset(new DirectProjectionSolverDivergencePenalty
        <dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>(projection_operator));
  }
  else if(this->param.projection_type == ProjectionType::DivergencePenalty &&
          this->param.solver_projection == SolverProjection::PCG)
  {
    if(projection_operator != nullptr)
    {
      delete projection_operator;
      projection_operator = nullptr;
    }

    projection_operator = new ProjectionOperatorDivergencePenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>(
        this->data,
        static_cast<typename std::underlying_type<DofHandlerSelector>::type>(DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type<QuadratureSelector>::type>(QuadratureSelector::velocity),
        projection_operator_data);

    ProjectionSolverData projection_solver_data;
    projection_solver_data.solver_tolerance_abs = this->param.abs_tol_projection;
    projection_solver_data.solver_tolerance_rel = this->param.rel_tol_projection;
    projection_solver_data.solver_projection = this->param.solver_projection;
    projection_solver_data.preconditioner_projection = this->param.preconditioner_projection;

    projection_solver.reset(new IterativeProjectionSolverDivergencePenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>(
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

    projection_operator = new ProjectionOperatorDivergenceAndContinuityPenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>(
        this->data,
        static_cast<typename std::underlying_type<DofHandlerSelector>::type>(DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type<QuadratureSelector>::type>(QuadratureSelector::velocity),
        projection_operator_data);

    ProjectionSolverData projection_solver_data;
    projection_solver_data.solver_tolerance_abs = this->param.abs_tol_projection;
    projection_solver_data.solver_tolerance_rel = this->param.rel_tol_projection;
    projection_solver_data.solver_projection = this->param.solver_projection;
    projection_solver_data.preconditioner_projection = this->param.preconditioner_projection;

    projection_solver.reset(new IterativeProjectionSolverDivergenceAndContinuityPenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>(
                              projection_operator,
                              projection_solver_data));
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesProjectionMethods<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
evaluate_velocity_divergence_term(parallel::distributed::Vector<value_type>        &dst,
                                  const parallel::distributed::Vector<value_type>  &src,
                                  const double                                     evaluation_time) const
{
  this->divergence_operator.evaluate(dst,src,evaluation_time);
}

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesProjectionMethods<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
apply_mass_matrix (parallel::distributed::Vector<value_type>       &dst,
                   parallel::distributed::Vector<value_type> const &src) const
{
  this->mass_matrix_operator.apply(dst,src);
}

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesProjectionMethods<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
evaluate_pressure_gradient_term (parallel::distributed::Vector<value_type>       &dst,
                                 parallel::distributed::Vector<value_type> const &src,
                                 value_type const                                evaluation_time) const
{
  this->gradient_operator.evaluate(dst,src,evaluation_time);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesProjectionMethods<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
rhs_add_viscous_term(parallel::distributed::Vector<value_type>  &dst,
                     const value_type                           evaluation_time) const
{
  this->viscous_operator.rhs_add(dst,evaluation_time);
}

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesProjectionMethods<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
rhs_ppe_laplace_add(parallel::distributed::Vector<value_type> &dst,
                    double const                              &evaluation_time) const
{
  const LaplaceOperatorData<dim> &data = this->laplace_operator.get_operator_data();

  // Set correct time for evaluation of functions on pressure Dirichlet boundaries
  // (not needed for pressure Neumann boundaries because all functions are ZeroFunction in Neumann BC map!)
  for(typename std::map<types::boundary_id, std_cxx11::shared_ptr<Function<dim> > >::const_iterator
        it = data.bc->dirichlet.begin(); it != data.bc->dirichlet.end(); ++it)
  {
    it->second->set_time(evaluation_time);
  }

  this->laplace_operator.rhs_add(dst);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
unsigned int DGNavierStokesProjectionMethods<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
solve_pressure (parallel::distributed::Vector<value_type>        &dst,
                const parallel::distributed::Vector<value_type>  &src) const
{
  unsigned int n_iter = this->pressure_poisson_solver->solve(dst,src);

  return n_iter;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
unsigned int DGNavierStokesProjectionMethods<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
solve_projection (parallel::distributed::Vector<value_type>       &dst,
                  const parallel::distributed::Vector<value_type> &src,
                  const parallel::distributed::Vector<value_type> &velocity_n,
                  double const                                    cfl,
                  double const                                    time_step_size) const
{
  // Update projection operator
  if(this->param.projection_type != ProjectionType::NoPenalty)
    this->projection_operator->calculate_array_penalty_parameter(velocity_n,cfl,time_step_size);

  unsigned int n_iter = this->projection_solver->solve(dst,src);

  return n_iter;
}


#endif /* INCLUDE_DGNAVIERSTOKESPROJECTIONMETHODS_H_ */
