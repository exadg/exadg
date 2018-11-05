/*
 * dg_navier_stokes_projection_methods.h
 *
 *  Created on: Nov 7, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_PROJECTION_METHODS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_PROJECTION_METHODS_H_

#include "../../incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_base.h"

namespace IncNS
{
/*
 * Base class for projection type splitting methods such as the high-order dual splitting scheme
 * (velocity-correction) or pressure correction schemes
 */
template<int dim, int degree_u, int degree_p, typename Number>
class DGNavierStokesProjectionMethods : public DGNavierStokesBase<dim, degree_u, degree_p, Number>
{
public:
  typedef DGNavierStokesBase<dim, degree_u, degree_p, Number> BASE;

  typedef typename BASE::VectorType VectorType;

  typedef typename BASE::Postprocessor Postprocessor;

  DGNavierStokesProjectionMethods(parallel::distributed::Triangulation<dim> const & triangulation,
                                  InputParameters<dim> const &                      parameters_in,
                                  std::shared_ptr<Postprocessor> postprocessor_in)
    : BASE(triangulation, parameters_in, postprocessor_in)
  {
    AssertThrow(degree_p > 0,
                ExcMessage("Polynomial degree of pressure shape functions has to be larger than "
                           "zero for dual splitting scheme and pressure-correction scheme."));
  }

  virtual ~DGNavierStokesProjectionMethods()
  {
  }

  virtual void
  setup_solvers(double const & time_step_size,
                double const & scaling_factor_time_derivative_term) = 0;

  // velocity divergence
  void
  evaluate_velocity_divergence_term(VectorType &       dst,
                                    VectorType const & src,
                                    double const       evaluation_time) const;

  // pressure gradient term
  void
  evaluate_pressure_gradient_term(VectorType &       dst,
                                  VectorType const & src,
                                  double const       evaluation_time) const;

  // rhs viscous term (add)
  void
  rhs_add_viscous_term(VectorType & dst, double const evaluation_time) const;

  // rhs pressure Poisson equation: inhomogeneous parts of boundary face
  // integrals of negative Laplace operator
  void
  rhs_ppe_laplace_add(VectorType & dst, double const & evaluation_time) const;

  // solve pressure step
  unsigned int
  solve_pressure(VectorType & dst, VectorType const & src) const;

  // apply projection operator
  void
  apply_projection_operator(VectorType & dst, VectorType const & src) const;

  // Evaluate residual of steady, coupled incompressible Navier-Stokes equations
  void
  evaluate_nonlinear_residual_steady(VectorType &       dst_u,
                                     VectorType &       dst_p,
                                     VectorType const & src_u,
                                     VectorType const & src_p,
                                     double const &     evaluation_time);

  // apply homogeneous Laplace operator
  void
  apply_laplace_operator(VectorType & dst, VectorType const & src) const;

protected:
  virtual void
  setup_pressure_poisson_solver(double const time_step_size);

  // Pressure Poisson equation
  Poisson::LaplaceOperator<dim, degree_p, Number> laplace_operator;

  std::shared_ptr<PreconditionerBase<Number>> preconditioner_pressure_poisson;

  std::shared_ptr<IterativeSolverBase<VectorType>> pressure_poisson_solver;
};

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesProjectionMethods<dim, degree_u, degree_p, Number>::setup_pressure_poisson_solver(
  double const time_step_size)
{
  // setup Laplace operator
  Poisson::LaplaceOperatorData<dim> laplace_operator_data;
  laplace_operator_data.dof_index  = this->get_dof_index_pressure();
  laplace_operator_data.quad_index = this->get_quad_index_pressure();
  laplace_operator_data.IP_factor  = this->param.IP_factor_pressure;

  // TODO: do this in derived classes
  if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    laplace_operator_data.operator_is_singular = this->param.pure_dirichlet_bc;
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    // One can show that the linear system of equations of the PPE is consistent
    // in case of the pressure-correction scheme if the velocity Dirichlet BC is consistent.
    // So there should be no need to solve a tranformed linear system of equations.
    //    laplace_operator_data.operator_is_singular = false;

    // In principle, it works (since the linear system of equations is consistent)
    // but we detected no convergence for some test cases and specific parameters.
    // Hence, for reasons of robustness we also solve a transformed linear system of equations
    // in case of the pressure-correction scheme.
    laplace_operator_data.operator_is_singular = this->param.pure_dirichlet_bc;
  }

  if(this->param.use_approach_of_ferrer == true)
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout
      << "Approach of Ferrer et al. is applied: IP_factor_pressure is scaled by time_step_size/time_step_size_ref!"
      << std::endl;

    // only makes sense in case of constant time step sizes
    laplace_operator_data.IP_factor =
      this->param.IP_factor_pressure / time_step_size * this->param.deltat_ref;
  }

  laplace_operator_data.bc = this->boundary_descriptor_laplace;

  laplace_operator_data.periodic_face_pairs_level0 = this->periodic_face_pairs;
  laplace_operator.initialize(this->mapping, this->data, laplace_operator_data);

  // setup preconditioner
  if(this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::Jacobi)
  {
    preconditioner_pressure_poisson.reset(
      new JacobiPreconditioner<Poisson::LaplaceOperator<dim, degree_p, Number>>(laplace_operator));
  }
  else if(this->param.preconditioner_pressure_poisson ==
          PreconditionerPressurePoisson::GeometricMultigrid)
  {
    MultigridData mg_data;
    mg_data = this->param.multigrid_data_pressure_poisson;

    // use single precision for multigrid
    typedef float MultigridNumber;

    typedef MyMultigridPreconditionerDG<dim,
                                        Number,
                                        Poisson::LaplaceOperator<dim, degree_p, MultigridNumber>>
      MULTIGRID;

    preconditioner_pressure_poisson.reset(new MULTIGRID());

    std::shared_ptr<MULTIGRID> mg_preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(preconditioner_pressure_poisson);

    mg_preconditioner->initialize(mg_data,
                                  this->dof_handler_p,
                                  this->mapping,
                                  laplace_operator.get_operator_data().bc->dirichlet_bc,
                                  (void *)&laplace_operator.get_operator_data());
  }
  else
  {
    AssertThrow(
      this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::None ||
        this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::Jacobi ||
        this->param.preconditioner_pressure_poisson ==
          PreconditionerPressurePoisson::GeometricMultigrid,
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
       this->param.preconditioner_pressure_poisson ==
         PreconditionerPressurePoisson::GeometricMultigrid)
    {
      solver_data.use_preconditioner = true;
    }

    // setup solver
    pressure_poisson_solver.reset(
      new CGSolver<Poisson::LaplaceOperator<dim, degree_p, Number>,
                   PreconditionerBase<Number>,
                   VectorType>(laplace_operator, *preconditioner_pressure_poisson, solver_data));
  }
  else if(this->param.solver_pressure_poisson == SolverPressurePoisson::FGMRES)
  {
    FGMRESSolverData solver_data;
    // use default value of max_iter
    solver_data.solver_tolerance_abs = this->param.abs_tol_pressure;
    solver_data.solver_tolerance_rel = this->param.rel_tol_pressure;
    solver_data.max_n_tmp_vectors    = this->param.max_n_tmp_vectors_pressure_poisson;
    // use default value of update_preconditioner (=false)

    if(this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::Jacobi ||
       this->param.preconditioner_pressure_poisson ==
         PreconditionerPressurePoisson::GeometricMultigrid)
    {
      solver_data.use_preconditioner = true;
    }

    pressure_poisson_solver.reset(new FGMRESSolver<Poisson::LaplaceOperator<dim, degree_p, Number>,
                                                   PreconditionerBase<Number>,
                                                   VectorType>(laplace_operator,
                                                               *preconditioner_pressure_poisson,
                                                               solver_data));
  }
  else
  {
    AssertThrow(
      this->param.solver_viscous == SolverViscous::PCG ||
        this->param.solver_viscous == SolverViscous::FGMRES,
      ExcMessage(
        "Specified  solver for pressure Poisson equation not implemented - possibilities are PCG and FGMRES"));
  }
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesProjectionMethods<dim, degree_u, degree_p, Number>::evaluate_velocity_divergence_term(
  VectorType &       dst,
  VectorType const & src,
  double const       evaluation_time) const
{
  this->divergence_operator.evaluate(dst, src, evaluation_time);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesProjectionMethods<dim, degree_u, degree_p, Number>::evaluate_pressure_gradient_term(
  VectorType &       dst,
  VectorType const & src,
  double const       evaluation_time) const
{
  this->gradient_operator.evaluate(dst, src, evaluation_time);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesProjectionMethods<dim, degree_u, degree_p, Number>::rhs_add_viscous_term(
  VectorType & dst,
  double const evaluation_time) const
{
  this->viscous_operator.rhs_add(dst, evaluation_time);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesProjectionMethods<dim, degree_u, degree_p, Number>::rhs_ppe_laplace_add(
  VectorType &   dst,
  double const & evaluation_time) const
{
  this->laplace_operator.rhs_add(dst, evaluation_time);
}

template<int dim, int degree_u, int degree_p, typename Number>
unsigned int
DGNavierStokesProjectionMethods<dim, degree_u, degree_p, Number>::solve_pressure(
  VectorType &       dst,
  VectorType const & src) const
{
  //  typedef float MultigridNumber;
  //  typedef MyMultigridPreconditionerLaplace<dim, Number,
  //      LaplaceOperator<dim, degree_p, MultigridNumber>, LaplaceOperatorData<dim> > MULTIGRID;
  //
  //  std::shared_ptr<MULTIGRID> mg_preconditioner
  //    = std::dynamic_pointer_cast<MULTIGRID>(preconditioner_pressure_poisson);
  //
  //  CheckMultigrid<dim,Number,LaplaceOperator<dim,degree_p, Number>,MULTIGRID>
  //    check_multigrid(this->laplace_operator,mg_preconditioner);
  //  check_multigrid.check();

  unsigned int n_iter = this->pressure_poisson_solver->solve(dst, src);

  return n_iter;
}


template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesProjectionMethods<dim, degree_u, degree_p, Number>::apply_laplace_operator(
  VectorType &       dst,
  VectorType const & src) const
{
  this->laplace_operator.vmult(dst, src);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesProjectionMethods<dim, degree_u, degree_p, Number>::apply_projection_operator(
  VectorType &       dst,
  VectorType const & src) const
{
  AssertThrow(this->projection_operator.get() != 0,
              ExcMessage("Projection operator is not initialized correctly."));

  this->projection_operator->vmult(dst, src);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesProjectionMethods<dim, degree_u, degree_p, Number>::
  evaluate_nonlinear_residual_steady(VectorType &       dst_u,
                                     VectorType &       dst_p,
                                     VectorType const & src_u,
                                     VectorType const & src_p,
                                     double const &     evaluation_time)
{
  // velocity-block

  // set dst_u to zero. This is necessary since subsequent operators
  // call functions of type ..._add
  dst_u = 0.0;

  if(this->param.right_hand_side == true)
  {
    this->body_force_operator.evaluate(dst_u, evaluation_time);
    // Shift body force term to the left-hand side of the equation.
    // This works since body_force_operator is the first operator
    // that is evaluated.
    dst_u *= -1.0;
  }

  if(this->param.equation_type == EquationType::NavierStokes)
    this->convective_operator.evaluate_add(dst_u, src_u, evaluation_time);

  this->viscous_operator.evaluate_add(dst_u, src_u, evaluation_time);

  // gradient operator scaled by scaling_factor_continuity
  this->gradient_operator.evaluate_add(dst_u, src_p, evaluation_time);

  // pressure-block

  this->divergence_operator.evaluate(dst_p, src_u, evaluation_time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  dst_p *= -1.0;
}


} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_PROJECTION_METHODS_H_ \
        */
