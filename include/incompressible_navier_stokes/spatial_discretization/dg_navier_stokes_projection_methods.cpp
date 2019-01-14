/*
 * dg_navier_stokes_projection_methods.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include "dg_navier_stokes_projection_methods.h"

namespace IncNS
{
template<int dim, int degree_u, int degree_p, typename Number>
DGNavierStokesProjectionMethods<dim, degree_u, degree_p, Number>::DGNavierStokesProjectionMethods(
  parallel::distributed::Triangulation<dim> const & triangulation,
  InputParameters<dim> const &                      parameters_in,
  std::shared_ptr<Postprocessor>                    postprocessor_in)
  : BASE(triangulation, parameters_in, postprocessor_in)
{
  AssertThrow(degree_p > 0,
              ExcMessage("Polynomial degree of pressure shape functions has to be larger than "
                         "zero for dual splitting scheme and pressure-correction scheme."));
}

template<int dim, int degree_u, int degree_p, typename Number>
DGNavierStokesProjectionMethods<dim, degree_u, degree_p, Number>::~DGNavierStokesProjectionMethods()
{
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesProjectionMethods<dim, degree_u, degree_p, Number>::setup_pressure_poisson_solver()
{
  initialize_laplace_operator();

  initialize_preconditioner_pressure_poisson();

  initialize_solver_pressure_poisson();
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesProjectionMethods<dim, degree_u, degree_p, Number>::initialize_laplace_operator()
{
  // setup Laplace operator
  Poisson::LaplaceOperatorData<dim> laplace_operator_data;
  laplace_operator_data.dof_index  = this->get_dof_index_pressure();
  laplace_operator_data.quad_index = this->get_quad_index_pressure();
  laplace_operator_data.IP_factor  = this->param.IP_factor_pressure;

  /*
   * In case of pure Dirichlet boundary conditions for the velocity (or more precisely pure Neumann
   * boundary conditions for the pressure), the pressure Poisson equation is singular (i.e. vectors
   * describing a constant pressure state form the nullspace of the discrete pressure Poisson
   * operator). To solve the pressure Poisson equation in that case, a Krylov subspace projection is
   * applied during the solution of the linear system of equations.
   *
   * Strictly speaking, this projection is only needed if the linear system of equations
   *
   *  A * p = rhs (with nullspace spanned by vector p0)
   *
   * is not consistent, meaning that
   *
   *  p0^T * rhs != 0 ,
   *
   *  so that
   *
   *  p0^T * A * p = (A^T * p0)^T * p = 0 != p0 * rhs (since A is symmetric).
   */
  if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    // the Krylov subspace projection is needed for the dual splitting scheme since the linear
    // system of equations is not consistent for this splitting method (due to the boundary
    // conditions).
    laplace_operator_data.operator_is_singular = this->param.pure_dirichlet_bc;
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    // One can show that the linear system of equations of the pressure Poisson equation is
    // consistent in case of the pressure-correction scheme if the velocity Dirichlet BC is
    // consistent. So there should be no need to solve a transformed linear system of equations.
    //    laplace_operator_data.operator_is_singular = false;

    // In principle, it works (since the linear system of equations is consistent)
    // but we detected no convergence for some test cases and specific parameters.
    // Hence, for reasons of robustness we also solve a transformed linear system of equations
    // in case of the pressure-correction scheme.
    laplace_operator_data.operator_is_singular = this->param.pure_dirichlet_bc;
  }

  laplace_operator_data.bc                   = this->boundary_descriptor_laplace;
  laplace_operator_data.use_cell_based_loops = this->param.use_cell_based_face_loops;
  laplace_operator_data.implement_block_diagonal_preconditioner_matrix_free =
    this->param.implement_block_diagonal_preconditioner_matrix_free;

  laplace_operator.reinit(this->mapping, this->data, this->constraint_p, laplace_operator_data);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesProjectionMethods<dim, degree_u, degree_p, Number>::
  initialize_preconditioner_pressure_poisson()
{
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

    typedef PreconditionableOperator<dim, MultigridNumber>           MG_BASE;
    typedef Poisson::LaplaceOperator<dim, degree_p, MultigridNumber> MG_OPERATOR;

    typedef MultigridPreconditionerBase<dim, Number, MultigridNumber> MULTIGRID;

    preconditioner_pressure_poisson.reset(new MULTIGRID(std::shared_ptr<MG_BASE>(new MG_OPERATOR)));

    std::shared_ptr<MULTIGRID> mg_preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(preconditioner_pressure_poisson);

    mg_preconditioner->initialize(mg_data,
                                  this->dof_handler_p,
                                  this->mapping,
                                  laplace_operator.get_operator_data(),
                                  &laplace_operator.get_operator_data().bc->dirichlet_bc,
                                  &this->periodic_face_pairs);
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
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesProjectionMethods<dim, degree_u, degree_p, Number>::
  initialize_solver_pressure_poisson()
{
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
        "Specified solver for pressure Poisson equation not implemented - possibilities are PCG and FGMRES"));
  }
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesProjectionMethods<dim, degree_u, degree_p, Number>::do_rhs_add_viscous_term(
  VectorType & dst,
  double const evaluation_time) const
{
  this->viscous_operator.rhs_add(dst, evaluation_time);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
DGNavierStokesProjectionMethods<dim, degree_u, degree_p, Number>::do_rhs_ppe_laplace_add(
  VectorType &   dst,
  double const & evaluation_time) const
{
  this->laplace_operator.rhs_add(dst, evaluation_time);
}

template<int dim, int degree_u, int degree_p, typename Number>
unsigned int
DGNavierStokesProjectionMethods<dim, degree_u, degree_p, Number>::do_solve_pressure(
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

} // namespace IncNS

#include "dg_navier_stokes_projection_methods.hpp"
