/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#include <exadg/incompressible_navier_stokes/preconditioners/multigrid_preconditioner_momentum.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_pressure_correction.h>
#include <exadg/solvers_and_preconditioners/preconditioners/block_jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/inverse_mass_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim, typename Number>
OperatorPressureCorrection<dim, Number>::OperatorPressureCorrection(
  std::shared_ptr<Grid<dim, Number> const>       grid_in,
  unsigned int const                             degree_u_in,
  std::shared_ptr<BoundaryDescriptor<dim> const> boundary_descriptor_in,
  std::shared_ptr<FieldFunctions<dim> const>     field_functions_in,
  InputParameters const &                        parameters_in,
  std::string const &                            field_in,
  MPI_Comm const &                               mpi_comm_in)
  : ProjectionBase(grid_in,
                   degree_u_in,
                   boundary_descriptor_in,
                   field_functions_in,
                   parameters_in,
                   field_in,
                   mpi_comm_in)
{
}

template<int dim, typename Number>
OperatorPressureCorrection<dim, Number>::~OperatorPressureCorrection()
{
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::setup(
  std::shared_ptr<MatrixFree<dim, Number>>     matrix_free,
  std::shared_ptr<MatrixFreeData<dim, Number>> matrix_free_data,
  std::string const &                          dof_index_temperature)
{
  ProjectionBase::setup(matrix_free, matrix_free_data, dof_index_temperature);

  setup_inverse_mass_operator_pressure();
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::setup_solvers(double const &     scaling_factor_mass,
                                                       VectorType const & velocity)
{
  this->pcout << std::endl << "Setup incompressible Navier-Stokes solver ..." << std::endl;

  ProjectionBase::setup_solvers(scaling_factor_mass, velocity);

  setup_momentum_solver();

  ProjectionBase::setup_pressure_poisson_solver();

  ProjectionBase::setup_projection_solver();

  this->pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::setup_momentum_solver()
{
  initialize_momentum_preconditioner();

  initialize_momentum_solver();
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::initialize_momentum_preconditioner()
{
  if(this->param.preconditioner_momentum == MomentumPreconditioner::InverseMassMatrix)
  {
    momentum_preconditioner.reset(
      new InverseMassPreconditioner<dim, dim, Number>(this->get_matrix_free(),
                                                      this->get_dof_index_velocity(),
                                                      this->get_quad_index_velocity_linear()));
  }
  else if(this->param.preconditioner_momentum == MomentumPreconditioner::PointJacobi)
  {
    momentum_preconditioner.reset(
      new JacobiPreconditioner<MomentumOperator<dim, Number>>(this->momentum_operator));
  }
  else if(this->param.preconditioner_momentum == MomentumPreconditioner::BlockJacobi)
  {
    momentum_preconditioner.reset(
      new BlockJacobiPreconditioner<MomentumOperator<dim, Number>>(this->momentum_operator));
  }
  else if(this->param.preconditioner_momentum == MomentumPreconditioner::Multigrid)
  {
    typedef MultigridPreconditioner<dim, Number> Multigrid;

    momentum_preconditioner.reset(new Multigrid(this->mpi_comm));

    std::shared_ptr<Multigrid> mg_preconditioner =
      std::dynamic_pointer_cast<Multigrid>(momentum_preconditioner);

    auto & dof_handler = this->get_dof_handler_u();
    mg_preconditioner->initialize(this->param.multigrid_data_momentum,
                                  &dof_handler.get_triangulation(),
                                  dof_handler.get_fe(),
                                  this->get_mapping(),
                                  this->momentum_operator,
                                  this->param.multigrid_operator_type_momentum,
                                  this->param.ale_formulation,
                                  &this->momentum_operator.get_data().bc->dirichlet_bc,
                                  &this->grid->periodic_faces);
  }
  else
  {
    AssertThrow(this->param.preconditioner_momentum == MomentumPreconditioner::None,
                ExcNotImplemented());
  }
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::initialize_momentum_solver()
{
  if(this->param.solver_momentum == SolverMomentum::CG)
  {
    // setup solver data
    Krylov::SolverDataCG solver_data;
    solver_data.max_iter             = this->param.solver_data_momentum.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_momentum.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_momentum.rel_tol;
    if(this->param.preconditioner_momentum != MomentumPreconditioner::None)
      solver_data.use_preconditioner = true;

    // setup solver
    momentum_linear_solver.reset(
      new Krylov::SolverCG<MomentumOperator<dim, Number>, PreconditionerBase<Number>, VectorType>(
        this->momentum_operator, *momentum_preconditioner, solver_data));
  }
  else if(this->param.solver_momentum == SolverMomentum::GMRES)
  {
    // setup solver data
    Krylov::SolverDataGMRES solver_data;
    solver_data.max_iter             = this->param.solver_data_momentum.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_momentum.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_momentum.rel_tol;
    solver_data.max_n_tmp_vectors    = this->param.solver_data_momentum.max_krylov_size;
    solver_data.compute_eigenvalues  = false;
    if(this->param.preconditioner_momentum != MomentumPreconditioner::None)
      solver_data.use_preconditioner = true;

    // setup solver
    momentum_linear_solver.reset(new Krylov::SolverGMRES<MomentumOperator<dim, Number>,
                                                         PreconditionerBase<Number>,
                                                         VectorType>(
      this->momentum_operator, *momentum_preconditioner, solver_data, this->mpi_comm));
  }
  else if(this->param.solver_momentum == SolverMomentum::FGMRES)
  {
    Krylov::SolverDataFGMRES solver_data;
    solver_data.max_iter             = this->param.solver_data_momentum.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_momentum.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_momentum.rel_tol;
    solver_data.max_n_tmp_vectors    = this->param.solver_data_momentum.max_krylov_size;
    if(this->param.preconditioner_momentum != MomentumPreconditioner::None)
      solver_data.use_preconditioner = true;

    momentum_linear_solver.reset(new Krylov::SolverFGMRES<MomentumOperator<dim, Number>,
                                                          PreconditionerBase<Number>,
                                                          VectorType>(this->momentum_operator,
                                                                      *momentum_preconditioner,
                                                                      solver_data));
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified solver for momentum equation is not implemented."));
  }


  // Navier-Stokes equations with an implicit treatment of the convective term
  if(this->param.nonlinear_problem_has_to_be_solved())
  {
    // nonlinear_operator;
    nonlinear_operator.initialize(*this);

    // setup Newton solver
    momentum_newton_solver.reset(
      new Newton::Solver<VectorType,
                         NonlinearMomentumOperator<dim, Number>,
                         MomentumOperator<dim, Number>,
                         Krylov::SolverBase<VectorType>>(this->param.newton_solver_data_momentum,
                                                         nonlinear_operator,
                                                         this->momentum_operator,
                                                         *momentum_linear_solver));
  }
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::setup_inverse_mass_operator_pressure()
{
  // inverse mass operator pressure (needed for pressure update in case of rotational
  // formulation)
  inverse_mass_pressure.initialize(this->get_matrix_free(),
                                   this->get_dof_index_pressure(),
                                   this->get_quad_index_pressure());
}

template<int dim, typename Number>
unsigned int
OperatorPressureCorrection<dim, Number>::solve_linear_momentum_equation(
  VectorType &       solution,
  VectorType const & rhs,
  bool const &       update_preconditioner,
  double const &     scaling_factor_mass)
{
  this->momentum_operator.set_scaling_factor_mass_operator(scaling_factor_mass);

  // Note that there is no need to set the evaluation time for the momentum_operator
  // in this because because this function is only called if the convective term is not considered
  // in the momentum_operator (Stokes eq. or explicit treatment of convective term).

  auto linear_iterations = momentum_linear_solver->solve(solution, rhs, update_preconditioner);

  return linear_iterations;
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::rhs_add_viscous_term(VectorType & dst,
                                                              double const time) const
{
  ProjectionBase::do_rhs_add_viscous_term(dst, time);
}

template<int dim, typename Number>
std::tuple<unsigned int, unsigned int>
OperatorPressureCorrection<dim, Number>::solve_nonlinear_momentum_equation(
  VectorType &       dst,
  VectorType const & rhs_vector,
  double const &     time,
  bool const &       update_preconditioner,
  double const &     scaling_factor_mass)
{
  // update nonlinear operator
  nonlinear_operator.update(rhs_vector, time, scaling_factor_mass);

  // Set time and scaling_factor_mass for linear operator
  this->momentum_operator.set_time(time);
  this->momentum_operator.set_scaling_factor_mass_operator(scaling_factor_mass);

  // Solve nonlinear problem
  Newton::UpdateData update;
  update.do_update             = update_preconditioner;
  update.threshold_newton_iter = this->param.update_preconditioner_momentum_every_newton_iter;

  std::tuple<unsigned int, unsigned int> iter = momentum_newton_solver->solve(dst, update);

  return iter;
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::evaluate_nonlinear_residual(
  VectorType &       dst,
  VectorType const & src,
  VectorType const * rhs_vector,
  double const &     time,
  double const &     scaling_factor_mass) const
{
  this->mass_operator.apply_scale(dst, scaling_factor_mass, src);

  // always evaluate convective term since this function is only called
  // if a nonlinear problem has to be solved, i.e., if the convective operator
  // has to be considered
  this->convective_operator.evaluate_nonlinear_operator_add(dst, src, time);

  // viscous term
  this->viscous_operator.set_time(time);
  this->viscous_operator.evaluate_add(dst, src);

  // rhs vector
  dst.add(-1.0, *rhs_vector);
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::evaluate_nonlinear_residual_steady(
  VectorType &       dst_u,
  VectorType &       dst_p,
  VectorType const & src_u,
  VectorType const & src_p,
  double const &     time) const
{
  // velocity-block

  // set dst_u to zero. This is necessary since subsequent operators
  // call functions of type ..._add
  dst_u = 0.0;

  if(this->param.right_hand_side == true)
  {
    this->rhs_operator.evaluate(dst_u, time);
    // Shift body force term to the left-hand side of the equation.
    // This works since body_force_operator is the first operator
    // that is evaluated.
    dst_u *= -1.0;
  }

  if(this->param.convective_problem())
    this->convective_operator.evaluate_nonlinear_operator_add(dst_u, src_u, time);

  if(this->param.viscous_problem())
  {
    this->viscous_operator.set_time(time);
    this->viscous_operator.evaluate_add(dst_u, src_u);
  }

  // gradient operator scaled by scaling_factor_continuity
  this->gradient_operator.evaluate_add(dst_u, src_p, time);

  // pressure-block

  this->divergence_operator.evaluate(dst_p, src_u, time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  dst_p *= -1.0;
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::apply_momentum_operator(VectorType &       dst,
                                                                 VectorType const & src)
{
  this->momentum_operator.apply(dst, src);
}


template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::rhs_pressure_gradient_term(VectorType & dst,
                                                                    double const time) const
{
  this->gradient_operator.rhs(dst, time);
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::rhs_pressure_gradient_term_dirichlet_bc_from_dof_vector(
  VectorType &       dst,
  VectorType const & pressure) const
{
  this->gradient_operator.rhs_bc_from_dof_vector(dst, pressure);
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::
  evaluate_pressure_gradient_term_dirichlet_bc_from_dof_vector(VectorType &       dst,
                                                               VectorType const & src,
                                                               VectorType const & pressure) const
{
  this->gradient_operator.evaluate_bc_from_dof_vector(dst, src, pressure);
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::apply_inverse_pressure_mass_operator(
  VectorType &       dst,
  VectorType const & src) const
{
  inverse_mass_pressure.apply(dst, src);
}

template<int dim, typename Number>
unsigned int
OperatorPressureCorrection<dim, Number>::solve_pressure(VectorType &       dst,
                                                        VectorType const & src,
                                                        bool const update_preconditioner) const
{
  return ProjectionBase::do_solve_pressure(dst, src, update_preconditioner);
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::rhs_ppe_laplace_add(VectorType &   dst,
                                                             double const & time) const
{
  ProjectionBase::do_rhs_ppe_laplace_add(dst, time);
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::rhs_ppe_laplace_add_dirichlet_bc_from_dof_vector(
  VectorType &       dst,
  VectorType const & src) const
{
  this->laplace_operator.rhs_add_dirichlet_bc_from_dof_vector(dst, src);
}

template class OperatorPressureCorrection<2, float>;
template class OperatorPressureCorrection<2, double>;

template class OperatorPressureCorrection<3, float>;
template class OperatorPressureCorrection<3, double>;

} // namespace IncNS
} // namespace ExaDG
