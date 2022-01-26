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

#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_projection_methods.h>
#include <exadg/poisson/preconditioners/multigrid_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/utilities/check_multigrid.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim, typename Number>
OperatorProjectionMethods<dim, Number>::OperatorProjectionMethods(
  std::shared_ptr<Grid<dim> const>                  grid_in,
  std::shared_ptr<GridMotionInterface<dim, Number>> grid_motion_in,
  std::shared_ptr<BoundaryDescriptor<dim> const>    boundary_descriptor_in,
  std::shared_ptr<FieldFunctions<dim> const>        field_functions_in,
  Parameters const &                                parameters_in,
  std::string const &                               field_in,
  MPI_Comm const &                                  mpi_comm_in)
  : Base(grid_in,
         grid_motion_in,
         boundary_descriptor_in,
         field_functions_in,
         parameters_in,
         field_in,
         mpi_comm_in)
{
}

template<int dim, typename Number>
OperatorProjectionMethods<dim, Number>::~OperatorProjectionMethods()
{
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::setup(
  std::shared_ptr<MatrixFree<dim, Number>>     matrix_free,
  std::shared_ptr<MatrixFreeData<dim, Number>> matrix_free_data,
  std::string const &                          dof_index_temperature)
{
  Base::setup(matrix_free, matrix_free_data, dof_index_temperature);

  initialize_laplace_operator();
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::update_after_grid_motion()
{
  Base::update_after_grid_motion();

  // update SIPG penalty parameter of Laplace operator which depends on the deformation
  // of elements
  laplace_operator.update_penalty_parameter();
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::setup_pressure_poisson_solver()
{
  initialize_preconditioner_pressure_poisson();

  initialize_solver_pressure_poisson();
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::initialize_laplace_operator()
{
  // setup Laplace operator
  Poisson::LaplaceOperatorData<0, dim> laplace_operator_data;
  laplace_operator_data.dof_index  = this->get_dof_index_pressure();
  laplace_operator_data.quad_index = this->get_quad_index_pressure();

  /*
   * In case no Dirichlet boundary conditions as prescribed for the pressure, the pressure Poisson
   * equation is singular (i.e. vectors describing a constant pressure state form the nullspace
   * of the discrete pressure Poisson operator). To solve the pressure Poisson equation in that
   * case, a Krylov subspace projection is applied during the solution of the linear system of
   * equations.
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
    laplace_operator_data.operator_is_singular = this->is_pressure_level_undefined();
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
    laplace_operator_data.operator_is_singular = this->is_pressure_level_undefined();
  }

  laplace_operator_data.bc                   = this->boundary_descriptor_laplace;
  laplace_operator_data.use_cell_based_loops = this->param.use_cell_based_face_loops;
  laplace_operator_data.implement_block_diagonal_preconditioner_matrix_free =
    this->param.implement_block_diagonal_preconditioner_matrix_free;

  laplace_operator_data.kernel_data.IP_factor = this->param.IP_factor_pressure;

  laplace_operator.initialize(this->get_matrix_free(),
                              this->get_constraint_p(),
                              laplace_operator_data);
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::initialize_preconditioner_pressure_poisson()
{
  // setup preconditioner
  if(this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::None)
  {
    // do nothing, preconditioner will not be used
  }
  else if(this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::PointJacobi)
  {
    preconditioner_pressure_poisson =
      std::make_shared<JacobiPreconditioner<Poisson::LaplaceOperator<dim, Number, 1>>>(
        laplace_operator);
  }
  else if(this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::Multigrid)
  {
    MultigridData mg_data;
    mg_data = this->param.multigrid_data_pressure_poisson;

    typedef Poisson::MultigridPreconditioner<dim, Number, 1> Multigrid;

    preconditioner_pressure_poisson = std::make_shared<Multigrid>(this->mpi_comm);

    std::shared_ptr<Multigrid> mg_preconditioner =
      std::dynamic_pointer_cast<Multigrid>(preconditioner_pressure_poisson);

    auto & dof_handler = this->get_dof_handler_p();
    mg_preconditioner->initialize(mg_data,
                                  &dof_handler.get_triangulation(),
                                  dof_handler.get_fe(),
                                  this->get_mapping(),
                                  laplace_operator.get_data(),
                                  this->param.ale_formulation,
                                  &laplace_operator.get_data().bc->dirichlet_bc,
                                  &this->grid->periodic_faces);
  }
  else
  {
    AssertThrow(
      false, ExcMessage("Specified preconditioner for pressure Poisson equation not implemented"));
  }
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::initialize_solver_pressure_poisson()
{
  if(this->param.solver_pressure_poisson == SolverPressurePoisson::CG)
  {
    // setup solver data
    Krylov::SolverDataCG solver_data;
    solver_data.max_iter             = this->param.solver_data_pressure_poisson.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_pressure_poisson.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_pressure_poisson.rel_tol;
    // use default value of update_preconditioner (=false)

    if(this->param.preconditioner_pressure_poisson != PreconditionerPressurePoisson::None)
    {
      solver_data.use_preconditioner = true;
    }

    // setup solver
    pressure_poisson_solver =
      std::make_shared<Krylov::SolverCG<Poisson::LaplaceOperator<dim, Number, 1>,
                                        PreconditionerBase<Number>,
                                        VectorType>>(laplace_operator,
                                                     *preconditioner_pressure_poisson,
                                                     solver_data);
  }
  else if(this->param.solver_pressure_poisson == SolverPressurePoisson::FGMRES)
  {
    Krylov::SolverDataFGMRES solver_data;
    solver_data.max_iter             = this->param.solver_data_pressure_poisson.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_pressure_poisson.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_pressure_poisson.rel_tol;
    solver_data.max_n_tmp_vectors    = this->param.solver_data_pressure_poisson.max_krylov_size;
    // use default value of update_preconditioner (=false)

    if(this->param.preconditioner_pressure_poisson != PreconditionerPressurePoisson::None)
    {
      solver_data.use_preconditioner = true;
    }

    pressure_poisson_solver =
      std::make_shared<Krylov::SolverFGMRES<Poisson::LaplaceOperator<dim, Number, 1>,
                                            PreconditionerBase<Number>,
                                            VectorType>>(laplace_operator,
                                                         *preconditioner_pressure_poisson,
                                                         solver_data);
  }
  else
  {
    AssertThrow(false,
                ExcMessage("Specified solver for pressure Poisson equation is not implemented."));
  }
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::do_rhs_add_viscous_term(VectorType & dst,
                                                                double const time) const
{
  this->viscous_operator.set_time(time);
  this->viscous_operator.rhs_add(dst);
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::do_rhs_ppe_laplace_add(VectorType &   dst,
                                                               double const & time) const
{
  this->laplace_operator.set_time(time);
  this->laplace_operator.rhs_add(dst);
}

template<int dim, typename Number>
unsigned int
OperatorProjectionMethods<dim, Number>::do_solve_pressure(VectorType &       dst,
                                                          VectorType const & src,
                                                          bool const update_preconditioner) const
{
  // Check multigrid algorithm
  //  std::shared_ptr<MultigridPoisson> mg_preconditioner
  //    = std::dynamic_pointer_cast<MultigridPoisson>(preconditioner_pressure_poisson);
  //
  //  CheckMultigrid<dim, Number, Poisson::LaplaceOperator<dim, Number>, MultigridPoisson,
  //  MultigridNumber>
  //    check_multigrid(this->laplace_operator, mg_preconditioner, this->mpi_comm);
  //  check_multigrid.check();

  // Use multigrid as a solver (use double precision here)
  //  std::shared_ptr<MultigridPoisson> mg_preconditioner
  //    = std::dynamic_pointer_cast<MultigridPoisson>(preconditioner_pressure_poisson);
  //  unsigned int n_iter = mg_preconditioner->solve(dst,src);

  // call pressure Poisson solver
  unsigned int n_iter = this->pressure_poisson_solver->solve(dst, src, update_preconditioner);

  return n_iter;
}


template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::apply_laplace_operator(VectorType &       dst,
                                                               VectorType const & src) const
{
  this->laplace_operator.vmult(dst, src);
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::apply_projection_operator(VectorType &       dst,
                                                                  VectorType const & src) const
{
  AssertThrow(this->projection_operator.get() != 0,
              ExcMessage("Projection operator is not initialized correctly."));

  this->projection_operator->vmult(dst, src);
}

template class OperatorProjectionMethods<2, float>;
template class OperatorProjectionMethods<2, double>;

template class OperatorProjectionMethods<3, float>;
template class OperatorProjectionMethods<3, double>;

} // namespace IncNS
} // namespace ExaDG
