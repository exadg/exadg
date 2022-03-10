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

#ifndef INCLUDE_LAPLACE_DG_LAPLACE_OPERATION_H_
#define INCLUDE_LAPLACE_DG_LAPLACE_OPERATION_H_

// ExaDG
#include <exadg/functions_and_boundary_conditions/interface_coupling.h>
#include <exadg/grid/grid.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/operators/rhs_operator.h>
#include <exadg/poisson/spatial_discretization/laplace_operator.h>
#include <exadg/poisson/user_interface/analytical_solution.h>
#include <exadg/poisson/user_interface/boundary_descriptor.h>
#include <exadg/poisson/user_interface/field_functions.h>
#include <exadg/poisson/user_interface/parameters.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_base.h>

namespace ExaDG
{
namespace Poisson
{
template<int dim, int n_components, typename Number>
class Operator : public dealii::Subscriptor
{
private:
  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : dealii::numbers::invalid_unsigned_int);

  typedef LaplaceOperator<dim, Number, n_components> Laplace;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef dealii::LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

public:
  Operator(std::shared_ptr<Grid<dim> const>                     grid,
           std::shared_ptr<BoundaryDescriptor<rank, dim> const> boundary_descriptor,
           std::shared_ptr<FieldFunctions<dim> const>           field_functions,
           Parameters const &                                   param,
           std::string const &                                  field,
           MPI_Comm const &                                     mpi_comm);

  void
  fill_matrix_free_data(MatrixFreeData<dim, Number> & matrix_free_data) const;

  void
  setup(std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free,
        std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data);

  void
  setup_solver();

  void
  initialize_dof_vector(VectorType & src) const;

  /*
   * Prescribe initial conditions using a specified analytical function.
   */
  void
  prescribe_initial_conditions(VectorType & src) const;

  void
  rhs(VectorType & dst, double const time = 0.0) const;

  void
  vmult(VectorType & dst, VectorType const & src) const;

  unsigned int
  solve(VectorType & sol, VectorType const & rhs, double const time) const;

  dealii::DoFHandler<dim> const &
  get_dof_handler() const;

  dealii::types::global_dof_index
  get_number_of_dofs() const;

  double
  get_n10() const;

  double
  get_average_convergence_rate() const;

  // Multiphysics coupling via "Cached" boundary conditions
  std::shared_ptr<ContainerInterfaceData<dim, n_components, Number>>
  get_container_interface_data();

  std::shared_ptr<TimerTree>
  get_timings() const;

#ifdef DEAL_II_WITH_TRILINOS
  void
  init_system_matrix(dealii::TrilinosWrappers::SparseMatrix & system_matrix,
                     MPI_Comm const &                         mpi_comm) const;

  void
  calculate_system_matrix(dealii::TrilinosWrappers::SparseMatrix & system_matrix) const;

  void
  vmult_matrix_based(VectorTypeDouble &                             dst,
                     dealii::TrilinosWrappers::SparseMatrix const & system_matrix,
                     VectorTypeDouble const &                       src) const;
#endif

#ifdef DEAL_II_WITH_PETSC
  void
  init_system_matrix(dealii::PETScWrappers::MPI::SparseMatrix & system_matrix,
                     MPI_Comm const &                           mpi_comm) const;

  void
  calculate_system_matrix(dealii::PETScWrappers::MPI::SparseMatrix & system_matrix) const;

  void
  vmult_matrix_based(VectorTypeDouble &                               dst,
                     dealii::PETScWrappers::MPI::SparseMatrix const & system_matrix,
                     VectorTypeDouble const &                         src) const;
#endif

private:
  std::string
  get_dof_name() const;

  std::string
  get_quad_name() const;

  std::string
  get_quad_gauss_lobatto_name() const;

  unsigned int
  get_dof_index() const;

  unsigned int
  get_quad_index() const;

  unsigned int
  get_quad_index_gauss_lobatto() const;

  void
  distribute_dofs();

  void
  setup_operators();

  /*
   * Grid
   */
  std::shared_ptr<Grid<dim> const> grid;

  /*
   * User interface: Boundary conditions and field functions.
   */
  std::shared_ptr<BoundaryDescriptor<rank, dim> const> boundary_descriptor;
  std::shared_ptr<FieldFunctions<dim> const>           field_functions;

  /*
   * List of parameters.
   */
  Parameters const & param;

  std::string const field;

  /*
   * Basic finite element ingredients.
   */
  std::shared_ptr<dealii::FiniteElement<dim>> fe;

  dealii::DoFHandler<dim> dof_handler;

  mutable dealii::AffineConstraints<Number> affine_constraints;

  std::string const dof_index                = "laplace";
  std::string const quad_index               = "laplace";
  std::string const quad_index_gauss_lobatto = "laplace_gauss_lobatto";

  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free;
  std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data;

  /*
   * Interface coupling
   */
  std::shared_ptr<ContainerInterfaceData<dim, n_components, Number>>
    interface_data_dirichlet_cached;

  RHSOperator<dim, Number, n_components> rhs_operator;

  Laplace laplace_operator;

  std::shared_ptr<PreconditionerBase<Number>>     preconditioner;
  std::shared_ptr<Krylov::SolverBase<VectorType>> iterative_solver;

  /*
   * MPI
   */
  MPI_Comm const mpi_comm;

  /*
   * Output to screen.
   */
  dealii::ConditionalOStream pcout;
};
} // namespace Poisson
} // namespace ExaDG

#endif
