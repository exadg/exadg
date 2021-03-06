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
#include <exadg/grid/grid.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/operators/rhs_operator.h>
#include <exadg/poisson/spatial_discretization/laplace_operator.h>
#include <exadg/poisson/user_interface/analytical_solution.h>
#include <exadg/poisson/user_interface/boundary_descriptor.h>
#include <exadg/poisson/user_interface/field_functions.h>
#include <exadg/poisson/user_interface/input_parameters.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_base.h>

namespace ExaDG
{
namespace Poisson
{
using namespace dealii;

template<int dim, typename Number, int n_components = 1>
class Operator : public dealii::Subscriptor
{
private:
  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : numbers::invalid_unsigned_int);

  typedef LaplaceOperator<dim, Number, n_components> Laplace;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;
#ifdef DEAL_II_WITH_TRILINOS
  typedef LinearAlgebra::distributed::Vector<double> VectorTypeDouble;
#endif

public:
  Operator(std::shared_ptr<Grid<dim, Number> const>             grid,
           unsigned int const                                   degree,
           std::shared_ptr<BoundaryDescriptor<rank, dim> const> boundary_descriptor,
           std::shared_ptr<FieldFunctions<dim> const>           field_functions,
           InputParameters const &                              param,
           std::string const &                                  field,
           MPI_Comm const &                                     mpi_comm);

  void
  fill_matrix_free_data(MatrixFreeData<dim, Number> & matrix_free_data) const;

  void
  setup(std::shared_ptr<MatrixFree<dim, Number>>     matrix_free,
        std::shared_ptr<MatrixFreeData<dim, Number>> matrix_free_data);

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

  DoFHandler<dim> const &
  get_dof_handler() const;

  types::global_dof_index
  get_number_of_dofs() const;

  double
  get_n10() const;

  double
  get_average_convergence_rate() const;

  unsigned int
  get_degree() const;

  unsigned int
  get_dof_index() const;

  unsigned int
  get_quad_index() const;

  unsigned int
  get_quad_index_gauss_lobatto() const;

  std::shared_ptr<TimerTree>
  get_timings() const;

#ifdef DEAL_II_WITH_TRILINOS
  void
  init_system_matrix(TrilinosWrappers::SparseMatrix & system_matrix,
                     MPI_Comm const &                 mpi_comm) const;

  void
  calculate_system_matrix(TrilinosWrappers::SparseMatrix & system_matrix) const;

  void
  vmult_matrix_based(VectorTypeDouble &                     dst,
                     TrilinosWrappers::SparseMatrix const & system_matrix,
                     VectorTypeDouble const &               src) const;
#endif

#ifdef DEAL_II_WITH_PETSC
  void
  init_system_matrix(PETScWrappers::MPI::SparseMatrix & system_matrix,
                     MPI_Comm const &                   mpi_comm) const;

  void
  calculate_system_matrix(PETScWrappers::MPI::SparseMatrix & system_matrix) const;

  void
  vmult_matrix_based(VectorTypeDouble &                       dst,
                     PETScWrappers::MPI::SparseMatrix const & system_matrix,
                     VectorTypeDouble const &                 src) const;
#endif

private:
  std::string
  get_dof_name() const;

  std::string
  get_quad_name() const;

  std::string
  get_quad_gauss_lobatto_name() const;

  void
  distribute_dofs();

  void
  setup_operators();

  /*
   * Grid
   */
  std::shared_ptr<Grid<dim, Number> const> grid;

  /*
   * Polynomial degree
   */
  unsigned int const degree;

  /*
   * User interface: Boundary conditions and field functions.
   */
  std::shared_ptr<BoundaryDescriptor<rank, dim> const> boundary_descriptor;
  std::shared_ptr<FieldFunctions<dim> const>           field_functions;

  /*
   * List of input parameters.
   */
  InputParameters const & param;

  std::string const field;

  /*
   * Basic finite element ingredients.
   */
  std::shared_ptr<FiniteElement<dim>> fe;

  DoFHandler<dim> dof_handler;

  mutable AffineConstraints<Number> affine_constraints;

  std::string const dof_index                = "laplace";
  std::string const quad_index               = "laplace";
  std::string const quad_index_gauss_lobatto = "laplace_gauss_lobatto";

  std::shared_ptr<MatrixFree<dim, Number>>     matrix_free;
  std::shared_ptr<MatrixFreeData<dim, Number>> matrix_free_data;

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
  ConditionalOStream pcout;
};
} // namespace Poisson
} // namespace ExaDG

#endif
