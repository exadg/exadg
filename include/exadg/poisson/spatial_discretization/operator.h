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
  Operator(std::shared_ptr<Grid<dim> const>                      grid,
           std::shared_ptr<dealii::Mapping<dim> const>           mapping,
           std::shared_ptr<MultigridMappings<dim, Number>> const multigrid_mappings,
           std::shared_ptr<BoundaryDescriptor<rank, dim> const>  boundary_descriptor,
           std::shared_ptr<FieldFunctions<dim> const>            field_functions,
           Parameters const &                                    param,
           std::string const &                                   field,
           MPI_Comm const &                                      mpi_comm);

  void
  fill_matrix_free_data(MatrixFreeData<dim, Number> & matrix_free_data) const;

  /**
   * Call this setup() function if the dealii::MatrixFree object can be set up by the present class.
   */
  void
  setup();

  /**
   * Call this setup() function if the dealii::MatrixFree object needs to be created outside this
   * class. The typical use case would be multiphysics-coupling with one MatrixFree object handed
   * over to several single-field solvers.
   */
  void
  setup(std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free,
        std::shared_ptr<MatrixFreeData<dim, Number> const>     matrix_free_data);

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

  void
  evaluate(VectorType & dst, VectorType const & src, double const time = 0.0) const;

  unsigned int
  solve(VectorType & sol, VectorType const & rhs, double const time) const;

  /*
   * Setters and getters.
   */

  std::shared_ptr<dealii::MatrixFree<dim, Number> const>
  get_matrix_free() const;

  dealii::DoFHandler<dim> const &
  get_dof_handler() const;

  dealii::types::global_dof_index
  get_number_of_dofs() const;

  double
  get_n10() const;

  double
  get_average_convergence_rate() const;

  // Multiphysics coupling via "Cached" boundary conditions
  std::shared_ptr<ContainerInterfaceData<rank, dim, double>>
  get_container_interface_data() const;

  std::shared_ptr<TimerTree>
  get_timings() const;

  std::shared_ptr<dealii::Mapping<dim> const>
  get_mapping() const;

  // TODO: we currently need this function public for precice-based FSI
  unsigned int
  get_dof_index() const;

  unsigned int
  get_quad_index() const;

private:
  std::string
  get_dof_name() const;

  unsigned int
  get_dof_index_periodicity_and_hanging_node_constraints() const;

  std::string
  get_dof_name_periodicity_and_hanging_node_constraints() const;

  std::string
  get_quad_name() const;

  std::string
  get_quad_gauss_lobatto_name() const;

  unsigned int
  get_quad_index_gauss_lobatto() const;

  void
  initialize_dof_handler_and_constraints();

  void
  setup_coupling_boundary_conditions();

  void
  setup_operators();

  void
  setup_preconditioner_and_solver();

  /*
   * Grid
   */
  std::shared_ptr<Grid<dim> const> grid;

  /*
   * Mapping
   */
  std::shared_ptr<dealii::Mapping<dim> const> mapping;

  std::shared_ptr<MultigridMappings<dim, Number>> const multigrid_mappings;

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

  // This AffineConstraints object applies homogeneous boundary conditions as needed by vmult()/
  // apply() functions in iterative solvers for linear systems of equations and preconditioners
  // such as multigrid, implemented via dealii::MatrixFree and FEEvaluation::read_dof_values()
  // (or gather_evaluate()).
  // To deal with inhomogeneous boundary data, a separate object of type AffineConstraints is
  // needed (see below).
  mutable dealii::AffineConstraints<Number> affine_constraints;

  // To treat inhomogeneous Dirichlet BCs correctly in the context of matrix-free operator
  // evaluation using dealii::MatrixFree/FEEvaluation, we need a separate AffineConstraints
  // object containing only periodicity and hanging node constraints. This is only relevant
  // for continuous Galerkin discretizations.
  dealii::AffineConstraints<Number> affine_constraints_periodicity_and_hanging_nodes;

  std::string const dof_index = "dof";
  std::string const dof_index_periodicity_and_handing_node_constraints =
    "dof_periodicity_hanging_nodes";

  std::string const quad_index               = "quad";
  std::string const quad_index_gauss_lobatto = "quad_gauss_lobatto";

  std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free;
  std::shared_ptr<MatrixFreeData<dim, Number> const>     matrix_free_data;

  /*
   * Interface coupling
   */
  // TODO: The PDE operator should only have read access to interface data
  mutable std::shared_ptr<ContainerInterfaceData<rank, dim, double>>
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
