/*
 * operator.h
 *
 *  Created on: 2016
 *      Author: Fehn/Munch
 */

#ifndef INCLUDE_LAPLACE_DG_LAPLACE_OPERATION_H_
#define INCLUDE_LAPLACE_DG_LAPLACE_OPERATION_H_

// deal.II
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/numerics/vector_tools.h>

// operators
#include "../../convection_diffusion/spatial_discretization/operators/rhs_operator.h"
#include "../../operators/inverse_mass_matrix.h"
#include "laplace_operator.h"

// solvers/preconditioners
#include "../../solvers_and_preconditioners/preconditioner/inverse_mass_matrix_preconditioner.h"
#include "../../solvers_and_preconditioners/preconditioner/jacobi_preconditioner.h"
#include "../../solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h"
#include "../preconditioner/multigrid_preconditioner.h"

// user interface
#include "../../convection_diffusion/user_interface/boundary_descriptor.h"
#include "../user_interface/analytical_solution.h"
#include "../user_interface/field_functions.h"
#include "../user_interface/input_parameters.h"

// functionalities
#include "../../functionalities/matrix_free_wrapper.h"

// postprocessor
#include "../../convection_diffusion/postprocessor/postprocessor_base.h"

namespace Poisson
{
template<int dim, typename Number, int n_components = 1>
class Operator : public dealii::Subscriptor
{
private:
  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : numbers::invalid_unsigned_int);

  typedef float MultigridNumber;
  // use this line for double-precision multigrid
  //  typedef Number MultigridNumber;

  typedef MultigridPreconditioner<dim, Number, MultigridNumber, n_components> Multigrid;
  typedef LaplaceOperator<dim, Number, n_components>                          Laplace;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;
#ifdef DEAL_II_WITH_TRILINOS
  typedef LinearAlgebra::distributed::Vector<double> VectorTypeDouble;
#endif

  typedef std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    PeriodicFaces;

public:
  Operator(parallel::TriangulationBase<dim> const &                       triangulation,
           Mapping<dim> const &                                           mapping,
           PeriodicFaces const                                            periodic_face_pairs,
           std::shared_ptr<ConvDiff::BoundaryDescriptor<rank, dim>> const boundary_descriptor,
           std::shared_ptr<FieldFunctions<dim>> const                     field_functions,
           InputParameters const &                                        param,
           MPI_Comm const &                                               mpi_comm);

  void
  append_data_structures(MatrixFreeWrapper<dim, Number> & matrix_free_wrapper,
                         std::string const &              field) const;

  void
  setup(std::shared_ptr<MatrixFreeWrapper<dim, Number>> matrix_free_wrapper);

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

#ifdef DEAL_II_WITH_TRILINOS
  void
  init_system_matrix(TrilinosWrappers::SparseMatrix & system_matrix) const;

  void
  calculate_system_matrix(TrilinosWrappers::SparseMatrix & system_matrix) const;

  void
  vmult_matrix_based(VectorTypeDouble &                     dst,
                     TrilinosWrappers::SparseMatrix const & system_matrix,
                     VectorTypeDouble const &               src) const;
#endif

private:
  unsigned int
  get_dof_index() const;

  unsigned int
  get_quad_index() const;

  void
  distribute_dofs();

  void
  setup_operators();

  /*
   * Mapping
   */
  Mapping<dim> const & mapping;

  /*
   * Periodic face pairs: This variable is only needed when using a multigrid preconditioner
   */
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_face_pairs;

  /*
   * User interface: Boundary conditions and field functions.
   */
  std::shared_ptr<ConvDiff::BoundaryDescriptor<rank, dim>> boundary_descriptor;
  std::shared_ptr<FieldFunctions<dim>>                     field_functions;

  /*
   * List of input parameters.
   */
  InputParameters const & param;

  /*
   * Basic finite element ingredients.
   */
  std::shared_ptr<FiniteElement<dim>> fe;

  DoFHandler<dim> dof_handler;

  mutable AffineConstraints<double> constraint_matrix;

  std::string const dof_index  = "laplace";
  std::string const quad_index = "laplace";

  mutable std::string field;

  std::shared_ptr<MatrixFreeWrapper<dim, Number>> matrix_free_wrapper;
  std::shared_ptr<MatrixFree<dim, Number>>        matrix_free;

  ConvDiff::RHSOperator<dim, Number, n_components> rhs_operator;

  Laplace laplace_operator;

  std::shared_ptr<PreconditionerBase<Number>>      preconditioner;
  std::shared_ptr<IterativeSolverBase<VectorType>> iterative_solver;

  /*
   * MPI
   */
  MPI_Comm const & mpi_comm;

  /*
   * Output to screen.
   */
  ConditionalOStream pcout;
};
} // namespace Poisson

#endif
