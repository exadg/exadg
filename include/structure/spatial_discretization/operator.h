/*
 * operator.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_
#define INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_

// deal.II
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/vector_tools.h>

// user interface
#include "../user_interface/boundary_descriptor.h"
#include "../user_interface/field_functions.h"
#include "../user_interface/input_parameters.h"

// operators
#include "../../structure/spatial_discretization/operators/linear_operator.h"
#include "../../structure/spatial_discretization/operators/nonlinear_operator.h"
#include "../../structure/spatial_discretization/operators/rhs_operator.h"

// solvers
#include "../../solvers_and_preconditioners/newton/newton_solver.h"
#include "../../solvers_and_preconditioners/preconditioner/preconditioner_base.h"
#include "../../solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h"

using namespace dealii;

namespace Structure
{
// forward declaration
template<int dim, typename Number>
class Operator;

template<int dim, typename Number>
class ResidualOperator
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef Operator<dim, Number> PDEOperator;

public:
  ResidualOperator() : pde_operator(nullptr), rhs_vector(nullptr), time(0.0)
  {
  }

  void
  initialize(PDEOperator const & pde_operator)
  {
    this->pde_operator = &pde_operator;
  }

  void
  update(VectorType const & rhs_vector, double const & time)
  {
    this->rhs_vector = &rhs_vector;
    this->time       = time;
  }

  /*
   * The implementation of the Newton solver requires a function called
   * 'initialize_vector_for_newton_solver'.
   */
  void
  initialize_vector_for_newton_solver(VectorType & src) const
  {
    pde_operator->initialize_dof_vector(src);
  }

  /*
   * The implementation of the Newton solver requires a function called
   * 'evaluate_nonlinear_residual'.
   */
  void
  evaluate_nonlinear_residual(VectorType & dst, VectorType const & src) const
  {
    pde_operator->evaluate_nonlinear_residual(dst, src, *rhs_vector, time);
  }

private:
  PDEOperator const * pde_operator;

  VectorType const * rhs_vector;
  double             time;
};

template<int dim, typename Number>
class LinearizedOperator : public dealii::Subscriptor
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef Operator<dim, Number> PDEOperator;

public:
  LinearizedOperator() : dealii::Subscriptor(), pde_operator(nullptr), time(0.0)
  {
  }

  void
  initialize(PDEOperator const & pde_operator)
  {
    this->pde_operator = &pde_operator;
  }

  /*
   * The implementation of the Newton solver requires a function called
   * 'set_solution_linearization'.
   */
  void
  set_solution_linearization(VectorType const & solution_linearization) const
  {
    pde_operator->set_solution_linearization(solution_linearization);
  }

  void
  update(double const & time)
  {
    this->time = time;
  }

  /*
   * The implementation of linear solvers in deal.ii requires that a function called 'vmult' is
   * provided.
   */
  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    pde_operator->apply_linearized_operator(dst, src, time);
  }

private:
  PDEOperator const * pde_operator;

  double time;
};

template<int dim, typename Number>
class Operator : public dealii::Subscriptor
{
private:
  typedef float MultigridNumber;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    PeriodicFaces;

public:
  /*
   * Constructor.
   */
  Operator(parallel::TriangulationBase<dim> &             triangulation_in,
           Mapping<dim> const &                           mapping_in,
           unsigned int const &                           degree_in,
           PeriodicFaces const &                          periodic_face_pairs_in,
           std::shared_ptr<BoundaryDescriptor<dim>> const boundary_descriptor_in,
           std::shared_ptr<FieldFunctions<dim>> const     field_functions_in,
           std::shared_ptr<MaterialDescriptor> const      material_descriptor_in,
           InputParameters const &                        param_in,
           MPI_Comm const &                               mpi_comm_in);

  /*
   * Setup function. Initializes basic finite element components, matrix-free object, and basic
   * operators. This function does not perform the setup related to the solution of linear systems
   * of equations.
   */
  void
  setup();

  /*
   * This function initializes operators, preconditioners, and solvers related to the solution of
   * linear systems of equation required for implicit formulations.
   */
  void
  setup_solver();

  /*
   * Initialization of dof-vector.
   */
  void
  initialize_dof_vector(VectorType & src) const;

  void
  reinitialize_matrix_free();

  /*
   * Prescribe initial conditions using a specified analytical/initial solution function.
   */
  void
  prescribe_initial_conditions(VectorType & src, double const time) const;

  /*
   * This function calculates the volume force term as well as contributions
   * from inhomogeneous boundary conditions.
   */
  void
  rhs(VectorType & dst, double const time = 0.0) const;

  void
  evaluate_nonlinear_residual(VectorType &       dst,
                              VectorType const & src,
                              VectorType const & rhs_vector,
                              double const       time) const;

  void
  set_solution_linearization(VectorType const & vector) const;

  void
  apply_linearized_operator(VectorType & dst, VectorType const & src, double const time) const;

  /*
   * This function solves the (non-)linear system of equations.
   */
  void
  solve_nonlinear(VectorType &       sol,
                  VectorType const & rhs,
                  double const       time,
                  bool const         update_preconditioner,
                  unsigned int &     newton_iterations,
                  unsigned int &     linear_iterations);

  unsigned int
  solve_linear(VectorType &       sol,
               VectorType const & rhs,
               double const       time,
               bool const         update_preconditioner);

  /*
   * Move mesh by a certain amount specified by the vector and update internal
   * data structures.
   */
  void
  move_mesh(const VectorType & solution);

  /*
   * Setters and getters.
   */
  MatrixFree<dim, Number> const &
  get_matrix_free() const;

  Mapping<dim> const &
  get_mapping() const;

  DoFHandler<dim> const &
  get_dof_handler() const;

private:
  /*
   * Initializes DoFHandler.
   */
  void
  distribute_dofs();

  /*
   * Initializes MatrixFree-object.
   */
  void
  initialize_matrix_free();

  /*
   * Initializes operators.
   */
  void
  setup_operators();

  /*
   * Initializes preconditioner.
   */
  void
  initialize_preconditioner();

  /*
   * Initializes solver.
   */
  void
  initialize_solver();

  /*
   * Mapping
   */
  Mapping<dim> const & mapping;

  /*
   * Periodic boundaries.
   */
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_face_pairs;

  /*
   * User interface.
   */
  std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor;
  std::shared_ptr<FieldFunctions<dim>>     field_functions;
  std::shared_ptr<MaterialDescriptor>      material_descriptor;

  /*
   * List of input parameters.
   */
  InputParameters const & param;

  /*
   * MPI communicator
   */
  MPI_Comm const & mpi_comm;

  /*
   * Basic finite element ingredients.
   */
  unsigned int const degree;

  FESystem<dim>             fe;
  DoFHandler<dim>           dof_handler;
  AffineConstraints<double> constraint_matrix;

  MatrixFree<dim, Number> matrix_free;

  /*
   * Output to screen.
   */
  ConditionalOStream pcout;

  /*
   * Basic operators.
   */
  RHSOperator<dim, Number> rhs_operator;

  LinearOperator<dim, Number>    elasticity_operator_linear;
  NonLinearOperator<dim, Number> elasticity_operator_nonlinear;
  OperatorData<dim>              operator_data;

  // operators required for Newton solver
  ResidualOperator<dim, Number>   residual_operator;
  LinearizedOperator<dim, Number> linearized_operator;

  /*
   * Solution of (non-)linear systems of equations
   */
  std::shared_ptr<PreconditionerBase<Number>> preconditioner;
  std::shared_ptr<PreconditionerBase<double>> preconditioner_amg;

  std::shared_ptr<IterativeSolverBase<VectorType>> iterative_solver;

  std::shared_ptr<NewtonSolver<VectorType,
                               ResidualOperator<dim, Number>,
                               LinearizedOperator<dim, Number>,
                               IterativeSolverBase<VectorType>>>
    non_linear_solver;
};

} // namespace Structure

#endif
