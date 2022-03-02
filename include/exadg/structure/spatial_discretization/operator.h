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

#ifndef INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_
#define INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_

// deal.II
#include <deal.II/fe/fe_system.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/interface_coupling.h>
#include <exadg/grid/grid.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/operators/inverse_mass_operator.h>
#include <exadg/operators/mass_operator.h>
#include <exadg/solvers_and_preconditioners/newton/newton_solver.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_base.h>
#include <exadg/structure/spatial_discretization/interface.h>
#include <exadg/structure/spatial_discretization/operators/body_force_operator.h>
#include <exadg/structure/spatial_discretization/operators/linear_operator.h>
#include <exadg/structure/spatial_discretization/operators/nonlinear_operator.h>
#include <exadg/structure/user_interface/boundary_descriptor.h>
#include <exadg/structure/user_interface/field_functions.h>
#include <exadg/structure/user_interface/parameters.h>

namespace ExaDG
{
namespace Structure
{
// forward declaration
template<int dim, typename Number>
class Operator;

/*
 * This operator provides the interface required by the non-linear Newton solver.
 * Requests to evaluate the residual for example are simply handed over to the
 * operators that implement the physics.
 */
template<int dim, typename Number>
class ResidualOperator
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef Operator<dim, Number> PDEOperator;

public:
  ResidualOperator()
    : pde_operator(nullptr), const_vector(nullptr), scaling_factor_mass(0.0), time(0.0)
  {
  }

  void
  initialize(PDEOperator const & pde_operator)
  {
    this->pde_operator = &pde_operator;
  }

  void
  update(VectorType const & const_vector, double const factor, double const time)
  {
    this->const_vector        = &const_vector;
    this->scaling_factor_mass = factor;
    this->time                = time;
  }

  /*
   * The implementation of the Newton solver requires a function called
   * 'evaluate_residual'.
   */
  void
  evaluate_residual(VectorType & dst, VectorType const & src) const
  {
    pde_operator->evaluate_nonlinear_residual(dst, src, *const_vector, scaling_factor_mass, time);
  }

private:
  PDEOperator const * pde_operator;

  VectorType const * const_vector;

  double scaling_factor_mass;
  double time;
};

/*
 * This operator implements the interface required by the non-linear Newton solver.
 * Requests to apply the linearized operator are simply handed over to the
 * operators that implement the physics.
 */
template<int dim, typename Number>
class LinearizedOperator : public dealii::Subscriptor
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef Operator<dim, Number> PDEOperator;

public:
  LinearizedOperator()
    : dealii::Subscriptor(), pde_operator(nullptr), scaling_factor_mass(0.0), time(0.0)
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
  update(double const factor, double const time)
  {
    this->scaling_factor_mass = factor;
    this->time                = time;
  }

  /*
   * The implementation of linear solvers in deal.ii requires that a function called 'vmult' is
   * provided.
   */
  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    pde_operator->apply_linearized_operator(dst, src, scaling_factor_mass, time);
  }

private:
  PDEOperator const * pde_operator;

  double scaling_factor_mass;
  double time;
};

template<int dim, typename Number>
class Operator : public dealii::Subscriptor, public Interface::Operator<Number>
{
private:
  typedef float MultigridNumber;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  /*
   * Constructor.
   */
  Operator(std::shared_ptr<Grid<dim> const>               grid_in,
           std::shared_ptr<BoundaryDescriptor<dim> const> boundary_descriptor_in,
           std::shared_ptr<FieldFunctions<dim> const>     field_functions_in,
           std::shared_ptr<MaterialDescriptor const>      material_descriptor_in,
           Parameters const &                             param_in,
           std::string const &                            field_in,
           MPI_Comm const &                               mpi_comm_in);

  void
  fill_matrix_free_data(MatrixFreeData<dim, Number> & matrix_free_data) const;

  /*
   * Setup function. Initializes basic operators. This function does not perform the setup
   * related to the solution of linear systems of equations.
   */
  void
  setup(std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free,
        std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data);

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

  /*
   * Prescribe initial conditions using a specified initial solution function.
   */
  void
  prescribe_initial_displacement(VectorType & displacement, double const time) const;

  void
  prescribe_initial_velocity(VectorType & velocity, double const time) const;

  void
  compute_initial_acceleration(VectorType &       acceleration,
                               VectorType const & displacement,
                               double const       time) const;

  void
  apply_mass_operator(VectorType & dst, VectorType const & src) const;

  /*
   * This function calculates the right-hand side of the linear system
   * of equations for linear elasticity problems.
   */
  void
  compute_rhs_linear(VectorType & dst, double const time) const;

  /*
   * This function evaluates the nonlinear residual which is required by
   * the Newton solver.
   */
  void
  evaluate_nonlinear_residual(VectorType &       dst,
                              VectorType const & src,
                              VectorType const & const_vector,
                              double const       factor,
                              double const       time) const;

  void
  set_solution_linearization(VectorType const & vector) const;

  void
  apply_linearized_operator(VectorType &       dst,
                            VectorType const & src,
                            double const       factor,
                            double const       time) const;

  void
  apply_nonlinear_operator(VectorType &       dst,
                           VectorType const & src,
                           double const       factor,
                           double const       time) const;

  void
  apply_linear_operator(VectorType &       dst,
                        VectorType const & src,
                        double const       factor,
                        double const       time) const;

  void
  set_constrained_values_to_zero(VectorType & vector) const;

  bool
  check_constrained_values_are_zero(VectorType const & vector) const;

  /*
   * This function solves the (non-)linear system of equations.
   */
  std::tuple<unsigned int, unsigned int>
  solve_nonlinear(VectorType &       sol,
                  VectorType const & rhs,
                  double const       factor,
                  double const       time,
                  bool const         update_preconditioner) const;

  unsigned int
  solve_linear(VectorType &       sol,
               VectorType const & rhs,
               double const       factor,
               double const       time) const;

  /*
   * Setters and getters.
   */
  dealii::MatrixFree<dim, Number> const &
  get_matrix_free() const;

  dealii::Mapping<dim> const &
  get_mapping() const;

  dealii::DoFHandler<dim> const &
  get_dof_handler() const;

  dealii::types::global_dof_index
  get_number_of_dofs() const;

  // Multiphysics coupling via "Cached" boundary conditions
  std::shared_ptr<ContainerInterfaceData<dim, dim, Number>>
  get_container_interface_data_neumann();

  std::shared_ptr<ContainerInterfaceData<dim, dim, Number>>
  get_container_interface_data_dirichlet();

private:
  /*
   * Initializes dealii::DoFHandler.
   */
  void
  distribute_dofs();

  std::string
  get_dof_name() const;

  std::string
  get_dof_name_mass() const;

  std::string
  get_quad_name() const;

  std::string
  get_quad_gauss_lobatto_name() const;

  unsigned int
  get_dof_index_mass() const;

  unsigned int
  get_dof_index() const;

  unsigned int
  get_quad_index() const;

  unsigned int
  get_quad_index_gauss_lobatto() const;

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
   * Grid
   */
  std::shared_ptr<Grid<dim> const> grid;

  /*
   * User interface.
   */
  std::shared_ptr<BoundaryDescriptor<dim> const> boundary_descriptor;
  std::shared_ptr<FieldFunctions<dim> const>     field_functions;
  std::shared_ptr<MaterialDescriptor const>      material_descriptor;

  /*
   * List of parameters.
   */
  Parameters const & param;

  std::string const field;

  /*
   * Basic finite element ingredients.
   */
  dealii::FESystem<dim>             fe;
  dealii::DoFHandler<dim>           dof_handler;
  dealii::AffineConstraints<Number> affine_constraints;
  // constraints for mass operator (i.e., do not apply any constraints)
  dealii::AffineConstraints<Number> constraints_mass;

  std::string const dof_index                = "dof";
  std::string const dof_index_mass           = "dof_mass";
  std::string const quad_index               = "quad";
  std::string const quad_index_gauss_lobatto = "quad_gauss_lobatto";

  /*
   * Matrix-free operator evaluation.
   */
  std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data;
  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free;

  /*
   * Interface coupling
   */
  std::shared_ptr<ContainerInterfaceData<dim, dim, Number>> interface_data_dirichlet_cached;
  std::shared_ptr<ContainerInterfaceData<dim, dim, Number>> interface_data_neumann_cached;

  /*
   * Basic operators.
   */
  BodyForceOperator<dim, Number> body_force_operator;

  LinearOperator<dim, Number>    elasticity_operator_linear;
  NonLinearOperator<dim, Number> elasticity_operator_nonlinear;
  OperatorData<dim>              operator_data;

  // The mass operator is only relevant for unsteady problems:
  // it is used to compute the initial acceleration and to evaluate
  // the mass operator term applied to a constant vector (independent
  // of new displacements) appearing on the right-hand side for linear
  // problems and in the residual for nonlinear problems.
  MassOperator<dim, dim, Number> mass_operator;

  /*
   * Solution of nonlinear systems of equations
   */

  // operators required for Newton solver
  mutable ResidualOperator<dim, Number>   residual_operator;
  mutable LinearizedOperator<dim, Number> linearized_operator;

  typedef Newton::Solver<VectorType,
                         ResidualOperator<dim, Number>,
                         LinearizedOperator<dim, Number>,
                         Krylov::SolverBase<VectorType>>
    NewtonSolver;

  std::shared_ptr<NewtonSolver> newton_solver;

  /*
   * Solution of linear systems of equations
   */
  std::shared_ptr<PreconditionerBase<Number>> preconditioner;

  std::shared_ptr<Krylov::SolverBase<VectorType>> linear_solver;

  // mass operator inversion
  std::shared_ptr<PreconditionerBase<Number>>     mass_preconditioner;
  std::shared_ptr<Krylov::SolverBase<VectorType>> mass_solver;

  /*
   * MPI communicator
   */
  MPI_Comm const mpi_comm;

  /*
   * Output to screen.
   */
  dealii::ConditionalOStream pcout;
};

} // namespace Structure
} // namespace ExaDG

#endif
