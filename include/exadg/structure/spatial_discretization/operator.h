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
#include <exadg/grid/grid.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/operators/inverse_mass_operator.h>
#include <exadg/operators/mass_operator.h>
#include <exadg/solvers_and_preconditioners/newton/newton_solver.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_base.h>
#include <exadg/structure/spatial_discretization/interface.h>
#include <exadg/structure/spatial_discretization/operators/body_force_operator.h>
#include <exadg/structure/spatial_discretization/operators/linear_operator.h>
#include <exadg/structure/spatial_discretization/operators/mass_operator.h>
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
  update(VectorType const & const_vector, double const scaling_factor_mass, double const time)
  {
    this->const_vector        = &const_vector;
    this->scaling_factor_mass = scaling_factor_mass;
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
  update(double const scaling_factor_mass, double const time)
  {
    this->scaling_factor_mass = scaling_factor_mass;
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
  Operator(std::shared_ptr<Grid<dim> const>               grid,
           std::shared_ptr<dealii::Mapping<dim> const>    mapping,
           std::shared_ptr<BoundaryDescriptor<dim> const> boundary_descriptor,
           std::shared_ptr<FieldFunctions<dim> const>     field_functions,
           std::shared_ptr<MaterialDescriptor const>      material_descriptor,
           Parameters const &                             param,
           std::string const &                            field,
           MPI_Comm const &                               mpi_comm);

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

  /*
   * This function initializes operators, preconditioners, and solvers related to the solution of
   * (non-)linear systems of equation.
   */
  void
  setup_solver(double const & scaling_factor_acceleration, double const & scaling_factor_velocity);

  /*
   * Initialization of dof-vector.
   */
  void
  initialize_dof_vector(VectorType & src) const final;

  /*
   * Prescribe initial conditions using a specified initial solution function.
   */
  void
  prescribe_initial_displacement(VectorType & displacement, double const time) const final;

  void
  prescribe_initial_velocity(VectorType & velocity, double const time) const final;

  /*
   * This computes the initial acceleration field by evaluating all PDE terms for the given
   * initial condition, shifting all terms to the right-hand side of the equations, and solving a
   * mass matrix system to obtain the initial acceleration.
   */
  void
  compute_initial_acceleration(VectorType &       initial_acceleration,
                               VectorType const & initial_displacement,
                               double const       time) const final;

  void
  evaluate_mass_operator(VectorType & dst, VectorType const & src) const final;

  void
  apply_add_damping_operator(VectorType & dst, VectorType const & src) const final;

  /*
   * This function evaluates the nonlinear residual which is required by the Newton solver. In order
   * to evaluate inhomogeneous Dirichlet boundary conditions correctly, inhomogeneous Dirichlet
   * degrees of freedom need to be set correctly in the src-vector prior to calling this function.
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
  evaluate_elasticity_operator(VectorType &       dst,
                               VectorType const & src,
                               double const       factor,
                               double const       time) const;

  void
  apply_elasticity_operator(VectorType &       dst,
                            VectorType const & src,
                            VectorType const & linearization,
                            double const       factor,
                            double const       time) const;

  /*
   * This function solves the system of equations for nonlinear problems. This function needs to
   * make sure that Dirichlet degrees of freedom are filled correctly with their inhomogeneous
   * boundary data before calling the nonlinear solver.
   */
  std::tuple<unsigned int, unsigned int>
  solve_nonlinear(VectorType &       sol,
                  VectorType const & const_vector,
                  double const       scaling_factor_acceleration,
                  double const       scaling_factor_velocity,
                  double const       time,
                  bool const         update_preconditioner) const final;

  /*
   * This function calculates the right-hand side of the linear system of equations for linear
   * elasticity problems.
   */
  void
  rhs(VectorType & dst, double const time) const final;

  /*
   * This function solves the system of equations for linear problems.
   *
   * Before calling this function, make sure that the function rhs() has been called.
   */
  unsigned int
  solve_linear(VectorType &       sol,
               VectorType const & rhs,
               double const       scaling_factor_acceleration,
               double const       scaling_factor_velocity,
               double const       time,
               bool const         update_preconditioner) const final;

  /*
   * Setters and getters.
   */
  std::shared_ptr<dealii::MatrixFree<dim, Number> const>
  get_matrix_free() const;

  dealii::Mapping<dim> const &
  get_mapping() const;

  dealii::DoFHandler<dim> const &
  get_dof_handler() const;

  dealii::types::global_dof_index
  get_number_of_dofs() const;

  // Multiphysics coupling via "Cached" boundary conditions
  std::shared_ptr<ContainerInterfaceData<1, dim, double>>
  get_container_interface_data_neumann() const;

  std::shared_ptr<ContainerInterfaceData<1, dim, double>>
  get_container_interface_data_dirichlet() const;

  // TODO: we currently need this function public for precice-based FSI
  unsigned int
  get_dof_index() const;

private:
  /*
   * Initializes dealii::DoFHandler.
   */
  void
  initialize_dof_handler_and_constraints();

  std::string
  get_dof_name() const;

  std::string
  get_dof_name_periodicity_and_hanging_node_constraints() const;

  std::string
  get_quad_name() const;

  std::string
  get_quad_gauss_lobatto_name() const;

  unsigned int
  get_dof_index_periodicity_and_hanging_node_constraints() const;

  unsigned int
  get_quad_index() const;

  unsigned int
  get_quad_index_gauss_lobatto() const;

  /**
   * Scaling factor for mass matrix assuming a weak damping operator leading to a scaled mass
   * matrix.
   */
  double
  compute_scaling_factor_mass(double const scaling_factor_acceleration,
                              double const scaling_factor_velocity) const;

  /**
   * Setup of "cached" boundary conditions for coupling with other domains.
   */
  void
  setup_coupling_boundary_conditions();

  /**
   * Initializes operators.
   */
  void
  setup_operators();

  /**
   * Initializes preconditioner.
   */
  void
  initialize_preconditioner();

  /**
   * Initializes solver.
   */
  void
  initialize_solver();

  /*
   * Grid
   */
  std::shared_ptr<Grid<dim> const> grid;

  /*
   * Mapping
   */
  std::shared_ptr<dealii::Mapping<dim> const> mapping;

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
  std::shared_ptr<dealii::FiniteElement<dim>> fe;
  dealii::DoFHandler<dim>                     dof_handler;

  // AffineConstraints object as needed by iterative solvers and preconditioners for linear systems
  // of equations. This constraint object contains additional constraints from Dirichlet boundary
  // conditions as compared to the constraint object below. Note that the present constraint object
  // can treat Dirichlet boundaries only in a homogeneous manner.
  dealii::AffineConstraints<Number> affine_constraints;

  // To treat inhomogeneous Dirichlet BCs correctly in the context of matrix-free operator
  // evaluation using dealii::MatrixFree/FEEvaluation, we need a separate AffineConstraints
  // object containing only periodicity and hanging node constraints.
  // When using the standard AffineConstraints object including Dirichlet boundary conditions,
  // inhomogeneous boundary data would be ignored by dealii::FEEvaluation::read_dof_values().
  // While dealii::FEEvaluation::read_dof_values_plain() would take into account inhomogeneous
  // Dirichlet data using the standard AffineConstraints object, hanging-node constraints would
  // not be resolved correctly.
  // The solution/workaround is to use dealii::FEEvaluation::read_dof_values() for a correct
  // handling of hanging nodes, but to exclude Dirichlet degrees of freedom from the
  // AffineConstraints object so that it is possible to read inhomogeneous boundary data when
  // calling dealii::FEEvaluation::read_dof_values(). This inhomogeneous boundary data needs to be
  // set beforehand in separate routines.
  dealii::AffineConstraints<Number> affine_constraints_periodicity_and_hanging_nodes;

  std::string const dof_index = "dof";
  std::string const dof_index_periodicity_and_handing_node_constraints =
    "dof_periodicity_hanging_nodes";

  std::string const quad_index               = "quad";
  std::string const quad_index_gauss_lobatto = "quad_gauss_lobatto";

  /*
   * Matrix-free operator evaluation.
   */
  std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free;
  std::shared_ptr<MatrixFreeData<dim, Number> const>     matrix_free_data;

  /*
   * Interface coupling
   */
  // TODO: The PDE operator should only have read access to interface data
  mutable std::shared_ptr<ContainerInterfaceData<1, dim, double>> interface_data_dirichlet_cached;
  mutable std::shared_ptr<ContainerInterfaceData<1, dim, double>> interface_data_neumann_cached;

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
  Structure::MassOperator<dim, Number> mass_operator;

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
