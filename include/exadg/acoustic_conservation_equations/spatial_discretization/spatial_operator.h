/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_SPATIAL_DISCRETIZATION_SPATIAL_OPERATOR_BASE_H_
#define EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_SPATIAL_DISCRETIZATION_SPATIAL_OPERATOR_BASE_H_

// deal.II
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>

// ExaDG
#include <exadg/acoustic_conservation_equations/spatial_discretization/interface.h>
#include <exadg/acoustic_conservation_equations/spatial_discretization/operators/operator.h>
#include <exadg/acoustic_conservation_equations/user_interface/boundary_descriptor.h>
#include <exadg/acoustic_conservation_equations/user_interface/field_functions.h>
#include <exadg/acoustic_conservation_equations/user_interface/parameters.h>
#include <exadg/grid/grid.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/operators/inverse_mass_operator.h>

namespace ExaDG
{
namespace Acoustics
{
/**
 * Spatial operator to solve for the acoustic conservation equations:
 *
 * 1/c^2 * dp/dt + div (rho * u) = f
 * d(rho * u)/dt + grad p = 0
 *
 * This operator solves for the pressure p as well as the velocity that is
 * already scaled by the density rho * u.
 */
template<int dim, typename Number>
class SpatialOperator : public Interface::SpatialOperator<Number>
{
  using BlockVectorType = typename Interface::SpatialOperator<Number>::BlockVectorType;

public:
  static unsigned int const block_index_pressure = 0;
  static unsigned int const block_index_velocity = 1;

  /*
   * Constructor.
   */
  SpatialOperator(std::shared_ptr<Grid<dim> const>               grid,
                  std::shared_ptr<dealii::Mapping<dim> const>    mapping,
                  std::shared_ptr<BoundaryDescriptor<dim> const> boundary_descriptor,
                  std::shared_ptr<FieldFunctions<dim> const>     field_functions,
                  Parameters const &                             parameters,
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
   * over to several single-field solvers. Another typical use case is the use of an ALE
   * formulation.
   */
  void
  setup(std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free_in,
        std::shared_ptr<MatrixFreeData<dim, Number> const>     matrix_free_data_in);

  /*
   * Getters and setters.
   */
  dealii::MatrixFree<dim, Number> const &
  get_matrix_free() const;

  std::string
  get_dof_name_pressure() const;

  unsigned int
  get_dof_index_pressure() const;

  std::string
  get_dof_name_velocity() const;

  unsigned int
  get_dof_index_velocity() const;

  unsigned int
  get_quad_index_pressure() const;

  unsigned int
  get_quad_index_velocity() const;

  unsigned int
  get_quad_index_pressure_velocity() const;

  std::shared_ptr<dealii::Mapping<dim> const>
  get_mapping() const;

  dealii::FiniteElement<dim> const &
  get_fe_p() const;

  dealii::FiniteElement<dim> const &
  get_fe_u() const;

  dealii::DoFHandler<dim> const &
  get_dof_handler_p() const;

  dealii::DoFHandler<dim> const &
  get_dof_handler_u() const;

  dealii::AffineConstraints<Number> const &
  get_constraint_p() const;

  dealii::AffineConstraints<Number> const &
  get_constraint_u() const;

  dealii::types::global_dof_index
  get_number_of_dofs() const;

  /*
   * Initialization of vectors.
   */
  void
  initialize_dof_vector(BlockVectorType & dst) const final;

  /*
   * Prescribe initial conditions using a specified analytical/initial solution function.
   */
  void
  prescribe_initial_conditions(BlockVectorType & dst, double const time) const final;

  /*
   *  This function is used in case of explicit time integration:
   *  This function evaluates the right-hand side operator, the
   *  convective and viscous terms (subsequently multiplied by -1.0 in order
   *  to shift these terms to the right-hand side of the equations)
   *  and finally applies the inverse mass operator.
   */
  void
  evaluate(BlockVectorType & dst, BlockVectorType const & src, double const time) const final;

  /*
   * Operators.
   */

  // acoustic operator
  void
  evaluate_acoustic_operator(BlockVectorType &       dst,
                             BlockVectorType const & src,
                             double const            time) const;

  /**
   * This function applies the inverse mass matrix and scales the pressure by c^2. This is because
   * the PDE we are solving for reads as
   *
   * 1/c^2 dp/dt + div  u = f
   *       du/dt + grad p = 0
   */
  void
  apply_scaled_inverse_mass_operator(BlockVectorType & dst, BlockVectorType const & src) const;

  // Calculate time step size according to local CFL criterion
  double
  calculate_time_step_cfl() const final;

private:
  void
  initialize_dof_handler_and_constraints();

  void
  initialize_operators();

  /*
   * Grid
   */
  std::shared_ptr<Grid<dim> const> grid;

  /*
   * dealii::Mapping (In case of moving meshes (ALE), this is the dynamic mapping describing the
   * deformed configuration.)
   */
  std::shared_ptr<dealii::Mapping<dim> const> mapping;

  /*
   * User interface: Boundary conditions and field functions.
   */
  std::shared_ptr<BoundaryDescriptor<dim> const> boundary_descriptor;
  std::shared_ptr<FieldFunctions<dim> const>     field_functions;

  /*
   * List of parameters.
   */
  Parameters const & param;

  /*
   * A name describing the field being solved.
   */
  std::string const field;

  /*
   * Basic finite element ingredients.
   */
  std::shared_ptr<dealii::FiniteElement<dim>> fe_p;
  std::shared_ptr<dealii::FiniteElement<dim>> fe_u;

  dealii::DoFHandler<dim> dof_handler_p;
  dealii::DoFHandler<dim> dof_handler_u;

  dealii::AffineConstraints<Number> constraint_p, constraint_u;

  std::string const dof_index_p = "pressure";
  std::string const dof_index_u = "velocity";

  std::string const quad_index_p = "pressure";
  std::string const quad_index_u = "velocity";

  // Quadrature that works for both, pressure and velocity, i.e. n_q=(max(k_p,k_u)+1)^dim.
  // This quadrature is needed for the acoustic operator which iterates over pressure and
  // velocity quadrature points in the same loop (for performance reasons).
  std::string const quad_index_p_u = "pressure_velocity";

  std::shared_ptr<MatrixFreeData<dim, Number> const>     matrix_free_data;
  std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free;

  /*
   * Basic operators.
   */
  Operator<dim, Number> acoustic_operator;

  /*
   * Inverse mass operator
   */
  InverseMassOperator<dim, 1, Number>   inverse_mass_pressure;
  InverseMassOperator<dim, dim, Number> inverse_mass_velocity;

  MPI_Comm const mpi_comm;

  dealii::ConditionalOStream pcout;
};

} // namespace Acoustics
} // namespace ExaDG

#endif /* EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_SPATIAL_DISCRETIZATION_SPATIAL_OPERATOR_BASE_H_ */
