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

#ifndef INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_H_
#define INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_H_

// deal.II
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/compressible_navier_stokes/spatial_discretization/calculators.h>
#include <exadg/compressible_navier_stokes/spatial_discretization/interface.h>
#include <exadg/compressible_navier_stokes/spatial_discretization/kernels_and_operators.h>
#include <exadg/compressible_navier_stokes/user_interface/boundary_descriptor.h>
#include <exadg/compressible_navier_stokes/user_interface/field_functions.h>
#include <exadg/compressible_navier_stokes/user_interface/parameters.h>
#include <exadg/grid/grid.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/operators/inverse_mass_operator.h>
#include <exadg/operators/navier_stokes_calculators.h>

namespace ExaDG
{
namespace CompNS
{
template<int dim, typename Number>
class Operator : public dealii::Subscriptor, public Interface::Operator<Number>
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  Operator(std::shared_ptr<GridManager<dim> const>        grid,
           std::shared_ptr<BoundaryDescriptor<dim> const> boundary_descriptor,
           std::shared_ptr<FieldFunctions<dim> const>     field_functions,
           Parameters const &                             param,
           std::string const &                            field,
           MPI_Comm const &                               mpi_comm);

  void
  fill_matrix_free_data(MatrixFreeData<dim, Number> & matrix_free_data) const;

  void
  setup(std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free,
        std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data);

  dealii::types::global_dof_index
  get_number_of_dofs() const;

  // initialization of DoF vectors
  void
  initialize_dof_vector(VectorType & src) const;

  void
  initialize_dof_vector_scalar(VectorType & src) const;

  void
  initialize_dof_vector_dim_components(VectorType & src) const;

  // set initial conditions
  void
  prescribe_initial_conditions(VectorType & src, double const time) const;

  /*
   *  This function is used in case of explicit time integration:
   *  This function evaluates the right-hand side operator, the
   *  convective and viscous terms (subsequently multiplied by -1.0 in order
   *  to shift these terms to the right-hand side of the equations)
   *  and finally applies the inverse mass operator.
   */
  void
  evaluate(VectorType & dst, VectorType const & src, Number const time) const;

  void
  evaluate_convective(VectorType & dst, VectorType const & src, Number const time) const;

  void
  evaluate_viscous(VectorType & dst, VectorType const & src, Number const time) const;

  void
  evaluate_convective_and_viscous(VectorType &       dst,
                                  VectorType const & src,
                                  Number const       time) const;

  void
  apply_inverse_mass(VectorType & dst, VectorType const & src) const;

  // getters
  dealii::MatrixFree<dim, Number> const &
  get_matrix_free() const;

  dealii::Mapping<dim> const &
  get_mapping() const;

  dealii::FESystem<dim> const &
  get_fe() const;

  dealii::DoFHandler<dim> const &
  get_dof_handler() const;

  dealii::DoFHandler<dim> const &
  get_dof_handler_scalar() const;

  dealii::DoFHandler<dim> const &
  get_dof_handler_vector() const;

  unsigned int
  get_dof_index_vector() const;

  unsigned int
  get_dof_index_scalar() const;

  unsigned int
  get_quad_index_standard() const;

  // pressure
  void
  compute_pressure(VectorType & dst, VectorType const & src) const;

  // velocity
  void
  compute_velocity(VectorType & dst, VectorType const & src) const;

  // temperature
  void
  compute_temperature(VectorType & dst, VectorType const & src) const;

  // vorticity
  void
  compute_vorticity(VectorType & dst, VectorType const & src) const;

  // divergence
  void
  compute_divergence(VectorType & dst, VectorType const & src) const;

  // shear rate
  void
  compute_shear_rate(VectorType & dst, VectorType const & src) const;

  double
  get_wall_time_operator_evaluation() const;

  // global CFL criterion: calculates the time step size for a given global maximum velocity
  double
  calculate_time_step_cfl_global() const;

  // Calculate time step size according to diffusion term
  double
  calculate_time_step_diffusion() const;

private:
  double
  calculate_minimum_element_length() const;

  void
  distribute_dofs();

  void
  setup_operators();

  unsigned int
  get_dof_index_all() const;

  unsigned int
  get_quad_index_overintegration_conv() const;

  unsigned int
  get_quad_index_overintegration_vis() const;

  unsigned int
  get_quad_index_l2_projections() const;

  /*
   * Grid
   */
  std::shared_ptr<GridManager<dim> const> grid;

  /*
   * User interface: Boundary conditions and field functions.
   */
  std::shared_ptr<BoundaryDescriptor<dim> const> boundary_descriptor;
  std::shared_ptr<FieldFunctions<dim> const>     field_functions;

  /*
   * List of parameters.
   */
  Parameters const & param;

  std::string const field;

  /*
   * Basic finite element ingredients.
   */

  std::shared_ptr<dealii::FESystem<dim>> fe;        // all (dim+2) components: (rho, rho u, rho E)
  std::shared_ptr<dealii::FESystem<dim>> fe_vector; // e.g. velocity
  dealii::FE_DGQ<dim>                    fe_scalar; // scalar quantity, e.g, pressure

  // dealii::Quadrature points
  unsigned int n_q_points_conv;
  unsigned int n_q_points_visc;

  // dealii::DoFHandler
  dealii::DoFHandler<dim> dof_handler;        // all (dim+2) components: (rho, rho u, rho E)
  dealii::DoFHandler<dim> dof_handler_vector; // e.g. velocity
  dealii::DoFHandler<dim> dof_handler_scalar; // scalar quantity, e.g, pressure

  std::string const dof_index_all    = "all_fields";
  std::string const dof_index_vector = "vector";
  std::string const dof_index_scalar = "scalar";

  std::string const quad_index_standard             = "standard";
  std::string const quad_index_overintegration_conv = "overintegration_conv";
  std::string const quad_index_overintegration_vis  = "overintegration_vis";

  std::string const quad_index_l2_projections = quad_index_standard;
  // alternative: use more accurate over-integration strategy
  //  std::string const quad_index_l2_projections = quad_index_overintegration_conv;

  /*
   * Constraints.
   */
  dealii::AffineConstraints<Number> constraint;

  /*
   * Matrix-free operator evaluation.
   */
  std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data;
  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free;

  /*
   * Basic operators.
   */
  MassOperator<dim, Number>       mass_operator;
  BodyForceOperator<dim, Number>  body_force_operator;
  ConvectiveOperator<dim, Number> convective_operator;
  ViscousOperator<dim, Number>    viscous_operator;

  /*
   * Merged operators.
   */
  CombinedOperator<dim, Number> combined_operator;

  InverseMassOperator<dim, dim + 2, Number> inverse_mass_all;
  InverseMassOperator<dim, dim, Number>     inverse_mass_vector;
  InverseMassOperator<dim, 1, Number>       inverse_mass_scalar;

  // L2 projections to calculate derived quantities
  p_u_T_Calculator<dim, Number>     p_u_T_calculator;
  VorticityCalculator<dim, Number>  vorticity_calculator;
  DivergenceCalculator<dim, Number> divergence_calculator;
  ShearRateCalculator<dim, Number>  shear_rate_calculator;

  /*
   * MPI
   */
  MPI_Comm const mpi_comm;

  /*
   * Output to screen.
   */
  dealii::ConditionalOStream pcout;

  // wall time for operator evaluation
  mutable double wall_time_operator_evaluation;
};

} // namespace CompNS
} // namespace ExaDG

#endif /* INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_ */
