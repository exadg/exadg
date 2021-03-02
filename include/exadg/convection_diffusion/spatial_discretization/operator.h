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
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

// ExaDG
#include <exadg/convection_diffusion/spatial_discretization/interface.h>
#include <exadg/convection_diffusion/spatial_discretization/operators/combined_operator.h>
#include <exadg/convection_diffusion/user_interface/boundary_descriptor.h>
#include <exadg/convection_diffusion/user_interface/field_functions.h>
#include <exadg/convection_diffusion/user_interface/input_parameters.h>
#include <exadg/matrix_free/matrix_free_wrapper.h>
#include <exadg/operators/inverse_mass_operator.h>
#include <exadg/operators/mass_operator.h>
#include <exadg/operators/rhs_operator.h>
#include <exadg/solvers_and_preconditioners/preconditioner/preconditioner_base.h>

namespace ExaDG
{
namespace ConvDiff
{
using namespace dealii;

template<int dim, typename Number>
class Operator : public dealii::Subscriptor, public Interface::Operator<Number>
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    PeriodicFaces;

public:
  /*
   * Constructor.
   */
  Operator(parallel::TriangulationBase<dim> const &       triangulation,
           Mapping<dim> const &                           mapping,
           unsigned int const                             degree,
           PeriodicFaces const                            periodic_face_pairs,
           std::shared_ptr<BoundaryDescriptor<dim>> const boundary_descriptor,
           std::shared_ptr<FieldFunctions<dim>> const     field_functions,
           InputParameters const &                        param,
           std::string const &                            field,
           MPI_Comm const &                               mpi_comm);


  void
  fill_matrix_free_data(MatrixFreeData<dim, Number> & matrix_free_data) const;

  /*
   * Setup function. Initializes basic finite element components, matrix-free object, and basic
   * operators. This function does not perform the setup related to the solution of linear systems
   * of equations.
   */
  void
  setup(std::shared_ptr<MatrixFree<dim, Number>>     matrix_free_in,
        std::shared_ptr<MatrixFreeData<dim, Number>> matrix_free_data_in,
        std::string const &                          dof_index_velocity_external_in = "");

  /*
   * This function initializes operators, preconditioners, and solvers related to the solution of
   * linear systems of equation required for implicit formulations.
   */
  void
  setup_solver(double const scaling_factor_mass = -1.0, VectorType const * velocity = nullptr);

  /*
   * Initialization of dof-vector.
   */
  void
  initialize_dof_vector(VectorType & src) const;

  /*
   * Initialization of velocity dof-vector (in case of numerical velocity field).
   */
  void
  initialize_dof_vector_velocity(VectorType & src) const;

  /*
   * Obtain velocity dof-vector by interpolation of specified analytical velocity field.
   */
  void
  interpolate_velocity(VectorType & velocity, double const time) const;

  /*
   * Obtain velocity dof-vector by L2-projection using the specified analytical velocity field.
   */
  void
  project_velocity(VectorType & velocity, double const time) const;

  /*
   * Prescribe initial conditions using a specified analytical function.
   */
  void
  prescribe_initial_conditions(VectorType & src, double const evaluation_time) const;

  /*
   * This function is used in case of explicit time integration:
   *
   * It evaluates the right-hand side operator, the convective and diffusive terms (subsequently
   * multiplied by -1.0 in order to shift these terms to the right-hand side of the equations) and
   * finally applies the inverse mass operator.
   */
  void
  evaluate_explicit_time_int(VectorType &       dst,
                             VectorType const & src,
                             double const       evaluation_time,
                             VectorType const * velocity = nullptr) const;

  /*
   * This function evaluates the convective term which is needed when using an explicit formulation
   * for the convective term.
   */
  void
  evaluate_convective_term(VectorType &       dst,
                           VectorType const & src,
                           double const       evaluation_time,
                           VectorType const * velocity = nullptr) const;

  /*
   * This function is called by OIF sub-stepping algorithm. It evaluates the convective term,
   * multiplies the result by -1.0 and applies the inverse mass operator.
   */
  void
  evaluate_oif(VectorType &       dst,
               VectorType const & src,
               double const       evaluation_time,
               VectorType const * velocity = nullptr) const;

  /*
   * This function calculates the inhomogeneous parts of all operators arising e.g. from
   * inhomogeneous boundary conditions or the solution at previous instants of time occurring in the
   * discrete time derivative term.
   *
   * Note that the convective operator only has a contribution to the right-hand side if it is
   * formulated implicitly in time. In case of an explicit treatment the whole convective operator
   * (call function evaluate_convective_term() instead of rhs()) has to be added to the right-hand
   * side of the equations.
   */
  void
  rhs(VectorType &       dst,
      double const       evaluation_time = 0.0,
      VectorType const * velocity        = nullptr) const;

  /*
   * This function applies the mass operator to the src-vector and writes the result to the
   * dst-vector.
   */
  void
  apply_mass_operator(VectorType & dst, VectorType const & src) const;

  /*
   * This function applies the mass operator to the src-vector and adds the result to the
   * dst-vector.
   */
  void
  apply_mass_operator_add(VectorType & dst, VectorType const & src) const;

  /*
   * This function applies the convective operator to the src-vector and writes the result to the
   * dst-vector. It is needed for throughput measurements of the matrix-free implementation.
   */
  void
  apply_convective_term(VectorType & dst, VectorType const & src) const;

  void
  update_convective_term(double const evaluation_time, VectorType const * velocity = nullptr) const;

  /*
   * This function applies the diffusive operator to the src-vector and writes the result to the
   * dst-vector. It is needed for throughput measurements of the matrix-free implementation.
   */
  void
  apply_diffusive_term(VectorType & dst, VectorType const & src) const;

  /*
   * This function applies the combined mass-convection-diffusion operator to the src-vector
   * and writes the result to the dst-vector. It is needed for throughput measurements of the
   * matrix-free implementation.
   */
  void
  apply_conv_diff_operator(VectorType & dst, VectorType const & src) const;

  void
  update_conv_diff_operator(double const       evaluation_time,
                            double const       scaling_factor,
                            VectorType const * velocity = nullptr);

  /*
   * This function solves the linear system of equations in case of implicit time integration or
   * steady-state problems (potentially involving the mass, convective, and diffusive
   * operators).
   */
  unsigned int
  solve(VectorType &       sol,
        VectorType const & rhs,
        bool const         update_preconditioner,
        double const       scaling_factor = -1.0,
        double const       time           = -1.0,
        VectorType const * velocity       = nullptr);

  /*
   * Calculate time step size according to local CFL criterion
   */

  // use numerical velocity field
  double
  calculate_time_step_cfl_numerical_velocity(VectorType const & velocity,
                                             double const       cfl,
                                             double const       exponent_degree) const;

  // use analytical velocity field
  double
  calculate_time_step_cfl_analytical_velocity(double const time,
                                              double const cfl,
                                              double const exponent_degree) const;

  // Calculate maximum velocity (required for global CFL criterion).
  double
  calculate_maximum_velocity(double const time) const;

  // Calculate minimum element length (required for global CFL criterion).
  double
  calculate_minimum_element_length() const;

  /*
   * Setters and getters.
   */

  MatrixFree<dim, Number> const &
  get_matrix_free() const;

  DoFHandler<dim> const &
  get_dof_handler() const;

  DoFHandler<dim> const &
  get_dof_handler_velocity() const;

  unsigned int
  get_polynomial_degree() const;

  types::global_dof_index
  get_number_of_dofs() const;

  std::string
  get_dof_name() const;

  void
  update_after_mesh_movement();

  unsigned int
  get_dof_index() const;

  unsigned int
  get_quad_index() const;

private:
  /*
   * Initializes DoFHandlers.
   */
  void
  distribute_dofs();

  bool
  needs_own_dof_handler_velocity() const;

  std::string
  get_quad_name() const;

  std::string
  get_quad_name_overintegration() const;

  std::string
  get_dof_name_velocity() const;

  /*
   * Dof index for velocity (in case of numerical velocity field)
   */
  unsigned int
  get_dof_index_velocity() const;

  unsigned int
  get_quad_index_overintegration() const;

  /*
   * Initializes the preconditioner.
   */
  void
  initialize_preconditioner();

  /*
   * Initializes the solver.
   */
  void
  initialize_solver();

  /*
   * Mapping
   */
  Mapping<dim> const & mapping;

  /*
   * Polynomial degree of shape function
   */
  unsigned int const degree;

  /*
   * Periodic face pairs: This variable is only needed when using a multigrid preconditioner
   */
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_face_pairs;

  /*
   * User interface: Boundary conditions and field functions.
   */
  std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor;
  std::shared_ptr<FieldFunctions<dim>>     field_functions;

  /*
   * List of input parameters.
   */
  InputParameters const & param;

  std::string const field;

  /*
   * Basic finite element ingredients.
   */
  FE_DGQ<dim>     fe;
  DoFHandler<dim> dof_handler;

  /*
   * Numerical velocity field.
   */
  std::shared_ptr<FESystem<dim>>   fe_velocity;
  std::shared_ptr<DoFHandler<dim>> dof_handler_velocity;

  /*
   * Constraints.
   */
  AffineConstraints<Number> affine_constraints;

  std::string const dof_index_std      = "conv_diff";
  std::string const dof_index_velocity = "conv_diff_velocity";

  std::string const quad_index_std             = "conv_diff";
  std::string const quad_index_overintegration = "conv_diff_overintegration";

  std::string dof_index_velocity_external;

  /*
   * Matrix-free operator evaluation.
   */
  std::shared_ptr<MatrixFree<dim, Number>>     matrix_free;
  std::shared_ptr<MatrixFreeData<dim, Number>> matrix_free_data;

  /*
   * Basic operators.
   */
  std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> convective_kernel;
  std::shared_ptr<Operators::DiffusiveKernel<dim, Number>>  diffusive_kernel;

  MassOperator<dim, 1, Number>        mass_operator;
  InverseMassOperator<dim, 1, Number> inverse_mass_operator;
  ConvectiveOperator<dim, Number>     convective_operator;
  DiffusiveOperator<dim, Number>      diffusive_operator;
  RHSOperator<dim, Number>            rhs_operator;

  /*
   * Combined operator.
   */
  CombinedOperator<dim, Number> combined_operator;

  /*
   * Solvers and preconditioners
   */
  std::shared_ptr<PreconditionerBase<Number>>      preconditioner;
  std::shared_ptr<IterativeSolverBase<VectorType>> iterative_solver;

  /*
   * MPI
   */
  MPI_Comm const mpi_comm;

  /*
   * Output to screen.
   */
  ConditionalOStream pcout;
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_ */
