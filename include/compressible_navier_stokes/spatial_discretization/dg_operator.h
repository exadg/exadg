/*
 * dg_operator.h
 *
 *  Created on: 2018
 *      Author: fehn
 */

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_OPERATOR_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_OPERATOR_H_

// deal.II
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/vector_tools.h>

// user interface
#include "../../compressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../../compressible_navier_stokes/user_interface/field_functions.h"
#include "../../compressible_navier_stokes/user_interface/input_parameters.h"

// operators
#include "../../compressible_navier_stokes/spatial_discretization/comp_navier_stokes_operators.h"
#include "comp_navier_stokes_calculators.h"
#include "operators/inverse_mass_matrix.h"

// interface
#include "interface.h"

// time step calculation
#include "time_integration/time_step_calculation.h"

// postprocessor
#include "../postprocessor/postprocessor_base.h"

namespace CompNS
{
template<int dim, typename Number>
class DGOperator : public dealii::Subscriptor, public Interface::Operator<Number>
{
public:
  enum class DofHandlerSelector
  {
    all_components = 0,
    vector         = 1,
    scalar         = 2,
    n_variants     = scalar + 1
  };

  enum class QuadratureSelector
  {
    overintegration_conv = 0,
    overintegration_vis  = 1,
    standard             = 2,
    n_variants           = standard + 1
  };

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef PostProcessorBase<dim, Number> Postprocessor;

  static const unsigned int dof_index_all =
    static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
      DofHandlerSelector::all_components);
  static const unsigned int dof_index_vector =
    static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
      DofHandlerSelector::vector);
  static const unsigned int dof_index_scalar =
    static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
      DofHandlerSelector::scalar);

  static const unsigned int quad_index_standard =
    static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::standard);
  static const unsigned int quad_index_overintegration_conv =
    static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::overintegration_conv);
  static const unsigned int quad_index_overintegration_vis =
    static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::overintegration_vis);

  // specify quadrature rule for calculation of derived quantities (p, u, T)
  static const unsigned int quad_index_l2_projections = quad_index_standard;

  // alternative: use more accurate over-integration strategy
  //  static const unsigned int quad_index_l2_projections = quad_index_overintegration_conv;

  DGOperator(parallel::TriangulationBase<dim> const & triangulation,
             InputParameters const &              param_in,
             std::shared_ptr<Postprocessor>       postprocessor_in);

  void
  setup(std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_density_in,
        std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_velocity_in,
        std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_pressure_in,
        std::shared_ptr<BoundaryDescriptorEnergy<dim>> boundary_descriptor_energy_in,
        std::shared_ptr<FieldFunctions<dim>>           field_functions_in);

  types::global_dof_index
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
   *  and finally applies the inverse mass matrix operator.
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
  MatrixFree<dim, Number> const &
  get_matrix_free() const;

  Mapping<dim> const &
  get_mapping() const;

  FESystem<dim> const &
  get_fe() const;

  DoFHandler<dim> const &
  get_dof_handler() const;

  DoFHandler<dim> const &
  get_dof_handler_scalar() const;

  DoFHandler<dim> const &
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

  double
  get_wall_time_operator_evaluation() const;

  void
  do_postprocessing(VectorType const & solution,
                    double const       time,
                    int const          time_step_number) const;

  double
  calculate_minimum_element_length() const;

  unsigned int
  get_polynomial_degree() const;

private:
  void
  create_dofs();

  void
  initialize_matrix_free();

  void
  setup_operators();

  void
  setup_postprocessor();

  // Input parameters
  InputParameters const & param;

  // finite element
  std::shared_ptr<FESystem<dim>> fe;        // all (dim+2) components: (rho, rho u, rho E)
  std::shared_ptr<FESystem<dim>> fe_vector; // e.g. velocity
  FE_DGQ<dim>                    fe_scalar; // scalar quantity, e.g, pressure

  // mapping
  unsigned int                          mapping_degree;
  std::shared_ptr<MappingQGeneric<dim>> mapping;

  // Quadrature points
  unsigned int n_q_points_conv;
  unsigned int n_q_points_visc;

  // DoFHandler
  DoFHandler<dim> dof_handler;        // all (dim+2) components: (rho, rho u, rho E)
  DoFHandler<dim> dof_handler_vector; // e.g. velocity
  DoFHandler<dim> dof_handler_scalar; // scalar quantity, e.g, pressure

  MatrixFree<dim, Number> matrix_free;

  std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_density;
  std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_velocity;
  std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_pressure;
  std::shared_ptr<BoundaryDescriptorEnergy<dim>> boundary_descriptor_energy;
  std::shared_ptr<FieldFunctions<dim>>           field_functions;

  // DG operators

  MassMatrixOperator<dim, Number> mass_matrix_operator;
  BodyForceOperator<dim, Number>  body_force_operator;
  ConvectiveOperator<dim, Number> convective_operator;
  ViscousOperator<dim, Number>    viscous_operator;
  CombinedOperator<dim, Number>   combined_operator;

  InverseMassMatrixOperator<dim, dim + 2, Number> inverse_mass_all;
  InverseMassMatrixOperator<dim, dim, Number>     inverse_mass_vector;
  InverseMassMatrixOperator<dim, 1, Number>       inverse_mass_scalar;

  // L2 projections to calculate derived quantities
  p_u_T_Calculator<dim, Number>     p_u_T_calculator;
  VorticityCalculator<dim, Number>  vorticity_calculator;
  DivergenceCalculator<dim, Number> divergence_calculator;

  // postprocessor
  std::shared_ptr<Postprocessor> postprocessor;

  // wall time for operator evaluation
  mutable double wall_time_operator_evaluation;
};

} // namespace CompNS

#endif /* INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_ */
