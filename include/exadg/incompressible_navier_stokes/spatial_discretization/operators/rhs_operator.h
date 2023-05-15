/*
 * body_force_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_RHS_OPERATOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_RHS_OPERATOR_H_

#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/mapping_flags.h>

namespace ExaDG
{
namespace IncNS
{
namespace Operators
{
template<int dim>
struct RHSKernelData
{
  RHSKernelData()
    : boussinesq_term(false),
      boussinesq_dynamic_part_only(false),
      thermal_expansion_coefficient(1.0),
      reference_temperature(0.0)
  {
  }

  std::shared_ptr<dealii::Function<dim>> f;

  // Boussinesq term
  bool                                   boussinesq_term;
  bool                                   boussinesq_dynamic_part_only;
  double                                 thermal_expansion_coefficient;
  double                                 reference_temperature;
  std::shared_ptr<dealii::Function<dim>> gravitational_force;
};

template<int dim, typename Number>
class RHSKernel
{
private:
  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;

  typedef CellIntegrator<dim, dim, Number> Integrator;
  typedef CellIntegrator<dim, 1, Number>   IntegratorScalar;

public:
  void
  reinit(RHSKernelData<dim> const & data_in) const
  {
    data = data_in;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells = dealii::update_JxW_values |
                  dealii::update_quadrature_points; // q-points due to rhs function f

    // no face integrals

    return flags;
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux(Integrator const &       integrator,
                    IntegratorScalar const & integrator_temperature,
                    unsigned int const       q,
                    Number const &           time) const
  {
    dealii::Point<dim, scalar> q_points = integrator.quadrature_point(q);

    vector f = FunctionEvaluator<1, dim, Number>::value(*(data.f), q_points, time);

    if(data.boussinesq_term)
    {
      vector g =
        FunctionEvaluator<1, dim, Number>::value(*(data.gravitational_force), q_points, time);
      scalar T     = integrator_temperature.get_value(q);
      scalar T_ref = data.reference_temperature;
      // solve only for the dynamic pressure variations
      if(data.boussinesq_dynamic_part_only)
        f += g * (-data.thermal_expansion_coefficient * (T - T_ref));
      else // includes hydrostatic component
        f += g * (1.0 - data.thermal_expansion_coefficient * (T - T_ref));
    }

    return f;
  }

private:
  mutable RHSKernelData<dim> data;
};

} // namespace Operators


template<int dim>
struct RHSOperatorData
{
  RHSOperatorData() : dof_index(0), quad_index(0), dof_index_scalar(2)
  {
  }

  unsigned int dof_index;
  unsigned int quad_index;

  unsigned int dof_index_scalar;

  Operators::RHSKernelData<dim> kernel_data;
};

template<int dim, typename Number>
class RHSOperator
{
public:
  typedef RHSOperator<dim, Number> This;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> Integrator;

  typedef CellIntegrator<dim, 1, Number> IntegratorScalar;

  RHSOperator();

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free,
             RHSOperatorData<dim> const &            data);

  void
  evaluate(VectorType & dst, Number const evaluation_time) const;

  void
  evaluate_add(VectorType & dst, Number const evaluation_time) const;

  void
  set_temperature(VectorType const & T);

private:
  void
  do_cell_integral(Integrator & integrator, IntegratorScalar & integrator_temperature) const;

  void
  cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst,
            VectorType const &                      src,
            Range const &                           cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  RHSOperatorData<dim> data;

  mutable double time;

  Operators::RHSKernel<dim, Number> kernel;

  VectorType const * temperature;
};

} // namespace IncNS
} // namespace ExaDG


#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_RHS_OPERATOR_H_ \
        */
