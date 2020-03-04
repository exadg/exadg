/*
 * body_force_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_RHS_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_RHS_OPERATOR_H_

#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "../../../functionalities/evaluate_functions.h"
#include "../../../operators/mapping_flags.h"

using namespace dealii;

namespace IncNS
{
namespace Operators
{
template<int dim>
struct RHSKernelData
{
  RHSKernelData()
    : boussinesq_term(false), thermal_expansion_coefficient(1.0), reference_temperature(0.0)
  {
  }

  std::shared_ptr<Function<dim>> f;

  // Boussinesq term
  bool                           boussinesq_term;
  double                         thermal_expansion_coefficient;
  double                         reference_temperature;
  std::shared_ptr<Function<dim>> gravitational_force;
};

template<int dim, typename Number>
class RHSKernel
{
private:
  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

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

    flags.cells = update_JxW_values | update_quadrature_points; // q-points due to rhs function f

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
    Point<dim, scalar> q_points = integrator.quadrature_point(q);

    vector f = FunctionEvaluator<dim, Number, 1>::value(data.f, q_points, time);

    if(data.boussinesq_term)
    {
      vector g = FunctionEvaluator<dim, Number, 1>::value(data.gravitational_force, q_points, time);
      scalar T = integrator_temperature.get_value(q);
      scalar T_ref = data.reference_temperature;
      f += g * (1.0 - data.thermal_expansion_coefficient * (T - T_ref));
      // TODO: for the dual splitting scheme we observed in one example that it might be
      // advantageous to only solve for the dynamic pressure variations and drop the constant 1.0 in
      // the above term that leads to a static pressure, in order improve robustness of the
      // splitting scheme.
      //      f += g * (- data.thermal_expansion_coefficient * (T - T_ref));
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

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> Integrator;

  typedef CellIntegrator<dim, 1, Number> IntegratorScalar;

  RHSOperator();

  void
  reinit(MatrixFree<dim, Number> const & matrix_free_in, RHSOperatorData<dim> const & data_in);

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
  cell_loop(MatrixFree<dim, Number> const & matrix_free,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const;

  MatrixFree<dim, Number> const * matrix_free;

  RHSOperatorData<dim> data;

  mutable double time;

  Operators::RHSKernel<dim, Number> kernel;

  VectorType const * temperature;
};
} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_RHS_OPERATOR_H_ \
        */
