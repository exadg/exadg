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

#ifndef INCLUDE_OPERATORS_RHS_OPERATOR
#define INCLUDE_OPERATORS_RHS_OPERATOR

#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/mapping_flags.h>

namespace ExaDG
{
namespace Operators
{
template<int dim>
struct RHSKernelData
{
  std::shared_ptr<dealii::Function<dim>> f;
};

template<int dim, typename Number, int n_components = 1>
class RHSKernel
{
private:
  typedef CellIntegrator<dim, n_components, Number> IntegratorCell;

  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : dealii::numbers::invalid_unsigned_int);

  typedef dealii::VectorizedArray<Number>   scalar;
  typedef dealii::Tensor<rank, dim, scalar> value;

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
    value
    get_volume_flux(IntegratorCell const & integrator,
                    unsigned int const     q,
                    Number const &         time) const
  {
    dealii::Point<dim, scalar> q_points = integrator.quadrature_point(q);

    return FunctionEvaluator<rank, dim, Number>::value(*(data.f), q_points, time);
  }

private:
  mutable RHSKernelData<dim> data;
};

} // namespace Operators


template<int dim>
struct RHSOperatorData
{
  RHSOperatorData() : dof_index(0), quad_index(0)
  {
  }

  unsigned int dof_index;
  unsigned int quad_index;

  Operators::RHSKernelData<dim> kernel_data;
};

template<int dim, typename Number, int n_components = 1>
class RHSOperator
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef RHSOperator<dim, Number, n_components> This;

  typedef CellIntegrator<dim, n_components, Number> IntegratorCell;

  typedef std::pair<unsigned int, unsigned int> Range;

public:
  /*
   * Constructor.
   */
  RHSOperator();

  /*
   * Initialization.
   */
  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free,
             RHSOperatorData<dim> const &            data);

  /*
   * Evaluate operator and overwrite dst-vector.
   */
  void
  evaluate(VectorType & dst, double const evaluation_time) const;

  /*
   * Evaluate operator and add to dst-vector.
   */
  void
  evaluate_add(VectorType & dst, double const evaluation_time) const;

private:
  void
  do_cell_integral(IntegratorCell & integrator) const;

  /*
   * The right-hand side operator involves only cell integrals so we only need a function looping
   * over all cells and computing the cell integrals.
   */
  void
  cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst,
            VectorType const &                      src,
            Range const &                           cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  RHSOperatorData<dim> data;

  mutable double time;

  Operators::RHSKernel<dim, Number, n_components> kernel;
};

} // namespace ExaDG

#endif
