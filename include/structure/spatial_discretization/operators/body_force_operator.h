/*
 * body_force_operator.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_RHS_OPERATOR_H_
#define INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_RHS_OPERATOR_H_

// deal.II
#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "../../../operators/mapping_flags.h"

using namespace dealii;

namespace Structure
{
template<int dim>
struct BodyForceData
{
  BodyForceData() : dof_index(0), quad_index(0), pull_back_body_force(false)
  {
  }

  unsigned int dof_index;
  unsigned int quad_index;

  std::shared_ptr<Function<dim>> function;

  bool pull_back_body_force;
};

template<int dim, typename Number>
class BodyForceOperator
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef BodyForceOperator<dim, Number> This;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> IntegratorCell;

public:
  /*
   * Constructor.
   */
  BodyForceOperator();

  /*
   * Initialization.
   */
  void
  initialize(MatrixFree<dim, Number> const & matrix_free, BodyForceData<dim> const & data);

  static MappingFlags
  get_mapping_flags();

  /*
   * Evaluate operator and add to dst-vector.
   */
  void
  evaluate_add(VectorType & dst, VectorType const & src, double const time) const;

private:
  void
  cell_loop(MatrixFree<dim, Number> const & matrix_free,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const;

  MatrixFree<dim, Number> const * matrix_free;

  BodyForceData<dim> data;

  double mutable time;
};

} // namespace Structure

#endif
