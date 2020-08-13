/*
 * streamfunction_calculator_rhs_operator.h
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_STREAMFUNCTION_CALCULATOR_RHS_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_STREAMFUNCTION_CALCULATOR_RHS_OPERATOR_H_

#include <deal.II/lac/la_parallel_vector.h>
#include "../../../matrix_free/integrators.h"

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

/*
 *  This function calculates the right-hand side of the Laplace equation that is solved in order to
 * obtain the streamfunction psi
 *
 *    - laplace(psi) = omega  (where omega is the vorticity).
 *
 *  Note that this function can only be used for the two-dimensional case (dim==2).
 */
template<int dim, typename Number>
class StreamfunctionCalculatorRHSOperator
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef StreamfunctionCalculatorRHSOperator<dim, Number> This;

  typedef CellIntegrator<dim, dim, Number> IntegratorVector;
  typedef CellIntegrator<dim, 1, Number>   IntegratorScalar;

public:
  StreamfunctionCalculatorRHSOperator();

  void
  initialize(MatrixFree<dim, Number> const & matrix_free_in,
             unsigned int const              dof_index_u_in,
             unsigned int const              dof_index_u_scalar_in,
             unsigned int const              quad_index_in);

  void
  apply(VectorType & dst, VectorType const & src) const;

private:
  void
  cell_loop(MatrixFree<dim, Number> const & matrix_free,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const;

  MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index_u;
  unsigned int dof_index_u_scalar;
  unsigned int quad_index;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_STREAMFUNCTION_CALCULATOR_RHS_OPERATOR_H_ \
        */
