/*
 * streamfunction_calculator_rhs_operator.h
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_STREAMFUNCTION_CALCULATOR_RHS_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_STREAMFUNCTION_CALCULATOR_RHS_OPERATOR_H_

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>

using namespace dealii;

namespace IncNS
{
/*
 *  This function calculates the right-hand side of the Laplace equation that is solved in order to
 * obtain the streamfunction psi
 *
 *    - laplace(psi) = omega  (where omega is the vorticity).
 *
 *  Note that this function can only be used for the two-dimensional case (dim==2).
 */
template<int dim, int degree, typename Number>
class StreamfunctionCalculatorRHSOperator
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef StreamfunctionCalculatorRHSOperator<dim, degree, Number> This;

  typedef FEEvaluation<dim, degree, degree + 1, dim, Number> FEEval;
  typedef FEEvaluation<dim, degree, degree + 1, 1, Number>   FEEvalScalar;

public:
  StreamfunctionCalculatorRHSOperator();

  void
  initialize(MatrixFree<dim, Number> const & data_in,
             unsigned int const              dof_index_u_in,
             unsigned int const              dof_index_u_scalar_in,
             unsigned int const              quad_index_in);

  void
  apply(VectorType & dst, VectorType const & src) const;

private:
  void
  cell_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const;

  MatrixFree<dim, Number> const * data;

  unsigned int dof_index_u;
  unsigned int dof_index_u_scalar;
  unsigned int quad_index;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_STREAMFUNCTION_CALCULATOR_RHS_OPERATOR_H_ \
        */
