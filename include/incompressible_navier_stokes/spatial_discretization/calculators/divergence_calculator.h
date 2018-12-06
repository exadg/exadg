/*
 * divergence_calculator.h
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_DIVERGENCE_CALCULATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_DIVERGENCE_CALCULATOR_H_

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>

using namespace dealii;

namespace IncNS
{
template<int dim, int degree, typename Number>
class DivergenceCalculator
{
private:
  typedef DivergenceCalculator<dim, degree, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number> scalar;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef FEEvaluation<dim, degree, degree + 1, dim, Number> FEEval;
  typedef FEEvaluation<dim, degree, degree + 1, 1, Number>   FEEvalScalar;

public:
  DivergenceCalculator();

  void
  initialize(MatrixFree<dim, Number> const & data_in,
             unsigned int const              dof_index_u_in,
             unsigned int const              dof_index_u_scalar_in,
             unsigned int const              quad_index_in);

  void
  compute_divergence(VectorType & dst, VectorType const & src) const;

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

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_DIVERGENCE_CALCULATOR_H_ \
        */
