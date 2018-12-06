/*
 * vorticity_calculator.h
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_VORTICITY_CALCULATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_VORTICITY_CALCULATOR_H_

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>

using namespace dealii;

namespace IncNS
{
template<int dim, int degree, typename Number>
class VorticityCalculator
{
private:
  typedef VorticityCalculator<dim, degree, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  static const unsigned int number_vorticity_components = (dim == 2) ? 1 : dim;

  typedef FEEvaluation<dim, degree, degree + 1, dim, Number> FEEval;

public:
  VorticityCalculator();

  void
  initialize(MatrixFree<dim, Number> const & data_in,
             unsigned int const              dof_index_in,
             unsigned int const              quad_index_in);

  void
  compute_vorticity(VectorType & dst, VectorType const & src) const;

private:
  void
  cell_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const;

  MatrixFree<dim, Number> const * data;

  unsigned int dof_index;
  unsigned int quad_index;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_VORTICITY_CALCULATOR_H_ \
        */
