/*
 * curl_compute.h
 *
 *  Created on: Jun 28, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CURL_COMPUTE_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CURL_COMPUTE_H_

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim, typename FEEval>
struct CurlCompute
{
  static Tensor<1, dim, VectorizedArray<typename FEEval::number_type>>
  compute(FEEval & fe_eval, const unsigned int q_point)
  {
    return fe_eval.get_curl(q_point);
  }
};

// use partial specialization of templates to handle the 2-dimensional case
template<typename FEEval>
struct CurlCompute<2, FEEval>
{
  static Tensor<1, 2, VectorizedArray<typename FEEval::number_type>>
  compute(FEEval & fe_eval, const unsigned int q_point)
  {
    Tensor<1, 2, VectorizedArray<typename FEEval::number_type>> curl;
    /*
     * fe_eval = / phi \   _____\   grad(fe_eval) = / d(phi)/dx1   d(phi)/dx2 \
     *           \  0  /        /                   \     0             0     /
     */
    Tensor<2, 2, VectorizedArray<typename FEEval::number_type>> temp =
      fe_eval.get_gradient(q_point);
    /*
     *         __    /  0  \     /   d(phi)/dx2 \
     *  curl = \/ X  |  0  |  =  | - d(phi)/dx1 |
     *               \ phi /     \      0       /
     */
    curl[0] = temp[0][1];  //   d(phi)/dx2
    curl[1] = -temp[0][0]; // - d(phi)/dx1
    return curl;
  }
};

} // namespace IncNS
} // namespace ExaDG


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CURL_COMPUTE_H_ */
