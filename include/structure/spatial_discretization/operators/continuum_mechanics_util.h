/*
 * continuum_mechanics_util.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_CONTINUUM_MECHANICS_UTIL_H_
#define INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_CONTINUUM_MECHANICS_UTIL_H_

// deal.II
#include <deal.II/base/derivative_form.h>

#include "tensor_util.h"

using namespace dealii;

namespace Structure
{
template<int dim>
struct Info
{
  static constexpr unsigned int n_stress_components = dim == 1 ? 1 : (dim == 2 ? 3 : 6);
};

template<int dim, typename Number>
Tensor<1, Info<dim>::n_stress_components, VectorizedArray<Number>>
  tensor_to_vector(Tensor<2, dim, VectorizedArray<Number>> gradient_in)
{
  if(dim == 2)
  {
    Tensor<1, Info<dim>::n_stress_components, VectorizedArray<Number>> vector_in;
    vector_in[0] = gradient_in[0][0];
    vector_in[1] = gradient_in[1][1];
    vector_in[2] = gradient_in[1][0] + gradient_in[0][1];
    return vector_in;
  }
  else // dim==3
  {
    Tensor<1, Info<dim>::n_stress_components, VectorizedArray<Number>> vector_in;
    vector_in[0] = gradient_in[0][0];
    vector_in[1] = gradient_in[1][1];
    vector_in[2] = gradient_in[2][2];
    vector_in[3] = gradient_in[0][1] + gradient_in[1][0];
    vector_in[4] = gradient_in[1][2] + gradient_in[2][1];
    vector_in[5] = gradient_in[0][2] + gradient_in[2][0];
    return vector_in;
  }
}

template<int dim, typename Number>
Tensor<1, Info<dim>::n_stress_components, VectorizedArray<Number>>
  symmetrize(Tensor<2, dim, VectorizedArray<Number>> gradient_in)
{
  if(dim == 2)
  {
    Tensor<1, Info<dim>::n_stress_components, VectorizedArray<Number>> vector_in;
    vector_in[0] = gradient_in[0][0];
    vector_in[1] = gradient_in[1][1];
    vector_in[2] = (gradient_in[1][0] + gradient_in[0][1]) * 0.5;
    return vector_in;
  }
  else // dim==3
  {
    Tensor<1, Info<dim>::n_stress_components, VectorizedArray<Number>> vector_in;
    vector_in[0] = gradient_in[0][0];
    vector_in[1] = gradient_in[1][1];
    vector_in[2] = gradient_in[2][2];
    vector_in[3] = (gradient_in[0][1] + gradient_in[1][0]) * 0.5;
    vector_in[4] = (gradient_in[1][2] + gradient_in[2][1]) * 0.5;
    vector_in[5] = (gradient_in[0][2] + gradient_in[2][0]) * 0.5;
    return vector_in;
  }
}

template<int dim, typename Number>
Tensor<2, dim, VectorizedArray<Number>>
  vector_to_tensor(Tensor<1, Info<dim>::n_stress_components, VectorizedArray<Number>> vector_out)
{
  if(dim == 2)
  {
    Tensor<2, dim, VectorizedArray<Number>> gradient_out;
    gradient_out[0][0] = vector_out[0];
    gradient_out[1][1] = vector_out[1];

    gradient_out[0][1] = vector_out[2];
    gradient_out[1][0] = vector_out[2];
    return gradient_out;
  }
  else // dim==3
  {
    Tensor<2, dim, VectorizedArray<Number>> gradient_out;
    gradient_out[0][0] = vector_out[0];
    gradient_out[1][1] = vector_out[1];
    gradient_out[2][2] = vector_out[2];

    gradient_out[0][1] = vector_out[3];
    gradient_out[1][0] = vector_out[3];

    gradient_out[1][2] = vector_out[4];
    gradient_out[2][1] = vector_out[4];

    gradient_out[0][2] = vector_out[5];
    gradient_out[2][0] = vector_out[5];
    return gradient_out;
  }
}

/**
 * \brief computes the Cauchy stress tensor sigma from 2nd Piola-Kirchhoff stress S and the
 * deformation gradient F.
 */
template<int dim, typename Number>
inline const Tensor<1, Info<dim>::n_stress_components, VectorizedArray<Number>>
get_sigma(const Tensor<1, Info<dim>::n_stress_components, VectorizedArray<Number>> & S,
          const Tensor<2, dim, VectorizedArray<Number>> &                            F)
{
  // compute the determinant of the deformation gradient
  auto const det_F = determinant(F);

  // redo Voigt notation of S (now S is normal 2nd order tensor)
  auto const S_tensor = vector_to_tensor<dim, Number>(S);

  // compute Cauchy stresses
  auto const sigma_tensor = (F * S_tensor * transpose(F)) / det_F;

  // use Voigt notation on sigma (now sigma is a 1st order tensor)
  auto const sigma = tensor_to_vector<dim, Number>(sigma_tensor);

  // return sigma in Voigt notation
  return sigma;
}

template<int dim, typename Number>
Tensor<2, dim, VectorizedArray<Number>>
get_F(const Tensor<2, dim, VectorizedArray<Number>> & H)
{
  return add_identity(H);
}

template<int dim, typename Number>
Tensor<1, Info<dim>::n_stress_components, VectorizedArray<Number>>
get_E(const Tensor<2, dim, VectorizedArray<Number>> & F)
{
  return 0.5 * tensor_to_vector<dim, Number>(sub_identity(transpose(F) * F));
}

} // namespace Structure

#endif
