/*
 * continuum_mechanics.h
 *
 *  Created on: 03.05.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_CONTINUUM_MECHANICS_H_
#define INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_CONTINUUM_MECHANICS_H_

// deal.II
#include <deal.II/base/tensor.h>
#include <deal.II/physics/transformations.h>

namespace Structure
{
using namespace dealii;

template<int dim>
struct Info
{
  static constexpr unsigned int n_stress_components = dim * (dim + 1) / 2;
};

template<int dim, typename Number = double>
inline DEAL_II_ALWAYS_INLINE //
    Tensor<2, dim, VectorizedArray<Number>>
    add_identity(Tensor<2, dim, VectorizedArray<Number>> gradient)
{
  for(unsigned int i = 0; i < dim; i++)
    gradient[i][i] = gradient[i][i] + 1.0;
  return gradient;
}

template<int dim, typename Number = double>
inline DEAL_II_ALWAYS_INLINE //
    Tensor<2, dim, VectorizedArray<Number>>
    subtract_identity(Tensor<2, dim, VectorizedArray<Number>> gradient)
{
  for(unsigned int i = 0; i < dim; i++)
    gradient[i][i] = gradient[i][i] - 1.0;
  return gradient;
}


template<typename Number = double>
inline DEAL_II_ALWAYS_INLINE //
  Tensor<2, 2, Number>
  get_rotation_matrix(const Tensor<2, 2> grad_u)
{
  const double curl  = (grad_u[1][0] - grad_u[0][1]);
  const double angle = std::atan(curl);
  return Physics::Transformations::Rotations::rotation_matrix_2d(-angle);
}

template<typename Number = double>
inline DEAL_II_ALWAYS_INLINE //
  Tensor<2, 3, Number>
  get_rotation_matrix(const Tensor<2, 3> & grad_u)
{
  const Point<3> curl(grad_u[2][1] - grad_u[1][2],
                      grad_u[0][2] - grad_u[2][0],
                      grad_u[1][0] - grad_u[0][1]);
  const double   tan_angle = std::sqrt(curl * curl);
  const double   angle     = std::atan(tan_angle);
  if(std::abs(angle) < 1e-9)
  {
    static const double       rotation[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    static const Tensor<2, 3> rot(rotation);
    return rot;
  }
  const Point<3> axis = curl / tan_angle;
  return Physics::Transformations::Rotations::rotation_matrix_3d(axis, -angle);
}

template<int dim, typename Number = double>
inline DEAL_II_ALWAYS_INLINE //
  Tensor<2, dim, VectorizedArray<Number>>
  get_rotation_matrix(const Tensor<2, dim, VectorizedArray<Number>> grad_u)
{
  Tensor<2, dim, VectorizedArray<Number>> result;
  for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; v++)
  {
    Tensor<2, dim, Number> temp;
    for(unsigned int i = 0; i < dim; i++)
      for(unsigned int j = 0; j < dim; j++)
        temp[i][j] = grad_u[i][j][v];

    temp = get_rotation_matrix(temp);


    for(unsigned int i = 0; i < dim; i++)
      for(unsigned int j = 0; j < dim; j++)
        result[i][j][v] = temp[i][j];
  }
  return result;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
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
inline DEAL_II_ALWAYS_INLINE //
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
inline DEAL_II_ALWAYS_INLINE //
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
 * This function computes the Cauchy stress tensor sigma from 2nd Piola-Kirchhoff stress S and the
 * deformation gradient F. This function is required for an updated Lagrangian formulation.
 */
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  Tensor<1, Info<dim>::n_stress_components, VectorizedArray<Number>>
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
inline DEAL_II_ALWAYS_INLINE //
  Tensor<2, dim, VectorizedArray<Number>>
  get_F(const Tensor<2, dim, VectorizedArray<Number>> & H)
{
  return add_identity(H);
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  Tensor<1, Info<dim>::n_stress_components, VectorizedArray<Number>>
  get_E(const Tensor<2, dim, VectorizedArray<Number>> & F)
{
  return 0.5 * tensor_to_vector<dim, Number>(subtract_identity(transpose(F) * F));
}
} // namespace Structure



#endif /* INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_CONTINUUM_MECHANICS_H_ */
