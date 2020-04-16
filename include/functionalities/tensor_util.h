/*
 * tensor_util.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_TENSOR_UTIL_H_
#define INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_TENSOR_UTIL_H_

// deal.II
#include <deal.II/base/tensor.h>
#include <deal.II/physics/transformations.h>

using namespace dealii;

template<int dim, typename Number = double>
Tensor<2, dim, VectorizedArray<Number>>
  add_identity(Tensor<2, dim, VectorizedArray<Number>> gradient)
{
  for(unsigned int i = 0; i < dim; i++)
    gradient[i][i] = gradient[i][i] + 1.0;
  return gradient;
}

template<int dim, typename Number = double>
Tensor<2, dim, VectorizedArray<Number>>
  subtract_identity(Tensor<2, dim, VectorizedArray<Number>> gradient)
{
  for(unsigned int i = 0; i < dim; i++)
    gradient[i][i] = gradient[i][i] - 1.0;
  return gradient;
}


template<typename Number = double>
Tensor<2, 2, Number>
get_rotation_matrix(const Tensor<2, 2> grad_u)
{
  const double curl  = (grad_u[1][0] - grad_u[0][1]);
  const double angle = std::atan(curl);
  return Physics::Transformations::Rotations::rotation_matrix_2d(-angle);
}

template<typename Number = double>
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

#endif
