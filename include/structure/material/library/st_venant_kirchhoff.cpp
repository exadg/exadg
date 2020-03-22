/*
 * st_venant_kirchhoff.cpp
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#include "st_venant_kirchhoff.h"

namespace Structure
{
template<int dim, typename Number>
StVenantKirchhoff<dim, Number>::StVenantKirchhoff(StVenantKirchhoffData const & data)
{
  Number const E            = data.E;
  Number const nu           = data.nu;
  Type2D const type_two_dim = data.type_two_dim;

  f0 = make_vectorized_array<Number>(dim == 3 ? E * (1 - nu) / (1 + nu) / (1 - 2 * nu) :
                                                (type_two_dim == Type2D::PlainStress ?
                                                   E * (1) / (1 - nu * nu) :
                                                   E * (1 - nu) / (1 + nu) / (1 - 2 * nu)));

  f1 = make_vectorized_array<Number>(dim == 3 ? E * (nu) / (1 + nu) / (1 - 2 * nu) :
                                                (type_two_dim == Type2D::PlainStress ?
                                                   E * (nu) / (1 - nu * nu) :
                                                   E * (nu) / (1 + nu) / (1 - 2 * nu)));

  f2 = make_vectorized_array<Number>(dim == 3 ? E * (1 - 2 * nu) / 2 / (1 + nu) / (1 - 2 * nu) :
                                                (type_two_dim == Type2D::PlainStress ?
                                                   E * (1 - nu) / 2 / (1 - nu * nu) :
                                                   E * (1 - 2 * nu) / 2 / (1 + nu) / (1 - 2 * nu)));

  for(int i = 0; i < dim; i++)
    for(int j = 0; j < dim; j++)
      if(i == j)
        C[i][j] = f0;
      else
        C[i][j] = f1;

  for(int i = dim; i < s; i++)
    C[i][i] = f2;
}

template<int dim, typename Number>
void
StVenantKirchhoff<dim, Number>::reinit(
  const Tensor<1, StVenantKirchhoff<dim, Number>::s, VectorizedArray<Number>> & E) const
{
  this->E = E;
}

template<int dim, typename Number>
Tensor<1, StVenantKirchhoff<dim, Number>::s, VectorizedArray<Number>>
StVenantKirchhoff<dim, Number>::get_S() const
{
#ifdef false
  const VectorizedArray<Number> f0 = this->f0;
  const VectorizedArray<Number> f1 = this->f1;
  const VectorizedArray<Number> f2 = this->f2;

  Tensor<1, s, VectorizedArray<Number>> vec_out;

  if(dim == 3)
  {
    vec_out[0] = f0 * E[0] + f1 * E[1] + f1 * E[2];
    vec_out[1] = f1 * E[0] + f0 * E[1] + f1 * E[2];
    vec_out[2] = f1 * E[0] + f1 * E[1] + f0 * E[2];
    vec_out[3] = f2 * E[3];
    vec_out[4] = f2 * E[4];
    vec_out[5] = f2 * E[5];
  }
  else
  {
    vec_out[0] = f0 * E[0] + f1 * E[1];
    vec_out[1] = f1 * E[0] + f0 * E[1];
    vec_out[2] = f2 * E[2];
  }

  return vec_out;
#else
  return C * E;
#endif
}

template<int dim, typename Number>

const Tensor<2, StVenantKirchhoff<dim, Number>::s, VectorizedArray<Number>> &
StVenantKirchhoff<dim, Number>::get_dSdE() const
{
  return C;
}

template class StVenantKirchhoff<2, float>;
template class StVenantKirchhoff<2, double>;

template class StVenantKirchhoff<3, float>;
template class StVenantKirchhoff<3, double>;

} // namespace Structure
