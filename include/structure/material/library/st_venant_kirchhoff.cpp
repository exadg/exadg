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
}

template<int dim, typename Number>
Tensor<2, dim, VectorizedArray<Number>> StVenantKirchhoff<dim, Number>::evaluate_stress(
  Tensor<2, dim, VectorizedArray<Number>> const & E) const
{
  Tensor<2, dim, VectorizedArray<Number>> S;

  if(dim == 3)
  {
    S[0][0] = f0 * E[0][0] + f1 * E[1][1] + f1 * E[2][2];
    S[1][1] = f1 * E[0][0] + f0 * E[1][1] + f1 * E[2][2];
    S[2][2] = f1 * E[0][0] + f1 * E[1][1] + f0 * E[2][2];
    S[0][1] = f2 * (E[0][1] + E[1][0]);
    S[1][2] = f2 * (E[1][2] + E[2][1]);
    S[0][2] = f2 * (E[0][2] + E[2][0]);
    S[1][0] = f2 * (E[0][1] + E[1][0]);
    S[2][1] = f2 * (E[1][2] + E[2][1]);
    S[2][0] = f2 * (E[0][2] + E[2][0]);
  }
  else
  {
    S[0][0] = f0 * E[0][0] + f1 * E[1][1];
    S[1][1] = f1 * E[0][0] + f0 * E[1][1];
    S[0][1] = f2 * (E[0][1] + E[1][0]);
    S[1][0] = f2 * (E[0][1] + E[1][0]);
  }

  return S;
}

template<int dim, typename Number>
Tensor<2, dim, VectorizedArray<Number>>
  StVenantKirchhoff<dim, Number>::apply_C(Tensor<2, dim, VectorizedArray<Number>> const & E) const
{
  return evaluate_stress(E);
}

template class StVenantKirchhoff<2, float>;
template class StVenantKirchhoff<2, double>;

template class StVenantKirchhoff<3, float>;
template class StVenantKirchhoff<3, double>;

} // namespace Structure
