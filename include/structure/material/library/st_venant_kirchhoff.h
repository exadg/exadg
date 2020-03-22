/*
 * st_venant_kirchhoff.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef STRUCTURE_MATERIAL_LIBRARY_STVENANTKIRCHHOFF
#define STRUCTURE_MATERIAL_LIBRARY_STVENANTKIRCHHOFF

#include "../material.h"

namespace Structure
{
struct StVenantKirchhoffData : public MaterialData
{
  StVenantKirchhoffData(MaterialType const & type,
                        double const &       E,
                        double const &       nu,
                        Type2D const &       type_two_dim)
    : MaterialData(type), E(E), nu(nu), type_two_dim(type_two_dim)
  {
  }

  double E;
  double nu;
  Type2D type_two_dim;
};

template<int dim, typename Number>
class StVenantKirchhoff : public Material<dim, Number>
{
public:
  static const int s = (dim == 2 ? 3 : 6);

  StVenantKirchhoff(StVenantKirchhoffData const & data);

  void
  reinit(const Tensor<1, s, VectorizedArray<Number>> & vec_in) const;

  Tensor<1, s, VectorizedArray<Number>>
  get_S() const;

  const Tensor<2, s, VectorizedArray<Number>> &
  get_dSdE() const;

private:
  VectorizedArray<Number> f0;
  VectorizedArray<Number> f1;
  VectorizedArray<Number> f2;

  mutable Tensor<2, s, VectorizedArray<Number>> C;
  mutable Tensor<1, s, VectorizedArray<Number>> E;
};
} // namespace Structure

#endif
