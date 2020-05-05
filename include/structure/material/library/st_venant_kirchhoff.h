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
  StVenantKirchhoff(StVenantKirchhoffData const & data);

  Tensor<2, dim, VectorizedArray<Number>>
    evaluate_stress(Tensor<2, dim, VectorizedArray<Number>> const & E) const;

  Tensor<2, dim, VectorizedArray<Number>>
    apply_C(Tensor<2, dim, VectorizedArray<Number>> const & E) const;

private:
  VectorizedArray<Number> f0;
  VectorizedArray<Number> f1;
  VectorizedArray<Number> f2;
};
} // namespace Structure

#endif
