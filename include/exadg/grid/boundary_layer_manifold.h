/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_EXADG_GRID_BOUNDARY_LAYER_MANIFOLD_H_
#define INCLUDE_EXADG_GRID_BOUNDARY_LAYER_MANIFOLD_H_

#include <deal.II/grid/manifold_lib.h>

namespace ExaDG
{
/**
 * A "boundary-layer" manifold that corresponds to a triangulation generated via
 *
 *   dealii::GridGenerator::hyper_rectangle(tria,
 *                                          dealii::Point<dim>(-dimensions / 2.0),
 *                                          dealii::Point<dim>(dimensions / 2.0));
 *
 * where dealii::Tensor<1, dim> dimensions; describes the physical dimensions in the dim coordinate
 * directions.
 */
template<int dim>
class BoundaryLayerManifold : public dealii::ChartManifold<dim, dim, dim>
{
public:
  BoundaryLayerManifold(dealii::Tensor<1, dim> const & dimensions_in,
                        double const                   grid_stretch_factor_in)
  {
    dimensions          = dimensions_in;
    grid_stretch_factor = grid_stretch_factor_in;
  }

  /*
   *  push_forward operation that maps point xi in reference coordinates [0,1]^d to
   *  point x in physical coordinates
   */
  dealii::Point<dim>
  push_forward(dealii::Point<dim> const & xi) const final
  {
    dealii::Point<dim> x;

    x[0] = xi[0] * dimensions[0] - dimensions[0] / 2.0;
    x[1] = grid_transform_y(xi[1]);

    if(dim == 3)
      x[2] = xi[2] * dimensions[2] - dimensions[2] / 2.0;

    return x;
  }

  /*
   *  pull_back operation that maps point x in physical coordinates
   *  to point xi in reference coordinates [0,1]^d
   */
  dealii::Point<dim>
  pull_back(dealii::Point<dim> const & x) const final
  {
    dealii::Point<dim> xi;

    xi[0] = x[0] / dimensions[0] + 0.5;
    xi[1] = inverse_grid_transform_y(x[1]);

    if(dim == 3)
      xi[2] = x[2] / dimensions[2] + 0.5;

    return xi;
  }

  std::unique_ptr<dealii::Manifold<dim>>
  clone() const final
  {
    return std::make_unique<BoundaryLayerManifold<dim>>(dimensions, grid_stretch_factor);
  }

  /*
   *  maps eta in [0,1] --> y in [-1,1]*length_y/2.0 (using a hyperbolic mesh stretching)
   */
  double
  grid_transform_y(double const & eta) const
  {
    double y = 0.0;

    if(grid_stretch_factor >= 0)
      y = dimensions[1] / 2.0 * std::tanh(grid_stretch_factor * (2. * eta - 1.)) /
          std::tanh(grid_stretch_factor);
    else // use a negative grid_stretch_factorTOR deactivate grid stretching
      y = dimensions[1] / 2.0 * (2. * eta - 1.);

    return y;
  }

  /*
   * inverse mapping:
   *
   *  maps y in [-1,1]*length_y/2.0 --> eta in [0,1]
   */
  double
  inverse_grid_transform_y(double const & y) const
  {
    double eta = 0.0;

    if(grid_stretch_factor >= 0)
      eta = (std::atanh(y * std::tanh(grid_stretch_factor) * 2.0 / dimensions[1]) /
               grid_stretch_factor +
             1.0) /
            2.0;
    else // use a negative grid_stretch_factorTOR deactivate grid stretching
      eta = (2. * y / dimensions[1] + 1.) / 2.0;

    return eta;
  }

private:
  dealii::Tensor<1, dim> dimensions;

  // use a negative grid_stretch_factor to deactivate grid stretching
  double grid_stretch_factor = 1.8;
};
} // namespace ExaDG



#endif /* INCLUDE_EXADG_GRID_BOUNDARY_LAYER_MANIFOLD_H_ */
