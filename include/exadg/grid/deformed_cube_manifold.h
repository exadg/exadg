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

#ifndef APPLICATIONS_GRID_TOOLS_DEFORMED_CUBE_MANIFOLD_H_
#define APPLICATIONS_GRID_TOOLS_DEFORMED_CUBE_MANIFOLD_H_

#include <deal.II/grid/manifold_lib.h>

namespace ExaDG
{
template<int dim>
class DeformedCubeManifold : public dealii::ChartManifold<dim, dim, dim>
{
public:
  DeformedCubeManifold(double const       left,
                       double const       right,
                       double const       deformation,
                       unsigned int const frequency = 1)
    : left(left), right(right), deformation(deformation), frequency(frequency)
  {
  }

  dealii::Point<dim>
  push_forward(dealii::Point<dim> const & chart_point) const override
  {
    double sinval = deformation;
    for(unsigned int d = 0; d < dim; ++d)
      sinval *=
        std::sin(frequency * dealii::numbers::PI * (chart_point(d) - left) / (right - left));
    dealii::Point<dim> space_point;
    for(unsigned int d = 0; d < dim; ++d)
      space_point(d) = chart_point(d) + sinval;
    return space_point;
  }

  dealii::Point<dim>
  pull_back(dealii::Point<dim> const & space_point) const override
  {
    dealii::Point<dim> x = space_point;
    dealii::Point<dim> one;
    for(unsigned int d = 0; d < dim; ++d)
      one(d) = 1.;

    // Newton iteration to solve the nonlinear equation given by the point
    dealii::Tensor<1, dim> sinvals;
    for(unsigned int d = 0; d < dim; ++d)
      sinvals[d] = std::sin(frequency * dealii::numbers::PI * (x(d) - left) / (right - left));

    double sinval = deformation;
    for(unsigned int d = 0; d < dim; ++d)
      sinval *= sinvals[d];
    dealii::Tensor<1, dim> residual = space_point - x - sinval * one;
    unsigned int           its      = 0;
    while(residual.norm() > 1e-12 && its < 100)
    {
      dealii::Tensor<2, dim> jacobian;
      for(unsigned int d = 0; d < dim; ++d)
        jacobian[d][d] = 1.;
      for(unsigned int d = 0; d < dim; ++d)
      {
        double sinval_der =
          deformation * frequency / (right - left) * dealii::numbers::PI *
          std::cos(frequency * dealii::numbers::PI * (x(d) - left) / (right - left));
        for(unsigned int e = 0; e < dim; ++e)
          if(e != d)
            sinval_der *= sinvals[e];
        for(unsigned int e = 0; e < dim; ++e)
          jacobian[e][d] += sinval_der;
      }

      x += invert(jacobian) * residual;

      for(unsigned int d = 0; d < dim; ++d)
        sinvals[d] = std::sin(frequency * dealii::numbers::PI * (x(d) - left) / (right - left));

      sinval = deformation;
      for(unsigned int d = 0; d < dim; ++d)
        sinval *= sinvals[d];
      residual = space_point - x - sinval * one;
      ++its;
    }
    AssertThrow(residual.norm() < 1e-12, dealii::ExcMessage("Newton for point did not converge."));
    return x;
  }

  std::unique_ptr<dealii::Manifold<dim>>
  clone() const override
  {
    return std::make_unique<DeformedCubeManifold<dim>>(left, right, deformation, frequency);
  }

private:
  double const       left;
  double const       right;
  double const       deformation;
  unsigned int const frequency;
};

} // namespace ExaDG

#endif /* APPLICATIONS_GRID_TOOLS_DEFORMED_CUBE_MANIFOLD_H_ */
