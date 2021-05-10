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

#ifndef APPLICATIONS_PERIODIC_HILL_MANIFOLD_H_
#define APPLICATIONS_PERIODIC_HILL_MANIFOLD_H_

namespace ExaDG
{
using namespace dealii;

double
m_to_mm(double coordinate)
{
  return 1000.0 * coordinate;
}

double
mm_to_m(double coordinate)
{
  return 0.001 * coordinate;
}

/*
 * This function returns the distance by which points are shifted in y-direction due to the hill,
 * i.e., a value of 0 is returned at x=0H, 9H, and a value of -H at x=4.5H.
 */
double
f(double x_m, double const H, double const LENGTH)
{
  if(x_m > LENGTH / 2.0)
    x_m = LENGTH - x_m;

  double x = m_to_mm(x_m);
  double y = 0.0;

  AssertThrow(x_m <= LENGTH / 2.0 + 1.e-12, ExcMessage("Parameter out of bounds."));

  if(x <= 9.0)
    y =
      -m_to_mm(H) +
      std::min(m_to_mm(H), m_to_mm(H) + 6.775070969851e-3 * x * x - 2.124527775800e-3 * x * x * x);
  else if(x > 9.0 && x <= 14.0)
    y = -m_to_mm(H) + 2.507355893131e1 + 9.754803562315e-1 * x - 1.016116352781e-1 * x * x +
        1.889794677828e-3 * x * x * x;
  else if(x > 14.0 && x <= 20.0)
    y = -m_to_mm(H) + 2.579601052357e1 + 8.206693007457e-1 * x - 9.055370274339e-2 * x * x +
        1.626510569859e-3 * x * x * x;
  else if(x > 20.0 && x <= 30.0)
    y = -m_to_mm(H) + 4.046435022819e1 - 1.379581654948 * x + 1.945884504128e-2 * x * x -
        2.070318932190e-4 * x * x * x;
  else if(x > 30.0 && x <= 40.0)
    y = -m_to_mm(H) + 1.792461334664e1 + 8.743920332081e-1 * x - 5.567361123058e-2 * x * x +
        6.277731764683e-4 * x * x * x;
  else if(x > 40.0 && x <= 54.0)
    y = -m_to_mm(H) + std::max(0.0,
                               5.639011190988e1 - 2.010520359035 * x + 1.644919857549e-2 * x * x +
                                 2.674976141766e-5 * x * x * x);
  else if(x > 54.0)
    y = -m_to_mm(H);
  else
    AssertThrow(false, ExcMessage("Not implemented."));

  return mm_to_m(y);
}

template<int dim>
class PeriodicHillManifold : public ChartManifold<dim>
{
public:
  PeriodicHillManifold(double const H,
                       double const LENGTH,
                       double const HEIGHT,
                       double const GRID_STRETCH_FAC)
    : ChartManifold<dim>(), H(H), LENGTH(LENGTH), HEIGHT(HEIGHT), GRID_STRETCH_FAC(GRID_STRETCH_FAC)
  {
  }

  Point<dim>
  push_forward(Point<dim> const & xi) const override
  {
    Point<dim> x = xi;

    // transform y-coordinate only
    double const gamma           = GRID_STRETCH_FAC;
    double const xi_1_normalized = (xi[1] - H) / HEIGHT;
    double       xi_1_hat = std::tanh(gamma * (2.0 * xi_1_normalized - 1.0)) / std::tanh(gamma);
    x[1] = H + (xi_1_hat + 1.0) / 2.0 * HEIGHT + (1.0 - xi_1_hat) / 2.0 * f(xi[0], H, LENGTH);

    return x;
  }

  Point<dim>
  pull_back(Point<dim> const & x) const override
  {
    Point<dim> xi = x;

    // transform y-coordinate only
    double const f_x      = f(x[0], H, LENGTH);
    double const xi_1_hat = (2.0 * x[1] - 2.0 * H - HEIGHT - f_x) / (HEIGHT - f_x);
    double const gamma    = GRID_STRETCH_FAC;
    xi[1] = HEIGHT / 2.0 * (std::atanh(xi_1_hat * std::tanh(gamma)) / gamma + 1) + H;

    return xi;
  }

  std::unique_ptr<Manifold<dim>>
  clone() const override
  {
    return std::make_unique<PeriodicHillManifold<dim>>(H, LENGTH, HEIGHT, GRID_STRETCH_FAC);
  }

private:
  double const H, LENGTH, HEIGHT, GRID_STRETCH_FAC;
};

} // namespace ExaDG


#endif /* APPLICATIONS_PERIODIC_HILL_MANIFOLD_H_ */
