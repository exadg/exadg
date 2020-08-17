/*
 * linear_interpolation.h
 *
 *  Created on: May 18, 2019
 *      Author: fehn
 */

// deal.II
#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, typename Number>
Number
linear_interpolation_1d(double const &                              y,
                        std::vector<Number> const &                 y_values,
                        std::vector<Tensor<1, dim, Number>> const & solution_values,
                        unsigned int const &                        component);

/*
 *  2D interpolation for rectangular cross-sections
 */
template<int dim, typename Number>
Number
linear_interpolation_2d_cartesian(Point<dim> const &                          point,
                                  std::vector<Number> const &                 y_values,
                                  std::vector<Number> const &                 z_values,
                                  std::vector<Tensor<1, dim, Number>> const & solution_values,
                                  unsigned int const &                        component);

/*
 *  2D interpolation for cylindrical cross-sections
 */
template<int dim, typename Number>
Number
linear_interpolation_2d_cylindrical(Number const                                r_in,
                                    Number const                                phi,
                                    std::vector<Number> const &                 r_values,
                                    std::vector<Number> const &                 phi_values,
                                    std::vector<Tensor<1, dim, Number>> const & solution_values,
                                    unsigned int const &                        component);

} // namespace ExaDG
