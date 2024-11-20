/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef INCLUDE_EXADG_TIME_INTEGRATION_LAMBDA_FUNCTIONS_ALE_H_
#define INCLUDE_EXADG_TIME_INTEGRATION_LAMBDA_FUNCTIONS_ALE_H_

// deal.II
#include <deal.II/base/exceptions.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace ExaDG
{
/**
 * This class is a collection of helper / utility functions required by time integration routines in
 * case of Arbitrary Lagrangian-Eulerian (ALE) methods with moving domains / grids. We realize these
 * utility functions as lambda functions so that the time integrator classes do not depend on data
 * structures owned by other classes.
 */
template<int dim, typename Number>
class HelpersALE
{
public:
  /**
   * This function moves the grid. It only affects basic deal.II data structures such as
   * dealii::Triangulation and dealii::Mapping. All dependent data structures need to be updated
   * separately.
   *
   * The default implementation causes the program to terminate, in order to avoid uninitialized /
   * undefined functions leading to erroneous results.
   */
  std::function<void(double const & time)> move_grid = [](double const & time) {
    (void)time;
    AssertThrow(false, dealii::ExcMessage("The function move_grid() has not been implemented."));
  };

  /**
   * This function updates the PDE operator after the grid has been moved. This function makes sure
   * that weak forms are evaluated correctly after the grid has been moved.
   *
   * The default implementation causes the program to terminate, in order to avoid uninitialized /
   * undefined functions leading to erroneous results.
   */
  std::function<void()> update_pde_operator_after_grid_motion = []() {
    AssertThrow(
      false,
      dealii::ExcMessage(
        "The function update_pde_operator_after_grid_motion() has not been implemented."));
  };

  /**
   * This function fills a DoF-vector describing the grid coordinates given an underlying
   * dof-handler.
   *
   * The default implementation causes the program to terminate, in order to avoid uninitialized /
   * undefined functions leading to erroneous results.
   */
  std::function<void(dealii::LinearAlgebra::distributed::Vector<Number> & grid_coordinates,
                     dealii::DoFHandler<dim> const &                      dof_handler)>
    fill_grid_coordinates_vector =
      [](dealii::LinearAlgebra::distributed::Vector<Number> & grid_coordinates,
         dealii::DoFHandler<dim> const &                      dof_handler) {
        (void)grid_coordinates;
        (void)dof_handler;

        AssertThrow(false,
                    dealii::ExcMessage(
                      "The function fill_grid_coordinates_vector() has not been implemented."));
      };
};
} // namespace ExaDG


#endif /* INCLUDE_EXADG_TIME_INTEGRATION_LAMBDA_FUNCTIONS_ALE_H_ */
