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

#ifndef INCLUDE_LAPLACE_INPUT_PARAMETERS_H_
#define INCLUDE_LAPLACE_INPUT_PARAMETERS_H_

#include <exadg/grid/grid_data.h>
#include <exadg/poisson/user_interface/enum_types.h>
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_parameters.h>
#include <exadg/solvers_and_preconditioners/solvers/solver_data.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace Poisson
{
class Parameters
{
public:
  // standard constructor that initializes parameters with default values
  Parameters();

  void
  check() const;

  bool
  involves_h_multigrid() const;

  void
  print(dealii::ConditionalOStream const & pcout, std::string const & name) const;

private:
  void
  print_parameters_mathematical_model(dealii::ConditionalOStream const & pcout) const;

  void
  print_parameters_spatial_discretization(dealii::ConditionalOStream const & pcout) const;

  void
  print_parameters_solver(dealii::ConditionalOStream const & pcout) const;

  void
  print_parameters_numerical_parameters(dealii::ConditionalOStream const & pcout) const;

public:
  /**************************************************************************************/
  /*                                                                                    */
  /*                                 MATHEMATICAL MODEL                                 */
  /*                                                                                    */
  /**************************************************************************************/

  // if the right-hand side f is unequal zero, set right_hand_side = true
  bool right_hand_side;

  /**************************************************************************************/
  /*                                                                                    */
  /*                              SPATIAL DISCRETIZATION                                */
  /*                                                                                    */
  /**************************************************************************************/

  // Grid data
  GridData grid;

  // Mapping
  unsigned int mapping_degree;

  // type of spatial discretization approach
  SpatialDiscretization spatial_discretization;

  // polynomial degree of shape functions
  unsigned int degree;

  // Symmetric interior penalty Galerkin (SIPG) discretization
  // interior penalty parameter scaling factor: default value is 1.0
  double IP_factor;


  /**************************************************************************************/
  /*                                                                                    */
  /*                                       SOLVER                                       */
  /*                                                                                    */
  /**************************************************************************************/

  // description: see enum declaration
  LinearSolver solver;

  // solver data
  SolverData solver_data;
  bool       compute_performance_metrics;

  // description: see enum declaration
  Preconditioner preconditioner;

  // description: see declaration of MultigridData
  MultigridData multigrid_data;

  /**************************************************************************************/
  /*                                                                                    */
  /*                                NUMERICAL PARAMETERS                                */
  /*                                                                                    */
  /**************************************************************************************/

  // By default, the matrix-free implementation performs separate loops over all cells,
  // interior faces, and boundary faces. For a certain type of operations, however, it
  // is necessary to perform the face-loop as a loop over all faces of a cell with an
  // outer loop over all cells, e.g., preconditioners operating on the level of
  // individual cells (for example block Jacobi). With this parameter, the loop structure
  // can be changed to such an algorithm (cell_based_face_loops).
  bool enable_cell_based_face_loops;
};

} // namespace Poisson
} // namespace ExaDG

#endif /* INCLUDE_LAPLACE_INPUT_PARAMETERS_H_ */
