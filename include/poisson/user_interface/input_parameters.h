/*
 * input_parameters.h
 *
 *  Created on:
 *      Author:
 */

#ifndef INCLUDE_LAPLACE_INPUT_PARAMETERS_H_
#define INCLUDE_LAPLACE_INPUT_PARAMETERS_H_

#include "../../grid/enum_types.h"
#include "../../utilities/print_functions.h"

#include "../../solvers_and_preconditioners/multigrid/multigrid_input_parameters.h"
#include "../../solvers_and_preconditioners/solvers/solver_data.h"

#include "enum_types.h"

namespace ExaDG
{
namespace Poisson
{
using namespace dealii;

class InputParameters
{
public:
  // standard constructor that initializes parameters with default values
  InputParameters();

  void
  check_input_parameters();

  void
  print(ConditionalOStream & pcout, std::string const & name);

private:
  void
  print_parameters_mathematical_model(ConditionalOStream & pcout);

  void
  print_parameters_spatial_discretization(ConditionalOStream & pcout);

  void
  print_parameters_solver(ConditionalOStream & pcout);

  void
  print_parameters_numerical_parameters(ConditionalOStream & pcout);

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

  // triangulation type
  TriangulationType triangulation_type;

  // Type of mapping (polynomial degree) use for geometry approximation
  MappingType mapping;

  // type of spatial discretization approach
  SpatialDiscretization spatial_discretization;

  // Symmetric interior penalty Galerkin (SIPG) discretization
  // interior penalty parameter scaling factor: default value is 1.0
  double IP_factor;


  /**************************************************************************************/
  /*                                                                                    */
  /*                                       SOLVER                                       */
  /*                                                                                    */
  /**************************************************************************************/

  // description: see enum declaration
  Solver solver;

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
