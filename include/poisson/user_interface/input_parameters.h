/*
 * input_parameters.h
 *
 *  Created on:
 *      Author:
 */

#ifndef INCLUDE_LAPLACE_INPUT_PARAMETERS_H_
#define INCLUDE_LAPLACE_INPUT_PARAMETERS_H_

#include "../../solvers_and_preconditioners/multigrid/multigrid_input_parameters.h"
#include "../../solvers_and_preconditioners/solvers/solver_data.h"
#include "../include/functionalities/print_functions.h"
#include "postprocessor/error_calculation_data.h"
#include "postprocessor/output_data.h"

#include "enum_types.h"

namespace Poisson
{
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

  // number of space dimensions
  unsigned int dim;

  // if the rhs f is unequal zero, set right_hand_side = true
  bool right_hand_side;

  /**************************************************************************************/
  /*                                                                                    */
  /*                              SPATIAL DISCRETIZATION                                */
  /*                                                                                    */
  /**************************************************************************************/

  // triangulation type
  TriangulationType triangulation_type;

  // Polynomial degree of shape functions
  unsigned int degree;

  // Type of mapping (polynomial degree) use for geometry approximation
  MappingType mapping;

  // Number of mesh refinement steps
  unsigned int h_refinements;

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

  /**************************************************************************************/
  /*                                                                                    */
  /*                               OUTPUT AND POSTPROCESSING                            */
  /*                                                                                    */
  /**************************************************************************************/

  // writing output
  OutputData output_data;
};

} // namespace Poisson
#endif /* INCLUDE_LAPLACE_INPUT_PARAMETERS_H_ */
