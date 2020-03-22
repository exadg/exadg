/*
 * input_parameters.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_USER_INTERFACE_INPUT_PARAMETERS_H_
#define INCLUDE_STRUCTURE_USER_INTERFACE_INPUT_PARAMETERS_H_

// deal.II
#include <deal.II/base/exceptions.h>

#include "../../functionalities/print_functions.h"

#include "../../solvers_and_preconditioners/multigrid/multigrid_input_parameters.h"
#include "../../solvers_and_preconditioners/solvers/solver_data.h"

#include "../../structure/user_interface/enum_types.h"

namespace Structure
{
class InputParameters
{
public:
  // standard constructor that initializes parameters with default values
  InputParameters()
    : // MATHEMATICAL MODEL
      problem_type(ProblemType::Undefined),
      right_hand_side(false),
      large_deformation(false),
      updated_formulation(true),

      // PHYSICAL QUANTITIES

      // TEMPORAL DISCRETIZATION
      end_time(1.0),
      delta_t(1.0),
      time_step_control(false),

      // SPATIAL DISCRETIZATION
      triangulation_type(TriangulationType::Undefined),
      mapping(MappingType::Affine),

      // SOLVER
      solver(Solver::Undefined),
      solver_data(SolverData(1e4, 1.e-12, 1.e-6, 100)),
      preconditioner(Preconditioner::Undefined),
      multigrid_data(MultigridData())
  {
  }

  void
  check_input_parameters()
  {
    // MATHEMATICAL MODEL
    AssertThrow(problem_type != ProblemType::Undefined, ExcMessage("Parameter must be defined."));
    AssertThrow(problem_type != ProblemType::Unsteady,
                ExcMessage("Unsteady solver is not implemented."));

    if(problem_type == ProblemType::QuasiStatic)
    {
      AssertThrow(large_deformation == true,
                  ExcMessage("QuasiStatic solver only implemented for nonlinear formulation."));
    }

    // SPATIAL DISCRETIZATION
    AssertThrow(triangulation_type != TriangulationType::Undefined,
                ExcMessage("Parameter must be defined."));

    // SOLVER
    AssertThrow(solver != Solver::Undefined, ExcMessage("Parameter must be defined."));
    AssertThrow(preconditioner != Preconditioner::Undefined,
                ExcMessage("Parameter must be defined."));
  }

  void
  print(ConditionalOStream & pcout, std::string const & name)
  {
    pcout << std::endl << name << std::endl;

    // MATHEMATICAL MODEL
    print_parameters_mathematical_model(pcout);

    // PHYSICAL QUANTITIES
    print_parameters_physical_quantities(pcout);

    // TEMPORAL DISCRETIZATION
    print_parameters_temporal_discretization(pcout);

    // SPATIAL DISCRETIZATION
    print_parameters_spatial_discretization(pcout);

    // SOLVER
    print_parameters_solver(pcout);
  }

  void
  print_parameters_mathematical_model(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Mathematical model:" << std::endl;

    print_parameter(pcout, "Problem type", enum_to_string(problem_type));

    print_parameter(pcout, "Right-hand side", right_hand_side);

    print_parameter(pcout, "Large deformation", large_deformation);

    print_parameter(pcout, "updated Lagrangian formulation", updated_formulation);
  }

  void
  print_parameters_physical_quantities(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Physical quantities:" << std::endl;

    (void)pcout;
  }

  void
  print_parameters_temporal_discretization(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Temporal discretization:" << std::endl;

    if(problem_type == ProblemType::QuasiStatic || problem_type == ProblemType::Unsteady)
    {
      print_parameter(pcout, "End time", end_time);
      print_parameter(pcout, "Delta t", delta_t);
      print_parameter(pcout, "Adapt time step size", time_step_control);
    }
  }

  void
  print_parameters_spatial_discretization(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Spatial Discretization:" << std::endl;

    print_parameter(pcout, "Triangulation type", enum_to_string(triangulation_type));
    print_parameter(pcout, "Mapping", enum_to_string(mapping));
  }

  void
  print_parameters_solver(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Solver:" << std::endl;

    print_parameter(pcout, "Solver", enum_to_string(solver));

    solver_data.print(pcout);

    print_parameter(pcout, "Preconditioner", enum_to_string(preconditioner));

    if(preconditioner == Preconditioner::Multigrid)
    {
      multigrid_data.print(pcout);
    }
  }

  /**************************************************************************************/
  /*                                                                                    */
  /*                                 MATHEMATICAL MODEL                                 */
  /*                                                                                    */
  /**************************************************************************************/

  // description: see enum declaration
  ProblemType problem_type;

  // if the rhs f is unequal zero, set right_hand_side = true
  bool right_hand_side;

  // are large deformations to be expected, than compute with non linear method
  bool large_deformation;

  // which formulation of the weak form is used
  // updated Lagrangian formulation = true
  // total Lagrangian formulation = false
  // only relevant for nonlinear formulation
  bool updated_formulation;

  /**************************************************************************************/
  /*                                                                                    */
  /*                                 PHYSICAL QUANTITIES                                */
  /*                                                                                    */
  /**************************************************************************************/


  /**************************************************************************************/
  /*                                                                                    */
  /*                             TEMPORAL DISCRETIZATION                                */
  /*                                                                                    */
  /**************************************************************************************/

  // these parameters are currently used for quasi-static solver (... and later for
  // unsteady solver as well)
  double end_time;
  double delta_t;
  bool   time_step_control;

  /**************************************************************************************/
  /*                                                                                    */
  /*                              SPATIAL DISCRETIZATION                                */
  /*                                                                                    */
  /**************************************************************************************/

  // triangulation type
  TriangulationType triangulation_type;

  // Type of mapping (polynomial degree) use for geometry approximation
  MappingType mapping;


  /**************************************************************************************/
  /*                                                                                    */
  /*                                       SOLVER                                       */
  /*                                                                                    */
  /**************************************************************************************/

  // description: see enum declaration
  Solver solver;

  // solver data
  SolverData solver_data;

  // description: see enum declaration
  Preconditioner preconditioner;

  // description: see declaration of MultigridData
  MultigridData multigrid_data;
};

} // namespace Structure

#endif
