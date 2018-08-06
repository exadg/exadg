/*
 * InputParametersConvDiff.h
 *
 *  Created on:
 *      Author:
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_INPUT_PARAMETERS_H_
#define INCLUDE_CONVECTION_DIFFUSION_INPUT_PARAMETERS_H_

#include "../../solvers_and_preconditioners/multigrid/multigrid_input_parameters.h"
#include "../include/functionalities/print_functions.h"
#include "postprocessor/error_calculation_data.h"
#include "postprocessor/output_data.h"

namespace Laplace
{
/**************************************************************************************/
/*                                                                                    */
/*                                       SOLVER                                       */
/*                                                                                    */
/**************************************************************************************/

/*
 *   Solver for linear system of equations
 */
enum class Solver
{
  Undefined,
  PCG,
  GMRES
};

/*
 *  Preconditioner type for solution of linear system of equations
 */
enum class Preconditioner
{
  Undefined,
  None,
  InverseMassMatrix,
  PointJacobi,
  BlockJacobi,
  Multigrid
};

/**************************************************************************************/
/*                                                                                    */
/*                               OUTPUT AND POSTPROCESSING                            */
/*                                                                                    */
/**************************************************************************************/


class InputParameters
{
public:
  // standard constructor that initializes parameters with default values
  InputParameters()
    : // MATHEMATICAL MODEL
      right_hand_side(false),

      // PHYSICAL QUANTITIES

      // TEMPORAL DISCRETIZATION

      // SPATIAL DISCRETIZATION
      IP_factor(1.0),

      // SOLVER
      solver(Solver::Undefined),
      use_right_preconditioner(true),
      max_n_tmp_vectors(30),
      abs_tol(1.e-20),
      rel_tol(1.e-12),
      max_iter(std::numeric_limits<unsigned int>::max()),
      preconditioner(Preconditioner::Undefined),
      multigrid_data(MultigridData()),

      // OUTPUT AND POSTPROCESSING
      print_input_parameters(true)
  {
  }

  /*
   *  This function is implemented in the header file of the test case
   *  that has to be solved.
   */
  void
  set_input_parameters();

  void
  check_input_parameters()
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    pcout << std::endl << "List of input parameters:" << std::endl;

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

    // NUMERICAL PARAMETERS
    print_parameters_numerical_parameters(pcout);

    // OUTPUT AND POSTPROCESSING
    print_parameters_output_and_postprocessing(pcout);
  }

  void
  print_parameters_mathematical_model(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Mathematical model:" << std::endl;

    // right hand side
    print_parameter(pcout, "Right-hand side", right_hand_side);
  }

  void
  print_parameters_physical_quantities(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Physical quantities:" << std::endl;
  }

  void
  print_parameters_temporal_discretization(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Temporal discretization:" << std::endl;
  }

  void
  print_parameters_spatial_discretization(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Spatial Discretization:" << std::endl;

    print_parameter(pcout, "IP factor viscous term", IP_factor);
  }

  void
  print_parameters_solver(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Solver:" << std::endl;

    std::string str_solver[] = {"Undefined", "PCG", "GMRES"};

    print_parameter(pcout, "Solver", str_solver[(int)solver]);

    if(solver == Solver::GMRES)
    {
      print_parameter(pcout, "Use right preconditioner", use_right_preconditioner);
      print_parameter(pcout, "max_n_tmp_vectors", max_n_tmp_vectors);
    }

    print_parameter(pcout, "Absolute solver tolerance", abs_tol);
    print_parameter(pcout, "Relative solver tolerance", rel_tol);
    print_parameter(pcout, "Maximum number of iterations", max_iter);

    std::string str_precon[] = {
      "Undefined", "None", "InverseMassMatrix", "PointJacobi", "BlockJacobi", "GMG"};

    print_parameter(pcout, "Preconditioner", str_precon[(int)preconditioner]);

    if(preconditioner == Preconditioner::Multigrid)
      multigrid_data.print(pcout);
  }


  void
  print_parameters_numerical_parameters(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Numerical parameters:" << std::endl;
  }


  void
  print_parameters_output_and_postprocessing(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Output and postprocessing:" << std::endl;
  }


  /**************************************************************************************/
  /*                                                                                    */
  /*                                 MATHEMATICAL MODEL                                 */
  /*                                                                                    */
  /**************************************************************************************/

  // if the rhs f is unequal zero, set right_hand_side = true
  bool right_hand_side;

  /**************************************************************************************/
  /*                                                                                    */
  /*                              SPATIAL DISCRETIZATION                                */
  /*                                                                                    */
  /**************************************************************************************/

  // diffusive term: Symmetric interior penalty discretization Galerkin (SIPG)
  // interior penalty parameter scaling factor: default value is 1.0
  double IP_factor;



  /**************************************************************************************/
  /*                                                                                    */
  /*                                       SOLVER                                       */
  /*                                                                                    */
  /**************************************************************************************/

  // description: see enum declaration
  Solver solver;

  // use right-preconditioner in case of GMRES solver
  bool use_right_preconditioner;

  // max_n_temp_vectors for GMRES solver
  unsigned int max_n_tmp_vectors;

  // solver tolerances
  double       abs_tol;
  double       rel_tol;
  unsigned int max_iter;

  // description: see enum declaration
  Preconditioner preconditioner;

  // description: see declaration of MultigridData
  MultigridData multigrid_data;

  /**************************************************************************************/
  /*                                                                                    */
  /*                               OUTPUT AND POSTPROCESSING                            */
  /*                                                                                    */
  /**************************************************************************************/

  // print a list of all input parameters at the beginning of the simulation
  bool print_input_parameters;
};

} // namespace Laplace
#endif /* INCLUDE_CONVECTION_DIFFUSION_INPUT_PARAMETERS_H_ */
