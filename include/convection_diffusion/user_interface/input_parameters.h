/*
 * InputParametersConvDiff.h
 *
 *  Created on: Aug 17, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_INPUT_PARAMETERS_H_
#define INCLUDE_CONVECTION_DIFFUSION_INPUT_PARAMETERS_H_

#include <deal.II/base/exceptions.h>
#include "../../functionalities/print_functions.h"
#include "../../postprocessor/error_calculation_data.h"
#include "../../postprocessor/output_data.h"
#include "../../solvers_and_preconditioners/multigrid/multigrid_input_parameters.h"
#include "../../solvers_and_preconditioners/solvers/solver_data.h"

namespace ConvDiff
{
/**************************************************************************************/
/*                                                                                    */
/*                                 MATHEMATICAL MODEL                                 */
/*                                                                                    */
/**************************************************************************************/

/*
 *  ProblemType describes whether a steady or an unsteady problem has to be solved
 */
enum class ProblemType
{
  Undefined,
  Steady,
  Unsteady
};


/*
 *  EquationType describes the physical/mathematical model that has to be solved,
 *  i.e., diffusion problem, convective problem or convection-diffusion problem
 */
enum class EquationType
{
  Undefined,
  Convection,
  Diffusion,
  ConvectionDiffusion
};

/**************************************************************************************/
/*                                                                                    */
/*                                 PHYSICAL QUANTITIES                                */
/*                                                                                    */
/**************************************************************************************/

// there are currently no enums for this section



/**************************************************************************************/
/*                                                                                    */
/*                             TEMPORAL DISCRETIZATION                                */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Temporal discretization method:
 *  ExplRK: Explicit Runge-Kutta methods (implemented for orders 1-4)
 *  BDF: backward differentiation formulae (implemented for order 1-3)
 */
enum class TemporalDiscretization
{
  Undefined,
  ExplRK,
  BDF
};

/*
 *  For the BDF time integrator, the convective term can be either
 *  treated explicitly or implicitly
 */
enum class TreatmentOfConvectiveTerm
{
  Undefined,
  Explicit,    // additive decomposition (IMEX)
  ExplicitOIF, // operator-integration-factor splitting (Maday et al. 1990)
  Implicit
};

/*
 *  Temporal discretization method for OIF splitting:
 *
 *    Explicit Runge-Kutta methods
 */
enum class TimeIntegratorRK
{
  Undefined,
  ExplRK1Stage1,
  ExplRK2Stage2,
  ExplRK3Stage3,
  ExplRK4Stage4,
  ExplRK3Stage4Reg2C,
  ExplRK3Stage7Reg2, // optimized for maximum time step sizes in DG context
  ExplRK4Stage5Reg2C,
  ExplRK4Stage8Reg2, // optimized for maximum time step sizes in DG context
  ExplRK4Stage5Reg3C,
  ExplRK5Stage9Reg2S
};

/*
 * calculation of time step size
 */
enum class TimeStepCalculation
{
  Undefined,
  ConstTimeStepUserSpecified,
  ConstTimeStepCFL,
  ConstTimeStepDiffusion,
  ConstTimeStepCFLAndDiffusion,
  ConstTimeStepMaxEfficiency
};

/**************************************************************************************/
/*                                                                                    */
/*                               SPATIAL DISCRETIZATION                               */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Numerical flux formulation of convective term
 */

enum class NumericalFluxConvectiveOperator
{
  Undefined,
  CentralFlux,
  LaxFriedrichsFlux
};

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

/*
 * Elementwise preconditioner for block Jacobi preconditioner (only relevant for
 * elementwise iterative solution procedure)
 */
enum class PreconditionerBlockDiagonal
{
  Undefined,
  None,
  InverseMassMatrix
};

/*
 * Specify the operator type to be used for multigrid (which can differ from the
 * equation type)
 */
enum class MultigridOperatorType
{
  Undefined,
  ReactionDiffusion,
  ReactionConvection,
  ReactionConvectionDiffusion
};

/**************************************************************************************/
/*                                                                                    */
/*                               OUTPUT AND POSTPROCESSING                            */
/*                                                                                    */
/**************************************************************************************/

// there are currently no enums for this section



class InputParameters
{
public:
  // standard constructor that initializes parameters with default values
  InputParameters()
    : // MATHEMATICAL MODEL
      problem_type(ProblemType::Undefined),
      equation_type(EquationType::Undefined),
      right_hand_side(false),

      // PHYSICAL QUANTITIES
      start_time(0.),
      end_time(-1.),
      diffusivity(0.),

      // TEMPORAL DISCRETIZATION
      temporal_discretization(TemporalDiscretization::Undefined),
      time_integrator_rk(TimeIntegratorRK::Undefined),
      order_time_integrator(1),
      start_with_low_order(true),
      treatment_of_convective_term(TreatmentOfConvectiveTerm::Undefined),
      calculation_of_time_step_size(TimeStepCalculation::Undefined),
      time_step_size(-1.),
      cfl_number(-1.),
      time_integrator_oif(TimeIntegratorRK::Undefined),
      cfl_oif(-1.),
      diffusion_number(-1.),
      c_eff(-1.),
      exponent_fe_degree_convection(1.5),
      exponent_fe_degree_diffusion(3.0),

      // SPATIAL DISCRETIZATION
      numerical_flux_convective_operator(NumericalFluxConvectiveOperator::Undefined),
      IP_factor(1.0),

      // SOLVER
      solver(Solver::Undefined),
      use_right_preconditioner(true),
      max_n_tmp_vectors(30),
      abs_tol(1.e-20),
      rel_tol(1.e-12),
      max_iter(std::numeric_limits<unsigned int>::max()),
      preconditioner(Preconditioner::Undefined),
      update_preconditioner(false),
      implement_block_diagonal_preconditioner_matrix_free(false),
      preconditioner_block_diagonal(PreconditionerBlockDiagonal::InverseMassMatrix),
      block_jacobi_solver_data(SolverData(1000, 1.e-12, 1.e-2)),
      mg_operator_type(MultigridOperatorType::Undefined),
      multigrid_data(MultigridData()),

      // NUMERICAL PARAMETERS
      use_cell_based_face_loops(false),
      runtime_optimization(false),

      // OUTPUT AND POSTPROCESSING
      print_input_parameters(false),

      // write output
      output_data(OutputData()),

      // calculation of errors
      error_data(ErrorCalculationData()),

      output_solver_info_every_timesteps(1)
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
    // MATHEMATICAL MODEL
    AssertThrow(problem_type != ProblemType::Undefined, ExcMessage("parameter must be defined"));

    AssertThrow(equation_type != EquationType::Undefined, ExcMessage("parameter must be defined"));


    // PHYSICAL QUANTITIES
    AssertThrow(end_time > start_time, ExcMessage("parameter must be defined"));

    // Set the diffusivity whenever the diffusive term is involved.
    if(equation_type == EquationType::Diffusion ||
       equation_type == EquationType::ConvectionDiffusion)
      AssertThrow(diffusivity > (0.0 + 1.0e-12), ExcMessage("parameter must be defined"));


    // TEMPORAL DISCRETIZATION
    AssertThrow(temporal_discretization != TemporalDiscretization::Undefined,
                ExcMessage("parameter must be defined"));

    if(temporal_discretization == TemporalDiscretization::BDF)
    {
      AssertThrow(treatment_of_convective_term != TreatmentOfConvectiveTerm::Undefined,
                  ExcMessage("parameter must be defined"));
    }

    if(temporal_discretization == TemporalDiscretization::ExplRK)
    {
      AssertThrow(time_integrator_rk != TimeIntegratorRK::Undefined,
                  ExcMessage("parameter must be defined"));
    }

    AssertThrow(calculation_of_time_step_size != TimeStepCalculation::Undefined,
                ExcMessage("parameter must be defined"));

    if(calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepUserSpecified)
      AssertThrow(time_step_size > 0.0, ExcMessage("parameter must be defined"));

    if(calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepMaxEfficiency)
      AssertThrow(c_eff > 0., ExcMessage("parameter must be defined"));

    if(temporal_discretization == TemporalDiscretization::ExplRK)
    {
      AssertThrow(order_time_integrator >= 1 && order_time_integrator <= 4,
                  ExcMessage("Specified order of time integrator ExplRK not implemented!"));

      // for the explicit RK method both the convective and the diffusive term are
      // treated explicitly -> one has to specify both the CFL-number and the Diffusion-number
      AssertThrow(cfl_number > 0., ExcMessage("parameter must be defined"));
      AssertThrow(diffusion_number > 0., ExcMessage("parameter must be defined"));
    }

    if(temporal_discretization == TemporalDiscretization::BDF)
    {
      AssertThrow(order_time_integrator >= 1 && order_time_integrator <= 4,
                  ExcMessage("Specified order of time integrator BDF not implemented!"));

      if(treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
      {
        AssertThrow(time_integrator_oif != TimeIntegratorRK::Undefined,
                    ExcMessage("parameter must be defined"));

        AssertThrow(cfl_number > 0., ExcMessage("parameter must be defined"));
        AssertThrow(cfl_oif > 0., ExcMessage("parameter must be defined"));
      }
    }

    // SPATIAL DISCRETIZATION
    if(equation_type == EquationType::Convection ||
       equation_type == EquationType::ConvectionDiffusion || runtime_optimization == true)
    {
      AssertThrow(numerical_flux_convective_operator != NumericalFluxConvectiveOperator::Undefined,
                  ExcMessage("parameter must be defined"));
    }


    // SOLVER
    if(temporal_discretization != TemporalDiscretization::ExplRK)
    {
      AssertThrow(solver != Solver::Undefined, ExcMessage("parameter must be defined"));

      AssertThrow(preconditioner != Preconditioner::Undefined,
                  ExcMessage("parameter must be defined"));

      if(preconditioner == Preconditioner::Multigrid)
      {
        AssertThrow(mg_operator_type != MultigridOperatorType::Undefined,
                    ExcMessage("parameter must be defined"));
      }
    }


    // NUMERICAL PARAMETERS


    // OUTPUT AND POSTPROCESSING
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
    // If a linear system of equations has to be solved:
    // for the currently implementation this means
    if(temporal_discretization != TemporalDiscretization::ExplRK)
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

    /*
     *  The definition of string-arrays in this function is somehow redundant with the
     *  enum declarations but I think C++ does not offer a more elaborate conversion
     *  from enums to strings
     */

    // equation type
    std::string str_equation_type[] = {"Undefined",
                                       "Convection",
                                       "Diffusion",
                                       "ConvectionDiffusion"};

    print_parameter(pcout, "Equation type", str_equation_type[(int)equation_type]);

    // right hand side
    print_parameter(pcout, "Right-hand side", right_hand_side);
  }

  void
  print_parameters_physical_quantities(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Physical quantities:" << std::endl;

    // start and end time
    if(true /*problem_type == ProblemType::Unsteady*/)
    {
      print_parameter(pcout, "Start time", start_time);
      print_parameter(pcout, "End time", end_time);
    }

    // diffusivity
    if(equation_type == EquationType::Diffusion ||
       equation_type == EquationType::ConvectionDiffusion)
    {
      print_parameter(pcout, "Diffusivity", diffusivity);
    }
  }

  void
  print_parameters_temporal_discretization(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Temporal discretization:" << std::endl;

    /*
     *  The definition of string-arrays in this function is somehow redundant with the
     *  enum declarations but I think C++ does not offer a more elaborate conversion
     *  from enums to strings
     */

    std::string str_temp_discret[] = {"Undefined", "ExplicitRungeKutta", "BDF"};

    print_parameter(pcout,
                    "Temporal discretization method",
                    str_temp_discret[(int)temporal_discretization]);

    if(temporal_discretization == TemporalDiscretization::ExplRK)
    {
      std::string str_expl_time_int[] = {"Undefined",
                                         "ExplRK1Stage1",
                                         "ExplRK2Stage2",
                                         "ExplRK3Stage3",
                                         "ExplRK4Stage4",
                                         "ExplRK3Stage4Reg2C",
                                         "ExplRK3Stage7Reg2",
                                         "ExplRK4Stage5Reg2C",
                                         "ExplRK4Stage8Reg2",
                                         "ExplRK4Stage5Reg3C",
                                         "ExplRK5Stage9Reg2S"};

      print_parameter(pcout,
                      "Explicit time integrator",
                      str_expl_time_int[(int)time_integrator_rk]);
    }

    if(temporal_discretization == TemporalDiscretization::BDF)
    {
      print_parameter(pcout, "Order of time integrator", order_time_integrator);

      print_parameter(pcout, "Start with low order method", start_with_low_order);

      std::string str_treatment_conv[] = {"Undefined", "Explicit", "ExplicitOIF", "Implicit"};

      print_parameter(pcout,
                      "Treatment of convective term",
                      str_treatment_conv[(int)treatment_of_convective_term]);

      if(treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
      {
        std::string str_time_int_oif[] = {"Undefined",
                                          "ExplRK1Stage1",
                                          "ExplRK2Stage2",
                                          "ExplRK3Stage3",
                                          "ExplRK4Stage4",
                                          "ExplRK3Stage4Reg2C",
                                          "ExplRK3Stage7Reg2",
                                          "ExplRK4Stage5Reg2C",
                                          "ExplRK4Stage8Reg2",
                                          "ExplRK4Stage5Reg3C",
                                          "ExplRK5Stage9Reg2S"};

        print_parameter(pcout,
                        "Time integrator for OIF splitting",
                        str_time_int_oif[(int)time_integrator_oif]);
      }
    }

    std::string str_time_step_calc[] = {"Undefined",
                                        "ConstTimeStepUserSpecified",
                                        "ConstTimeStepCFL",
                                        "ConstTimeStepDiffusion",
                                        "ConstTimeStepCFLAndDiffusion"};

    print_parameter(pcout,
                    "Calculation of time step size",
                    str_time_step_calc[(int)calculation_of_time_step_size]);


    // here we do not print quantities such as  cfl_number, diffusion_number, time_step_size
    // because this is done by the time integration scheme (or the functions that
    // calculate the time step size)
  }

  void
  print_parameters_spatial_discretization(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Spatial Discretization:" << std::endl;

    if(equation_type == EquationType::Convection ||
       equation_type == EquationType::ConvectionDiffusion)
    {
      std::string str_num_flux_convective[] = {"Undefined", "Central flux", "Lax-Friedrichs flux"};

      print_parameter(pcout,
                      "Numerical flux convective term",
                      str_num_flux_convective[(int)numerical_flux_convective_operator]);
    }

    if(equation_type == EquationType::Diffusion ||
       equation_type == EquationType::ConvectionDiffusion)
    {
      print_parameter(pcout, "IP factor viscous term", IP_factor);
    }
  }

  void
  print_parameters_solver(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Solver:" << std::endl;

    /*
     *  The definition of string-arrays in this function is somehow redundant with the
     *  enum declarations but I think C++ does not offer a more elaborate conversion
     *  from enums to strings
     */

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
      "Undefined", "None", "InverseMassMatrix", "PointJacobi", "BlockJacobi", "Multigrid"};

    print_parameter(pcout, "Preconditioner", str_precon[(int)preconditioner]);

    print_parameter(pcout, "Update preconditioner", update_preconditioner);

    print_parameter(pcout,
                    "Block Jacobi matrix-free",
                    implement_block_diagonal_preconditioner_matrix_free);

    if(implement_block_diagonal_preconditioner_matrix_free)
    {
      std::string str_precon[] = {"Undefined", "None", "InverseMassMatrix"};

      print_parameter(pcout,
                      "Preconditioner block diagonal",
                      str_precon[(int)preconditioner_block_diagonal]);

      block_jacobi_solver_data.print(pcout);
    }

    if(preconditioner == Preconditioner::Multigrid)
    {
      std::string str_mg[] = {"Undefined",
                              "Reaction-Diffusion",
                              "Reaction-Convection",
                              "Reaction-Convection-Diffusion"};

      print_parameter(pcout, "MG Operator type", str_mg[(int)mg_operator_type]);
      multigrid_data.print(pcout);
    }
  }


  void
  print_parameters_numerical_parameters(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Numerical parameters:" << std::endl;

    print_parameter(pcout, "Use cell-based face loops", use_cell_based_face_loops);
    print_parameter(pcout,
                    "Block Jacobi implemented matrix-free",
                    implement_block_diagonal_preconditioner_matrix_free);
    print_parameter(pcout, "Runtime optimization", runtime_optimization);
  }


  void
  print_parameters_output_and_postprocessing(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Output and postprocessing:" << std::endl;

    output_data.print(pcout, true /*problem_type == ProblemType::Unsteady*/);

    error_data.print(pcout, true /*problem_type == ProblemType::Unsteady*/);
  }


  /**************************************************************************************/
  /*                                                                                    */
  /*                                 MATHEMATICAL MODEL                                 */
  /*                                                                                    */
  /**************************************************************************************/

  // description: see enum declaration
  ProblemType problem_type;

  // description: see enum declaration
  EquationType equation_type;

  // if the rhs f is unequal zero, set right_hand_side = true
  bool right_hand_side;

  /**************************************************************************************/
  /*                                                                                    */
  /*                                 PHYSICAL QUANTITIES                                */
  /*                                                                                    */
  /**************************************************************************************/

  // start time of simulation
  double start_time;

  // end time of simulation
  double end_time;

  // kinematic diffusivity
  double diffusivity;



  /**************************************************************************************/
  /*                                                                                    */
  /*                             TEMPORAL DISCRETIZATION                                */
  /*                                                                                    */
  /**************************************************************************************/

  // temporal discretization method
  TemporalDiscretization temporal_discretization;

  // description: see enum declaration (only relevant explicit time integration ExplRK)
  TimeIntegratorRK time_integrator_rk;

  // order of time integration scheme (only relevant for BDF time integration)
  unsigned int order_time_integrator;

  // start with low order (only relevant for BDF time integration)
  bool start_with_low_order;

  // description: see enum declaration (only relevant for BDF time integration)
  TreatmentOfConvectiveTerm treatment_of_convective_term;

  // calculation of time step size
  TimeStepCalculation calculation_of_time_step_size;

  // user specified time step size:  note that this time_step_size is the first
  // in a series of time_step_size's when performing temporal convergence tests,
  // i.e., delta_t = time_step_size, time_step_size/2, ...
  double time_step_size;

  // cfl number ("global" CFL number, can be larger than critical CFL in case
  // of operator-integration-factor splitting)
  double cfl_number;

  // specify the time integration scheme that is used for the OIF substepping of the
  // convective term (only relevant for BDF time integration)
  TimeIntegratorRK time_integrator_oif;

  // cfl number for operator-integration-factor splitting (has to be smaller than the
  // critical time step size arising from the CFL restriction)
  double cfl_oif;

  // diffusion number (relevant number for limitation of time step size
  // when treating the diffusive term explicitly)
  double diffusion_number;

  // C_eff: constant that has to be specified for time step calculation method
  // MaxEfficiency, which means that the time step is selected such that the errors of
  // the temporal and spatial discretization are comparable
  double c_eff;

  // exponent of fe_degree used in the calculation of the convective time step size
  double exponent_fe_degree_convection;

  // exponent of fe_degree used in the calculation of the diffusion time step size
  double exponent_fe_degree_diffusion;

  /**************************************************************************************/
  /*                                                                                    */
  /*                              SPATIAL DISCRETIZATION                                */
  /*                                                                                    */
  /**************************************************************************************/

  // convective term: the convective term is written in divergence formulation

  // description: see enum declaration
  NumericalFluxConvectiveOperator numerical_flux_convective_operator;

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

  // update preconditioner in case of varying parameters
  bool update_preconditioner;

  // Implement block diagonal (block Jacobi) preconditioner in a matrix-free way
  // by solving the block Jacobi problems elementwise using iterative solvers and
  // matrix-free operator evaluation
  bool implement_block_diagonal_preconditioner_matrix_free;

  // description: see enum declaration
  PreconditionerBlockDiagonal preconditioner_block_diagonal;

  // solver data for block Jacobi preconditioner (only relevant for elementwise
  // iterative solution procedure)
  SolverData block_jacobi_solver_data;

  // description: see enum declaration
  MultigridOperatorType mg_operator_type;

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
  bool use_cell_based_face_loops;

  // Runtime optimization: Evaluate volume and surface integrals of convective term,
  // diffusive term and rhs term in one function (local_apply, local_apply_face,
  // local_apply_boundary_face) instead of implementing each operator seperately and
  // subsequently looping over all operators.
  // Note: if runtime_optimization == false:
  //   If an operator is not used (e.g. purely diffusive problem) the volume and
  //   surface integrals of this operator are simply not evaluated
  // Note: if runtime_optimization == true:
  //  ensure that the rhs-function, velocity-field and that the diffusivity is zero
  //  if the rhs operator, convective operator or diffusive operator is "inactive"
  //  because the volume and surface integrals of these operators will always be evaluated
  bool runtime_optimization;

  /**************************************************************************************/
  /*                                                                                    */
  /*                               OUTPUT AND POSTPROCESSING                            */
  /*                                                                                    */
  /**************************************************************************************/

  // print a list of all input parameters at the beginning of the simulation
  bool print_input_parameters;

  // writing output
  OutputData output_data;

  // calculation of errors
  ErrorCalculationData error_data;

  // show solver performance (wall time, number of iterations) every ... timesteps
  unsigned int output_solver_info_every_timesteps;
};

} // namespace ConvDiff
#endif /* INCLUDE_CONVECTION_DIFFUSION_INPUT_PARAMETERS_H_ */
