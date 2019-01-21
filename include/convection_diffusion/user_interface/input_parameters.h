/*
 * input_parameters.h
 *
 *  Created on: Aug 17, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_INPUT_PARAMETERS_H_
#define INCLUDE_CONVECTION_DIFFUSION_INPUT_PARAMETERS_H_

#include <deal.II/base/exceptions.h>
#include "../../functionalities/print_functions.h"
#include "../../functionalities/restart_data.h"
#include "../../postprocessor/error_calculation_data.h"
#include "../../postprocessor/output_data.h"
#include "../../solvers_and_preconditioners/multigrid/multigrid_input_parameters.h"
#include "../../solvers_and_preconditioners/solvers/solver_data.h"

#include "enum_types.h"

namespace ConvDiff
{
class InputParameters
{
public:
  // standard constructor that initializes parameters with default values
  InputParameters()
    : // MATHEMATICAL MODEL
      problem_type(ProblemType::Undefined),
      equation_type(EquationType::Undefined),
      type_velocity_field(TypeVelocityField::Analytical),
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
      adaptive_time_stepping(false),
      adaptive_time_stepping_limiting_factor(1.2),
      time_step_size(-1.),
      max_number_of_time_steps(std::numeric_limits<unsigned int>::max()),
      cfl(-1.),
      max_velocity(std::numeric_limits<double>::min()),
      time_integrator_oif(TimeIntegratorRK::Undefined),
      cfl_oif(-1.),
      diffusion_number(-1.),
      c_eff(-1.),
      exponent_fe_degree_convection(1.5),
      exponent_fe_degree_diffusion(3.0),

      // SPATIAL DISCRETIZATION
      degree_mapping(1),
      numerical_flux_convective_operator(NumericalFluxConvectiveOperator::Undefined),
      IP_factor(1.0),

      // SOLVER
      solver(Solver::Undefined),
      solver_data(SolverData(1e4, 1.e-12, 1.e-6, 100)),
      preconditioner(Preconditioner::Undefined),
      update_preconditioner(false),
      update_preconditioner_every_time_steps(1),
      implement_block_diagonal_preconditioner_matrix_free(false),
      preconditioner_block_diagonal(PreconditionerBlockDiagonal::InverseMassMatrix),
      block_jacobi_solver_data(SolverData(1000, 1.e-12, 1.e-2, 1000)),
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

      output_solver_info_every_timesteps(1),

      // restart
      restart_data(RestartData())
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

    if(calculation_of_time_step_size == TimeStepCalculation::UserSpecified)
      AssertThrow(time_step_size > 0.0, ExcMessage("parameter must be defined"));

    if(calculation_of_time_step_size == TimeStepCalculation::MaxEfficiency)
      AssertThrow(c_eff > 0., ExcMessage("parameter must be defined"));

    if(calculation_of_time_step_size == TimeStepCalculation::CFL)
    {
      AssertThrow(
        equation_type == EquationType::Convection ||
          equation_type == EquationType::ConvectionDiffusion,
        ExcMessage(
          "Type of time step calculation CFL does not make sense for the specified equation type."));
    }

    if(calculation_of_time_step_size == TimeStepCalculation::Diffusion)
    {
      AssertThrow(
        equation_type == EquationType::Diffusion ||
          equation_type == EquationType::ConvectionDiffusion,
        ExcMessage(
          "Type of time step calculation Diffusion does not make sense for the specified equation type."));
    }

    if(calculation_of_time_step_size == TimeStepCalculation::CFLAndDiffusion)
    {
      AssertThrow(
        equation_type == EquationType::ConvectionDiffusion,
        ExcMessage(
          "Type of time step calculation CFLAndDiffusion does not make sense for the specified equation type."));
    }

    if(adaptive_time_stepping == true)
    {
      AssertThrow(calculation_of_time_step_size == TimeStepCalculation::CFL ||
                    calculation_of_time_step_size == TimeStepCalculation::CFLAndDiffusion,
                  ExcMessage(
                    "Adaptive time stepping can only be used in combination with CFL condition."));
    }

    if(temporal_discretization == TemporalDiscretization::ExplRK)
    {
      AssertThrow(order_time_integrator >= 1 && order_time_integrator <= 4,
                  ExcMessage("Specified order of time integrator ExplRK not implemented!"));

      // for the explicit RK method both the convective and the diffusive term are
      // treated explicitly -> one has to specify both the CFL-number and the Diffusion-number
      AssertThrow(cfl > 0., ExcMessage("parameter must be defined"));
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

        AssertThrow(cfl > 0., ExcMessage("parameter must be defined"));
        AssertThrow(cfl_oif > 0., ExcMessage("parameter must be defined"));
      }
    }

    // SPATIAL DISCRETIZATION
    AssertThrow(degree_mapping > 0, ExcMessage("Invalid parameter."));

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

        if(treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
        {
          AssertThrow(mg_operator_type != MultigridOperatorType::ReactionConvection &&
                        mg_operator_type != MultigridOperatorType::ReactionConvectionDiffusion,
                      ExcMessage(
                        "Invalid solver parameters. The convective term is treated explicitly."));
        }
      }
    }

    if(implement_block_diagonal_preconditioner_matrix_free)
    {
      AssertThrow(
        use_cell_based_face_loops == true,
        ExcMessage(
          "Cell based face loops have to be used for matrix-free implementation of block diagonal preconditioner."));
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

    print_parameter(pcout, "Problem type", enum_to_string(problem_type));
    print_parameter(pcout, "Equation type", enum_to_string(equation_type));
    print_parameter(pcout, "Type of velocity field", enum_to_string(type_velocity_field));
    print_parameter(pcout, "Right-hand side", right_hand_side);
  }

  void
  print_parameters_physical_quantities(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Physical quantities:" << std::endl;

    if(problem_type == ProblemType::Unsteady)
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

    print_parameter(pcout,
                    "Temporal discretization method",
                    enum_to_string(temporal_discretization));

    if(temporal_discretization == TemporalDiscretization::ExplRK)
    {
      print_parameter(pcout, "Explicit time integrator", enum_to_string(time_integrator_rk));
    }

    // maximum number of time steps
    print_parameter(pcout, "Maximum number of time steps", max_number_of_time_steps);

    if(temporal_discretization == TemporalDiscretization::BDF)
    {
      print_parameter(pcout, "Order of time integrator", order_time_integrator);
      print_parameter(pcout, "Start with low order method", start_with_low_order);
      print_parameter(pcout,
                      "Treatment of convective term",
                      enum_to_string(treatment_of_convective_term));

      if(treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
      {
        print_parameter(pcout,
                        "Time integrator for OIF splitting",
                        enum_to_string(time_integrator_oif));
      }
    }

    print_parameter(pcout,
                    "Calculation of time step size",
                    enum_to_string(calculation_of_time_step_size));

    print_parameter(pcout, "Adaptive time stepping", adaptive_time_stepping);

    if(adaptive_time_stepping)
    {
      print_parameter(pcout,
                      "Adaptive time stepping limiting factor",
                      adaptive_time_stepping_limiting_factor);
    }


    // here we do not print quantities such as  cfl, diffusion_number, time_step_size
    // because this is done by the time integration scheme (or the functions that
    // calculate the time step size)
  }

  void
  print_parameters_spatial_discretization(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Spatial Discretization:" << std::endl;

    print_parameter(pcout, "Polynomial degree of mapping", degree_mapping);

    if(equation_type == EquationType::Convection ||
       equation_type == EquationType::ConvectionDiffusion)
    {
      print_parameter(pcout,
                      "Numerical flux convective term",
                      enum_to_string(numerical_flux_convective_operator));
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

    print_parameter(pcout, "Solver", enum_to_string(solver));

    solver_data.print(pcout);

    print_parameter(pcout, "Preconditioner", enum_to_string(preconditioner));

    if(preconditioner != Preconditioner::None)
    {
      print_parameter(pcout, "Update preconditioner", update_preconditioner);

      if(update_preconditioner)
        print_parameter(pcout, "Update every time steps", update_preconditioner_every_time_steps);
    }

    print_parameter(pcout,
                    "Block Jacobi matrix-free",
                    implement_block_diagonal_preconditioner_matrix_free);

    if(implement_block_diagonal_preconditioner_matrix_free)
    {
      print_parameter(pcout,
                      "Preconditioner block diagonal",
                      enum_to_string(preconditioner_block_diagonal));

      block_jacobi_solver_data.print(pcout);
    }

    if(preconditioner == Preconditioner::Multigrid)
    {
      print_parameter(pcout, "Multigrid operator type", enum_to_string(mg_operator_type));
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

    output_data.print(pcout, problem_type == ProblemType::Unsteady);

    error_data.print(pcout, problem_type == ProblemType::Unsteady);

    restart_data.print(pcout);
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

  // description: see enum declaration
  TypeVelocityField type_velocity_field;

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

  // use adaptive time stepping?
  bool adaptive_time_stepping;

  // This parameter defines by which factor the time step size is allowed to increase
  // or to decrease in case of adaptive time step, e.g., if one wants to avoid large
  // jumps in the time step size. A factor of 1 implies that the time step size can not
  // change at all, while a factor towards infinity implies that arbitrary changes in
  // the time step size are allowed from one time step to the next.
  double adaptive_time_stepping_limiting_factor;

  // user specified time step size:  note that this time_step_size is the first
  // in a series of time_step_size's when performing temporal convergence tests,
  // i.e., delta_t = time_step_size, time_step_size/2, ...
  double time_step_size;

  // maximum number of time steps
  unsigned int max_number_of_time_steps;

  // cfl number ("global" CFL number, can be larger than critical CFL in case
  // of operator-integration-factor splitting)
  double cfl;

  // estimation of maximum velocity required for CFL condition
  double max_velocity;

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

  // Polynomial degree of shape functions used for geometry approximation (mapping from
  // parameter space to physical space)
  unsigned int degree_mapping;


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

  // solver data
  SolverData solver_data;

  // description: see enum declaration
  Preconditioner preconditioner;

  // update preconditioner in case of varying parameters
  bool update_preconditioner;

  // update preconditioner every ... time step. Only relevant if update preconditioner
  // is set to true.
  unsigned int update_preconditioner_every_time_steps;

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

  // Restart
  RestartData restart_data;
};

} // namespace ConvDiff
#endif /* INCLUDE_CONVECTION_DIFFUSION_INPUT_PARAMETERS_H_ */
