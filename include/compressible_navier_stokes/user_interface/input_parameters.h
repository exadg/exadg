/*
 * input_parameters.h
 *
 *  Created on: 2018
 *      Author: fehn
 */

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_INPUT_PARAMETERS_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_INPUT_PARAMETERS_H_

#include "../../incompressible_navier_stokes/postprocessor/kinetic_energy_data.h"
#include "../../incompressible_navier_stokes/postprocessor/kinetic_energy_spectrum_data.h"
#include "../../incompressible_navier_stokes/postprocessor/lift_and_drag_data.h"
#include "../../incompressible_navier_stokes/postprocessor/pressure_difference_data.h"
#include "../../incompressible_navier_stokes/postprocessor/turbulent_channel_data.h"
#include "functionalities/print_functions.h"
#include "functionalities/restart_data.h"
#include "postprocessor/error_calculation_data.h"
#include "postprocessor/output_data.h"

namespace CompNS
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
 *  i.e., Euler euqations or Navier-Stokes equations
 */
enum class EquationType
{
  Undefined,
  Euler,
  NavierStokes
};

/*
 *  For energy boundary conditions, one can prescribe the temperature or the energy
 */
enum class EnergyBoundaryVariable
{
  Undefined,
  Energy,
  Temperature
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
 *
 *    Explicit Runge-Kutta methods
 */
enum class TemporalDiscretization
{
  Undefined,
  ExplRK, // specify order of time integration scheme (order = stages)
  ExplRK3Stage4Reg2C,
  ExplRK3Stage7Reg2, // optimized for maximum time step sizes in DG context
  ExplRK4Stage5Reg2C,
  ExplRK4Stage8Reg2, // optimized for maximum time step sizes in DG context
  ExplRK4Stage5Reg3C,
  ExplRK5Stage9Reg2S,
  SSPRK // specify order and stages of time integration scheme
};

/*
 *  For the BDF time integrator, the convective term can be either
 *  treated explicitly or implicitly
 */
enum class TreatmentOfConvectiveTerm
{
  Undefined,
  Explicit,
  Implicit
};

/*
 * calculation of time step size
 */
enum class TimeStepCalculation
{
  Undefined,
  UserSpecified,
  CFL,
  Diffusion,
  CFLAndDiffusion
};

/**************************************************************************************/
/*                                                                                    */
/*                               SPATIAL DISCRETIZATION                               */
/*                                                                                    */
/**************************************************************************************/



/**************************************************************************************/
/*                                                                                    */
/*                                       SOLVER                                       */
/*                                                                                    */
/**************************************************************************************/



/**************************************************************************************/
/*                                                                                    */
/*                               OUTPUT AND POSTPROCESSING                            */
/*                                                                                    */
/**************************************************************************************/

struct OutputDataCompNavierStokes : public OutputData
{
  OutputDataCompNavierStokes()
    : write_velocity(false),
      write_pressure(false),
      write_temperature(false),
      write_vorticity(false),
      write_divergence(false),
      write_processor_id(false)
  {
  }

  void
  print(ConditionalOStream & pcout, bool unsteady)
  {
    OutputData::print(pcout, unsteady);

    print_parameter(pcout, "Write velocity", write_velocity);
    print_parameter(pcout, "Write pressure", write_pressure);
    print_parameter(pcout, "Write temperature", write_temperature);
    print_parameter(pcout, "Write vorticity", write_vorticity);
    print_parameter(pcout, "Write divergence", write_divergence);
    print_parameter(pcout, "Write processor ID", write_processor_id);
  }

  // write velocity
  bool write_velocity;

  // write pressure
  bool write_pressure;

  // write temperature
  bool write_temperature;

  // write vorticity of velocity field
  bool write_vorticity;

  // write divergence of velocity field
  bool write_divergence;

  // write processor ID to scalar field in order to visualize the
  // distribution of cells to processors
  bool write_processor_id;
};

template<int dim>
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
      dynamic_viscosity(0.),
      reference_density(1.0),
      heat_capacity_ratio(1.4),
      thermal_conductivity(0.0262),
      specific_gas_constant(287.058),
      max_temperature(273.15),

      // TEMPORAL DISCRETIZATION
      temporal_discretization(TemporalDiscretization::Undefined),
      order_time_integrator(1),
      stages(1),
      calculation_of_time_step_size(TimeStepCalculation::Undefined),
      time_step_size(-1.),
      max_number_of_time_steps(std::numeric_limits<unsigned int>::max()),
      max_velocity(-1.),
      cfl_number(-1.),
      diffusion_number(-1.),
      exponent_fe_degree_cfl(2.0),
      exponent_fe_degree_viscous(4.0),

      // SPATIAL DISCRETIZATION
      degree_mapping(1),
      IP_factor(1.0),

      // SOLVER

      // NUMERICAL PARAMETERS
      detect_instabilities(true),
      use_combined_operator(false),

      // OUTPUT AND POSTPROCESSING
      print_input_parameters(false),
      calculate_velocity(false),
      calculate_pressure(false),

      // write output for visualization of results
      output_data(OutputDataCompNavierStokes()),

      // calculation of errors
      error_data(ErrorCalculationData()),

      output_solver_info_every_timesteps(1),

      // lift and drag
      lift_and_drag_data(LiftAndDragData()),

      // pressure difference
      pressure_difference_data(PressureDifferenceData<dim>()),

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


    // TEMPORAL DISCRETIZATION
    AssertThrow(temporal_discretization != TemporalDiscretization::Undefined,
                ExcMessage("parameter must be defined"));

    AssertThrow(calculation_of_time_step_size != TimeStepCalculation::Undefined,
                ExcMessage("parameter must be defined"));

    if(calculation_of_time_step_size == TimeStepCalculation::UserSpecified)
      AssertThrow(time_step_size > 0.0, ExcMessage("parameter must be defined"));

    if(temporal_discretization == TemporalDiscretization::ExplRK)
    {
      AssertThrow(order_time_integrator >= 1 && order_time_integrator <= 4,
                  ExcMessage("Specified order of time integrator ExplRK not implemented!"));
    }

    if(temporal_discretization == TemporalDiscretization::SSPRK)
    {
      AssertThrow(stages >= 1, ExcMessage("Specify number of RK stages!"));
    }

    if(calculation_of_time_step_size == TimeStepCalculation::CFLAndDiffusion)
    {
      AssertThrow(max_velocity >= 0.0, ExcMessage("Invalid parameter max_velocity."));
      AssertThrow(cfl_number > 0.0, ExcMessage("parameter must be defined"));
      AssertThrow(diffusion_number > 0.0, ExcMessage("parameter must be defined"));
    }


    // SPATIAL DISCRETIZATION
    AssertThrow(degree_mapping > 0, ExcMessage("Invalid parameter."));

    // SOLVER


    // NUMERICAL PARAMETERS


    // OUTPUT AND POSTPROCESSING
    if(lift_and_drag_data.calculate_lift_and_drag == true)
    {
      AssertThrow(
        calculate_velocity == true && calculate_pressure == true,
        ExcMessage(
          "Invalid parameters. One has to calculate velocity and pressure in order to compute lift and drag forces."));
    }
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
    // for the current implementation this means
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

    print_parameter(pcout, "Dynamic viscosity", dynamic_viscosity);
    print_parameter(pcout, "Reference density", reference_density);
    print_parameter(pcout, "Heat capacity ratio", heat_capacity_ratio);
    print_parameter(pcout, "Thermal conductivity", thermal_conductivity);
    print_parameter(pcout, "Specific gas constant", specific_gas_constant);
  }

  void
  print_parameters_temporal_discretization(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Temporal discretization:" << std::endl;

    std::string str_temp_discret[] = {"Undefined",
                                      "Classical, explicit Runge-Kutta",
                                      "ExplRK3Stage4Reg2C",
                                      "ExplRK3Stage7Reg2",
                                      "ExplRK4Stage5Reg2C",
                                      "ExplRK4Stage8Reg2",
                                      "ExplRK4Stage5Reg3C",
                                      "ExplRK5Stage9Reg2S",
                                      "SSPRK"};

    print_parameter(pcout,
                    "Temporal discretization method",
                    str_temp_discret[(int)temporal_discretization]);

    if(temporal_discretization == TemporalDiscretization::ExplRK)
    {
      print_parameter(pcout, "Order of time integrator", order_time_integrator);
    }

    if(temporal_discretization == TemporalDiscretization::SSPRK)
    {
      print_parameter(pcout, "Order of time integrator", order_time_integrator);
      print_parameter(pcout, "Number of stages", stages);
    }

    std::string str_time_step_calc[] = {
      "Undefined", "UserSpecified", "CFL", "Diffusion", "CFLAndDiffusion"};

    print_parameter(pcout,
                    "Calculation of time step size",
                    str_time_step_calc[(int)calculation_of_time_step_size]);

    // maximum number of time steps
    print_parameter(pcout, "Maximum number of time steps", max_number_of_time_steps);


    // here we do not print quantities such as  cfl_number, diffusion_number, time_step_size
    // because this is done by the time integration scheme (or the functions that
    // calculate the time step size)
  }

  void
  print_parameters_spatial_discretization(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Spatial Discretization:" << std::endl;

    print_parameter(pcout, "Polynomial degree of mapping", degree_mapping);

    print_parameter(pcout, "IP factor viscous term", IP_factor);
  }

  void
  print_parameters_solver(ConditionalOStream & /*pcout*/)
  {
    /*
    pcout << std::endl
          << "Solver:" << std::endl;
    */
  }


  void
  print_parameters_numerical_parameters(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Numerical parameters:" << std::endl;

    print_parameter(pcout, "Detect instabilities", detect_instabilities);
    print_parameter(pcout, "Use combined operator", use_combined_operator);
  }


  void
  print_parameters_output_and_postprocessing(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Output and postprocessing:" << std::endl;

    print_parameter(pcout, "Calculate velocity field", calculate_velocity);
    print_parameter(pcout, "Calculate pressure field", calculate_pressure);

    output_data.print(pcout, problem_type == ProblemType::Unsteady);
    error_data.print(pcout, problem_type == ProblemType::Unsteady);

    // kinetic energy
    kinetic_energy_data.print(pcout);

    // kinetic energy spectrum
    kinetic_energy_spectrum_data.print(pcout);

    // turbulent channel data
    turb_ch_data.print(pcout);

    // restart
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

  // dynamic viscosity
  double dynamic_viscosity;

  // reference density needed to calculate the kinematic viscosity from the specified
  // dynamic viscosity
  double reference_density;

  // heat_capacity_ratio
  double heat_capacity_ratio;

  // thermal conductivity
  double thermal_conductivity;

  // specific gas constant
  double specific_gas_constant;

  // maximum temperature (needed to calculate time step size according to CFL condition)
  double max_temperature;

  /**************************************************************************************/
  /*                                                                                    */
  /*                             TEMPORAL DISCRETIZATION                                */
  /*                                                                                    */
  /**************************************************************************************/

  // temporal discretization method
  TemporalDiscretization temporal_discretization;

  // order of time integration scheme
  unsigned int order_time_integrator;

  // number of Runge-Kutta stages
  unsigned int stages;

  // calculation of time step size
  TimeStepCalculation calculation_of_time_step_size;

  // user specified time step size:  note that this time_step_size is the first
  // in a series of time_step_size's when performing temporal convergence tests,
  // i.e., delta_t = time_step_size, time_step_size/2, ...
  double time_step_size;

  // maximum number of time steps
  unsigned int max_number_of_time_steps;

  // maximum velocity needed when calculating the time step according to cfl-condition
  double max_velocity;

  // cfl number
  double cfl_number;

  // diffusion number (relevant number for limitation of time step size
  // when treating the diffusive term explicitly)
  double diffusion_number;

  // exponent of fe_degree used in the calculation of the CFL time step size
  double exponent_fe_degree_cfl;

  // exponent of fe_degree used in the calculation of the diffusion time step size
  double exponent_fe_degree_viscous;

  /**************************************************************************************/
  /*                                                                                    */
  /*                              SPATIAL DISCRETIZATION                                */
  /*                                                                                    */
  /**************************************************************************************/

  // Polynomial degree of shape functions used for geometry approximation (mapping from
  // parameter space to physical space)
  unsigned int degree_mapping;

  // diffusive term: Symmetric interior penalty Galerkin (SIPG) discretization
  // interior penalty parameter scaling factor: default value is 1.0
  double IP_factor;

  /**************************************************************************************/
  /*                                                                                    */
  /*                                       SOLVER                                       */
  /*                                                                                    */
  /**************************************************************************************/



  /**************************************************************************************/
  /*                                                                                    */
  /*                                NUMERICAL PARAMETERS                                */
  /*                                                                                    */
  /**************************************************************************************/

  // detect instabilities (norm of solution vector grows by a large factor from one time
  // step to the next
  bool detect_instabilities;

  // use combined operator for viscous term and convective term in order to improve run
  // time
  bool use_combined_operator;

  /**************************************************************************************/
  /*                                                                                    */
  /*                               OUTPUT AND POSTPROCESSING                            */
  /*                                                                                    */
  /**************************************************************************************/

  // print a list of all input parameters at the beginning of the simulation
  bool print_input_parameters;

  // calculate velocity field
  bool calculate_velocity;

  // calculate pressure field
  bool calculate_pressure;

  // writing output
  OutputDataCompNavierStokes output_data;

  // calculation of errors
  ErrorCalculationData error_data;

  // show solver performance (wall time, number of iterations) every ... timesteps
  unsigned int output_solver_info_every_timesteps;

  // computation of lift and drag coefficients
  LiftAndDragData lift_and_drag_data;

  // computation of pressure difference between two points
  PressureDifferenceData<dim> pressure_difference_data;

  // kinetic energy
  KineticEnergyData kinetic_energy_data;

  // kinetic energy spectrum
  KineticEnergySpectrumData kinetic_energy_spectrum_data;

  // turbulent channel
  TurbulentChannelData turb_ch_data;

  // Restart
  RestartData restart_data;
};

} // namespace CompNS

#endif /* INCLUDE_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_INPUT_PARAMETERS_H_*/
