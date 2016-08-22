/*
 * InputParametersConvDiff.h
 *
 *  Created on: Aug 17, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INPUTPARAMETERSCONVDIFF_H_
#define INCLUDE_INPUTPARAMETERSCONVDIFF_H_

#include "../include/PrintFunctions.h"

/**************************************************************************************/
/*                                                                                    */
/*                                 MATHEMATICAL MODEL                                 */
/*                                                                                    */
/**************************************************************************************/

/*
 *  EquationType describes the physical/mathematical model that has to be solved,
 *  i.e., diffusion problem, convective problem or convection-diffusion problem
 */
enum class EquationTypeConvDiff
{
  Undefined,
  Convection,
  Diffusion,
  ConvectionDiffusion
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


class InputParametersConvDiff
{
public:
  // standard constructor that initializes parameters with default values
  InputParametersConvDiff()
    :
    // MATHEMATICAL MODEL
    equation_type(EquationTypeConvDiff::Undefined),
    right_hand_side(false),

    // PHYSICAL QUANTITIES
    start_time(0.),
    end_time(-1.),
    diffusivity(0.),

    // TEMPORAL DISCRETIZATION
    order_time_integrator(1),
    cfl_number(-1.),
    diffusion_number(-1.),

    // SPATIAL DISCRETIZATION
    numerical_flux_convective_operator(NumericalFluxConvectiveOperator::Undefined),
    IP_factor(1.0),

    // NUMERICAL PARAMETERS
    runtime_optimization(false),

    // OUTPUT AND POSTPROCESSING
    print_input_parameters(false),
    write_output(false),
    output_prefix("solution"),
    output_start_time(std::numeric_limits<double>::max()),
    output_interval_time(std::numeric_limits<double>::max()),

    analytical_solution_available(false),
    error_calc_start_time(std::numeric_limits<double>::max()),
    error_calc_interval_time(std::numeric_limits<double>::max())
  {}

  void set_input_parameters();

  void check_input_parameters()
  {
    // MATHEMATICAL MODEL
    AssertThrow(equation_type != EquationTypeConvDiff::Undefined,
        ExcMessage("parameter must be defined"));

    // PHYSICAL QUANTITIES
    AssertThrow(end_time > start_time, ExcMessage("parameter must be defined"));

    // Set the diffusivity whenever the diffusive term is involved.
    if(equation_type == EquationTypeConvDiff::Diffusion ||
       equation_type == EquationTypeConvDiff::ConvectionDiffusion)
    AssertThrow(diffusivity > (0.0 + 1.0e-10), ExcMessage("parameter must be defined"));

    // TEMPORAL DISCRETIZATION
    AssertThrow(cfl_number > 0., ExcMessage("parameter must be defined"));
    AssertThrow(diffusion_number > 0., ExcMessage("parameter must be defined"));

    // SPATIAL DISCRETIZATION
    if(equation_type == EquationTypeConvDiff::Convection ||
       equation_type == EquationTypeConvDiff::ConvectionDiffusion ||
       runtime_optimization == true)
    {
      AssertThrow(numerical_flux_convective_operator != 
                  NumericalFluxConvectiveOperator::Undefined,
                  ExcMessage("parameter must be defined"));
    }

    // NUMERICAL PARAMETERS

    // OUTPUT AND POSTPROCESSING

  }



  void print(ConditionalOStream &pcout)
  {
    pcout << std::endl
          << "List of input parameters:" << std::endl;

    // MATHEMATICAL MODEL
    print_parameters_mathematical_model(pcout);

    // PHYSICAL QUANTITIES
    print_parameters_physical_quantities(pcout);

    // TEMPORAL DISCRETIZATION
    print_parameters_temporal_discretization(pcout);

    // SPATIAL DISCRETIZATION
    print_parameters_spatial_discretization(pcout);
 
    // NUMERICAL PARAMETERS
    print_parameters_numerical_parameters(pcout);
 
    // OUTPUT AND POSTPROCESSING
    print_parameters_output_and_postprocessing(pcout);
  }

  void print_parameters_mathematical_model(ConditionalOStream &pcout)
  {
     pcout << std::endl
           << "Mathematical model:" << std::endl;

    /*
     *  The definition of string-arrays in this function is somehow redundant with the 
     *  enum declarations but I think C++ does not offer a more elaborate conversion 
     *  from enums to strings
     */

     // equation type
     std::string str_equation_type[] = {"Undefined", 
                                        "Convection", 
                                        "Diffusion",
                                        "ConvectionDiffusion" };

     print_parameter(pcout, 
                     "Equation type", 
                     str_equation_type[(int)equation_type]);
     
    // right hand side
     print_parameter(pcout,"Right-hand side",right_hand_side);
  }
  
  void print_parameters_physical_quantities(ConditionalOStream &pcout)
  {
    pcout << std::endl
          << "Physical quantities:" << std::endl;

    // start and end time
    if(true /*problem_type == ProblemType::Unsteady*/)
    {
      print_parameter(pcout,"Start time",start_time);
      print_parameter(pcout,"End time",end_time);
    }
    
    // diffusivity
    if(equation_type == EquationTypeConvDiff::Diffusion ||
       equation_type == EquationTypeConvDiff::ConvectionDiffusion)
    {
      print_parameter(pcout,"Diffusivity",diffusivity);
    }
  }  

  void print_parameters_temporal_discretization(ConditionalOStream &pcout)
  {
    pcout << std::endl
          << "Physical quantities:" << std::endl;

    print_parameter(pcout,"Order of time integrator",order_time_integrator);


    // here we do not print quantities such as  cfl_number, diffusion_number, time_step_size
    // because this is done by the time integration scheme (or the functions that 
    // calculate the time step size)
  }

  void print_parameters_spatial_discretization(ConditionalOStream &pcout)
  {
    pcout << std::endl
          << "Spatial Discretization:" << std::endl;
   
    if(equation_type == EquationTypeConvDiff::Convection ||
       equation_type == EquationTypeConvDiff::ConvectionDiffusion)
    {
      std::string str_num_flux_convective[] = { "Undefined",
                                                "Central flux",
                                                "Lax-Friedrichs flux" };

      print_parameter(pcout,
                      "Numerical flux convective term",
                      str_num_flux_convective[(int)numerical_flux_convective_operator]);
    }

    if(equation_type == EquationTypeConvDiff::Diffusion ||
       equation_type == EquationTypeConvDiff::ConvectionDiffusion)
    { 
      print_parameter(pcout,"IP factor viscous term",IP_factor);
    }
  }


  void print_parameters_numerical_parameters(ConditionalOStream &pcout)
  {
    pcout << std::endl
          << "Numerical parameters:" << std::endl;

    print_parameter(pcout,"Runtime optimization",runtime_optimization);
  }


  void print_parameters_output_and_postprocessing(ConditionalOStream &pcout)
  {
    pcout << std::endl
          << "Output and postprocessing:" << std::endl;
   
    // output for visualization of results
    print_parameter(pcout,"Write output",write_output);
    if(write_output == true)
    {
      print_parameter(pcout,"Name of output files",output_prefix);
      if(true /*problem_type == ProblemType::Unsteady*/)
      {
        print_parameter(pcout,"Output start time",output_start_time);
        print_parameter(pcout,"Output interval time",output_interval_time);
      }
    }

    // calculation of error
    print_parameter(pcout,"Calculate error",analytical_solution_available);
    if(analytical_solution_available == true /*&&
       problem_type == ProblemType::Unsteady*/)
    {
      print_parameter(pcout,"Error calculation start time",error_calc_start_time);
      print_parameter(pcout,"Error calculation interval time",error_calc_interval_time);
    }
  }

 
  /**************************************************************************************/
  /*                                                                                    */
  /*                                 MATHEMATICAL MODEL                                 */
  /*                                                                                    */
  /**************************************************************************************/

  // description: see enum declaration
  EquationTypeConvDiff equation_type;

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

  // order of BDF time integration scheme and extrapolation scheme
  unsigned int order_time_integrator;

  // cfl number
  double cfl_number;

  // diffusion number (relevant number for limitation of time step size when treating the diffusive term explicitly)
  double diffusion_number;



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
  /*                                NUMERICAL PARAMETERS                                */
  /*                                                                                    */
  /**************************************************************************************/

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

  // set write_output = true in order to write files for visualization
  bool write_output;

  // name of generated output files
  std::string output_prefix;

  // before then no output will be written
  double output_start_time;

  // specifies the time interval in which output is written
  double output_interval_time;

  // to calculate the error an analytical solution to the problem has to be available
  bool analytical_solution_available;

  // before then no error calculation will be performed
  double error_calc_start_time;

  // specifies the time interval in which error calculation is performed
  double error_calc_interval_time;
};


#endif /* INCLUDE_INPUTPARAMETERSCONVDIFF_H_ */
