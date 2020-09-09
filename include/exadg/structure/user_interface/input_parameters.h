/*
 * input_parameters.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_STRUCTURE_USER_INTERFACE_INPUT_PARAMETERS_H_
#define INCLUDE_EXADG_STRUCTURE_USER_INTERFACE_INPUT_PARAMETERS_H_

// deal.II
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/grid/enum_types.h>
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_input_parameters.h>
#include <exadg/solvers_and_preconditioners/newton/newton_solver_data.h>
#include <exadg/solvers_and_preconditioners/solvers/solver_data.h>
#include <exadg/structure/user_interface/enum_types.h>
#include <exadg/time_integration/enum_types.h>
#include <exadg/time_integration/restart_data.h>
#include <exadg/time_integration/solver_info_data.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

class InputParameters
{
public:
  // standard constructor that initializes parameters with default values
  InputParameters()
    : // MATHEMATICAL MODEL
      problem_type(ProblemType::Undefined),
      body_force(false),
      large_deformation(false),
      pull_back_body_force(false),
      pull_back_traction(false),

      // PHYSICAL QUANTITIES
      density(1.0),

      // TEMPORAL DISCRETIZATION
      start_time(0.0),
      end_time(1.0),
      time_step_size(1.0),
      max_number_of_time_steps(std::numeric_limits<unsigned int>::max()),
      gen_alpha_type(GenAlphaType::GenAlpha),
      spectral_radius(1.0),
      solver_info_data(SolverInfoData()),
      restarted_simulation(false),
      restart_data(RestartData()),

      // quasi-static solver
      load_increment(1.0),
      adjust_load_increment(false),
      desired_newton_iterations(10),

      // SPATIAL DISCRETIZATION
      triangulation_type(TriangulationType::Undefined),
      mapping(MappingType::Affine),

      // SOLVER
      newton_solver_data(Newton::SolverData(1e4, 1.e-12, 1.e-6)),
      solver(Solver::Undefined),
      solver_data(SolverData(1e4, 1.e-12, 1.e-6, 100)),
      preconditioner(Preconditioner::AMG),
      update_preconditioner(false),
      update_preconditioner_every_time_steps(1),
      update_preconditioner_every_newton_iterations(10),
      multigrid_data(MultigridData())
  {
  }

  void
  check_input_parameters()
  {
    // MATHEMATICAL MODEL
    AssertThrow(problem_type != ProblemType::Undefined, ExcMessage("Parameter must be defined."));

    if(problem_type == ProblemType::QuasiStatic)
    {
      AssertThrow(large_deformation == true,
                  ExcMessage("QuasiStatic solver only implemented for nonlinear formulation."));
    }

    if(problem_type == ProblemType::Unsteady)
    {
      AssertThrow(restarted_simulation == false, ExcMessage("Restart has not been implemented."));
    }

    // SPATIAL DISCRETIZATION
    AssertThrow(triangulation_type != TriangulationType::Undefined,
                ExcMessage("Parameter must be defined."));

    // SOLVER
    AssertThrow(solver != Solver::Undefined, ExcMessage("Parameter must be defined."));
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

    print_parameter(pcout, "Body force", body_force);

    print_parameter(pcout, "Large deformation", large_deformation);

    if(large_deformation)
    {
      print_parameter(pcout, "Pull back body force", pull_back_body_force);
      print_parameter(pcout, "Pull back traction", pull_back_traction);
    }
  }

  void
  print_parameters_physical_quantities(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Physical quantities:" << std::endl;

    if(problem_type == ProblemType::Unsteady)
    {
      print_parameter(pcout, "Density", density);
    }
  }

  void
  print_parameters_temporal_discretization(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Temporal discretization:" << std::endl;

    if(problem_type == ProblemType::QuasiStatic)
    {
      print_parameter(pcout, "load_increment", load_increment);
      print_parameter(pcout, "Adjust load increment", adjust_load_increment);
      print_parameter(pcout, "Desired Newton iterations", desired_newton_iterations);
    }

    if(problem_type == ProblemType::Unsteady)
    {
      print_parameter(pcout, "Start time", start_time);
      print_parameter(pcout, "End time", end_time);
      print_parameter(pcout, "Max. number of time steps", max_number_of_time_steps);
      print_parameter(pcout, "Time integration type", enum_to_string(gen_alpha_type));
      print_parameter(pcout, "Spectral radius", spectral_radius);
      solver_info_data.print(pcout);
      if(restarted_simulation)
        restart_data.print(pcout);
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

    // nonlinear solver
    if(large_deformation)
    {
      pcout << std::endl << "Newton:" << std::endl;
      newton_solver_data.print(pcout);
    }

    // linear solver
    pcout << std::endl << "Linear solver:" << std::endl;
    print_parameter(pcout, "Solver", enum_to_string(solver));
    solver_data.print(pcout);

    // preconditioner for linear system of equations
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

  // set true in order to consider body forces
  bool body_force;

  // are large deformations to be expected, than compute with non linear method
  bool large_deformation;

  // For nonlinear problems with large deformations, it is important to specify whether
  // the body forces are formulated with respect to the current deformed configuration
  // or the reference configuration.
  //
  // Option 1: pull_back_body_force = false
  // In this case, the body force is specified as force per undeformed volume. A typical
  // use case are density-proportional forces such as the gravitational force. The body
  // is then directly described in reference space, b_0 = rho_0 * g, and the pull-back to
  // the reference configuration is deactivated.
  //
  // Option 2: pull_back_body_force = true
  // The body force is specified as force per deformed volume, and the body force needs
  // to be pulled-back according to b_0 = dv/dV * b, where the volume ratio dv/dV depends
  // on the current state of deformation.
  bool pull_back_body_force;

  // For nonlinear problems with large deformations, it is important to specify whether
  // the traction Neumann boundary condition is formulated with respect to the current
  // deformed configuration or the reference configuration. Both cases appear in practice,
  // so it needs to be specified by the user which formulation is to be used.
  //
  // Option 1: pull_back_traction = false
  // In this case, the traction is specified as a force per undeformed area, e.g., a
  // force of fixed amount distributed uniformly over a surface of the body. The force per
  // deformed area is an unknown. Hence, it is more natural to specify the traction in the
  // reference configuration and deactivate the pull-back from the current to the reference
  // configuration.
  //
  // Option 2: pull_back_traction = true
  // The traction is known as a force per area of the deformed body. In this case, the
  // traction needs to be pulled-back to the reference configuration, i.e., t_0 = da/dA * t,
  // where the surface area ratio da/dA depends on the current state of deformation.
  // A typical use case would be fluid-structure-interaction problems where the fluid
  // stresses are applied as traction boundary conditions for the structure. Note that
  // the direction of the traction vector does not change by this pull-back operation.
  bool pull_back_traction;

  /**************************************************************************************/
  /*                                                                                    */
  /*                                 PHYSICAL QUANTITIES                                */
  /*                                                                                    */
  /**************************************************************************************/

  // density rho_0 in initial configuration (only relevant for unsteady problems)
  double density;

  /**************************************************************************************/
  /*                                                                                    */
  /*                             TEMPORAL DISCRETIZATION                                */
  /*                                                                                    */
  /**************************************************************************************/

  double       start_time;
  double       end_time;
  double       time_step_size;
  unsigned int max_number_of_time_steps;

  GenAlphaType gen_alpha_type;

  // spectral radius rho_infty for generalized alpha time integration scheme
  double spectral_radius;

  // configure printing of solver performance (wall time, number of iterations)
  SolverInfoData solver_info_data;

  // set this variable to true to start the simulation from restart files
  bool restarted_simulation;

  // restart
  RestartData restart_data;

  // quasi-static solver

  // choose a value in [0,1] where 1 = maximum load (Neumann or Dirichlet)
  double load_increment;

  // adjust load increment adaptively depending on the number of iterations needed to
  // solve the system of equations
  bool adjust_load_increment;

  // in case of adaptively adjusting the load increment: specify a desired number of
  // Newton iterations according to which the load increment will be adjusted
  unsigned int desired_newton_iterations;

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

  // Newton solver data (only relevant for nonlinear problems)
  Newton::SolverData newton_solver_data;

  // description: see enum declaration
  Solver solver;

  // solver data
  SolverData solver_data;

  // description: see enum declaration
  Preconditioner preconditioner;

  // only relevant for nonlinear problems
  bool update_preconditioner;
  // ... every time steps (or load steps for QuasiStatic problems)
  unsigned int update_preconditioner_every_time_steps;
  // ... every Newton iterations
  unsigned int update_preconditioner_every_newton_iterations;

  // description: see declaration of MultigridData
  MultigridData multigrid_data;
};

} // namespace Structure
} // namespace ExaDG

#endif
