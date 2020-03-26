/*
 * input_parameters.h
 *
 *  Created on: Aug 17, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_INPUT_PARAMETERS_H_
#define INCLUDE_CONVECTION_DIFFUSION_INPUT_PARAMETERS_H_

// deal.II
#include <deal.II/base/exceptions.h>

#include "../../functionalities/enum_types.h"
#include "../../functionalities/restart_data.h"
#include "../../functionalities/solver_info_data.h"

#include "../../solvers_and_preconditioners/multigrid/multigrid_input_parameters.h"
#include "../../solvers_and_preconditioners/preconditioner/enum_types.h"
#include "../../solvers_and_preconditioners/solvers/enum_types.h"
#include "../../solvers_and_preconditioners/solvers/solver_data.h"
#include "../../time_integration/enum_types.h"
#include "enum_types.h"

namespace ConvDiff
{
class InputParameters
{
public:
  // standard constructor that initializes parameters with default values
  InputParameters();

  // check correctness of parameters
  void
  check_input_parameters();

  bool
  linear_system_including_convective_term_has_to_be_solved() const;

  bool
  convective_problem() const;

  bool
  diffusive_problem() const;

  bool
  linear_system_has_to_be_solved() const;

  TypeVelocityField
  get_type_velocity_field() const;

  // print functions
  void
  print(ConditionalOStream & pcout, std::string const & name);

private:
  void
  print_parameters_mathematical_model(ConditionalOStream & pcout);

  void
  print_parameters_physical_quantities(ConditionalOStream & pcout);

  void
  print_parameters_temporal_discretization(ConditionalOStream & pcout);

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

  // description: see enum declaration
  ProblemType problem_type;

  // description: see enum declaration
  EquationType equation_type;

  // Use true if an analytical function is used to prescribe the velocity field
  bool analytical_velocity_field;

  // Use Arbitrary Lagrangian-Eulerian (ALE) formulation
  bool ale_formulation;

  // set right_hand_side = true if the right-hand side f is unequal zero
  bool right_hand_side;

  // type of formulation of convective term
  FormulationConvectiveTerm formulation_convective_term;

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

  // description: see enum declaration (only relevant for explicit time integration)
  TimeIntegratorRK time_integrator_rk;

  // order of time integration scheme (only relevant for BDF time integration)
  unsigned int order_time_integrator;

  // start with low order (only relevant for BDF time integration)
  bool start_with_low_order;

  // description: see enum declaration (this parameter is ignored for steady problems or
  // unsteady problems with explicit Runge-Kutta time integration scheme). In case of
  // a purely diffusive problem, one also does not have to specify this parameter.
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

  // specify a maximum time step size in case of adaptive time stepping since the adaptive
  // time stepping algorithm would choose arbitrarily large time step sizes if the velocity field
  // is close to zero. This variable is only used for adaptive time stepping.
  double time_step_size_max;

  // Different variants are available for calculating the time step size based on a local CFL
  // criterion.
  CFLConditionType adaptive_time_stepping_cfl_type;

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

  // number of refinement steps for time step size
  //   default: use dt_refinements = 0
  unsigned int dt_refinements;

  // set this variable to true to start the simulation from restart files
  bool restarted_simulation;

  // Restart
  RestartData restart_data;

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
  Elementwise::Solver solver_block_diagonal;

  // description: see enum declaration
  Elementwise::Preconditioner preconditioner_block_diagonal;

  // solver data for block Jacobi preconditioner (only relevant for elementwise
  // iterative solution procedure)
  SolverData solver_data_block_diagonal;

  // description: see enum declaration
  MultigridOperatorType mg_operator_type;

  // description: see declaration of MultigridData
  MultigridData multigrid_data;

  // show solver performance (wall time, number of iterations)
  SolverInfoData solver_info_data;

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

  // Evaluate convective term and diffusive term at once instead of implementing each
  // operator separately and subsequently looping over all operators. This parameter is
  // only relevant in case of fully explicit time stepping. In case of semi-implicit or
  // fully implicit time integration the combined operator will always be used.
  bool use_combined_operator;

  // In case that the velocity field is prescribed analytically, it might be advantageous
  // from the point of view of computational costs to store the velocity field in a DoF
  // vector instead of repeatedly calling Function<dim>::value() whenever evaluating the
  // operator in iterative solvers. In other words, depending on the computer hardware it
  // might be more efficient to load a DoF vector and interpolate into the quadrature points
  // than calculating the velocity field by means of Function<dim>::value().
  // This strategy makes only sense in case of steady state problems or unsteady problems
  // with an implicit treatment of the convective term, i.e., in cases where the convective
  // term has to be evaluated more than once at a given time t.
  bool store_analytical_velocity_in_dof_vector;

  // filter solution after each time step
  bool filter_solution;

  // use 3/2 overintegration rule for convective term
  bool use_overintegration;
};

} // namespace ConvDiff
#endif /* INCLUDE_CONVECTION_DIFFUSION_INPUT_PARAMETERS_H_ */
