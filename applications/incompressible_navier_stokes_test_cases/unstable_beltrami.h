/*
 * unstable_beltrami.h
 *
 *  Created on: July, 2018
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BELTRAMI_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BELTRAMI_H_

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

// single or double precision?
//typedef float VALUE_TYPE;
typedef double VALUE_TYPE;

// set the number of space dimensions: dimension = 2, 3
unsigned int const DIMENSION = 3;

// set the polynomial degree of the shape functions for velocity and pressure
unsigned int const FE_DEGREE_VELOCITY = 4;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 2;
unsigned int const REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_MIN;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// set problem specific parameters like physical dimensions, etc.
const double VISCOSITY = 1.0/(2.0e3);

// output folders and names
std::string OUTPUT_FOLDER = "output/unstable_beltrami/div_normal_conti_penalty_CFL0-2_reltol_1e-8/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME = "nu_inf_l2_k87";

template<int dim>
void InputParameters<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::NavierStokes;
  formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
  right_hand_side = false;

  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = 20.0;
  viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  solver_type = SolverType::Unsteady;
  temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  calculation_of_time_step_size = TimeStepCalculation::CFL;
  max_velocity = std::sqrt(2.0);
  cfl = 0.1;
  cfl_exponent_fe_degree_velocity = 1.5;
  time_step_size = 1.0e-3;
  order_time_integrator = 2; // 1; // 2; // 3;
  start_with_low_order = false; // true; // false;


  // SPATIAL DISCRETIZATION

  // triangulation
  triangulation_type = TriangulationType::Distributed;

  // mapping
  degree_mapping = FE_DEGREE_VELOCITY;

  // convective term

  // viscous term
  IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  pure_dirichlet_bc = true;
  adjust_pressure_level = AdjustPressureLevel::ApplyAnalyticalMeanValue;

  // div-div and continuity penalty
  use_divergence_penalty = true;
  divergence_penalty_factor = 1.0e0;
  use_continuity_penalty = true;
  continuity_penalty_factor = divergence_penalty_factor;
  add_penalty_terms_to_monolithic_system = false;


  // PROJECTION METHODS

  // pressure Poisson equation
  solver_pressure_poisson = SolverPressurePoisson::CG;
  solver_data_pressure_poisson = SolverData(1000,1.e-12,1.e-8,100);
  preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

  // projection step
  solver_projection = SolverProjection::CG;
  solver_data_projection = SolverData(1000, 1.e-12, 1.e-12);
  preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  order_extrapolation_pressure_nbc = order_time_integrator <=2 ? order_time_integrator : 2;

  // viscous step
  solver_viscous = SolverViscous::CG;
  solver_data_viscous = SolverData(1000,1.e-12,1.e-8);
  preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; //Multigrid;

  // PRESSURE-CORRECTION SCHEME

  // momentum step

  // Newton solver
  newton_solver_data_momentum = NewtonSolverData(100,1.e-12,1.e-8);

  // linear solver
  solver_momentum = SolverMomentum::GMRES;
  solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-8, 100);
  preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
  update_preconditioner_momentum = false;

  // formulation
  order_pressure_extrapolation = 1;
  rotational_formulation = true;


  // COUPLED NAVIER-STOKES SOLVER
  use_scaling_continuity = false;

  // nonlinear solver (Newton solver)
  newton_solver_data_coupled = NewtonSolverData(100,1.e-12,1.e-8);

  // linear solver
  solver_coupled = SolverCoupled::GMRES;
  solver_data_coupled = SolverData(1e3, 1.e-12, 1.e-8, 100);

  // preconditioning linear solver
  preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

  // preconditioner velocity/momentum block
  preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

  // preconditioner Schur-complement block
  preconditioner_pressure_block = SchurComplementPreconditioner::CahouetChabard;
  discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;


  // OUTPUT AND POSTPROCESSING

  // write output for visualization of results
  output_data.write_output = false; //true;
  output_data.output_folder = OUTPUT_FOLDER_VTU;
  output_data.output_name = OUTPUT_NAME;
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/100;
  output_data.write_divergence = false;
  output_data.degree = FE_DEGREE_VELOCITY;

  // calculation of error
  error_data.analytical_solution_available = true;
  error_data.calculate_relative_errors = true;
  error_data.calculate_H1_seminorm_velocity = true;
  error_data.error_calc_start_time = start_time;
  //error_data.error_calc_interval_time = (end_time-start_time);
  error_data.calculate_every_time_steps = 10;
  error_data.write_errors_to_file = true;
  error_data.filename_prefix = OUTPUT_FOLDER + OUTPUT_NAME + "_error";

  // kinetic energy
  kinetic_energy_data.calculate = true;
  kinetic_energy_data.calculate_every_time_steps = 10;
  kinetic_energy_data.viscosity = VISCOSITY;
  kinetic_energy_data.filename_prefix = OUTPUT_FOLDER + OUTPUT_NAME;

  // output of solver information
  solver_info_data.print_to_screen = true;
  solver_info_data.interval_time = (end_time-start_time)/10;
}

/**************************************************************************************/
/*                                                                                    */
/*                        GENERATE GRID AND SET BOUNDARY INDICATORS                   */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
void create_grid_and_set_boundary_ids(
    std::shared_ptr<parallel::Triangulation<dim>>     triangulation,
    unsigned int const                                n_refine_space,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >            &periodic_faces)
{
  AssertThrow(dim == 3, ExcMessage("This test case can only be used for dim==3!"));

  const double left = 0.0, right = 1.0;
  GridGenerator::hyper_cube(*triangulation,left,right);

  // periodicity in x-,y-, and z-direction

  // x-direction
  triangulation->begin()->face(0)->set_all_boundary_ids(0);
  triangulation->begin()->face(1)->set_all_boundary_ids(1);
  // y-direction
  triangulation->begin()->face(2)->set_all_boundary_ids(2);
  triangulation->begin()->face(3)->set_all_boundary_ids(3);
  // z-direction
  triangulation->begin()->face(4)->set_all_boundary_ids(4);
  triangulation->begin()->face(5)->set_all_boundary_ids(5);

  auto tria = dynamic_cast<Triangulation<dim>*>(&*triangulation);
  GridTools::collect_periodic_faces(*tria, 0, 1, 0 /*x-direction*/, periodic_faces);
  GridTools::collect_periodic_faces(*tria, 2, 3, 1 /*y-direction*/, periodic_faces);
  GridTools::collect_periodic_faces(*tria, 4, 5, 2 /*z-direction*/, periodic_faces);

  triangulation->add_periodicity(periodic_faces);

  // global refinements
  triangulation->refine_global(n_refine_space);
}

/**************************************************************************************/
/*                                                                                    */
/*    FUNCTIONS (ANALYTICAL SOLUTION, BOUNDARY CONDITIONS, VELOCITY FIELD, etc.)      */
/*                                                                                    */
/**************************************************************************************/

namespace IncNS
{

template<int dim>
class AnalyticalSolutionVelocity : public Function<dim>
{
public:
  AnalyticalSolutionVelocity (const unsigned int  n_components = dim,
                              const double        time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  double value (const Point<dim>    &p,
                const unsigned int  component = 0) const
  {
    double t = this->get_time();
    double result = 0.0;

    const double pi = numbers::PI;

    if (component == 0)
      result = std::sin(2.0*pi*p[0])*std::sin(2.0*pi*p[1]);
    else if (component == 1)
      result = std::cos(2.0*pi*p[0])*std::cos(2.0*pi*p[1]);
    else if (component == 2)
      result = std::sqrt(2.0)*std::sin(2.0*pi*p[0])*std::cos(2.0*pi*p[1]);

    result *= std::exp(-8.0*pi*pi*VISCOSITY*t);

    return result;
  }

  Tensor<1,dim,double> gradient (const Point<dim>    &p,
                                 const unsigned int  component = 0) const
  {
    double t = this->get_time();
    Tensor<1,dim,double> result;

    const double pi = numbers::PI;

    AssertThrow(dim==3, ExcMessage("not implemented."));

    if (component == 0)
    {
      result[0] = 2.0*pi*std::cos(2.0*pi*p[0])*std::sin(2.0*pi*p[1]);
      result[1] = 2.0*pi*std::sin(2.0*pi*p[0])*std::cos(2.0*pi*p[1]);
      result[2] = 0.0;
    }
    else if (component == 1)
    {
      result[0] = -2.0*pi*std::sin(2.0*pi*p[0])*std::cos(2.0*pi*p[1]);
      result[1] = -2.0*pi*std::cos(2.0*pi*p[0])*std::sin(2.0*pi*p[1]);
      result[2] = 0.0;
    }
    else if (component == 2)
    {
      result[0] = 2.0*pi*std::sqrt(2.0)*std::cos(2.0*pi*p[0])*std::cos(2.0*pi*p[1]);
      result[1] = -2.0*pi*std::sqrt(2.0)*std::sin(2.0*pi*p[0])*std::sin(2.0*pi*p[1]);
      result[2] = 0.0;
    }

    result *= std::exp(-8.0*pi*pi*VISCOSITY*t);

    return result;
  }
};


template<int dim>
class AnalyticalSolutionPressure : public Function<dim>
{
public:
  AnalyticalSolutionPressure (const double time = 0.)
    :
    Function<dim>(1 /*n_components*/, time)
  {}

  double value (const Point<dim>   &p,
                const unsigned int /*component*/) const
  {
    double t = this->get_time();
    const double pi = numbers::PI;

    double result = - 0.5*(+ std::sin(2.0*pi*p[0])*std::sin(2.0*pi*p[0])
                           + std::cos(2.0*pi*p[1])*std::cos(2.0*pi*p[1]) - 1.0);

    result *= std::exp(-16.0*pi*pi*VISCOSITY*t);

    return result;
  }
};

template<int dim>
class PressureBC_dudt : public Function<dim>
{
public:
  PressureBC_dudt (const double time = 0.)
    :
    Function<dim>(dim, time)
  {}

  double value (const Point<dim>    &p,
                const unsigned int  component = 0) const
  {
    double t = this->get_time();
    double result = 0.0;

    const double pi = numbers::PI;

    if (component == 0)
      result = std::sin(2.0*pi*p[0])*std::sin(2.0*pi*p[1]);
    else if (component == 1)
      result = std::cos(2.0*pi*p[0])*std::cos(2.0*pi*p[1]);
    else if (component == 2)
      result = std::sqrt(2.0)*std::sin(2.0*pi*p[0])*std::cos(2.0*pi*p[1]);

    result *= -8.0*pi*pi*VISCOSITY*std::exp(-8.0*pi*pi*VISCOSITY*t);

    return result;
  }
};

template<int dim>
void set_boundary_conditions(
    std::shared_ptr<BoundaryDescriptorU<dim> > /*boundary_descriptor_velocity*/,
    std::shared_ptr<BoundaryDescriptorP<dim> > /*boundary_descriptor_pressure*/)
{

}

template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
  field_functions->initial_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
  field_functions->analytical_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
}

template<int dim>
void set_analytical_solution(std::shared_ptr<AnalyticalSolution<dim> > analytical_solution)
{
  analytical_solution->velocity.reset(new AnalyticalSolutionVelocity<dim>());
  analytical_solution->pressure.reset(new AnalyticalSolutionPressure<dim>());
}

#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

template<int dim, int degree_u, int degree_p, typename Number>
std::shared_ptr<PostProcessorBase<dim, degree_u, degree_p, Number> >
construct_postprocessor(InputParameters<dim> const &param)
{
  PostProcessorData<dim> pp_data;

  pp_data.output_data = param.output_data;
  pp_data.error_data = param.error_data;
  pp_data.lift_and_drag_data = param.lift_and_drag_data;
  pp_data.pressure_difference_data = param.pressure_difference_data;
  pp_data.mass_data = param.mass_data;
  pp_data.kinetic_energy_data = param.kinetic_energy_data;

  std::shared_ptr<PostProcessor<dim,degree_u,degree_p,Number> > pp;
  pp.reset(new PostProcessor<dim,degree_u,degree_p,Number>(pp_data));

  return pp;
}

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_UNSTABLE_BELTRAMI_H_ */
