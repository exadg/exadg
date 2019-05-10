/*
 * TaylorVortex.h
 *
 *  Created on: Aug 19, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TAYLOR_VORTEX_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TAYLOR_VORTEX_H_

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

// shear layer roll-up problem (Fischer and Mullen, Filter-based stabilization of spectral element methods)

// set the number of space dimensions: dimension = 2, 3
unsigned int const DIMENSION = 2;

// set the polynomial degree of the shape functions for velocity and pressure
unsigned int const FE_DEGREE_VELOCITY = 7;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 5;
unsigned int const REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_MIN;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

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
  end_time = 4.0;
  viscosity = 1.0e-4; // Re = 10^4


  // TEMPORAL DISCRETIZATION
  solver_type = SolverType::Unsteady;
  temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  calculation_of_time_step_size = TimeStepCalculation::CFL;
  adaptive_time_stepping = true;
  max_velocity = 1.5;
  cfl = 0.4;
  cfl_exponent_fe_degree_velocity = 1.5;
  time_step_size = 1.0e-4;
  order_time_integrator = 2;
  start_with_low_order = true;


  // SPATIAL DISCRETIZATION

  // triangulation
  triangulation_type = TriangulationType::Distributed;

  // polynomial degrees
  degree_u = FE_DEGREE_VELOCITY;
  degree_p = FE_DEGREE_PRESSURE;

  // mapping
  degree_mapping = FE_DEGREE_VELOCITY;

  // convective term
  upwind_factor = 0.5;

  // viscous term
  IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  pure_dirichlet_bc = true;
  adjust_pressure_level = AdjustPressureLevel::ApplyZeroMeanValue;

  // PROJECTION METHODS

  // pressure Poisson equation
  solver_pressure_poisson = SolverPressurePoisson::CG;
  solver_data_pressure_poisson = SolverData(1000,1.e-12,1.e-6,100);
  preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;
  multigrid_data_pressure_poisson.type = MultigridType::phMG;
  multigrid_data_pressure_poisson.p_sequence = PSequenceType::Bisect;
  multigrid_data_pressure_poisson.dg_to_cg_transfer = DG_To_CG_Transfer::None;
  multigrid_data_pressure_poisson.coarse_problem.solver = MultigridCoarseGridSolver::Chebyshev;

  // projection step
  solver_projection = SolverProjection::CG;
  solver_data_projection = SolverData(1000, 1.e-12, 1.e-6);
  preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  order_extrapolation_pressure_nbc = order_time_integrator <=2 ? order_time_integrator : 2;

  // viscous step
  solver_viscous = SolverViscous::CG;
  solver_data_viscous = SolverData(1000,1.e-12,1.e-6);
  preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;


  // OUTPUT AND POSTPROCESSING

  // write output for visualization of results
  output_data.write_output = true;
  output_data.output_folder = "/data/fehn/navierstokes/applications/output/shear_layer/";
  output_data.output_name = "output";
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/400;
  output_data.write_divergence = true;
  output_data.write_vorticity = true;
  output_data.degree = FE_DEGREE_VELOCITY;

  // output of solver information
  solver_info_data.print_to_screen = true;
  solver_info_data.interval_time = (end_time-start_time)/400;
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
  const double left = 0.0, right = 1.0;
  GridGenerator::hyper_cube(*triangulation,left,right);

  // use periodic boundary conditions
  // x-direction
  triangulation->begin()->face(0)->set_all_boundary_ids(0+10);
  triangulation->begin()->face(1)->set_all_boundary_ids(1+10);
  // y-direction
  triangulation->begin()->face(2)->set_all_boundary_ids(2+10);
  triangulation->begin()->face(3)->set_all_boundary_ids(3+10);

  auto tria = dynamic_cast<Triangulation<dim>*>(&*triangulation);
  GridTools::collect_periodic_faces(*tria, 0+10, 1+10, 0, periodic_faces);
  GridTools::collect_periodic_faces(*tria, 2+10, 3+10, 1, periodic_faces);
  triangulation->add_periodicity(periodic_faces);

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
class InitialSolutionVelocity : public Function<dim>
{
public:
  InitialSolutionVelocity (const unsigned int  n_components = dim,
                              const double        time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  double value (const Point<dim>    &p,
                const unsigned int  component = 0) const
  {
    double result = 0.0;

    const double pi = numbers::PI;
    if(component == 0)
      if(p[1]<= 0.5)
        result = std::tanh(30.0 * (p[1]-0.25));
      else
        result = std::tanh(30.0 * (0.75-p[1]));
    else if(component == 1)
      result = 0.05*std::sin(2.0*pi*p[0]);

    return result;
  }
};

template<int dim>
void set_boundary_conditions(
    std::shared_ptr<BoundaryDescriptorU<dim> > /*boundary_descriptor_velocity*/,
    std::shared_ptr<BoundaryDescriptorP<dim> > /*boundary_descriptor_pressure*/)
{
  // nothing to do (periodic BCs)
}


template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution_velocity.reset(new InitialSolutionVelocity<dim>());
  field_functions->initial_solution_pressure.reset(new InitialSolutionVelocity<dim>());
  field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
}

template<int dim>
void set_analytical_solution(std::shared_ptr<AnalyticalSolution<dim> > analytical_solution)
{
  analytical_solution->velocity.reset(new InitialSolutionVelocity<dim>());
  analytical_solution->pressure.reset(new Functions::ZeroFunction<dim>(1));
}

#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, Number> >
construct_postprocessor(InputParameters<dim> const &param)
{
  PostProcessorData<dim> pp_data;

  pp_data.output_data = param.output_data;
  pp_data.error_data = param.error_data;
  pp_data.lift_and_drag_data = param.lift_and_drag_data;
  pp_data.pressure_difference_data = param.pressure_difference_data;
  pp_data.mass_data = param.mass_data;

  std::shared_ptr<PostProcessor<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data));

  return pp;
}

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TAYLOR_VORTEX_H_ */
