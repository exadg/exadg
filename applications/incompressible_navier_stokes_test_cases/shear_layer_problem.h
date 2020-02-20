/*
 * shear_layer_problem.h
 *
 *  Created on: May, 2019
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TAYLOR_VORTEX_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TAYLOR_VORTEX_H_

#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

/*
 * For a description of the test case, see
 *
 * Brown and Minion (J. Comput. Phys. 122 (1995), 165-183)
 */

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// convergence studies in space or time
unsigned int const DEGREE_MIN = 7;
unsigned int const DEGREE_MAX = 7;

unsigned int const REFINE_SPACE_MIN = 4;
unsigned int const REFINE_SPACE_MAX = 4;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;

bool const INVISCID = true;
double const VISCOSITY = INVISCID ? 0.0 : 1.0e-4; // Re = 10^4
double const RHO = 30.0;

std::string const OUTPUT_FOLDER = "output/shear_layer/";
std::string const OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string const OUTPUT_NAME = "l4_k7_bdf2_adaptive_cfl0-25";


namespace IncNS
{
void set_input_parameters(InputParameters &param)
{
  // MATHEMATICAL MODEL
  param.dim = 2;
  param.problem_type = ProblemType::Unsteady;
  if(INVISCID)
    param.equation_type = EquationType::Euler;
  else
    param.equation_type = EquationType::NavierStokes;
  param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
  param.right_hand_side = false;

  // PHYSICAL QUANTITIES
  param.start_time = 0.0;
  param.end_time = 4.0;
  param.viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  param.solver_type = SolverType::Unsteady;
  param.temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
  param.adaptive_time_stepping = true;
  param.adaptive_time_stepping_limiting_factor = 3.0;
  param.max_velocity = 1.5;
  param.cfl = 0.25;
  param.cfl_exponent_fe_degree_velocity = 1.5;
  param.time_step_size = 1.0e-4;
  param.order_time_integrator = 2;
  param.start_with_low_order = true;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = (param.end_time-param.start_time)/40;


  // SPATIAL DISCRETIZATION
  param.triangulation_type = TriangulationType::Distributed;
  param.degree_u = DEGREE_MIN;
  param.degree_p = DegreePressure::MixedOrder;
  param.mapping = MappingType::Isoparametric;
  param.h_refinements = REFINE_SPACE_MIN;

  // convective term
  if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
    param.upwind_factor = 0.5;

  // viscous term
  param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  param.pure_dirichlet_bc = true;
  param.adjust_pressure_level = AdjustPressureLevel::ApplyZeroMeanValue;

  // velocity-pressure coupling terms
  param.gradp_formulation = FormulationPressureGradientTerm::Weak;
  param.divu_formulation = FormulationVelocityDivergenceTerm::Weak;

  // penalty terms
  param.use_divergence_penalty = true;
  param.divergence_penalty_factor = 1.0e0;
  param.use_continuity_penalty = true;
  param.continuity_penalty_factor = param.divergence_penalty_factor;
  param.continuity_penalty_components = ContinuityPenaltyComponents::Normal;
  param.continuity_penalty_use_boundary_data = true;
  param.apply_penalty_terms_in_postprocessing_step = true;

  // PROJECTION METHODS

  // formulation
  param.store_previous_boundary_values = false;

  // pressure Poisson equation
  param.solver_pressure_poisson = SolverPressurePoisson::CG;
  param.solver_data_pressure_poisson = SolverData(1000,1.e-12,1.e-6,100);
  param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;
  param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
  param.multigrid_data_pressure_poisson.coarse_problem.solver = MultigridCoarseGridSolver::Chebyshev;
  param.multigrid_data_pressure_poisson.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::PointJacobi;

  // projection step
  param.solver_projection = SolverProjection::CG;
  param.solver_data_projection = SolverData(1000, 1.e-12, 1.e-6);
  param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  param.order_extrapolation_pressure_nbc = param.order_time_integrator <=2 ? param.order_time_integrator : 2;

  // viscous step
  param.solver_viscous = SolverViscous::CG;
  param.solver_data_viscous = SolverData(1000,1.e-12,1.e-6);
  param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;
}

}

/************************************************************************************************************/
/*                                                                                                          */
/*                                       CREATE GRID AND SET BOUNDARY IDs                                   */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim>
void
create_grid_and_set_boundary_ids(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
                                 unsigned int const                            n_refine_space,
                                 std::vector<GridTools::PeriodicFacePair<typename
                                   Triangulation<dim>::cell_iterator> >        &periodic_faces)
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

/************************************************************************************************************/
/*                                                                                                          */
/*                                               MESH MOTION                                                */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim>
std::shared_ptr<Function<dim>>
set_mesh_movement_function()
{
  std::shared_ptr<Function<dim>> mesh_motion;
  mesh_motion.reset(new Functions::ZeroFunction<dim>(dim));

  return mesh_motion;
}

namespace IncNS
{

/************************************************************************************************************/
/*                                                                                                          */
/*                         FUNCTIONS (INITIAL/BOUNDARY CONDITIONS, RIGHT-HAND SIDE, etc.)                   */
/*                                                                                                          */
/************************************************************************************************************/

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

    const double rho = RHO;
    const double delta = 0.05;
    const double pi = numbers::PI;
    if(component == 0)
      result = std::tanh(rho * (0.25 - std::abs(0.5-p[1])));
    else if(component == 1)
      result = delta*std::sin(2.0*pi*p[0]);

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

/************************************************************************************************************/
/*                                                                                                          */
/*                                              POSTPROCESSOR                                               */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, Number> >
construct_postprocessor(InputParameters const &param, MPI_Comm const &mpi_comm)
{
  PostProcessorData<dim> pp_data;

  // write output for visualization of results
  pp_data.output_data.write_output = true;
  pp_data.output_data.output_folder = OUTPUT_FOLDER_VTU;
  pp_data.output_data.output_name = "output";
  pp_data.output_data.output_start_time = param.start_time;
  pp_data.output_data.output_interval_time = (param.end_time-param.start_time)/40;
  pp_data.output_data.write_divergence = true;
  pp_data.output_data.write_vorticity = true;
  pp_data.output_data.degree = param.degree_u;

  // kinetic energy
  pp_data.kinetic_energy_data.calculate = true;
  pp_data.kinetic_energy_data.evaluate_individual_terms = false;
  pp_data.kinetic_energy_data.calculate_every_time_steps = 1;
  pp_data.kinetic_energy_data.viscosity = VISCOSITY;
  pp_data.kinetic_energy_data.filename = OUTPUT_FOLDER + OUTPUT_NAME;

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data, mpi_comm));

  return pp;
}

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TAYLOR_VORTEX_H_ */
