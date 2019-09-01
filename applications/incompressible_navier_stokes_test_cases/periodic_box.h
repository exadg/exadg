/*
 * deformed_cube.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_3D_TAYLOR_GREEN_VORTEX_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_3D_TAYLOR_GREEN_VORTEX_H_

#include "../grid_tools/periodic_box.h"
#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// convergence studies in space or time
unsigned int const DEGREE_MIN = 1;
unsigned int const DEGREE_MAX = 10;

unsigned int const REFINE_SPACE_MIN = 0;
unsigned int const REFINE_SPACE_MAX = 0;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;

// set problem specific parameters like physical dimensions, etc.

enum class MeshType{ Cartesian, Curvilinear };
const MeshType MESH_TYPE = MeshType::Cartesian;

namespace IncNS
{
void set_input_parameters(InputParameters &param)
{
  // MATHEMATICAL MODEL
  param.dim = 3;
  param.problem_type = ProblemType::Unsteady;
  param.equation_type = EquationType::NavierStokes;
  param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
  param.right_hand_side = false;

  // PHYSICAL QUANTITIES
  param.start_time = 0.0;
  param.end_time = 1.0;
  param.viscosity = 1.0;

  // TEMPORAL DISCRETIZATION
  param.solver_type = SolverType::Unsteady;
  param.temporal_discretization = TemporalDiscretization::BDFPressureCorrection; //BDFCoupledSolution;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit;
  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
  param.cfl = 1.0;
  param.max_velocity = 1.0;

  // NUMERICAL PARAMETERS
  param.quad_rule_linearization = QuadratureRuleLinearization::Standard; //Overintegration32k;

  // SPATIAL DISCRETIZATION
  param.triangulation_type = TriangulationType::Distributed;
  param.degree_u = DEGREE_MIN;
  param.degree_p = DegreePressure::EqualOrder; //MixedOrder; // use EqualOrder so that we can also start with k=1 for the velocity!
  param.h_refinements = REFINE_SPACE_MIN;

  // mapping
  param.mapping = MappingType::Affine; // Isoparametric;

  // convective term
  if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
    param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

  // viscous term
  param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's (only periodic BCs -> pure_dirichlet_bc = true)
  param.pure_dirichlet_bc = true;

  // div-div and continuity penalty
  param.use_divergence_penalty = true;
  param.divergence_penalty_factor = 1.0e0;
  param.use_continuity_penalty = true;
  param.continuity_penalty_factor = param.divergence_penalty_factor;
  param.add_penalty_terms_to_monolithic_system = false;

  // TURBULENCE
  param.use_turbulence_model = false;
  param.turbulence_model = TurbulenceEddyViscosityModel::Sigma;
  // Smagorinsky: 0.165
  // Vreman: 0.28
  // WALE: 0.50
  // Sigma: 1.35
  param.turbulence_model_constant = 1.35;

  // PROJECTION METHODS

  // pressure Poisson equation
  param.solver_pressure_poisson = SolverPressurePoisson::CG;
  param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::None;

  // projection step
  param.solver_projection = SolverProjection::CG;
  param.preconditioner_projection = PreconditionerProjection::None;

  // HIGH-ORDER DUAL SPLITTING SCHEME

  // viscous step
  param.solver_viscous = SolverViscous::CG;
  param.preconditioner_viscous = PreconditionerViscous::None;

  // PRESSURE-CORRECTION SCHEME

  // momentum step

  // linear solver
  param.solver_momentum = SolverMomentum::GMRES;
  param.preconditioner_momentum = MomentumPreconditioner::None;

  // COUPLED NAVIER-STOKES SOLVER

  // linear solver
  param.solver_coupled = SolverCoupled::GMRES;

  // preconditioning linear solver
  param.preconditioner_coupled = PreconditionerCoupled::None;

  param.preconditioner_velocity_block = MomentumPreconditioner::None;
  param.preconditioner_pressure_block = SchurComplementPreconditioner::None;
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
                                 unsigned int const                                n_refine_space,
                                 std::vector<GridTools::PeriodicFacePair<typename
                                   Triangulation<dim>::cell_iterator> >            &periodic_faces,
                                 unsigned int const                                n_subdivisions = 1)
{
  double const left = -1.0, right = 1.0;
  double const deformation = 0.1;

  bool curvilinear_mesh = false;
  if(MESH_TYPE == MeshType::Cartesian)
  {
    // do nothing
  }
  else if(MESH_TYPE == MeshType::Curvilinear)
  {
    curvilinear_mesh = true;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  create_periodic_box(triangulation,
                      n_refine_space,
                      periodic_faces,
                      n_subdivisions,
                      left,
                      right,
                      curvilinear_mesh,
                      deformation);
}

/************************************************************************************************************/
/*                                                                                                          */
/*                         FUNCTIONS (INITIAL/BOUNDARY CONDITIONS, RIGHT-HAND SIDE, etc.)                   */
/*                                                                                                          */
/************************************************************************************************************/

namespace IncNS
{

template<int dim>
void set_boundary_conditions(
    std::shared_ptr<BoundaryDescriptorU<dim> > /*boundary_descriptor_velocity*/,
    std::shared_ptr<BoundaryDescriptorP<dim> > /*boundary_descriptor_pressure*/)
{
  // test case with pure periodic BC
  // boundary descriptors remain empty for velocity and pressure
}


template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
  field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(dim));
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
construct_postprocessor(InputParameters const &param)
{
  (void)param;

  PostProcessorData<dim> pp_data;

  // no postprocessing

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data));

  return pp;
}

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_3D_TAYLOR_GREEN_VORTEX_H_ */
