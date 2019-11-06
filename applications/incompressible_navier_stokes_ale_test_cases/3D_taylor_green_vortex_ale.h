/*
 * 3D_taylor_green_vortex.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_3D_TAYLOR_GREEN_VORTEX_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_3D_TAYLOR_GREEN_VORTEX_H_

#include "../grid_tools/deformed_cube_manifold.h"
#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"
#include "../grid_tools/mesh_movement_functions.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// convergence studies in space or time
unsigned int const DEGREE_MIN = 3;
unsigned int const DEGREE_MAX = 3;

unsigned int const REFINE_SPACE_MIN = 2;
unsigned int const REFINE_SPACE_MAX = 2;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;

// set problem specific parameters like physical dimensions, etc.
const double Re = 1600.0;

const double V_0 = 1.0;
const double L = 1.0;
const double p_0 = 0.0;

const double VISCOSITY = V_0*L/Re;
const double MAX_VELOCITY = V_0;
const double CHARACTERISTIC_TIME = L/V_0;

std::string OUTPUT_FOLDER = "output/taylor_green_vortex/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME = "test";

enum class MeshType{ Cartesian, Curvilinear };
const MeshType MESH_TYPE = MeshType::Cartesian;

// only relevant for Cartesian mesh
const unsigned int N_CELLS_1D_COARSE_GRID = 1;

// mesh movement
const double LEFT = - numbers::PI * L;
const double RIGHT = numbers::PI * L;
MeshMovementShape const MESH_MOVEMENT_SHAPE = MeshMovementShape::Sin;
MeshMovementAdvanceInTime const MESH_MOVEMENT_ADVANCE_IN_TIME = MeshMovementAdvanceInTime::SinSquared;
double const MESH_MOVEMENT_AMPLITUDE = 0.04;
double const MESH_MOVEMENT_FREQUENCY = 0.25;
double const SPATIAL_NUMBER_OF_OSCILLATIONS = 1.0;
bool const MESH_MOVEMENT_DAMPED_TOWARDS_BOUNDARIES = false;
const double START_TIME = 0.0;
const double END_TIME = 20.0*CHARACTERISTIC_TIME;

namespace IncNS
{
void set_input_parameters(InputParameters &param)
{
  // ALE
  param.ale_formulation                          = true;
  param.neumann_with_variable_normal_vector      = false;

  // MATHEMATICAL MODEL
  param.dim = 3;
  param.problem_type = ProblemType::Unsteady;
  param.equation_type = EquationType::NavierStokes;
  param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
  param.right_hand_side = false;

  // PHYSICAL QUANTITIES
  param.start_time = START_TIME;
  param.end_time = END_TIME;
  param.viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  param.solver_type = SolverType::Unsteady;
  param.temporal_discretization = TemporalDiscretization::BDFCoupledSolution; //BDFPressureCorrection; //BDFCoupledSolution;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit; //Explicit; //Implicit;
  param.time_integrator_oif = TimeIntegratorOIF::ExplRK2Stage2;
  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
  param.adaptive_time_stepping = true;
  param.max_velocity = MAX_VELOCITY;
  param.cfl_oif = 0.45; //0.2; //0.125;
  param.cfl = param.cfl_oif * 1.0;
  param.cfl_exponent_fe_degree_velocity = 1.5;
  param.time_step_size = 1.0e-3; // 1.0e-4;
  param.order_time_integrator = 2; // 1; // 2; // 3;
  param.start_with_low_order = true; // true; // false;
  param.dt_refinements = REFINE_TIME_MIN;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = CHARACTERISTIC_TIME;

  // NUMERICAL PARAMETERS
  param.implement_block_diagonal_preconditioner_matrix_free = false;
  param.use_cell_based_face_loops = false;

  // SPATIAL DISCRETIZATION
  param.triangulation_type = TriangulationType::Distributed;
  param.degree_u = DEGREE_MIN;
  param.degree_p = DegreePressure::MixedOrder;
  param.h_refinements = REFINE_SPACE_MIN;

  // mapping
  if(MESH_TYPE == MeshType::Cartesian)
    param.mapping = MappingType::Affine;
  else
    param.mapping = MappingType::Isoparametric;

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
  param.solver_data_pressure_poisson = SolverData(1000,1.e-12,1.e-3,100);
  param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;
  param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
  param.multigrid_data_pressure_poisson.coarse_problem.solver = MultigridCoarseGridSolver::Chebyshev;
  param.multigrid_data_pressure_poisson.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::PointJacobi;

  // projection step
  param.solver_projection = SolverProjection::CG;
  param.solver_data_projection = SolverData(1000, 1.e-12, 1.e-3);
  param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;
  param.multigrid_data_projection.type = MultigridType::phMG;
  param.multigrid_data_projection.smoother_data.smoother = MultigridSmoother::Chebyshev;
  param.multigrid_data_projection.coarse_problem.solver = MultigridCoarseGridSolver::Chebyshev;
  param.multigrid_data_projection.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::PointJacobi;
  param.preconditioner_block_diagonal_projection = Elementwise::Preconditioner::InverseMassMatrix;
  param.solver_data_block_diagonal_projection = SolverData(1000,1.e-12,1.e-2,1000);
  param.update_preconditioner_projection = false;
  param.update_preconditioner_projection_every_time_steps = 10;

  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  param.order_extrapolation_pressure_nbc = param.order_time_integrator <=2 ? param.order_time_integrator : 2;

  // viscous step
  param.solver_viscous = SolverViscous::CG;
  param.solver_data_viscous = SolverData(1000,1.e-12,1.e-3);
  param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;
  param.multigrid_data_viscous.type = MultigridType::cphMG;
  param.multigrid_data_viscous.smoother_data.smoother = MultigridSmoother::Chebyshev; //Jacobi;
  param.multigrid_data_viscous.smoother_data.preconditioner = PreconditionerSmoother::PointJacobi; //BlockJacobi;
  param.multigrid_data_viscous.smoother_data.relaxation_factor = 0.7;
  param.update_preconditioner_viscous = false;

  // PRESSURE-CORRECTION SCHEME

  // momentum step

  // Newton solver
  param.newton_solver_data_momentum = NewtonSolverData(100,1.e-20,1.e-6);

  // linear solver
  param.solver_momentum = SolverMomentum::GMRES;
  if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
    param.solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-2, 100);
  else
    param.solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-6, 100);

  param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
  param.update_preconditioner_momentum = true;

  // formulation
  param.order_pressure_extrapolation = param.order_time_integrator-1;
  param.rotational_formulation = true;


  // COUPLED NAVIER-STOKES SOLVER
  param.use_scaling_continuity = false;

  // nonlinear solver (Newton solver)
  param.newton_solver_data_coupled = NewtonSolverData(100,1.e-12,1.e-6);

  // linear solver
  param.solver_coupled = SolverCoupled::GMRES;
  param.solver_data_coupled = SolverData(1e3, 1.e-12, 1.e-6, 100);

  // preconditioning linear solver
  param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

  // preconditioner velocity/momentum block
  param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

  // preconditioner Schur-complement block
  param.preconditioner_pressure_block = SchurComplementPreconditioner::CahouetChabard; //PressureConvectionDiffusion;
  param.discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;
  param.exact_inversion_of_laplace_operator = false;
  param.solver_data_pressure_block = SolverData(1e4, 1.e-12, 1.e-6, 100);
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
  const double left = LEFT, right = RIGHT;
  std::vector<unsigned int> repetitions({N_CELLS_1D_COARSE_GRID,
                                         N_CELLS_1D_COARSE_GRID,
                                         N_CELLS_1D_COARSE_GRID});

  Point<dim> point1(left,left,left), point2(right,right,right);
  GridGenerator::subdivided_hyper_rectangle(*triangulation,repetitions,point1,point2);

  if(MESH_TYPE == MeshType::Cartesian)
  {
    // do nothing
  }
  else if(MESH_TYPE == MeshType::Curvilinear)
  {
    double const deformation = 0.5;
    unsigned int const frequency = 2;
    static DeformedCubeManifold<dim> manifold(left, right, deformation, frequency);
    triangulation->set_all_manifold_ids(1);
    triangulation->set_manifold(1, manifold);

    std::vector<bool> vertex_touched(triangulation->n_vertices(), false);

    for(typename Triangulation<dim>::cell_iterator cell = triangulation->begin();
        cell != triangulation->end(); ++cell)
    {
      for (unsigned int v=0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
      {
        if (vertex_touched[cell->vertex_index(v)]==false)
        {
          Point<dim> &vertex = cell->vertex(v);
          Point<dim> new_point = manifold.push_forward(vertex);
          vertex = new_point;
          vertex_touched[cell->vertex_index(v)] = true;
        }
      }
    }
  }

  AssertThrow(dim == 3, ExcMessage("This test case can only be used for dim==3!"));

  typename Triangulation<dim>::cell_iterator cell = triangulation->begin(), endc = triangulation->end();
  for(;cell!=endc;++cell)
  {
   for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
   {
     // x-direction
     if((std::fabs(cell->face(face_number)->center()(0) - left)< 1e-12))
       cell->face(face_number)->set_all_boundary_ids (0);
     else if((std::fabs(cell->face(face_number)->center()(0) - right)< 1e-12))
       cell->face(face_number)->set_all_boundary_ids (1);
     // y-direction
     else if((std::fabs(cell->face(face_number)->center()(1) - left)< 1e-12))
       cell->face(face_number)->set_all_boundary_ids (2);
     else if((std::fabs(cell->face(face_number)->center()(1) - right)< 1e-12))
       cell->face(face_number)->set_all_boundary_ids (3);
     // z-direction
     else if((std::fabs(cell->face(face_number)->center()(2) - left)< 1e-12))
       cell->face(face_number)->set_all_boundary_ids (4);
     else if((std::fabs(cell->face(face_number)->center()(2) - right)< 1e-12))
       cell->face(face_number)->set_all_boundary_ids (5);
   }
  }

  auto tria = dynamic_cast<Triangulation<dim>*>(&*triangulation);
  GridTools::collect_periodic_faces(*tria, 0, 1, 0 /*x-direction*/, periodic_faces);
  GridTools::collect_periodic_faces(*tria, 2, 3, 1 /*y-direction*/, periodic_faces);
  GridTools::collect_periodic_faces(*tria, 4, 5, 2 /*z-direction*/, periodic_faces);

  triangulation->add_periodicity(periodic_faces);

  // perform global refinements
  triangulation->refine_global(n_refine_space);
}

/************************************************************************************************************/
/*                                                                                                          */
/*                         FUNCTIONS (INITIAL/BOUNDARY CONDITIONS, RIGHT-HAND SIDE, etc.)                   */
/*                                                                                                          */
/************************************************************************************************************/

namespace IncNS
{

/*
 *  This function is used to prescribe initial conditions for the velocity field
 */
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

    if (component == 0)
      result = V_0*std::sin(p[0]/L)*std::cos(p[1]/L)*std::cos(p[2]/L);
    else if (component == 1)
      result = -V_0*std::cos(p[0]/L)*std::sin(p[1]/L)*std::cos(p[2]/L);
    else if (component == 2)
      result = 0.0;

    return result;
  }
};

template<int dim>
class InitialSolutionPressure : public Function<dim>
{
public:
  InitialSolutionPressure (const double time = 0.)
    :
    Function<dim>(1 /*n_components*/, time)
  {}

  double value (const Point<dim>   &p,
                const unsigned int /*component*/) const
  {
    double result = 0.0;

    result = p_0 + V_0 * V_0 / 16.0 * (std::cos(2.0*p[0]/L) + std::cos(2.0*p[1]/L)) * (std::cos(2.0*p[2]/L) + 2.0);

    return result;
  }
};


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
  field_functions->initial_solution_velocity.reset(new InitialSolutionVelocity<dim>());
  field_functions->initial_solution_pressure.reset(new InitialSolutionPressure<dim>());
  field_functions->analytical_solution_pressure.reset(new InitialSolutionPressure<dim>());
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));

  MeshMovementData<dim> data;
  data.temporal = MESH_MOVEMENT_ADVANCE_IN_TIME;
  data.shape = MESH_MOVEMENT_SHAPE;
  data.dimensions[0] = std::abs(RIGHT-LEFT);
  data.dimensions[1] = std::abs(RIGHT-LEFT);
  data.dimensions[2] = std::abs(RIGHT-LEFT);
  data.amplitude = MESH_MOVEMENT_AMPLITUDE;
  data.frequency = MESH_MOVEMENT_FREQUENCY;
  data.t_start = START_TIME;
  data.t_end = END_TIME;
  data.spatial_number_of_oscillations = SPATIAL_NUMBER_OF_OSCILLATIONS;
  data.damp_towards_bondaries = MESH_MOVEMENT_DAMPED_TOWARDS_BOUNDARIES;
  field_functions->mesh_movement.reset(new CubeMeshMovementFunctions<dim>(data));
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
  PostProcessorData<dim> pp_data;

  // write output for visualization of results
  pp_data.output_data.write_output = true;
  pp_data.output_data.output_folder = OUTPUT_FOLDER_VTU;
  pp_data.output_data.output_name = OUTPUT_NAME;
  pp_data.output_data.output_start_time = param.start_time;
  pp_data.output_data.output_interval_time = (param.end_time-param.start_time)/20;
  pp_data.output_data.write_vorticity = true;
  pp_data.output_data.write_divergence = true;
  pp_data.output_data.write_velocity_magnitude = true;
  pp_data.output_data.write_vorticity_magnitude = true;
  pp_data.output_data.write_q_criterion = true;
  pp_data.output_data.write_processor_id = true;
  pp_data.output_data.degree = param.degree_u;

  // calculate div and mass error
  pp_data.mass_data.calculate_error = false; //true;
  pp_data.mass_data.start_time = 0.0;
  pp_data.mass_data.sample_every_time_steps = 1e2;
  pp_data.mass_data.filename_prefix = OUTPUT_FOLDER + OUTPUT_NAME;
  pp_data.mass_data.reference_length_scale = 1.0;

  // kinetic energy
  pp_data.kinetic_energy_data.calculate = true;
  pp_data.kinetic_energy_data.evaluate_individual_terms = true;
  pp_data.kinetic_energy_data.calculate_every_time_steps = 1;
  pp_data.kinetic_energy_data.viscosity = VISCOSITY;
  pp_data.kinetic_energy_data.filename_prefix = OUTPUT_FOLDER + OUTPUT_NAME;

  // kinetic energy spectrum
  pp_data.kinetic_energy_spectrum_data.calculate = true;
  pp_data.kinetic_energy_spectrum_data.calculate_every_time_interval = 0.5;
  pp_data.kinetic_energy_spectrum_data.filename_prefix = OUTPUT_FOLDER + OUTPUT_NAME + "_energy_spectrum";
  pp_data.kinetic_energy_spectrum_data.degree = param.degree_u;
  pp_data.kinetic_energy_spectrum_data.evaluation_points_per_cell = (param.degree_u + 1) * 2.0;

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data));

  return pp;
}

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_3D_TAYLOR_GREEN_VORTEX_H_ */
