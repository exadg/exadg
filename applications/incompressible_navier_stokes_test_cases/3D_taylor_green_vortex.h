/*
 * 3D_taylor_green_vortex.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_3D_TAYLOR_GREEN_VORTEX_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_3D_TAYLOR_GREEN_VORTEX_H_

#include "../grid_tools/periodic_box.h"
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

unsigned int const REFINE_SPACE_MIN = 4;
unsigned int const REFINE_SPACE_MAX = 4;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;

// set problem specific parameters like physical dimensions, etc.
double const Re = 1600.0;

double const V_0 = 1.0;
double const L = 1.0;
double const p_0 = 0.0;

double const pi = numbers::PI;
double const LEFT = - pi * L, RIGHT = pi * L;

double const VISCOSITY = V_0*L/Re;
double const MAX_VELOCITY = V_0;
double const CHARACTERISTIC_TIME = L/V_0;
double const END_TIME = 20.0*CHARACTERISTIC_TIME;

std::string const OUTPUT_FOLDER = "output/taylor_green_vortex/";
std::string const  OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string const OUTPUT_NAME = "test_exploit_symmetry"; //"test_ale";

enum class MeshType{ Cartesian, Curvilinear };
MeshType const MESH_TYPE = MeshType::Cartesian;

// only relevant for Cartesian mesh
unsigned int const N_CELLS_1D_COARSE_GRID = 1;

// moving mesh
bool const ALE = false; //true;

bool const EXPLOIT_SYMMETRIE = false; //true;

namespace IncNS
{
void set_input_parameters(InputParameters &param)
{
  // MATHEMATICAL MODEL
  param.dim = 3;
  param.problem_type = ProblemType::Unsteady;
  param.equation_type = EquationType::NavierStokes;
  param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation; //DivergenceFormulation;
  param.right_hand_side = false;

  // ALE
  param.ale_formulation                     = ALE;
  param.neumann_with_variable_normal_vector = false;

  // PHYSICAL QUANTITIES
  param.start_time = 0.0;
  param.end_time = END_TIME;
  param.viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  param.solver_type = SolverType::Unsteady;
  param.temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme; //BDFPressureCorrection; //BDFCoupledSolution;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit; //Explicit; //Implicit;
  param.time_integrator_oif = TimeIntegratorOIF::ExplRK2Stage2;
  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
  param.adaptive_time_stepping = true;
  param.max_velocity = MAX_VELOCITY;
  param.cfl_oif = 0.2; //0.2; //0.125;
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
  param.continuity_penalty_components = ContinuityPenaltyComponents::Normal;
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

  // formulation
  // this test case only has periodic boundaries so that this parameter is not used.
  // Deactivate in order to reduce memory requirements
  param.store_previous_boundary_values = false;

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
  double const deformation = 0.5;

  if(ALE)
  {
    AssertThrow(MESH_TYPE == MeshType::Cartesian,
        ExcMessage("Taylor-Green vortex: Parameter MESH_TYPE is invalid for ALE."));
  }

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

  if(EXPLOIT_SYMMETRIE == false)
  {
    create_periodic_box(triangulation,
                        n_refine_space,
                        periodic_faces,
                        N_CELLS_1D_COARSE_GRID,
                        LEFT,
                        RIGHT,
                        curvilinear_mesh,
                        deformation);
  }
  else
  {
    GridGenerator::subdivided_hyper_cube(*triangulation, N_CELLS_1D_COARSE_GRID, 0.0, RIGHT);

    if(curvilinear_mesh)
    {
      unsigned int const               frequency = 2;
      static DeformedCubeManifold<dim> manifold(0.0, RIGHT, deformation, frequency);
      triangulation->set_all_manifold_ids(1);
      triangulation->set_manifold(1, manifold);

      std::vector<bool> vertex_touched(triangulation->n_vertices(), false);

      for(typename Triangulation<dim>::cell_iterator cell = triangulation->begin();
          cell != triangulation->end();
          ++cell)
      {
        for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          if(vertex_touched[cell->vertex_index(v)] == false)
          {
            Point<dim> & vertex                   = cell->vertex(v);
            Point<dim>   new_point                = manifold.push_forward(vertex);
            vertex                                = new_point;
            vertex_touched[cell->vertex_index(v)] = true;
          }
        }
      }
    }

    // perform global refinements
    triangulation->refine_global(n_refine_space);
  }
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
    std::shared_ptr<BoundaryDescriptorU<dim> > boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> > boundary_descriptor_pressure)
{
  if(EXPLOIT_SYMMETRIE)
  {
    typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

    boundary_descriptor_velocity->symmetry_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim))); // function will not be used
    boundary_descriptor_pressure->neumann_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim))); // dg_u/dt=0 for dual splitting
  }
  else
  {
    // test case with pure periodic BC
    // boundary descriptors remain empty for velocity and pressure
  }
}


template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution_velocity.reset(new InitialSolutionVelocity<dim>());
  field_functions->initial_solution_pressure.reset(new InitialSolutionPressure<dim>());
  field_functions->analytical_solution_pressure.reset(new InitialSolutionPressure<dim>());
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));

  if(ALE)
  {
    MeshMovementData<dim> data;
    data.temporal = MeshMovementAdvanceInTime::Sin;
    data.shape = MeshMovementShape::Sin;
    data.dimensions[0] = std::abs(RIGHT-LEFT);
    data.dimensions[1] = std::abs(RIGHT-LEFT);
    data.dimensions[2] = std::abs(RIGHT-LEFT);
    data.amplitude = RIGHT/6.0; // use a value <= RIGHT/4.0
    data.period = END_TIME/40.0; // END_TIME/2.0;
    data.t_start = 0.0;
    data.t_end = END_TIME;
    data.spatial_number_of_oscillations = 1.0;
    field_functions->mesh_movement.reset(new CubeMeshMovementFunctions<dim>(data));
  }
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
  pp_data.mass_data.calculate_error = false;
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
  pp_data.kinetic_energy_spectrum_data.calculate = false; //true;
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
