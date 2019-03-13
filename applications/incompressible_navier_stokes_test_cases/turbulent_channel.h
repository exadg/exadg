/*
 * TurbulentChannel.h
 *
 *  Created on: Oct 14, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_

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
unsigned int const FE_DEGREE_VELOCITY = 3;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY - 1;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 3;
unsigned int const REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_MIN;

// can only be used for GridStretchType::TransformGridCells, otherwise coarse grid consists of 1 cell
const unsigned int N_CELLS_1D_COARSE_GRID = 1;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

std::string OUTPUT_FOLDER = "output_comp_ns/turbulent_channel/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME = "Re180_l3_k3";

// set problem specific parameters like physical dimensions, etc.
double const DIMENSIONS_X1 = 2.0*numbers::PI;
double const DIMENSIONS_X2 = 2.0;
double const DIMENSIONS_X3 = numbers::PI;

// nu = 1/180  coarsest meshes: l2_ku3 or l3_ku2
// nu = 1/395
// nu = 1/590
// nu = 1/950
double const VISCOSITY = 1./180.; // critical value: 1./50. - 1./75.

//18.3 for Re_tau = 180
//20.1 for Re_tau = 395
//21.3 for Re_tau = 590
//22.4 for Re_tau = 950
double const MAX_VELOCITY = 18.3;

// flow-through time based on mean centerline velocity
const double CHARACTERISTIC_TIME = DIMENSIONS_X1/MAX_VELOCITY;

double const START_TIME = 0.0;
double const END_TIME = 200.0*CHARACTERISTIC_TIME; // 50.0;

double const SAMPLE_START_TIME = 100.0*CHARACTERISTIC_TIME; // 30.0;
double const SAMPLE_END_TIME = END_TIME;

// use a negative GRID_STRETCH_FAC to deactivate grid stretching
const double GRID_STRETCH_FAC = 1.8;

enum class GridStretchType{ TransformGridCells, VolumeManifold };
GridStretchType GRID_STRETCH_TYPE = GridStretchType::TransformGridCells; //VolumeManifold;

template<int dim>
void InputParameters<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::NavierStokes;
  formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation; //LaplaceFormulation; //DivergenceFormulation;
  right_hand_side = true;


  // PHYSICAL QUANTITIES
  start_time = START_TIME;
  end_time = END_TIME;
  viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  solver_type = SolverType::Unsteady;
  temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  calculation_of_time_step_size = TimeStepCalculation::CFL;
  max_velocity = MAX_VELOCITY;
  cfl = 1.7;
  cfl_exponent_fe_degree_velocity = 1.5;
  time_step_size = 1.0e-1;
  order_time_integrator = 2; // 1; // 2; // 3;
  start_with_low_order = true;


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

  // TURBULENCE
  use_turbulence_model = false;
  turbulence_model = TurbulenceEddyViscosityModel::Sigma;
  // Smagorinsky: 0.165
  // Vreman: 0.28
  // WALE: 0.50
  // Sigma: 1.35
  turbulence_model_constant = 1.35;

  // PROJECTION METHODS

  // pressure Poisson equation
  solver_pressure_poisson = SolverPressurePoisson::CG;
  solver_data_pressure_poisson = SolverData(1000,1.e-12,1.e-6,100);
  preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

  // projection step
  solver_projection = SolverProjection::CG;
  solver_data_projection = SolverData(1000, 1.e-12, 1.e-6);
  preconditioner_projection = PreconditionerProjection::InverseMassMatrix; //BlockJacobi; //PointJacobi; //InverseMassMatrix;
  update_preconditioner_projection = true;


  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  order_extrapolation_pressure_nbc = order_time_integrator <=2 ? order_time_integrator : 2;

  // viscous step
  solver_viscous = SolverViscous::CG;
  solver_data_viscous = SolverData(1000,1.e-12,1.e-6);
  preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; //Multigrid;

  // PRESSURE-CORRECTION SCHEME

  // formulation
  order_pressure_extrapolation = 1; // use 0 for non-incremental formulation
  rotational_formulation = true;

  // momentum step

  // Newton solver
  newton_solver_data_momentum = NewtonSolverData(100,1.e-12,1.e-6);

  // linear solver
  solver_momentum = SolverMomentum::GMRES;
  solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-6, 100);
  preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
  update_preconditioner_momentum = false;

  // COUPLED NAVIER-STOKES SOLVER
  use_scaling_continuity = false;

  // nonlinear solver (Newton solver)
  newton_solver_data_coupled = NewtonSolverData(100,1.e-12,1.e-6);

  // linear solver
  solver_coupled = SolverCoupled::GMRES; //GMRES; //FGMRES;
  solver_data_coupled = SolverData(1e3, 1.e-12, 1.e-6, 100);

  // preconditioning linear solver
  preconditioner_coupled = PreconditionerCoupled::BlockTriangular;
  update_preconditioner_coupled = false;

  // preconditioner velocity/momentum block
  preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix; //Multigrid;
  multigrid_data_velocity_block.smoother_data.smoother = MultigridSmoother::Jacobi; //Jacobi; //Chebyshev; //GMRES;
  multigrid_data_velocity_block.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi; //PointJacobi; //BlockJacobi;
  multigrid_data_velocity_block.smoother_data.iterations = 4;
  multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
  multigrid_data_velocity_block.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

  // preconditioner Schur-complement block
  preconditioner_pressure_block = SchurComplementPreconditioner::CahouetChabard; //PressureConvectionDiffusion;
  discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;


  // OUTPUT AND POSTPROCESSING

  // write output for visualization of results
  output_data.write_output = true;
  output_data.output_folder = OUTPUT_FOLDER_VTU;
  output_data.output_name = OUTPUT_NAME;
  output_data.output_start_time = start_time;
  output_data.output_interval_time = 1.0;
  output_data.write_divergence = true;
  output_data.degree = FE_DEGREE_VELOCITY;

  // output of solver information
  solver_info_data.print_to_screen = true;
  solver_info_data.interval_time = CHARACTERISTIC_TIME;

  // calculate div and mass error
  mass_data.calculate_error = false; //true;
  mass_data.start_time = START_TIME;
  mass_data.sample_every_time_steps = 1e0; //1e2;
  mass_data.filename_prefix = OUTPUT_FOLDER + OUTPUT_NAME;
  mass_data.reference_length_scale = 1.0;

  // turbulent channel statistics
  turb_ch_data.calculate_statistics = true;
  turb_ch_data.cells_are_stretched = false;
  if(GRID_STRETCH_TYPE == GridStretchType::VolumeManifold)
    turb_ch_data.cells_are_stretched = true;
  turb_ch_data.sample_start_time = SAMPLE_START_TIME;
  turb_ch_data.sample_end_time = SAMPLE_END_TIME;
  turb_ch_data.sample_every_timesteps = 10;
  turb_ch_data.viscosity = VISCOSITY;
  turb_ch_data.filename_prefix = OUTPUT_FOLDER + OUTPUT_NAME;
}

/**************************************************************************************/
/*                                                                                    */
/*    FUNCTIONS (ANALYTICAL SOLUTION, BOUNDARY CONDITIONS, VELOCITY FIELD, etc.)      */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
class AnalyticalSolutionVelocity : public Function<dim>
{
public:
  AnalyticalSolutionVelocity (const unsigned int  n_components = dim,
                              const double        time = 0.)
    :
    Function<dim>(n_components, time)
  {
    /*
     * Use the following line to obtain different initial velocity fields.
     * Without this line, the initial field is always the same as long as the
     * same number of processors is used.
     */
//    srand(std::time(NULL));
  }

  double value (const Point<dim>    &p,
                const unsigned int  component = 0) const
  {
    double result = 0.0;

    const double tol = 1.e-12;
    AssertThrow(std::abs(p[1])<DIMENSIONS_X2/2.0+tol,ExcMessage("Invalid geometry parameters."));

    AssertThrow(dim==3, ExcMessage("Dimension has to be dim==3."));

    // TODO
    if(component == 0)
      result = -MAX_VELOCITY*(pow(p[1],6.0)-1.0)*(1.0+((double)rand()/RAND_MAX-1.0)*0.5-2./MAX_VELOCITY*std::sin(p[2]*8.));
    else if(component == 2)
      result = (pow(p[1],6.0)-1.0)*std::sin(p[0]*8.)*2.;

  //  if(component == 0)
  //  {
  //    double factor = 1.0;
  //    result = -MAX_VELOCITY*(pow(p[1],6.0)-1.0)*(1.0+((double)rand()/RAND_MAX-0.5)*factor);
  //  }

    return result;
  }
};

template<int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide (const double time = 0.)
    :
    Function<dim>(dim, time)
  {}

  double value (const Point<dim>    &/*p*/,
                const unsigned int  component = 0) const
  {
    double result = 0.0;

    //channel flow with periodic bc
    if(component==0)
      return 1.0;
    else
      return 0.0;

    return result;
  }
};

/**************************************************************************************/
/*                                                                                    */
/*         GENERATE GRID, SET BOUNDARY INDICATORS AND FILL BOUNDARY DESCRIPTOR        */
/*                                                                                    */
/**************************************************************************************/

#include "../grid_tools/grid_functions_turbulent_channel.h"

template<int dim>
void create_grid_and_set_boundary_conditions(
    std::shared_ptr<parallel::Triangulation<dim>>     triangulation,
    unsigned int const                                n_refine_space,
    std::shared_ptr<BoundaryDescriptorU<dim> >        boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> >        boundary_descriptor_pressure,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >            &periodic_faces)
{
  /* --------------- Generate grid ------------------- */
  if(GRID_STRETCH_TYPE == GridStretchType::TransformGridCells)
  {
    Point<dim> coordinates;
    coordinates[0] = DIMENSIONS_X1;
    // dimension in y-direction is 2.0, function grid_transform() maps the y-coordinate from [0,1] to [-1,1]
    coordinates[1] = DIMENSIONS_X2/2.0;
    if (dim == 3)
     coordinates[2] = DIMENSIONS_X3;

    // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
    std::vector<unsigned int> refinements(dim, N_CELLS_1D_COARSE_GRID);
    GridGenerator::subdivided_hyper_rectangle (*triangulation, refinements,Point<dim>(),coordinates);

    typename Triangulation<dim>::cell_iterator cell = triangulation->begin(), endc = triangulation->end();
    for(;cell!=endc;++cell)
    {
      for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
      {
        // x-direction
        if((std::fabs(cell->face(face_number)->center()(0) - 0.0)< 1e-12))
          cell->face(face_number)->set_all_boundary_ids (0+10);
        else if((std::fabs(cell->face(face_number)->center()(0) - DIMENSIONS_X1)< 1e-12))
          cell->face(face_number)->set_all_boundary_ids (1+10);
        // z-direction
        else if((std::fabs(cell->face(face_number)->center()(2) - 0-0)< 1e-12))
          cell->face(face_number)->set_all_boundary_ids (2+10);
        else if((std::fabs(cell->face(face_number)->center()(2) - DIMENSIONS_X3)< 1e-12))
          cell->face(face_number)->set_all_boundary_ids (3+10);
      }
    }
  }
  else if (GRID_STRETCH_TYPE == GridStretchType::VolumeManifold)
  {
    Tensor<1,dim> dimensions;
    dimensions[0] = DIMENSIONS_X1;
    dimensions[1] = DIMENSIONS_X2;
    if(dim==3)
      dimensions[2] = DIMENSIONS_X3;

    // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
    AssertThrow(N_CELLS_1D_COARSE_GRID == 1,
        ExcMessage("Only N_CELLS_1D_COARSE_GRID=1 possible for curvilinear grid."));

    std::vector<unsigned int> refinements(dim, N_CELLS_1D_COARSE_GRID);
    GridGenerator::subdivided_hyper_rectangle (*triangulation, refinements,Point<dim>(-dimensions/2.0),Point<dim>(dimensions/2.0));

    // manifold
    unsigned int manifold_id = 1;
    for (typename Triangulation<dim>::cell_iterator cell = triangulation->begin(); cell != triangulation->end(); ++cell)
    {
      cell->set_all_manifold_ids(manifold_id);
    }

    // apply mesh stretching towards no-slip boundaries in y-direction
    static const ManifoldTurbulentChannel<dim> manifold(dimensions);
    triangulation->set_manifold(manifold_id, manifold);

    //periodicity in x- and z-direction
    //add 10 to avoid conflicts with dirichlet boundary, which is 0
    triangulation->begin()->face(0)->set_all_boundary_ids(0+10);
    triangulation->begin()->face(1)->set_all_boundary_ids(1+10);
    //periodicity in z-direction
    if (dim == 3)
    {
      triangulation->begin()->face(4)->set_all_boundary_ids(2+10);
      triangulation->begin()->face(5)->set_all_boundary_ids(3+10);
    }
  }

   auto tria = dynamic_cast<Triangulation<dim>*>(&*triangulation);
   GridTools::collect_periodic_faces(*tria, 0+10, 1+10, 0, periodic_faces);
   if (dim == 3)
     GridTools::collect_periodic_faces(*tria, 2+10, 3+10, 2, periodic_faces);

   triangulation->add_periodicity(periodic_faces);

   // perform global refinements
   triangulation->refine_global(n_refine_space);

   if(GRID_STRETCH_TYPE == GridStretchType::TransformGridCells)
   {
     // perform grid transform
     GridTools::transform (&grid_transform<dim>, *triangulation);
   }

   // set boundary conditions
   typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

   // fill boundary descriptor velocity
   boundary_descriptor_velocity->dirichlet_bc.insert(pair(0,new AnalyticalSolutionVelocity<dim>(dim)));

   // fill boundary descriptor pressure
   boundary_descriptor_pressure->neumann_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim)));
}


template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
  field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->right_hand_side.reset(new RightHandSide<dim>());
}

template<int dim>
void set_analytical_solution(std::shared_ptr<AnalyticalSolution<dim> > analytical_solution)
{
  analytical_solution->velocity.reset(new Functions::ZeroFunction<dim>(dim));
  analytical_solution->pressure.reset(new Functions::ZeroFunction<dim>(1));
}

// Postprocessor

#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"
#include "../../include/incompressible_navier_stokes/postprocessor/statistics_manager.h"

template<int dim>
struct PostProcessorDataTurbulentChannel
{
  PostProcessorData<dim> pp_data;
  TurbulentChannelData turb_ch_data;
};

template<int dim, int degree_u, int degree_p, typename Number>
class PostProcessorTurbulentChannel : public PostProcessor<dim, degree_u, degree_p, Number>
{
public:
  typedef PostProcessor<dim, degree_u, degree_p, Number> Base;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef typename Base::NavierStokesOperator NavierStokesOperator;

  PostProcessorTurbulentChannel(PostProcessorDataTurbulentChannel<dim> const & pp_data_turb_channel)
    :
    Base(pp_data_turb_channel.pp_data),
    turb_ch_data(pp_data_turb_channel.turb_ch_data)
  {}

  void setup(NavierStokesOperator const                &navier_stokes_operator_in,
             DoFHandler<dim> const                     &dof_handler_velocity_in,
             DoFHandler<dim> const                     &dof_handler_pressure_in,
             Mapping<dim> const                        &mapping_in,
             MatrixFree<dim,Number> const              &matrix_free_data_in,
             DofQuadIndexData const                    &dof_quad_index_data_in,
             std::shared_ptr<AnalyticalSolution<dim> > analytical_solution_in)
  {
    // call setup function of base class
    Base::setup(
        navier_stokes_operator_in,
        dof_handler_velocity_in,
        dof_handler_pressure_in,
        mapping_in,
        matrix_free_data_in,
        dof_quad_index_data_in,
        analytical_solution_in);

    // perform setup of turbulent channel related things
    statistics_turb_ch.reset(new StatisticsManager<dim>(dof_handler_velocity_in,mapping_in));
    statistics_turb_ch->setup(&grid_transform_y,turb_ch_data);
  }

  void do_postprocessing(VectorType const &velocity,
                         VectorType const &intermediate_velocity,
                         VectorType const &pressure,
                         double const     time,
                         int const        time_step_number)
  {
    Base::do_postprocessing(
	      velocity,
        intermediate_velocity,
        pressure,
        time,
        time_step_number);
   
    statistics_turb_ch->evaluate(velocity,time,time_step_number);
  }

  TurbulentChannelData turb_ch_data;
  std::shared_ptr<StatisticsManager<dim> > statistics_turb_ch;
};

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

  PostProcessorDataTurbulentChannel<dim> pp_data_turb_ch;
  pp_data_turb_ch.pp_data = pp_data;
  pp_data_turb_ch.turb_ch_data = param.turb_ch_data;

  std::shared_ptr<PostProcessorBase<dim,degree_u,degree_p,Number> > pp;
  pp.reset(new PostProcessorTurbulentChannel<dim,degree_u,degree_p,Number>(pp_data_turb_ch));

  return pp;
}


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_ */
