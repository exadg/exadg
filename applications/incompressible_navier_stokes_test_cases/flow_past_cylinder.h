/*
 * FlowPastCylinder.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_CYLINDER_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_CYLINDER_H_

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

// single or double precision?
//typedef float VALUE_TYPE;
typedef double VALUE_TYPE;

// set the number of space dimensions: dimension = 2, 3
unsigned int const DIMENSION = 2;

// set the polynomial degree of the shape functions for velocity and pressure
unsigned int const FE_DEGREE_VELOCITY = 3;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1; // FE_DEGREE_VELOCITY; // FE_DEGREE_VELOCITY - 1;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 1;
unsigned int const REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_MIN;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = 0; //REFINE_STEPS_TIME_MIN;

// mesh
#include "../grid_tools/mesh_flow_past_cylinder.h"

// set problem specific parameters like physical dimensions, etc.
ProblemType PROBLEM_TYPE = ProblemType::Unsteady;
const unsigned int TEST_CASE = 3; // 1, 2 or 3
const double Um = (DIMENSION == 2 ? (TEST_CASE==1 ? 0.3 : 1.5) : (TEST_CASE==1 ? 0.45 : 2.25));

const double END_TIME = 8.0;

std::string OUTPUT_FOLDER = "output/FPC/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME = "flow_past_cylinder";

template<int dim>
void InputParameters<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = PROBLEM_TYPE;
  equation_type = EquationType::NavierStokes;
  formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  right_hand_side = false;


  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = END_TIME;
  viscosity = 1.e-3;


  // TEMPORAL DISCRETIZATION
  solver_type = SolverType::Unsteady;
  temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  calculation_of_time_step_size = TimeStepCalculation::CFL;
  adaptive_time_stepping = true;
  max_velocity = Um;
  cfl = 0.7;//0.6;//2.5e-1;
  cfl_exponent_fe_degree_velocity = 1.5;
  time_step_size = 1.0e-3;
  order_time_integrator = 2; // 1; // 2; // 3;
  start_with_low_order = true; // true; // false;


  // SPATIAL DISCRETIZATION

  // triangulation
  triangulation_type = TriangulationType::Distributed;

  // polynomial degrees
  degree_u = FE_DEGREE_VELOCITY;
  degree_p = FE_DEGREE_PRESSURE;

  // mapping
  degree_mapping = FE_DEGREE_VELOCITY;

  // convective term
  if(formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
    upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

  // viscous term
  IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  pure_dirichlet_bc = false;

  // PROJECTION METHODS

  // pressure Poisson equation
  solver_pressure_poisson = SolverPressurePoisson::CG; //FGMRES;
  solver_data_pressure_poisson = SolverData(1000,1.e-12,1.e-6,30);
  preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;
  multigrid_data_pressure_poisson.type = MultigridType::pMG;
  multigrid_data_pressure_poisson.dg_to_cg_transfer = DG_To_CG_Transfer::Fine;
  multigrid_data_pressure_poisson.smoother_data.smoother = MultigridSmoother::Chebyshev;
  multigrid_data_pressure_poisson.smoother_data.iterations = 5;
  multigrid_data_pressure_poisson.coarse_problem.solver = MultigridCoarseGridSolver::CG;
  multigrid_data_pressure_poisson.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::AMG;
 
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
  preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; //BlockJacobi; //Multigrid;
  update_preconditioner_viscous = true;
  update_preconditioner_viscous_every_time_steps = 10;
  multigrid_data_viscous.type = MultigridType::pMG;
  multigrid_data_viscous.dg_to_cg_transfer = DG_To_CG_Transfer::Coarse;
  multigrid_data_viscous.coarse_problem.solver = MultigridCoarseGridSolver::CG;
  multigrid_data_viscous.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::PointJacobi;

  // PRESSURE-CORRECTION SCHEME

  // momentum step

  // Newton solver
  newton_solver_data_momentum = NewtonSolverData(100,1.e-12,1.e-8);

  // linear solver
  solver_momentum = SolverMomentum::FGMRES; //GMRES; //FGMRES;
  solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-8, 100);
  preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
  multigrid_data_momentum.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;
  multigrid_data_momentum.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::PointJacobi;
  update_preconditioner_momentum = false;

  // formulation
  order_pressure_extrapolation = 1;
  rotational_formulation = true;


  // COUPLED NAVIER-STOKES SOLVER

  // nonlinear solver (Newton solver)
  newton_solver_data_coupled = NewtonSolverData(100,1.e-12,1.e-8);

  // linear solver
  solver_coupled = SolverCoupled::FGMRES; //GMRES; //FGMRES;
  solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-8, 100);

  // preconditioning linear solver
  preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

  // preconditioner velocity/momentum block
  preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

  // preconditioner Schur-complement block
  preconditioner_pressure_block = SchurComplementPreconditioner::PressureConvectionDiffusion;
  discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;
  multigrid_data_pressure_block.coarse_problem.solver = MultigridCoarseGridSolver::CG;
  multigrid_data_pressure_block.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::PointJacobi;

  // OUTPUT AND POSTPROCESSING

  // write output for visualization of results
  output_data.write_output = true;
  output_data.output_folder = OUTPUT_FOLDER_VTU;
  output_data.output_name = OUTPUT_NAME;
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/20;
  output_data.write_divergence = true;
  output_data.degree = FE_DEGREE_VELOCITY;

  // output of solver information
  solver_info_data.print_to_screen = true;
  solver_info_data.interval_time = (end_time-start_time)/20;

  // lift and drag
  lift_and_drag_data.calculate_lift_and_drag = true;
  lift_and_drag_data.viscosity = viscosity;
  const double U = Um * (DIMENSION == 2 ? 2./3. : 4./9.);
  if(DIMENSION == 2)
    lift_and_drag_data.reference_value = 1.0/2.0*pow(U,2.0)*D;
  else if(DIMENSION == 3)
    lift_and_drag_data.reference_value = 1.0/2.0*pow(U,2.0)*D*H;

  // surfaces for calculation of lift and drag coefficients have boundary_ID = 2
  lift_and_drag_data.boundary_IDs.insert(2);

  lift_and_drag_data.filename_lift = OUTPUT_FOLDER + OUTPUT_NAME;
  lift_and_drag_data.filename_drag = OUTPUT_FOLDER + OUTPUT_NAME;

  // pressure difference
  pressure_difference_data.calculate_pressure_difference = true;
  if(DIMENSION == 2)
  {
    Point<dim> point_1_2D((X_C-D/2.0),Y_C), point_2_2D((X_C+D/2.0),Y_C);
    pressure_difference_data.point_1 = point_1_2D;
    pressure_difference_data.point_2 = point_2_2D;
  }
  else if(DIMENSION == 3)
  {
    Point<dim> point_1_3D((X_C-D/2.0),Y_C,H/2.0), point_2_3D((X_C+D/2.0),Y_C,H/2.0);
    pressure_difference_data.point_1 = point_1_3D;
    pressure_difference_data.point_2 = point_2_3D;
  }

  pressure_difference_data.filename = OUTPUT_FOLDER + OUTPUT_NAME;

  mass_data.calculate_error = false; //true;
  mass_data.start_time = 0.0;
  mass_data.sample_every_time_steps = 1;
  mass_data.filename_prefix = OUTPUT_FOLDER + OUTPUT_NAME;
}

/**************************************************************************************/
/*                                                                                    */
/*                        GENERATE GRID AND SET BOUNDARY INDICATORS                   */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
void create_grid_and_set_boundary_ids(
   std::shared_ptr<parallel::Triangulation<dim>>    triangulation,
   unsigned int const                               n_refine_space,
   std::vector<GridTools::PeriodicFacePair<typename
     Triangulation<dim>::cell_iterator> >           &/*periodic_faces*/)
{
 Point<dim> center;
 center[0] = X_C;
 center[1] = Y_C;

 Point<3> center_cyl_manifold;
 center_cyl_manifold[0] = center[0];
 center_cyl_manifold[1] = center[1];

 // apply this manifold for all mesh types
 Point<3> direction;
 direction[2] = 1.;

 static std::shared_ptr<Manifold<dim> > cylinder_manifold;

 if(MANIFOLD_TYPE == ManifoldType::SurfaceManifold)
 {
   cylinder_manifold = std::shared_ptr<Manifold<dim> >(dim == 2 ? static_cast<Manifold<dim>*>(new SphericalManifold<dim>(center)) :
                                           reinterpret_cast<Manifold<dim>*>(new CylindricalManifold<3>(direction, center_cyl_manifold)));
 }
 else if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
 {
   cylinder_manifold = std::shared_ptr<Manifold<dim> >(static_cast<Manifold<dim>*>(new MyCylindricalManifold<dim>(center)));
 }
 else
 {
   AssertThrow(MANIFOLD_TYPE == ManifoldType::SurfaceManifold || MANIFOLD_TYPE == ManifoldType::VolumeManifold,
       ExcMessage("Specified manifold type not implemented"));
 }

 create_triangulation(*triangulation);
 triangulation->set_manifold(MANIFOLD_ID, *cylinder_manifold);

 // generate vector of manifolds and apply manifold to all cells that have been marked
 static std::vector<std::shared_ptr<Manifold<dim> > > manifold_vec;
 manifold_vec.resize(manifold_ids.size());

 for(unsigned int i=0;i<manifold_ids.size();++i)
 {
   for (typename Triangulation<dim>::cell_iterator cell = triangulation->begin(); cell != triangulation->end(); ++cell)
   {
     if(cell->manifold_id() == manifold_ids[i])
     {
       manifold_vec[i] = std::shared_ptr<Manifold<dim> >(
           static_cast<Manifold<dim>*>(new OneSidedCylindricalManifold<dim>(cell,face_ids[i],center)));
       triangulation->set_manifold(manifold_ids[i],*(manifold_vec[i]));
     }
   }
 }

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

    if(component == 0 && std::abs(p[0]-(dim==2 ? L1: 0.0))<1.e-12)
    {
      const double pi = numbers::PI;
      const double T = 1.0;
      double coefficient = Utilities::fixed_power<dim-1>(4.) * Um / Utilities::fixed_power<2*dim-2>(H);
      if(TEST_CASE < 3)
      {
        if(PROBLEM_TYPE == ProblemType::Steady)
        {
          result = coefficient * p[1] * (H-p[1]);
        }
        else if(PROBLEM_TYPE == ProblemType::Unsteady)
        {
          result = coefficient * p[1] * (H-p[1]) * ( (t/T)<1.0 ? std::sin(pi/2.*t/T) : 1.0);
        }
      }
      if(TEST_CASE == 3)
        result = coefficient * p[1] * (H-p[1]) * std::sin(pi*t/END_TIME);
      if (dim == 3)
        result *= p[2] * (H-p[2]);
    }

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

    if(component == 0 && std::abs(p[0]-(dim==2 ? L1 : 0.0))<1.e-12)
    {
      const double pi = numbers::PI;
      const double T = 1.0;
      double coefficient = Utilities::fixed_power<dim-1>(4.) * Um / Utilities::fixed_power<2*dim-2>(H);
      if(TEST_CASE < 3)
        result = coefficient * p[1] * (H-p[1]) * ( (t/T)<1.0 ? (pi/2./T)*std::cos(pi/2.*t/T) : 0.0);
      if(TEST_CASE == 3)
        result = coefficient * p[1] * (H-p[1]) * std::cos(pi*t/END_TIME)*pi/END_TIME;
      if (dim == 3)
        result *= p[2] * (H-p[2]);
    }

    return result;
  }
};

template<int dim>
void set_boundary_conditions(
    std::shared_ptr<BoundaryDescriptorU<dim> > boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> > boundary_descriptor_pressure)
{
 // set boundary conditions
 typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

 // fill boundary descriptor velocity
 boundary_descriptor_velocity->dirichlet_bc.insert(pair(0,new AnalyticalSolutionVelocity<dim>()));
 boundary_descriptor_velocity->dirichlet_bc.insert(pair(2,new AnalyticalSolutionVelocity<dim>()));
 boundary_descriptor_velocity->neumann_bc.insert(pair(1,new Functions::ZeroFunction<dim>(dim)));

 // fill boundary descriptor pressure
 boundary_descriptor_pressure->neumann_bc.insert(pair(0,new PressureBC_dudt<dim>()));
 boundary_descriptor_pressure->neumann_bc.insert(pair(2,new PressureBC_dudt<dim>()));
 boundary_descriptor_pressure->dirichlet_bc.insert(pair(1,new Functions::ZeroFunction<dim>(1)));
}


template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
  field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
}

template<int dim>
void set_analytical_solution(std::shared_ptr<AnalyticalSolution<dim> > analytical_solution)
{
  analytical_solution->velocity.reset(new Functions::ZeroFunction<dim>(dim));
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

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_CYLINDER_H_ */
