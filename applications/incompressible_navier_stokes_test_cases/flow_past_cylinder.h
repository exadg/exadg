/*
 * flow_past_cylinder.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_CYLINDER_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_CYLINDER_H_

#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// convergence studies in space or time
unsigned int const DEGREE_MIN = 3; // degree_velocity >= 2 for mixed-order formulation (degree_pressure >= 1)
unsigned int const DEGREE_MAX = DEGREE_MIN;

unsigned int const REFINE_SPACE_MIN = 0;
unsigned int const REFINE_SPACE_MAX = REFINE_SPACE_MIN;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = REFINE_TIME_MIN;

// select test case according to Schaefer and Turek benchmark definition: 2D-1/2/3, 3D-1/2/3
ProblemType PROBLEM_TYPE = ProblemType::Unsteady;
unsigned int const DIMENSION = 2;
unsigned int const TEST_CASE = 3; // 1, 2 or 3
double const Um = (DIMENSION == 2 ? (TEST_CASE==1 ? 0.3 : 1.5) : (TEST_CASE==1 ? 0.45 : 2.25));

// physical parameters
double const VISCOSITY = 1.0e-3;

// end time: use a large value for TEST_CASE = 1 (steady problem)
// in order to not stop pseudo-timestepping approach before having converged
double const END_TIME = (TEST_CASE==1) ? 1000.0 : 8.0;

// CFL number (use CFL <= 0.4 - 0.6 for adaptive time stepping)
double const CFL = 0.4;

// physical dimensions
double const X_0 = 0.0; // origin (x-coordinate)
double const Y_0 = 0.0; // origin (y-coordinate)
double const L1 = 0.3; // x-coordinate of inflow boundary (2d test cases)
double const L2 = 2.5; // x-coordinate of outflow boundary (=length for 3d test cases)
double const H = 0.41; // height of channel
double const X_1 = L1; // left x-coordinate of mesh block around the cylinder
double const X_2 = 0.7; // right x-coordinate of mesh block around the cylinder
double const X_C = (X_2+X_1)/2.0; // center of cylinder (x-coordinate)
double const Y_C = 0.2; // center of cylinder (y-coordinate)
double const D = 0.1; // cylinder diameter
double const R = D/2.0; // cylinder radius

// MeshType
// Type1: no refinement around cylinder surface (coarsest mesh has 34 elements in 2D)
// Type2: two layers of spherical cells around cylinder (used in Fehn et al. (JCP, 2017, "On the stability of projection methods ...")),
//        (coarsest mesh has 50 elements in 2D)
// Type3: coarse mesh has only one element in direction perpendicular to flow direction,
//        one layer of spherical cells around cylinder for coarsest mesh (coarsest mesh has 12 elements in 2D)
// Type4: no refinement around cylinder, coarsest mesh consists of 4 cells for the block that
//        that surrounds the cylinder (coarsest mesh has 8 elements in 2D)
enum class MeshType{ Type1, Type2, Type3, Type4 };
MeshType const MESH_TYPE = MeshType::Type2;

// ManifoldType
// Surface manifold: when refining the mesh only the cells close to the manifold-surface are curved (should not be used!)
// Volume manifold: when refining the mesh all child cells are curved since it is a volume manifold
enum class ManifoldType{ SurfaceManifold, VolumeManifold };
ManifoldType const MANIFOLD_TYPE = ManifoldType::VolumeManifold;

// solver tolerances
double const ABS_TOL_LINEAR = 1.e-12;
double const REL_TOL_LINEAR = 1.e-6; // use 1.e-3 or smaller for pseudo-timestepping approach

double const ABS_TOL_NONLINEAR = 1.e-12;
double const REL_TOL_NONLINEAR = 1.e-8;

// writing output
std::string OUTPUT_FOLDER = "output/FPC/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME_VTU = "flow_past_cylinder";
std::string OUTPUT_NAME_LIFT = "lift";
std::string OUTPUT_NAME_DRAG = "drag";
std::string OUTPUT_NAME_DELTA_P = "pressure_difference";

namespace IncNS
{
void set_input_parameters(InputParameters &param)
{
  // MATHEMATICAL MODEL
  param.dim = DIMENSION;
  param.problem_type = PROBLEM_TYPE;
  param.equation_type = EquationType::NavierStokes;
  param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  param.right_hand_side = false;


  // PHYSICAL QUANTITIES
  param.start_time = 0.0;
  param.end_time = END_TIME;
  param.viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  param.solver_type = SolverType::Unsteady;
  param.temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
  param.adaptive_time_stepping = true;
  param.max_velocity = Um;
  param.cfl = CFL;
  param.cfl_exponent_fe_degree_velocity = 1.5;
  param.time_step_size = 1.0e-3;
  param.order_time_integrator = 2;
  param.start_with_low_order = true;
  param.dt_refinements = REFINE_TIME_MIN;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = (param.end_time-param.start_time)/8.0;

  // pseudo-timestepping for steady-state problems
  param.convergence_criterion_steady_problem = ConvergenceCriterionSteadyProblem::SolutionIncrement; //ResidualSteadyNavierStokes;
  param.abs_tol_steady = 1.e-12;
  param.rel_tol_steady = 1.e-8;

  // SPATIAL DISCRETIZATION
  param.triangulation_type = TriangulationType::Distributed;
  param.degree_u = DEGREE_MIN;
  param.degree_p = DegreePressure::MixedOrder;
  param.mapping = MappingType::Isoparametric;
  param.h_refinements = REFINE_SPACE_MIN;

  // convective term
  if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
    param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

  param.use_divergence_penalty = true;
  param.use_continuity_penalty = true;

  // viscous term
  param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  param.pure_dirichlet_bc = false;

  // NUMERICAL PARAMETERS 
  param.implement_block_diagonal_preconditioner_matrix_free = false;
  param.use_cell_based_face_loops = false;
  param.quad_rule_linearization = QuadratureRuleLinearization::Overintegration32k;

  // PROJECTION METHODS

  // pressure Poisson equation
  param.solver_pressure_poisson = SolverPressurePoisson::CG; //FGMRES;
  param.solver_data_pressure_poisson = SolverData(1000,ABS_TOL_LINEAR,REL_TOL_LINEAR,30);
  param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;
  param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
  param.multigrid_data_pressure_poisson.smoother_data.smoother = MultigridSmoother::Chebyshev;
  param.multigrid_data_pressure_poisson.smoother_data.iterations = 5;
  param.multigrid_data_pressure_poisson.coarse_problem.solver = MultigridCoarseGridSolver::CG;
  param.multigrid_data_pressure_poisson.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::AMG;
 
  // projection step
  param.solver_projection = SolverProjection::CG;
  param.solver_data_projection = SolverData(1000, ABS_TOL_LINEAR, REL_TOL_LINEAR);
  param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  param.order_extrapolation_pressure_nbc = param.order_time_integrator <=2 ? param.order_time_integrator : 2;

  // viscous step
  param.solver_viscous = SolverViscous::CG;
  param.solver_data_viscous = SolverData(1000,ABS_TOL_LINEAR,REL_TOL_LINEAR);
  param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; //BlockJacobi; //Multigrid;
  param.update_preconditioner_viscous = false;
  param.multigrid_data_viscous.type = MultigridType::phMG;
  param.multigrid_data_viscous.coarse_problem.solver = MultigridCoarseGridSolver::CG;
  param.multigrid_data_viscous.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::PointJacobi;

  // PRESSURE-CORRECTION SCHEME

  // momentum step

  // Newton solver
  param.newton_solver_data_momentum = NewtonSolverData(100,ABS_TOL_NONLINEAR,REL_TOL_NONLINEAR);

  // linear solver
  param.solver_momentum = SolverMomentum::FGMRES; //GMRES; //FGMRES;
  param.solver_data_momentum = SolverData(1e4, ABS_TOL_LINEAR, REL_TOL_LINEAR, 100);
  param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
  param.multigrid_data_momentum.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;
  param.multigrid_data_momentum.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::PointJacobi;
  param.update_preconditioner_momentum = false;

  // formulation
  param.order_pressure_extrapolation = 1;
  param.rotational_formulation = true;


  // COUPLED NAVIER-STOKES SOLVER

  // nonlinear solver (Newton solver)
  param.newton_solver_data_coupled = NewtonSolverData(100,ABS_TOL_NONLINEAR,REL_TOL_NONLINEAR);

  // linear solver
  param.solver_coupled = SolverCoupled::FGMRES; //FGMRES;
  param.solver_data_coupled = SolverData(1e4, ABS_TOL_LINEAR, REL_TOL_LINEAR, 100);

  param.update_preconditioner_coupled = true;

  // preconditioning linear solver
  param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

  // preconditioner velocity/momentum block
  param.preconditioner_velocity_block = MomentumPreconditioner::Multigrid;
  param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionConvectionDiffusion;
  param.multigrid_data_velocity_block.type = MultigridType::phMG;
  param.multigrid_data_velocity_block.smoother_data.smoother = MultigridSmoother::Jacobi; //Chebyshev; 
  param.multigrid_data_velocity_block.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
  param.multigrid_data_velocity_block.smoother_data.iterations = 5;
  param.multigrid_data_velocity_block.coarse_problem.solver = MultigridCoarseGridSolver::GMRES; //CG;
  param.multigrid_data_velocity_block.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::BlockJacobi; //PointJacobi; //AMG;
  param.multigrid_data_velocity_block.coarse_problem.solver_data.rel_tol = 1.e-3;
  param.multigrid_data_velocity_block.coarse_problem.amg_data.data.smoother_type = "Chebyshev";
  param.multigrid_data_velocity_block.coarse_problem.amg_data.data.smoother_sweeps = 1;

  // preconditioner Schur-complement block
  param.preconditioner_pressure_block = SchurComplementPreconditioner::PressureConvectionDiffusion;
  param.discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;
  param.multigrid_data_pressure_block.type = MultigridType::phMG;
  param.multigrid_data_pressure_block.coarse_problem.solver = MultigridCoarseGridSolver::CG;
  param.multigrid_data_pressure_block.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::PointJacobi; //AMG;
  param.multigrid_data_pressure_block.coarse_problem.solver_data.rel_tol = 1.e-3;
  param.multigrid_data_pressure_block.coarse_problem.amg_data.data.smoother_type = "Chebyshev";
  param.multigrid_data_pressure_block.coarse_problem.amg_data.data.smoother_sweeps = 1;
}

}

/************************************************************************************************************/
/*                                                                                                          */
/*                                       CREATE GRID AND SET BOUNDARY IDs                                   */
/*                                                                                                          */
/************************************************************************************************************/

#include "../grid_tools/mesh_flow_past_cylinder.h"

template<int dim>
void
create_grid_and_set_boundary_ids(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
                                 unsigned int const                            n_refine_space,
                                 std::vector<GridTools::PeriodicFacePair<typename
                                   Triangulation<dim>::cell_iterator> >        &periodic_faces)
{
 (void)periodic_faces;

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


/************************************************************************************************************/
/*                                                                                                          */
/*                         FUNCTIONS (INITIAL/BOUNDARY CONDITIONS, RIGHT-HAND SIDE, etc.)                   */
/*                                                                                                          */
/************************************************************************************************************/

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
  pp_data.output_data.output_name = OUTPUT_NAME_VTU;
  pp_data.output_data.output_start_time = param.start_time;
  pp_data.output_data.output_interval_time = (param.end_time-param.start_time)/20;
  pp_data.output_data.write_divergence = true;
  pp_data.output_data.degree = param.degree_u;

  // lift and drag
  pp_data.lift_and_drag_data.calculate_lift_and_drag = true;
  pp_data.lift_and_drag_data.viscosity = param.viscosity;
  const double U = Um * (DIMENSION == 2 ? 2./3. : 4./9.);
  if(DIMENSION == 2)
    pp_data.lift_and_drag_data.reference_value = 1.0/2.0*pow(U,2.0)*D;
  else if(DIMENSION == 3)
    pp_data.lift_and_drag_data.reference_value = 1.0/2.0*pow(U,2.0)*D*H;

  // surfaces for calculation of lift and drag coefficients have boundary_ID = 2
  pp_data.lift_and_drag_data.boundary_IDs.insert(2);

  pp_data.lift_and_drag_data.filename_lift = OUTPUT_FOLDER + OUTPUT_NAME_LIFT;
  pp_data.lift_and_drag_data.filename_drag = OUTPUT_FOLDER + OUTPUT_NAME_DRAG;

  // pressure difference
  pp_data.pressure_difference_data.calculate_pressure_difference = true;
  if(DIMENSION == 2)
  {
    Point<dim> point_1_2D((X_C-D/2.0),Y_C), point_2_2D((X_C+D/2.0),Y_C);
    pp_data.pressure_difference_data.point_1 = point_1_2D;
    pp_data.pressure_difference_data.point_2 = point_2_2D;
  }
  else if(DIMENSION == 3)
  {
    Point<dim> point_1_3D((X_C-D/2.0),Y_C,H/2.0), point_2_3D((X_C+D/2.0),Y_C,H/2.0);
    pp_data.pressure_difference_data.point_1 = point_1_3D;
    pp_data.pressure_difference_data.point_2 = point_2_3D;
  }

  pp_data.pressure_difference_data.filename = OUTPUT_FOLDER + OUTPUT_NAME_DELTA_P;

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data));

  return pp;
}

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_CYLINDER_H_ */
