/*
 * tum.h
 *
 *  Created on: Sep 8, 2017
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TUM_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TUM_H_

#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// convergence studies in space or time
unsigned int const DEGREE_MIN = 2;
unsigned int const DEGREE_MAX = 2;

unsigned int const REFINE_SPACE_MIN = 3;
unsigned int const REFINE_SPACE_MAX = 3;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;

// set problem specific parameters like physical dimensions, etc.
const ProblemType PROBLEM_TYPE = ProblemType::Unsteady;
const double L = 1.0;
const double MAX_VELOCITY = 1.0;
const double VISCOSITY = 1.0e-3;

std::string OUTPUT_FOLDER = "output/tum/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME = "test";

namespace IncNS
{
void set_input_parameters(InputParameters &param)
{
  // MATHEMATICAL MODEL
  param.dim = 2;
  param.problem_type = PROBLEM_TYPE;
  param.equation_type = EquationType::NavierStokes;
  param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
  param.use_outflow_bc_convective_term = true;
  param.right_hand_side = false;


  // PHYSICAL QUANTITIES
  param.start_time = 0.0;
  param.end_time = 50.0;
  param.viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  param.solver_type = SolverType::Unsteady;
  param.temporal_discretization = TemporalDiscretization::BDFPressureCorrection;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit;
  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
  // best practice: use adaptive time stepping for this test case to avoid adjusting the CFL number
  param.adaptive_time_stepping = true;
  param.max_velocity = 1.0;
  param.cfl = 0.25;
  param.cfl_exponent_fe_degree_velocity = 1.5;
  param.time_step_size = 5.0e-2;
  param.order_time_integrator = 2; // 1; // 2; // 3;
  param.start_with_low_order = true; // true; // false;
  param.dt_refinements = REFINE_TIME_MIN;

  // pseudo-timestepping for steady-state problems
  param.convergence_criterion_steady_problem = ConvergenceCriterionSteadyProblem::ResidualSteadyNavierStokes;
  param.abs_tol_steady = 1.e-12;
  param.rel_tol_steady = 1.e-10;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = (param.end_time-param.start_time)/200;

  // SPATIAL DISCRETIZATION
  param.triangulation_type = TriangulationType::Distributed;
  param.degree_u = DEGREE_MIN;
  param.degree_p = DegreePressure::MixedOrder;
  param.mapping = MappingType::Isoparametric;
  param.h_refinements = REFINE_SPACE_MIN;

  // convective term
  if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
    param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

  // viscous term
  param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  param.pure_dirichlet_bc = false;

  // PROJECTION METHODS

  // pressure Poisson equation
  param.solver_pressure_poisson = SolverPressurePoisson::CG;
  param.solver_data_pressure_poisson = SolverData(1000,1.e-12,1.e-6,100);
  param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

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
  param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; //Multigrid;

  // PRESSURE-CORRECTION SCHEME

  // momentum step

  // Newton solver
  param.newton_solver_data_momentum = NewtonSolverData(100,1.e-12,1.e-6);

  // linear solver
  param.solver_momentum = SolverMomentum::GMRES; //GMRES; //FGMRES;
  param.solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-2, 100);
  param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
  param.update_preconditioner_momentum = true;

  // formulation
  param.order_pressure_extrapolation = 1;
  param.rotational_formulation = true;


  // COUPLED NAVIER-STOKES SOLVER

  // nonlinear solver (Newton solver)
  param.newton_solver_data_coupled = NewtonSolverData(100,1.e-12,1.e-6);

  // linear solver
  param.solver_coupled = SolverCoupled::FGMRES; //FGMRES;
  param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-2, 100);

  // preconditioning linear solver
  param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;
  param.update_preconditioner_coupled = true;

  // preconditioner velocity/momentum block
  param.preconditioner_velocity_block = MomentumPreconditioner::Multigrid;
  param.multigrid_data_velocity_block.smoother_data.smoother = MultigridSmoother::Jacobi; //Jacobi; //Chebyshev; //GMRES;
  param.multigrid_data_velocity_block.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi; //PointJacobi; //BlockJacobi;
  param.multigrid_data_velocity_block.smoother_data.iterations = 5;
  param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
  param.multigrid_data_velocity_block.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

  // preconditioner Schur-complement block
  param.preconditioner_pressure_block = SchurComplementPreconditioner::PressureConvectionDiffusion;
  param.discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;
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
 (void)periodic_faces;

 if(dim == 2)
 {
   Triangulation<dim> tria_h1, tria_h2, tria_h3, tria_v1, tria_v2, tria_v3, tria_v4, tria_v5;

   GridGenerator::subdivided_hyper_rectangle(tria_h1,
                                             std::vector<unsigned int>({4,1}),
                                             Point<dim>(0.0,0.0),
                                             Point<dim>(4.0*L,-1.0*L));

   GridGenerator::subdivided_hyper_rectangle(tria_h2,
                                             std::vector<unsigned int>({1,1}),
                                             Point<dim>(4.0*L,-4.0*L),
                                             Point<dim>(5.0*L,-5.0*L));

   GridGenerator::subdivided_hyper_rectangle(tria_h3,
                                             std::vector<unsigned int>({5,1}),
                                             Point<dim>(5.0*L,0.0*L),
                                             Point<dim>(10.0*L,-1.0*L));

   GridGenerator::subdivided_hyper_rectangle(tria_v1,
                                             std::vector<unsigned int>({1,4}),
                                             Point<dim>(1.0*L,-1.0*L),
                                             Point<dim>(2.0*L,-5.0*L));

   GridGenerator::subdivided_hyper_rectangle(tria_v2,
                                             std::vector<unsigned int>({1,4}),
                                             Point<dim>(3.0*L,-1.0*L),
                                             Point<dim>(4.0*L,-5.0*L));

   GridGenerator::subdivided_hyper_rectangle(tria_v3,
                                             std::vector<unsigned int>({1,4}),
                                             Point<dim>(5.0*L,-1.0*L),
                                             Point<dim>(6.0*L,-5.0*L));

   GridGenerator::subdivided_hyper_rectangle(tria_v4,
                                             std::vector<unsigned int>({1,4}),
                                             Point<dim>(7.0*L,-1.0*L),
                                             Point<dim>(8.0*L,-5.0*L));

   GridGenerator::subdivided_hyper_rectangle(tria_v5,
                                             std::vector<unsigned int>({1,4}),
                                             Point<dim>(9.0*L,-1.0*L),
                                             Point<dim>(10.0*L,-5.0*L));

   // merge
   Triangulation<dim> tmp1, tmp2;
   GridGenerator::merge_triangulations (tria_h1, tria_v1, tmp1);
   GridGenerator::merge_triangulations (tmp1, tria_v2, tmp2);
   GridGenerator::merge_triangulations (tmp2, tria_h2, tmp1);
   GridGenerator::merge_triangulations (tmp1, tria_v3, tmp2);
   GridGenerator::merge_triangulations (tmp2, tria_h3, tmp1);
   GridGenerator::merge_triangulations (tmp1, tria_v4, tmp2);
   GridGenerator::merge_triangulations (tmp2, tria_v5, *triangulation);

   // global refinements
   triangulation->refine_global(n_refine_space);
 }
 else if(dim == 3)
 {
   AssertThrow(false, ExcMessage("NotImplemented"));
 }

 // set boundary indicator
 // all boundaries have ID = 0 by default -> Dirichlet boundaries
 typename Triangulation<dim>::cell_iterator cell = triangulation->begin(), endc = triangulation->end();
 for(;cell!=endc;++cell)
 {
   for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
   {
     // inflow boundary
    if ((std::fabs(cell->face(face_number)->center()(0))< 1e-12))
       cell->face(face_number)->set_boundary_id (1);

    // outflow boundary
    if ((std::fabs(cell->face(face_number)->center()(1) - (-5.0*L))< 1e-12) && (cell->face(face_number)->center()(0)- 9.0*L)>=0)
       cell->face(face_number)->set_boundary_id (2);
   }
 }
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

    // constant velocity
    if(PROBLEM_TYPE == ProblemType::Steady)
    {
      if(component == 0 && (std::abs(p[0])<1.0e-15))
        result = MAX_VELOCITY;
    }
    else if(PROBLEM_TYPE == ProblemType::Unsteady)
    {
      const double T = 0.5;
      const double pi = numbers::PI;
      if(component == 0 && (std::abs(p[0])<1.0e-15))
      {
        result = t<T ? std::sin(pi/2.*t/T) : 1.0;

        result *= MAX_VELOCITY; //*(1.0 - std::pow(p[1]/L+0.5,2.0));
      }
    }

    return result;
  }
};


template<int dim>
void set_boundary_conditions(
    std::shared_ptr<BoundaryDescriptorU<dim> > boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> > boundary_descriptor_pressure)
 {
  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  // fill boundary descriptor velocity
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim)));
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(1,new AnalyticalSolutionVelocity<dim>()));
  boundary_descriptor_velocity->neumann_bc.insert(pair(2,new Functions::ZeroFunction<dim>(dim)));

  // fill boundary descriptor pressure
  boundary_descriptor_pressure->neumann_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim)));
  boundary_descriptor_pressure->neumann_bc.insert(pair(1,new Functions::ZeroFunction<dim>(dim)));
  boundary_descriptor_pressure->dirichlet_bc.insert(pair(2,new Functions::ZeroFunction<dim>(1)));
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
  pp_data.output_data.output_name = OUTPUT_NAME;
  pp_data.output_data.output_start_time = param.start_time;
  pp_data.output_data.output_interval_time = (param.end_time-param.start_time)/200;
  pp_data.output_data.write_divergence = true;
  pp_data.output_data.write_streamfunction = false;
  pp_data.output_data.degree = param.degree_u;

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data));

  return pp;
}

}


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TUM_H_ */
