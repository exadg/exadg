/*
 * Poiseuille.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_POISEUILLE_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_POISEUILLE_H_

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
const double MAX_VELOCITY = 1.0;
const double VISCOSITY = 1.0e-1;

const double H = 2.0;
const double L = 4.0;

namespace IncNS
{
void set_input_parameters(InputParameters &param)
{
  // MATHEMATICAL MODEL
  param.dim = 2;
  param.problem_type = PROBLEM_TYPE;
  param.equation_type = EquationType::NavierStokes;
  param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
  param.use_outflow_bc_convective_term = true;
  param.right_hand_side = false;


  // PHYSICAL QUANTITIES
  param.start_time = 0.0;
  param.end_time = 100.0;
  param.viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  param.solver_type = SolverType::Unsteady;
  param.temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
  param.adaptive_time_stepping = true;
  param.max_velocity = MAX_VELOCITY;
  param.cfl = 2.0e-1;
  param.time_step_size = 1.0e-1;
  param.max_number_of_time_steps = 1e8;
  param.order_time_integrator = 2; // 1; // 2; // 3;
  param.start_with_low_order = true; // true; // false;
  param.dt_refinements = REFINE_TIME_MIN;

  param.convergence_criterion_steady_problem = ConvergenceCriterionSteadyProblem::SolutionIncrement;
  param.abs_tol_steady = 1.e-12;
  param.rel_tol_steady = 1.e-6;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = (param.end_time-param.start_time)/100;

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
  param.pure_dirichlet_bc = false;


  // PROJECTION METHODS

  // pressure Poisson equation
  param.solver_pressure_poisson = SolverPressurePoisson::CG;
  param.solver_data_pressure_poisson = SolverData(1000,1.e-20,1.e-6,100);
  param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

  // projection step
  param.solver_projection = SolverProjection::CG;
  param.solver_data_projection = SolverData(1000, 1.e-20, 1.e-12);
  param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;


  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  param.order_extrapolation_pressure_nbc = param.order_time_integrator <=2 ? param.order_time_integrator : 2;

  // viscous step
  param.solver_viscous = SolverViscous::CG;
  param.solver_data_viscous = SolverData(1000,1.e-20,1.e-6);
  param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;

  // PRESSURE-CORRECTION SCHEME

  // formulation
  param.order_pressure_extrapolation = 1;
  param.rotational_formulation = true;

  // momentum step

  // Newton solver
  param.newton_solver_data_momentum = NewtonSolverData(100,1.e-14,1.e-6);

  // linear solver
  param.solver_momentum = SolverMomentum::GMRES;
  param.solver_data_momentum = SolverData(1e4, 1.e-20, 1.e-6, 100);
  param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
  param.update_preconditioner_momentum = false;
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

  std::vector<unsigned int> repetitions({1,1});
  Point<dim> point1(0.0,-H/2.), point2(L,H/2.);
  GridGenerator::subdivided_hyper_rectangle(*triangulation,repetitions,point1,point2);

  // left boundary has ID=1, right boundary has ID=2, all other boundaries have ID=0
  typename Triangulation<dim>::cell_iterator cell = triangulation->begin(), endc = triangulation->end();
  for(;cell!=endc;++cell)
  {
    for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
    {
     if ((std::fabs(cell->face(face_number)->center()(0) - 0.0)< 1e-12))
         cell->face(face_number)->set_boundary_id (1);
     if ((std::fabs(cell->face(face_number)->center()(0) - L)< 1e-12))
        cell->face(face_number)->set_boundary_id (2);
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
class PressureInflowBC : public Function<dim>
{
public:
  PressureInflowBC (const double time = 0.)
    :
    Function<dim>(1 /*n_components*/, time)
  {}

  double value (const Point<dim>   &p,
                const unsigned int /*component = 0*/) const
  {
    double t = this->get_time();

    // assuming a parabolic velocity profile
    double const pressure_gradient = -2.*VISCOSITY*MAX_VELOCITY;
    double const pi = numbers::PI;
    double const T = 10.0;

    return (p[0]-L) * pressure_gradient * std::sin(pi*t/T);
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
  boundary_descriptor_velocity->neumann_bc.insert(pair(1,new Functions::ZeroFunction<dim>(dim)));
  boundary_descriptor_velocity->neumann_bc.insert(pair(2,new Functions::ZeroFunction<dim>(dim)));

  // fill boundary descriptor pressure
  boundary_descriptor_pressure->neumann_bc.insert(pair(0,new Functions::ZeroFunction<dim>(1)));
  boundary_descriptor_pressure->dirichlet_bc.insert(pair(1,new PressureInflowBC<dim>()));
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
  pp_data.output_data.output_folder = "output/poiseuille/vtu/";
  pp_data.output_data.output_name = "poiseuille_pressure_inflow";
  pp_data.output_data.output_start_time = param.start_time;
  pp_data.output_data.output_interval_time = (param.end_time-param.start_time)/100;
  pp_data.output_data.write_velocity_magnitude = true;
  pp_data.output_data.degree = param.degree_u;

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data));

  return pp;
}

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_POISEUILLE_H_ */
