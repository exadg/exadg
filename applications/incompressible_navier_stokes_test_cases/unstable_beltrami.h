/*
 * unstable_beltrami.h
 *
 *  Created on: July, 2018
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BELTRAMI_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BELTRAMI_H_

#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// convergence studies in space or time
unsigned int const DEGREE_MIN = 4;
unsigned int const DEGREE_MAX = 4;

unsigned int const REFINE_SPACE_MIN = 2;
unsigned int const REFINE_SPACE_MAX = 2;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;

// set problem specific parameters like physical dimensions, etc.
const double VISCOSITY = 1.0/(2.0e3);

// output folders and names
std::string OUTPUT_FOLDER = "output/unstable_beltrami/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME = "test";

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
  param.end_time = 20.0;
  param.viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  param.solver_type = SolverType::Unsteady;
  param.temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
  param.max_velocity = std::sqrt(2.0);
  param.cfl = 0.1;
  param.cfl_exponent_fe_degree_velocity = 1.5;
  param.time_step_size = 1.0e-3;
  param.order_time_integrator = 2; // 1; // 2; // 3;
  param.start_with_low_order = false; // true; // false;
  param.dt_refinements = REFINE_TIME_MIN;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = (param.end_time-param.start_time)/10;

  // SPATIAL DISCRETIZATION
  param.triangulation_type = TriangulationType::Distributed;
  param.degree_u = DEGREE_MIN;
  param.degree_p = DegreePressure::MixedOrder;
  param.mapping = MappingType::Isoparametric;
  param.h_refinements = REFINE_SPACE_MIN;

  // convective term

  // viscous term
  param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  param.pure_dirichlet_bc = true;
  param.adjust_pressure_level = AdjustPressureLevel::ApplyAnalyticalMeanValue;

  // div-div and continuity penalty
  param.use_divergence_penalty = true;
  param.divergence_penalty_factor = 1.0e0;
  param.use_continuity_penalty = true;
  param.continuity_penalty_factor = param.divergence_penalty_factor;
  param.add_penalty_terms_to_monolithic_system = false;


  // PROJECTION METHODS

  // pressure Poisson equation
  param.solver_pressure_poisson = SolverPressurePoisson::CG;
  param.solver_data_pressure_poisson = SolverData(1000,1.e-12,1.e-8,100);
  param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

  // projection step
  param.solver_projection = SolverProjection::CG;
  param.solver_data_projection = SolverData(1000, 1.e-12, 1.e-12);
  param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  param.order_extrapolation_pressure_nbc = param.order_time_integrator <=2 ? param.order_time_integrator : 2;

  // viscous step
  param.solver_viscous = SolverViscous::CG;
  param.solver_data_viscous = SolverData(1000,1.e-12,1.e-8);
  param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; //Multigrid;

  // PRESSURE-CORRECTION SCHEME

  // momentum step

  // Newton solver
  param.newton_solver_data_momentum = NewtonSolverData(100,1.e-12,1.e-8);

  // linear solver
  param.solver_momentum = SolverMomentum::GMRES;
  param.solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-8, 100);
  param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
  param.update_preconditioner_momentum = false;

  // formulation
  param.order_pressure_extrapolation = 1;
  param.rotational_formulation = true;


  // COUPLED NAVIER-STOKES SOLVER
  param.use_scaling_continuity = false;

  // nonlinear solver (Newton solver)
  param.newton_solver_data_coupled = NewtonSolverData(100,1.e-12,1.e-8);

  // linear solver
  param.solver_coupled = SolverCoupled::GMRES;
  param.solver_data_coupled = SolverData(1e3, 1.e-12, 1.e-8, 100);

  // preconditioning linear solver
  param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

  // preconditioner velocity/momentum block
  param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

  // preconditioner Schur-complement block
  param.preconditioner_pressure_block = SchurComplementPreconditioner::CahouetChabard;
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
  AssertThrow(dim == 3, ExcMessage("This test case can only be used for dim==3!"));

  const double left = 0.0, right = 1.0;
  GridGenerator::hyper_cube(*triangulation,left,right);

  // periodicity in x-,y-, and z-direction

  // x-direction
  triangulation->begin()->face(0)->set_all_boundary_ids(0);
  triangulation->begin()->face(1)->set_all_boundary_ids(1);
  // y-direction
  triangulation->begin()->face(2)->set_all_boundary_ids(2);
  triangulation->begin()->face(3)->set_all_boundary_ids(3);
  // z-direction
  triangulation->begin()->face(4)->set_all_boundary_ids(4);
  triangulation->begin()->face(5)->set_all_boundary_ids(5);

  auto tria = dynamic_cast<Triangulation<dim>*>(&*triangulation);
  GridTools::collect_periodic_faces(*tria, 0, 1, 0 /*x-direction*/, periodic_faces);
  GridTools::collect_periodic_faces(*tria, 2, 3, 1 /*y-direction*/, periodic_faces);
  GridTools::collect_periodic_faces(*tria, 4, 5, 2 /*z-direction*/, periodic_faces);

  triangulation->add_periodicity(periodic_faces);

  // global refinements
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

    const double pi = numbers::PI;

    if (component == 0)
      result = std::sin(2.0*pi*p[0])*std::sin(2.0*pi*p[1]);
    else if (component == 1)
      result = std::cos(2.0*pi*p[0])*std::cos(2.0*pi*p[1]);
    else if (component == 2)
      result = std::sqrt(2.0)*std::sin(2.0*pi*p[0])*std::cos(2.0*pi*p[1]);

    result *= std::exp(-8.0*pi*pi*VISCOSITY*t);

    return result;
  }

  Tensor<1,dim,double> gradient (const Point<dim>    &p,
                                 const unsigned int  component = 0) const
  {
    double t = this->get_time();
    Tensor<1,dim,double> result;

    const double pi = numbers::PI;

    AssertThrow(dim==3, ExcMessage("not implemented."));

    if (component == 0)
    {
      result[0] = 2.0*pi*std::cos(2.0*pi*p[0])*std::sin(2.0*pi*p[1]);
      result[1] = 2.0*pi*std::sin(2.0*pi*p[0])*std::cos(2.0*pi*p[1]);
      result[2] = 0.0;
    }
    else if (component == 1)
    {
      result[0] = -2.0*pi*std::sin(2.0*pi*p[0])*std::cos(2.0*pi*p[1]);
      result[1] = -2.0*pi*std::cos(2.0*pi*p[0])*std::sin(2.0*pi*p[1]);
      result[2] = 0.0;
    }
    else if (component == 2)
    {
      result[0] = 2.0*pi*std::sqrt(2.0)*std::cos(2.0*pi*p[0])*std::cos(2.0*pi*p[1]);
      result[1] = -2.0*pi*std::sqrt(2.0)*std::sin(2.0*pi*p[0])*std::sin(2.0*pi*p[1]);
      result[2] = 0.0;
    }

    result *= std::exp(-8.0*pi*pi*VISCOSITY*t);

    return result;
  }
};


template<int dim>
class AnalyticalSolutionPressure : public Function<dim>
{
public:
  AnalyticalSolutionPressure (const double time = 0.)
    :
    Function<dim>(1 /*n_components*/, time)
  {}

  double value (const Point<dim>   &p,
                const unsigned int /*component*/) const
  {
    double t = this->get_time();
    const double pi = numbers::PI;

    double result = - 0.5*(+ std::sin(2.0*pi*p[0])*std::sin(2.0*pi*p[0])
                           + std::cos(2.0*pi*p[1])*std::cos(2.0*pi*p[1]) - 1.0);

    result *= std::exp(-16.0*pi*pi*VISCOSITY*t);

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

    const double pi = numbers::PI;

    if (component == 0)
      result = std::sin(2.0*pi*p[0])*std::sin(2.0*pi*p[1]);
    else if (component == 1)
      result = std::cos(2.0*pi*p[0])*std::cos(2.0*pi*p[1]);
    else if (component == 2)
      result = std::sqrt(2.0)*std::sin(2.0*pi*p[0])*std::cos(2.0*pi*p[1]);

    result *= -8.0*pi*pi*VISCOSITY*std::exp(-8.0*pi*pi*VISCOSITY*t);

    return result;
  }
};

template<int dim>
void set_boundary_conditions(
    std::shared_ptr<BoundaryDescriptorU<dim> > /*boundary_descriptor_velocity*/,
    std::shared_ptr<BoundaryDescriptorP<dim> > /*boundary_descriptor_pressure*/)
{

}

template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
  field_functions->initial_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
  field_functions->analytical_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
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
  pp_data.output_data.write_output = false; //true;
  pp_data.output_data.output_folder = OUTPUT_FOLDER_VTU;
  pp_data.output_data.output_name = OUTPUT_NAME;
  pp_data.output_data.output_start_time = param.start_time;
  pp_data.output_data.output_interval_time = (param.end_time-param.start_time)/100;
  pp_data.output_data.write_divergence = false;
  pp_data.output_data.degree = param.degree_u;

  // calculation of velocity error
  pp_data.error_data_u.analytical_solution_available = true;
  pp_data.error_data_u.analytical_solution.reset(new AnalyticalSolutionVelocity<dim>());
  pp_data.error_data_u.calculate_H1_seminorm_error = true;
  pp_data.error_data_u.calculate_relative_errors = true;
  pp_data.error_data_u.error_calc_start_time = param.start_time;
  pp_data.error_data_u.error_calc_interval_time = 10;
  pp_data.error_data_u.write_errors_to_file = true;
  pp_data.error_data_u.folder = OUTPUT_FOLDER;
  pp_data.error_data_u.name = OUTPUT_NAME + "_velocity";

  // ... pressure error
  pp_data.error_data_p.analytical_solution_available = true;
  pp_data.error_data_p.analytical_solution.reset(new AnalyticalSolutionPressure<dim>());
  pp_data.error_data_p.calculate_relative_errors = true;
  pp_data.error_data_p.error_calc_start_time = param.start_time;
  pp_data.error_data_p.error_calc_interval_time = 10;
  pp_data.error_data_p.write_errors_to_file = true;
  pp_data.error_data_p.folder = OUTPUT_FOLDER;
  pp_data.error_data_p.name = OUTPUT_NAME + "_pressure";

  // kinetic energy
  pp_data.kinetic_energy_data.calculate = true;
  pp_data.kinetic_energy_data.calculate_every_time_steps = 10;
  pp_data.kinetic_energy_data.viscosity = VISCOSITY;
  pp_data.kinetic_energy_data.filename_prefix = OUTPUT_FOLDER + OUTPUT_NAME + "_energy";

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data));

  return pp;
}

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_UNSTABLE_BELTRAMI_H_ */
