/*
 * mantle_convection.h
 *
 *  Created on: Jan 26, 2020
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_

#include "../../include/convection_diffusion/postprocessor/postprocessor.h"
#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// number of space dimensions
unsigned int const DIM = 2;

// convergence studies in space
unsigned int const DEGREE = 3;
unsigned int const REFINE_SPACE = 4;

// number of scalar quantities
unsigned int const N_SCALARS = 1;

// set problem specific parameters like physical dimensions, etc.
double const R0 = 6371000.0 - 2890000.0;
double const R1 = 6371000.0 - 35000.0;

double const YEAR_IN_SECONDS = 3600.0*24.0*365.0;

double const START_TIME = 0.0;
double const END_TIME = 1.0e8*YEAR_IN_SECONDS;

double const CFL = 1.0;
double const MAX_VELOCITY = 1.0/YEAR_IN_SECONDS;
bool const ADAPTIVE_TIME_STEPPING = true;

double const BETA = 2.0e-5;
double const KINEMATIC_VISCOSITY = 1.0e21/3300.0;
double const THERMAL_DIFFUSIVITY = 1.0e-6;

double const T_REF = 293.0;
double const T0 = 4000.0 + 273.0 - T_REF;
double const T1 = 700.0 + 273.0 - T_REF;

double const RELTOL = 1.e-6;

// output
bool const WRITE_OUTPUT = true;
std::string const OUTPUT_FOLDER = "/data/fehn/navierstokes/applications/output/mantle_convection/";
std::string const OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string const OUTPUT_NAME = "test";
double const OUTPUT_START_TIME = START_TIME;
double const OUTPUT_INTERVAL_TIME = (END_TIME-START_TIME)/100.0;

// restart
bool const WRITE_RESTART = false;
double const RESTART_INTERVAL_TIME = 10.0;

namespace IncNS
{
void set_input_parameters(InputParameters &param)
{
  // MATHEMATICAL MODEL
  param.dim = DIM;
  param.problem_type = ProblemType::Unsteady;
  param.equation_type = EquationType::Stokes;
  param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
  param.right_hand_side = true;
  param.boussinesq_term = true;


  // PHYSICAL QUANTITIES
  param.start_time = START_TIME;
  param.end_time = END_TIME;
  param.viscosity = KINEMATIC_VISCOSITY;
  param.thermal_expansion_coefficient = BETA;
  param.reference_temperature = T_REF;

  // TEMPORAL DISCRETIZATION
  param.solver_type = SolverType::Unsteady;
  param.temporal_discretization = TemporalDiscretization::BDFCoupledSolution;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  param.time_integrator_oif = TimeIntegratorOIF::ExplRK3Stage7Reg2;
  param.adaptive_time_stepping = ADAPTIVE_TIME_STEPPING;
  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
  param.max_velocity = MAX_VELOCITY;
  param.cfl_exponent_fe_degree_velocity = 1.5;
  param.cfl = CFL;
  param.time_step_size = 1.0e-1;
  param.order_time_integrator = 2;
  param.start_with_low_order = true;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = (END_TIME-START_TIME)/10.;

  // restart
  param.restart_data.write_restart = WRITE_RESTART;
  param.restart_data.interval_time = RESTART_INTERVAL_TIME;
  param.restart_data.filename = OUTPUT_FOLDER + OUTPUT_NAME + "_fluid";

  // SPATIAL DISCRETIZATION
  param.triangulation_type = TriangulationType::Distributed;
  param.degree_u = DEGREE;
  param.degree_p = DegreePressure::MixedOrder;
  param.mapping = MappingType::Isoparametric;
  param.h_refinements = REFINE_SPACE;

  // convective term
  if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
    param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

  // viscous term
  param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  param.pure_dirichlet_bc = true;

  // div-div and continuity penalty
  param.use_divergence_penalty = true;
  param.divergence_penalty_factor = 1.0;
  param.use_continuity_penalty = true;
  param.continuity_penalty_factor = param.divergence_penalty_factor;
  param.continuity_penalty_components = ContinuityPenaltyComponents::Normal;
  param.continuity_penalty_use_boundary_data = true;
  param.apply_penalty_terms_in_postprocessing_step = true;
  param.type_penalty_parameter = TypePenaltyParameter::ConvectiveTerm;

  // NUMERICAL PARAMETERS
  param.implement_block_diagonal_preconditioner_matrix_free = false;
  param.use_cell_based_face_loops = false;

  // PROJECTION METHODS

  // pressure Poisson equation
  param.solver_pressure_poisson = SolverPressurePoisson::CG;
  param.solver_data_pressure_poisson = SolverData(1000,1.e-30,RELTOL,100);
  param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;
  param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
  param.multigrid_data_pressure_poisson.coarse_problem.solver = MultigridCoarseGridSolver::Chebyshev;
  param.multigrid_data_pressure_poisson.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::PointJacobi;
  param.multigrid_data_pressure_poisson.smoother_data.smoother = MultigridSmoother::Chebyshev;
  param.multigrid_data_pressure_poisson.smoother_data.preconditioner = PreconditionerSmoother::PointJacobi;

  // projection step
  param.solver_projection = SolverProjection::CG;
  param.solver_data_projection = SolverData(1000, 1.e-30, RELTOL);
  param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulation
  param.order_extrapolation_pressure_nbc = param.order_time_integrator <=2 ? param.order_time_integrator : 2;

  // viscous step
  param.solver_viscous = SolverViscous::CG;
  param.solver_data_viscous = SolverData(1000,1.e-30,RELTOL);
  param.preconditioner_viscous = PreconditionerViscous::Multigrid; //InverseMassMatrix;
  param.multigrid_data_viscous.type = MultigridType::cphMG;


  // PRESSURE-CORRECTION SCHEME

  // formulation
  param.order_pressure_extrapolation = 1;
  param.rotational_formulation = true;

  // momentum step

  // Newton solver
  param.newton_solver_data_momentum = NewtonSolverData(100,1.e-30,RELTOL);

  // linear solver
  // use FGMRES for matrix-free BlockJacobi or Multigrid with Krylov methods as smoother/coarse grid solver
  param.solver_momentum = SolverMomentum::FGMRES;
  param.solver_data_momentum = SolverData(1e4,1.e-30,1.e-6,100);
  param.preconditioner_momentum = MomentumPreconditioner::Multigrid;
  param.multigrid_data_momentum.type = MultigridType::cphMG;
  param.multigrid_operator_type_momentum = MultigridOperatorType::ReactionDiffusion;

  // COUPLED NAVIER-STOKES SOLVER

  // nonlinear solver (Newton solver)
  param.newton_solver_data_coupled = NewtonSolverData(100,1.e-30,RELTOL);

  // linear solver
  param.solver_coupled = SolverCoupled::GMRES;
  param.solver_data_coupled = SolverData(1e3, 1.e-30, RELTOL, 100);
  param.use_scaling_continuity = false;

  // preconditioner linear solver
  param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;
  param.update_preconditioner_coupled = false;

  // preconditioner velocity/momentum block
  param.preconditioner_velocity_block = MomentumPreconditioner::Multigrid;
  param.multigrid_data_velocity_block.type = MultigridType::cphMG;
  param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;
  param.multigrid_data_velocity_block.smoother_data.smoother = MultigridSmoother::Chebyshev;
  param.multigrid_data_velocity_block.smoother_data.preconditioner = PreconditionerSmoother::PointJacobi;
  param.multigrid_data_velocity_block.smoother_data.iterations = 5;

  // preconditioner Schur-complement block
  param.preconditioner_pressure_block = SchurComplementPreconditioner::InverseMassMatrix;
}

}

namespace ConvDiff
{
void set_input_parameters(InputParameters &param, unsigned int const scalar_index)
{
  // MATHEMATICAL MODEL
  param.dim = DIM;
  param.problem_type = ProblemType::Unsteady;
  param.equation_type = EquationType::ConvectionDiffusion;
  param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
  param.analytical_velocity_field = false;
  param.right_hand_side = false;

  // PHYSICAL QUANTITIES
  param.start_time = START_TIME;
  param.end_time = END_TIME;
  param.diffusivity = THERMAL_DIFFUSIVITY;

  // TEMPORAL DISCRETIZATION
  param.temporal_discretization = TemporalDiscretization::BDF;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit;
  param.adaptive_time_stepping = ADAPTIVE_TIME_STEPPING;
  param.order_time_integrator = 2;
  param.start_with_low_order = true;
  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
  param.time_step_size = 1.0e-2;
  param.cfl = CFL;
  param.max_velocity = MAX_VELOCITY;
  param.exponent_fe_degree_convection = 1.5;
  param.exponent_fe_degree_diffusion = 3.0;
  param.diffusion_number = 0.01;

  // restart
  param.restart_data.write_restart = WRITE_RESTART;
  param.restart_data.interval_time = RESTART_INTERVAL_TIME;
  param.restart_data.filename = OUTPUT_FOLDER + OUTPUT_NAME + "_scalar_" + std::to_string(scalar_index);

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = (END_TIME-START_TIME)/10.;

  // SPATIAL DISCRETIZATION

  // triangulation
  param.triangulation_type = TriangulationType::Distributed;

  // polynomial degree
  param.degree = DEGREE;
  param.mapping = MappingType::Isoparametric;

  // h-refinements
  param.h_refinements = REFINE_SPACE;

  // convective term
  param.numerical_flux_convective_operator = NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

  // viscous term
  param.IP_factor = 1.0;

  // NUMERICAL PARAMETERS
  param.implement_block_diagonal_preconditioner_matrix_free = false;
  param.use_cell_based_face_loops = false;

  // SOLVER
  param.solver = Solver::GMRES; //CG;
  param.solver_data = SolverData(1e4, 1.e-30, RELTOL, 100);
  param.preconditioner = Preconditioner::InverseMassMatrix;
  param.multigrid_data.type = MultigridType::pMG;
  param.multigrid_data.p_sequence = PSequenceType::DecreaseByOne; //Bisect;
  param.mg_operator_type = MultigridOperatorType::ReactionDiffusion;
  param.update_preconditioner = false;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = (END_TIME-START_TIME)/10.;

  // NUMERICAL PARAMETERS
  param.use_combined_operator = true;
  param.filter_solution = false;
  param.use_overintegration = true;
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

  GridGenerator::hyper_shell(*triangulation,
                             Point<dim>(),
                             R0,
                             R1,
                             12);

  Point<dim> center = dim == 2 ? Point<dim>(0.,0.) : Point<dim>(0.,0.,0.);

  typename Triangulation<dim>::cell_iterator cell;
  for(cell = triangulation->begin(); cell != triangulation->end(); ++cell)
  {
    for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
    {
      bool face_at_outer_boundary = true;
      for (unsigned int v=0; v < GeometryInfo<dim-1>::vertices_per_cell; ++v)
      {
        if (std::abs(center.distance(cell->face(f)->vertex(v)) - R1) > 1e-12*R1)
          face_at_outer_boundary = false;
      }

      if (face_at_outer_boundary)
        cell->face(f)->set_boundary_id(1);
    }
  }

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
 class RightHandSide : public Function<dim>
 {
 public:
   RightHandSide (const double time = 0.)
     :
     Function<dim>(dim, time)
   {}

   double value (const Point<dim>    &p,
                 const unsigned int  component) const
   {
     double const r = p.norm();

     double g = -(1.245e-6 * r + 7.714e13 / r / r) * p[component] / r;

     return g;
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
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(1,new Functions::ZeroFunction<dim>(dim)));

  // fill boundary descriptor pressure
  boundary_descriptor_pressure->neumann_bc.insert(pair(0,new Functions::ZeroFunction<dim>(dim)));
  boundary_descriptor_pressure->neumann_bc.insert(pair(1,new Functions::ZeroFunction<dim>(dim)));
}

template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
  field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  field_functions->gravitational_force.reset(new RightHandSide<dim>());
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
  (void)param;

  PostProcessorData<dim> pp_data;

  // write output for visualization of results
  pp_data.output_data.write_output = WRITE_OUTPUT;
  pp_data.output_data.output_folder = OUTPUT_FOLDER_VTU;
  pp_data.output_data.output_name = OUTPUT_NAME + "_fluid";
  pp_data.output_data.output_start_time = OUTPUT_START_TIME;
  pp_data.output_data.output_interval_time = OUTPUT_INTERVAL_TIME;
  pp_data.output_data.write_processor_id = true;
  pp_data.output_data.degree = param.degree_u;

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data, mpi_comm));

  return pp;
}

}

namespace ConvDiff
{

/************************************************************************************************************/
/*                                                                                                          */
/*                         FUNCTIONS (INITIAL/BOUNDARY CONDITIONS, RIGHT-HAND SIDE, etc.)                   */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim>
class InitialTemperature : public Function<dim>
{
public:
  InitialTemperature (const unsigned int n_components = 1,
                      const double       time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  double value (const Point<dim>    &p,
                const unsigned int  /*component = 0*/) const
  {
    const double r = p.norm();
    const double h = R1 - R0;
    const double s = (r - R0) / h;
    const double q = (dim == 3) ? std::max(0.0, cos(numbers::PI * abs(p(2) / R1))) : 1.0;
    const double phi = std::atan2(p(0), p(1));
    const double tau = s + 0.2 * s * (1 - s) * std::sin(6 * phi) * q;
    return T0 * (1.0 - tau) + T1 * tau;
  }
};

template<int dim>
void
set_boundary_conditions(std::shared_ptr<ConvDiff::BoundaryDescriptor<0,dim> > boundary_descriptor, unsigned int scalar_index = 0)
{
  (void)scalar_index; // only one scalar quantity considered

  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  boundary_descriptor->dirichlet_bc.insert(pair(0,new Functions::ConstantFunction<dim>(T0)));
  boundary_descriptor->dirichlet_bc.insert(pair(1,new Functions::ConstantFunction<dim>(T1)));
}

template<int dim>
void
set_field_functions(std::shared_ptr<ConvDiff::FieldFunctions<dim> > field_functions, unsigned int scalar_index = 0)
{
  (void)scalar_index; // only one scalar quantity considered

  field_functions->initial_solution.reset(new InitialTemperature<dim>());
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->velocity.reset(new Functions::ZeroFunction<dim>(dim));
}

/************************************************************************************************************/
/*                                                                                                          */
/*                                              POSTPROCESSOR                                               */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, Number> >
construct_postprocessor(ConvDiff::InputParameters const &param,
                        MPI_Comm const &                 mpi_comm,
                        unsigned int const              scalar_index)
{
  PostProcessorData<dim> pp_data;
  pp_data.output_data.write_output = WRITE_OUTPUT;
  pp_data.output_data.output_folder = OUTPUT_FOLDER_VTU;
  pp_data.output_data.output_name = OUTPUT_NAME + "_scalar_" + std::to_string(scalar_index);
  pp_data.output_data.output_start_time = OUTPUT_START_TIME;
  pp_data.output_data.output_interval_time = OUTPUT_INTERVAL_TIME;
  pp_data.output_data.degree = param.degree;

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data, mpi_comm));

  return pp;
}

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_ */
