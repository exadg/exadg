/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FDA_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FDA_H_

// ExaDG
#include <exadg/functions_and_boundary_conditions/linear_interpolation.h>

// FDA nozzle benchmark application
#include "include/functions.h"
#include "include/grid.h"
#include "include/postprocessor.h"

namespace ExaDG
{
namespace IncNS
{
namespace Precursor
{
// set the throat Reynolds number Re_throat = U_{mean,throat} * (2 R_throat) / nu
double const Re = 3500; // 500; //2000; //3500; //5000; //6500; //8000;

// kinematic viscosity (same viscosity for all Reynolds numbers)
double const viscosity = 3.31e-6;

double const area_inflow = FDANozzle::R_OUTER * FDANozzle::R_OUTER * dealii::numbers::PI;
double const area_throat = FDANozzle::R_INNER * FDANozzle::R_INNER * dealii::numbers::PI;

double const mean_velocity_throat = Re * viscosity / (2.0 * FDANozzle::R_INNER);
double const target_flow_rate     = mean_velocity_throat * area_throat;
double const mean_velocity_inflow = target_flow_rate / area_inflow;

double const max_velocity     = 2.0 * target_flow_rate / area_inflow;
double const max_velocity_cfl = 2.0 * target_flow_rate / area_throat;

// use prescribed velocity profile at inflow superimposed by random perturbations (white noise)?
// If yes, specify amplitude of perturbations relative to maximum velocity on centerline.
// Can be used with and without precursor approach
bool const   use_random_perturbations    = false;
double const factor_random_perturbations = 0.02;

// set this variable to true in order to switch off the precursor and simulate on the actual domain
// only
bool const switch_off_precursor = false;

// start and end time

// estimation of flow-through time T_0 (through nozzle section)
// based on the mean velocity through throat
double const T_0                  = FDANozzle::LENGTH_THROAT / mean_velocity_throat;
double const start_time_precursor = -500.0 * T_0; // let the flow develop
double const start_time_nozzle    = 0.0 * T_0;
double const end_time             = 250.0 * T_0; // 150.0*T_0;

// postprocessing

// output folder
std::string const directory = "output/fda/Re3500/";

// flow-rate
std::string const filename_flowrate = "precursor_mean_velocity";

// sampling of axial and radial velocity profiles

// sampling interval should last over (100-200) * T_0 according to preliminary results.
double const       sample_start_time      = 50.0 * T_0; // let the flow develop
double const       sample_end_time        = end_time;   // that's the only reasonable choice
unsigned int const sample_every_timesteps = 1;

// line plot data
unsigned int const n_points_line_axial           = 400;
unsigned int const n_points_line_radial          = 64;
unsigned int const n_points_line_circumferential = 32;

// vtu-output
double const output_start_time_precursor = start_time_precursor;
double const output_start_time_nozzle    = start_time_nozzle;
double const output_interval_time        = 5.0 * T_0; // 10.0*T_0;

/*
 *  Most of the parameters are the same for both domains, so we write
 *  this function for the actual domain and only "correct" the parameters
 *  for the precursor by passing an additional parameter is_precursor.
 */
void
do_set_parameters(Parameters & param, bool const is_precursor = false)
{
  // MATHEMATICAL MODEL
  param.problem_type                   = ProblemType::Unsteady;
  param.equation_type                  = EquationType::NavierStokes;
  param.use_outflow_bc_convective_term = true;
  param.formulation_viscous_term       = FormulationViscousTerm::LaplaceFormulation;
  param.formulation_convective_term    = FormulationConvectiveTerm::DivergenceFormulation;
  param.right_hand_side                = true;


  // PHYSICAL QUANTITIES
  param.start_time = start_time_nozzle;
  if(is_precursor)
    param.start_time = start_time_precursor;

  param.end_time  = end_time;
  param.viscosity = viscosity;


  // TEMPORAL DISCRETIZATION
  param.solver_type = SolverType::Unsteady;

  //  param.temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  //  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  //  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
  //  param.adaptive_time_stepping = true;
  param.temporal_discretization                = TemporalDiscretization::BDFPressureCorrection;
  param.treatment_of_convective_term           = TreatmentOfConvectiveTerm::Implicit;
  param.calculation_of_time_step_size          = TimeStepCalculation::CFL;
  param.adaptive_time_stepping_limiting_factor = 3.0;
  param.max_velocity                           = max_velocity_cfl;
  param.cfl                                    = 4.0;
  param.cfl_exponent_fe_degree_velocity        = 1.5;
  param.time_step_size                         = 1.0e-1;
  param.order_time_integrator                  = 2;
  param.start_with_low_order                   = true;

  // output of solver information
  param.solver_info_data.interval_time = T_0;


  // SPATIAL DISCRETIZATION
  param.grid.triangulation_type = TriangulationType::Distributed;
  param.mapping_degree          = param.degree_u;
  param.degree_p                = DegreePressure::MixedOrder;

  // convective term
  param.upwind_factor = 1.0;

  // viscous term
  param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;
  param.IP_factor_viscous      = 1.0;

  // div-div and continuity penalty terms
  param.use_divergence_penalty                     = true;
  param.divergence_penalty_factor                  = 1.0e0;
  param.use_continuity_penalty                     = true;
  param.continuity_penalty_factor                  = param.divergence_penalty_factor;
  param.apply_penalty_terms_in_postprocessing_step = true;

  // TURBULENCE
  param.turbulence_model_data.is_active        = false;
  param.turbulence_model_data.turbulence_model = TurbulenceEddyViscosityModel::Sigma;
  // Smagorinsky: 0.165, Vreman: 0.28, WALE: 0.50, Sigma: 1.35
  param.turbulence_model_data.constant = 1.35;

  // PROJECTION METHODS

  // pressure Poisson equation
  param.IP_factor_pressure                   = 1.0;
  param.solver_data_pressure_poisson         = SolverData(1000, 1.e-12, 1.e-3, 100);
  param.solver_pressure_poisson              = SolverPressurePoisson::CG; // FGMRES;
  param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
  param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
  if(is_precursor)
    param.multigrid_data_pressure_poisson.type = MultigridType::phMG;
  param.multigrid_data_pressure_poisson.smoother_data.smoother   = MultigridSmoother::Chebyshev;
  param.multigrid_data_pressure_poisson.smoother_data.iterations = 5;
  param.multigrid_data_pressure_poisson.coarse_problem.solver    = MultigridCoarseGridSolver::CG;
  param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
    MultigridCoarseGridPreconditioner::AMG;


  // projection step
  param.solver_projection                = SolverProjection::CG;
  param.solver_data_projection           = SolverData(1000, 1.e-12, 1.e-3);
  param.preconditioner_projection        = PreconditionerProjection::InverseMassMatrix;
  param.update_preconditioner_projection = true;


  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  param.order_extrapolation_pressure_nbc =
    param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

  // viscous step
  param.solver_viscous         = SolverViscous::CG;
  param.solver_data_viscous    = SolverData(1000, 1.e-12, 1.e-3);
  param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;


  // PRESSURE-CORRECTION SCHEME

  // formulation
  param.order_pressure_extrapolation = 1;    // use 0 for non-incremental formulation
  param.rotational_formulation       = true; // use false for standard formulation

  // momentum step

  // Newton solver
  param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-12, 1.e-3);

  // linear solver
  if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
    param.solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-1, 100);
  else
    param.solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-3, 100);

  param.solver_momentum                = SolverMomentum::GMRES;
  param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
  param.update_preconditioner_momentum = false;

  // COUPLED NAVIER-STOKES SOLVER
  param.use_scaling_continuity = false;

  // nonlinear solver (Newton solver)
  param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-20, 1.e-3);

  // linear solver
  param.solver_coupled = SolverCoupled::GMRES; // GMRES; //FGMRES;
  if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
    param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-1, 100);
  else
    param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-3, 100);

  // preconditioning linear solver
  param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
  param.update_preconditioner_coupled = false;

  // preconditioner velocity/momentum block
  param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

  // preconditioner Schur-complement block
  param.preconditioner_pressure_block =
    SchurComplementPreconditioner::CahouetChabard; // PressureConvectionDiffusion;

  // Chebyshev moother
  param.multigrid_data_pressure_block.smoother_data.smoother = MultigridSmoother::Chebyshev;
  param.multigrid_data_pressure_block.coarse_problem.solver  = MultigridCoarseGridSolver::Chebyshev;
}


template<int dim, typename Number>
class PrecursorDomain : public Domain<dim, Number>
{
public:
  PrecursorDomain(std::string                               parameter_file,
                  MPI_Comm const &                          comm,
                  std::shared_ptr<InflowDataStorage<dim>> & inflow_data)
    : Domain<dim, Number>(parameter_file, comm), inflow_data_storage(inflow_data)
  {
  }

  void
  set_parameters() final
  {
    do_set_parameters(this->param, true);
  }

  void
  create_grid() final
  {
    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> &                        tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & periodic_face_pairs,
          unsigned int const                                       global_refinements,
          std::vector<unsigned int> const &                        vector_local_refinements) {
        (void)vector_local_refinements;

        FDANozzle::create_grid_and_set_boundary_ids_precursor(tria,
                                                              global_refinements,
                                                              periodic_face_pairs);
      };

    GridUtilities::create_fine_and_coarse_triangulations<dim>(*this->grid,
                                                              this->mpi_comm,
                                                              this->param.grid,
                                                              this->param.involves_h_multigrid(),
                                                              lambda_create_triangulation,
                                                              {} /* no local refinements */);
  }

  void
  set_boundary_descriptor() final
  {
    /*
     *  FILL BOUNDARY DESCRIPTORS
     */
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    // fill boundary descriptor velocity
    // no slip boundaries at lower and upper wall with ID=0
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));

    // fill boundary descriptor pressure
    // no slip boundaries at lower and upper wall with ID=0
    this->boundary_descriptor->pressure->neumann_bc.insert(0);
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(
      new InitialSolutionVelocity<dim>(max_velocity));
    this->field_functions->initial_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->analytical_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));

    // prescribe body force for the turbulent pipe flow (precursor) to adjust the desired flow rate
    flow_rate_controller.reset(new FlowRateController(target_flow_rate,
                                                      viscosity,
                                                      max_velocity,
                                                      FDANozzle::R_OUTER,
                                                      mean_velocity_inflow,
                                                      FDANozzle::D,
                                                      start_time_precursor));

    this->field_functions->right_hand_side.reset(new RightHandSide<dim>(*flow_rate_controller));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    std::shared_ptr<PostProcessorBase<dim, Number>> pp;

    PostProcessorData<dim> pp_data;
    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = output_start_time_precursor;
    pp_data.output_data.time_control_data.trigger_interval = output_interval_time;
    pp_data.output_data.directory                = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename                 = this->output_parameters.filename + "_precursor";
    pp_data.output_data.write_divergence         = true;
    pp_data.output_data.write_processor_id       = true;
    pp_data.output_data.mean_velocity.is_active  = true;
    pp_data.output_data.mean_velocity.start_time = sample_start_time;
    pp_data.output_data.mean_velocity.end_time   = sample_end_time;
    pp_data.output_data.mean_velocity.trigger_every_time_steps = 1;
    pp_data.output_data.degree                                 = this->param.degree_u;

    PostProcessorDataFDA<dim> pp_data_fda;
    pp_data_fda.pp_data = pp_data;

    // inflow data
    // prescribe solution at the right boundary of the precursor domain
    // as weak Dirichlet boundary condition at the left boundary of the nozzle domain
    AssertThrow(inflow_data_storage.get(),
                dealii::ExcMessage("inflow_data_storage is uninitialized."));

    pp_data_fda.inflow_data.write_inflow_data = true;
    pp_data_fda.inflow_data.inflow_geometry   = InflowGeometry::Cylindrical;
    pp_data_fda.inflow_data.normal_direction  = 2;
    pp_data_fda.inflow_data.normal_coordinate = FDANozzle::Z2_PRECURSOR;
    pp_data_fda.inflow_data.n_points_y        = inflow_data_storage->n_points_r;
    pp_data_fda.inflow_data.n_points_z        = inflow_data_storage->n_points_phi;
    pp_data_fda.inflow_data.y_values          = &inflow_data_storage->r_values;
    pp_data_fda.inflow_data.z_values          = &inflow_data_storage->phi_values;
    pp_data_fda.inflow_data.array             = &inflow_data_storage->velocity_values;

    // calculation of flow rate (use volume-based computation)
    pp_data_fda.mean_velocity_data.calculate = true;
    pp_data_fda.mean_velocity_data.directory = this->output_parameters.directory;
    pp_data_fda.mean_velocity_data.filename  = filename_flowrate;
    dealii::Tensor<1, dim, double> direction;
    direction[2]                                 = 1.0;
    pp_data_fda.mean_velocity_data.direction     = direction;
    pp_data_fda.mean_velocity_data.write_to_file = true;

    pp.reset(new PostProcessorFDA<dim, Number>(pp_data_fda,
                                               this->mpi_comm,
                                               area_inflow,
                                               flow_rate_controller,
                                               inflow_data_storage,
                                               true /* use precursor */,
                                               false));

    return pp;
  }

private:
  std::shared_ptr<InflowDataStorage<dim>> inflow_data_storage;

  std::shared_ptr<FlowRateController> flow_rate_controller;
};

template<int dim, typename Number>
class MainDomain : public Domain<dim, Number>
{
public:
  MainDomain(std::string                               parameter_file,
             MPI_Comm const &                          comm,
             std::shared_ptr<InflowDataStorage<dim>> & inflow_data,
             bool const                                precursor_is_active)
    : Domain<dim, Number>(parameter_file, comm),
      inflow_data_storage(inflow_data),
      use_precursor(precursor_is_active)
  {
  }

  void
  set_parameters() final
  {
    do_set_parameters(this->param, false);
  }

  void
  create_grid() final
  {
    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> &                        tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & periodic_face_pairs,
          unsigned int const                                       global_refinements,
          std::vector<unsigned int> const &                        vector_local_refinements) {
        (void)vector_local_refinements;

        FDANozzle::create_grid_and_set_boundary_ids_nozzle(tria,
                                                           global_refinements,
                                                           periodic_face_pairs);
      };

    GridUtilities::create_fine_and_coarse_triangulations<dim>(*this->grid,
                                                              this->mpi_comm,
                                                              this->param.grid,
                                                              this->param.involves_h_multigrid(),
                                                              lambda_create_triangulation,
                                                              {} /* no local refinements */);
  }

  void
  set_boundary_descriptor() final
  {
    /*
     *  FILL BOUNDARY DESCRIPTORS
     */
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    // fill boundary descriptor velocity
    // no slip boundaries at the upper and lower wall with ID=0
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));

    // inflow boundary condition at left boundary with ID=1: prescribe velocity profile which
    // is obtained as the results of the simulation on the precursor domain
    AssertThrow(inflow_data_storage.get(),
                dealii::ExcMessage("inflow_data_storage is not initialized."));

    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(1, new InflowProfile<dim>(*inflow_data_storage)));

    // outflow boundary condition at right boundary with ID=2
    this->boundary_descriptor->velocity->neumann_bc.insert(
      pair(2, new dealii::Functions::ZeroFunction<dim>(dim)));

    // fill boundary descriptor pressure
    // no slip boundaries at the upper and lower wall with ID=0
    this->boundary_descriptor->pressure->neumann_bc.insert(0);

    // inflow boundary condition at left boundary with ID=1
    this->boundary_descriptor->pressure->neumann_bc.insert(1);

    // outflow boundary condition at right boundary with ID=2: set pressure to zero
    this->boundary_descriptor->pressure->dirichlet_bc.insert(
      pair(2, new dealii::Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(
      new InitialSolutionVelocity<dim>(max_velocity));
    this->field_functions->initial_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->analytical_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    std::shared_ptr<PostProcessorBase<dim, Number>> pp;

    // write output for visualization of results
    PostProcessorData<dim> pp_data;
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = output_start_time_nozzle;
    pp_data.output_data.time_control_data.trigger_interval = output_interval_time;
    pp_data.output_data.directory                = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename                 = this->output_parameters.filename + "_nozzle";
    pp_data.output_data.write_divergence         = true;
    pp_data.output_data.write_processor_id       = true;
    pp_data.output_data.mean_velocity.is_active  = true;
    pp_data.output_data.mean_velocity.start_time = sample_start_time;
    pp_data.output_data.mean_velocity.end_time   = sample_end_time;
    pp_data.output_data.mean_velocity.trigger_every_time_steps = 1;
    pp_data.output_data.degree                                 = this->param.degree_u;

    PostProcessorDataFDA<dim> pp_data_fda;
    pp_data_fda.pp_data = pp_data;

    // evaluation of quantities along lines
    pp_data_fda.line_plot_data.time_control_data_statistics.time_control_data.is_active = true;
    pp_data_fda.line_plot_data.time_control_data_statistics.time_control_data.start_time =
      sample_start_time;
    pp_data_fda.line_plot_data.time_control_data_statistics.time_control_data.end_time = end_time;
    pp_data_fda.line_plot_data.time_control_data_statistics.time_control_data
      .trigger_every_time_steps = sample_every_timesteps;
    pp_data_fda.line_plot_data.time_control_data_statistics
      .write_preliminary_results_every_nth_time_step = sample_every_timesteps * 100;
    pp_data_fda.line_plot_data.directory             = this->output_parameters.directory;

    // lines
    std::shared_ptr<LineCircumferentialAveraging<dim>> axial_profile, radial_profile_z1,
      radial_profile_z2, radial_profile_z3, radial_profile_z4, radial_profile_z5, radial_profile_z6,
      radial_profile_z7, radial_profile_z8, radial_profile_z9, radial_profile_z10,
      radial_profile_z11, radial_profile_z12;

    axial_profile.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z1.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z2.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z3.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z4.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z5.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z6.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z7.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z8.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z9.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z10.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z11.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z12.reset(new LineCircumferentialAveraging<dim>());

    double z_1 = -0.088, z_2 = -0.064, z_3 = -0.048, z_4 = -0.02, z_5 = -0.008, z_6 = 0.0,
           z_7 = 0.008, z_8 = 0.016, z_9 = 0.024, z_10 = 0.032, z_11 = 0.06, z_12 = 0.08;

    // begin and end points of all lines
    axial_profile->begin      = dealii::Point<dim>(0, 0, FDANozzle::Z1_INFLOW);
    axial_profile->end        = dealii::Point<dim>(0, 0, FDANozzle::Z2_OUTFLOW);
    radial_profile_z1->begin  = dealii::Point<dim>(0, 0, z_1);
    radial_profile_z1->end    = dealii::Point<dim>(FDANozzle::radius_function(z_1), 0, z_1);
    radial_profile_z2->begin  = dealii::Point<dim>(0, 0, z_2);
    radial_profile_z2->end    = dealii::Point<dim>(FDANozzle::radius_function(z_2), 0, z_2);
    radial_profile_z3->begin  = dealii::Point<dim>(0, 0, z_3);
    radial_profile_z3->end    = dealii::Point<dim>(FDANozzle::radius_function(z_3), 0, z_3);
    radial_profile_z4->begin  = dealii::Point<dim>(0, 0, z_4);
    radial_profile_z4->end    = dealii::Point<dim>(FDANozzle::radius_function(z_4), 0, z_4);
    radial_profile_z5->begin  = dealii::Point<dim>(0, 0, z_5);
    radial_profile_z5->end    = dealii::Point<dim>(FDANozzle::radius_function(z_5), 0, z_5);
    radial_profile_z6->begin  = dealii::Point<dim>(0, 0, z_6);
    radial_profile_z6->end    = dealii::Point<dim>(FDANozzle::radius_function(z_6), 0, z_6);
    radial_profile_z7->begin  = dealii::Point<dim>(0, 0, z_7);
    radial_profile_z7->end    = dealii::Point<dim>(FDANozzle::radius_function(z_7), 0, z_7);
    radial_profile_z8->begin  = dealii::Point<dim>(0, 0, z_8);
    radial_profile_z8->end    = dealii::Point<dim>(FDANozzle::radius_function(z_8), 0, z_8);
    radial_profile_z9->begin  = dealii::Point<dim>(0, 0, z_9);
    radial_profile_z9->end    = dealii::Point<dim>(FDANozzle::radius_function(z_9), 0, z_9);
    radial_profile_z10->begin = dealii::Point<dim>(0, 0, z_10);
    radial_profile_z10->end   = dealii::Point<dim>(FDANozzle::radius_function(z_10), 0, z_10);
    radial_profile_z11->begin = dealii::Point<dim>(0, 0, z_11);
    radial_profile_z11->end   = dealii::Point<dim>(FDANozzle::radius_function(z_11), 0, z_11);
    radial_profile_z12->begin = dealii::Point<dim>(0, 0, z_12);
    radial_profile_z12->end   = dealii::Point<dim>(FDANozzle::radius_function(z_12), 0, z_12);

    // number of points
    axial_profile->n_points      = n_points_line_axial;
    radial_profile_z1->n_points  = n_points_line_radial;
    radial_profile_z2->n_points  = n_points_line_radial;
    radial_profile_z3->n_points  = n_points_line_radial;
    radial_profile_z4->n_points  = n_points_line_radial;
    radial_profile_z5->n_points  = n_points_line_radial;
    radial_profile_z6->n_points  = n_points_line_radial;
    radial_profile_z7->n_points  = n_points_line_radial;
    radial_profile_z8->n_points  = n_points_line_radial;
    radial_profile_z9->n_points  = n_points_line_radial;
    radial_profile_z10->n_points = n_points_line_radial;
    radial_profile_z11->n_points = n_points_line_radial;
    radial_profile_z12->n_points = n_points_line_radial;

    axial_profile->average_circumferential      = false;
    radial_profile_z1->average_circumferential  = true;
    radial_profile_z2->average_circumferential  = true;
    radial_profile_z3->average_circumferential  = true;
    radial_profile_z4->average_circumferential  = true;
    radial_profile_z5->average_circumferential  = true;
    radial_profile_z6->average_circumferential  = true;
    radial_profile_z7->average_circumferential  = true;
    radial_profile_z8->average_circumferential  = true;
    radial_profile_z9->average_circumferential  = true;
    radial_profile_z10->average_circumferential = true;
    radial_profile_z11->average_circumferential = true;
    radial_profile_z12->average_circumferential = true;

    radial_profile_z1->n_points_circumferential  = n_points_line_circumferential;
    radial_profile_z2->n_points_circumferential  = n_points_line_circumferential;
    radial_profile_z3->n_points_circumferential  = n_points_line_circumferential;
    radial_profile_z4->n_points_circumferential  = n_points_line_circumferential;
    radial_profile_z5->n_points_circumferential  = n_points_line_circumferential;
    radial_profile_z6->n_points_circumferential  = n_points_line_circumferential;
    radial_profile_z7->n_points_circumferential  = n_points_line_circumferential;
    radial_profile_z8->n_points_circumferential  = n_points_line_circumferential;
    radial_profile_z9->n_points_circumferential  = n_points_line_circumferential;
    radial_profile_z10->n_points_circumferential = n_points_line_circumferential;
    radial_profile_z11->n_points_circumferential = n_points_line_circumferential;
    radial_profile_z12->n_points_circumferential = n_points_line_circumferential;

    dealii::Tensor<1, dim, double> normal;
    normal[2]                         = 1.0;
    radial_profile_z1->normal_vector  = normal;
    radial_profile_z2->normal_vector  = normal;
    radial_profile_z3->normal_vector  = normal;
    radial_profile_z4->normal_vector  = normal;
    radial_profile_z5->normal_vector  = normal;
    radial_profile_z6->normal_vector  = normal;
    radial_profile_z7->normal_vector  = normal;
    radial_profile_z8->normal_vector  = normal;
    radial_profile_z9->normal_vector  = normal;
    radial_profile_z10->normal_vector = normal;
    radial_profile_z11->normal_vector = normal;
    radial_profile_z12->normal_vector = normal;

    // quantities

    // no additional averaging in space for centerline velocity
    std::shared_ptr<Quantity> quantity_velocity;
    quantity_velocity.reset(new Quantity());
    quantity_velocity->type = QuantityType::Velocity;

    axial_profile->quantities.push_back(quantity_velocity);
    radial_profile_z1->quantities.push_back(quantity_velocity);
    radial_profile_z2->quantities.push_back(quantity_velocity);
    radial_profile_z3->quantities.push_back(quantity_velocity);
    radial_profile_z4->quantities.push_back(quantity_velocity);
    radial_profile_z5->quantities.push_back(quantity_velocity);
    radial_profile_z6->quantities.push_back(quantity_velocity);
    radial_profile_z7->quantities.push_back(quantity_velocity);
    radial_profile_z8->quantities.push_back(quantity_velocity);
    radial_profile_z9->quantities.push_back(quantity_velocity);
    radial_profile_z10->quantities.push_back(quantity_velocity);
    radial_profile_z11->quantities.push_back(quantity_velocity);
    radial_profile_z12->quantities.push_back(quantity_velocity);

    // names
    axial_profile->name      = "axial_profile";
    radial_profile_z1->name  = "radial_profile_z1";
    radial_profile_z2->name  = "radial_profile_z2";
    radial_profile_z3->name  = "radial_profile_z3";
    radial_profile_z4->name  = "radial_profile_z4";
    radial_profile_z5->name  = "radial_profile_z5";
    radial_profile_z6->name  = "radial_profile_z6";
    radial_profile_z7->name  = "radial_profile_z7";
    radial_profile_z8->name  = "radial_profile_z8";
    radial_profile_z9->name  = "radial_profile_z9";
    radial_profile_z10->name = "radial_profile_z10";
    radial_profile_z11->name = "radial_profile_z11";
    radial_profile_z12->name = "radial_profile_z12";

    // insert lines
    pp_data_fda.line_plot_data.lines.push_back(axial_profile);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z1);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z2);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z3);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z4);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z5);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z6);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z7);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z8);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z9);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z10);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z11);
    pp_data_fda.line_plot_data.lines.push_back(radial_profile_z12);

    AssertThrow(inflow_data_storage.get(),
                dealii::ExcMessage("inflow_data_storage is uninitialized."));

    std::shared_ptr<FlowRateController> flow_rate_controller_dummy;

    pp.reset(new PostProcessorFDA<dim, Number>(pp_data_fda,
                                               this->mpi_comm,
                                               area_inflow,
                                               flow_rate_controller_dummy,
                                               inflow_data_storage,
                                               use_precursor,
                                               use_random_perturbations));

    return pp;
  }

private:
  std::shared_ptr<InflowDataStorage<dim>> inflow_data_storage;

  bool const use_precursor;
};


template<int dim, typename Number>
class Application : public Precursor::ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : Precursor::ApplicationBase<dim, Number>(input_file, comm)
  {
    this->switch_off_precursor = switch_off_precursor;

    // compute number of points for inflow data array depending on spatial resolution of problem
    ResolutionParameters resolution_main;

    dealii::ParameterHandler prm;
    prm.enter_subsection("Main");
    {
      resolution_main.add_parameters(prm);
    }
    prm.leave_subsection();

    prm.parse_input(input_file, "", true, true);

    unsigned int const n_points =
      20 * (resolution_main.degree + 1) * dealii::Utilities::pow(2, resolution_main.refine_space);

    inflow_data_storage.reset(new InflowDataStorage<dim>(n_points,
                                                         FDANozzle::R_OUTER,
                                                         max_velocity,
                                                         use_random_perturbations,
                                                         factor_random_perturbations));

    this->precursor =
      std::make_shared<PrecursorDomain<dim, Number>>(input_file, comm, inflow_data_storage);
    this->main = std::make_shared<MainDomain<dim, Number>>(input_file,
                                                           comm,
                                                           inflow_data_storage,
                                                           this->precursor_is_active());
  }

private:
  std::shared_ptr<InflowDataStorage<dim>> inflow_data_storage;
};

} // namespace Precursor
} // namespace IncNS
} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/precursor/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FDA_H_ */
