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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_NATURAL_CONVECTION_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_NATURAL_CONVECTION_H_

#include <exadg/grid/mesh_movement_functions.h>

namespace ExaDG
{
namespace FTI
{
template<int dim, typename Number>
class Application : public FTI::ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : FTI::ApplicationBase<dim, Number>(input_file, comm, 1)
  {
  }

private:
  void
  set_parameters() final
  {
    using namespace IncNS;

    // MATHEMATICAL MODEL
    this->param.problem_type                = ProblemType::Unsteady;
    this->param.equation_type               = EquationType::NavierStokes;
    this->param.formulation_viscous_term    = FormulationViscousTerm::LaplaceFormulation;
    this->param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
    this->param.ale_formulation             = ALE;
    this->param.mesh_movement_type          = MeshMovementType::Function;
    this->param.right_hand_side             = true;
    this->param.boussinesq_term             = true;


    // PHYSICAL QUANTITIES
    this->param.start_time                    = start_time;
    this->param.end_time                      = end_time;
    this->param.viscosity                     = kinematic_viscosity;
    this->param.thermal_expansion_coefficient = beta;
    this->param.reference_temperature         = T_ref;


    // TEMPORAL DISCRETIZATION
    this->param.solver_type                     = SolverType::Unsteady;
    this->param.temporal_discretization         = TemporalDiscretization::BDFCoupledSolution;
    this->param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    this->param.adaptive_time_stepping          = adaptive_time_stepping;
    this->param.order_time_integrator           = 2;
    this->param.start_with_low_order            = true;
    this->param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    this->param.max_velocity                    = max_velocity;
    this->param.cfl_exponent_fe_degree_velocity = 1.5;
    this->param.cfl                             = CFL;
    this->param.time_step_size                  = 1.0e-1;

    // output of solver information
    this->param.solver_info_data.interval_time = (end_time - start_time) / 10.;

    // restart
    this->param.restart_data.write_restart = write_restart;
    this->param.restart_data.interval_time = restart_interval_time;
    this->param.restart_data.filename =
      this->output_parameters.directory + this->output_parameters.filename + "_fluid";

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.grid.mapping_degree     = 1;
    this->param.degree_p                = DegreePressure::MixedOrder;

    // convective term
    if(this->param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      this->param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

    // viscous term
    this->param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // div-div and continuity penalty
    this->param.use_divergence_penalty                     = true;
    this->param.divergence_penalty_factor                  = 1.0e0;
    this->param.use_continuity_penalty                     = true;
    this->param.continuity_penalty_factor                  = this->param.divergence_penalty_factor;
    this->param.continuity_penalty_components              = ContinuityPenaltyComponents::Normal;
    this->param.continuity_penalty_use_boundary_data       = true;
    this->param.apply_penalty_terms_in_postprocessing_step = true;
    this->param.type_penalty_parameter                     = TypePenaltyParameter::ConvectiveTerm;

    // NUMERICAL PARAMETERS
    this->param.implement_block_diagonal_preconditioner_matrix_free = false;
    this->param.use_cell_based_face_loops                           = false;

    // PROJECTION METHODS

    // pressure Poisson equation
    this->param.solver_pressure_poisson              = SolverPressurePoisson::CG;
    this->param.solver_data_pressure_poisson         = SolverData(1000, ABS_TOL, REL_TOL, 100);
    this->param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    this->param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
    this->param.multigrid_data_pressure_poisson.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev;
    this->param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::PointJacobi;
    this->param.multigrid_data_pressure_poisson.smoother_data.smoother =
      MultigridSmoother::Chebyshev;
    this->param.multigrid_data_pressure_poisson.smoother_data.preconditioner =
      PreconditionerSmoother::PointJacobi;

    // projection step
    this->param.solver_projection         = SolverProjection::CG;
    this->param.solver_data_projection    = SolverData(1000, ABS_TOL, REL_TOL);
    this->param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulation
    this->param.order_extrapolation_pressure_nbc =
      this->param.order_time_integrator <= 2 ? this->param.order_time_integrator : 2;

    // viscous step
    this->param.solver_viscous         = SolverViscous::CG;
    this->param.solver_data_viscous    = SolverData(1000, ABS_TOL, REL_TOL);
    this->param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;


    // PRESSURE-CORRECTION SCHEME

    // formulation
    this->param.order_pressure_extrapolation = 1;
    this->param.rotational_formulation       = true;

    // momentum step

    // Newton solver
    this->param.newton_solver_data_momentum = Newton::SolverData(100, ABS_TOL, REL_TOL);

    // linear solver
    this->param.solver_momentum = SolverMomentum::GMRES;
    if(this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      this->param.solver_data_momentum = SolverData(1e4, ABS_TOL_LINEAR, REL_TOL_LINEAR, 100);
    else
      this->param.solver_data_momentum = SolverData(1e4, ABS_TOL, REL_TOL, 100);

    this->param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;


    // COUPLED NAVIER-STOKES SOLVER

    this->param.use_scaling_continuity = false;

    // nonlinear solver (Newton solver)
    this->param.newton_solver_data_coupled = Newton::SolverData(100, ABS_TOL, REL_TOL);

    // linear solver
    this->param.solver_coupled = SolverCoupled::GMRES;
    if(this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      this->param.solver_data_coupled = SolverData(1e3, ABS_TOL_LINEAR, REL_TOL_LINEAR, 100);
    else
      this->param.solver_data_coupled = SolverData(1e3, ABS_TOL, REL_TOL, 100);

    // preconditioner linear solver
    this->param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    this->param.update_preconditioner_coupled = false;

    // preconditioner velocity/momentum block
    this->param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

    // preconditioner Schur-complement block
    this->param.preconditioner_pressure_block =
      SchurComplementPreconditioner::PressureConvectionDiffusion;
  }

  void
  set_parameters_scalar(unsigned int const scalar_index) final
  {
    using namespace ConvDiff;

    Parameters param;

    // MATHEMATICAL MODEL
    param.problem_type                = ProblemType::Unsteady;
    param.equation_type               = EquationType::ConvectionDiffusion;
    param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
    param.analytical_velocity_field   = false;
    param.right_hand_side             = false;
    param.ale_formulation             = ALE;

    // PHYSICAL QUANTITIES
    param.start_time  = start_time;
    param.end_time    = end_time;
    param.diffusivity = thermal_diffusivity;

    // TEMPORAL DISCRETIZATION
    param.temporal_discretization       = TemporalDiscretization::BDF;
    param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Explicit;
    param.adaptive_time_stepping        = adaptive_time_stepping;
    param.order_time_integrator         = 2;
    param.start_with_low_order          = true;
    param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    param.time_step_size                = 1.0e-2;
    param.cfl                           = CFL;
    param.max_velocity                  = max_velocity;
    param.exponent_fe_degree_convection = 1.5;
    param.exponent_fe_degree_diffusion  = 3.0;
    param.diffusion_number              = 0.01;

    // restart
    param.restart_data.write_restart = write_restart;
    param.restart_data.interval_time = restart_interval_time;
    param.restart_data.filename      = this->output_parameters.directory +
                                  this->output_parameters.filename + "_scalar_" +
                                  std::to_string(scalar_index);

    // output of solver information
    param.solver_info_data.interval_time = (end_time - start_time) / 10.;

    // SPATIAL DISCRETIZATION
    param.grid.triangulation_type = TriangulationType::Distributed;
    param.grid.mapping_degree     = 1;

    // convective term
    param.numerical_flux_convective_operator = NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

    // viscous term
    param.IP_factor = 1.0;

    // NUMERICAL PARAMETERS
    param.implement_block_diagonal_preconditioner_matrix_free = false;
    param.use_cell_based_face_loops                           = false;

    // SOLVER
    param.solver                    = ConvDiff::Solver::CG;
    param.solver_data               = SolverData(1e3, ABS_TOL, REL_TOL, 100);
    param.preconditioner            = Preconditioner::InverseMassMatrix;
    param.multigrid_data.type       = MultigridType::phMG;
    param.multigrid_data.p_sequence = PSequenceType::Bisect;
    param.mg_operator_type          = MultigridOperatorType::ReactionDiffusion;
    param.update_preconditioner     = false;

    // output of solver information
    param.solver_info_data.interval_time = (end_time - start_time) / 10.;

    // NUMERICAL PARAMETERS
    param.use_combined_operator = true;
    param.use_overintegration   = true;

    this->scalar_param[scalar_index] = param;
  }

  void
  create_grid() final
  {
    dealii::GridGenerator::hyper_cube(*this->grid->triangulation, left, right);

    // set boundary IDs: 0 by default, set left boundary to 1
    for(auto cell : this->grid->triangulation->cell_iterators())
    {
      for(auto const & f : cell->face_indices())
      {
        if((std::fabs(cell->face(f)->center()(0) - left) < 1e-12))
        {
          cell->face(f)->set_boundary_id(1);
        }

        // lower and upper boundary
        if((std::fabs(cell->face(f)->center()(1) - left) < 1e-12) ||
           (std::fabs(cell->face(f)->center()(1) - right) < 1e-12))
        {
          cell->face(f)->set_boundary_id(2);
        }
      }
    }

    this->grid->triangulation->refine_global(this->param.grid.n_refine_global);
  }

  std::shared_ptr<dealii::Function<dim>>
  create_mesh_movement_function() final
  {
    std::shared_ptr<dealii::Function<dim>> mesh_motion;

    MeshMovementData<dim> data;
    data.temporal      = MeshMovementAdvanceInTime::Sin;
    data.shape         = MeshMovementShape::SineAligned; // SineZeroAtBoundary; //SineAligned;
    data.dimensions[0] = std::abs(right - left);
    data.dimensions[1] = std::abs(right - left);
    data.amplitude     = 0.08 * (right - left); // A_max = (right-left)/(2*pi)
    data.period        = end_time;
    data.t_start       = 0.0;
    data.t_end         = end_time;
    data.spatial_number_of_oscillations = 1.0;
    mesh_motion.reset(new CubeMeshMovementFunctions<dim>(data));

    return mesh_motion;
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    // fill boundary descriptor velocity
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(1, new dealii::Functions::ZeroFunction<dim>(dim)));
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(2, new dealii::Functions::ZeroFunction<dim>(dim)));

    // fill boundary descriptor pressure
    this->boundary_descriptor->pressure->neumann_bc.insert(0);
    this->boundary_descriptor->pressure->neumann_bc.insert(1);
    this->boundary_descriptor->pressure->neumann_bc.insert(2);
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(
      new dealii::Functions::ZeroFunction<dim>(dim));
    this->field_functions->initial_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->analytical_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    std::vector<double> gravity = std::vector<double>(dim, 0.0);
    gravity[1]                  = -g;
    this->field_functions->gravitational_force.reset(
      new dealii::Functions::ConstantFunction<dim>(gravity));
  }

  std::shared_ptr<IncNS::PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    IncNS::PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = start_time;
    pp_data.output_data.time_control_data.trigger_interval = output_interval_time;
    pp_data.output_data.directory          = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = this->output_parameters.filename + "_fluid";
    pp_data.output_data.write_processor_id = true;
    pp_data.output_data.degree             = this->param.degree_u;
    pp_data.output_data.write_higher_order = true;

    std::shared_ptr<IncNS::PostProcessorBase<dim, Number>> pp;
    pp.reset(new IncNS::PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  void
  set_boundary_descriptor_scalar(unsigned int scalar_index = 0) final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    this->scalar_boundary_descriptor[scalar_index]->dirichlet_bc.insert(
      pair(0, new dealii::Functions::ConstantFunction<dim>(T_ref)));
    this->scalar_boundary_descriptor[scalar_index]->dirichlet_bc.insert(
      pair(1, new dealii::Functions::ConstantFunction<dim>(T_ref + delta_T)));
    this->scalar_boundary_descriptor[scalar_index]->neumann_bc.insert(
      pair(2, new dealii::Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions_scalar(unsigned int scalar_index = 0) final
  {
    this->scalar_field_functions[scalar_index]->initial_solution.reset(
      new dealii::Functions::ConstantFunction<dim>(T_ref));
    this->scalar_field_functions[scalar_index]->right_hand_side.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->scalar_field_functions[scalar_index]->velocity.reset(
      new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>>
  create_postprocessor_scalar(unsigned int const scalar_index) final
  {
    ConvDiff::PostProcessorData<dim> pp_data;
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = start_time;
    pp_data.output_data.time_control_data.trigger_interval = output_interval_time;
    pp_data.output_data.directory = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename =
      this->output_parameters.filename + "_scalar_" + std::to_string(scalar_index);
    pp_data.output_data.degree             = this->scalar_param[scalar_index].degree;
    pp_data.output_data.write_higher_order = true;

    std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>> pp;
    pp.reset(new ConvDiff::PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  // Problem specific parameters
  double const L        = 1.0;
  double const T_ref    = 300.0;
  double const delta_T  = 1.0;
  double const g        = 10.0;
  double const beta     = 1.0 / 300.0;
  double const Prandtl  = 1.0;
  double const Rayleigh = 1.0e8;

  // dependent parameters
  double const kinematic_viscosity =
    std::sqrt(g * beta * delta_T * std::pow(L, 3.0) * Prandtl / Rayleigh);
  double const thermal_diffusivity = kinematic_viscosity / Prandtl;

  double const left  = -L / 2.0;
  double const right = L / 2.0;

  double const U                   = std::sqrt(g * beta * delta_T * L);
  double const characteristic_time = L / U;
  double const start_time          = 0.0;
  double const end_time            = 10.0 * characteristic_time;

  double const CFL                    = 0.3;
  double const max_velocity           = 1.0;
  bool const   adaptive_time_stepping = true;

  // vtu output
  double const output_interval_time = (end_time - start_time) / 100.0;

  // restart
  bool const   write_restart         = false;
  double const restart_interval_time = 10.0;

  // moving mesh (ALE)
  bool const ALE = false;

  // solver tolerances
  double const ABS_TOL = 1.e-12;
  double const REL_TOL = 1.e-6;

  double const ABS_TOL_LINEAR = 1.e-12;
  double const REL_TOL_LINEAR = 1.e-2;
};

} // namespace FTI

} // namespace ExaDG

#include <exadg/incompressible_flow_with_transport/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_NATURAL_CONVECTION_H_ */
