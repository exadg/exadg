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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_RAYLEIGH_BENARD_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_RAYLEIGH_BENARD_H_

namespace ExaDG
{
namespace FTI
{
template<int dim>
class TemperatureBC : public dealii::Function<dim>
{
public:
  TemperatureBC(double const T_ref,
                double const delta_T,
                double const length,
                double const characteristic_time)
    : dealii::Function<dim>(1, 0.0),
      T_ref(T_ref),
      delta_T(delta_T),
      length(length),
      characteristic_time(characteristic_time)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const /*component = 0*/) const
  {
    double       t           = this->get_time();
    double const time_factor = std::max(0.0, 1.0 - t / characteristic_time);

    double perturbation = time_factor;
    perturbation *=
      0.25 * delta_T * std::pow(std::sin(dealii::numbers::PI * p[0] / (length / 4.)), 2.0);
    if(dim == 3)
      perturbation *= std::pow(std::sin(dealii::numbers::PI * p[2] / (length / 4.)), 2.0);

    return (T_ref + delta_T + perturbation);
  }

private:
  double const T_ref, delta_T, length, characteristic_time;
};

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
    this->param.temporal_discretization         = TemporalDiscretization::BDFDualSplittingScheme;
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
    this->param.divergence_penalty_factor                  = 1.0;
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
    this->param.solver_data_pressure_poisson         = SolverData(1000, 1.e-12, 1.e-6, 100);
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
    this->param.solver_data_projection    = SolverData(1000, 1.e-12, 1.e-6);
    this->param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulation
    this->param.order_extrapolation_pressure_nbc =
      this->param.order_time_integrator <= 2 ? this->param.order_time_integrator : 2;

    // viscous step
    this->param.solver_viscous         = SolverViscous::CG;
    this->param.solver_data_viscous    = SolverData(1000, 1.e-12, 1.e-6);
    this->param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;


    // PRESSURE-CORRECTION SCHEME

    // formulation
    this->param.order_pressure_extrapolation = 1;
    this->param.rotational_formulation       = true;

    // momentum step

    // Newton solver
    this->param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-20, 1.e-6);

    // linear solver
    this->param.solver_momentum         = SolverMomentum::GMRES;
    this->param.solver_data_momentum    = SolverData(1e4, 1.e-12, 1.e-6, 100);
    this->param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;


    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    this->param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-12, 1.e-6);

    // linear solver
    this->param.solver_coupled      = SolverCoupled::GMRES;
    this->param.solver_data_coupled = SolverData(1e3, 1.e-12, 1.e-6, 100);

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
    param.solver_data               = SolverData(1e4, 1.e-12, 1.e-6, 100);
    param.preconditioner            = Preconditioner::InverseMassMatrix;
    param.multigrid_data.type       = MultigridType::pMG;
    param.multigrid_data.p_sequence = PSequenceType::Bisect;
    param.mg_operator_type          = MultigridOperatorType::ReactionDiffusion;
    param.update_preconditioner     = false;

    // output of solver information
    param.solver_info_data.interval_time = (end_time - start_time) / 10.;

    // NUMERICAL PARAMETERS
    param.use_combined_operator = true;

    this->scalar_param[scalar_index] = param;
  }

  void
  create_grid() final
  {
    if(dim == 2)
    {
      std::vector<unsigned int> repetitions({8, 1});
      dealii::Point<dim>        point1(-L / 2., 0.0), point2(L / 2., H);
      dealii::GridGenerator::subdivided_hyper_rectangle(*this->grid->triangulation,
                                                        repetitions,
                                                        point1,
                                                        point2);
    }
    else if(dim == 3)
    {
      std::vector<unsigned int> repetitions({8, 1, 8});
      dealii::Point<dim>        point1(-L / 2., 0.0, -L / 2.), point2(L / 2., H, L / 2.);
      dealii::GridGenerator::subdivided_hyper_rectangle(*this->grid->triangulation,
                                                        repetitions,
                                                        point1,
                                                        point2);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    // set boundary IDs: 0 by default, set left boundary to 1
    for(auto cell : this->grid->triangulation->active_cell_iterators())
    {
      for(auto const & f : cell->face_indices())
      {
        if((std::fabs(cell->face(f)->center()(1) - 0.0) < 1e-12))
          cell->face(f)->set_boundary_id(1);

        // periodicity in x-direction
        if((std::fabs(cell->face(f)->center()(0) + L / 2.) < 1e-12))
          cell->face(f)->set_boundary_id(10);
        if((std::fabs(cell->face(f)->center()(0) - L / 2.) < 1e-12))
          cell->face(f)->set_boundary_id(11);

        // periodicity in z-direction
        if(dim == 3)
        {
          if((std::fabs(cell->face(f)->center()(2) + L / 2.) < 1e-12))
            cell->face(f)->set_boundary_id(12);
          if((std::fabs(cell->face(f)->center()(2) - L / 2.) < 1e-12))
            cell->face(f)->set_boundary_id(13);
        }
      }
    }

    dealii::GridTools::collect_periodic_faces(
      *this->grid->triangulation, 10, 11, 0, this->grid->periodic_faces);
    if(dim == 3)
      dealii::GridTools::collect_periodic_faces(
        *this->grid->triangulation, 12, 13, 2, this->grid->periodic_faces);

    this->grid->triangulation->add_periodicity(this->grid->periodic_faces);

    this->grid->triangulation->refine_global(this->param.grid.n_refine_global);
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

    // fill boundary descriptor pressure
    this->boundary_descriptor->pressure->neumann_bc.insert(0);
    this->boundary_descriptor->pressure->neumann_bc.insert(1);
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
    pp_data.output_data.write_output       = this->output_parameters.write;
    pp_data.output_data.directory          = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = this->output_parameters.filename + "_fluid";
    pp_data.output_data.start_time         = start_time;
    pp_data.output_data.interval_time      = output_interval_time;
    pp_data.output_data.write_processor_id = true;
    pp_data.output_data.write_surface_mesh = true;
    pp_data.output_data.degree             = this->param.degree_u;
    pp_data.output_data.write_higher_order = false;

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
      pair(1, new TemperatureBC<dim>(T_ref, delta_T, L, characteristic_time)));
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
    pp_data.output_data.write_output = this->output_parameters.write;
    pp_data.output_data.directory    = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename =
      this->output_parameters.filename + "_scalar_" + std::to_string(scalar_index);
    pp_data.output_data.start_time         = start_time;
    pp_data.output_data.interval_time      = output_interval_time;
    pp_data.output_data.degree             = this->scalar_param[scalar_index].degree;
    pp_data.output_data.write_higher_order = true;

    std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>> pp;
    pp.reset(new ConvDiff::PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  // physical quantities
  double const L = 8.0;
  double const H = 1.0;

  double const Prandtl = 1.0;
  double const Re      = std::sqrt(1.0e8);
  double const Ra      = Re * Re * Prandtl;
  double const g       = 10.0;
  double const T_ref   = 0.0;
  double const beta    = 1.0 / 300.0;
  double const U       = 1.0;

  double const kinematic_viscosity = U * H / Re;
  double const thermal_diffusivity = kinematic_viscosity / Prandtl;

  // u^2 = g * beta * Delta_T * h
  double const delta_T = std::pow(U, 2.0) / beta / g / H;

  double const start_time          = 0.0;
  double const characteristic_time = H / U;
  double const end_time            = 200.0 * characteristic_time;

  // time stepping
  double const CFL                    = 0.4;
  double const max_velocity           = 1.0;
  bool const   adaptive_time_stepping = true;

  // vtu output
  double const output_interval_time = (end_time - start_time) / 100.0;
};

} // namespace FTI

} // namespace ExaDG

#include <exadg/incompressible_flow_with_transport/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_RAYLEIGH_BENARD_H_ */
