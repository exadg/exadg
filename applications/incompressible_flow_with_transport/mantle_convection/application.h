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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_MANTLE_CONVECTION_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_MANTLE_CONVECTION_H_

namespace ExaDG
{
namespace FTI
{
template<int dim>
class RightHandSide : public dealii::Function<dim>
{
public:
  RightHandSide() : dealii::Function<dim>(dim, 0.0)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component) const
  {
    double const r = p.norm();

    double g = -p[component] / r;

    return g;
  }
};

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
    if(dim == 2)
      perturbation *=
        0.25 * delta_T *
        std::pow(std::sin(dealii::numbers::PI * (p[0] + std::sqrt(2)) / (length / 2.)) *
                   std::sin(dealii::numbers::PI * (p[1] + std::sqrt(3)) / (length / 2.)),
                 2.0);
    else if(dim == 3)
      perturbation *=
        0.25 * delta_T *
        std::pow(std::sin(dealii::numbers::PI * (p[0] + std::sqrt(2)) / (length / 2.)) *
                   std::sin(dealii::numbers::PI * (p[1] + std::sqrt(3)) / (length / 2.)) *
                   std::sin(dealii::numbers::PI * (p[2] + std::sqrt(5)) / (length / 2.)),
                 2.0);
    else
      AssertThrow(false, dealii::ExcMessage("not implemented."));

    return (T_ref + delta_T + perturbation);
  }

private:
  double const T_ref, delta_T, length, characteristic_time;
};

template<int dim, typename Number>
class Application : public FTI::ApplicationBase<dim, Number>
{
public:
  // physical quantities
  double const R0 = 0.55;
  double const R1 = 1.0;

  double const Ra                  = 1.0e8;
  double const beta                = Ra;
  double const kinematic_viscosity = 1.0;
  double const thermal_diffusivity = 1.0;

  double const T0 = 1.0;
  double const T1 = 0.0;

  double const H = R1 - R0;
  double const U = std::sqrt(Ra * kinematic_viscosity * thermal_diffusivity / (H * H));
  double const characteristic_time = H / U;
  double const start_time          = 0.0;
  double const end_time            = 200.0 * characteristic_time;

  // CFL > 4 did not show further speed-up for 2d example
  double const CFL                    = 2.0; // 0.4;
  bool const   adaptive_time_stepping = true;

  // solver tolerance
  double const reltol = 1.e-3;

  // vtu output
  double const output_interval_time = (end_time - start_time) / 200.0;

  Application(std::string input_file, MPI_Comm const & comm)
    : FTI::ApplicationBase<dim, Number>(input_file, comm, 1)
  {
    // parse application-specific parameters
    dealii::ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  void
  set_parameters() final
  {
    using namespace IncNS;

    // MATHEMATICAL MODEL
    this->param.problem_type                 = ProblemType::Steady;
    this->param.equation_type                = EquationType::Stokes;
    this->param.formulation_viscous_term     = FormulationViscousTerm::LaplaceFormulation;
    this->param.formulation_convective_term  = FormulationConvectiveTerm::ConvectiveFormulation;
    this->param.right_hand_side              = true;
    this->param.boussinesq_term              = true;
    this->param.boussinesq_dynamic_part_only = true;

    // PHYSICAL QUANTITIES
    this->param.start_time                    = start_time;
    this->param.end_time                      = end_time;
    this->param.viscosity                     = kinematic_viscosity;
    this->param.thermal_expansion_coefficient = beta;
    this->param.reference_temperature         = T1;

    // TEMPORAL DISCRETIZATION
    this->param.solver_type                     = SolverType::Steady;
    this->param.temporal_discretization         = TemporalDiscretization::BDFCoupledSolution;
    this->param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    this->param.adaptive_time_stepping          = adaptive_time_stepping;
    this->param.order_time_integrator           = 2;
    this->param.start_with_low_order            = true;
    this->param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    this->param.max_velocity                    = U;
    this->param.cfl_exponent_fe_degree_velocity = 1.5;
    this->param.cfl                             = CFL;
    this->param.time_step_size                  = characteristic_time / 1.e0;
    this->param.time_step_size_max              = characteristic_time / 1.e0;

    // output of solver information
    this->param.solver_info_data.interval_time = output_interval_time;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.grid.mapping_degree     = this->param.degree_u;
    this->param.degree_p                = DegreePressure::MixedOrder;

    // convective term
    if(this->param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      this->param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

    // viscous term
    this->param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // div-div and continuity penalty
    this->param.use_divergence_penalty                     = false;
    this->param.divergence_penalty_factor                  = 1.0;
    this->param.use_continuity_penalty                     = false;
    this->param.continuity_penalty_factor                  = this->param.divergence_penalty_factor;
    this->param.continuity_penalty_components              = ContinuityPenaltyComponents::Normal;
    this->param.continuity_penalty_use_boundary_data       = true;
    this->param.apply_penalty_terms_in_postprocessing_step = false;
    this->param.type_penalty_parameter                     = TypePenaltyParameter::ConvectiveTerm;

    // NUMERICAL PARAMETERS
    this->param.implement_block_diagonal_preconditioner_matrix_free = false;
    this->param.use_cell_based_face_loops                           = false;

    // PROJECTION METHODS

    // pressure Poisson equation
    this->param.solver_pressure_poisson              = SolverPressurePoisson::CG;
    this->param.solver_data_pressure_poisson         = SolverData(1000, 1.e-30, reltol, 100);
    this->param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    this->param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;

    // projection step
    this->param.solver_projection         = SolverProjection::CG;
    this->param.solver_data_projection    = SolverData(1000, 1.e-30, reltol);
    this->param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulation
    this->param.order_extrapolation_pressure_nbc =
      this->param.order_time_integrator <= 2 ? this->param.order_time_integrator : 2;

    // viscous step
    this->param.solver_viscous              = SolverViscous::CG;
    this->param.solver_data_viscous         = SolverData(1000, 1.e-30, reltol);
    this->param.multigrid_data_viscous.type = MultigridType::cphMG;


    // PRESSURE-CORRECTION SCHEME

    // formulation
    this->param.order_pressure_extrapolation = 1;
    this->param.rotational_formulation       = true;

    // momentum step

    // Newton solver
    this->param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-30, reltol);

    // linear solver
    this->param.solver_momentum                  = SolverMomentum::CG;
    this->param.solver_data_momentum             = SolverData(1e4, 1.e-30, reltol, 100);
    this->param.preconditioner_momentum          = MomentumPreconditioner::Multigrid;
    this->param.multigrid_data_momentum.type     = MultigridType::cphMG;
    this->param.multigrid_operator_type_momentum = MultigridOperatorType::ReactionDiffusion;

    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    this->param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-30, reltol);

    // linear solver
    this->param.solver_coupled         = SolverCoupled::GMRES;
    this->param.solver_data_coupled    = SolverData(1e3, 1.e-30, reltol, 100);
    this->param.use_scaling_continuity = false;

    // preconditioner linear solver
    this->param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    this->param.update_preconditioner_coupled = false;

    // preconditioner velocity/momentum block
    this->param.preconditioner_velocity_block          = MomentumPreconditioner::Multigrid;
    this->param.multigrid_data_velocity_block.type     = MultigridType::cphMG;
    this->param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;
    this->param.multigrid_data_velocity_block.smoother_data.smoother = MultigridSmoother::Chebyshev;
    this->param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::PointJacobi;
    this->param.multigrid_data_velocity_block.smoother_data.iterations = 5;

    // preconditioner Schur-complement block
    this->param.preconditioner_pressure_block = SchurComplementPreconditioner::InverseMassMatrix;
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
    param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Implicit;
    param.adaptive_time_stepping        = adaptive_time_stepping;
    param.order_time_integrator         = 2;
    param.start_with_low_order          = true;
    param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    param.time_step_size                = 1.0e-2;
    param.cfl                           = CFL;
    param.max_velocity                  = U;
    param.exponent_fe_degree_convection = 1.5;
    param.time_step_size                = characteristic_time / 1.e0;
    param.time_step_size_max            = characteristic_time / 1.e0;

    // output of solver information
    param.solver_info_data.interval_time = output_interval_time;

    // SPATIAL DISCRETIZATION
    param.grid.triangulation_type = TriangulationType::Distributed;
    param.grid.mapping_degree     = param.degree;

    // convective term
    param.numerical_flux_convective_operator = NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

    // viscous term
    param.IP_factor = 1.0;

    // NUMERICAL PARAMETERS
    param.implement_block_diagonal_preconditioner_matrix_free = false;
    param.use_cell_based_face_loops                           = false;

    // SOLVER
    param.solver                = ConvDiff::Solver::GMRES; // CG;
    param.solver_data           = SolverData(1e4, 1.e-30, reltol, 100);
    param.preconditioner        = Preconditioner::InverseMassMatrix;
    param.multigrid_data.type   = MultigridType::cphMG;
    param.mg_operator_type      = MultigridOperatorType::ReactionDiffusion;
    param.update_preconditioner = false;

    // NUMERICAL PARAMETERS
    param.use_combined_operator = true;
    param.use_overintegration   = true;

    this->scalar_param[scalar_index] = param;
  }

  void
  create_grid() final
  {
    dealii::GridGenerator::hyper_shell(
      *this->grid->triangulation, dealii::Point<dim>(), R0, R1, (dim == 3) ? 48 : 12);

    dealii::Point<dim> center =
      dim == 2 ? dealii::Point<dim>(0., 0.) : dealii::Point<dim>(0., 0., 0.);

    for(auto cell : this->grid->triangulation->active_cell_iterators())
    {
      for(unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
      {
        bool face_at_outer_boundary = true;
        for(unsigned int v = 0; v < dealii::GeometryInfo<dim - 1>::vertices_per_cell; ++v)
        {
          if(std::abs(center.distance(cell->face(f)->vertex(v)) - R1) > 1e-12 * R1)
            face_at_outer_boundary = false;
        }

        if(face_at_outer_boundary)
          cell->face(f)->set_boundary_id(1);
      }
    }

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
    this->boundary_descriptor->pressure->neumann_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));
    this->boundary_descriptor->pressure->neumann_bc.insert(
      pair(1, new dealii::Functions::ZeroFunction<dim>(dim)));
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
    this->field_functions->gravitational_force.reset(new RightHandSide<dim>());
  }

  std::shared_ptr<IncNS::PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    IncNS::PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output       = this->write_output;
    pp_data.output_data.directory          = this->output_directory + "vtu/";
    pp_data.output_data.filename           = this->output_name + "_fluid";
    pp_data.output_data.start_time         = start_time;
    pp_data.output_data.interval_time      = output_interval_time;
    pp_data.output_data.write_processor_id = true;
    pp_data.output_data.degree             = this->param.degree_u;
    pp_data.output_data.write_higher_order = (dim == 2);

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
      pair(0, new TemperatureBC<dim>(T1, T0 - T1, H, characteristic_time)));
    this->scalar_boundary_descriptor[scalar_index]->dirichlet_bc.insert(
      pair(1, new dealii::Functions::ConstantFunction<dim>(T1)));
  }

  void
  set_field_functions_scalar(unsigned int scalar_index = 0) final
  {
    this->scalar_field_functions[scalar_index]->initial_solution.reset(
      new dealii::Functions::ConstantFunction<dim>(T1));
    this->scalar_field_functions[scalar_index]->right_hand_side.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->scalar_field_functions[scalar_index]->velocity.reset(
      new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>>
  create_postprocessor_scalar(unsigned int const scalar_index) final
  {
    ConvDiff::PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output = this->write_output;
    pp_data.output_data.directory    = this->output_directory + "vtu/";
    pp_data.output_data.filename   = this->output_name + "_scalar_" + std::to_string(scalar_index);
    pp_data.output_data.start_time = start_time;
    pp_data.output_data.interval_time      = output_interval_time;
    pp_data.output_data.degree             = this->scalar_param[scalar_index].degree;
    pp_data.output_data.write_higher_order = (dim == 2);
    pp_data.output_data.write_surface_mesh = true;

    std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>> pp;
    pp.reset(new ConvDiff::PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

} // namespace FTI

} // namespace ExaDG

#include <exadg/incompressible_flow_with_transport/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_MANTLE_CONVECTION_H_ */
