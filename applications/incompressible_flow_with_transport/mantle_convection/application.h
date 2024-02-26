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
  value(dealii::Point<dim> const & p, unsigned int const component) const final
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
  value(dealii::Point<dim> const & p, unsigned int const /*component = 0*/) const final
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

template<int dim, typename Number>
class Fluid : public FluidBase<dim, Number>
{
public:
  Fluid(std::string parameter_file, MPI_Comm const & comm)
    : FluidBase<dim, Number>(parameter_file, comm)
  {
  }

private:
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
    this->param.grid.triangulation_type     = TriangulationType::Distributed;
    this->param.mapping_degree              = this->param.degree_u;
    this->param.mapping_degree_coarse_grids = this->param.mapping_degree;
    this->param.degree_p                    = DegreePressure::MixedOrder;

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
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) final
  {
    auto const lambda_create_triangulation = [&](dealii::Triangulation<dim, dim> & tria,
                                                 std::vector<dealii::GridTools::PeriodicFacePair<
                                                   typename dealii::Triangulation<
                                                     dim>::cell_iterator>> & periodic_face_pairs,
                                                 unsigned int const          global_refinements,
                                                 std::vector<unsigned int> const &
                                                   vector_local_refinements) {
      (void)periodic_face_pairs;
      (void)vector_local_refinements;

      AssertThrow(
        this->param.grid.triangulation_type != TriangulationType::FullyDistributed,
        dealii::ExcMessage(
          "Manifolds might not be applied correctly for TriangulationType::FullyDistributed. "
          "Try to use another triangulation type, or try to fix these limitations in ExaDG or deal.II."));

      dealii::GridGenerator::hyper_shell(tria, dealii::Point<dim>(), R0, R1, (dim == 3) ? 48 : 12);

      dealii::Point<dim> center =
        dim == 2 ? dealii::Point<dim>(0., 0.) : dealii::Point<dim>(0., 0., 0.);

      for(auto cell : tria.cell_iterators())
      {
        for(auto const & f : cell->face_indices())
        {
          if(cell->face(f)->at_boundary())
          {
            bool face_at_outer_boundary = true;
            for(auto const & v : cell->face(f)->vertex_indices())
            {
              if(std::abs(center.distance(cell->face(f)->vertex(v)) - R1) > 1e-12 * R1)
              {
                face_at_outer_boundary = false;
                break;
              }
            }

            if(face_at_outer_boundary)
              cell->face(f)->set_boundary_id(1);
          }
        }
      }

      tria.refine_global(global_refinements);
    };

    GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
                                                            {} /* no local refinements */);

    // mappings
    GridUtilities::create_mapping_with_multigrid(mapping,
                                                 multigrid_mappings,
                                                 this->param.grid.element_type,
                                                 this->param.mapping_degree,
                                                 this->param.mapping_degree_coarse_grids,
                                                 this->param.involves_h_multigrid());
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
    this->field_functions->gravitational_force.reset(new RightHandSide<dim>());
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
    pp_data.output_data.filename           = this->output_parameters.filename;
    pp_data.output_data.write_processor_id = true;
    pp_data.output_data.degree             = this->param.degree_u;
    pp_data.output_data.write_higher_order = (dim == 2);

    std::shared_ptr<IncNS::PostProcessorBase<dim, Number>> pp;
    pp.reset(new IncNS::PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

template<int dim, typename Number>
class Scalar : public ScalarBase<dim, Number>
{
public:
  Scalar(std::string parameter_file, MPI_Comm const & comm)
    : ScalarBase<dim, Number>(parameter_file, comm)
  {
  }

private:
  void
  set_parameters() final
  {
    using namespace ConvDiff;

    // MATHEMATICAL MODEL
    this->param.problem_type                = ProblemType::Unsteady;
    this->param.equation_type               = EquationType::ConvectionDiffusion;
    this->param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
    this->param.analytical_velocity_field   = false;
    this->param.right_hand_side             = false;

    // PHYSICAL QUANTITIES
    this->param.start_time  = start_time;
    this->param.end_time    = end_time;
    this->param.diffusivity = thermal_diffusivity;

    // TEMPORAL DISCRETIZATION
    this->param.temporal_discretization       = TemporalDiscretization::BDF;
    this->param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Implicit;
    this->param.adaptive_time_stepping        = adaptive_time_stepping;
    this->param.order_time_integrator         = 2;
    this->param.start_with_low_order          = true;
    this->param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    this->param.time_step_size                = 1.0e-2;
    this->param.cfl                           = CFL;
    this->param.max_velocity                  = U;
    this->param.exponent_fe_degree_convection = 1.5;
    this->param.time_step_size                = characteristic_time / 1.e0;
    this->param.time_step_size_max            = characteristic_time / 1.e0;

    // output of solver information
    this->param.solver_info_data.interval_time = output_interval_time;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type     = TriangulationType::Distributed;
    this->param.mapping_degree              = this->param.degree;
    this->param.mapping_degree_coarse_grids = this->param.mapping_degree;

    // convective term
    this->param.numerical_flux_convective_operator =
      NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

    // viscous term
    this->param.IP_factor = 1.0;

    // NUMERICAL PARAMETERS
    this->param.implement_block_diagonal_preconditioner_matrix_free = false;
    this->param.use_cell_based_face_loops                           = false;

    // SOLVER
    this->param.solver                = ConvDiff::Solver::GMRES; // CG;
    this->param.solver_data           = SolverData(1e4, 1.e-30, reltol, 100);
    this->param.preconditioner        = Preconditioner::InverseMassMatrix;
    this->param.multigrid_data.type   = MultigridType::cphMG;
    this->param.mg_operator_type      = MultigridOperatorType::ReactionDiffusion;
    this->param.update_preconditioner = false;

    // NUMERICAL PARAMETERS
    this->param.use_combined_operator = true;
    this->param.use_overintegration   = true;
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    this->boundary_descriptor->dirichlet_bc.insert(
      pair(0, new TemperatureBC<dim>(T1, T0 - T1, H, characteristic_time)));
    this->boundary_descriptor->dirichlet_bc.insert(
      pair(1, new dealii::Functions::ConstantFunction<dim>(T1)));
  }


  void
  set_field_functions() final
  {
    this->field_functions->initial_solution.reset(new dealii::Functions::ConstantFunction<dim>(T1));
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->velocity.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    ConvDiff::PostProcessorData<dim> pp_data;
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = start_time;
    pp_data.output_data.time_control_data.trigger_interval = output_interval_time;
    pp_data.output_data.directory          = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = this->output_parameters.filename;
    pp_data.output_data.degree             = this->param.degree;
    pp_data.output_data.write_higher_order = (dim == 2);
    pp_data.output_data.write_surface_mesh = true;

    std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>> pp;
    pp.reset(new ConvDiff::PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
    this->fluid = std::make_shared<Fluid<dim, Number>>(input_file, comm);

    // create one (or even more) scalar fields
    this->scalars.resize(1);
    this->scalars[0] = std::make_shared<Scalar<dim, Number>>(input_file, comm);
  }
};

} // namespace FTI

} // namespace ExaDG

#include <exadg/incompressible_flow_with_transport/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_MANTLE_CONVECTION_H_ */
