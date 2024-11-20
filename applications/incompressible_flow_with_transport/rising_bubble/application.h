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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_RISING_BUBBLE_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_RISING_BUBBLE_H_

namespace ExaDG
{
namespace FTI
{
template<int dim>
class InitialTemperature : public dealii::Function<dim>
{
public:
  InitialTemperature(double const T_ref, double const delta_T, double const length)
    : dealii::Function<dim>(1, 0.0), T_ref(T_ref), delta_T(delta_T), L(length)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const /*component = 0*/) const final
  {
    double radius = 0.0;
    if(dim == 2)
    {
      dealii::Point<dim> center = dealii::Point<dim>(0.5 * L, 0.35 * L);
      radius                    = p.distance(center);
    }
    else if(dim == 3)
    {
      dealii::Point<dim> center = dealii::Point<dim>(0.5 * L, 0.35 * L, 0.5 * L);
      radius                    = p.distance(center);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    double const r_b = 0.25 * L;
    double const factor =
      radius < r_b ? delta_T / 2.0 * (1 + std::cos(dealii::numbers::PI * radius / r_b)) : 0.0;

    return (T_ref + factor);
  }

private:
  double const T_ref, delta_T, L;
};

double const L = 1000.0;

double const g       = 9.81;
double const T_ref   = 303.15;
double const beta    = 1.0 / T_ref;
double const delta_T = 0.5;

double const kinematic_viscosity = 0.0;
double const thermal_diffusivity = kinematic_viscosity;

double const start_time = 0.0;
double const end_time   = 1000.0;

double const CFL                    = 0.3;
double const max_velocity           = 1.0;
bool const   adaptive_time_stepping = true;

// output
double const output_interval_time = (end_time - start_time) / 100.0;

// solver tolerances
double const ABS_TOL = 1.e-12; // 1.e-10 was necessary for dual splitting scheme and refine level 8
double const REL_TOL = 1.e-6;

double const ABS_TOL_LINEAR = 1.e-12;
double const REL_TOL_LINEAR = 1.e-2;

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
    this->param.order_time_integrator           = 2;
    this->param.start_with_low_order            = true;
    this->param.adaptive_time_stepping          = adaptive_time_stepping;
    this->param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    this->param.max_velocity                    = max_velocity;
    this->param.cfl_exponent_fe_degree_velocity = 1.5;
    this->param.cfl                             = CFL;
    this->param.time_step_size                  = 1.0e-1;

    // output of solver information
    this->param.solver_info_data.interval_time = (end_time - start_time) / 10.;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type     = TriangulationType::Distributed;
    this->param.mapping_degree              = 1;
    this->param.mapping_degree_coarse_grids = this->param.mapping_degree;
    this->param.degree_p                    = DegreePressure::MixedOrder;

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

    if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
    {
      this->param.solver_momentum         = SolverMomentum::CG;
      this->param.solver_data_momentum    = SolverData(1000, ABS_TOL, REL_TOL);
      this->param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
    }


    // PRESSURE-CORRECTION SCHEME

    // formulation
    this->param.order_pressure_extrapolation = 1;
    this->param.rotational_formulation       = true;

    // momentum step
    if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
    {
      // Newton solver
      this->param.newton_solver_data_momentum = Newton::SolverData(100, ABS_TOL, REL_TOL);

      // linear solver
      this->param.solver_momentum = SolverMomentum::GMRES;
      if(this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
        this->param.solver_data_momentum = SolverData(1e4, ABS_TOL_LINEAR, REL_TOL_LINEAR, 100);
      else
        this->param.solver_data_momentum = SolverData(1e4, ABS_TOL, REL_TOL, 100);

      this->param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
    }


    // COUPLED NAVIER-STOKES SOLVER

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
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) final
  {
    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> &                        tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & periodic_face_pairs,
          unsigned int const                                       global_refinements,
          std::vector<unsigned int> const &                        vector_local_refinements) {
        (void)periodic_face_pairs;
        (void)vector_local_refinements;

        if(dim == 2)
        {
          double const left = 0.0, right = L;
          dealii::GridGenerator::hyper_cube(tria, left, right);
        }
        else if(dim == 3)
        {
          std::vector<unsigned int> repetitions({1, 1, 1});
          dealii::Point<dim>        point1(0.0, 0.0, 0.0), point2(L, 1.5 * L, L);
          dealii::GridGenerator::subdivided_hyper_rectangle(tria, repetitions, point1, point2);
        }
        else
        {
          AssertThrow(false, dealii::ExcMessage("Not implemented."));
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

    // fill boundary descriptor pressure
    this->boundary_descriptor->pressure->neumann_bc.insert(0);
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
    pp_data.output_data.filename           = this->output_parameters.filename;
    pp_data.output_data.degree             = this->param.degree_u;
    pp_data.output_data.write_higher_order = true;

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
    this->param.equation_type               = EquationType::Convection;
    this->param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
    this->param.analytical_velocity_field   = false;
    this->param.right_hand_side             = false;

    // PHYSICAL QUANTITIES
    this->param.start_time  = start_time;
    this->param.end_time    = end_time;
    this->param.diffusivity = thermal_diffusivity;

    // TEMPORAL DISCRETIZATION
    this->param.temporal_discretization       = TemporalDiscretization::BDF;
    this->param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Explicit;
    this->param.adaptive_time_stepping        = adaptive_time_stepping;
    this->param.order_time_integrator         = 2;
    this->param.start_with_low_order          = true;
    this->param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    this->param.time_step_size                = 1.0e-2;
    this->param.cfl                           = CFL;
    this->param.max_velocity                  = max_velocity;
    this->param.exponent_fe_degree_convection = 1.5;
    this->param.exponent_fe_degree_diffusion  = 3.0;
    this->param.diffusion_number              = 0.01;

    // output of solver information
    this->param.solver_info_data.interval_time = (end_time - start_time) / 10.;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type     = TriangulationType::Distributed;
    this->param.mapping_degree              = 1;
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
    this->param.solver                    = ConvDiff::Solver::CG;
    this->param.solver_data               = SolverData(1e3, ABS_TOL, REL_TOL, 100);
    this->param.preconditioner            = Preconditioner::InverseMassMatrix;
    this->param.multigrid_data.type       = MultigridType::phMG;
    this->param.multigrid_data.p_sequence = PSequenceType::Bisect;
    this->param.mg_operator_type          = MultigridOperatorType::ReactionDiffusion;
    this->param.update_preconditioner     = false;


    // output of solver information
    this->param.solver_info_data.interval_time = (end_time - start_time) / 10.;

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
      pair(0, new dealii::Functions::ConstantFunction<dim>(T_ref)));
  }


  void
  set_field_functions() final
  {
    this->field_functions->initial_solution.reset(new InitialTemperature<dim>(T_ref, delta_T, L));
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
    pp_data.output_data.write_higher_order = true;

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

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_RISING_BUBBLE_H_ */
