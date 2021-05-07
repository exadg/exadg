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
using namespace dealii;

template<int dim>
class InitialTemperature : public Function<dim>
{
public:
  InitialTemperature(double const T_ref, double const delta_T, double const length)
    : Function<dim>(1, 0.0), T_ref(T_ref), delta_T(delta_T), L(length)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const /*component = 0*/) const
  {
    double radius = 0.0;
    if(dim == 2)
    {
      Point<dim> center = Point<dim>(0.5 * L, 0.35 * L);
      radius            = p.distance(center);
    }
    else if(dim == 3)
    {
      Point<dim> center = Point<dim>(0.5 * L, 0.35 * L, 0.5 * L);
      radius            = p.distance(center);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    double const r_b = 0.25 * L;
    double const factor =
      radius < r_b ? delta_T / 2.0 * (1 + std::cos(numbers::PI * radius / r_b)) : 0.0;

    return (T_ref + factor);
  }

private:
  double const T_ref, delta_T, L;
};

template<int dim, typename Number>
class Application : public FTI::ApplicationBase<dim, Number>
{
public:
  typedef typename ApplicationBase<dim, Number>::PeriodicFaces PeriodicFaces;

  Application(std::string input_file) : FTI::ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

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
  double const ABS_TOL =
    1.e-12; // 1.e-10 was necessary for dual splitting scheme and refine level 8
  double const REL_TOL = 1.e-6;

  double const ABS_TOL_LINEAR = 1.e-12;
  double const REL_TOL_LINEAR = 1.e-2;

  void
  set_input_parameters(IncNS::InputParameters & param)
  {
    using namespace IncNS;

    // MATHEMATICAL MODEL
    param.problem_type                = ProblemType::Unsteady;
    param.equation_type               = EquationType::NavierStokes;
    param.formulation_viscous_term    = FormulationViscousTerm::LaplaceFormulation;
    param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
    param.right_hand_side             = true;
    param.boussinesq_term             = true;


    // PHYSICAL QUANTITIES
    param.start_time                    = start_time;
    param.end_time                      = end_time;
    param.viscosity                     = kinematic_viscosity;
    param.thermal_expansion_coefficient = beta;
    param.reference_temperature         = T_ref;


    // TEMPORAL DISCRETIZATION
    param.solver_type                     = SolverType::Unsteady;
    param.temporal_discretization         = TemporalDiscretization::BDFDualSplittingScheme;
    param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    param.order_time_integrator           = 2;
    param.start_with_low_order            = true;
    param.adaptive_time_stepping          = adaptive_time_stepping;
    param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    param.max_velocity                    = max_velocity;
    param.cfl_exponent_fe_degree_velocity = 1.5;
    param.cfl                             = CFL;
    param.time_step_size                  = 1.0e-1;

    // output of solver information
    param.solver_info_data.interval_time = (end_time - start_time) / 10.;

    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_p           = DegreePressure::MixedOrder;
    param.mapping            = MappingType::Affine;

    // convective term
    if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // div-div and continuity penalty
    param.use_divergence_penalty                     = true;
    param.divergence_penalty_factor                  = 1.0;
    param.use_continuity_penalty                     = true;
    param.continuity_penalty_factor                  = param.divergence_penalty_factor;
    param.continuity_penalty_components              = ContinuityPenaltyComponents::Normal;
    param.continuity_penalty_use_boundary_data       = true;
    param.apply_penalty_terms_in_postprocessing_step = true;
    param.type_penalty_parameter                     = TypePenaltyParameter::ConvectiveTerm;

    // NUMERICAL PARAMETERS
    param.implement_block_diagonal_preconditioner_matrix_free = false;
    param.use_cell_based_face_loops                           = false;

    // PROJECTION METHODS

    // pressure Poisson equation
    param.solver_pressure_poisson              = SolverPressurePoisson::CG;
    param.solver_data_pressure_poisson         = SolverData(1000, ABS_TOL, REL_TOL, 100);
    param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
    param.multigrid_data_pressure_poisson.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev;
    param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::PointJacobi;
    param.multigrid_data_pressure_poisson.smoother_data.smoother = MultigridSmoother::Chebyshev;
    param.multigrid_data_pressure_poisson.smoother_data.preconditioner =
      PreconditionerSmoother::PointJacobi;

    // projection step
    param.solver_projection         = SolverProjection::CG;
    param.solver_data_projection    = SolverData(1000, ABS_TOL, REL_TOL);
    param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulation
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // viscous step
    param.solver_viscous         = SolverViscous::CG;
    param.solver_data_viscous    = SolverData(1000, ABS_TOL, REL_TOL);
    param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;


    // PRESSURE-CORRECTION SCHEME

    // formulation
    param.order_pressure_extrapolation = 1;
    param.rotational_formulation       = true;

    // momentum step

    // Newton solver
    param.newton_solver_data_momentum = Newton::SolverData(100, ABS_TOL, REL_TOL);

    // linear solver
    param.solver_momentum = SolverMomentum::GMRES;
    if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      param.solver_data_momentum = SolverData(1e4, ABS_TOL_LINEAR, REL_TOL_LINEAR, 100);
    else
      param.solver_data_momentum = SolverData(1e4, ABS_TOL, REL_TOL, 100);

    param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;


    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    param.newton_solver_data_coupled = Newton::SolverData(100, ABS_TOL, REL_TOL);

    // linear solver
    param.solver_coupled = SolverCoupled::GMRES;
    if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      param.solver_data_coupled = SolverData(1e3, ABS_TOL_LINEAR, REL_TOL_LINEAR, 100);
    else
      param.solver_data_coupled = SolverData(1e3, ABS_TOL, REL_TOL, 100);

    // preconditioner linear solver
    param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    param.update_preconditioner_coupled = false;

    // preconditioner velocity/momentum block
    param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block =
      SchurComplementPreconditioner::PressureConvectionDiffusion;
  }

  void
  set_input_parameters_scalar(ConvDiff::InputParameters & param, unsigned int const scalar_index)
  {
    (void)scalar_index;

    using namespace ConvDiff;

    // MATHEMATICAL MODEL
    param.problem_type                = ProblemType::Unsteady;
    param.equation_type               = EquationType::Convection;
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

    // triangulation
    param.triangulation_type = TriangulationType::Distributed;

    // mapping
    param.mapping = MappingType::Affine;

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
  }

  void
  create_grid(std::shared_ptr<Triangulation<dim>> triangulation,
              PeriodicFaces &                     periodic_faces,
              unsigned int const                  n_refine_space,
              std::shared_ptr<Mapping<dim>> &     mapping,
              unsigned int const                  mapping_degree)
  {
    (void)periodic_faces;

    if(dim == 2)
    {
      double const left = 0.0, right = L;
      GridGenerator::hyper_cube(*triangulation, left, right);
    }
    else if(dim == 3)
    {
      std::vector<unsigned int> repetitions({1, 1, 1});
      Point<dim>                point1(0.0, 0.0, 0.0), point2(L, 1.5 * L, L);
      GridGenerator::subdivided_hyper_rectangle(*triangulation, repetitions, point1, point2);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    triangulation->refine_global(n_refine_space);

    mapping.reset(new MappingQGeneric<dim>(mapping_degree));
  }

  void
  set_boundary_conditions(
    std::shared_ptr<IncNS::BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
    std::shared_ptr<IncNS::BoundaryDescriptorP<dim>> boundary_descriptor_pressure)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // fill boundary descriptor velocity
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));

    // fill boundary descriptor pressure
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));
  }

  void
  set_field_functions(std::shared_ptr<IncNS::FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
    std::vector<double> gravity = std::vector<double>(dim, 0.0);
    gravity[1]                  = -g;
    field_functions->gravitational_force.reset(new Functions::ConstantFunction<dim>(gravity));
  }

  std::shared_ptr<IncNS::PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    IncNS::PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output       = this->write_output;
    pp_data.output_data.directory          = this->output_directory + "vtu/";
    pp_data.output_data.filename           = this->output_name + "_fluid";
    pp_data.output_data.start_time         = start_time;
    pp_data.output_data.interval_time      = output_interval_time;
    pp_data.output_data.degree             = degree;
    pp_data.output_data.write_higher_order = true;

    std::shared_ptr<IncNS::PostProcessorBase<dim, Number>> pp;
    pp.reset(new IncNS::PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }

  void
  set_boundary_conditions_scalar(
    std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> boundary_descriptor,
    unsigned int                                       scalar_index = 0)
  {
    (void)scalar_index; // only one scalar quantity considered

    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    boundary_descriptor->dirichlet_bc.insert(pair(0, new Functions::ConstantFunction<dim>(T_ref)));
  }

  void
  set_field_functions_scalar(std::shared_ptr<ConvDiff::FieldFunctions<dim>> field_functions,
                             unsigned int                                   scalar_index = 0)
  {
    (void)scalar_index; // only one scalar quantity considered

    field_functions->initial_solution.reset(new InitialTemperature<dim>(T_ref, delta_T, L));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->velocity.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>>
  construct_postprocessor_scalar(unsigned int const degree,
                                 MPI_Comm const &   mpi_comm,
                                 unsigned int const scalar_index)
  {
    ConvDiff::PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output = this->write_output;
    pp_data.output_data.directory    = this->output_directory + "vtu/";
    pp_data.output_data.filename   = this->output_name + "_scalar_" + std::to_string(scalar_index);
    pp_data.output_data.start_time = start_time;
    pp_data.output_data.interval_time      = output_interval_time;
    pp_data.output_data.degree             = degree;
    pp_data.output_data.write_higher_order = true;

    std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>> pp;
    pp.reset(new ConvDiff::PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace FTI

template<int dim, typename Number>
std::shared_ptr<FTI::ApplicationBase<dim, Number>>
get_application(std::string input_file)
{
  return std::make_shared<FTI::Application<dim, Number>>(input_file);
}

} // namespace ExaDG


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_RISING_BUBBLE_H_ */
