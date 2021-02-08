/*
 * mantle_convection.h
 *
 *  Created on: Jan 26, 2020
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_MANTLE_CONVECTION_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_MANTLE_CONVECTION_H_

namespace ExaDG
{
namespace FTI
{
using namespace dealii;

template<int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide() : Function<dim>(dim, 0.0)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component) const
  {
    double const r = p.norm();

    double g = -p[component] / r;

    return g;
  }
};

template<int dim>
class TemperatureBC : public Function<dim>
{
public:
  TemperatureBC(double const T_ref,
                double const delta_T,
                double const length,
                double const characteristic_time)
    : Function<dim>(1, 0.0),
      T_ref(T_ref),
      delta_T(delta_T),
      length(length),
      characteristic_time(characteristic_time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int /*component = 0*/) const
  {
    double       t           = this->get_time();
    double const time_factor = std::max(0.0, 1.0 - t / characteristic_time);

    double perturbation = time_factor;
    if(dim == 2)
      perturbation *= 0.25 * delta_T *
                      std::pow(std::sin(numbers::PI * (p[0] + std::sqrt(2)) / (length / 2.)) *
                                 std::sin(numbers::PI * (p[1] + std::sqrt(3)) / (length / 2.)),
                               2.0);
    else if(dim == 3)
      perturbation *= 0.25 * delta_T *
                      std::pow(std::sin(numbers::PI * (p[0] + std::sqrt(2)) / (length / 2.)) *
                                 std::sin(numbers::PI * (p[1] + std::sqrt(3)) / (length / 2.)) *
                                 std::sin(numbers::PI * (p[2] + std::sqrt(5)) / (length / 2.)),
                               2.0);
    else
      AssertThrow(false, ExcMessage("not implemented."));

    return (T_ref + delta_T + perturbation);
  }

private:
  double const T_ref, delta_T, length, characteristic_time;
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

  void
  set_input_parameters(IncNS::InputParameters & param)
  {
    using namespace IncNS;

    // MATHEMATICAL MODEL
    param.problem_type                 = ProblemType::Steady;
    param.equation_type                = EquationType::Stokes;
    param.formulation_viscous_term     = FormulationViscousTerm::LaplaceFormulation;
    param.formulation_convective_term  = FormulationConvectiveTerm::ConvectiveFormulation;
    param.right_hand_side              = true;
    param.boussinesq_term              = true;
    param.boussinesq_dynamic_part_only = true;

    // PHYSICAL QUANTITIES
    param.start_time                    = start_time;
    param.end_time                      = end_time;
    param.viscosity                     = kinematic_viscosity;
    param.thermal_expansion_coefficient = beta;
    param.reference_temperature         = T1;

    // TEMPORAL DISCRETIZATION
    param.solver_type                     = SolverType::Steady;
    param.temporal_discretization         = TemporalDiscretization::BDFCoupledSolution;
    param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    param.adaptive_time_stepping          = adaptive_time_stepping;
    param.order_time_integrator           = 2;
    param.start_with_low_order            = true;
    param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    param.max_velocity                    = U;
    param.cfl_exponent_fe_degree_velocity = 1.5;
    param.cfl                             = CFL;
    param.time_step_size                  = characteristic_time / 1.e0;
    param.time_step_size_max              = characteristic_time / 1.e0;

    // output of solver information
    param.solver_info_data.interval_time = output_interval_time;

    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_p           = DegreePressure::MixedOrder;
    param.mapping            = MappingType::Isoparametric;

    // convective term
    if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // div-div and continuity penalty
    param.use_divergence_penalty                     = false;
    param.divergence_penalty_factor                  = 1.0;
    param.use_continuity_penalty                     = false;
    param.continuity_penalty_factor                  = param.divergence_penalty_factor;
    param.continuity_penalty_components              = ContinuityPenaltyComponents::Normal;
    param.continuity_penalty_use_boundary_data       = true;
    param.apply_penalty_terms_in_postprocessing_step = false;
    param.type_penalty_parameter                     = TypePenaltyParameter::ConvectiveTerm;

    // NUMERICAL PARAMETERS
    param.implement_block_diagonal_preconditioner_matrix_free = false;
    param.use_cell_based_face_loops                           = false;

    // PROJECTION METHODS

    // pressure Poisson equation
    param.solver_pressure_poisson              = SolverPressurePoisson::CG;
    param.solver_data_pressure_poisson         = SolverData(1000, 1.e-30, reltol, 100);
    param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;

    // projection step
    param.solver_projection         = SolverProjection::CG;
    param.solver_data_projection    = SolverData(1000, 1.e-30, reltol);
    param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulation
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // viscous step
    param.solver_viscous              = SolverViscous::CG;
    param.solver_data_viscous         = SolverData(1000, 1.e-30, reltol);
    param.multigrid_data_viscous.type = MultigridType::cphMG;


    // PRESSURE-CORRECTION SCHEME

    // formulation
    param.order_pressure_extrapolation = 1;
    param.rotational_formulation       = true;

    // momentum step

    // Newton solver
    param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-30, reltol);

    // linear solver
    param.solver_momentum                  = SolverMomentum::CG;
    param.solver_data_momentum             = SolverData(1e4, 1.e-30, reltol, 100);
    param.preconditioner_momentum          = MomentumPreconditioner::Multigrid;
    param.multigrid_data_momentum.type     = MultigridType::cphMG;
    param.multigrid_operator_type_momentum = MultigridOperatorType::ReactionDiffusion;

    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-30, reltol);

    // linear solver
    param.solver_coupled         = SolverCoupled::GMRES;
    param.solver_data_coupled    = SolverData(1e3, 1.e-30, reltol, 100);
    param.use_scaling_continuity = false;

    // preconditioner linear solver
    param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    param.update_preconditioner_coupled = false;

    // preconditioner velocity/momentum block
    param.preconditioner_velocity_block          = MomentumPreconditioner::Multigrid;
    param.multigrid_data_velocity_block.type     = MultigridType::cphMG;
    param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;
    param.multigrid_data_velocity_block.smoother_data.smoother = MultigridSmoother::Chebyshev;
    param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::PointJacobi;
    param.multigrid_data_velocity_block.smoother_data.iterations = 5;

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block = SchurComplementPreconditioner::InverseMassMatrix;
  }

  void
  set_input_parameters_scalar(ConvDiff::InputParameters & param, unsigned int const scalar_index)
  {
    (void)scalar_index;

    using namespace ConvDiff;

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

    // triangulation
    param.triangulation_type = TriangulationType::Distributed;

    // mapping
    param.mapping = MappingType::Isoparametric;

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
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              PeriodicFaces &                                   periodic_faces,
              unsigned int const                                n_refine_space,
              std::shared_ptr<Mapping<dim>> &                   mapping,
              unsigned int const                                mapping_degree)
  {
    (void)periodic_faces;

    GridGenerator::hyper_shell(*triangulation, Point<dim>(), R0, R1, (dim == 3) ? 48 : 12);

    Point<dim> center = dim == 2 ? Point<dim>(0., 0.) : Point<dim>(0., 0., 0.);

    for(auto cell : triangulation->active_cell_iterators())
    {
      for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      {
        bool face_at_outer_boundary = true;
        for(unsigned int v = 0; v < GeometryInfo<dim - 1>::vertices_per_cell; ++v)
        {
          if(std::abs(center.distance(cell->face(f)->vertex(v)) - R1) > 1e-12 * R1)
            face_at_outer_boundary = false;
        }

        if(face_at_outer_boundary)
          cell->face(f)->set_boundary_id(1);
      }
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
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(1, new Functions::ZeroFunction<dim>(dim)));

    // fill boundary descriptor pressure
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_pressure->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(dim)));
  }

  void
  set_field_functions(std::shared_ptr<IncNS::FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->gravitational_force.reset(new RightHandSide<dim>());
  }

  std::shared_ptr<IncNS::PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    IncNS::PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output         = this->write_output;
    pp_data.output_data.output_folder        = this->output_directory + "vtu/";
    pp_data.output_data.output_name          = this->output_name + "_fluid";
    pp_data.output_data.output_start_time    = start_time;
    pp_data.output_data.output_interval_time = output_interval_time;
    pp_data.output_data.write_processor_id   = true;
    pp_data.output_data.degree               = degree;
    pp_data.output_data.write_higher_order   = (dim == 2);

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

    boundary_descriptor->dirichlet_bc.insert(
      pair(0, new TemperatureBC<dim>(T1, T0 - T1, H, characteristic_time)));
    boundary_descriptor->dirichlet_bc.insert(pair(1, new Functions::ConstantFunction<dim>(T1)));
  }

  void
  set_field_functions_scalar(std::shared_ptr<ConvDiff::FieldFunctions<dim>> field_functions,
                             unsigned int                                   scalar_index = 0)
  {
    (void)scalar_index; // only one scalar quantity considered

    field_functions->initial_solution.reset(new Functions::ConstantFunction<dim>(T1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->velocity.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>>
  construct_postprocessor_scalar(unsigned int const degree,
                                 MPI_Comm const &   mpi_comm,
                                 unsigned int const scalar_index)
  {
    ConvDiff::PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output  = this->write_output;
    pp_data.output_data.output_folder = this->output_directory + "vtu/";
    pp_data.output_data.output_name = this->output_name + "_scalar_" + std::to_string(scalar_index);
    pp_data.output_data.output_start_time    = start_time;
    pp_data.output_data.output_interval_time = output_interval_time;
    pp_data.output_data.degree               = degree;
    pp_data.output_data.write_higher_order   = (dim == 2);
    pp_data.output_data.write_surface_mesh   = true;

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

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_MANTLE_CONVECTION_H_ */
