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
using namespace dealii;

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
  value(Point<dim> const & p, unsigned int const /*component = 0*/) const
  {
    double       t           = this->get_time();
    double const time_factor = std::max(0.0, 1.0 - t / characteristic_time);

    double perturbation = time_factor;
    perturbation *= 0.25 * delta_T * std::pow(std::sin(numbers::PI * p[0] / (length / 4.)), 2.0);
    if(dim == 3)
      perturbation *= std::pow(std::sin(numbers::PI * p[2] / (length / 4.)), 2.0);

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

  void
  set_input_parameters(IncNS::InputParameters & param) final
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
    param.adaptive_time_stepping          = adaptive_time_stepping;
    param.order_time_integrator           = 2;
    param.start_with_low_order            = true;
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
    param.solver_data_pressure_poisson         = SolverData(1000, 1.e-12, 1.e-6, 100);
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
    param.solver_data_projection    = SolverData(1000, 1.e-12, 1.e-6);
    param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulation
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // viscous step
    param.solver_viscous         = SolverViscous::CG;
    param.solver_data_viscous    = SolverData(1000, 1.e-12, 1.e-6);
    param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;


    // PRESSURE-CORRECTION SCHEME

    // formulation
    param.order_pressure_extrapolation = 1;
    param.rotational_formulation       = true;

    // momentum step

    // Newton solver
    param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-20, 1.e-6);

    // linear solver
    param.solver_momentum         = SolverMomentum::GMRES;
    param.solver_data_momentum    = SolverData(1e4, 1.e-12, 1.e-6, 100);
    param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;


    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-12, 1.e-6);

    // linear solver
    param.solver_coupled      = SolverCoupled::GMRES;
    param.solver_data_coupled = SolverData(1e3, 1.e-12, 1.e-6, 100);

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
  set_input_parameters_scalar(ConvDiff::InputParameters & param,
                              unsigned int const          scalar_index) final
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
    param.use_overintegration   = true;
  }

  void
  create_grid(std::shared_ptr<Triangulation<dim>> triangulation,
              PeriodicFaces &                     periodic_faces,
              unsigned int const                  n_refine_space,
              std::shared_ptr<Mapping<dim>> &     mapping,
              unsigned int const                  mapping_degree) final
  {
    if(dim == 2)
    {
      std::vector<unsigned int> repetitions({8, 1});
      Point<dim>                point1(-L / 2., 0.0), point2(L / 2., H);
      GridGenerator::subdivided_hyper_rectangle(*triangulation, repetitions, point1, point2);
    }
    else if(dim == 3)
    {
      std::vector<unsigned int> repetitions({8, 1, 8});
      Point<dim>                point1(-L / 2., 0.0, -L / 2.), point2(L / 2., H, L / 2.);
      GridGenerator::subdivided_hyper_rectangle(*triangulation, repetitions, point1, point2);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    // set boundary IDs: 0 by default, set left boundary to 1
    for(auto cell : triangulation->active_cell_iterators())
    {
      for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
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

    auto tria = dynamic_cast<Triangulation<dim> *>(&*triangulation);
    GridTools::collect_periodic_faces(*tria, 10, 11, 0, periodic_faces);
    if(dim == 3)
      GridTools::collect_periodic_faces(*tria, 12, 13, 2, periodic_faces);

    triangulation->add_periodicity(periodic_faces);

    triangulation->refine_global(n_refine_space);

    mapping.reset(new MappingQGeneric<dim>(mapping_degree));
  }

  void
  set_boundary_conditions(
    std::shared_ptr<IncNS::BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
    std::shared_ptr<IncNS::BoundaryDescriptorP<dim>> boundary_descriptor_pressure) final
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
  set_field_functions(std::shared_ptr<IncNS::FieldFunctions<dim>> field_functions) final
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
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm) final
  {
    IncNS::PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output       = this->write_output;
    pp_data.output_data.directory          = this->output_directory + "vtu/";
    pp_data.output_data.filename           = this->output_name + "_fluid";
    pp_data.output_data.start_time         = start_time;
    pp_data.output_data.interval_time      = output_interval_time;
    pp_data.output_data.write_processor_id = true;
    pp_data.output_data.write_surface_mesh = true;
    pp_data.output_data.degree             = degree;
    pp_data.output_data.write_higher_order = false;

    std::shared_ptr<IncNS::PostProcessorBase<dim, Number>> pp;
    pp.reset(new IncNS::PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }

  void
  set_boundary_conditions_scalar(
    std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> boundary_descriptor,
    unsigned int                                       scalar_index = 0) final
  {
    (void)scalar_index; // only one scalar quantity considered

    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    boundary_descriptor->dirichlet_bc.insert(pair(0, new Functions::ConstantFunction<dim>(T_ref)));
    boundary_descriptor->dirichlet_bc.insert(
      pair(1, new TemperatureBC<dim>(T_ref, delta_T, L, characteristic_time)));
  }

  void
  set_field_functions_scalar(std::shared_ptr<ConvDiff::FieldFunctions<dim>> field_functions,
                             unsigned int                                   scalar_index = 0) final
  {
    (void)scalar_index; // only one scalar quantity considered

    field_functions->initial_solution.reset(new Functions::ConstantFunction<dim>(T_ref));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->velocity.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>>
  construct_postprocessor_scalar(unsigned int const degree,
                                 MPI_Comm const &   mpi_comm,
                                 unsigned int const scalar_index) final
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

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_RAYLEIGH_BENARD_H_ */
