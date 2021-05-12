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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_SHEAR_LAYER_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_SHEAR_LAYER_H_

/*
 * For a description of the test case, see
 *
 * Brown and Minion (J. Comput. Phys. 122 (1995), 165-183)
 */

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim>
class InitialSolutionVelocity : public Function<dim>
{
public:
  InitialSolutionVelocity(double const rho, double const delta)
    : Function<dim>(dim, 0.0), rho(rho), delta(delta)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const component = 0) const
  {
    double result = 0.0;
    if(component == 0)
      result = std::tanh(rho * (0.25 - std::abs(0.5 - p[1])));
    else if(component == 1)
      result = delta * std::sin(2.0 * numbers::PI * p[0]);

    return result;
  }

private:
  double const rho, delta;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  typedef typename ApplicationBase<dim, Number>::PeriodicFaces PeriodicFaces;

  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  bool const   inviscid  = true;
  double const viscosity = inviscid ? 0.0 : 1.0e-4; // Re = 10^4
  double const rho       = 30.0;
  double const delta     = 0.05;

  double const start_time = 0.0;
  double const end_time   = 4.0;

  void
  set_input_parameters(InputParameters & param) final
  {
    // MATHEMATICAL MODEL
    param.problem_type = ProblemType::Unsteady;
    if(inviscid)
      param.equation_type = EquationType::Euler;
    else
      param.equation_type = EquationType::NavierStokes;
    param.formulation_viscous_term    = FormulationViscousTerm::LaplaceFormulation;
    param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
    param.right_hand_side             = false;

    // PHYSICAL QUANTITIES
    param.start_time = start_time;
    param.end_time   = end_time;
    param.viscosity  = viscosity;


    // TEMPORAL DISCRETIZATION
    param.solver_type                            = SolverType::Unsteady;
    param.temporal_discretization                = TemporalDiscretization::BDFDualSplittingScheme;
    param.treatment_of_convective_term           = TreatmentOfConvectiveTerm::Explicit;
    param.calculation_of_time_step_size          = TimeStepCalculation::CFL;
    param.adaptive_time_stepping                 = true;
    param.adaptive_time_stepping_limiting_factor = 3.0;
    param.max_velocity                           = 1.5;
    param.cfl                                    = 0.25;
    param.cfl_exponent_fe_degree_velocity        = 1.5;
    param.time_step_size                         = 1.0e-4;
    param.order_time_integrator                  = 2;
    param.start_with_low_order                   = true;

    // output of solver information
    param.solver_info_data.interval_time = (param.end_time - param.start_time) / 40;


    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_p           = DegreePressure::MixedOrder;
    param.mapping            = MappingType::Isoparametric;

    // convective term
    if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      param.upwind_factor = 0.5;

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // velocity-pressure coupling terms
    param.gradp_formulation = FormulationPressureGradientTerm::Weak;
    param.divu_formulation  = FormulationVelocityDivergenceTerm::Weak;

    // penalty terms
    param.use_divergence_penalty                     = true;
    param.divergence_penalty_factor                  = 1.0e0;
    param.use_continuity_penalty                     = true;
    param.continuity_penalty_factor                  = param.divergence_penalty_factor;
    param.continuity_penalty_components              = ContinuityPenaltyComponents::Normal;
    param.continuity_penalty_use_boundary_data       = true;
    param.apply_penalty_terms_in_postprocessing_step = true;

    // PROJECTION METHODS

    // formulation
    param.store_previous_boundary_values = false;

    // pressure Poisson equation
    param.solver_pressure_poisson              = SolverPressurePoisson::CG;
    param.solver_data_pressure_poisson         = SolverData(1000, 1.e-12, 1.e-6, 100);
    param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
    param.multigrid_data_pressure_poisson.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev;
    param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::PointJacobi;

    // projection step
    param.solver_projection         = SolverProjection::CG;
    param.solver_data_projection    = SolverData(1000, 1.e-12, 1.e-6);
    param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // viscous step
    param.solver_viscous         = SolverViscous::CG;
    param.solver_data_viscous    = SolverData(1000, 1.e-12, 1.e-6);
    param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;
  }

  void
  create_grid(std::shared_ptr<Triangulation<dim>> triangulation,
              PeriodicFaces &                     periodic_faces,
              unsigned int const                  n_refine_space,
              std::shared_ptr<Mapping<dim>> &     mapping,
              unsigned int const                  mapping_degree) final
  {
    double const left = 0.0, right = 1.0;
    GridGenerator::hyper_cube(*triangulation, left, right);

    // use periodic boundary conditions
    // x-direction
    triangulation->begin()->face(0)->set_all_boundary_ids(0);
    triangulation->begin()->face(1)->set_all_boundary_ids(1);
    // y-direction
    triangulation->begin()->face(2)->set_all_boundary_ids(2);
    triangulation->begin()->face(3)->set_all_boundary_ids(3);

    auto tria = dynamic_cast<Triangulation<dim> *>(&*triangulation);
    GridTools::collect_periodic_faces(*tria, 0, 1, 0, periodic_faces);
    GridTools::collect_periodic_faces(*tria, 2, 3, 1, periodic_faces);
    triangulation->add_periodicity(periodic_faces);

    triangulation->refine_global(n_refine_space);

    mapping.reset(new MappingQGeneric<dim>(mapping_degree));
  }

  void
  set_boundary_conditions(
    std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure) final
  {
    (void)boundary_descriptor_velocity;
    (void)boundary_descriptor_pressure;
  }


  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions) final
  {
    field_functions->initial_solution_velocity.reset(new InitialSolutionVelocity<dim>(rho, delta));
    field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm) final
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output     = this->write_output;
    pp_data.output_data.directory        = this->output_directory + "vtu/";
    pp_data.output_data.filename         = this->output_name;
    pp_data.output_data.start_time       = start_time;
    pp_data.output_data.interval_time    = (end_time - start_time) / 40;
    pp_data.output_data.write_divergence = true;
    pp_data.output_data.write_vorticity  = true;
    pp_data.output_data.degree           = degree;

    // kinetic energy
    pp_data.kinetic_energy_data.calculate                  = true;
    pp_data.kinetic_energy_data.evaluate_individual_terms  = false;
    pp_data.kinetic_energy_data.calculate_every_time_steps = 1;
    pp_data.kinetic_energy_data.viscosity                  = viscosity;
    pp_data.kinetic_energy_data.directory                  = this->output_directory;
    pp_data.kinetic_energy_data.filename                   = this->output_name;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace IncNS

template<int dim, typename Number>
std::shared_ptr<IncNS::ApplicationBase<dim, Number>>
get_application(std::string input_file)
{
  return std::make_shared<IncNS::Application<dim, Number>>(input_file);
}

} // namespace ExaDG


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_SHEAR_LAYER_H_ */
