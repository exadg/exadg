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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_COUETTE_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_COUETTE_H_

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim>
class AnalyticalSolutionVelocity : public Function<dim>
{
public:
  AnalyticalSolutionVelocity(double const H, double const max_velocity)
    : Function<dim>(dim, 0.0), H(H), max_velocity(max_velocity)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const component = 0) const
  {
    double result = 0.0;

    if(component == 0)
      result = max_velocity * (p[1] + H / 2.) / H;

    return result;
  }

private:
  double const H, max_velocity;
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

  double const H            = 1.0;
  double const L            = 4.0;
  double const max_velocity = 1.0;

  double const start_time = 0.0;
  double const end_time   = 10.0;

  void
  set_input_parameters(InputParameters & param) final
  {
    // MATHEMATICAL MODEL
    param.problem_type             = ProblemType::Steady; // Unsteady;
    param.equation_type            = EquationType::NavierStokes;
    param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
    param.right_hand_side          = false;


    // PHYSICAL QUANTITIES
    param.start_time = start_time;
    param.end_time   = end_time;
    param.viscosity  = 1.0;


    // TEMPORAL DISCRETIZATION
    param.solver_type = SolverType::Steady; // Unsteady;
    param.temporal_discretization =
      TemporalDiscretization::BDFCoupledSolution; // BDFDualSplittingScheme;
    param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Implicit; // Explicit;
    param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    param.max_velocity                  = max_velocity;
    param.cfl                           = 1.0e-1;
    param.time_step_size                = 1.0e-1;
    param.order_time_integrator         = 2;    // 1; // 2; // 3;
    param.start_with_low_order          = true; // true; // false;

    // output of solver information
    param.solver_info_data.interval_time = (param.end_time - param.start_time) / 10;

    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_p           = DegreePressure::MixedOrder;
    param.mapping            = MappingType::Isoparametric;

    // convective term

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // divergence and continuity penalty terms
    param.apply_penalty_terms_in_postprocessing_step = false;

    // PROJECTION METHODS

    // pressure Poisson equation
    param.solver_pressure_poisson         = SolverPressurePoisson::CG;
    param.solver_data_pressure_poisson    = SolverData(1000, 1.e-20, 1.e-6);
    param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

    // projection step
    param.solver_projection         = SolverProjection::CG;
    param.solver_data_projection    = SolverData(1000, 1.e-20, 1.e-12);
    param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // viscous step
    param.solver_viscous         = SolverViscous::CG;
    param.solver_data_viscous    = SolverData(1000, 1.e-20, 1.e-6);
    param.preconditioner_viscous = PreconditionerViscous::Multigrid;

    // PRESSURE-CORRECTION SCHEME

    // momentum step

    // Newton solver
    param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-20, 1.e-6);

    // linear solver
    param.solver_momentum                = SolverMomentum::GMRES;
    param.solver_data_momentum           = SolverData(1e4, 1.e-20, 1.e-4, 100);
    param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
    param.update_preconditioner_momentum = false;

    // formulation
    param.order_pressure_extrapolation = 1;
    param.rotational_formulation       = true;

    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-14, 1.e-14);

    // linear solver
    param.solver_coupled      = SolverCoupled::FGMRES;
    param.solver_data_coupled = SolverData(1e4, 1.e-14, 1.e-2, 100);

    // preconditioning linear solver
    param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    param.update_preconditioner_coupled = true;

    // preconditioner velocity/momentum block
    param.preconditioner_velocity_block          = MomentumPreconditioner::Multigrid;
    param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;
    param.multigrid_data_velocity_block.type     = MultigridType::hMG;
    param.multigrid_data_velocity_block.smoother_data.smoother = MultigridSmoother::Jacobi;
    param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::BlockJacobi;
    param.multigrid_data_velocity_block.smoother_data.iterations        = 5;
    param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
    param.multigrid_data_velocity_block.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block =
      SchurComplementPreconditioner::PressureConvectionDiffusion;
  }

  void
  create_grid(std::shared_ptr<Triangulation<dim>> triangulation,
              PeriodicFaces &                     periodic_faces,
              unsigned int const                  n_refine_space,
              std::shared_ptr<Mapping<dim>> &     mapping,
              unsigned int const                  mapping_degree) final
  {
    (void)periodic_faces;

    std::vector<unsigned int> repetitions(dim, 1);
    repetitions[0] = 2;
    Point<dim> point1, point2;
    point1[0] = 0.0;
    point1[1] = -H / 2;
    if(dim == 3)
      point1[2] = 0.0;

    point2[0] = L;
    point2[1] = H / 2;
    if(dim == 3)
      point2[2] = H;

    GridGenerator::subdivided_hyper_rectangle(*triangulation, repetitions, point1, point2);

    // set boundary indicator
    for(auto cell : triangulation->active_cell_iterators())
    {
      for(unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell;
          ++face_number)
      {
        if((std::fabs(cell->face(face_number)->center()(0) - L) < 1e-12))
          cell->face(face_number)->set_boundary_id(1);
      }
    }

    triangulation->refine_global(n_refine_space);

    mapping.reset(new MappingQGeneric<dim>(mapping_degree));
  }

  void
  set_boundary_conditions(
    std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure) final
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // fill boundary descriptor velocity
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(0, new AnalyticalSolutionVelocity<dim>(H, max_velocity)));
    boundary_descriptor_velocity->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(dim)));

    // fill boundary descriptor pressure
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_pressure->dirichlet_bc.insert(pair(1, new Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions) final
  {
    field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
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
    pp_data.output_data.interval_time    = (end_time - start_time) / 20;
    pp_data.output_data.write_divergence = true;
    pp_data.output_data.degree           = degree;

    // calculation of velocity error
    pp_data.error_data_u.analytical_solution_available = true;
    pp_data.error_data_u.analytical_solution.reset(
      new AnalyticalSolutionVelocity<dim>(H, max_velocity));
    pp_data.error_data_u.calculate_relative_errors = false;
    pp_data.error_data_u.error_calc_start_time     = start_time;
    pp_data.error_data_u.error_calc_interval_time  = (end_time - start_time) / 20;
    pp_data.error_data_u.name                      = "velocity";

    // ... pressure error
    pp_data.error_data_p.analytical_solution_available = true;
    pp_data.error_data_p.analytical_solution.reset(new Functions::ZeroFunction<dim>(1));
    pp_data.error_data_p.calculate_relative_errors = false;
    pp_data.error_data_p.error_calc_start_time     = start_time;
    pp_data.error_data_p.error_calc_interval_time  = (end_time - start_time) / 20;
    pp_data.error_data_p.name                      = "pressure";

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


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_COUETTE_H_ */
