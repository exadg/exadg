/*
 * kovasznay.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_KOVASZNAY_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_KOVASZNAY_H_

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

enum class InitializeSolutionWith
{
  ZeroFunction,
  AnalyticalSolution
};

template<int dim>
class AnalyticalSolutionVelocity : public Function<dim>
{
public:
  AnalyticalSolutionVelocity(double const lambda) : Function<dim>(dim, 0.0), lambda(lambda)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double const pi = numbers::PI;

    double result = 0.0;
    if(component == 0)
      result = 1.0 - std::exp(lambda * p[0]) * std::cos(2 * pi * p[1]);
    else if(component == 1)
      result = lambda / 2.0 / pi * std::exp(lambda * p[0]) * std::sin(2 * pi * p[1]);

    return result;
  }

private:
  double const lambda;
};

template<int dim>
class AnalyticalSolutionPressure : public Function<dim>
{
public:
  AnalyticalSolutionPressure(double const lambda)
    : Function<dim>(1 /*n_components*/, 0.0), lambda(lambda)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int /*component*/) const
  {
    double const result = 0.5 * (1.0 - std::exp(2.0 * lambda * p[0]));

    return result;
  }

private:
  double const lambda;
};

template<int dim>
class NeumannBoundaryVelocity : public Function<dim>
{
public:
  NeumannBoundaryVelocity(FormulationViscousTerm const & formulation_viscous, double const lambda)
    : Function<dim>(dim, 0.0), formulation_viscous(formulation_viscous), lambda(lambda)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double const pi = numbers::PI;

    double result = 0.0;
    if(formulation_viscous == FormulationViscousTerm::LaplaceFormulation)
    {
      if(component == 0)
        result = -lambda * std::exp(lambda) * std::cos(2 * pi * p[1]);
      else if(component == 1)
        result = std::pow(lambda, 2.0) / 2 / pi * std::exp(lambda) * std::sin(2 * pi * p[1]);
    }
    else if(formulation_viscous == FormulationViscousTerm::DivergenceFormulation)
    {
      if(component == 0)
        result = -2.0 * lambda * std::exp(lambda) * std::cos(2 * pi * p[1]);
      else if(component == 1)
        result =
          (std::pow(lambda, 2.0) / 2 / pi + 2.0 * pi) * std::exp(lambda) * std::sin(2 * pi * p[1]);
    }
    else
    {
      AssertThrow(formulation_viscous == FormulationViscousTerm::LaplaceFormulation ||
                    formulation_viscous == FormulationViscousTerm::DivergenceFormulation,
                  ExcMessage("Specified formulation of viscous term is not implemented!"));
    }

    return result;
  }

private:
  FormulationViscousTerm const formulation_viscous;
  double const                 lambda;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  InitializeSolutionWith const initialize_solution_with =
    InitializeSolutionWith::AnalyticalSolution;

  FormulationViscousTerm const formulation_viscous = FormulationViscousTerm::LaplaceFormulation;

  double const viscosity = 2.5e-2;
  double const lambda =
    0.5 / viscosity -
    std::pow(0.25 / std::pow(viscosity, 2.0) + 4.0 * std::pow(numbers::PI, 2.0), 0.5);

  double const start_time = 0.0;
  double const end_time   = 1.0;

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.problem_type                = ProblemType::Unsteady;
    param.equation_type               = EquationType::NavierStokes;
    param.formulation_viscous_term    = formulation_viscous;
    param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
    param.right_hand_side             = false;


    // PHYSICAL QUANTITIES
    param.start_time = start_time;
    param.end_time   = end_time;
    param.viscosity  = viscosity;


    // TEMPORAL DISCRETIZATION
    param.solver_type                   = SolverType::Unsteady;
    param.temporal_discretization       = TemporalDiscretization::BDFDualSplittingScheme;
    param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Explicit;
    param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    param.max_velocity                  = 3.6;
    param.cfl                           = 1.0e-2;
    param.time_step_size                = 1.0e-3;
    param.order_time_integrator         = 3;    // 1; // 2; // 3;
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

    // PROJECTION METHODS

    // pressure Poisson equation
    param.solver_pressure_poisson         = SolverPressurePoisson::CG;
    param.solver_data_pressure_poisson    = SolverData(1000, 1.e-20, 1.e-6, 100);
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
    param.newton_solver_data_coupled.abs_tol  = 1.e-12;
    param.newton_solver_data_coupled.rel_tol  = 1.e-6;
    param.newton_solver_data_coupled.max_iter = 1e2;

    // linear solver
    param.solver_coupled      = SolverCoupled::FGMRES;
    param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-3, 1000);

    // preconditioning linear solver
    param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    param.update_preconditioner_coupled = true;

    // preconditioner velocity/momentum block
    param.preconditioner_velocity_block = MomentumPreconditioner::Multigrid;
    param.multigrid_data_velocity_block.smoother_data.smoother =
      MultigridSmoother::Jacobi; // Jacobi; //Chebyshev; //GMRES;
    param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::BlockJacobi; // PointJacobi; //BlockJacobi;
    param.multigrid_data_velocity_block.smoother_data.iterations        = 5;
    param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
    param.multigrid_data_velocity_block.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block =
      SchurComplementPreconditioner::PressureConvectionDiffusion;
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    (void)periodic_faces;

    const double left = -1.0, right = 1.0;
    GridGenerator::hyper_cube(*triangulation, left, right);

    // set boundary indicator
    for(auto cell : triangulation->active_cell_iterators())
    {
      for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      {
        if((std::fabs(cell->face(f)->center()(0) - right) < 1e-12))
          cell->face(f)->set_boundary_id(1);
      }
    }

    triangulation->refine_global(n_refine_space);
  }

  void
  set_boundary_conditions(std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
                          std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // fill boundary descriptor velocity
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(0, new AnalyticalSolutionVelocity<dim>(lambda)));
    boundary_descriptor_velocity->neumann_bc.insert(
      pair(1, new NeumannBoundaryVelocity<dim>(formulation_viscous, lambda)));

    // fill boundary descriptor pressure
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_pressure->dirichlet_bc.insert(
      pair(1, new AnalyticalSolutionPressure<dim>(lambda)));
  }


  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    std::shared_ptr<Function<dim>> initial_solution_velocity;
    std::shared_ptr<Function<dim>> initial_solution_pressure;
    if(initialize_solution_with == InitializeSolutionWith::ZeroFunction)
    {
      initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
      initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    }
    else if(initialize_solution_with == InitializeSolutionWith::AnalyticalSolution)
    {
      initial_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>(lambda));
      initial_solution_pressure.reset(new AnalyticalSolutionPressure<dim>(lambda));
    }

    field_functions->initial_solution_velocity = initial_solution_velocity;
    field_functions->initial_solution_pressure = initial_solution_pressure;
    field_functions->analytical_solution_pressure.reset(
      new AnalyticalSolutionPressure<dim>(lambda));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output         = this->write_output;
    pp_data.output_data.output_folder        = this->output_directory + "vtu/";
    pp_data.output_data.output_name          = this->output_name;
    pp_data.output_data.output_start_time    = start_time;
    pp_data.output_data.output_interval_time = (end_time - start_time) / 20;
    pp_data.output_data.write_divergence     = true;
    pp_data.output_data.degree               = degree;

    // calculation of velocity error
    pp_data.error_data_u.analytical_solution_available = true;
    pp_data.error_data_u.analytical_solution.reset(new AnalyticalSolutionVelocity<dim>(lambda));
    pp_data.error_data_u.error_calc_start_time    = start_time;
    pp_data.error_data_u.error_calc_interval_time = (end_time - start_time) / 20;
    pp_data.error_data_u.name                     = "velocity";

    // ... pressure error
    pp_data.error_data_p.analytical_solution_available = true;
    pp_data.error_data_p.analytical_solution.reset(new AnalyticalSolutionPressure<dim>(lambda));
    pp_data.error_data_p.error_calc_start_time    = start_time;
    pp_data.error_data_p.error_calc_interval_time = (end_time - start_time) / 20;
    pp_data.error_data_p.name                     = "pressure";

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


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_KOVASZNAY_H_ */
