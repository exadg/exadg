/*
 * cavity.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

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

  double const L = 1.0;

  double const start_time = 0.0;
  double const end_time   = 10.0;

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.problem_type                = ProblemType::Steady;
    param.equation_type               = EquationType::NavierStokes;
    param.formulation_viscous_term    = FormulationViscousTerm::LaplaceFormulation;
    param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
    param.right_hand_side             = false;


    // PHYSICAL QUANTITIES
    param.start_time = start_time;
    param.end_time   = end_time;
    param.viscosity  = 1.0e-2;


    // TEMPORAL DISCRETIZATION
    param.solver_type                     = SolverType::Steady;
    param.temporal_discretization         = TemporalDiscretization::BDFCoupledSolution;
    param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Implicit;
    param.time_integrator_oif             = TimeIntegratorOIF::ExplRK3Stage7Reg2;
    param.adaptive_time_stepping          = false;
    param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    param.max_velocity                    = 1.0;
    param.cfl_exponent_fe_degree_velocity = 1.5;
    // Explicit: CFL_crit = 0.35 (0.4 unstable), ExplicitOIF: CFL_crit,oif = 3.0 (3.5 unstable)
    param.cfl_oif               = 3.0;
    param.cfl                   = param.cfl_oif * 1.0;
    param.time_step_size        = 1.0e-1;
    param.order_time_integrator = 2;
    param.start_with_low_order  = true;

    // pseudo-timestepping for steady-state problems
    param.convergence_criterion_steady_problem =
      ConvergenceCriterionSteadyProblem::ResidualSteadyNavierStokes;
    param.abs_tol_steady = 1.e-12;
    param.rel_tol_steady = 1.e-8;

    // restart
    param.restart_data.write_restart       = false;
    param.restart_data.interval_time       = 5.0;
    param.restart_data.interval_wall_time  = 1.e6;
    param.restart_data.interval_time_steps = 1e8;
    param.restart_data.filename            = "output/cavity/cavity_restart";

    // output of solver information
    param.solver_info_data.interval_time_steps = 1;
    param.solver_info_data.interval_time       = (param.end_time - param.start_time) / 10;


    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_p           = DegreePressure::MixedOrder;
    param.mapping            = MappingType::Isoparametric;

    // convective term
    if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      param.upwind_factor = 0.5;

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // div-div and continuity penalty
    param.use_divergence_penalty                     = true;
    param.divergence_penalty_factor                  = 1.0e0;
    param.use_continuity_penalty                     = true;
    param.continuity_penalty_factor                  = param.divergence_penalty_factor;
    param.apply_penalty_terms_in_postprocessing_step = false;

    // NUMERICAL PARAMETERS
    param.implement_block_diagonal_preconditioner_matrix_free = false;
    param.use_cell_based_face_loops                           = false;
    param.solver_data_block_diagonal = SolverData(1000, 1.e-12, 1.e-2, 1000);
    param.quad_rule_linearization    = QuadratureRuleLinearization::Overintegration32k;

    // PROJECTION METHODS

    // pressure Poisson equation
    param.solver_data_pressure_poisson    = SolverData(1000, 1.e-12, 1.e-8, 100);
    param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

    // projection step
    param.solver_projection         = SolverProjection::CG;
    param.solver_data_projection    = SolverData(1000, 1.e-12, 1.e-8);
    param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;


    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // viscous step
    param.solver_viscous         = SolverViscous::CG;
    param.solver_data_viscous    = SolverData(1000, 1.e-12, 1.e-8);
    param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; // Multigrid;


    // PRESSURE-CORRECTION SCHEME

    // momentum step

    // Newton solver
    param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-20, 1.e-6);

    // linear solver
    param.solver_momentum = SolverMomentum::GMRES; // FGMRES;
    if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      param.solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-2, 100);
    else
      param.solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-6, 100);
    param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
    param.update_preconditioner_momentum = true;
    param.multigrid_data_momentum.smoother_data.smoother = MultigridSmoother::Jacobi;
    param.multigrid_data_momentum.smoother_data.preconditioner =
      PreconditionerSmoother::BlockJacobi;
    param.multigrid_data_momentum.smoother_data.iterations        = 5;
    param.multigrid_data_momentum.smoother_data.relaxation_factor = 0.7;
    param.multigrid_data_momentum.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

    // formulation
    param.order_pressure_extrapolation = 1;
    param.rotational_formulation       = true;


    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-12, 1.e-8);

    // linear solver
    param.solver_coupled = SolverCoupled::FGMRES; // FGMRES;
    if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-2, 1000);
    else
      param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-6, 1000);

    // preconditioning linear solver
    param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    param.update_preconditioner_coupled = true;

    // preconditioner velocity/momentum block
    param.preconditioner_velocity_block = MomentumPreconditioner::Multigrid;
    param.multigrid_operator_type_velocity_block =
      MultigridOperatorType::ReactionConvectionDiffusion;
    param.multigrid_data_velocity_block.smoother_data.smoother =
      MultigridSmoother::Jacobi; // GMRES;
    param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::BlockJacobi;
    param.multigrid_data_velocity_block.smoother_data.iterations        = 5;
    param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;

    param.multigrid_data_velocity_block.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

    param.exact_inversion_of_velocity_block = false; // true;
    param.solver_data_velocity_block        = SolverData(1e4, 1.e-12, 1.e-6, 100);

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block =
      SchurComplementPreconditioner::PressureConvectionDiffusion;
    param.discretization_of_laplacian = DiscretizationOfLaplacian::Classical;
    param.multigrid_data_pressure_block.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev;
    param.exact_inversion_of_laplace_operator = false;
    param.solver_data_pressure_block          = SolverData(1e4, 1.e-12, 1.e-6, 100);
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    (void)periodic_faces;

    const double left = 0.0, right = L;
    GridGenerator::hyper_cube(*triangulation, left, right);
    triangulation->refine_global(n_refine_space);

    // set boundary indicator
    for(auto cell : triangulation->active_cell_iterators())
    {
      for(unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell;
          ++face_number)
      {
        if((std::fabs(cell->face(face_number)->center()(1) - L) < 1e-12))
          cell->face(face_number)->set_boundary_id(1);
      }
    }
  }

  void
  set_boundary_conditions(std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
                          std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure)
  {
    // all boundaries have ID = 0 by default -> Dirichlet boundaries

    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // fill boundary descriptor velocity
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));
    std::vector<double> velocity = std::vector<double>(dim, 0.0);
    velocity[0]                  = 1.0;
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(1, new Functions::ConstantFunction<dim>(velocity)));

    // fill boundary descriptor pressure
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_pressure->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(dim)));
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
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
    pp_data.output_data.output_interval_time = (end_time - start_time) / 100;
    pp_data.output_data.write_divergence     = true;
    pp_data.output_data.write_vorticity      = true;
    pp_data.output_data.write_streamfunction = true; // false;
    pp_data.output_data.write_processor_id   = true;
    pp_data.output_data.degree               = degree;

    // consider line plots only for two-dimensional case
    if(dim == 2)
    {
      // line plot data
      pp_data.line_plot_data.calculate           = false;
      pp_data.line_plot_data.line_data.directory = this->output_directory;

      // which quantities
      std::shared_ptr<Quantity> quantity_u;
      quantity_u.reset(new Quantity());
      quantity_u->type = QuantityType::Velocity;
      std::shared_ptr<Quantity> quantity_p;
      quantity_p.reset(new Quantity());
      quantity_p->type = QuantityType::Pressure;

      // lines
      std::shared_ptr<Line<dim>> vert_line, hor_line;

      // vertical line
      vert_line.reset(new Line<dim>());
      vert_line->begin    = Point<dim>(0.5, 0.0);
      vert_line->end      = Point<dim>(0.5, 1.0);
      vert_line->name     = this->output_name + "_vert_line";
      vert_line->n_points = 100001; // 2001;
      vert_line->quantities.push_back(quantity_u);
      vert_line->quantities.push_back(quantity_p);
      pp_data.line_plot_data.line_data.lines.push_back(vert_line);

      // horizontal line
      hor_line.reset(new Line<dim>());
      hor_line->begin    = Point<dim>(0.0, 0.5);
      hor_line->end      = Point<dim>(1.0, 0.5);
      hor_line->name     = this->output_name + "_hor_line";
      hor_line->n_points = 10001; // 2001;
      hor_line->quantities.push_back(quantity_u);
      hor_line->quantities.push_back(quantity_p);
      pp_data.line_plot_data.line_data.lines.push_back(hor_line);
    }

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
  return std::shared_ptr<IncNS::ApplicationBase<dim, Number>>(
    new IncNS::Application<dim, Number>(input_file));
}

} // namespace ExaDG


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_ */
