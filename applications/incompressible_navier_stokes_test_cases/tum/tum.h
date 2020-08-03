/*
 * tum.h
 *
 *  Created on: Sep 8, 2017
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TUM_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TUM_H_

namespace IncNS
{
namespace TUM
{
template<int dim>
class AnalyticalSolutionVelocity : public Function<dim>
{
public:
  AnalyticalSolutionVelocity(double const L, double const max_velocity)
    : Function<dim>(dim, 0.0), L(L), max_velocity(max_velocity)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    (void)p;

    double result = 0.0;
    if(component == 0)
    {
      double const t  = this->get_time();
      double const T  = 0.5;
      double const pi = numbers::PI;

      result = t < T ? std::sin(pi / 2. * t / T) : 1.0;

      // constant profile
      result *= max_velocity;
      // parabolic profile
      //      result *= max_velocity * (1.0 - std::pow(p[1]/L+0.5,2.0));
    }

    return result;
  }

private:
  double const L, max_velocity;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  void
  add_parameters(ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("Application");
      prm.add_parameter("OutputDirectory",  output_directory, "Directory where output is written.");
      prm.add_parameter("OutputName",       output_name,      "Name of output files.");
    prm.leave_subsection();
    // clang-format on
  }

  std::string output_directory = "output/tum/", output_name = "test";

  double const L            = 1.0;
  double const max_velocity = 1.0;
  double const viscosity    = 1.0e-3;

  double const start_time = 0.0;
  double const end_time   = 50.0;

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.problem_type                   = ProblemType::Unsteady;
    param.equation_type                  = EquationType::NavierStokes;
    param.formulation_viscous_term       = FormulationViscousTerm::LaplaceFormulation;
    param.formulation_convective_term    = FormulationConvectiveTerm::ConvectiveFormulation;
    param.use_outflow_bc_convective_term = true;
    param.right_hand_side                = false;


    // PHYSICAL QUANTITIES
    param.start_time = start_time;
    param.end_time   = end_time;
    param.viscosity  = viscosity;


    // TEMPORAL DISCRETIZATION
    param.solver_type                   = SolverType::Unsteady;
    param.temporal_discretization       = TemporalDiscretization::BDFDualSplittingScheme;
    param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Explicit;
    param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    // best practice: use adaptive time stepping for this test case to avoid adjusting the CFL
    // number
    param.adaptive_time_stepping          = true;
    param.max_velocity                    = max_velocity;
    param.cfl                             = 0.4;
    param.cfl_exponent_fe_degree_velocity = 1.5;
    param.time_step_size                  = 5.0e-2;
    param.order_time_integrator           = 2;    // 1; // 2; // 3;
    param.start_with_low_order            = true; // true; // false;

    // pseudo-timestepping for steady-state problems
    param.convergence_criterion_steady_problem =
      ConvergenceCriterionSteadyProblem::ResidualSteadyNavierStokes;
    param.abs_tol_steady = 1.e-12;
    param.rel_tol_steady = 1.e-10;

    // output of solver information
    param.solver_info_data.interval_time = (param.end_time - param.start_time) / 200;

    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_p           = DegreePressure::MixedOrder;
    param.mapping            = MappingType::Isoparametric;

    // convective term
    if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // PROJECTION METHODS

    // pressure Poisson equation
    param.solver_pressure_poisson                    = SolverPressurePoisson::CG;
    param.solver_data_pressure_poisson               = SolverData(1000, 1.e-12, 1.e-3, 100);
    param.preconditioner_pressure_poisson            = PreconditionerPressurePoisson::Multigrid;
    param.multigrid_data_pressure_poisson.type       = MultigridType::cphMG;
    param.multigrid_data_pressure_poisson.p_sequence = PSequenceType::Bisect;
    // MG smoother
    param.multigrid_data_pressure_poisson.smoother_data.smoother   = MultigridSmoother::Chebyshev;
    param.multigrid_data_pressure_poisson.smoother_data.iterations = 5;
    // MG coarse grid solver
    param.multigrid_data_pressure_poisson.coarse_problem.solver = MultigridCoarseGridSolver::CG;
    param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;
    param.multigrid_data_pressure_poisson.coarse_problem.solver_data.rel_tol = 1.e-3;

    // projection step
    param.solver_projection         = SolverProjection::CG;
    param.solver_data_projection    = SolverData(1000, 1.e-12, 1.e-3);
    param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // viscous step
    param.solver_viscous         = SolverViscous::CG;
    param.solver_data_viscous    = SolverData(1000, 1.e-12, 1.e-3);
    param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;

    // PRESSURE-CORRECTION SCHEME

    // momentum step

    // Newton solver
    param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-12, 1.e-6);

    // linear solver
    param.solver_momentum                = SolverMomentum::GMRES; // GMRES; //FGMRES;
    param.solver_data_momentum           = SolverData(1e4, 1.e-12, 1.e-2, 100);
    param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
    param.update_preconditioner_momentum = true;

    // formulation
    param.order_pressure_extrapolation = 1;
    param.rotational_formulation       = true;


    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-12, 1.e-6);

    // linear solver
    param.solver_coupled      = SolverCoupled::FGMRES; // FGMRES;
    param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-2, 100);

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
    param.discretization_of_laplacian = DiscretizationOfLaplacian::Classical;
  }

  void create_triangulation(Triangulation<2> & tria)
  {
    std::vector<Triangulation<2>> tria_vector(8);

    GridGenerator::subdivided_hyper_rectangle(tria_vector[0],
                                              std::vector<unsigned int>({4, 1}),
                                              Point<2>(0.0, 0.0),
                                              Point<2>(4.0 * L, -1.0 * L));

    GridGenerator::subdivided_hyper_rectangle(tria_vector[1],
                                              std::vector<unsigned int>({1, 4}),
                                              Point<2>(1.0 * L, -1.0 * L),
                                              Point<2>(2.0 * L, -5.0 * L));

    GridGenerator::subdivided_hyper_rectangle(tria_vector[2],
                                              std::vector<unsigned int>({1, 4}),
                                              Point<2>(3.0 * L, -1.0 * L),
                                              Point<2>(4.0 * L, -5.0 * L));

    GridGenerator::subdivided_hyper_rectangle(tria_vector[3],
                                              std::vector<unsigned int>({1, 1}),
                                              Point<2>(4.0 * L, -4.0 * L),
                                              Point<2>(5.0 * L, -5.0 * L));

    GridGenerator::subdivided_hyper_rectangle(tria_vector[4],
                                              std::vector<unsigned int>({1, 4}),
                                              Point<2>(5.0 * L, -1.0 * L),
                                              Point<2>(6.0 * L, -5.0 * L));

    GridGenerator::subdivided_hyper_rectangle(tria_vector[5],
                                              std::vector<unsigned int>({5, 1}),
                                              Point<2>(5.0 * L, 0.0 * L),
                                              Point<2>(10.0 * L, -1.0 * L));

    GridGenerator::subdivided_hyper_rectangle(tria_vector[6],
                                              std::vector<unsigned int>({1, 4}),
                                              Point<2>(7.0 * L, -1.0 * L),
                                              Point<2>(8.0 * L, -5.0 * L));

    GridGenerator::subdivided_hyper_rectangle(tria_vector[7],
                                              std::vector<unsigned int>({1, 4}),
                                              Point<2>(9.0 * L, -1.0 * L),
                                              Point<2>(10.0 * L, -5.0 * L));

    std::vector<Triangulation<2> const *> tria_vector_ptr(tria_vector.size());
    for(unsigned int i = 0; i < tria_vector.size(); ++i)
      tria_vector_ptr[i] = &tria_vector[i];

    GridGenerator::merge_triangulations(tria_vector_ptr, tria);
  }

  void create_triangulation(Triangulation<3> & tria)
  {
    Triangulation<2> tria_2d;
    create_triangulation(tria_2d);
    GridGenerator::extrude_triangulation(tria_2d, 2, L, tria);
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    (void)periodic_faces;

    create_triangulation(*triangulation);

    // global refinements
    triangulation->refine_global(n_refine_space);

    // set boundary indicator
    // all boundaries have ID = 0 by default -> Dirichlet boundaries
    for(auto cell : triangulation->active_cell_iterators())
    {
      for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      {
        // inflow boundary
        if((std::fabs(cell->face(f)->center()(0)) < 1e-12))
          cell->face(f)->set_boundary_id(1);

        // outflow boundary
        if((std::fabs(cell->face(f)->center()(1) - (-5.0 * L)) < 1e-12) &&
           (cell->face(f)->center()(0) - 9.0 * L) >= 0)
          cell->face(f)->set_boundary_id(2);

        // after extrude triangulation to create the 3d mesh, the boundary IDs on the z-orthogonal
        // boundaries are not 0!
        if(dim == 3)
        {
          if((std::fabs(cell->face(f)->center()(2)) < 1e-12) ||
             (std::fabs(cell->face(f)->center()(2) - (L)) < 1e-12))
            cell->face(f)->set_boundary_id(0);
        }
      }
    }
  }

  void
  set_boundary_conditions(std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
                          std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // fill boundary descriptor velocity
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(1, new AnalyticalSolutionVelocity<dim>(L, max_velocity)));
    boundary_descriptor_velocity->neumann_bc.insert(pair(2, new Functions::ZeroFunction<dim>(dim)));

    // fill boundary descriptor pressure
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_pressure->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_pressure->dirichlet_bc.insert(pair(2, new Functions::ZeroFunction<dim>(1)));
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
    pp_data.output_data.write_output              = true;
    pp_data.output_data.output_folder             = output_directory + "vtu/";
    pp_data.output_data.output_name               = output_name;
    pp_data.output_data.output_start_time         = start_time;
    pp_data.output_data.output_interval_time      = (end_time - start_time) / 200;
    pp_data.output_data.write_divergence          = true;
    pp_data.output_data.write_vorticity           = true;
    pp_data.output_data.write_vorticity_magnitude = true;
    pp_data.output_data.write_q_criterion         = true;
    pp_data.output_data.write_velocity_magnitude  = true;
    pp_data.output_data.write_surface_mesh        = true;
    pp_data.output_data.write_boundary_IDs        = true;
    pp_data.output_data.write_higher_order        = false;
    pp_data.output_data.degree                    = degree;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace TUM
} // namespace IncNS

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TUM_H_ */
