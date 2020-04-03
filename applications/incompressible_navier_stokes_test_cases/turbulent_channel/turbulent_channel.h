/*
 * turbulent_channel.h
 *
 *  Created on: Oct 14, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_

#include "../../../include/postprocessor/statistics_manager.h"

namespace IncNS
{
namespace TurbulentChannel
{
// set problem specific parameters like physical dimensions, etc.
double const DIMENSIONS_X1 = 2.0 * numbers::PI;
double const DIMENSIONS_X2 = 2.0;
double const DIMENSIONS_X3 = numbers::PI;

// nu = 1/180  coarsest meshes: l2_ku3 or l3_ku2
// nu = 1/395
// nu = 1/590
// nu = 1/950
double const VISCOSITY = 1. / 180.; // critical value: 1./50. - 1./75.

// 18.3 for Re_tau = 180
// 20.1 for Re_tau = 395
// 21.3 for Re_tau = 590
// 22.4 for Re_tau = 950
double const MAX_VELOCITY = 18.3;

// flow-through time based on mean centerline velocity
const double CHARACTERISTIC_TIME = DIMENSIONS_X1 / MAX_VELOCITY;

double const START_TIME = 0.0;
double const END_TIME   = 200.0 * CHARACTERISTIC_TIME; // 50.0;

double const SAMPLE_START_TIME = 100.0 * CHARACTERISTIC_TIME; // 30.0;
double const SAMPLE_END_TIME   = END_TIME;

// use a negative GRID_STRETCH_FAC to deactivate grid stretching
const double GRID_STRETCH_FAC = 1.8;

/*
 *  maps eta in [0,1] --> y in [-1,1]*length_y/2.0 (using a hyperbolic mesh stretching)
 */
double
grid_transform_y(const double & eta)
{
  double y = 0.0;

  if(GRID_STRETCH_FAC >= 0)
    y = DIMENSIONS_X2 / 2.0 * std::tanh(GRID_STRETCH_FAC * (2. * eta - 1.)) /
        std::tanh(GRID_STRETCH_FAC);
  else // use a negative GRID_STRETCH_FACTOR deactivate grid stretching
    y = DIMENSIONS_X2 / 2.0 * (2. * eta - 1.);

  return y;
}

/*
 * inverse mapping:
 *
 *  maps y in [-1,1]*length_y/2.0 --> eta in [0,1]
 */
double
inverse_grid_transform_y(const double & y)
{
  double eta = 0.0;

  if(GRID_STRETCH_FAC >= 0)
    eta =
      (std::atanh(y * std::tanh(GRID_STRETCH_FAC) * 2.0 / DIMENSIONS_X2) / GRID_STRETCH_FAC + 1.0) /
      2.0;
  else // use a negative GRID_STRETCH_FACTOR deactivate grid stretching
    eta = (2. * y / DIMENSIONS_X2 + 1.) / 2.0;

  return eta;
}

template<int dim>
class ManifoldTurbulentChannel : public ChartManifold<dim, dim, dim>
{
public:
  ManifoldTurbulentChannel(Tensor<1, dim> const & dimensions_in)
  {
    dimensions = dimensions_in;
  }

  /*
   *  push_forward operation that maps point xi in reference coordinates [0,1]^d to
   *  point x in physical coordinates
   */
  Point<dim>
  push_forward(const Point<dim> & xi) const
  {
    Point<dim> x;

    x[0] = xi[0] * dimensions[0] - dimensions[0] / 2.0;
    x[1] = grid_transform_y(xi[1]);

    if(dim == 3)
      x[2] = xi[2] * dimensions[2] - dimensions[2] / 2.0;

    return x;
  }

  /*
   *  pull_back operation that maps point x in physical coordinates
   *  to point xi in reference coordinates [0,1]^d
   */
  Point<dim>
  pull_back(const Point<dim> & x) const
  {
    Point<dim> xi;

    xi[0] = x[0] / dimensions[0] + 0.5;
    xi[1] = inverse_grid_transform_y(x[1]);

    if(dim == 3)
      xi[2] = x[2] / dimensions[2] + 0.5;

    return xi;
  }

  std::unique_ptr<Manifold<dim>>
  clone() const override
  {
    return std_cxx14::make_unique<ManifoldTurbulentChannel<dim>>(dimensions);
  }

private:
  Tensor<1, dim> dimensions;
};

template<int dim>
class InitialSolutionVelocity : public Function<dim>
{
public:
  InitialSolutionVelocity() : Function<dim>(dim, 0.0)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    AssertThrow(std::abs(p[1]) < DIMENSIONS_X2 / 2.0 + 1.e-12,
                ExcMessage("Invalid geometry parameters."));

    AssertThrow(dim == 3, ExcMessage("Dimension has to be dim==3."));

    double result = 0.0;
    // use turbulent-like profile with superimposed vortices and random noise to initiate a
    // turbulent flow
    if(component == 0)
      result =
        -MAX_VELOCITY * (pow(p[1], 6.0) - 1.0) *
        (1.0 + ((double)rand() / RAND_MAX - 1.0) * 0.5 - 2. / MAX_VELOCITY * std::sin(p[2] * 8.));
    else if(component == 2)
      result = (pow(p[1], 6.0) - 1.0) * std::sin(p[0] * 8.) * 2.;

    return result;
  }
};

template<int dim>
struct MyPostProcessorData
{
  PostProcessorData<dim> pp_data;
  TurbulentChannelData   turb_ch_data;
};

template<int dim, typename Number>
class MyPostProcessor : public PostProcessor<dim, Number>
{
public:
  typedef PostProcessor<dim, Number> Base;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef typename Base::Operator Operator;

  MyPostProcessor(MyPostProcessorData<dim> const & pp_data_turb_channel, MPI_Comm const & mpi_comm)
    : Base(pp_data_turb_channel.pp_data, mpi_comm), turb_ch_data(pp_data_turb_channel.turb_ch_data)
  {
  }

  void
  setup(Operator const & pde_operator)
  {
    // call setup function of base class
    Base::setup(pde_operator);

    // perform setup of turbulent channel related things
    statistics_turb_ch.reset(new StatisticsManager<dim, Number>(pde_operator.get_dof_handler_u(),
                                                                pde_operator.get_mapping()));

    statistics_turb_ch->setup(&grid_transform_y, turb_ch_data);
  }

  void
  do_postprocessing(VectorType const & velocity,
                    VectorType const & pressure,
                    double const       time,
                    int const          time_step_number)
  {
    Base::do_postprocessing(velocity, pressure, time, time_step_number);

    statistics_turb_ch->evaluate(velocity, time, time_step_number);
  }

  TurbulentChannelData                            turb_ch_data;
  std::shared_ptr<StatisticsManager<dim, Number>> statistics_turb_ch;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application() : ApplicationBase<dim, Number>("")
  {
  }

  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    parse_input(input_file, prm, true, true);
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

  std::string output_directory = "output/turbulent_channel/", output_name = "test";

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.problem_type             = ProblemType::Unsteady;
    param.equation_type            = EquationType::NavierStokes;
    param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
    param.right_hand_side          = true;


    // PHYSICAL QUANTITIES
    param.start_time = START_TIME;
    param.end_time   = END_TIME;
    param.viscosity  = VISCOSITY;


    // TEMPORAL DISCRETIZATION
    param.solver_type                     = SolverType::Unsteady;
    param.temporal_discretization         = TemporalDiscretization::BDFDualSplittingScheme;
    param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    param.adaptive_time_stepping          = true;
    param.max_velocity                    = MAX_VELOCITY;
    param.cfl                             = 0.4;
    param.cfl_exponent_fe_degree_velocity = 1.5;
    param.time_step_size                  = 1.0e-1;
    param.order_time_integrator           = 2; // 1; // 2; // 3;
    param.start_with_low_order            = true;

    // output of solver information
    param.solver_info_data.interval_time = CHARACTERISTIC_TIME;

    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_p           = DegreePressure::MixedOrder;
    param.mapping            = MappingType::Isoparametric;

    // convective term
    param.upwind_factor = 0.5;

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // special case: pure DBC's
    param.pure_dirichlet_bc = true;

    // TURBULENCE
    param.use_turbulence_model = false;
    param.turbulence_model     = TurbulenceEddyViscosityModel::Sigma;
    // Smagorinsky: 0.165
    // Vreman: 0.28
    // WALE: 0.50
    // Sigma: 1.35
    param.turbulence_model_constant = 1.35;

    // PROJECTION METHODS

    // pressure Poisson equation
    param.solver_pressure_poisson         = SolverPressurePoisson::CG;
    param.solver_data_pressure_poisson    = SolverData(1000, 1.e-12, 1.e-6, 100);
    param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

    // projection step
    param.solver_projection      = SolverProjection::CG;
    param.solver_data_projection = SolverData(1000, 1.e-12, 1.e-6);
    param.preconditioner_projection =
      PreconditionerProjection::InverseMassMatrix; // BlockJacobi; //PointJacobi;
                                                   // //InverseMassMatrix;
    param.update_preconditioner_projection = true;


    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // viscous step
    param.solver_viscous         = SolverViscous::CG;
    param.solver_data_viscous    = SolverData(1000, 1.e-12, 1.e-6);
    param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; // Multigrid;

    // PRESSURE-CORRECTION SCHEME

    // formulation
    param.order_pressure_extrapolation = 1; // use 0 for non-incremental formulation
    param.rotational_formulation       = true;

    // momentum step

    // Newton solver
    param.newton_solver_data_momentum = NewtonSolverData(100, 1.e-12, 1.e-6);

    // linear solver
    param.solver_momentum                = SolverMomentum::GMRES;
    param.solver_data_momentum           = SolverData(1e4, 1.e-12, 1.e-6, 100);
    param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
    param.update_preconditioner_momentum = false;

    // COUPLED NAVIER-STOKES SOLVER
    param.use_scaling_continuity = false;

    // nonlinear solver (Newton solver)
    param.newton_solver_data_coupled = NewtonSolverData(100, 1.e-12, 1.e-6);

    // linear solver
    param.solver_coupled      = SolverCoupled::GMRES; // GMRES; //FGMRES;
    param.solver_data_coupled = SolverData(1e3, 1.e-12, 1.e-6, 100);

    // preconditioning linear solver
    param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    param.update_preconditioner_coupled = false;

    // preconditioner velocity/momentum block
    param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix; // Multigrid;
    param.multigrid_data_velocity_block.smoother_data.smoother =
      MultigridSmoother::Jacobi; // Jacobi; //Chebyshev; //GMRES;
    param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::BlockJacobi; // PointJacobi; //BlockJacobi;
    param.multigrid_data_velocity_block.smoother_data.iterations        = 4;
    param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
    param.multigrid_data_velocity_block.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block =
      SchurComplementPreconditioner::CahouetChabard; // PressureConvectionDiffusion;
    param.discretization_of_laplacian = DiscretizationOfLaplacian::Classical;
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    Tensor<1, dim> dimensions;
    dimensions[0] = DIMENSIONS_X1;
    dimensions[1] = DIMENSIONS_X2;
    if(dim == 3)
      dimensions[2] = DIMENSIONS_X3;

    GridGenerator::hyper_rectangle(*triangulation,
                                   Point<dim>(-dimensions / 2.0),
                                   Point<dim>(dimensions / 2.0));

    // manifold
    unsigned int manifold_id = 1;
    for(auto cell : triangulation->active_cell_iterators())
    {
      cell->set_all_manifold_ids(manifold_id);
    }

    // apply mesh stretching towards no-slip boundaries in y-direction
    static const ManifoldTurbulentChannel<dim> manifold(dimensions);
    triangulation->set_manifold(manifold_id, manifold);

    // periodicity in x--direction
    triangulation->begin()->face(0)->set_all_boundary_ids(0 + 10);
    triangulation->begin()->face(1)->set_all_boundary_ids(1 + 10);
    // periodicity in z-direction
    if(dim == 3)
    {
      triangulation->begin()->face(4)->set_all_boundary_ids(2 + 10);
      triangulation->begin()->face(5)->set_all_boundary_ids(3 + 10);
    }

    auto tria = dynamic_cast<Triangulation<dim> *>(&*triangulation);
    GridTools::collect_periodic_faces(*tria, 0 + 10, 1 + 10, 0, periodic_faces);
    if(dim == 3)
      GridTools::collect_periodic_faces(*tria, 2 + 10, 3 + 10, 2, periodic_faces);

    triangulation->add_periodicity(periodic_faces);

    // perform global refinements
    triangulation->refine_global(n_refine_space);
  }

  void
  set_boundary_conditions(std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
                          std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution_velocity.reset(new InitialSolutionVelocity<dim>());
    field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    std::vector<double> forcing = std::vector<double>(dim, 0.0);
    forcing[0]                  = 1.0; // constant forcing in x_1-direction
    field_functions->right_hand_side.reset(new Functions::ConstantFunction<dim>(forcing));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output         = true;
    pp_data.output_data.output_folder        = output_directory + "vtu/";
    pp_data.output_data.output_name          = output_name;
    pp_data.output_data.output_start_time    = START_TIME;
    pp_data.output_data.output_interval_time = 1.0;
    pp_data.output_data.write_divergence     = true;
    pp_data.output_data.degree               = degree;
    pp_data.output_data.write_higher_order   = false;

    // calculate div and mass error
    pp_data.mass_data.calculate_error         = false; // true;
    pp_data.mass_data.start_time              = START_TIME;
    pp_data.mass_data.sample_every_time_steps = 1e0; // 1e2;
    pp_data.mass_data.filename_prefix         = output_directory + output_name;
    pp_data.mass_data.reference_length_scale  = 1.0;

    MyPostProcessorData<dim> pp_data_turb_ch;
    pp_data_turb_ch.pp_data = pp_data;

    // turbulent channel statistics
    pp_data_turb_ch.turb_ch_data.calculate_statistics   = true;
    pp_data_turb_ch.turb_ch_data.cells_are_stretched    = true;
    pp_data_turb_ch.turb_ch_data.sample_start_time      = SAMPLE_START_TIME;
    pp_data_turb_ch.turb_ch_data.sample_end_time        = SAMPLE_END_TIME;
    pp_data_turb_ch.turb_ch_data.sample_every_timesteps = 10;
    pp_data_turb_ch.turb_ch_data.viscosity              = VISCOSITY;
    pp_data_turb_ch.turb_ch_data.filename_prefix        = output_directory + output_name;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new MyPostProcessor<dim, Number>(pp_data_turb_ch, mpi_comm));

    return pp;
  }
};

} // namespace TurbulentChannel
} // namespace IncNS

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_ */
