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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_

// ExaDG
#include <exadg/postprocessor/statistics_manager.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

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
double const CHARACTERISTIC_TIME = DIMENSIONS_X1 / MAX_VELOCITY;

double const START_TIME = 0.0;
double const END_TIME   = 200.0 * CHARACTERISTIC_TIME;

double const SAMPLE_START_TIME       = 100.0 * CHARACTERISTIC_TIME;
double const SAMPLE_END_TIME         = END_TIME;
unsigned int SAMPLE_EVERY_TIME_STEPS = 10;

// use a negative GRID_STRETCH_FAC to deactivate grid stretching
double const GRID_STRETCH_FAC = 1.8;

/*
 *  maps eta in [0,1] --> y in [-1,1]*length_y/2.0 (using a hyperbolic mesh stretching)
 */
double
grid_transform_y(double const & eta)
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
inverse_grid_transform_y(double const & y)
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
  push_forward(Point<dim> const & xi) const
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
  pull_back(Point<dim> const & x) const
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
    return std::make_unique<ManifoldTurbulentChannel<dim>>(dimensions);
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
  value(Point<dim> const & p, unsigned int const component = 0) const
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
  typedef typename ApplicationBase<dim, Number>::PeriodicFaces PeriodicFaces;

  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  // solver tolerances
  double const ABS_TOL = 1.e-12;
  double const REL_TOL = 1.e-3;

  double const ABS_TOL_LINEAR = 1.e-12;
  double const REL_TOL_LINEAR = 1.e-2;

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.problem_type                = ProblemType::Unsteady;
    param.equation_type               = EquationType::NavierStokes;
    param.formulation_viscous_term    = FormulationViscousTerm::LaplaceFormulation;
    param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
    param.right_hand_side             = true;


    // PHYSICAL QUANTITIES
    param.start_time = START_TIME;
    param.end_time   = END_TIME;
    param.viscosity  = VISCOSITY;


    // TEMPORAL DISCRETIZATION
    param.solver_type                     = SolverType::Unsteady;
    param.temporal_discretization         = TemporalDiscretization::BDFDualSplittingScheme;
    param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    param.order_time_integrator           = 2;
    param.start_with_low_order            = true;
    param.adaptive_time_stepping          = true;
    param.max_velocity                    = MAX_VELOCITY;
    param.cfl                             = 0.3;
    param.cfl_exponent_fe_degree_velocity = 1.5;
    param.time_step_size                  = 1.0e-1;

    // output of solver information
    param.solver_info_data.interval_time       = CHARACTERISTIC_TIME;
    param.solver_info_data.interval_time_steps = 1;

    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_p           = DegreePressure::MixedOrder;
    param.mapping            = MappingType::Isoparametric;

    // convective term
    if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      param.upwind_factor = 0.5;

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // velocity pressure coupling terms
    param.gradp_formulation = FormulationPressureGradientTerm::Weak;
    param.divu_formulation  = FormulationVelocityDivergenceTerm::Weak;

    // div-div and continuity penalty
    param.use_divergence_penalty                     = true;
    param.divergence_penalty_factor                  = 1.0e0;
    param.use_continuity_penalty                     = true;
    param.continuity_penalty_factor                  = param.divergence_penalty_factor;
    param.continuity_penalty_components              = ContinuityPenaltyComponents::Normal;
    param.apply_penalty_terms_in_postprocessing_step = true;
    param.continuity_penalty_use_boundary_data       = true;


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
    param.solver_pressure_poisson              = SolverPressurePoisson::CG;
    param.solver_data_pressure_poisson         = SolverData(1000, ABS_TOL, REL_TOL, 100);
    param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;

    // projection step
    param.solver_projection         = SolverProjection::CG;
    param.solver_data_projection    = SolverData(1000, ABS_TOL, REL_TOL);
    param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;


    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // viscous step
    param.solver_viscous         = SolverViscous::CG;
    param.solver_data_viscous    = SolverData(1000, ABS_TOL, REL_TOL);
    param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;

    // PRESSURE-CORRECTION SCHEME

    // formulation
    param.order_pressure_extrapolation = 1; // use 0 for non-incremental formulation
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
    param.use_scaling_continuity = false;

    // nonlinear solver (Newton solver)
    param.newton_solver_data_coupled = Newton::SolverData(100, ABS_TOL, REL_TOL);

    // linear solver
    param.solver_coupled = SolverCoupled::GMRES;
    if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      param.solver_data_coupled = SolverData(1e3, ABS_TOL_LINEAR, REL_TOL_LINEAR, 100);
    else
      param.solver_data_coupled = SolverData(1e3, ABS_TOL, REL_TOL, 100);

    // preconditioning linear solver
    param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

    // preconditioner velocity/momentum block
    param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block      = SchurComplementPreconditioner::CahouetChabard;
    param.multigrid_data_pressure_block.type = MultigridType::cphMG;
  }

  void
  create_grid(std::shared_ptr<Triangulation<dim>> triangulation,
              PeriodicFaces &                     periodic_faces,
              unsigned int const                  n_refine_space,
              std::shared_ptr<Mapping<dim>> &     mapping,
              unsigned int const                  mapping_degree)
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

    mapping.reset(new MappingQGeneric<dim>(mapping_degree));
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
    pp_data.output_data.write_output         = this->write_output;
    pp_data.output_data.output_folder        = this->output_directory + "vtu/";
    pp_data.output_data.output_name          = this->output_name;
    pp_data.output_data.output_start_time    = START_TIME;
    pp_data.output_data.output_interval_time = 1.0 * CHARACTERISTIC_TIME;
    pp_data.output_data.degree               = degree;
    pp_data.output_data.write_higher_order   = false;

    // calculate div and mass error
    pp_data.mass_data.calculate               = false; // true;
    pp_data.mass_data.start_time              = START_TIME;
    pp_data.mass_data.sample_every_time_steps = 1e0;
    pp_data.mass_data.directory               = this->output_directory;
    pp_data.mass_data.filename                = this->output_name;
    pp_data.mass_data.reference_length_scale  = 1.0;

    MyPostProcessorData<dim> pp_data_turb_ch;
    pp_data_turb_ch.pp_data = pp_data;

    // turbulent channel statistics
    pp_data_turb_ch.turb_ch_data.calculate              = true;
    pp_data_turb_ch.turb_ch_data.cells_are_stretched    = true;
    pp_data_turb_ch.turb_ch_data.sample_start_time      = SAMPLE_START_TIME;
    pp_data_turb_ch.turb_ch_data.sample_end_time        = SAMPLE_END_TIME;
    pp_data_turb_ch.turb_ch_data.sample_every_timesteps = SAMPLE_EVERY_TIME_STEPS;
    pp_data_turb_ch.turb_ch_data.viscosity              = VISCOSITY;
    pp_data_turb_ch.turb_ch_data.directory              = this->output_directory;
    pp_data_turb_ch.turb_ch_data.filename               = this->output_name;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new MyPostProcessor<dim, Number>(pp_data_turb_ch, mpi_comm));

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


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_ */
