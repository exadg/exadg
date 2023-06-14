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
#include <exadg/utilities/numbers.h>

namespace ExaDG
{
namespace IncNS
{
// set problem specific parameters like physical dimensions, etc.
double const DIMENSIONS_X1 = 2.0 * dealii::numbers::PI;
double const DIMENSIONS_X2 = 2.0;
double const DIMENSIONS_X3 = dealii::numbers::PI;

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
class ManifoldTurbulentChannel : public dealii::ChartManifold<dim, dim, dim>
{
public:
  ManifoldTurbulentChannel(dealii::Tensor<1, dim> const & dimensions_in)
  {
    dimensions = dimensions_in;
  }

  /*
   *  push_forward operation that maps point xi in reference coordinates [0,1]^d to
   *  point x in physical coordinates
   */
  dealii::Point<dim>
  push_forward(dealii::Point<dim> const & xi) const final
  {
    dealii::Point<dim> x;

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
  dealii::Point<dim>
  pull_back(dealii::Point<dim> const & x) const final
  {
    dealii::Point<dim> xi;

    xi[0] = x[0] / dimensions[0] + 0.5;
    xi[1] = inverse_grid_transform_y(x[1]);

    if(dim == 3)
      xi[2] = x[2] / dimensions[2] + 0.5;

    return xi;
  }

  std::unique_ptr<dealii::Manifold<dim>>
  clone() const final
  {
    return std::make_unique<ManifoldTurbulentChannel<dim>>(dimensions);
  }

private:
  dealii::Tensor<1, dim> dimensions;
};

template<int dim>
class InitialSolutionVelocity : public dealii::Function<dim>
{
public:
  InitialSolutionVelocity() : dealii::Function<dim>(dim, 0.0)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    AssertThrow(std::abs(p[1]) < DIMENSIONS_X2 / 2.0 + 1.e-12,
                dealii::ExcMessage("Invalid geometry parameters."));

    AssertThrow(dim == 3, dealii::ExcMessage("Dimension has to be dim==3."));

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

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef typename Base::Operator Operator;

  MyPostProcessor(MyPostProcessorData<dim> const & pp_data_turb_channel, MPI_Comm const & mpi_comm)
    : Base(pp_data_turb_channel.pp_data, mpi_comm), turb_ch_data(pp_data_turb_channel.turb_ch_data)
  {
  }

  void
  setup(Operator const & pde_operator) final
  {
    // call setup function of base class
    Base::setup(pde_operator);

    // perform setup of turbulent channel related things
    statistics_turb_ch.reset(new StatisticsManager<dim, Number>(pde_operator.get_dof_handler_u(),
                                                                *pde_operator.get_mapping()));

    statistics_turb_ch->setup(&grid_transform_y, turb_ch_data);
  }

  void
  do_postprocessing(VectorType const &     velocity,
                    VectorType const &     pressure,
                    double const           time,
                    types::time_step const time_step_number) final
  {
    Base::do_postprocessing(velocity, pressure, time, time_step_number);

    if(statistics_turb_ch->time_control_statistics.time_control.needs_evaluation(time,
                                                                                 time_step_number))
    {
      statistics_turb_ch->evaluate(velocity, Utilities::is_unsteady_timestep(time_step_number));
    }

    if(statistics_turb_ch->time_control_statistics.write_preliminary_results(time,
                                                                             time_step_number))
    {
      statistics_turb_ch->write_output();
    }
  }

  TurbulentChannelData                            turb_ch_data;
  std::shared_ptr<StatisticsManager<dim, Number>> statistics_turb_ch;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

private:
  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type                = ProblemType::Unsteady;
    this->param.equation_type               = EquationType::NavierStokes;
    this->param.formulation_viscous_term    = FormulationViscousTerm::LaplaceFormulation;
    this->param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
    this->param.right_hand_side             = true;


    // PHYSICAL QUANTITIES
    this->param.start_time = START_TIME;
    this->param.end_time   = END_TIME;
    this->param.viscosity  = VISCOSITY;


    // TEMPORAL DISCRETIZATION
    this->param.solver_type                     = SolverType::Unsteady;
    this->param.temporal_discretization         = TemporalDiscretization::BDFDualSplittingScheme;
    this->param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    this->param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    this->param.order_time_integrator           = 2;
    this->param.start_with_low_order            = true;
    this->param.adaptive_time_stepping          = true;
    this->param.max_velocity                    = MAX_VELOCITY;
    this->param.cfl                             = 0.3;
    this->param.cfl_exponent_fe_degree_velocity = 1.5;
    this->param.time_step_size                  = 1.0e-1;

    // output of solver information
    this->param.solver_info_data.interval_time       = CHARACTERISTIC_TIME;
    this->param.solver_info_data.interval_time_steps = 1;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.mapping_degree          = this->param.degree_u;
    this->param.degree_p                = DegreePressure::MixedOrder;

    // convective term
    if(this->param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      this->param.upwind_factor = 0.5;

    // viscous term
    this->param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // velocity pressure coupling terms
    this->param.gradp_formulation = FormulationPressureGradientTerm::Weak;
    this->param.divu_formulation  = FormulationVelocityDivergenceTerm::Weak;

    // div-div and continuity penalty
    this->param.use_divergence_penalty                     = true;
    this->param.divergence_penalty_factor                  = 1.0e0;
    this->param.use_continuity_penalty                     = true;
    this->param.continuity_penalty_factor                  = this->param.divergence_penalty_factor;
    this->param.continuity_penalty_components              = ContinuityPenaltyComponents::Normal;
    this->param.apply_penalty_terms_in_postprocessing_step = true;
    this->param.continuity_penalty_use_boundary_data       = true;


    // TURBULENCE
    this->param.turbulence_model_data.is_active        = false;
    this->param.turbulence_model_data.turbulence_model = TurbulenceEddyViscosityModel::Sigma;
    // Smagorinsky: 0.165
    // Vreman: 0.28
    // WALE: 0.50
    // Sigma: 1.35
    this->param.turbulence_model_data.constant = 1.35;

    // PROJECTION METHODS

    // pressure Poisson equation
    this->param.solver_pressure_poisson              = SolverPressurePoisson::CG;
    this->param.solver_data_pressure_poisson         = SolverData(1000, ABS_TOL, REL_TOL, 100);
    this->param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    this->param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;

    // projection step
    this->param.solver_projection         = SolverProjection::CG;
    this->param.solver_data_projection    = SolverData(1000, ABS_TOL, REL_TOL);
    this->param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;


    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    this->param.order_extrapolation_pressure_nbc =
      this->param.order_time_integrator <= 2 ? this->param.order_time_integrator : 2;

    // viscous step
    this->param.solver_viscous         = SolverViscous::CG;
    this->param.solver_data_viscous    = SolverData(1000, ABS_TOL, REL_TOL);
    this->param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;

    // PRESSURE-CORRECTION SCHEME

    // formulation
    this->param.order_pressure_extrapolation = 1; // use 0 for non-incremental formulation
    this->param.rotational_formulation       = true;

    // momentum step

    // Newton solver
    this->param.newton_solver_data_momentum = Newton::SolverData(100, ABS_TOL, REL_TOL);

    // linear solver
    this->param.solver_momentum = SolverMomentum::GMRES;
    if(this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      this->param.solver_data_momentum = SolverData(1e4, ABS_TOL_LINEAR, REL_TOL_LINEAR, 100);
    else
      this->param.solver_data_momentum = SolverData(1e4, ABS_TOL, REL_TOL, 100);

    this->param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;

    // COUPLED NAVIER-STOKES SOLVER
    this->param.use_scaling_continuity = false;

    // nonlinear solver (Newton solver)
    this->param.newton_solver_data_coupled = Newton::SolverData(100, ABS_TOL, REL_TOL);

    // linear solver
    this->param.solver_coupled = SolverCoupled::GMRES;
    if(this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      this->param.solver_data_coupled = SolverData(1e3, ABS_TOL_LINEAR, REL_TOL_LINEAR, 100);
    else
      this->param.solver_data_coupled = SolverData(1e3, ABS_TOL, REL_TOL, 100);

    // preconditioning linear solver
    this->param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

    // preconditioner velocity/momentum block
    this->param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

    // preconditioner Schur-complement block
    this->param.preconditioner_pressure_block      = SchurComplementPreconditioner::CahouetChabard;
    this->param.multigrid_data_pressure_block.type = MultigridType::cphMG;
  }

  void
  create_grid() final
  {
    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> &                        tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & periodic_face_pairs,
          unsigned int const                                       global_refinements,
          std::vector<unsigned int> const &                        vector_local_refinements) {
        (void)vector_local_refinements;

        dealii::Tensor<1, dim> dimensions;
        dimensions[0] = DIMENSIONS_X1;
        dimensions[1] = DIMENSIONS_X2;
        if(dim == 3)
          dimensions[2] = DIMENSIONS_X3;

        dealii::GridGenerator::hyper_rectangle(tria,
                                               dealii::Point<dim>(-dimensions / 2.0),
                                               dealii::Point<dim>(dimensions / 2.0));

        // manifold
        unsigned int manifold_id = 1;
        for(auto cell : tria.cell_iterators())
        {
          cell->set_all_manifold_ids(manifold_id);
        }

        // apply mesh stretching towards no-slip boundaries in y-direction
        static const ManifoldTurbulentChannel<dim> manifold(dimensions);
        tria.set_manifold(manifold_id, manifold);

        // periodicity in x--direction
        tria.begin()->face(0)->set_all_boundary_ids(0 + 10);
        tria.begin()->face(1)->set_all_boundary_ids(1 + 10);
        // periodicity in z-direction
        if(dim == 3)
        {
          tria.begin()->face(4)->set_all_boundary_ids(2 + 10);
          tria.begin()->face(5)->set_all_boundary_ids(3 + 10);
        }

        dealii::GridTools::collect_periodic_faces(tria, 0 + 10, 1 + 10, 0, periodic_face_pairs);
        if(dim == 3)
        {
          dealii::GridTools::collect_periodic_faces(tria, 2 + 10, 3 + 10, 2, periodic_face_pairs);
        }

        tria.add_periodicity(periodic_face_pairs);

        tria.refine_global(global_refinements);
      };

    GridUtilities::create_fine_and_coarse_triangulations<dim>(*this->grid,
                                                              this->mpi_comm,
                                                              this->param.grid,
                                                              this->param.involves_h_multigrid(),
                                                              lambda_create_triangulation,
                                                              {} /* no local refinements */);
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));

    this->boundary_descriptor->pressure->neumann_bc.insert(0);
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(new InitialSolutionVelocity<dim>());
    this->field_functions->initial_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->analytical_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    std::vector<double> forcing = std::vector<double>(dim, 0.0);
    forcing[0]                  = 1.0; // constant forcing in x_1-direction
    this->field_functions->right_hand_side.reset(
      new dealii::Functions::ConstantFunction<dim>(forcing));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = START_TIME;
    pp_data.output_data.time_control_data.trigger_interval = 1.0 * CHARACTERISTIC_TIME;
    pp_data.output_data.directory          = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = this->output_parameters.filename;
    pp_data.output_data.degree             = this->param.degree_u;
    pp_data.output_data.write_higher_order = false;

    // calculate div and mass error
    pp_data.mass_data.time_control_data.is_active                = false; // true;
    pp_data.mass_data.time_control_data.start_time               = START_TIME;
    pp_data.mass_data.time_control_data.trigger_every_time_steps = 1e0;
    pp_data.mass_data.directory              = this->output_parameters.directory;
    pp_data.mass_data.filename               = this->output_parameters.filename;
    pp_data.mass_data.reference_length_scale = 1.0;

    MyPostProcessorData<dim> pp_data_turb_ch;
    pp_data_turb_ch.pp_data = pp_data;

    // turbulent channel statistics
    pp_data_turb_ch.turb_ch_data.time_control_data_statistics.time_control_data.is_active = true;
    pp_data_turb_ch.turb_ch_data.time_control_data_statistics.time_control_data.start_time =
      SAMPLE_START_TIME;
    pp_data_turb_ch.turb_ch_data.time_control_data_statistics.time_control_data.end_time =
      SAMPLE_END_TIME;
    pp_data_turb_ch.turb_ch_data.time_control_data_statistics.time_control_data
      .trigger_every_time_steps = SAMPLE_EVERY_TIME_STEPS;
    pp_data_turb_ch.turb_ch_data.time_control_data_statistics
      .write_preliminary_results_every_nth_time_step = SAMPLE_EVERY_TIME_STEPS * 100;

    pp_data_turb_ch.turb_ch_data.cells_are_stretched = true;
    pp_data_turb_ch.turb_ch_data.viscosity           = VISCOSITY;
    pp_data_turb_ch.turb_ch_data.directory           = this->output_parameters.directory;
    pp_data_turb_ch.turb_ch_data.filename            = this->output_parameters.filename;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new MyPostProcessor<dim, Number>(pp_data_turb_ch, this->mpi_comm));

    return pp;
  }

  // solver tolerances
  double const ABS_TOL = 1.e-12;
  double const REL_TOL = 1.e-3;

  double const ABS_TOL_LINEAR = 1.e-12;
  double const REL_TOL_LINEAR = 1.e-2;
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_ */
