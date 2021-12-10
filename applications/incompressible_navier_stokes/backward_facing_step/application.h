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
#include <exadg/functions_and_boundary_conditions/linear_interpolation.h>

// backward facing step application
#include "include/geometry.h"
#include "include/inflow_data_storage.h"
#include "include/postprocessor.h"

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim>
class InitialSolutionVelocity : public Function<dim>
{
public:
  InitialSolutionVelocity(double const max_velocity,
                          double const length,
                          double const height,
                          double const width)
    : Function<dim>(dim, 0.0),
      max_velocity(max_velocity),
      length(length),
      height(height),
      width(width)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const component = 0) const
  {
    AssertThrow(dim == 3, ExcMessage("Dimension has to be dim==3."));

    double const x = p[0] / length;
    double const y = (p[1] - height / 2.0) / (height / 2.0);
    double const z = p[2] / width;

    double const factor = 0.5;

    double result = 0.0;
    if(false) // only random perturbations
    {
      if(component == 0)
      {
        if(std::abs(y) < 1.0)
          result = -max_velocity * (pow(y, 6.0) - 1.0) *
                   (1.0 + ((double)rand() / RAND_MAX - 0.5) * factor);
        else
          result = 0.0;
      }
    }
    else // random perturbations and vortices
    {
      if(component == 0)
      {
        if(std::abs(y) < 1.0)
          result = -max_velocity * (pow(y, 6.0) - 1.0) *
                   (1.0 + (((double)rand() / RAND_MAX - 1.0) + std::sin(z * 8.0) * 0.5) * factor);
      }

      if(component == 2)
      {
        if(std::abs(y) < 1.0)
          result = -max_velocity * (pow(y, 6.0) - 1.0) * std::sin(x * 8.0) * 0.5 * factor;
      }
    }

    return result;
  }

private:
  double const max_velocity, length, height, width;
};

template<int dim>
class InflowProfile : public Function<dim>
{
public:
  InflowProfile(InflowDataStorage<dim> const & inflow_data_storage)
    : Function<dim>(dim, 0.0), data(inflow_data_storage)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const component = 0) const
  {
    double result = linear_interpolation_2d_cartesian(
      p, data.y_values, data.z_values, data.velocity_values, component);

    return result;
  }

private:
  InflowDataStorage<dim> const & data;
};

template<int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide() : Function<dim>(dim, 0.0)
  {
  }

  double
  value(Point<dim> const & /*p*/, unsigned int const component = 0) const
  {
    double result = 0.0;

    // channel flow with periodic boundary conditions.
    // The force is known (so no flow rate controller is used).
    if(component == 0)
      return 0.2844518;
    else
      return 0.0;

    return result;
  }
};


template<int dim, typename Number>
class Application : public ApplicationBasePrecursor<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBasePrecursor<dim, Number>(input_file, comm)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);

    inflow_data_storage.reset(new InflowDataStorage<dim>(n_points_inflow));
  }

  // consider a friction Reynolds number of Re_tau = u_tau * H / nu = 290
  // and body force f = tau_w/H with tau_w = u_tau^2.
  double const viscosity = 1.5268e-5;

  // precursor simulation: data structures for storage of inflow data
  std::shared_ptr<InflowDataStorage<dim>> inflow_data_storage;

  // number of points for inflow boundary condition
  unsigned int const n_points_inflow = 101;

  unsigned int const additional_refinements_precursor = 0;

  // start and end time
  double const Re_H                 = 5540.0;
  double const centerline_velocity  = Re_H * viscosity / Geometry::H;
  double const characteristic_time  = Geometry::H / centerline_velocity;
  double const start_time           = 0.0;
  double const precursor_start_time = -300.0 * characteristic_time;
  double const end_time             = 300.0 * characteristic_time;

  // postprocessing

  // sampling of statistical results
  double const       sample_start_time      = 100.0 * characteristic_time;
  unsigned int const sample_every_timesteps = 10;
  unsigned int const n_points_per_line      = 101;

  // solver tolerances
  double const ABS_TOL = 1.e-12;
  double const REL_TOL = 1.e-3;

  double const ABS_TOL_LINEAR = 1.e-12;
  double const REL_TOL_LINEAR = 1.e-2;

  /*
   *  Most of the input parameters are the same for both domains, so we write
   *  this function for the actual domain and only "correct" the parameters
   *  for the precursor by passing an additional parameter is_precursor.
   */
  void
  do_set_parameters(InputParameters &  param,
                    unsigned int const degree,
                    bool const         is_precursor = false)
  {
    // MATHEMATICAL MODEL
    param.problem_type                   = ProblemType::Unsteady;
    param.equation_type                  = EquationType::NavierStokes;
    param.use_outflow_bc_convective_term = true;
    param.formulation_viscous_term       = FormulationViscousTerm::LaplaceFormulation;
    param.formulation_convective_term    = FormulationConvectiveTerm::DivergenceFormulation;
    param.right_hand_side                = true;


    // PHYSICAL QUANTITIES
    if(is_precursor)
      param.start_time = precursor_start_time;
    else
      param.start_time = start_time;
    param.end_time  = end_time;
    param.viscosity = viscosity;


    // TEMPORAL DISCRETIZATION
    param.solver_type                     = SolverType::Unsteady;
    param.temporal_discretization         = TemporalDiscretization::BDFDualSplittingScheme;
    param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    param.order_time_integrator           = 2;
    param.start_with_low_order            = true;
    param.adaptive_time_stepping          = true;
    param.max_velocity                    = centerline_velocity;
    param.cfl                             = 0.3;
    param.cfl_exponent_fe_degree_velocity = 1.5;
    param.time_step_size                  = 1.0e-1;
    param.order_time_integrator           = 2;

    // output of solver information
    param.solver_info_data.interval_time = (end_time - start_time) / 100;

    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_u           = degree;
    param.degree_p           = DegreePressure::MixedOrder;
    param.mapping            = MappingType::Isoparametric;

    // convective term
    if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

    // variant Direct allows to use larger time step
    // sizes due to CFL condition at inflow boundary
    param.type_dirichlet_bc_convective = TypeDirichletBCs::Mirror;

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

    // formulation
    param.store_previous_boundary_values = true;

    // pressure Poisson equation
    param.solver_pressure_poisson              = SolverPressurePoisson::CG;
    param.solver_data_pressure_poisson         = SolverData(1e4, 1.e-12, 1.e-6, 100);
    param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
    param.multigrid_data_pressure_poisson.smoother_data.smoother = MultigridSmoother::Chebyshev;
    param.multigrid_data_pressure_poisson.smoother_data.preconditioner =
      PreconditionerSmoother::PointJacobi;
    param.multigrid_data_pressure_poisson.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev;
    param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::PointJacobi;

    // pressure Poisson equation
    param.solver_pressure_poisson              = SolverPressurePoisson::CG;
    param.solver_data_pressure_poisson         = SolverData(1000, ABS_TOL, REL_TOL, 100);
    param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
    param.multigrid_data_pressure_poisson.coarse_problem.solver = MultigridCoarseGridSolver::CG;
    param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;

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
    param.multigrid_data_pressure_block.coarse_problem.solver = MultigridCoarseGridSolver::CG;
    param.multigrid_data_pressure_block.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;
  }

  void
  set_parameters(unsigned int const degree) final
  {
    do_set_parameters(this->param, degree);
  }

  void
  set_parameters_precursor(unsigned int const degree) final
  {
    do_set_parameters(this->param_pre, degree, true);
  }

  std::shared_ptr<Grid<dim, Number>>
  create_grid(GridData const & grid_data) final
  {
    std::shared_ptr<Grid<dim, Number>> grid =
      std::make_shared<Grid<dim, Number>>(grid_data, this->mpi_comm);

    Geometry::create_grid(grid->triangulation, grid_data.n_refine_global, grid->periodic_faces);

    return grid;
  }

  std::shared_ptr<Grid<dim, Number>>
  create_grid_precursor(GridData const & grid_data) final
  {
    std::shared_ptr<Grid<dim, Number>> grid =
      std::make_shared<Grid<dim, Number>>(grid_data, this->mpi_comm);

    Geometry::create_grid_precursor(grid->triangulation,
                                    grid_data.n_refine_global + additional_refinements_precursor,
                                    grid->periodic_faces);

    return grid;
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // fill boundary descriptor velocity
    // no slip boundaries at the upper and lower wall with ID=0
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));

    // inflow boundary condition at left boundary with ID=2: prescribe velocity profile which
    // is obtained as the results of the precursor simulation
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(2, new InflowProfile<dim>(*inflow_data_storage)));

    // outflow boundary condition at right boundary with ID=1
    this->boundary_descriptor->velocity->neumann_bc.insert(
      pair(1, new Functions::ZeroFunction<dim>(dim)));

    // fill boundary descriptor pressure
    // no slip boundaries at the upper and lower wall with ID=0
    this->boundary_descriptor->pressure->neumann_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));

    // inflow boundary condition at left boundary with ID=2
    // the inflow boundary condition is time dependent (du/dt != 0) but, for simplicity,
    // we assume that this is negligible when using the dual splitting scheme
    this->boundary_descriptor->pressure->neumann_bc.insert(
      pair(2, new Functions::ZeroFunction<dim>(dim)));

    // outflow boundary condition at right boundary with ID=1: set pressure to zero
    this->boundary_descriptor->pressure->dirichlet_bc.insert(
      pair(1, new Functions::ZeroFunction<dim>(1)));
  }

  void
  set_boundary_descriptor_precursor() final
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // fill boundary descriptor velocity
    // no slip boundaries at lower and upper wall with ID=0
    this->boundary_descriptor_pre->velocity->dirichlet_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));

    // fill boundary descriptor pressure
    // no slip boundaries at lower and upper wall with ID=0
    this->boundary_descriptor_pre->pressure->neumann_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(
      new InitialSolutionVelocity<dim>(centerline_velocity,
                                       Geometry::LENGTH_CHANNEL,
                                       Geometry::HEIGHT_CHANNEL,
                                       Geometry::WIDTH_CHANNEL));
    //  this->field_functions->initial_solution_velocity.reset(new
    //  Functions::ZeroFunction<dim>(dim));
    this->field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    this->field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  void
  set_field_functions_precursor() final
  {
    this->field_functions_pre->initial_solution_velocity.reset(
      new InitialSolutionVelocity<dim>(centerline_velocity,
                                       Geometry::LENGTH_CHANNEL,
                                       Geometry::HEIGHT_CHANNEL,
                                       Geometry::WIDTH_CHANNEL));
    this->field_functions_pre->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    this->field_functions_pre->analytical_solution_pressure.reset(
      new Functions::ZeroFunction<dim>(1));
    this->field_functions_pre->right_hand_side.reset(new RightHandSide<dim>());
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    std::shared_ptr<PostProcessorBase<dim, Number>> pp;

    PostProcessorData<dim> pp_data;
    // write output for visualization of results
    pp_data.output_data.write_output       = this->write_output;
    pp_data.output_data.directory          = this->output_directory + "vtu/";
    pp_data.output_data.filename           = this->output_name;
    pp_data.output_data.start_time         = start_time;
    pp_data.output_data.interval_time      = (end_time - start_time) / 60;
    pp_data.output_data.write_divergence   = true;
    pp_data.output_data.write_q_criterion  = true;
    pp_data.output_data.write_boundary_IDs = true;
    pp_data.output_data.write_processor_id = true;
    pp_data.output_data.degree             = this->param.degree_u;
    pp_data.output_data.write_higher_order = false;

    PostProcessorDataBFS<dim> pp_data_bfs;
    pp_data_bfs.pp_data = pp_data;

    // line plot data: calculate statistics along lines
    pp_data_bfs.line_plot_data.line_data.directory                    = this->output_directory;
    pp_data_bfs.line_plot_data.statistics_data.calculate              = true;
    pp_data_bfs.line_plot_data.statistics_data.sample_start_time      = sample_start_time;
    pp_data_bfs.line_plot_data.statistics_data.sample_end_time        = end_time;
    pp_data_bfs.line_plot_data.statistics_data.sample_every_timesteps = sample_every_timesteps;
    pp_data_bfs.line_plot_data.statistics_data.write_output_every_timesteps =
      sample_every_timesteps * 10;

    // mean velocity
    std::shared_ptr<Quantity> quantity_velocity;
    quantity_velocity.reset(new Quantity());
    quantity_velocity->type = QuantityType::Velocity;

    // Reynolds stresses
    std::shared_ptr<Quantity> quantity_reynolds;
    quantity_reynolds.reset(new Quantity());
    quantity_reynolds->type = QuantityType::ReynoldsStresses;

    // skin friction
    Tensor<1, dim, double> normal;
    normal[1] = 1.0;
    Tensor<1, dim, double> tangent;
    tangent[0] = 1.0;
    std::shared_ptr<QuantitySkinFriction<dim>> quantity_skin_friction;
    quantity_skin_friction.reset(new QuantitySkinFriction<dim>());
    quantity_skin_friction->type           = QuantityType::SkinFriction;
    quantity_skin_friction->normal_vector  = normal;
    quantity_skin_friction->tangent_vector = tangent;
    quantity_skin_friction->viscosity      = viscosity;

    // mean pressure
    std::shared_ptr<Quantity> quantity_pressure;
    quantity_pressure.reset(new Quantity());
    quantity_pressure->type = QuantityType::Pressure;

    // mean pressure coefficient
    std::shared_ptr<QuantityPressureCoefficient<dim>> quantity_pressure_coeff;
    quantity_pressure_coeff.reset(new QuantityPressureCoefficient<dim>());
    quantity_pressure_coeff->type            = QuantityType::PressureCoefficient;
    quantity_pressure_coeff->reference_point = Point<dim>(Geometry::X1_COORDINATE_INFLOW, 0, 0);

    // lines
    std::shared_ptr<LineHomogeneousAveraging<dim>> vel_0, vel_1, vel_2, vel_3, vel_4, vel_5, vel_6,
      vel_7, vel_8, vel_9, vel_10, vel_11, Cp_1, Cp_2, Cf;
    vel_0.reset(new LineHomogeneousAveraging<dim>());
    vel_1.reset(new LineHomogeneousAveraging<dim>());
    vel_2.reset(new LineHomogeneousAveraging<dim>());
    vel_3.reset(new LineHomogeneousAveraging<dim>());
    vel_4.reset(new LineHomogeneousAveraging<dim>());
    vel_5.reset(new LineHomogeneousAveraging<dim>());
    vel_6.reset(new LineHomogeneousAveraging<dim>());
    vel_7.reset(new LineHomogeneousAveraging<dim>());
    vel_8.reset(new LineHomogeneousAveraging<dim>());
    vel_9.reset(new LineHomogeneousAveraging<dim>());
    vel_10.reset(new LineHomogeneousAveraging<dim>());
    vel_11.reset(new LineHomogeneousAveraging<dim>());
    Cp_1.reset(new LineHomogeneousAveraging<dim>());
    Cp_2.reset(new LineHomogeneousAveraging<dim>());
    Cf.reset(new LineHomogeneousAveraging<dim>());

    vel_0->average_homogeneous_direction  = true;
    vel_1->average_homogeneous_direction  = true;
    vel_2->average_homogeneous_direction  = true;
    vel_3->average_homogeneous_direction  = true;
    vel_4->average_homogeneous_direction  = true;
    vel_5->average_homogeneous_direction  = true;
    vel_6->average_homogeneous_direction  = true;
    vel_7->average_homogeneous_direction  = true;
    vel_8->average_homogeneous_direction  = true;
    vel_9->average_homogeneous_direction  = true;
    vel_10->average_homogeneous_direction = true;
    vel_11->average_homogeneous_direction = true;
    Cp_1->average_homogeneous_direction   = true;
    Cp_2->average_homogeneous_direction   = true;
    Cf->average_homogeneous_direction     = true;

    vel_0->averaging_direction  = 2;
    vel_1->averaging_direction  = 2;
    vel_2->averaging_direction  = 2;
    vel_3->averaging_direction  = 2;
    vel_4->averaging_direction  = 2;
    vel_5->averaging_direction  = 2;
    vel_6->averaging_direction  = 2;
    vel_7->averaging_direction  = 2;
    vel_8->averaging_direction  = 2;
    vel_9->averaging_direction  = 2;
    vel_10->averaging_direction = 2;
    vel_11->averaging_direction = 2;
    Cp_1->averaging_direction   = 2;
    Cp_2->averaging_direction   = 2;
    Cf->averaging_direction     = 2;

    // begin and end points of all lines
    double const H   = Geometry::H;
    double const eps = 1.e-8;
    vel_0->begin     = Point<dim>(Geometry::X1_COORDINATE_INFLOW + eps, 0, 0);
    vel_0->end       = Point<dim>(Geometry::X1_COORDINATE_INFLOW + eps, 2 * H, 0);
    vel_1->begin     = Point<dim>(0 * H, 0, 0);
    vel_1->end       = Point<dim>(0 * H, 2 * H, 0);
    vel_2->begin     = Point<dim>(1 * H, -1 * H, 0);
    vel_2->end       = Point<dim>(1 * H, 2 * H, 0);
    vel_3->begin     = Point<dim>(2 * H, -1 * H, 0);
    vel_3->end       = Point<dim>(2 * H, 2 * H, 0);
    vel_4->begin     = Point<dim>(3 * H, -1 * H, 0);
    vel_4->end       = Point<dim>(3 * H, 2 * H, 0);
    vel_5->begin     = Point<dim>(4 * H, -1 * H, 0);
    vel_5->end       = Point<dim>(4 * H, 2 * H, 0);
    vel_6->begin     = Point<dim>(5 * H, -1 * H, 0);
    vel_6->end       = Point<dim>(5 * H, 2 * H, 0);
    vel_7->begin     = Point<dim>(6 * H, -1 * H, 0);
    vel_7->end       = Point<dim>(6 * H, 2 * H, 0);
    vel_8->begin     = Point<dim>(7 * H, -1 * H, 0);
    vel_8->end       = Point<dim>(7 * H, 2 * H, 0);
    vel_9->begin     = Point<dim>(8 * H, -1 * H, 0);
    vel_9->end       = Point<dim>(8 * H, 2 * H, 0);
    vel_10->begin    = Point<dim>(9 * H, -1 * H, 0);
    vel_10->end      = Point<dim>(9 * H, 2 * H, 0);
    vel_11->begin    = Point<dim>(10 * H, -1 * H, 0);
    vel_11->end      = Point<dim>(10 * H, 2 * H, 0);
    Cp_1->begin      = Point<dim>(Geometry::X1_COORDINATE_INFLOW, 0, 0);
    Cp_1->end        = Point<dim>(0, 0, 0);
    Cp_2->begin      = Point<dim>(0, -H, 0);
    Cp_2->end        = Point<dim>(Geometry::X1_COORDINATE_OUTFLOW, -H, 0);
    Cf->begin        = Point<dim>(0, -H, 0);
    Cf->end          = Point<dim>(Geometry::X1_COORDINATE_OUTFLOW, -H, 0);

    // set the number of points along the lines
    vel_0->n_points  = n_points_per_line;
    vel_1->n_points  = n_points_per_line;
    vel_2->n_points  = n_points_per_line;
    vel_3->n_points  = n_points_per_line;
    vel_4->n_points  = n_points_per_line;
    vel_5->n_points  = n_points_per_line;
    vel_6->n_points  = n_points_per_line;
    vel_7->n_points  = n_points_per_line;
    vel_8->n_points  = n_points_per_line;
    vel_9->n_points  = n_points_per_line;
    vel_10->n_points = n_points_per_line;
    vel_11->n_points = n_points_per_line;
    Cp_1->n_points   = n_points_per_line;
    Cp_2->n_points   = n_points_per_line;
    Cf->n_points     = n_points_per_line;

    // set the quantities that we want to compute along the lines
    vel_0->quantities.push_back(quantity_velocity);
    vel_0->quantities.push_back(quantity_pressure);
    vel_0->quantities.push_back(quantity_reynolds);
    vel_1->quantities.push_back(quantity_velocity);
    vel_1->quantities.push_back(quantity_pressure);
    vel_1->quantities.push_back(quantity_reynolds);
    vel_2->quantities.push_back(quantity_velocity);
    vel_2->quantities.push_back(quantity_pressure);
    vel_2->quantities.push_back(quantity_reynolds);
    vel_3->quantities.push_back(quantity_velocity);
    vel_3->quantities.push_back(quantity_pressure);
    vel_3->quantities.push_back(quantity_reynolds);
    vel_4->quantities.push_back(quantity_velocity);
    vel_4->quantities.push_back(quantity_pressure);
    vel_4->quantities.push_back(quantity_reynolds);
    vel_5->quantities.push_back(quantity_velocity);
    vel_5->quantities.push_back(quantity_pressure);
    vel_5->quantities.push_back(quantity_reynolds);
    vel_6->quantities.push_back(quantity_velocity);
    vel_6->quantities.push_back(quantity_pressure);
    vel_6->quantities.push_back(quantity_reynolds);
    vel_7->quantities.push_back(quantity_velocity);
    vel_7->quantities.push_back(quantity_pressure);
    vel_7->quantities.push_back(quantity_reynolds);
    vel_8->quantities.push_back(quantity_velocity);
    vel_8->quantities.push_back(quantity_pressure);
    vel_8->quantities.push_back(quantity_reynolds);
    vel_9->quantities.push_back(quantity_velocity);
    vel_9->quantities.push_back(quantity_pressure);
    vel_9->quantities.push_back(quantity_reynolds);
    vel_10->quantities.push_back(quantity_velocity);
    vel_10->quantities.push_back(quantity_pressure);
    vel_10->quantities.push_back(quantity_reynolds);
    vel_11->quantities.push_back(quantity_velocity);
    vel_11->quantities.push_back(quantity_pressure);
    vel_11->quantities.push_back(quantity_reynolds);
    Cp_1->quantities.push_back(quantity_pressure);
    Cp_1->quantities.push_back(quantity_pressure_coeff);
    Cp_2->quantities.push_back(quantity_pressure);
    Cp_2->quantities.push_back(quantity_pressure_coeff);
    Cf->quantities.push_back(quantity_skin_friction);

    // set line names
    vel_0->name  = "vel_0";
    vel_1->name  = "vel_1";
    vel_2->name  = "vel_2";
    vel_3->name  = "vel_3";
    vel_4->name  = "vel_4";
    vel_5->name  = "vel_5";
    vel_6->name  = "vel_6";
    vel_7->name  = "vel_7";
    vel_8->name  = "vel_8";
    vel_9->name  = "vel_9";
    vel_10->name = "vel_10";
    vel_11->name = "vel_11";
    Cp_1->name   = "Cp_1";
    Cp_2->name   = "Cp_2";
    Cf->name     = "Cf";

    // insert lines
    pp_data_bfs.line_plot_data.line_data.lines.push_back(vel_0);
    pp_data_bfs.line_plot_data.line_data.lines.push_back(vel_1);
    pp_data_bfs.line_plot_data.line_data.lines.push_back(vel_2);
    pp_data_bfs.line_plot_data.line_data.lines.push_back(vel_3);
    pp_data_bfs.line_plot_data.line_data.lines.push_back(vel_4);
    pp_data_bfs.line_plot_data.line_data.lines.push_back(vel_5);
    pp_data_bfs.line_plot_data.line_data.lines.push_back(vel_6);
    pp_data_bfs.line_plot_data.line_data.lines.push_back(vel_7);
    pp_data_bfs.line_plot_data.line_data.lines.push_back(vel_8);
    pp_data_bfs.line_plot_data.line_data.lines.push_back(vel_9);
    pp_data_bfs.line_plot_data.line_data.lines.push_back(vel_10);
    pp_data_bfs.line_plot_data.line_data.lines.push_back(vel_11);
    pp_data_bfs.line_plot_data.line_data.lines.push_back(Cp_1);
    pp_data_bfs.line_plot_data.line_data.lines.push_back(Cp_2);
    pp_data_bfs.line_plot_data.line_data.lines.push_back(Cf);

    pp.reset(new PostProcessorBFS<dim, Number>(pp_data_bfs, this->mpi_comm));

    return pp;
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor_precursor() final
  {
    std::shared_ptr<PostProcessorBase<dim, Number>> pp;

    PostProcessorData<dim> pp_data;
    // write output for visualization of results
    pp_data.output_data.write_output       = this->write_output;
    pp_data.output_data.directory          = this->output_directory + "vtu/";
    pp_data.output_data.filename           = this->output_name + "_precursor";
    pp_data.output_data.start_time         = start_time;
    pp_data.output_data.interval_time      = (end_time - start_time) / 60;
    pp_data.output_data.write_divergence   = true;
    pp_data.output_data.write_q_criterion  = true;
    pp_data.output_data.write_boundary_IDs = true;
    pp_data.output_data.write_processor_id = true;
    pp_data.output_data.degree             = this->param_pre.degree_u;
    pp_data.output_data.write_higher_order = false;

    PostProcessorDataBFS<dim> pp_data_bfs;
    pp_data_bfs.pp_data = pp_data;

    // turbulent channel statistics
    pp_data_bfs.turb_ch_data.calculate              = true;
    pp_data_bfs.turb_ch_data.cells_are_stretched    = Geometry::use_grid_stretching_in_y_direction;
    pp_data_bfs.turb_ch_data.sample_start_time      = sample_start_time;
    pp_data_bfs.turb_ch_data.sample_end_time        = end_time;
    pp_data_bfs.turb_ch_data.sample_every_timesteps = sample_every_timesteps;
    pp_data_bfs.turb_ch_data.viscosity              = viscosity;
    pp_data_bfs.turb_ch_data.directory              = this->output_directory;
    pp_data_bfs.turb_ch_data.filename               = this->output_name + "_precursor";

    // use turbulent channel data to prescribe inflow velocity for BFS
    pp_data_bfs.inflow_data.write_inflow_data = true;
    pp_data_bfs.inflow_data.normal_direction  = 0; /* x-direction */
    pp_data_bfs.inflow_data.normal_coordinate = Geometry::X1_COORDINATE_OUTFLOW_CHANNEL;
    pp_data_bfs.inflow_data.n_points_y        = inflow_data_storage->n_points_y;
    pp_data_bfs.inflow_data.n_points_z        = inflow_data_storage->n_points_z;
    pp_data_bfs.inflow_data.y_values          = &inflow_data_storage->y_values;
    pp_data_bfs.inflow_data.z_values          = &inflow_data_storage->z_values;
    pp_data_bfs.inflow_data.array             = &inflow_data_storage->velocity_values;

    pp.reset(new PostProcessorBFS<dim, Number>(pp_data_bfs, this->mpi_comm));

    return pp;
  }
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application_precursor.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_ */
