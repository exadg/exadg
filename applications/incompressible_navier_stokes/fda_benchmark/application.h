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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FDA_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FDA_H_

// ExaDG
#include <exadg/functions_and_boundary_conditions/linear_interpolation.h>

// FDA nozzle benchmark application
#include "include/flow_rate_controller.h"
#include "include/grid.h"
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
  InitialSolutionVelocity(double const max_velocity)
    : Function<dim>(dim, 0.0), max_velocity(max_velocity)
  {
    srand(0); // initialize rand() to obtain reproducible results
  }

  double
  value(Point<dim> const & p, unsigned int const component = 0) const
  {
    AssertThrow(dim == 3, ExcMessage("Dimension has to be dim==3."));

    double result = 0.0;

    // flow in z-direction
    if(component == 2)
    {
      // assume parabolic profile u(r) = u_max * [1-(r/R)^2]
      //  -> u_max = 2 * u_mean = 2 * flow_rate / area
      double const R = FDANozzle::radius_function(p[2]);
      double const r = std::min(std::sqrt(p[0] * p[0] + p[1] * p[1]), R);

      // parabolic velocity profile
      double const max_velocity_z = max_velocity * std::pow(FDANozzle::R_OUTER / R, 2.0);

      result = max_velocity_z * (1.0 - pow(r / R, 2.0));

      // Add perturbation (sine + random) for the precursor to initiate
      // a turbulent flow in case the Reynolds number is large enough
      // (otherwise, the perturbations will be damped and the flow becomes laminar).
      // According to first numerical results, the perturbed flow returns to a laminar
      // steady state in the precursor domain for Reynolds numbers Re_t = 500, 2000,
      // 3500, 5000, and 6500.
      if(p[2] <= FDANozzle::Z2_PRECURSOR)
      {
        double const phi    = std::atan2(p[1], p[0]);
        double const factor = 0.5;

        double perturbation = factor * max_velocity_z * std::sin(4.0 * phi) *
                                std::sin(8.0 * numbers::PI * p[2] / FDANozzle::LENGTH_PRECURSOR) +
                              factor * max_velocity_z * ((double)rand() / RAND_MAX - 0.5) / 0.5;

        // the perturbations should fulfill the Dirichlet boundary conditions
        perturbation *= (1.0 - pow(r / R, 6.0));

        result += perturbation;
      }
    }

    return result;
  }

private:
  double const max_velocity;
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
    // compute polar coordinates (r, phi) from point p
    // given in Cartesian coordinates (x, y) = inflow plane
    double const r   = std::sqrt(p[0] * p[0] + p[1] * p[1]);
    double const phi = std::atan2(p[1], p[0]);

    double const result = linear_interpolation_2d_cylindrical(
      r, phi, data.r_values, data.phi_values, data.velocity_values, component);

    return result;
  }

private:
  InflowDataStorage<dim> const & data;
};


/*
 *  Right-hand side function: Implements the body force vector occurring on the
 *  right-hand side of the momentum equation of the Navier-Stokes equations.
 *  Only relevant for precursor simulation.
 */
template<int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide(FlowRateController const & flow_rate_controller)
    : Function<dim>(dim, 0.0), flow_rate_controller(flow_rate_controller)
  {
  }

  double
  value(Point<dim> const & /*p*/, unsigned int const component = 0) const
  {
    double result = 0.0;

    // Channel flow with periodic BCs in z-direction:
    // The flow is driven by a body force in z-direction
    if(component == 2)
    {
      result = flow_rate_controller.get_body_force();
    }

    return result;
  }

private:
  FlowRateController const & flow_rate_controller;
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

    flow_rate_controller.reset(new FlowRateController(target_flow_rate,
                                                      viscosity,
                                                      max_velocity,
                                                      FDANozzle::R_OUTER,
                                                      mean_velocity_inflow,
                                                      FDANozzle::D,
                                                      start_time_precursor));

    // compute number of points for inflow data array depending on spatial resolution of problem
    unsigned int n_points = 1;
    {
      unsigned int degree       = 1;
      unsigned int refine_space = 0;

      ParameterHandler prm;
      // clang-format off
      prm.enter_subsection("General");
        prm.add_parameter("DegreeMin",      degree,       "Polynomial degree of shape functions.", Patterns::Integer(1,15));
        prm.add_parameter("RefineSpaceMin", refine_space, "Number of mesh refinements.",           Patterns::Integer(0,20));
      prm.leave_subsection();
      // clang-format on
      prm.parse_input(input_file, "", true, true);

      n_points = 20 * (degree + 1) * Utilities::pow(2, refine_space);
    }

    inflow_data_storage.reset(new InflowDataStorage<dim>(n_points,
                                                         FDANozzle::R_OUTER,
                                                         max_velocity,
                                                         use_random_perturbations,
                                                         factor_random_perturbations));
  }

  // set the throat Reynolds number Re_throat = U_{mean,throat} * (2 R_throat) / nu
  double const Re = 3500; // 500; //2000; //3500; //5000; //6500; //8000;

  // kinematic viscosity (same viscosity for all Reynolds numbers)
  double const viscosity = 3.31e-6;

  double const area_inflow = FDANozzle::R_OUTER * FDANozzle::R_OUTER * numbers::PI;
  double const area_throat = FDANozzle::R_INNER * FDANozzle::R_INNER * numbers::PI;

  double const mean_velocity_throat = Re * viscosity / (2.0 * FDANozzle::R_INNER);
  double const target_flow_rate     = mean_velocity_throat * area_throat;
  double const mean_velocity_inflow = target_flow_rate / area_inflow;

  double const max_velocity     = 2.0 * target_flow_rate / area_inflow;
  double const max_velocity_cfl = 2.0 * target_flow_rate / area_throat;

  // prescribe velocity inflow profile for nozzle domain via precursor simulation?
  // If yes, specify additional mesh_refinements for precursor domain
  bool const         use_precursor                    = true;
  unsigned int const additional_refinements_precursor = 1;

  // use prescribed velocity profile at inflow superimposed by random perturbations (white noise)?
  // If yes, specify amplitude of perturbations relative to maximum velocity on centerline.
  // Can be used with and without precursor approach
  bool const   use_random_perturbations    = false;
  double const factor_random_perturbations = 0.02;

  std::shared_ptr<FlowRateController>     flow_rate_controller;
  std::shared_ptr<InflowDataStorage<dim>> inflow_data_storage;

  // start and end time

  // estimation of flow-through time T_0 (through nozzle section)
  // based on the mean velocity through throat
  double const T_0                  = FDANozzle::LENGTH_THROAT / mean_velocity_throat;
  double const start_time_precursor = -500.0 * T_0; // let the flow develop
  double const start_time_nozzle    = 0.0 * T_0;
  double const end_time             = 250.0 * T_0; // 150.0*T_0;

  // postprocessing

  // output folder
  std::string const directory = "output/fda/Re3500/";

  // flow-rate
  std::string const filename_flowrate = "precursor_mean_velocity";

  // sampling of axial and radial velocity profiles

  // sampling interval should last over (100-200) * T_0 according to preliminary results.
  double const       sample_start_time      = 50.0 * T_0; // let the flow develop
  double const       sample_end_time        = end_time;   // that's the only reasonable choice
  unsigned int const sample_every_timesteps = 1;

  // line plot data
  unsigned int const n_points_line_axial           = 400;
  unsigned int const n_points_line_radial          = 64;
  unsigned int const n_points_line_circumferential = 32;

  // vtu-output
  double const output_start_time_precursor = start_time_precursor;
  double const output_start_time_nozzle    = start_time_nozzle;
  double const output_interval_time        = 5.0 * T_0; // 10.0*T_0;

  /*
   *  Most of the parameters are the same for both domains, so we write
   *  this function for the actual domain and only "correct" the parameters
   *  for the precursor by passing an additional parameter is_precursor.
   */
  void
  do_set_parameters(Parameters & param, bool const is_precursor = false)
  {
    // MATHEMATICAL MODEL
    param.problem_type                   = ProblemType::Unsteady;
    param.equation_type                  = EquationType::NavierStokes;
    param.use_outflow_bc_convective_term = true;
    param.formulation_viscous_term       = FormulationViscousTerm::LaplaceFormulation;
    param.formulation_convective_term    = FormulationConvectiveTerm::DivergenceFormulation;
    param.right_hand_side                = true;


    // PHYSICAL QUANTITIES
    param.start_time = start_time_nozzle;
    if(is_precursor)
      param.start_time = start_time_precursor;

    param.end_time  = end_time;
    param.viscosity = viscosity;


    // TEMPORAL DISCRETIZATION
    param.solver_type = SolverType::Unsteady;

    //  param.temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
    //  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
    //  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    //  param.adaptive_time_stepping = true;
    param.temporal_discretization                = TemporalDiscretization::BDFPressureCorrection;
    param.treatment_of_convective_term           = TreatmentOfConvectiveTerm::Implicit;
    param.calculation_of_time_step_size          = TimeStepCalculation::CFL;
    param.adaptive_time_stepping_limiting_factor = 3.0;
    param.max_velocity                           = max_velocity_cfl;
    param.cfl                                    = 4.0;
    param.cfl_exponent_fe_degree_velocity        = 1.5;
    param.time_step_size                         = 1.0e-1;
    param.order_time_integrator                  = 2;
    param.start_with_low_order                   = true;

    // output of solver information
    param.solver_info_data.interval_time = T_0;


    // SPATIAL DISCRETIZATION
    param.grid.triangulation_type = TriangulationType::Distributed;
    param.grid.mapping_degree     = param.degree_u;
    param.degree_p                = DegreePressure::MixedOrder;

    // convective term
    param.upwind_factor = 1.0;

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;
    param.IP_factor_viscous      = 1.0;

    // div-div and continuity penalty terms
    param.use_divergence_penalty                     = true;
    param.divergence_penalty_factor                  = 1.0e0;
    param.use_continuity_penalty                     = true;
    param.continuity_penalty_factor                  = param.divergence_penalty_factor;
    param.apply_penalty_terms_in_postprocessing_step = true;

    // TURBULENCE
    param.use_turbulence_model = false;
    param.turbulence_model     = TurbulenceEddyViscosityModel::Sigma;
    // Smagorinsky: 0.165, Vreman: 0.28, WALE: 0.50, Sigma: 1.35
    param.turbulence_model_constant = 1.35;

    // PROJECTION METHODS

    // pressure Poisson equation
    param.IP_factor_pressure                   = 1.0;
    param.solver_data_pressure_poisson         = SolverData(1000, 1.e-12, 1.e-3, 100);
    param.solver_pressure_poisson              = SolverPressurePoisson::CG; // FGMRES;
    param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
    if(is_precursor)
      param.multigrid_data_pressure_poisson.type = MultigridType::phMG;
    param.multigrid_data_pressure_poisson.smoother_data.smoother   = MultigridSmoother::Chebyshev;
    param.multigrid_data_pressure_poisson.smoother_data.iterations = 5;
    param.multigrid_data_pressure_poisson.coarse_problem.solver    = MultigridCoarseGridSolver::CG;
    param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;


    // projection step
    param.solver_projection                = SolverProjection::CG;
    param.solver_data_projection           = SolverData(1000, 1.e-12, 1.e-3);
    param.preconditioner_projection        = PreconditionerProjection::InverseMassMatrix;
    param.update_preconditioner_projection = true;


    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // viscous step
    param.solver_viscous         = SolverViscous::CG;
    param.solver_data_viscous    = SolverData(1000, 1.e-12, 1.e-3);
    param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;


    // PRESSURE-CORRECTION SCHEME

    // formulation
    param.order_pressure_extrapolation = 1;    // use 0 for non-incremental formulation
    param.rotational_formulation       = true; // use false for standard formulation

    // momentum step

    // Newton solver
    param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-12, 1.e-3);

    // linear solver
    if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      param.solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-1, 100);
    else
      param.solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-3, 100);

    param.solver_momentum                = SolverMomentum::GMRES;
    param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
    param.update_preconditioner_momentum = false;

    // COUPLED NAVIER-STOKES SOLVER
    param.use_scaling_continuity = false;

    // nonlinear solver (Newton solver)
    param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-20, 1.e-3);

    // linear solver
    param.solver_coupled = SolverCoupled::GMRES; // GMRES; //FGMRES;
    if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-1, 100);
    else
      param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-3, 100);

    // preconditioning linear solver
    param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    param.update_preconditioner_coupled = false;

    // preconditioner velocity/momentum block
    param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block =
      SchurComplementPreconditioner::CahouetChabard; // PressureConvectionDiffusion;

    // Chebyshev moother
    param.multigrid_data_pressure_block.smoother_data.smoother = MultigridSmoother::Chebyshev;
    param.multigrid_data_pressure_block.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev;
  }

  void
  set_parameters() final
  {
    do_set_parameters(this->param);
  }

  void
  set_parameters_precursor() final
  {
    do_set_parameters(this->param_pre, true);
  }

  void
  create_grid() final
  {
    FDANozzle::create_grid_and_set_boundary_ids_nozzle(this->grid->triangulation,
                                                       this->param.grid.n_refine_global,
                                                       this->grid->periodic_faces);
  }

  void
  create_grid_precursor() final
  {
    Triangulation<2> tria_2d;
    GridGenerator::hyper_ball(tria_2d, Point<2>(), FDANozzle::R_OUTER);
    GridGenerator::extrude_triangulation(tria_2d,
                                         FDANozzle::N_CELLS_AXIAL_PRECURSOR + 1,
                                         FDANozzle::LENGTH_PRECURSOR,
                                         *this->grid_pre->triangulation);
    Tensor<1, dim> offset = Tensor<1, dim>();
    offset[2]             = FDANozzle::Z1_PRECURSOR;
    GridTools::shift(offset, *this->grid_pre->triangulation);

    /*
     *  MANIFOLDS
     */
    this->grid_pre->triangulation->set_all_manifold_ids(0);

    // first fill vectors of manifold_ids and face_ids
    std::vector<unsigned int> manifold_ids;
    std::vector<unsigned int> face_ids;

    for(auto cell : this->grid_pre->triangulation->active_cell_iterators())
    {
      for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      {
        bool face_at_sphere_boundary = true;
        for(unsigned int v = 0; v < GeometryInfo<dim - 1>::vertices_per_cell; ++v)
        {
          Point<dim> point = Point<dim>(0, 0, cell->face(f)->vertex(v)[2]);

          if(std::abs((cell->face(f)->vertex(v) - point).norm() - FDANozzle::R_OUTER) > 1e-12)
            face_at_sphere_boundary = false;
        }
        if(face_at_sphere_boundary)
        {
          face_ids.push_back(f);
          unsigned int manifold_id = manifold_ids.size() + 1;
          cell->set_all_manifold_ids(manifold_id);
          manifold_ids.push_back(manifold_id);
        }
      }
    }

    // generate vector of manifolds and apply manifold to all cells that have been marked
    static std::vector<std::shared_ptr<Manifold<dim>>> manifold_vec;
    manifold_vec.resize(manifold_ids.size());

    for(unsigned int i = 0; i < manifold_ids.size(); ++i)
    {
      for(auto cell : this->grid_pre->triangulation->active_cell_iterators())
      {
        if(cell->manifold_id() == manifold_ids[i])
        {
          manifold_vec[i] = std::shared_ptr<Manifold<dim>>(static_cast<Manifold<dim> *>(
            new OneSidedCylindricalManifold<dim>(cell, face_ids[i], Point<dim>())));
          this->grid_pre->triangulation->set_manifold(manifold_ids[i], *(manifold_vec[i]));
        }
      }
    }

    /*
     *  BOUNDARY ID's
     */
    for(auto cell : this->grid_pre->triangulation->active_cell_iterators())
    {
      for(unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      {
        // left boundary
        if((std::fabs(cell->face(face)->center()[2] - FDANozzle::Z1_PRECURSOR) < 1e-12))
        {
          cell->face(face)->set_boundary_id(0 + 10);
        }

        // right boundary
        if((std::fabs(cell->face(face)->center()[2] - FDANozzle::Z2_PRECURSOR) < 1e-12))
        {
          cell->face(face)->set_boundary_id(1 + 10);
        }
      }
    }

    GridTools::collect_periodic_faces(
      *this->grid->triangulation, 0 + 10, 1 + 10, 2, this->grid_pre->periodic_faces);
    this->grid_pre->triangulation->add_periodicity(this->grid_pre->periodic_faces);

    // perform global refinements
    this->grid_pre->triangulation->refine_global(this->param_pre.grid.n_refine_global +
                                                 additional_refinements_precursor);
  }

  void
  set_boundary_descriptor() final
  {
    /*
     *  FILL BOUNDARY DESCRIPTORS
     */
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // fill boundary descriptor velocity
    // no slip boundaries at the upper and lower wall with ID=0
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));

    // inflow boundary condition at left boundary with ID=1: prescribe velocity profile which
    // is obtained as the results of the simulation on DOMAIN 1
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(1, new InflowProfile<dim>(*inflow_data_storage)));

    // outflow boundary condition at right boundary with ID=2
    this->boundary_descriptor->velocity->neumann_bc.insert(
      pair(2, new Functions::ZeroFunction<dim>(dim)));

    // fill boundary descriptor pressure
    // no slip boundaries at the upper and lower wall with ID=0
    this->boundary_descriptor->pressure->neumann_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));

    // inflow boundary condition at left boundary with ID=1
    // the inflow boundary condition is time dependent (du/dt != 0) but, for simplicity,
    // we assume that this is negligible when using the dual splitting scheme
    this->boundary_descriptor->pressure->neumann_bc.insert(
      pair(1, new Functions::ZeroFunction<dim>(dim)));

    // outflow boundary condition at right boundary with ID=2: set pressure to zero
    this->boundary_descriptor->pressure->dirichlet_bc.insert(
      pair(2, new Functions::ZeroFunction<dim>(1)));
  }

  void
  set_boundary_descriptor_precursor() final
  {
    /*
     *  FILL BOUNDARY DESCRIPTORS
     */
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
      new InitialSolutionVelocity<dim>(max_velocity));
    this->field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    this->field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  void
  set_field_functions_precursor() final
  {
    this->field_functions_pre->initial_solution_velocity.reset(
      new InitialSolutionVelocity<dim>(max_velocity));
    this->field_functions_pre->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    this->field_functions_pre->analytical_solution_pressure.reset(
      new Functions::ZeroFunction<dim>(1));
    // prescribe body force for the turbulent pipe flow (precursor) to adjust the desired flow rate
    this->field_functions_pre->right_hand_side.reset(new RightHandSide<dim>(*flow_rate_controller));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    std::shared_ptr<PostProcessorBase<dim, Number>> pp;

    // write output for visualization of results
    PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output                         = this->write_output;
    pp_data.output_data.directory                            = this->output_directory + "vtu/";
    pp_data.output_data.filename                             = this->output_name + "_nozzle";
    pp_data.output_data.start_time                           = output_start_time_nozzle;
    pp_data.output_data.interval_time                        = output_interval_time;
    pp_data.output_data.write_divergence                     = true;
    pp_data.output_data.write_processor_id                   = true;
    pp_data.output_data.mean_velocity.calculate              = true;
    pp_data.output_data.mean_velocity.sample_start_time      = sample_start_time;
    pp_data.output_data.mean_velocity.sample_end_time        = sample_end_time;
    pp_data.output_data.mean_velocity.sample_every_timesteps = 1;
    pp_data.output_data.degree                               = this->param.degree_u;

    PostProcessorDataFDA<dim> pp_data_fda;
    pp_data_fda.pp_data = pp_data;

    // evaluation of quantities along lines
    pp_data_fda.line_plot_data.line_data.directory                    = this->output_directory;
    pp_data_fda.line_plot_data.statistics_data.calculate              = true;
    pp_data_fda.line_plot_data.statistics_data.sample_start_time      = sample_start_time;
    pp_data_fda.line_plot_data.statistics_data.sample_end_time        = end_time;
    pp_data_fda.line_plot_data.statistics_data.sample_every_timesteps = sample_every_timesteps;
    pp_data_fda.line_plot_data.statistics_data.write_output_every_timesteps =
      sample_every_timesteps * 100;

    // lines
    std::shared_ptr<LineCircumferentialAveraging<dim>> axial_profile, radial_profile_z1,
      radial_profile_z2, radial_profile_z3, radial_profile_z4, radial_profile_z5, radial_profile_z6,
      radial_profile_z7, radial_profile_z8, radial_profile_z9, radial_profile_z10,
      radial_profile_z11, radial_profile_z12;

    axial_profile.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z1.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z2.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z3.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z4.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z5.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z6.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z7.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z8.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z9.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z10.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z11.reset(new LineCircumferentialAveraging<dim>());
    radial_profile_z12.reset(new LineCircumferentialAveraging<dim>());

    double z_1 = -0.088, z_2 = -0.064, z_3 = -0.048, z_4 = -0.02, z_5 = -0.008, z_6 = 0.0,
           z_7 = 0.008, z_8 = 0.016, z_9 = 0.024, z_10 = 0.032, z_11 = 0.06, z_12 = 0.08;

    // begin and end points of all lines
    axial_profile->begin      = Point<dim>(0, 0, FDANozzle::Z1_INFLOW);
    axial_profile->end        = Point<dim>(0, 0, FDANozzle::Z2_OUTFLOW);
    radial_profile_z1->begin  = Point<dim>(0, 0, z_1);
    radial_profile_z1->end    = Point<dim>(FDANozzle::radius_function(z_1), 0, z_1);
    radial_profile_z2->begin  = Point<dim>(0, 0, z_2);
    radial_profile_z2->end    = Point<dim>(FDANozzle::radius_function(z_2), 0, z_2);
    radial_profile_z3->begin  = Point<dim>(0, 0, z_3);
    radial_profile_z3->end    = Point<dim>(FDANozzle::radius_function(z_3), 0, z_3);
    radial_profile_z4->begin  = Point<dim>(0, 0, z_4);
    radial_profile_z4->end    = Point<dim>(FDANozzle::radius_function(z_4), 0, z_4);
    radial_profile_z5->begin  = Point<dim>(0, 0, z_5);
    radial_profile_z5->end    = Point<dim>(FDANozzle::radius_function(z_5), 0, z_5);
    radial_profile_z6->begin  = Point<dim>(0, 0, z_6);
    radial_profile_z6->end    = Point<dim>(FDANozzle::radius_function(z_6), 0, z_6);
    radial_profile_z7->begin  = Point<dim>(0, 0, z_7);
    radial_profile_z7->end    = Point<dim>(FDANozzle::radius_function(z_7), 0, z_7);
    radial_profile_z8->begin  = Point<dim>(0, 0, z_8);
    radial_profile_z8->end    = Point<dim>(FDANozzle::radius_function(z_8), 0, z_8);
    radial_profile_z9->begin  = Point<dim>(0, 0, z_9);
    radial_profile_z9->end    = Point<dim>(FDANozzle::radius_function(z_9), 0, z_9);
    radial_profile_z10->begin = Point<dim>(0, 0, z_10);
    radial_profile_z10->end   = Point<dim>(FDANozzle::radius_function(z_10), 0, z_10);
    radial_profile_z11->begin = Point<dim>(0, 0, z_11);
    radial_profile_z11->end   = Point<dim>(FDANozzle::radius_function(z_11), 0, z_11);
    radial_profile_z12->begin = Point<dim>(0, 0, z_12);
    radial_profile_z12->end   = Point<dim>(FDANozzle::radius_function(z_12), 0, z_12);

    // number of points
    axial_profile->n_points      = n_points_line_axial;
    radial_profile_z1->n_points  = n_points_line_radial;
    radial_profile_z2->n_points  = n_points_line_radial;
    radial_profile_z3->n_points  = n_points_line_radial;
    radial_profile_z4->n_points  = n_points_line_radial;
    radial_profile_z5->n_points  = n_points_line_radial;
    radial_profile_z6->n_points  = n_points_line_radial;
    radial_profile_z7->n_points  = n_points_line_radial;
    radial_profile_z8->n_points  = n_points_line_radial;
    radial_profile_z9->n_points  = n_points_line_radial;
    radial_profile_z10->n_points = n_points_line_radial;
    radial_profile_z11->n_points = n_points_line_radial;
    radial_profile_z12->n_points = n_points_line_radial;

    axial_profile->average_circumferential      = false;
    radial_profile_z1->average_circumferential  = true;
    radial_profile_z2->average_circumferential  = true;
    radial_profile_z3->average_circumferential  = true;
    radial_profile_z4->average_circumferential  = true;
    radial_profile_z5->average_circumferential  = true;
    radial_profile_z6->average_circumferential  = true;
    radial_profile_z7->average_circumferential  = true;
    radial_profile_z8->average_circumferential  = true;
    radial_profile_z9->average_circumferential  = true;
    radial_profile_z10->average_circumferential = true;
    radial_profile_z11->average_circumferential = true;
    radial_profile_z12->average_circumferential = true;

    radial_profile_z1->n_points_circumferential  = n_points_line_circumferential;
    radial_profile_z2->n_points_circumferential  = n_points_line_circumferential;
    radial_profile_z3->n_points_circumferential  = n_points_line_circumferential;
    radial_profile_z4->n_points_circumferential  = n_points_line_circumferential;
    radial_profile_z5->n_points_circumferential  = n_points_line_circumferential;
    radial_profile_z6->n_points_circumferential  = n_points_line_circumferential;
    radial_profile_z7->n_points_circumferential  = n_points_line_circumferential;
    radial_profile_z8->n_points_circumferential  = n_points_line_circumferential;
    radial_profile_z9->n_points_circumferential  = n_points_line_circumferential;
    radial_profile_z10->n_points_circumferential = n_points_line_circumferential;
    radial_profile_z11->n_points_circumferential = n_points_line_circumferential;
    radial_profile_z12->n_points_circumferential = n_points_line_circumferential;

    Tensor<1, dim, double> normal;
    normal[2]                         = 1.0;
    radial_profile_z1->normal_vector  = normal;
    radial_profile_z2->normal_vector  = normal;
    radial_profile_z3->normal_vector  = normal;
    radial_profile_z4->normal_vector  = normal;
    radial_profile_z5->normal_vector  = normal;
    radial_profile_z6->normal_vector  = normal;
    radial_profile_z7->normal_vector  = normal;
    radial_profile_z8->normal_vector  = normal;
    radial_profile_z9->normal_vector  = normal;
    radial_profile_z10->normal_vector = normal;
    radial_profile_z11->normal_vector = normal;
    radial_profile_z12->normal_vector = normal;

    // quantities

    // no additional averaging in space for centerline velocity
    std::shared_ptr<Quantity> quantity_velocity;
    quantity_velocity.reset(new Quantity());
    quantity_velocity->type = QuantityType::Velocity;

    axial_profile->quantities.push_back(quantity_velocity);
    radial_profile_z1->quantities.push_back(quantity_velocity);
    radial_profile_z2->quantities.push_back(quantity_velocity);
    radial_profile_z3->quantities.push_back(quantity_velocity);
    radial_profile_z4->quantities.push_back(quantity_velocity);
    radial_profile_z5->quantities.push_back(quantity_velocity);
    radial_profile_z6->quantities.push_back(quantity_velocity);
    radial_profile_z7->quantities.push_back(quantity_velocity);
    radial_profile_z8->quantities.push_back(quantity_velocity);
    radial_profile_z9->quantities.push_back(quantity_velocity);
    radial_profile_z10->quantities.push_back(quantity_velocity);
    radial_profile_z11->quantities.push_back(quantity_velocity);
    radial_profile_z12->quantities.push_back(quantity_velocity);

    // names
    axial_profile->name      = "axial_profile";
    radial_profile_z1->name  = "radial_profile_z1";
    radial_profile_z2->name  = "radial_profile_z2";
    radial_profile_z3->name  = "radial_profile_z3";
    radial_profile_z4->name  = "radial_profile_z4";
    radial_profile_z5->name  = "radial_profile_z5";
    radial_profile_z6->name  = "radial_profile_z6";
    radial_profile_z7->name  = "radial_profile_z7";
    radial_profile_z8->name  = "radial_profile_z8";
    radial_profile_z9->name  = "radial_profile_z9";
    radial_profile_z10->name = "radial_profile_z10";
    radial_profile_z11->name = "radial_profile_z11";
    radial_profile_z12->name = "radial_profile_z12";

    // insert lines
    pp_data_fda.line_plot_data.line_data.lines.push_back(axial_profile);
    pp_data_fda.line_plot_data.line_data.lines.push_back(radial_profile_z1);
    pp_data_fda.line_plot_data.line_data.lines.push_back(radial_profile_z2);
    pp_data_fda.line_plot_data.line_data.lines.push_back(radial_profile_z3);
    pp_data_fda.line_plot_data.line_data.lines.push_back(radial_profile_z4);
    pp_data_fda.line_plot_data.line_data.lines.push_back(radial_profile_z5);
    pp_data_fda.line_plot_data.line_data.lines.push_back(radial_profile_z6);
    pp_data_fda.line_plot_data.line_data.lines.push_back(radial_profile_z7);
    pp_data_fda.line_plot_data.line_data.lines.push_back(radial_profile_z8);
    pp_data_fda.line_plot_data.line_data.lines.push_back(radial_profile_z9);
    pp_data_fda.line_plot_data.line_data.lines.push_back(radial_profile_z10);
    pp_data_fda.line_plot_data.line_data.lines.push_back(radial_profile_z11);
    pp_data_fda.line_plot_data.line_data.lines.push_back(radial_profile_z12);

    pp.reset(new PostProcessorFDA<dim, Number>(pp_data_fda,
                                               this->mpi_comm,
                                               area_inflow,
                                               *flow_rate_controller,
                                               *inflow_data_storage,
                                               use_precursor,
                                               use_random_perturbations));

    return pp;
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor_precursor() final
  {
    std::shared_ptr<PostProcessorBase<dim, Number>> pp;

    PostProcessorData<dim> pp_data;
    // write output for visualization of results
    pp_data.output_data.write_output                         = this->write_output;
    pp_data.output_data.directory                            = this->output_directory + "vtu/";
    pp_data.output_data.filename                             = this->output_name + "_precursor";
    pp_data.output_data.start_time                           = output_start_time_precursor;
    pp_data.output_data.interval_time                        = output_interval_time;
    pp_data.output_data.write_divergence                     = true;
    pp_data.output_data.write_processor_id                   = true;
    pp_data.output_data.mean_velocity.calculate              = true;
    pp_data.output_data.mean_velocity.sample_start_time      = sample_start_time;
    pp_data.output_data.mean_velocity.sample_end_time        = sample_end_time;
    pp_data.output_data.mean_velocity.sample_every_timesteps = 1;
    pp_data.output_data.degree                               = this->param_pre.degree_u;

    PostProcessorDataFDA<dim> pp_data_fda;
    pp_data_fda.pp_data = pp_data;

    // inflow data
    // prescribe solution at the right boundary of the precursor domain
    // as weak Dirichlet boundary condition at the left boundary of the nozzle domain
    pp_data_fda.inflow_data.write_inflow_data = true;
    pp_data_fda.inflow_data.inflow_geometry   = InflowGeometry::Cylindrical;
    pp_data_fda.inflow_data.normal_direction  = 2;
    pp_data_fda.inflow_data.normal_coordinate = FDANozzle::Z2_PRECURSOR;
    pp_data_fda.inflow_data.n_points_y        = inflow_data_storage->n_points_r;
    pp_data_fda.inflow_data.n_points_z        = inflow_data_storage->n_points_phi;
    pp_data_fda.inflow_data.y_values          = &inflow_data_storage->r_values;
    pp_data_fda.inflow_data.z_values          = &inflow_data_storage->phi_values;
    pp_data_fda.inflow_data.array             = &inflow_data_storage->velocity_values;

    // calculation of flow rate (use volume-based computation)
    pp_data_fda.mean_velocity_data.calculate = true;
    pp_data_fda.mean_velocity_data.directory = this->output_directory;
    pp_data_fda.mean_velocity_data.filename  = filename_flowrate;
    Tensor<1, dim, double> direction;
    direction[2]                                 = 1.0;
    pp_data_fda.mean_velocity_data.direction     = direction;
    pp_data_fda.mean_velocity_data.write_to_file = true;

    pp.reset(new PostProcessorFDA<dim, Number>(pp_data_fda,
                                               this->mpi_comm,
                                               area_inflow,
                                               *flow_rate_controller,
                                               *inflow_data_storage,
                                               use_precursor,
                                               use_random_perturbations));

    return pp;
  }
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application_precursor.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FDA_H_ */
