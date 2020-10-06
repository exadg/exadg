/*
 * flow_past_cylinder.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_CYLINDER_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_CYLINDER_H_

// ExaDG
#include <exadg/functions_and_boundary_conditions/linear_interpolation.h>

#include "grid.h"

namespace ExaDG
{
namespace IncNS
{
namespace FlowPastCylinder
{
using namespace dealii;

template<int dim>
class InflowBC : public Function<dim>
{
public:
  InflowBC(double const                                Um,
           double const                                H,
           double const                                end_time,
           unsigned int const                          test_case,
           bool const                                  use_random_perturbations,
           std::vector<double> const &                 y,
           std::vector<double> const &                 z,
           std::vector<Tensor<1, dim, double>> const & u)
    : Function<dim>(dim, 0.0),
      Um(Um),
      H(H),
      end_time(end_time),
      test_case(test_case),
      use_random_perturbations(use_random_perturbations),
      y_vector(y),
      z_vector(z),
      u_vector(u)
  {
  }

  double
  value(const Point<dim> & x, const unsigned int component = 0) const
  {
    double t      = this->get_time();
    double result = 0.0;

    if(component == 0)
    {
      const double pi = numbers::PI;
      const double T  = 1.0;
      double       coefficient =
        Utilities::fixed_power<dim - 1>(4.) * Um / Utilities::fixed_power<2 * dim - 2>(H);

      if(test_case == 1)
        result = coefficient * x[1] * (H - x[1]);
      else if(test_case == 2)
        result =
          coefficient * x[1] * (H - x[1]) * ((t / T) < 1.0 ? std::sin(pi / 2. * t / T) : 1.0);
      else if(test_case == 3)
        result = coefficient * x[1] * (H - x[1]) * std::sin(pi * t / end_time);
      else
        AssertThrow(false, ExcMessage("Not implemented."));

      if(dim == 3)
        result *= x[2] * (H - x[2]);

      if(use_random_perturbations)
      {
        double perturbation = 0.0;

        if(dim == 2)
          perturbation = linear_interpolation_1d(x[1], y_vector, u_vector, component);
        else if(dim == 3)
        {
          Point<dim> point_3d;
          point_3d[0] = x[0];
          point_3d[1] = x[1];
          point_3d[2] = x[2];

          perturbation =
            linear_interpolation_2d_cartesian(point_3d, y_vector, z_vector, u_vector, component);
        }
        else
          AssertThrow(false, ExcMessage("Not implemented."));

        result += perturbation;
      }
    }

    return result;
  }

private:
  double const       Um, H, end_time;
  unsigned int const test_case;

  // perturbations
  bool const                          use_random_perturbations;
  std::vector<double> const &         y_vector;
  std::vector<double> const &         z_vector;
  std::vector<Tensor<1, dim, double>> u_vector;
};

template<int dim>
class PressureBC_dudt : public Function<dim>
{
public:
  PressureBC_dudt(const double       Um,
                  const double       H,
                  const double       end_time,
                  const unsigned int test_case)
    : Function<dim>(dim, 0.0), Um(Um), H(H), end_time(end_time), test_case(test_case)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double t      = this->get_time();
    double result = 0.0;

    if(component == 0 && std::abs(p[0] - (dim == 2 ? L1 : 0.0)) < 1.e-12)
    {
      const double pi = numbers::PI;

      const double T = 1.0;
      double       coefficient =
        Utilities::fixed_power<dim - 1>(4.) * Um / Utilities::fixed_power<2 * dim - 2>(H);

      if(test_case == 2)
        result = coefficient * p[1] * (H - p[1]) *
                 ((t / T) < 1.0 ? (pi / 2. / T) * std::cos(pi / 2. * t / T) : 0.0);
      if(test_case == 3)
        result = coefficient * p[1] * (H - p[1]) * std::cos(pi * t / end_time) * pi / end_time;

      if(dim == 3)
        result *= p[2] * (H - p[2]);
    }

    return result;
  }

private:
  double const       Um, H, end_time;
  unsigned int const test_case;
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

    if(use_perturbation)
    {
      initialize_y_and_z_values();
      initialize_velocity_values();
    }
  }

  void
  initialize_y_and_z_values()
  {
    AssertThrow(n_points_y >= 2, ExcMessage("Variable n_points_y is invalid"));
    if(dim == 3)
      AssertThrow(n_points_z >= 2, ExcMessage("Variable n_points_z is invalid"));

    // 0 <= y <= H
    for(unsigned int i = 0; i < n_points_y; ++i)
      y_values[i] = double(i) / double(n_points_y - 1) * H;

    // 0 <= z <= H
    if(dim == 3)
      for(unsigned int i = 0; i < n_points_z; ++i)
        z_values[i] = double(i) / double(n_points_z - 1) * H;
  }

  void
  initialize_velocity_values()
  {
    AssertThrow(n_points_y >= 2, ExcMessage("Variable n_points_y is invalid"));
    if(dim == 3)
      AssertThrow(n_points_z >= 2, ExcMessage("Variable n_points_z is invalid"));

    for(unsigned int iy = 0; iy < n_points_y; ++iy)
    {
      for(unsigned int iz = 0; iz < n_points_z; ++iz)
      {
        Tensor<1, dim, double> velocity;

        if(use_perturbation == true)
        {
          // Add random perturbation
          double const y = y_values[iy];
          double const z = z_values[iz];
          double       coefficient =
            Utilities::fixed_power<dim - 1>(4.) * Um / Utilities::fixed_power<2 * dim - 2>(H);
          double perturbation =
            amplitude_perturbation * coefficient * ((double)rand() / RAND_MAX - 0.5) / 0.5;
          perturbation *= y * (H - y);
          if(dim == 3)
            perturbation *= z * (H - z);

          velocity[0] += perturbation;
        }

        velocity_values[iy * n_points_z + iz] = velocity;
      }
    }
  }

  void
  add_parameters(ParameterHandler & prm)
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    // clang-format off
    prm.enter_subsection("Application");
      prm.add_parameter("TestCase", test_case, "Number of test case.",
        Patterns::Integer(1,3));
      prm.add_parameter("CylinderType", cylinder_type_string, "Type of cylinder.",
        Patterns::Selection("circular|square"));
    prm.leave_subsection();
    // clang-format on
  }

  // string to read input parameter
  std::string cylinder_type_string = "circular";

  // select test case according to Schaefer and Turek benchmark definition: 2D-1/2/3, 3D-1/2/3
  unsigned int test_case = 3; // 1, 2 or 3

  ProblemType  problem_type = ProblemType::Unsteady;
  double const Um = (dim == 2 ? (test_case == 1 ? 0.3 : 1.5) : (test_case == 1 ? 0.45 : 2.25));

  double const viscosity = 1.e-3;

  // start and end time
  // use a large value for test_case = 1 (steady problem)
  // in order to not stop pseudo-timestepping approach before having converged
  double const start_time = 0.0;
  double const end_time   = (test_case == 1) ? 1000.0 : 8.0;

  // superimpose random perturbations at inflow
  bool const use_perturbation = false;
  // amplitude of perturbations relative to maximum velocity on centerline
  double const amplitude_perturbation = 0.25;

  unsigned int const n_points_y = 10;
  unsigned int const n_points_z = dim == 3 ? n_points_y : 1;

  std::vector<double> y_values = std::vector<double>(n_points_y);
  std::vector<double> z_values = std::vector<double>(n_points_z);

  std::vector<Tensor<1, dim, double>> velocity_values =
    std::vector<Tensor<1, dim, double>>(n_points_y * n_points_z);

  // solver tolerances
  double const ABS_TOL = 1.e-12;
  double const REL_TOL = 1.e-3;

  double const ABS_TOL_LINEAR = ABS_TOL;
  double const REL_TOL_LINEAR = REL_TOL;

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.problem_type             = problem_type;
    param.equation_type            = EquationType::NavierStokes;
    param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
    param.right_hand_side          = false;


    // PHYSICAL QUANTITIES
    param.start_time = start_time;
    param.end_time   = end_time;
    param.viscosity  = viscosity;


    // TEMPORAL DISCRETIZATION
    param.solver_type                     = SolverType::Unsteady;
    param.temporal_discretization         = TemporalDiscretization::BDFDualSplittingScheme;
    param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    param.time_integrator_oif             = TimeIntegratorOIF::ExplRK2Stage2;
    param.order_time_integrator           = 2;
    param.start_with_low_order            = true;
    param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    param.adaptive_time_stepping          = true;
    param.max_velocity                    = Um;
    param.cfl                             = 0.4; // use CFL <= 0.4 - 0.6 for adaptive time stepping
    param.cfl_oif                         = param.cfl;
    param.cfl_exponent_fe_degree_velocity = 1.5;
    param.time_step_size                  = 1.0e-3;
    param.time_step_size_max              = 1.e-2;

    // output of solver information
    param.solver_info_data.interval_time = (param.end_time - param.start_time) / 8.0;

    // pseudo-timestepping for steady-state problems
    param.convergence_criterion_steady_problem =
      ConvergenceCriterionSteadyProblem::SolutionIncrement; // ResidualSteadyNavierStokes;
    param.abs_tol_steady = 1.e-12;
    param.rel_tol_steady = 1.e-8;

    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_p           = DegreePressure::MixedOrder;
    param.mapping            = MappingType::Isoparametric;

    // convective term
    if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

    // divergence penalty
    param.use_divergence_penalty    = true;
    param.divergence_penalty_factor = 1.0e0;
    param.use_continuity_penalty    = true;
    param.continuity_penalty_factor = param.divergence_penalty_factor;

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // NUMERICAL PARAMETERS
    param.implement_block_diagonal_preconditioner_matrix_free = false;
    param.use_cell_based_face_loops                           = false;
    param.quad_rule_linearization = QuadratureRuleLinearization::Overintegration32k;

    // PROJECTION METHODS

    // formulation
    param.store_previous_boundary_values = true;

    // pressure Poisson equation
    param.solver_pressure_poisson              = SolverPressurePoisson::CG; // FGMRES;
    param.solver_data_pressure_poisson         = SolverData(1000, ABS_TOL, REL_TOL, 30);
    param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
    param.multigrid_data_pressure_poisson.smoother_data.smoother   = MultigridSmoother::Chebyshev;
    param.multigrid_data_pressure_poisson.smoother_data.iterations = 5;
    param.multigrid_data_pressure_poisson.coarse_problem.solver    = MultigridCoarseGridSolver::CG;
    param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;
    param.update_preconditioner_pressure_poisson = false;

    // projection step
    param.solver_projection              = SolverProjection::CG;
    param.solver_data_projection         = SolverData(1000, ABS_TOL, REL_TOL);
    param.preconditioner_projection      = PreconditionerProjection::InverseMassMatrix;
    param.multigrid_data_projection.type = MultigridType::phcMG;
    param.multigrid_data_projection.smoother_data.smoother = MultigridSmoother::Chebyshev;
    param.multigrid_data_projection.coarse_problem.solver  = MultigridCoarseGridSolver::CG;
    param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;
    param.update_preconditioner_projection                  = false;
    param.update_preconditioner_projection_every_time_steps = 10;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // viscous step
    param.solver_viscous                = SolverViscous::CG;
    param.solver_data_viscous           = SolverData(1000, ABS_TOL, REL_TOL);
    param.preconditioner_viscous        = PreconditionerViscous::InverseMassMatrix; // Multigrid;
    param.update_preconditioner_viscous = false;
    param.multigrid_data_viscous.type   = MultigridType::phcMG;
    param.multigrid_data_viscous.coarse_problem.solver = MultigridCoarseGridSolver::CG;
    param.multigrid_data_viscous.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;

    // PRESSURE-CORRECTION SCHEME

    // momentum step

    // Newton solver
    param.newton_solver_data_momentum = Newton::SolverData(100, ABS_TOL, REL_TOL);

    // linear solver
    param.solver_momentum                  = SolverMomentum::CG; // FGMRES;
    param.solver_data_momentum             = SolverData(1e4, ABS_TOL_LINEAR, REL_TOL_LINEAR, 100);
    param.preconditioner_momentum          = MomentumPreconditioner::InverseMassMatrix;
    param.multigrid_operator_type_momentum = MultigridOperatorType::ReactionDiffusion;
    param.multigrid_data_momentum.type     = MultigridType::phcMG;
    param.multigrid_data_momentum.coarse_problem.solver = MultigridCoarseGridSolver::CG;
    param.multigrid_data_momentum.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;
    param.update_preconditioner_momentum = false;

    // formulation
    param.order_pressure_extrapolation = 1;
    param.rotational_formulation       = true;


    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    param.newton_solver_data_coupled = Newton::SolverData(100, ABS_TOL, REL_TOL);

    // linear solver
    param.solver_coupled      = SolverCoupled::FGMRES; // FGMRES;
    param.solver_data_coupled = SolverData(1e4, ABS_TOL_LINEAR, REL_TOL_LINEAR, 100);

    param.update_preconditioner_coupled = false;

    // preconditioning linear solver
    param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

    // preconditioner velocity/momentum block
    param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix; // Multigrid;
    param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;
    param.multigrid_data_velocity_block.type     = MultigridType::phcMG;
    param.multigrid_data_velocity_block.smoother_data.smoother = MultigridSmoother::Chebyshev;
    param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::PointJacobi;
    param.multigrid_data_velocity_block.smoother_data.iterations = 5;
    param.multigrid_data_velocity_block.coarse_problem.solver    = MultigridCoarseGridSolver::CG;
    param.multigrid_data_velocity_block.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;
    param.multigrid_data_velocity_block.coarse_problem.solver_data.rel_tol           = 1.e-3;
    param.multigrid_data_velocity_block.coarse_problem.amg_data.data.smoother_type   = "Chebyshev";
    param.multigrid_data_velocity_block.coarse_problem.amg_data.data.smoother_sweeps = 1;

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block =
      SchurComplementPreconditioner::PressureConvectionDiffusion;
    param.discretization_of_laplacian        = DiscretizationOfLaplacian::Classical;
    param.multigrid_data_pressure_block.type = MultigridType::cphMG;
    param.multigrid_data_pressure_block.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev; // CG;
    //  param.multigrid_data_pressure_block.coarse_problem.preconditioner =
    //  MultigridCoarseGridPreconditioner::AMG;
    param.multigrid_data_pressure_block.coarse_problem.solver_data.rel_tol           = 1.e-3;
    param.multigrid_data_pressure_block.coarse_problem.amg_data.data.smoother_type   = "Chebyshev";
    param.multigrid_data_pressure_block.coarse_problem.amg_data.data.smoother_sweeps = 1;
  }


  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    if(auto tria_fully_dist =
         dynamic_cast<parallel::fullydistributed::Triangulation<dim> *>(&*triangulation))
    {
      const auto construction_data =
        TriangulationDescription::Utilities::create_description_from_triangulation_in_groups<dim,
                                                                                             dim>(
          [&](dealii::Triangulation<dim, dim> & tria) mutable {
            create_cylinder_grid<dim>(tria, n_refine_space, periodic_faces, cylinder_type_string);
          },
          [](dealii::Triangulation<dim, dim> & tria,
             const MPI_Comm                    comm,
             const unsigned int /* group_size */) {
            // metis partitioning
            GridTools::partition_triangulation(Utilities::MPI::n_mpi_processes(comm), tria);
            // p4est partitioning
            //            GridTools::partition_triangulation_zorder(Utilities::MPI::n_mpi_processes(comm),
            //            tria);
          },
          tria_fully_dist->get_communicator(),
          1 /* group size */);
      tria_fully_dist->create_triangulation(construction_data);
    }
    else if(auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> *>(&*triangulation))
    {
      create_cylinder_grid<dim>(*tria, n_refine_space, periodic_faces, cylinder_type_string);
    }
    else
    {
      AssertThrow(false, ExcMessage("Unknown triangulation!"));
    }
  }

  void
  set_boundary_conditions(std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
                          std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // fill boundary descriptor velocity
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(0,
           new InflowBC<dim>(
             Um, H, end_time, test_case, use_perturbation, y_values, z_values, velocity_values)));
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(2, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_velocity->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(dim)));

    // fill boundary descriptor pressure
    boundary_descriptor_pressure->neumann_bc.insert(
      pair(0, new PressureBC_dudt<dim>(Um, H, end_time, test_case)));
    boundary_descriptor_pressure->neumann_bc.insert(
      pair(2, new PressureBC_dudt<dim>(Um, H, end_time, test_case)));
    boundary_descriptor_pressure->dirichlet_bc.insert(pair(1, new Functions::ZeroFunction<dim>(1)));
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
    pp_data.output_data.output_interval_time = (end_time - start_time) / 20;
    pp_data.output_data.write_divergence     = true;
    pp_data.output_data.write_higher_order   = false;
    pp_data.output_data.write_processor_id   = true;
    pp_data.output_data.write_surface_mesh   = true;
    pp_data.output_data.write_boundary_IDs   = true;
    pp_data.output_data.write_grid           = true;
    pp_data.output_data.degree               = degree;

    // lift and drag
    pp_data.lift_and_drag_data.calculate_lift_and_drag = true;
    pp_data.lift_and_drag_data.viscosity               = viscosity;

    const double U = Um * (dim == 2 ? 2. / 3. : 4. / 9.);
    if(dim == 2)
      pp_data.lift_and_drag_data.reference_value = 1.0 / 2.0 * pow(U, 2.0) * D;
    else if(dim == 3)
      pp_data.lift_and_drag_data.reference_value = 1.0 / 2.0 * pow(U, 2.0) * D * H;

    // surface for calculation of lift and drag coefficients has boundary_ID = 2
    pp_data.lift_and_drag_data.boundary_IDs.insert(2);

    pp_data.lift_and_drag_data.filename_lift = this->output_directory + this->output_name + "_lift";
    pp_data.lift_and_drag_data.filename_drag = this->output_directory + this->output_name + "_drag";

    // pressure difference
    pp_data.pressure_difference_data.calculate_pressure_difference = true;
    if(dim == 2)
    {
      Point<dim> point_1_2D((X_C - D / 2.0), Y_C), point_2_2D((X_C + D / 2.0), Y_C);
      pp_data.pressure_difference_data.point_1 = point_1_2D;
      pp_data.pressure_difference_data.point_2 = point_2_2D;
    }
    else if(dim == 3)
    {
      Point<dim> point_1_3D((X_C - D / 2.0), Y_C, H / 2.0),
        point_2_3D((X_C + D / 2.0), Y_C, H / 2.0);
      pp_data.pressure_difference_data.point_1 = point_1_3D;
      pp_data.pressure_difference_data.point_2 = point_2_3D;
    }

    pp_data.pressure_difference_data.filename =
      this->output_directory + this->output_name + "_pressure_difference";

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace FlowPastCylinder
} // namespace IncNS
} // namespace ExaDG


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_CYLINDER_H_ */
