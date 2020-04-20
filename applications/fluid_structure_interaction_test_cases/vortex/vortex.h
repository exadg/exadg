/*
 * vortex.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_VORTEX_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_VORTEX_H_

#include "../../grid_tools/mesh_movement_functions.h"

namespace FSI
{
namespace Vortex
{
// set problem specific parameters like physical dimensions, etc.
double const U_X_MAX   = 1.0;
double const VISCOSITY = 2.5e-2; // 1.e-2; //2.5e-2;

double const LEFT  = -0.5;
double const RIGHT = 0.5;

double const END_TIME = 1.0;

IncNS::FormulationViscousTerm const FORMULATION_VISCOUS_TERM =
  IncNS::FormulationViscousTerm::LaplaceFormulation;

// moving mesh
bool const ALE = true;

template<int dim>
class AnalyticalSolutionVelocity : public Function<dim>
{
public:
  AnalyticalSolutionVelocity(const unsigned int n_components = dim, const double time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double t      = this->get_time();
    double result = 0.0;

    const double pi = numbers::PI;
    if(component == 0)
      result = -U_X_MAX * std::sin(2.0 * pi * p[1]) * std::exp(-4.0 * pi * pi * VISCOSITY * t);
    else if(component == 1)
      result = U_X_MAX * std::sin(2.0 * pi * p[0]) * std::exp(-4.0 * pi * pi * VISCOSITY * t);

    return result;
  }
};

template<int dim>
class AnalyticalSolutionPressure : public Function<dim>
{
public:
  AnalyticalSolutionPressure(const double time = 0.) : Function<dim>(1 /*n_components*/, time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int /*component*/) const
  {
    double t      = this->get_time();
    double result = 0.0;

    const double pi = numbers::PI;
    result          = -U_X_MAX * std::cos(2 * pi * p[0]) * std::cos(2 * pi * p[1]) *
             std::exp(-8.0 * pi * pi * VISCOSITY * t);

    return result;
  }
};

template<int dim>
class NeumannBoundaryVelocity : public Function<dim>
{
public:
  NeumannBoundaryVelocity(const double time = 0.) : Function<dim>(dim, time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    const double t  = this->get_time();
    const double pi = numbers::PI;

    double result = 0.0;
    // prescribe F_nu(u) / nu = grad(u)
    if(FORMULATION_VISCOUS_TERM == IncNS::FormulationViscousTerm::LaplaceFormulation)
    {
      if(component == 0)
      {
        if((std::abs(p[1] + 0.5) < 1e-12) && (p[0] < 0))
          result = U_X_MAX * 2.0 * pi * std::cos(2.0 * pi * p[1]) *
                   std::exp(-4.0 * pi * pi * VISCOSITY * t);
        else if((std::abs(p[1] - 0.5) < 1e-12) && (p[0] > 0))
          result = -U_X_MAX * 2.0 * pi * std::cos(2.0 * pi * p[1]) *
                   std::exp(-4.0 * pi * pi * VISCOSITY * t);
      }
      else if(component == 1)
      {
        if((std::abs(p[0] + 0.5) < 1e-12) && (p[1] > 0))
          result = -U_X_MAX * 2.0 * pi * std::cos(2.0 * pi * p[0]) *
                   std::exp(-4.0 * pi * pi * VISCOSITY * t);
        else if((std::abs(p[0] - 0.5) < 1e-12) && (p[1] < 0))
          result = U_X_MAX * 2.0 * pi * std::cos(2.0 * pi * p[0]) *
                   std::exp(-4.0 * pi * pi * VISCOSITY * t);
      }
    }
    // prescribe F_nu(u) / nu = ( grad(u) + grad(u)^T )
    else if(FORMULATION_VISCOUS_TERM == IncNS::FormulationViscousTerm::DivergenceFormulation)
    {
      const double pi = numbers::PI;
      if(component == 0)
      {
        if((std::abs(p[1] + 0.5) < 1e-12) && (p[0] < 0))
          result = -U_X_MAX * 2.0 * pi * (std::cos(2.0 * pi * p[0]) - std::cos(2.0 * pi * p[1])) *
                   std::exp(-4.0 * pi * pi * VISCOSITY * t);
        else if((std::abs(p[1] - 0.5) < 1e-12) && (p[0] > 0))
          result = U_X_MAX * 2.0 * pi * (std::cos(2.0 * pi * p[0]) - std::cos(2.0 * pi * p[1])) *
                   std::exp(-4.0 * pi * pi * VISCOSITY * t);
      }
      else if(component == 1)
      {
        if((std::abs(p[0] + 0.5) < 1e-12) && (p[1] > 0))
          result = -U_X_MAX * 2.0 * pi * (std::cos(2.0 * pi * p[0]) - std::cos(2.0 * pi * p[1])) *
                   std::exp(-4.0 * pi * pi * VISCOSITY * t);
        else if((std::abs(p[0] - 0.5) < 1e-12) && (p[1] < 0))
          result = U_X_MAX * 2.0 * pi * (std::cos(2.0 * pi * p[0]) - std::cos(2.0 * pi * p[1])) *
                   std::exp(-4.0 * pi * pi * VISCOSITY * t);
      }
    }
    else
    {
      AssertThrow(FORMULATION_VISCOUS_TERM == IncNS::FormulationViscousTerm::LaplaceFormulation ||
                    FORMULATION_VISCOUS_TERM ==
                      IncNS::FormulationViscousTerm::DivergenceFormulation,
                  ExcMessage("Specified formulation of viscous term is not implemented!"));
    }

    return result;
  }
};

template<int dim>
class NeumannBoundaryVelocityALE : public FunctionWithNormal<dim>
{
public:
  NeumannBoundaryVelocityALE() : FunctionWithNormal<dim>(dim, 0.0)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    const double         t  = this->get_time();
    const Tensor<1, dim> n  = this->get_normal_vector();
    const double         pi = numbers::PI;

    double result = 0.0;
    // prescribe F_nu(u) / nu = grad(u)
    if(FORMULATION_VISCOUS_TERM == IncNS::FormulationViscousTerm::LaplaceFormulation)
    {
      if(component == 0)
        result = 0 * n[0] - U_X_MAX * 2.0 * pi * std::cos(2 * pi * p[1]) *
                              std::exp(-4.0 * pi * pi * VISCOSITY * t) * n[1];
      else if(component == 1)
        result = U_X_MAX * 2.0 * pi * std::cos(2 * pi * p[0]) *
                   std::exp(-4.0 * pi * pi * VISCOSITY * t) * n[0] +
                 0 * n[1];
    }
    // prescribe F_nu(u) / nu = ( grad(u) + grad(u)^T )
    else if(FORMULATION_VISCOUS_TERM == IncNS::FormulationViscousTerm::DivergenceFormulation)
    {
      if(component == 0)
        result = 0 * n[0] + U_X_MAX * 2.0 * pi *
                              (std::cos(2.0 * pi * p[0]) - std::cos(2.0 * pi * p[1])) *
                              std::exp(-4.0 * pi * pi * VISCOSITY * t) * n[1];
      else if(component == 1)
        result = U_X_MAX * 2.0 * pi * (std::cos(2.0 * pi * p[0]) - std::cos(2.0 * pi * p[1])) *
                   std::exp(-4.0 * pi * pi * VISCOSITY * t) * n[0] +
                 0 * n[1];
    }
    else
    {
      AssertThrow(FORMULATION_VISCOUS_TERM == IncNS::FormulationViscousTerm::LaplaceFormulation ||
                    FORMULATION_VISCOUS_TERM ==
                      IncNS::FormulationViscousTerm::DivergenceFormulation,
                  ExcMessage("Specified formulation of viscous term is not implemented!"));
    }

    return result;
  }
};


template<int dim>
class PressureBC_dudt : public Function<dim>
{
public:
  PressureBC_dudt(const double time = 0.) : Function<dim>(dim, time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double t      = this->get_time();
    double result = 0.0;

    const double pi = numbers::PI;
    if(component == 0)
      result = U_X_MAX * 4.0 * pi * pi * VISCOSITY * std::sin(2.0 * pi * p[1]) *
               std::exp(-4.0 * pi * pi * VISCOSITY * t);
    else if(component == 1)
      result = -U_X_MAX * 4.0 * pi * pi * VISCOSITY * std::sin(2.0 * pi * p[0]) *
               std::exp(-4.0 * pi * pi * VISCOSITY * t);

    return result;
  }
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

  std::string output_directory = "output/vortex/", output_name = "fsi_test";

  double const end_time = END_TIME;

  void
  set_input_parameters_fluid(IncNS::InputParameters & param)
  {
    using namespace IncNS;

    // MATHEMATICAL MODEL
    param.problem_type                = ProblemType::Unsteady;
    param.equation_type               = EquationType::NavierStokes;
    param.formulation_viscous_term    = FORMULATION_VISCOUS_TERM;
    param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
    param.right_hand_side             = false;

    // ALE
    param.ale_formulation                     = ALE;
    param.neumann_with_variable_normal_vector = ALE;

    // PHYSICAL QUANTITIES
    param.start_time = 0.0;
    param.end_time   = END_TIME;
    param.viscosity  = VISCOSITY;

    // TEMPORAL DISCRETIZATION
    param.solver_type                     = SolverType::Unsteady;
    param.temporal_discretization         = TemporalDiscretization::BDFDualSplittingScheme;
    param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    param.order_time_integrator           = 2;
    param.start_with_low_order            = false;
    param.adaptive_time_stepping          = false;
    param.calculation_of_time_step_size   = TimeStepCalculation::CFL; // UserSpecified; //CFL;
    param.time_step_size                  = END_TIME;
    param.max_velocity                    = 1.4 * U_X_MAX;
    param.cfl                             = 0.2; // 0.4;
    param.cfl_exponent_fe_degree_velocity = 1.5;
    param.c_eff                           = 8.0;
    param.time_integrator_oif             = TimeIntegratorOIF::ExplRK3Stage7Reg2;
    param.cfl_oif                         = param.cfl / 1.0;

    // output of solver information
    param.solver_info_data.interval_time = 0.1 * (param.end_time - param.start_time);

    // restart
    param.restarted_simulation             = false;
    param.restart_data.write_restart       = false;
    param.restart_data.interval_time       = 0.25;
    param.restart_data.interval_wall_time  = 1.e6;
    param.restart_data.interval_time_steps = 1e8;
    param.restart_data.filename            = "output/vortex/vortex";


    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_p           = DegreePressure::MixedOrder;
    param.mapping            = MappingType::Affine; // initial mesh is a hypercube

    // convective term
    param.upwind_factor = 1.0;

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // velocity pressure coupling terms
    param.gradp_formulation = FormulationPressureGradientTerm::Weak;
    param.divu_formulation  = FormulationVelocityDivergenceTerm::Weak;

    // div-div and continuity penalty
    param.use_divergence_penalty               = true;
    param.divergence_penalty_factor            = 1.0e0;
    param.use_continuity_penalty               = true;
    param.continuity_penalty_factor            = param.divergence_penalty_factor;
    param.continuity_penalty_components        = ContinuityPenaltyComponents::Normal;
    param.continuity_penalty_use_boundary_data = true;
    if(param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
      param.apply_penalty_terms_in_postprocessing_step = false;
    else
      param.apply_penalty_terms_in_postprocessing_step = true;

    // NUMERICAL PARAMETERS
    param.implement_block_diagonal_preconditioner_matrix_free = false;
    param.use_cell_based_face_loops                           = false;
    param.quad_rule_linearization = QuadratureRuleLinearization::Overintegration32k;

    // PROJECTION METHODS

    // formulation
    param.store_previous_boundary_values = true;

    // pressure Poisson equation
    param.solver_pressure_poisson              = SolverPressurePoisson::CG;
    param.solver_data_pressure_poisson         = SolverData(1000, 1.e-12, 1.e-6, 100);
    param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    param.multigrid_data_pressure_poisson.type = MultigridType::pMG;
    param.multigrid_data_pressure_poisson.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev;
    param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::PointJacobi;
    param.multigrid_data_pressure_poisson.smoother_data.smoother = MultigridSmoother::Chebyshev;
    param.multigrid_data_pressure_poisson.smoother_data.preconditioner =
      PreconditionerSmoother::PointJacobi;

    // projection step
    param.solver_projection                        = SolverProjection::CG;
    param.solver_data_projection                   = SolverData(1000, 1.e-12, 1.e-6);
    param.preconditioner_projection                = PreconditionerProjection::InverseMassMatrix;
    param.preconditioner_block_diagonal_projection = Elementwise::Preconditioner::InverseMassMatrix;
    param.solver_data_block_diagonal_projection    = SolverData(1000, 1.e-12, 1.e-2, 1000);

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;
    param.formulation_convective_term_bc = FormulationConvectiveTerm::ConvectiveFormulation;

    // viscous step
    param.solver_viscous              = SolverViscous::CG;
    param.solver_data_viscous         = SolverData(1000, 1.e-12, 1.e-6);
    param.preconditioner_viscous      = PreconditionerViscous::InverseMassMatrix; // Multigrid;
    param.multigrid_data_viscous.type = MultigridType::hMG;
    param.multigrid_data_viscous.smoother_data.smoother = MultigridSmoother::Chebyshev;
    param.update_preconditioner_viscous                 = false;


    // PRESSURE-CORRECTION SCHEME

    // formulation
    param.order_pressure_extrapolation =
      std::min(2, (int)param.order_time_integrator) - 1; // J_p = J-1, but not larger than 1
    param.rotational_formulation = true;

    // momentum step

    // Newton solver
    param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-12, 1.e-6); // TODO

    // linear solver
    param.solver_momentum                = SolverMomentum::FGMRES;
    param.solver_data_momentum           = SolverData(1e4, 1.e-12, 1.e-6, 100);
    param.update_preconditioner_momentum = false;
    param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix; // Multigrid;
    param.multigrid_operator_type_momentum = MultigridOperatorType::ReactionDiffusion;

    // Jacobi smoother data
    //  param.multigrid_data_momentum.smoother_data.smoother = MultigridSmoother::Jacobi;
    //  param.multigrid_data_momentum.smoother_data.preconditioner =
    //  PreconditionerSmoother::BlockJacobi; param.multigrid_data_momentum.smoother_data.iterations
    //  = 5; param.multigrid_data_momentum.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

    // Chebyshev smoother data
    param.multigrid_data_momentum.smoother_data.smoother = MultigridSmoother::Chebyshev;
    param.multigrid_data_momentum.coarse_problem.solver  = MultigridCoarseGridSolver::Chebyshev;


    // COUPLED NAVIER-STOKES SOLVER
    param.use_scaling_continuity = false;

    // nonlinear solver (Newton solver)
    param.newton_solver_data_coupled =
      Newton::SolverData(100, 1.e-10, 1.e-6); // TODO did not converge with 1.e-12

    // linear solver
    param.solver_coupled      = SolverCoupled::FGMRES;
    param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-6, 100);

    // preconditioner linear solver
    param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    param.update_preconditioner_coupled = false;

    // preconditioner momentum block
    param.preconditioner_velocity_block          = MomentumPreconditioner::Multigrid;
    param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;
    param.multigrid_data_velocity_block.type     = MultigridType::phMG;
    param.multigrid_data_velocity_block.smoother_data.smoother =
      MultigridSmoother::Chebyshev; // GMRES;
    param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::BlockJacobi;
    param.multigrid_data_velocity_block.smoother_data.iterations        = 5;
    param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
    // coarse grid solver
    param.multigrid_data_velocity_block.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev; // GMRES;

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block =
      SchurComplementPreconditioner::PressureConvectionDiffusion;
    param.discretization_of_laplacian = DiscretizationOfLaplacian::Classical;
  }

  void
  set_input_parameters_poisson(Poisson::InputParameters & param)
  {
    using namespace Poisson;

    // MATHEMATICAL MODEL
    param.right_hand_side = false;

    // SPATIAL DISCRETIZATION
    param.triangulation_type     = TriangulationType::Distributed;
    param.mapping                = MappingType::Affine; // initial mesh is a hypercube
    param.spatial_discretization = SpatialDiscretization::CG;
    param.IP_factor              = 1.0e0;

    // SOLVER
    param.solver                    = Poisson::Solver::CG;
    param.solver_data.abs_tol       = 1.e-20;
    param.solver_data.rel_tol       = 1.e-10;
    param.solver_data.max_iter      = 1e4;
    param.preconditioner            = Preconditioner::Multigrid;
    param.multigrid_data.type       = MultigridType::cphMG;
    param.multigrid_data.p_sequence = PSequenceType::Bisect;
    // MG smoother
    param.multigrid_data.smoother_data.smoother        = MultigridSmoother::Chebyshev;
    param.multigrid_data.smoother_data.iterations      = 5;
    param.multigrid_data.smoother_data.smoothing_range = 20;
    // MG coarse grid solver
    param.multigrid_data.coarse_problem.solver         = MultigridCoarseGridSolver::CG;
    param.multigrid_data.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::AMG;
    param.multigrid_data.coarse_problem.solver_data.rel_tol = 1.e-3;
  }

  void
  create_grid_fluid(
    std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
    unsigned int const                                n_refine_space,
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
      periodic_faces)
  {
    (void)periodic_faces;

    // Uniform Cartesian grid
    const double left = -0.5, right = 0.5;
    GridGenerator::subdivided_hyper_cube(*triangulation, 2, left, right);

    typename Triangulation<dim>::cell_iterator cell = triangulation->begin(),
                                               endc = triangulation->end();
    for(; cell != endc; ++cell)
    {
      for(unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell;
          ++face_number)
      {
        if(((std::fabs(cell->face(face_number)->center()(0) - right) < 1e-12) &&
            (cell->face(face_number)->center()(1) < 0)) ||
           ((std::fabs(cell->face(face_number)->center()(0) - left) < 1e-12) &&
            (cell->face(face_number)->center()(1) > 0)) ||
           ((std::fabs(cell->face(face_number)->center()(1) - left) < 1e-12) &&
            (cell->face(face_number)->center()(0) < 0)) ||
           ((std::fabs(cell->face(face_number)->center()(1) - right) < 1e-12) &&
            (cell->face(face_number)->center()(0) > 0)))
        {
          cell->face(face_number)->set_boundary_id(1);
        }
      }
    }

    triangulation->refine_global(n_refine_space);
  }

  std::shared_ptr<Function<dim>>
  set_mesh_movement_function_fluid()
  {
    std::shared_ptr<Function<dim>> mesh_motion;

    MeshMovementData<dim> data;
    data.temporal      = MeshMovementAdvanceInTime::Sin;
    data.shape         = MeshMovementShape::Sin; // SineAligned;
    data.dimensions[0] = std::abs(RIGHT - LEFT);
    data.dimensions[1] = std::abs(RIGHT - LEFT);
    data.amplitude = 0.08 * (RIGHT - LEFT); // 0.12 * (RIGHT-LEFT); // A_max = (RIGHT-LEFT)/(2*pi)
    data.period    = 4.0 * END_TIME;
    data.t_start   = 0.0;
    data.t_end     = END_TIME;
    data.spatial_number_of_oscillations = 1.0;
    mesh_motion.reset(new CubeMeshMovementFunctions<dim>(data));

    return mesh_motion;
  }

  void
  set_boundary_conditions_fluid(
    std::shared_ptr<IncNS::BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
    std::shared_ptr<IncNS::BoundaryDescriptorP<dim>> boundary_descriptor_pressure)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // fill boundary descriptor velocity
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(0, new AnalyticalSolutionVelocity<dim>()));
    if(ALE)
      boundary_descriptor_velocity->neumann_bc.insert(
        pair(1, new NeumannBoundaryVelocityALE<dim>()));
    else
      boundary_descriptor_velocity->neumann_bc.insert(pair(1, new NeumannBoundaryVelocity<dim>()));

    // fill boundary descriptor pressure
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new PressureBC_dudt<dim>()));
    boundary_descriptor_pressure->dirichlet_bc.insert(
      pair(1, new AnalyticalSolutionPressure<dim>()));
  }

  void
  set_field_functions_fluid(std::shared_ptr<IncNS::FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
    field_functions->initial_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
    field_functions->analytical_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  void set_boundary_conditions_poisson(
    std::shared_ptr<Poisson::BoundaryDescriptor<1, dim>> boundary_descriptor)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // these lines show exemplarily how the boundary descriptors are filled
    std::shared_ptr<Function<dim>> bc = this->set_mesh_movement_function_fluid();
    boundary_descriptor->dirichlet_bc.insert(pair(0, bc));
    boundary_descriptor->dirichlet_bc.insert(pair(1, bc));
  }


  void
  set_field_functions_poisson(std::shared_ptr<Poisson::FieldFunctions<dim>> field_functions)
  {
    // these lines show exemplarily how the field functions are filled
    field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
  }

  std::shared_ptr<IncNS::PostProcessorBase<dim, Number>>
  construct_postprocessor_fluid(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    IncNS::PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output         = true;
    pp_data.output_data.output_folder        = output_directory + "vtu/";
    pp_data.output_data.output_name          = output_name;
    pp_data.output_data.output_start_time    = 0.0;
    pp_data.output_data.output_interval_time = end_time / 20;
    pp_data.output_data.write_higher_order   = false;
    pp_data.output_data.degree               = degree;

    // calculation of velocity error
    pp_data.error_data_u.analytical_solution_available = true;
    pp_data.error_data_u.analytical_solution.reset(new AnalyticalSolutionVelocity<dim>());
    pp_data.error_data_u.calculate_relative_errors = true;
    pp_data.error_data_u.error_calc_start_time     = 0.0;
    pp_data.error_data_u.error_calc_interval_time  = end_time;
    pp_data.error_data_u.name                      = "velocity";

    // ... pressure error
    pp_data.error_data_p.analytical_solution_available = true;
    pp_data.error_data_p.analytical_solution.reset(new AnalyticalSolutionPressure<dim>());
    pp_data.error_data_p.calculate_relative_errors = true;
    pp_data.error_data_p.error_calc_start_time     = 0.0;
    pp_data.error_data_p.error_calc_interval_time  = end_time;
    pp_data.error_data_p.name                      = "pressure";

    std::shared_ptr<IncNS::PostProcessorBase<dim, Number>> pp;
    pp.reset(new IncNS::PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace Vortex
} // namespace FSI

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_VORTEX_H_ */
