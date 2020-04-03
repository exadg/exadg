/*
 * cylinder_with_flag.h
 *
 *  Created on: Mar 18, 2020
 *      Author: fehn
 */

#ifndef APPLICATIONS_FSI_CYLINDER_WITH_FLAG_H_
#define APPLICATIONS_FSI_CYLINDER_WITH_FLAG_H_

#include "../../../include/functionalities/one_sided_cylindrical_manifold.h"

namespace FSI
{
namespace CylinderWithFlag
{
// set problem specific parameters like physical dimensions, etc.
double const U_X_MAX   = 1.0;
double const VISCOSITY = 0.01;

// physical dimensions (diameter D and center coordinate Y_C can be varied)
double const X_0    = 0.0;  // origin (x-coordinate)
double const Y_0    = 0.0;  // origin (y-coordinate)
double const L      = 2.5;  // x-coordinate of outflow boundary (=length for 3d test cases)
double const H      = 0.41; // height of channel
double const X_C    = 0.2;  // center of cylinder (x-coordinate)
double const Y_C    = 0.2;  // center of cylinder (y-coordinate)
double const X_2    = 2.0 * X_C;
double const D      = 0.1;     // cylinder diameter
double const R      = D / 2.0; // cylinder radius
double const T      = 0.02;    // thickness of flag
double const L_FLAG = 0.35;    // length of flag
double const X_3    = X_C + R + L_FLAG * 1.6;
double const Y_3    = H / 3.0;

unsigned int const BOUNDARY_ID_CYLINDER = 3;
unsigned int const BOUNDARY_ID_FLAG     = 4;

// manifold ID of spherical manifold
unsigned int const MANIFOLD_ID = 10;

// vectors of manifold_ids and face_ids
std::vector<unsigned int> manifold_ids;
std::vector<unsigned int> face_ids;

double const END_TIME = 8.0;

// moving mesh
bool const ALE = true;

double
function_space(double const x)
{
  double const AMPLITUDE = 2.0 * T;
  if(x > X_C + R)
  {
    return AMPLITUDE * (std::pow((x - (X_C + R)) / L_FLAG, 2.0) -
                        (1 - std::cos((x - (X_C + R)) / L_FLAG * 2.0 * numbers::PI)) / 4.0);
  }
  else
    return 0.0;
}

double
function(double const t)
{
  double const T = END_TIME;
  return std::sin(2.0 * numbers::PI * t / T);
}

double
dfdt(double const t)
{
  double const T = END_TIME;
  return 2.0 * numbers::PI * std::cos(2.0 * numbers::PI * t / T);
}

template<int dim>
class MeshMotion : public Function<dim>
{
public:
  MeshMotion() : Function<dim>(dim, 0.0)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double t      = this->get_time();
    double result = 0.0;

    if(component == 1)
      result = function_space(p[0]) * function(t);

    return result;
  }
};

template<int dim>
class InflowBC : public Function<dim>
{
public:
  InflowBC() : Function<dim>(dim, 0.0)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    (void)p;
    double result = 0.0;

    if(component == 0)
      result = U_X_MAX;

    return result;
  }
};

template<int dim>
class VelocityBendingWall : public Function<dim>
{
public:
  VelocityBendingWall() : Function<dim>(dim, 0.0)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double t      = this->get_time();
    double result = 0.0;

    if(component == 1)
      result = function_space(p[0]) * dfdt(t);

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

  std::string output_directory = "output/cylinder_with_flag/", output_name = "test";

  void
  set_input_parameters_fluid(IncNS::InputParameters & param)
  {
    using namespace IncNS;

    // MATHEMATICAL MODEL
    param.problem_type                   = ProblemType::Unsteady;
    param.equation_type                  = EquationType::NavierStokes;
    param.formulation_viscous_term       = FormulationViscousTerm::LaplaceFormulation;
    param.formulation_convective_term    = FormulationConvectiveTerm::ConvectiveFormulation;
    param.use_outflow_bc_convective_term = true;
    param.right_hand_side                = false;

    // ALE
    param.ale_formulation                     = ALE;
    param.neumann_with_variable_normal_vector = false;

    // PHYSICAL QUANTITIES
    param.start_time = 0.0;
    param.end_time   = END_TIME;
    param.viscosity  = VISCOSITY;

    // TEMPORAL DISCRETIZATION
    param.solver_type                     = SolverType::Unsteady;
    param.temporal_discretization         = TemporalDiscretization::BDFDualSplittingScheme;
    param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    param.order_time_integrator           = 2;
    param.start_with_low_order            = true;
    param.adaptive_time_stepping          = true;
    param.calculation_of_time_step_size   = TimeStepCalculation::CFL; // UserSpecified; //CFL;
    param.time_step_size                  = END_TIME;
    param.max_velocity                    = U_X_MAX;
    param.cfl                             = 1.5;
    param.cfl_exponent_fe_degree_velocity = 1.5;

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
    param.mapping            = MappingType::Isoparametric;

    // convective term
    param.upwind_factor = 1.0;

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
    param.continuity_penalty_use_boundary_data       = true;
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
    param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
    param.multigrid_data_pressure_poisson.smoother_data.smoother = MultigridSmoother::Chebyshev;
    param.multigrid_data_pressure_poisson.smoother_data.preconditioner =
      PreconditionerSmoother::PointJacobi;
    param.multigrid_data_pressure_poisson.coarse_problem.solver = MultigridCoarseGridSolver::CG;
    param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;
    param.multigrid_data_pressure_poisson.coarse_problem.solver_data.rel_tol = 1.e-3;

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
    param.solver_viscous         = SolverViscous::CG;
    param.solver_data_viscous    = SolverData(1000, 1.e-12, 1.e-6);
    param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;


    // PRESSURE-CORRECTION SCHEME

    // formulation
    param.order_pressure_extrapolation =
      std::min(2, (int)param.order_time_integrator) - 1; // J_p = J-1, but not larger than 1
    param.rotational_formulation = true;

    // momentum step

    // Newton solver
    param.newton_solver_data_momentum = NewtonSolverData(100, 1.e-12, 1.e-6);

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
    param.newton_solver_data_coupled = NewtonSolverData(100, 1.e-10, 1.e-6);

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
    param.mapping                = MappingType::Isoparametric;
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

  void create_triangulation(Triangulation<2> & tria)
  {
    std::vector<Triangulation<2>> tria_vec;
    tria_vec.resize(11);

    GridGenerator::general_cell(tria_vec[0],
                                {Point<2>(X_0, 0.0),
                                 Point<2>(X_C - R / std::sqrt(2.0), Y_C - R / std::sqrt(2.0)),
                                 Point<2>(X_0, H),
                                 Point<2>(X_C - R / std::sqrt(2.0), Y_C + R / std::sqrt(2.0))});

    GridGenerator::general_cell(tria_vec[1],
                                {Point<2>(X_0, 0.0),
                                 Point<2>(X_2, 0.0),
                                 Point<2>(X_C - R / std::sqrt(2.0), Y_C - R / std::sqrt(2.0)),
                                 Point<2>(X_C + R / std::sqrt(2.0), Y_C - R / std::sqrt(2.0))});

    GridGenerator::general_cell(tria_vec[2],
                                {Point<2>(X_C - R / std::sqrt(2.0), Y_C + R / std::sqrt(2.0)),
                                 Point<2>(X_C + R / std::sqrt(2.0), Y_C + R / std::sqrt(2.0)),
                                 Point<2>(X_0, H),
                                 Point<2>(X_2, H)});

    GridGenerator::general_cell(tria_vec[3],
                                {Point<2>(X_C + R / std::sqrt(2.0), Y_C - R / std::sqrt(2.0)),
                                 Point<2>(X_2, 0.0),
                                 Point<2>(X_C + R * std::cos(std::asin(T / (2.0 * R))),
                                          Y_C - T / 2.0),
                                 Point<2>(X_2, Y_C - T / 2.0)});

    GridGenerator::general_cell(tria_vec[4],
                                {Point<2>(X_C + R * std::cos(std::asin(T / (2.0 * R))),
                                          Y_C + T / 2.0),
                                 Point<2>(X_2, Y_C + T / 2.0),
                                 Point<2>(X_C + R / std::sqrt(2.0), Y_C + R / std::sqrt(2.0)),
                                 Point<2>(X_2, H)});

    GridGenerator::subdivided_hyper_rectangle(tria_vec[5],
                                              {1, 1} /* refinements x,y */,
                                              Point<2>(X_2, 0.0),
                                              Point<2>(X_C + R + L_FLAG, Y_C - T / 2.0));

    GridGenerator::subdivided_hyper_rectangle(tria_vec[6],
                                              {1, 1} /* refinements x,y */,
                                              Point<2>(X_2, Y_C + T / 2.0),
                                              Point<2>(X_C + R + L_FLAG, H));

    GridGenerator::general_cell(tria_vec[7],
                                {Point<2>(X_C + R + L_FLAG, 0.0),
                                 Point<2>(X_3, 0.0),
                                 Point<2>(X_C + R + L_FLAG, Y_C - T / 2.0),
                                 Point<2>(X_3, Y_3)});

    GridGenerator::general_cell(tria_vec[8],
                                {Point<2>(X_C + R + L_FLAG, Y_C + T / 2.0),
                                 Point<2>(X_3, 2.0 * Y_3),
                                 Point<2>(X_C + R + L_FLAG, H),
                                 Point<2>(X_3, H)});

    GridGenerator::general_cell(tria_vec[9],
                                {Point<2>(X_C + R + L_FLAG, Y_C - T / 2.0),
                                 Point<2>(X_3, Y_3),
                                 Point<2>(X_C + R + L_FLAG, Y_C + T / 2.0),
                                 Point<2>(X_3, 2.0 * Y_3)});

    GridGenerator::subdivided_hyper_rectangle(tria_vec[10],
                                              {8, 3} /* refinements x,y */,
                                              Point<2>(X_3, 0.0),
                                              Point<2>(L, H));

    std::vector<Triangulation<2> const *> tria_vec_ptr(tria_vec.size());
    for(unsigned int i = 0; i < tria_vec.size(); ++i)
      tria_vec_ptr[i] = &tria_vec[i];

    GridGenerator::merge_triangulations(tria_vec_ptr, tria);
  }

  void create_triangulation(Triangulation<3> & tria)
  {
    (void)tria;

    AssertThrow(false, ExcMessage("not implemented."));
  }

  void
  create_grid_fluid(
    std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
    unsigned int const                                n_refine_space,
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
      periodic_faces)
  {
    (void)periodic_faces;

    create_triangulation(*triangulation);

    triangulation->set_all_manifold_ids(0);

    Point<dim> center;
    center[0] = X_C;
    center[1] = Y_C;

    for(auto cell : triangulation->active_cell_iterators())
    {
      for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      {
        double const x   = cell->face(f)->center()(0);
        double const y   = cell->face(f)->center()(1);
        double const TOL = 1.e-12;

        // TODO
        // inflow: set boundary ID to 1
        if(std::fabs(x - X_0) < TOL)
        {
          cell->face(f)->set_boundary_id(1);
        }

        if(std::fabs(x - L) < TOL)
        {
          cell->face(f)->set_boundary_id(2);
        }

        if(std::fabs(cell->face(f)->center().distance(center)) < R + TOL)
        {
          cell->face(f)->set_boundary_id(BOUNDARY_ID_CYLINDER);
        }

        if(std::fabs(y - (Y_C - T / 2.0)) < TOL || std::fabs(y - (Y_C + T / 2.0)) < TOL ||
           (std::fabs(y - Y_C) < T / 2.0 + TOL && std::fabs(x - (X_C + R + L_FLAG)) < TOL))
        {
          cell->face(f)->set_boundary_id(BOUNDARY_ID_FLAG);
        }

        // manifold IDs
        bool face_at_sphere_boundary = true;
        for(unsigned int v = 0; v < GeometryInfo<2 - 1>::vertices_per_cell; ++v)
        {
          if(std::abs(center.distance(cell->face(f)->vertex(v)) - R) > TOL)
            face_at_sphere_boundary = false;
        }
        if(face_at_sphere_boundary)
        {
          face_ids.push_back(f);
          unsigned int manifold_id = MANIFOLD_ID + manifold_ids.size() + 1;
          cell->set_all_manifold_ids(manifold_id);
          manifold_ids.push_back(manifold_id);
        }
      }

      for(unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
      {
        bool face_at_sphere_boundary = true;
        for(unsigned int v = 0; v < GeometryInfo<2 - 1>::vertices_per_cell; ++v)
        {
          if(std::abs(center.distance(cell->face(f)->vertex(v)) - R) > 1e-12)
            face_at_sphere_boundary = false;
        }
        if(face_at_sphere_boundary)
        {
          face_ids.push_back(f);
          unsigned int manifold_id = MANIFOLD_ID + manifold_ids.size() + 1;
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
      for(auto cell : triangulation->active_cell_iterators())
      {
        if(cell->manifold_id() == manifold_ids[i])
        {
          manifold_vec[i] = std::shared_ptr<Manifold<dim>>(static_cast<Manifold<dim> *>(
            new OneSidedCylindricalManifold<dim>(cell, face_ids[i], center)));
          triangulation->set_manifold(manifold_ids[i], *(manifold_vec[i]));
        }
      }
    }

    triangulation->refine_global(n_refine_space);
  }


  void set_boundary_conditions_poisson(
    std::shared_ptr<ConvDiff::BoundaryDescriptor<1, dim>> boundary_descriptor)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // these lines show exemplarily how the boundary descriptors are filled
    boundary_descriptor->dirichlet_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc.insert(pair(1, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc.insert(pair(2, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc.insert(pair(3, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc.insert(pair(4, new MeshMotion<dim>()));
  }


  void
  set_field_functions_poisson(std::shared_ptr<Poisson::FieldFunctions<dim>> field_functions)
  {
    // these lines show exemplarily how the field functions are filled
    field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  void
  set_boundary_conditions_fluid(
    std::shared_ptr<IncNS::BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
    std::shared_ptr<IncNS::BoundaryDescriptorP<dim>> boundary_descriptor_pressure)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // fill boundary descriptor velocity
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_velocity->dirichlet_bc.insert(pair(1, new InflowBC<dim>()));
    boundary_descriptor_velocity->neumann_bc.insert(pair(2, new Functions::ZeroFunction<dim>()));
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(3, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_velocity->dirichlet_bc.insert(pair(4, new VelocityBendingWall<dim>()));

    // fill boundary descriptor pressure
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_pressure->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_pressure->dirichlet_bc.insert(pair(2, new Functions::ZeroFunction<dim>(1)));
    boundary_descriptor_pressure->neumann_bc.insert(pair(3, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_pressure->neumann_bc.insert(pair(4, new Functions::ZeroFunction<dim>(dim)));
  }

  void
  set_field_functions_fluid(std::shared_ptr<IncNS::FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<IncNS::PostProcessorBase<dim, Number>>
  construct_postprocessor_fluid(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    IncNS::PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output         = true;
    pp_data.output_data.output_folder        = output_directory + "vtu/";
    pp_data.output_data.output_name          = output_name;
    pp_data.output_data.write_boundary_IDs   = true;
    pp_data.output_data.output_start_time    = 0.0;
    pp_data.output_data.output_interval_time = END_TIME / 100;
    pp_data.output_data.write_higher_order   = false;
    pp_data.output_data.degree               = degree;

    std::shared_ptr<IncNS::PostProcessorBase<dim, Number>> pp;
    pp.reset(new IncNS::PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace CylinderWithFlag
} // namespace FSI

#endif /* APPLICATIONS_FSI_CYLINDER_WITH_FLAG_H_ */
