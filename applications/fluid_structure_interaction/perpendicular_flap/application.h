/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2022 by the ExaDG authors
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

#ifndef APPLICATIONS_FSI_PERPENDICULAR_FLAP_H_
#define APPLICATIONS_FSI_PERPENDICULAR_FLAP_H_

#include <exadg/grid/one_sided_cylindrical_manifold.h>

namespace ExaDG
{
// Perpendicular flap
double const U_MEAN          = 10;
double const FLUID_VISCOSITY = 1;
double const FLUID_DENSITY   = 1;

double const DENSITY_STRUCTURE       = 3.0e3;
double const POISSON_RATIO_STRUCTURE = 0.3;
// 1538462 being the shear modulus
double const E_STRUCTURE = 1538462 * 2.0 * (1.0 + POISSON_RATIO_STRUCTURE);

// physical dimensions (diameter D and center coordinate Y_C can be varied)
double const X_0         = -3.0; // origin (x-coordinate)
double const Y_0         = 0.0;  // origin (y-coordinate)
double const L           = 3.0;  // x-coordinate of outflow boundary
double const H           = 4.0;  // height of channel
double const CENTER_S    = 0;    // Center of the flap
double const THICKNESS_S = 0.1;  // Thickness of the flap
double const HEIGHT_S    = 1.0;  // Height of the flap
// boundary conditions
dealii::types::boundary_id const BOUNDARY_ID_WALLS       = 0;
dealii::types::boundary_id const BOUNDARY_ID_INFLOW      = 1;
dealii::types::boundary_id const BOUNDARY_ID_OUTFLOW     = 2;
dealii::types::boundary_id const BOUNDARY_ID_BOTTOM_WALL = 3;
dealii::types::boundary_id const BOUNDARY_ID_FLAG        = 4;

bool STRUCTURE_COVERS_FLAG_ONLY = true;

unsigned int N_CELLS_FLAG_Y = 5;

double const       END_TIME                            = 5;
double const       DELTA_T                             = 0.01;
double const       OUTPUT_INTERVAL_TIME                = 0.01;
unsigned int const OUTPUT_SOLVER_INFO_EVERY_TIME_STEPS = 1;

double const REL_TOL = 1.e-6;
double const ABS_TOL = 1.e-8;

double const REL_TOL_LINEARIZED = 1.e-6;
double const ABS_TOL_LINEARIZED = 1.e-12;

template<int dim>
class InflowBC : public dealii::Function<dim>
{
public:
  InflowBC() : dealii::Function<dim>(dim, 0.0)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const
  {
    (void)p;
    double result = 0.0;

    if(component == 0)
      result = U_MEAN;

    return result;
  }
};

template<int dim>
class SpatiallyVaryingE : public dealii::Function<dim>
{
public:
  SpatiallyVaryingE() : dealii::Function<dim>(1, 0.0)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const
  {
    (void)component;

    double const length_scale = 2.0 * THICKNESS_S;
    double const x            = p[0];
    double const value =
      (std::abs(x) < length_scale) ? std::cos(x / length_scale * 0.5 * dealii::numbers::PI) : 0.0;

    return 1. + 100. * value * value;
  }
};

namespace FluidFSI
{
template<int dim, typename Number>
class Application : public FluidFSI::ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : FluidFSI::ApplicationBase<dim, Number>(input_file, comm)
  {
  }

private:
  void
  set_parameters() final
  {
    using namespace IncNS;

    Parameters & param = this->param;

    // MATHEMATICAL MODEL
    param.problem_type                   = ProblemType::Unsteady;
    param.equation_type                  = EquationType::NavierStokes;
    param.formulation_viscous_term       = FormulationViscousTerm::LaplaceFormulation;
    param.formulation_convective_term    = FormulationConvectiveTerm::ConvectiveFormulation;
    param.use_outflow_bc_convective_term = true;
    param.right_hand_side                = false;

    // ALE
    param.ale_formulation                     = true;
    param.mesh_movement_type                  = MeshMovementType::Elasticity;
    param.neumann_with_variable_normal_vector = false;

    // PHYSICAL QUANTITIES
    param.start_time = 0.0;
    param.end_time   = END_TIME;
    param.viscosity  = FLUID_VISCOSITY;
    param.density    = FLUID_DENSITY;

    // TEMPORAL DISCRETIZATION
    param.solver_type                     = SolverType::Unsteady;
    param.temporal_discretization         = TemporalDiscretization::BDFDualSplittingScheme;
    param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    param.order_time_integrator           = 1;
    param.start_with_low_order            = true;
    param.adaptive_time_stepping          = false;
    param.calculation_of_time_step_size   = TimeStepCalculation::UserSpecified;
    param.time_step_size                  = DELTA_T;
    param.max_velocity                    = U_MEAN;
    param.cfl                             = 0.5;
    param.cfl_exponent_fe_degree_velocity = 1.5;

    // output of solver information
    param.solver_info_data.interval_time_steps = OUTPUT_SOLVER_INFO_EVERY_TIME_STEPS;

    // restart
    param.restarted_simulation             = false;
    param.restart_data.write_restart       = false;
    param.restart_data.interval_time       = 0.25;
    param.restart_data.interval_wall_time  = 1.e6;
    param.restart_data.interval_time_steps = 1e8;
    param.restart_data.filename            = "output/vortex/vortex";


    // SPATIAL DISCRETIZATION
    param.grid.triangulation_type = TriangulationType::Distributed;
    param.grid.mapping_degree     = param.degree_u;
    param.degree_p                = DegreePressure::MixedOrder;

    // convective term
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
    param.continuity_penalty_use_boundary_data       = true;
    param.apply_penalty_terms_in_postprocessing_step = true;

    // NUMERICAL PARAMETERS
    param.implement_block_diagonal_preconditioner_matrix_free = false;
    param.use_cell_based_face_loops                           = false;
    param.quad_rule_linearization = QuadratureRuleLinearization::Overintegration32k;

    // PROJECTION METHODS

    // pressure Poisson equation
    param.solver_pressure_poisson              = SolverPressurePoisson::CG;
    param.solver_data_pressure_poisson         = SolverData(1000, ABS_TOL, REL_TOL, 100);
    param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
    param.multigrid_data_pressure_poisson.smoother_data.smoother = MultigridSmoother::Chebyshev;
    param.multigrid_data_pressure_poisson.smoother_data.preconditioner =
      PreconditionerSmoother::PointJacobi;
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
    param.formulation_convective_term_bc = FormulationConvectiveTerm::ConvectiveFormulation;

    // viscous step
    param.solver_viscous         = SolverViscous::CG;
    param.solver_data_viscous    = SolverData(1000, ABS_TOL, REL_TOL);
    param.preconditioner_viscous = PreconditionerViscous::PointJacobi;


    // PRESSURE-CORRECTION SCHEME

    // formulation
    param.order_pressure_extrapolation =
      std::min(2, (int)param.order_time_integrator) - 1; // J_p = J-1, but not larger than 1
    param.rotational_formulation = true;

    // momentum step

    // Newton solver
    param.newton_solver_data_momentum = Newton::SolverData(100, ABS_TOL, REL_TOL);

    // linear solver
    param.solver_momentum = SolverMomentum::FGMRES;
    if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      param.solver_data_momentum = SolverData(1e4, ABS_TOL_LINEARIZED, REL_TOL_LINEARIZED, 100);
    else
      param.solver_data_momentum = SolverData(1e4, ABS_TOL, REL_TOL, 100);
    param.update_preconditioner_momentum = false;
    param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;


    // COUPLED NAVIER-STOKES SOLVER
    param.use_scaling_continuity = false;

    // nonlinear solver (Newton solver)
    param.newton_solver_data_coupled = Newton::SolverData(100, ABS_TOL, REL_TOL);

    // linear solver
    param.solver_coupled = SolverCoupled::FGMRES;
    if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      param.solver_data_coupled = SolverData(1e4, ABS_TOL_LINEARIZED, REL_TOL_LINEARIZED, 100);
    else
      param.solver_data_coupled = SolverData(1e4, ABS_TOL, REL_TOL, 100);

    // preconditioner linear solver
    param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    param.update_preconditioner_coupled = false;

    // preconditioner momentum block
    param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block =
      SchurComplementPreconditioner::PressureConvectionDiffusion;
  }

  void
  create_triangulation(dealii::Triangulation<2> & tria)
  {
    std::vector<dealii::Triangulation<2>> tria_vec;
    tria_vec.resize(5);

    dealii::GridGenerator::subdivided_hyper_rectangle(
      tria_vec[0],
      {16, 4} /* subdivisions x,y */,
      dealii::Point<2>(X_0, Y_0),
      dealii::Point<2>(CENTER_S - (0.5 * THICKNESS_S), HEIGHT_S));

    dealii::GridGenerator::subdivided_hyper_rectangle(
      tria_vec[1],
      {16, 12} /* subdivisions x,y */,
      dealii::Point<2>(X_0, HEIGHT_S),
      dealii::Point<2>(CENTER_S - (0.5 * THICKNESS_S), H));

    dealii::GridGenerator::subdivided_hyper_rectangle(
      tria_vec[2],
      {1, 12} /* subdivisions x,y */,
      dealii::Point<2>(CENTER_S - (0.5 * THICKNESS_S), HEIGHT_S),
      dealii::Point<2>(CENTER_S + (0.5 * THICKNESS_S), H));

    dealii::GridGenerator::subdivided_hyper_rectangle(
      tria_vec[3],
      {16, 12} /* subdivisions x,y */,
      dealii::Point<2>(CENTER_S + (0.5 * THICKNESS_S), HEIGHT_S),
      dealii::Point<2>(L, H));

    dealii::GridGenerator::subdivided_hyper_rectangle(
      tria_vec[4],
      {16, 4} /* subdivisions x,y */,
      dealii::Point<2>(CENTER_S + (0.5 * THICKNESS_S), Y_0),
      dealii::Point<2>(L, HEIGHT_S));

    std::vector<dealii::Triangulation<2> const *> tria_vec_ptr(tria_vec.size());
    for(unsigned int i = 0; i < tria_vec.size(); ++i)
      tria_vec_ptr[i] = &tria_vec[i];

    dealii::GridGenerator::merge_triangulations(tria_vec_ptr, tria);
  }

  void
  create_triangulation(dealii::Triangulation<3> & tria)
  {
    (void)tria;

    AssertThrow(false, dealii::ExcMessage("not implemented."));
  }

  void
  create_grid() final
  {
    create_triangulation(*this->grid->triangulation);

    this->grid->triangulation->set_all_manifold_ids(0);

    for(auto cell : this->grid->triangulation->active_cell_iterators())
    {
      for(unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
      {
        if(cell->face(f)->at_boundary())
        {
          double const x   = cell->face(f)->center()(0);
          double const y   = cell->face(f)->center()(1);
          double const TOL = 1.e-12;

          if(std::fabs(x - X_0) < TOL)
          {
            cell->face(f)->set_boundary_id(BOUNDARY_ID_INFLOW);
          }
          else if(std::fabs(y - Y_0) < TOL || std::fabs(y - H) < TOL)
          {
            cell->face(f)->set_boundary_id(BOUNDARY_ID_WALLS);
          }
          else if(std::fabs(x - L) < TOL)
          {
            cell->face(f)->set_boundary_id(BOUNDARY_ID_OUTFLOW);
          }
          else if(std::fabs(x - CENTER_S) < (THICKNESS_S * 0.5) + TOL && y < HEIGHT_S + TOL)
          {
            cell->face(f)->set_boundary_id(BOUNDARY_ID_FLAG);
          }
          else
          {
            AssertThrow(false, dealii::ExcNotImplemented());
          }
        }
      }
    }
    this->grid->triangulation->refine_global(this->param.grid.n_refine_global);
  }

  void
  set_boundary_descriptor() final
  {
    std::shared_ptr<IncNS::BoundaryDescriptor<dim>> boundary_descriptor = this->boundary_descriptor;

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<FunctionCached<1, dim>>>
      pair_fsi;

    // fill boundary descriptor velocity
    boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(BOUNDARY_ID_WALLS, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(BOUNDARY_ID_INFLOW, new InflowBC<dim>()));
    boundary_descriptor->velocity->neumann_bc.insert(
      pair(BOUNDARY_ID_OUTFLOW, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(BOUNDARY_ID_BOTTOM_WALL, new dealii::Functions::ZeroFunction<dim>(dim)));
    // fluid-structure interface
    boundary_descriptor->velocity->dirichlet_cached_bc.insert(
      pair_fsi(BOUNDARY_ID_FLAG, new FunctionCached<1, dim>()));

    // fill boundary descriptor pressure
    boundary_descriptor->pressure->neumann_bc.insert(BOUNDARY_ID_WALLS);
    boundary_descriptor->pressure->neumann_bc.insert(BOUNDARY_ID_INFLOW);
    boundary_descriptor->pressure->dirichlet_bc.insert(
      pair(BOUNDARY_ID_OUTFLOW, new dealii::Functions::ZeroFunction<dim>(1)));
    boundary_descriptor->pressure->neumann_bc.insert(BOUNDARY_ID_BOTTOM_WALL);
    boundary_descriptor->pressure->neumann_bc.insert(BOUNDARY_ID_FLAG);
  }

  void
  set_field_functions() final
  {
    std::shared_ptr<IncNS::FieldFunctions<dim>> field_functions = this->field_functions;

    field_functions->initial_solution_velocity.reset(new InflowBC<dim>());
    field_functions->initial_solution_pressure.reset(new dealii::Functions::ZeroFunction<dim>(1));
    field_functions->analytical_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<IncNS::PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    IncNS::PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output       = true;
    pp_data.output_data.directory          = this->output_parameters.directory;
    pp_data.output_data.filename           = this->output_parameters.filename + "_fluid";
    pp_data.output_data.write_boundary_IDs = true;
    pp_data.output_data.write_processor_id = true;
    pp_data.output_data.start_time         = 0.0;
    pp_data.output_data.interval_time      = OUTPUT_INTERVAL_TIME;
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.degree             = this->param.degree_u;

    std::shared_ptr<IncNS::PostProcessorBase<dim, Number>> pp;
    pp.reset(new IncNS::PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  void
  set_parameters_ale_poisson() final
  {
    using namespace Poisson;

    Parameters & param = this->ale_poisson_param;

    // MATHEMATICAL MODEL
    param.right_hand_side = false;

    // SPATIAL DISCRETIZATION
    param.spatial_discretization = SpatialDiscretization::CG;
    param.degree                 = this->param.grid.mapping_degree;

    // SOLVER
    param.solver         = Poisson::Solver::FGMRES;
    param.solver_data    = SolverData(1e4, ABS_TOL, REL_TOL, 100);
    param.preconditioner = Preconditioner::Multigrid;

    param.multigrid_data.type                          = MultigridType::phMG;
    param.multigrid_data.p_sequence                    = PSequenceType::Bisect;
    param.multigrid_data.smoother_data.smoother        = MultigridSmoother::Chebyshev;
    param.multigrid_data.coarse_problem.solver         = MultigridCoarseGridSolver::CG;
    param.multigrid_data.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::AMG;
  }

  void
  set_boundary_descriptor_ale_poisson() final
  {
    std::shared_ptr<Poisson::BoundaryDescriptor<1, dim>> boundary_descriptor =
      this->ale_poisson_boundary_descriptor;

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<FunctionCached<1, dim>>>
      pair_fsi;

    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_WALLS, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_INFLOW, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_OUTFLOW, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_BOTTOM_WALL, new dealii::Functions::ZeroFunction<dim>(dim)));

    // fluid-structure interface
    boundary_descriptor->dirichlet_cached_bc.insert(
      pair_fsi(BOUNDARY_ID_FLAG, new FunctionCached<1, dim>()));
  }


  void
  set_field_functions_ale_poisson() final
  {
    std::shared_ptr<Poisson::FieldFunctions<dim>> field_functions =
      this->ale_poisson_field_functions;

    field_functions->initial_solution.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  void
  set_parameters_ale_elasticity() final
  {
    using namespace Structure;

    Parameters & param = this->ale_elasticity_param;

    param.problem_type         = ProblemType::Steady;
    param.body_force           = false;
    param.pull_back_body_force = false;
    param.large_deformation    = false;
    param.pull_back_traction   = false;

    param.degree = this->param.grid.mapping_degree;

    param.newton_solver_data = Newton::SolverData(1e4, ABS_TOL, REL_TOL);
    param.solver             = Structure::Solver::FGMRES;
    if(param.large_deformation)
      param.solver_data = SolverData(1e4, ABS_TOL_LINEARIZED, REL_TOL_LINEARIZED, 100);
    else
      param.solver_data = SolverData(1e4, ABS_TOL, REL_TOL, 100);
    param.preconditioner                               = Preconditioner::Multigrid;
    param.multigrid_data.type                          = MultigridType::phMG;
    param.multigrid_data.coarse_problem.solver         = MultigridCoarseGridSolver::CG;
    param.multigrid_data.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::AMG;

    param.update_preconditioner                         = param.large_deformation;
    param.update_preconditioner_every_newton_iterations = 10;
  }

  void
  set_boundary_descriptor_ale_elasticity() final
  {
    std::shared_ptr<Structure::BoundaryDescriptor<dim>> boundary_descriptor =
      this->ale_elasticity_boundary_descriptor;

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
                                                                                  pair;
    typedef typename std::pair<dealii::types::boundary_id, dealii::ComponentMask> pair_mask;

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<FunctionCached<1, dim>>>
      pair_fsi;

    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_WALLS, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_component_mask.insert(
      pair_mask(BOUNDARY_ID_WALLS, dealii::ComponentMask()));
    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_INFLOW, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_component_mask.insert(
      pair_mask(BOUNDARY_ID_INFLOW, dealii::ComponentMask()));
    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_OUTFLOW, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_component_mask.insert(
      pair_mask(BOUNDARY_ID_OUTFLOW, dealii::ComponentMask()));
    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_BOTTOM_WALL, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_component_mask.insert(
      pair_mask(BOUNDARY_ID_BOTTOM_WALL, dealii::ComponentMask()));

    // fluid-structure interface
    boundary_descriptor->dirichlet_cached_bc.insert(
      pair_fsi(BOUNDARY_ID_FLAG, new FunctionCached<1, dim>()));
  }

  void
  set_material_descriptor_ale_elasticity() final
  {
    std::shared_ptr<Structure::MaterialDescriptor> material_descriptor =
      this->ale_elasticity_material_descriptor;

    using namespace Structure;

    typedef std::pair<dealii::types::material_id, std::shared_ptr<MaterialData>> Pair;

    MaterialType const type         = MaterialType::StVenantKirchhoff;
    Type2D const       two_dim_type = Type2D::PlaneStrain;

    double const                           E       = 1.0;
    double const                           poisson = 0.3;
    std::shared_ptr<dealii::Function<dim>> E_function;
    E_function.reset(new SpatiallyVaryingE<dim>());
    material_descriptor->insert(
      Pair(0, new StVenantKirchhoffData<dim>(type, E, poisson, two_dim_type, E_function)));
  }

  void
  set_field_functions_ale_elasticity() final
  {
    std::shared_ptr<Structure::FieldFunctions<dim>> field_functions =
      this->ale_elasticity_field_functions;

    field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    field_functions->initial_displacement.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    field_functions->initial_velocity.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }
};
} // namespace FluidFSI

// Structure
namespace StructureFSI
{
template<int dim, typename Number>
class Application : public StructureFSI::ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : StructureFSI::ApplicationBase<dim, Number>(input_file, comm)
  {
  }

  void
  set_parameters() final
  {
    using namespace Structure;

    Parameters & param = this->param;

    param.problem_type         = ProblemType::Unsteady;
    param.body_force           = false;
    param.pull_back_body_force = false;
    param.large_deformation    = true;
    param.pull_back_traction   = true;

    param.density = DENSITY_STRUCTURE;

    param.start_time                           = 0.0;
    param.end_time                             = END_TIME;
    param.time_step_size                       = DELTA_T;
    param.gen_alpha_type                       = GenAlphaType::GenAlpha;
    param.spectral_radius                      = 0.9;
    param.solver_info_data.interval_time_steps = OUTPUT_SOLVER_INFO_EVERY_TIME_STEPS;

    param.grid.triangulation_type = TriangulationType::Distributed;
    param.grid.mapping_degree     = param.degree;

    param.newton_solver_data = Newton::SolverData(1e4, ABS_TOL, REL_TOL);
    param.solver             = Structure::Solver::FGMRES;
    if(param.large_deformation)
      param.solver_data = SolverData(1e4, ABS_TOL_LINEARIZED, REL_TOL_LINEARIZED, 100);
    else
      param.solver_data = SolverData(1e4, ABS_TOL, REL_TOL, 100);
    param.preconditioner                               = Preconditioner::Multigrid;
    param.multigrid_data.type                          = MultigridType::phMG;
    param.multigrid_data.coarse_problem.solver         = MultigridCoarseGridSolver::CG;
    param.multigrid_data.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::AMG;

    param.update_preconditioner                         = true;
    param.update_preconditioner_every_time_steps        = 10;
    param.update_preconditioner_every_newton_iterations = 10;
  }

  void
  create_triangulation(dealii::Triangulation<2> & tria)
  {
    if(STRUCTURE_COVERS_FLAG_ONLY)
    {
      dealii::GridGenerator::subdivided_hyper_rectangle(
        tria,
        {1, N_CELLS_FLAG_Y} /* subdivisions x,y */,
        dealii::Point<2>(CENTER_S - (THICKNESS_S * 0.5), 0),
        dealii::Point<2>(CENTER_S + (THICKNESS_S * 0.5), HEIGHT_S));
    }
    else
    {
      Assert(false, dealii::ExcNotImplemented());
    }
  }

  void
  create_triangulation(dealii::Triangulation<3> & tria)
  {
    (void)tria;

    AssertThrow(false, dealii::ExcMessage("not implemented."));
  }

  void
  create_grid() final
  {
    create_triangulation(*this->grid->triangulation);

    this->grid->triangulation->set_all_manifold_ids(0);

    for(auto cell : this->grid->triangulation->active_cell_iterators())
    {
      for(unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
      {
        double const y   = cell->face(f)->center()(1);
        double const TOL = 1.e-12;

        if(cell->face(f)->at_boundary())
        {
          if(y < 0 + TOL)
          {
            cell->face(f)->set_boundary_id(BOUNDARY_ID_BOTTOM_WALL);
          }
          else if(y > 0 + TOL)
          {
            cell->face(f)->set_boundary_id(BOUNDARY_ID_FLAG);
          }
          else
          {
            AssertThrow(false, dealii::ExcInternalError());
          }
        }
      }
    }

    this->grid->triangulation->refine_global(this->param.grid.n_refine_global);
  }

  void
  set_boundary_descriptor() final
  {
    std::shared_ptr<Structure::BoundaryDescriptor<dim>> boundary_descriptor =
      this->boundary_descriptor;

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
                                                                                  pair;
    typedef typename std::pair<dealii::types::boundary_id, dealii::ComponentMask> pair_mask;

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<FunctionCached<1, dim>>>
      pair_fsi;

    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_BOTTOM_WALL, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_component_mask.insert(
      pair_mask(BOUNDARY_ID_BOTTOM_WALL, dealii::ComponentMask()));

    // fluid-structure interface
    boundary_descriptor->neumann_cached_bc.insert(
      pair_fsi(BOUNDARY_ID_FLAG, new FunctionCached<1, dim>()));
  }

  void
  set_field_functions() final
  {
    std::shared_ptr<Structure::FieldFunctions<dim>> field_functions = this->field_functions;

    field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    field_functions->initial_displacement.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    field_functions->initial_velocity.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  void
  set_material_descriptor() final
  {
    std::shared_ptr<Structure::MaterialDescriptor> material_descriptor = this->material_descriptor;

    using namespace Structure;

    typedef std::pair<dealii::types::material_id, std::shared_ptr<MaterialData>> Pair;

    MaterialType const type         = MaterialType::StVenantKirchhoff;
    Type2D const       two_dim_type = Type2D::PlaneStrain;

    material_descriptor->insert(Pair(
      0, new StVenantKirchhoffData<dim>(type, E_STRUCTURE, POISSON_RATIO_STRUCTURE, two_dim_type)));
  }

  std::shared_ptr<Structure::PostProcessor<dim, Number>>
  create_postprocessor() final
  {
    using namespace Structure;

    PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output       = true;
    pp_data.output_data.directory          = this->output_parameters.directory;
    pp_data.output_data.filename           = this->output_parameters.filename + "_structure";
    pp_data.output_data.write_processor_id = true;
    pp_data.output_data.start_time         = 0.0;
    pp_data.output_data.interval_time      = OUTPUT_INTERVAL_TIME;
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.degree             = this->param.degree;

    std::shared_ptr<PostProcessor<dim, Number>> post(
      new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return post;
  }
};

} // namespace StructureFSI

namespace FSI
{
template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
  {
    this->structure = std::make_shared<StructureFSI::Application<dim, Number>>(input_file, comm);
    this->fluid     = std::make_shared<FluidFSI::Application<dim, Number>>(input_file, comm);
  }
};

} // namespace FSI

} // namespace ExaDG

#include <exadg/fluid_structure_interaction/user_interface/implement_get_application.h>

#endif
