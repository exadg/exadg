/*
 * vortex.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_VORTEX_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_VORTEX_H_

#include "grid/deformed_cube_manifold.h"
#include "grid/mesh_movement_functions.h"
#include "grid/one_sided_cylindrical_manifold.h"

namespace ExaDG
{
namespace IncNS
{
namespace Vortex
{
using namespace dealii;

enum class MeshType
{
  UniformCartesian,
  ComplexSurfaceManifold,
  ComplexVolumeManifold,
  Curvilinear
};

template<int dim>
class AnalyticalSolutionVelocity : public Function<dim>
{
public:
  AnalyticalSolutionVelocity(double const u_x_max, double const viscosity)
    : Function<dim>(dim, 0.0), u_x_max(u_x_max), viscosity(viscosity)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double const t  = this->get_time();
    double const pi = numbers::PI;

    double result = 0.0;
    if(component == 0)
      result = -u_x_max * std::sin(2.0 * pi * p[1]) * std::exp(-4.0 * pi * pi * viscosity * t);
    else if(component == 1)
      result = u_x_max * std::sin(2.0 * pi * p[0]) * std::exp(-4.0 * pi * pi * viscosity * t);

    return result;
  }

private:
  double const u_x_max, viscosity;
};

template<int dim>
class AnalyticalSolutionPressure : public Function<dim>
{
public:
  AnalyticalSolutionPressure(double const u_x_max, double const viscosity)
    : Function<dim>(1 /*n_components*/, 0.0), u_x_max(u_x_max), viscosity(viscosity)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int /*component*/) const
  {
    double const t  = this->get_time();
    double const pi = numbers::PI;

    double const result = -u_x_max * std::cos(2 * pi * p[0]) * std::cos(2 * pi * p[1]) *
                          std::exp(-8.0 * pi * pi * viscosity * t);

    return result;
  }

private:
  double const u_x_max, viscosity;
};

template<int dim>
class NeumannBoundaryVelocity : public Function<dim>
{
public:
  NeumannBoundaryVelocity(double const                 u_x_max,
                          double const                 viscosity,
                          FormulationViscousTerm const formulation_viscous)
    : Function<dim>(dim, 0.0),
      u_x_max(u_x_max),
      viscosity(viscosity),
      formulation_viscous(formulation_viscous)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double const t  = this->get_time();
    double const pi = numbers::PI;

    double result = 0.0;
    // clang-format off
    // prescribe F_nu(u) / nu = grad(u)
    if(formulation_viscous == FormulationViscousTerm::LaplaceFormulation)
    {
      if(component==0)
      {
        if( (std::abs(p[1]+0.5)< 1e-12) && (p[0]<0) )
          result = u_x_max*2.0*pi*std::cos(2.0*pi*p[1])*std::exp(-4.0*pi*pi*viscosity*t);
        else if( (std::abs(p[1]-0.5)< 1e-12) && (p[0]>0) )
          result = -u_x_max*2.0*pi*std::cos(2.0*pi*p[1])*std::exp(-4.0*pi*pi*viscosity*t);
      }
      else if(component==1)
      {
        if( (std::abs(p[0]+0.5)< 1e-12) && (p[1]>0) )
          result = -u_x_max*2.0*pi*std::cos(2.0*pi*p[0])*std::exp(-4.0*pi*pi*viscosity*t);
        else if((std::abs(p[0]-0.5)< 1e-12) && (p[1]<0) )
          result = u_x_max*2.0*pi*std::cos(2.0*pi*p[0])*std::exp(-4.0*pi*pi*viscosity*t);
      }
    }
    // prescribe F_nu(u) / nu = ( grad(u) + grad(u)^T )
    else if(formulation_viscous == FormulationViscousTerm::DivergenceFormulation)
    {
      if(component==0)
      {
        if( (std::abs(p[1]+0.5)< 1e-12) && (p[0]<0) )
          result = -u_x_max*2.0*pi*(std::cos(2.0*pi*p[0]) - std::cos(2.0*pi*p[1]))*std::exp(-4.0*pi*pi*viscosity*t);
        else if( (std::abs(p[1]-0.5)< 1e-12) && (p[0]>0) )
          result = u_x_max*2.0*pi*(std::cos(2.0*pi*p[0]) - std::cos(2.0*pi*p[1]))*std::exp(-4.0*pi*pi*viscosity*t);
      }
      else if(component==1)
      {
        if( (std::abs(p[0]+0.5)< 1e-12) && (p[1]>0) )
          result = -u_x_max*2.0*pi*(std::cos(2.0*pi*p[0]) - std::cos(2.0*pi*p[1]))*std::exp(-4.0*pi*pi*viscosity*t);
        else if((std::abs(p[0]-0.5)< 1e-12) && (p[1]<0) )
          result = u_x_max*2.0*pi*(std::cos(2.0*pi*p[0]) - std::cos(2.0*pi*p[1]))*std::exp(-4.0*pi*pi*viscosity*t);
      }
    }
    else
    {
      AssertThrow(formulation_viscous == FormulationViscousTerm::LaplaceFormulation ||
                  formulation_viscous == FormulationViscousTerm::DivergenceFormulation,
                  ExcMessage("Specified formulation of viscous term is not implemented!"));
    }
    // clang-format on

    return result;
  }

private:
  double const                 u_x_max, viscosity;
  FormulationViscousTerm const formulation_viscous;
};

template<int dim>
class NeumannBoundaryVelocityALE : public FunctionWithNormal<dim>
{
public:
  NeumannBoundaryVelocityALE(double const                 u_x_max,
                             double const                 viscosity,
                             FormulationViscousTerm const formulation_viscous)
    : FunctionWithNormal<dim>(dim, 0.0),
      u_x_max(u_x_max),
      viscosity(viscosity),
      formulation_viscous(formulation_viscous)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double const t  = this->get_time();
    double const pi = numbers::PI;

    Tensor<1, dim> const n = this->get_normal_vector();

    double result = 0.0;
    // prescribe F_nu(u) / nu = grad(u)
    if(formulation_viscous == FormulationViscousTerm::LaplaceFormulation)
    {
      if(component == 0)
        result = 0 * n[0] - u_x_max * 2.0 * pi * std::cos(2 * pi * p[1]) *
                              std::exp(-4.0 * pi * pi * viscosity * t) * n[1];
      else if(component == 1)
        result = u_x_max * 2.0 * pi * std::cos(2 * pi * p[0]) *
                   std::exp(-4.0 * pi * pi * viscosity * t) * n[0] +
                 0 * n[1];
    }
    // prescribe F_nu(u) / nu = ( grad(u) + grad(u)^T )
    else if(formulation_viscous == FormulationViscousTerm::DivergenceFormulation)
    {
      if(component == 0)
        result = 0 * n[0] + u_x_max * 2.0 * pi *
                              (std::cos(2.0 * pi * p[0]) - std::cos(2.0 * pi * p[1])) *
                              std::exp(-4.0 * pi * pi * viscosity * t) * n[1];
      else if(component == 1)
        result = u_x_max * 2.0 * pi * (std::cos(2.0 * pi * p[0]) - std::cos(2.0 * pi * p[1])) *
                   std::exp(-4.0 * pi * pi * viscosity * t) * n[0] +
                 0 * n[1];
    }
    else
    {
      AssertThrow(formulation_viscous == FormulationViscousTerm::LaplaceFormulation ||
                    formulation_viscous == FormulationViscousTerm::DivergenceFormulation,
                  ExcMessage("Specified formulation of viscous term is not implemented!"));
    }

    return result;
  }

private:
  double const                 u_x_max, viscosity;
  FormulationViscousTerm const formulation_viscous;
};


template<int dim>
class PressureBC_dudt : public Function<dim>
{
public:
  PressureBC_dudt(double const u_x_max, double const viscosity)
    : Function<dim>(dim, 0.0), u_x_max(u_x_max), viscosity(viscosity)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double const t  = this->get_time();
    double const pi = numbers::PI;

    double result = 0.0;
    if(component == 0)
      result = u_x_max * 4.0 * pi * pi * viscosity * std::sin(2.0 * pi * p[1]) *
               std::exp(-4.0 * pi * pi * viscosity * t);
    else if(component == 1)
      result = -u_x_max * 4.0 * pi * pi * viscosity * std::sin(2.0 * pi * p[0]) *
               std::exp(-4.0 * pi * pi * viscosity * t);

    return result;
  }

private:
  double const u_x_max, viscosity;
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

  std::string output_directory = "output/vortex/", output_name = "ale";

  // set problem specific parameters like physical dimensions, etc.
  double const u_x_max   = 1.0;
  double const viscosity = 2.5e-2; // 1.e-2; //2.5e-2;

  double const left  = -0.5;
  double const right = 0.5;

  double const start_time = 0.0;
  double const end_time   = 1.0;

  FormulationViscousTerm const formulation_viscous = FormulationViscousTerm::LaplaceFormulation;

  MeshType const mesh_type = MeshType::UniformCartesian;

  bool const ALE = true;

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.problem_type                = ProblemType::Unsteady;
    param.equation_type               = EquationType::NavierStokes;
    param.formulation_viscous_term    = formulation_viscous;
    param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
    param.right_hand_side             = false;

    // ALE
    param.ale_formulation                     = ALE;
    param.mesh_movement_type                  = MeshMovementType::Analytical;
    param.neumann_with_variable_normal_vector = ALE;

    // PHYSICAL QUANTITIES
    param.start_time = start_time;
    param.end_time   = end_time;
    param.viscosity  = viscosity;


    // TEMPORAL DISCRETIZATION
    param.solver_type                  = SolverType::Unsteady;
    param.temporal_discretization      = TemporalDiscretization::BDFDualSplittingScheme;
    param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
    param.order_time_integrator        = 2;
    param.start_with_low_order         = false;
    param.adaptive_time_stepping       = false;
    param.calculation_of_time_step_size =
      TimeStepCalculation::UserSpecified; // UserSpecified; //CFL;
    param.time_step_size                  = end_time;
    param.max_velocity                    = 1.4 * u_x_max;
    param.cfl                             = 0.2; // 0.4;
    param.cfl_exponent_fe_degree_velocity = 1.5;
    param.c_eff                           = 8.0;
    param.time_integrator_oif             = TimeIntegratorOIF::ExplRK3Stage7Reg2;
    param.cfl_oif                         = param.cfl / 1.0;

    // output of solver information
    param.solver_info_data.interval_time = param.end_time - param.start_time;

    // restart
    param.restarted_simulation             = false;
    param.restart_data.write_restart       = false;
    param.restart_data.interval_time       = 0.25;
    param.restart_data.interval_wall_time  = 1.e6;
    param.restart_data.interval_time_steps = 1e8;
    param.restart_data.filename            = output_directory + output_name;


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
    param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
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
    param.preconditioner_pressure_block = SchurComplementPreconditioner::CahouetChabard;
    param.discretization_of_laplacian   = DiscretizationOfLaplacian::Classical;
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    (void)periodic_faces;

    if(ALE)
    {
      AssertThrow(mesh_type == MeshType::UniformCartesian,
                  ExcMessage("Taylor vortex problem: Parameter mesh_type is invalid for ALE."));
    }

    if(mesh_type == MeshType::UniformCartesian)
    {
      // Uniform Cartesian grid
      GridGenerator::subdivided_hyper_cube(*triangulation, 2, left, right);
    }
    else if(mesh_type == MeshType::ComplexSurfaceManifold)
    {
      // Complex Geometry
      Triangulation<dim> tria1, tria2;
      const double       radius = (right - left) * 0.25;
      const double       width  = right - left;
      GridGenerator::hyper_shell(
        tria1, Point<dim>(), radius, 0.5 * width * std::sqrt(dim), 2 * dim);
      tria1.reset_all_manifolds();
      if(dim == 2)
      {
        GridTools::rotate(numbers::PI / 4, tria1);
      }
      GridGenerator::hyper_ball(tria2, Point<dim>(), radius);
      GridGenerator::merge_triangulations(tria1, tria2, *triangulation);

      triangulation->set_all_manifold_ids(0);
      for(auto cell : triangulation->active_cell_iterators())
      {
        for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          bool face_at_sphere_boundary = true;
          for(unsigned int v = 0; v < GeometryInfo<dim - 1>::vertices_per_cell; ++v)
          {
            if(std::abs(cell->face(f)->vertex(v).norm() - radius) > 1e-12)
              face_at_sphere_boundary = false;
          }
          if(face_at_sphere_boundary)
          {
            cell->face(f)->set_all_manifold_ids(1);
          }
        }
      }
      static const SphericalManifold<dim> spherical_manifold;
      triangulation->set_manifold(1, spherical_manifold);

      // refine globally due to boundary conditions for vortex problem
      triangulation->refine_global(1);
    }
    else if(mesh_type == MeshType::ComplexVolumeManifold)
    {
      // Complex Geometry
      Triangulation<dim> tria1, tria2;
      const double       radius = (right - left) * 0.25;
      const double       width  = right - left;
      Point<dim>         center = Point<dim>();

      GridGenerator::hyper_shell(
        tria1, Point<dim>(), radius, 0.5 * width * std::sqrt(dim), 2 * dim);
      tria1.reset_all_manifolds();
      if(dim == 2)
      {
        GridTools::rotate(numbers::PI / 4, tria1);
      }
      GridGenerator::hyper_ball(tria2, Point<dim>(), radius);
      GridGenerator::merge_triangulations(tria1, tria2, *triangulation);

      // manifolds
      triangulation->set_all_manifold_ids(0);

      // first fill vectors of manifold_ids and face_ids
      std::vector<unsigned int> manifold_ids;
      std::vector<unsigned int> face_ids;

      for(auto cell : triangulation->active_cell_iterators())
      {
        for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          bool face_at_sphere_boundary = true;
          for(unsigned int v = 0; v < GeometryInfo<dim - 1>::vertices_per_cell; ++v)
          {
            if(std::abs(cell->face(f)->vertex(v).norm() - radius) > 1e-12)
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

      // refine globally due to boundary conditions for vortex problem
      triangulation->refine_global(1);
    }
    else if(mesh_type == MeshType::Curvilinear)
    {
      GridGenerator::subdivided_hyper_cube(*triangulation, 2, left, right);

      triangulation->set_all_manifold_ids(1);
      double const                     deformation = 0.1;
      unsigned int const               frequency   = 2;
      static DeformedCubeManifold<dim> manifold(left, right, deformation, frequency);
      triangulation->set_manifold(1, manifold);
    }

    for(auto cell : triangulation->active_cell_iterators())
    {
      for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      {
        if(((std::fabs(cell->face(f)->center()(0) - right) < 1e-12) &&
            (cell->face(f)->center()(1) < 0)) ||
           ((std::fabs(cell->face(f)->center()(0) - left) < 1e-12) &&
            (cell->face(f)->center()(1) > 0)) ||
           ((std::fabs(cell->face(f)->center()(1) - left) < 1e-12) &&
            (cell->face(f)->center()(0) < 0)) ||
           ((std::fabs(cell->face(f)->center()(1) - right) < 1e-12) &&
            (cell->face(f)->center()(0) > 0)))
        {
          cell->face(f)->set_boundary_id(1);
        }
      }
    }

    triangulation->refine_global(n_refine_space);
  }

  void
  set_boundary_conditions(std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
                          std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // fill boundary descriptor velocity
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(0, new AnalyticalSolutionVelocity<dim>(u_x_max, viscosity)));
    if(ALE)
      boundary_descriptor_velocity->neumann_bc.insert(
        pair(1, new NeumannBoundaryVelocityALE<dim>(u_x_max, viscosity, formulation_viscous)));
    else
      boundary_descriptor_velocity->neumann_bc.insert(
        pair(1, new NeumannBoundaryVelocity<dim>(u_x_max, viscosity, formulation_viscous)));

    // fill boundary descriptor pressure
    boundary_descriptor_pressure->neumann_bc.insert(
      pair(0, new PressureBC_dudt<dim>(u_x_max, viscosity)));
    boundary_descriptor_pressure->dirichlet_bc.insert(
      pair(1, new AnalyticalSolutionPressure<dim>(u_x_max, viscosity)));
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution_velocity.reset(
      new AnalyticalSolutionVelocity<dim>(u_x_max, viscosity));
    field_functions->initial_solution_pressure.reset(
      new AnalyticalSolutionPressure<dim>(u_x_max, viscosity));
    field_functions->analytical_solution_pressure.reset(
      new AnalyticalSolutionPressure<dim>(u_x_max, viscosity));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<Function<dim>>
  set_mesh_movement_function() override
  {
    std::shared_ptr<Function<dim>> mesh_motion;

    MeshMovementData<dim> data;
    data.temporal                       = MeshMovementAdvanceInTime::Sin;
    data.shape                          = MeshMovementShape::Sin; // SineAligned;
    data.dimensions[0]                  = std::abs(right - left);
    data.dimensions[1]                  = std::abs(right - left);
    data.amplitude                      = 0.08 * (right - left); // A_max = (RIGHT-LEFT)/(2*pi)
    data.period                         = 4.0 * end_time;
    data.t_start                        = 0.0;
    data.t_end                          = end_time;
    data.spatial_number_of_oscillations = 1.0;
    mesh_motion.reset(new CubeMeshMovementFunctions<dim>(data));

    return mesh_motion;
  }

  void
  set_input_parameters_poisson(Poisson::InputParameters & param) override
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

  void set_boundary_conditions_poisson(
    std::shared_ptr<Poisson::BoundaryDescriptor<1, dim>> boundary_descriptor) override
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    std::shared_ptr<Function<dim>> bc = this->set_mesh_movement_function();
    boundary_descriptor->dirichlet_bc.insert(pair(0, bc));
    boundary_descriptor->dirichlet_bc.insert(pair(1, bc));
  }

  void
  set_field_functions_poisson(
    std::shared_ptr<Poisson::FieldFunctions<dim>> field_functions) override
  {
    field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output                         = false;
    pp_data.output_data.output_folder                        = output_directory + "vtu/";
    pp_data.output_data.output_name                          = output_name;
    pp_data.output_data.output_start_time                    = start_time;
    pp_data.output_data.output_interval_time                 = (end_time - start_time) / 20;
    pp_data.output_data.write_vorticity                      = true;
    pp_data.output_data.write_divergence                     = true;
    pp_data.output_data.write_velocity_magnitude             = true;
    pp_data.output_data.write_vorticity_magnitude            = true;
    pp_data.output_data.write_processor_id                   = true;
    pp_data.output_data.mean_velocity.calculate              = true;
    pp_data.output_data.mean_velocity.sample_start_time      = start_time;
    pp_data.output_data.mean_velocity.sample_end_time        = end_time;
    pp_data.output_data.mean_velocity.sample_every_timesteps = 1;
    pp_data.output_data.write_higher_order                   = true;
    pp_data.output_data.degree                               = degree;

    // calculation of velocity error
    pp_data.error_data_u.analytical_solution_available = true;
    pp_data.error_data_u.analytical_solution.reset(
      new AnalyticalSolutionVelocity<dim>(u_x_max, viscosity));
    pp_data.error_data_u.calculate_relative_errors = true;
    pp_data.error_data_u.error_calc_start_time     = start_time;
    pp_data.error_data_u.error_calc_interval_time  = (end_time - start_time);
    pp_data.error_data_u.name                      = "velocity";

    // ... pressure error
    pp_data.error_data_p.analytical_solution_available = true;
    pp_data.error_data_p.analytical_solution.reset(
      new AnalyticalSolutionPressure<dim>(u_x_max, viscosity));
    pp_data.error_data_p.calculate_relative_errors = true;
    pp_data.error_data_p.error_calc_start_time     = start_time;
    pp_data.error_data_p.error_calc_interval_time  = (end_time - start_time);
    pp_data.error_data_p.name                      = "pressure";

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace Vortex
} // namespace IncNS
} // namespace ExaDG


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_VORTEX_H_ */
