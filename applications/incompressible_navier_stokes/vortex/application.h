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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_VORTEX_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_VORTEX_H_

// ExaDG
#include <exadg/grid/deformed_cube_manifold.h>
#include <exadg/grid/mesh_movement_functions.h>
#include <exadg/grid/one_sided_cylindrical_manifold.h>

namespace ExaDG
{
namespace IncNS
{
enum class MeshType
{
  UniformCartesian,
  ComplexSurfaceManifold,
  ComplexVolumeManifold,
  Curvilinear
};

template<int dim>
class AnalyticalSolutionVelocity : public dealii::Function<dim>
{
public:
  AnalyticalSolutionVelocity(double const u_x_max, double const viscosity)
    : dealii::Function<dim>(dim, 0.0), u_x_max(u_x_max), viscosity(viscosity)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const
  {
    double const t  = this->get_time();
    double const pi = dealii::numbers::PI;

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
class AnalyticalSolutionPressure : public dealii::Function<dim>
{
public:
  AnalyticalSolutionPressure(double const u_x_max, double const viscosity)
    : dealii::Function<dim>(1 /*n_components*/, 0.0), u_x_max(u_x_max), viscosity(viscosity)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const /*component*/) const
  {
    double const t  = this->get_time();
    double const pi = dealii::numbers::PI;

    double const result = -u_x_max * std::cos(2 * pi * p[0]) * std::cos(2 * pi * p[1]) *
                          std::exp(-8.0 * pi * pi * viscosity * t);

    return result;
  }

private:
  double const u_x_max, viscosity;
};

template<int dim>
class NeumannBoundaryVelocity : public dealii::Function<dim>
{
public:
  NeumannBoundaryVelocity(double const                 u_x_max,
                          double const                 viscosity,
                          FormulationViscousTerm const formulation_viscous)
    : dealii::Function<dim>(dim, 0.0),
      u_x_max(u_x_max),
      viscosity(viscosity),
      formulation_viscous(formulation_viscous)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const
  {
    double const t  = this->get_time();
    double const pi = dealii::numbers::PI;

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
                  dealii::ExcMessage("Specified formulation of viscous term is not implemented!"));
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
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const
  {
    double const t  = this->get_time();
    double const pi = dealii::numbers::PI;

    dealii::Tensor<1, dim> const n = this->get_normal_vector();

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
                  dealii::ExcMessage("Specified formulation of viscous term is not implemented!"));
    }

    return result;
  }

private:
  double const                 u_x_max, viscosity;
  FormulationViscousTerm const formulation_viscous;
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
    this->param.formulation_viscous_term    = formulation_viscous;
    this->param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
    this->param.right_hand_side             = false;

    // ALE
    this->param.ale_formulation                     = ALE;
    this->param.mesh_movement_type                  = MeshMovementType::Function;
    this->param.neumann_with_variable_normal_vector = ALE;

    // PHYSICAL QUANTITIES
    this->param.start_time = start_time;
    this->param.end_time   = end_time;
    this->param.viscosity  = viscosity;


    // TEMPORAL DISCRETIZATION
    this->param.solver_type                  = SolverType::Unsteady;
    this->param.temporal_discretization      = TemporalDiscretization::BDFDualSplittingScheme;
    this->param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
    this->param.order_time_integrator        = 2;
    this->param.start_with_low_order         = false;
    this->param.adaptive_time_stepping       = false;
    this->param.calculation_of_time_step_size =
      TimeStepCalculation::UserSpecified; // UserSpecified; //CFL;
    this->param.time_step_size                  = end_time;
    this->param.max_velocity                    = 1.4 * u_x_max;
    this->param.cfl                             = 0.2; // 0.4;
    this->param.cfl_exponent_fe_degree_velocity = 1.5;
    this->param.c_eff                           = 8.0;
    this->param.time_integrator_oif             = TimeIntegratorOIF::ExplRK3Stage7Reg2;
    this->param.cfl_oif                         = this->param.cfl / 1.0;

    // output of solver information
    this->param.solver_info_data.interval_time = this->param.end_time - this->param.start_time;

    // restart
    this->param.restarted_simulation             = false;
    this->param.restart_data.write_restart       = false;
    this->param.restart_data.interval_time       = 0.25;
    this->param.restart_data.interval_wall_time  = 1.e6;
    this->param.restart_data.interval_time_steps = 1e8;
    this->param.restart_data.filename =
      this->output_parameters.directory + this->output_parameters.filename;


    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.grid.mapping_degree     = this->param.degree_u;
    this->param.degree_p                = DegreePressure::MixedOrder;

    // convective term
    this->param.upwind_factor = 1.0;

    // viscous term
    this->param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // velocity pressure coupling terms
    this->param.gradp_formulation = FormulationPressureGradientTerm::Weak;
    this->param.divu_formulation  = FormulationVelocityDivergenceTerm::Weak;

    // div-div and continuity penalty
    this->param.use_divergence_penalty               = true;
    this->param.divergence_penalty_factor            = 1.0e0;
    this->param.use_continuity_penalty               = true;
    this->param.continuity_penalty_factor            = this->param.divergence_penalty_factor;
    this->param.continuity_penalty_components        = ContinuityPenaltyComponents::Normal;
    this->param.continuity_penalty_use_boundary_data = true;
    if(this->param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
      this->param.apply_penalty_terms_in_postprocessing_step = false;
    else
      this->param.apply_penalty_terms_in_postprocessing_step = true;

    // NUMERICAL PARAMETERS
    this->param.implement_block_diagonal_preconditioner_matrix_free = false;
    this->param.use_cell_based_face_loops                           = false;
    this->param.quad_rule_linearization = QuadratureRuleLinearization::Overintegration32k;

    // PROJECTION METHODS

    // pressure Poisson equation
    this->param.solver_pressure_poisson              = SolverPressurePoisson::CG;
    this->param.solver_data_pressure_poisson         = SolverData(1000, 1.e-12, 1.e-6, 100);
    this->param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    this->param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
    this->param.multigrid_data_pressure_poisson.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev;
    this->param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::PointJacobi;
    this->param.multigrid_data_pressure_poisson.smoother_data.smoother =
      MultigridSmoother::Chebyshev;
    this->param.multigrid_data_pressure_poisson.smoother_data.preconditioner =
      PreconditionerSmoother::PointJacobi;

    // projection step
    this->param.solver_projection         = SolverProjection::CG;
    this->param.solver_data_projection    = SolverData(1000, 1.e-12, 1.e-6);
    this->param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;
    this->param.preconditioner_block_diagonal_projection =
      Elementwise::Preconditioner::InverseMassMatrix;
    this->param.solver_data_block_diagonal_projection = SolverData(1000, 1.e-12, 1.e-2, 1000);

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    this->param.order_extrapolation_pressure_nbc =
      this->param.order_time_integrator <= 2 ? this->param.order_time_integrator : 2;
    this->param.formulation_convective_term_bc = FormulationConvectiveTerm::ConvectiveFormulation;

    // viscous step
    this->param.solver_viscous         = SolverViscous::CG;
    this->param.solver_data_viscous    = SolverData(1000, 1.e-12, 1.e-6);
    this->param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; // Multigrid;
    this->param.multigrid_data_viscous.type                   = MultigridType::hMG;
    this->param.multigrid_data_viscous.smoother_data.smoother = MultigridSmoother::Chebyshev;
    this->param.update_preconditioner_viscous                 = false;


    // PRESSURE-CORRECTION SCHEME

    // formulation
    this->param.order_pressure_extrapolation =
      std::min(2, (int)this->param.order_time_integrator) - 1; // J_p = J-1, but not larger than 1
    this->param.rotational_formulation = true;

    // momentum step

    // Newton solver
    this->param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-12, 1.e-6); // TODO

    // linear solver
    this->param.solver_momentum                = SolverMomentum::FGMRES;
    this->param.solver_data_momentum           = SolverData(1e4, 1.e-12, 1.e-6, 100);
    this->param.update_preconditioner_momentum = false;
    this->param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix; // Multigrid;
    this->param.multigrid_operator_type_momentum = MultigridOperatorType::ReactionDiffusion;

    // Jacobi smoother data
    //  this->param.multigrid_data_momentum.smoother_data.smoother = MultigridSmoother::Jacobi;
    //  this->param.multigrid_data_momentum.smoother_data.preconditioner =
    //  PreconditionerSmoother::BlockJacobi;
    //  this->param.multigrid_data_momentum.smoother_data.iterations = 5;
    //  this->param.multigrid_data_momentum.coarse_problem.solver =
    //  MultigridCoarseGridSolver::GMRES;

    // Chebyshev smoother data
    this->param.multigrid_data_momentum.smoother_data.smoother = MultigridSmoother::Chebyshev;
    this->param.multigrid_data_momentum.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev;


    // COUPLED NAVIER-STOKES SOLVER
    this->param.use_scaling_continuity = false;

    // nonlinear solver (Newton solver)
    this->param.newton_solver_data_coupled =
      Newton::SolverData(100, 1.e-10, 1.e-6); // TODO did not converge with 1.e-12

    // linear solver
    this->param.solver_coupled      = SolverCoupled::FGMRES;
    this->param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-6, 100);

    // preconditioner linear solver
    this->param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    this->param.update_preconditioner_coupled = false;

    // preconditioner momentum block
    this->param.preconditioner_velocity_block          = MomentumPreconditioner::Multigrid;
    this->param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;
    this->param.multigrid_data_velocity_block.type     = MultigridType::phMG;
    this->param.multigrid_data_velocity_block.smoother_data.smoother =
      MultigridSmoother::Chebyshev; // GMRES;
    this->param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::BlockJacobi;
    this->param.multigrid_data_velocity_block.smoother_data.iterations        = 5;
    this->param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
    // coarse grid solver
    this->param.multigrid_data_velocity_block.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev; // GMRES;

    // preconditioner Schur-complement block
    this->param.preconditioner_pressure_block = SchurComplementPreconditioner::CahouetChabard;
  }

  void
  create_grid() final
  {
    if(ALE)
    {
      AssertThrow(mesh_type == MeshType::UniformCartesian,
                  dealii::ExcMessage(
                    "Taylor vortex problem: Parameter mesh_type is invalid for ALE."));
    }

    if(mesh_type == MeshType::UniformCartesian)
    {
      // Uniform Cartesian grid
      dealii::GridGenerator::subdivided_hyper_cube(*this->grid->triangulation, 2, left, right);
    }
    else if(mesh_type == MeshType::ComplexSurfaceManifold or
            mesh_type == MeshType::ComplexVolumeManifold)
    {
      // Complex Geometry
      dealii::Triangulation<dim> tria1, tria2, tria_coarse;
      double const               radius = (right - left) * 0.25;
      double const               width  = right - left;
      dealii::GridGenerator::hyper_shell(
        tria1, dealii::Point<dim>(), radius, 0.5 * width * std::sqrt(dim), 2 * dim);
      tria1.reset_all_manifolds();
      if(dim == 2)
      {
        dealii::GridTools::rotate(dealii::numbers::PI / 4, tria1);
      }
      dealii::GridGenerator::hyper_ball(tria2, dealii::Point<dim>(), radius);
      tria2.reset_all_manifolds();
      dealii::GridGenerator::merge_triangulations(tria1, tria2, tria_coarse);

      // manifolds
      tria_coarse.set_all_manifold_ids(0);

      // vectors of manifold_ids and face_ids required only in case of volume manifold
      std::vector<unsigned int> manifold_ids;
      std::vector<unsigned int> face_ids;

      for(auto cell : tria_coarse.active_cell_iterators())
      {
        for(unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
        {
          if(cell->face(f)->at_boundary())
          {
            bool face_at_sphere_boundary = true;
            for(unsigned int v = 0; v < dealii::GeometryInfo<dim - 1>::vertices_per_cell; ++v)
            {
              if(std::abs(cell->face(f)->vertex(v).norm() - radius) > 1e-12)
              {
                face_at_sphere_boundary = false;
                break;
              }
            }

            if(face_at_sphere_boundary)
            {
              if(mesh_type == MeshType::ComplexSurfaceManifold)
              {
                cell->face(f)->set_all_manifold_ids(1);
              }
              else if(mesh_type == MeshType::ComplexVolumeManifold)
              {
                face_ids.push_back(f);
                unsigned int manifold_id = manifold_ids.size() + 1;
                cell->set_all_manifold_ids(manifold_id);
                manifold_ids.push_back(manifold_id);
                break;
              }
              else
              {
                AssertThrow(false, dealii::ExcMessage("Should not arrive here."));
              }
            }
          }
        }
      }

      // set manifolds
      if(mesh_type == MeshType::ComplexSurfaceManifold)
      {
        static const dealii::SphericalManifold<dim> spherical_manifold;
        this->grid->triangulation->set_manifold(1, spherical_manifold);
      }
      else if(mesh_type == MeshType::ComplexVolumeManifold)
      {
        // generate vector of manifolds and apply manifold to all cells that have been marked
        static std::vector<std::shared_ptr<dealii::Manifold<dim>>> manifold_vec;
        manifold_vec.resize(manifold_ids.size());

        for(unsigned int i = 0; i < manifold_ids.size(); ++i)
        {
          for(auto cell : tria_coarse.active_cell_iterators())
          {
            if(cell->manifold_id() == manifold_ids[i])
            {
              dealii::Point<dim> center = dealii::Point<dim>();
              manifold_vec[i] =
                std::shared_ptr<dealii::Manifold<dim>>(static_cast<dealii::Manifold<dim> *>(
                  new OneSidedCylindricalManifold<dim>(cell, face_ids[i], center)));
              tria_coarse.set_manifold(manifold_ids[i], *(manifold_vec[i]));
            }
          }
        }
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("Should not arrive here."));
      }

      tria_coarse.refine_global(1);
      // make sure that the triangulation does not have refinements at this point
      dealii::GridGenerator::flatten_triangulation(tria_coarse, *this->grid->triangulation);
    }
    else if(mesh_type == MeshType::Curvilinear)
    {
      dealii::GridGenerator::subdivided_hyper_cube(*this->grid->triangulation, 2, left, right);

      this->grid->triangulation->set_all_manifold_ids(1);
      double const                     deformation = 0.1;
      unsigned int const               frequency   = 2;
      static DeformedCubeManifold<dim> manifold(left, right, deformation, frequency);
      this->grid->triangulation->set_manifold(1, manifold);
    }

    // boundary IDs
    for(auto cell : this->grid->triangulation->active_cell_iterators())
    {
      for(unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
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

    this->grid->triangulation->refine_global(this->param.grid.n_refine_global);
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    // fill boundary descriptor velocity
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0, new AnalyticalSolutionVelocity<dim>(u_x_max, viscosity)));
    if(ALE)
      this->boundary_descriptor->velocity->neumann_bc.insert(
        pair(1, new NeumannBoundaryVelocityALE<dim>(u_x_max, viscosity, formulation_viscous)));
    else
      this->boundary_descriptor->velocity->neumann_bc.insert(
        pair(1, new NeumannBoundaryVelocity<dim>(u_x_max, viscosity, formulation_viscous)));

    // fill boundary descriptor pressure
    this->boundary_descriptor->pressure->neumann_bc.insert(0);
    this->boundary_descriptor->pressure->dirichlet_bc.insert(
      pair(1, new AnalyticalSolutionPressure<dim>(u_x_max, viscosity)));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(
      new AnalyticalSolutionVelocity<dim>(u_x_max, viscosity));
    this->field_functions->initial_solution_pressure.reset(
      new AnalyticalSolutionPressure<dim>(u_x_max, viscosity));
    this->field_functions->analytical_solution_pressure.reset(
      new AnalyticalSolutionPressure<dim>(u_x_max, viscosity));
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<dealii::Function<dim>>
  create_mesh_movement_function() final
  {
    std::shared_ptr<dealii::Function<dim>> mesh_motion;

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
  set_parameters_poisson() final
  {
    using namespace Poisson;

    // MATHEMATICAL MODEL
    this->poisson_param.right_hand_side = false;

    // SPATIAL DISCRETIZATION
    this->poisson_param.degree = this->param.grid.mapping_degree;

    this->poisson_param.spatial_discretization = SpatialDiscretization::CG;
    this->poisson_param.IP_factor              = 1.0e0;

    // SOLVER
    this->poisson_param.solver                    = Poisson::Solver::CG;
    this->poisson_param.solver_data.abs_tol       = 1.e-20;
    this->poisson_param.solver_data.rel_tol       = 1.e-10;
    this->poisson_param.solver_data.max_iter      = 1e4;
    this->poisson_param.preconditioner            = Preconditioner::Multigrid;
    this->poisson_param.multigrid_data.type       = MultigridType::cphMG;
    this->poisson_param.multigrid_data.p_sequence = PSequenceType::Bisect;
    // MG smoother
    this->poisson_param.multigrid_data.smoother_data.smoother        = MultigridSmoother::Chebyshev;
    this->poisson_param.multigrid_data.smoother_data.iterations      = 5;
    this->poisson_param.multigrid_data.smoother_data.smoothing_range = 20;
    // MG coarse grid solver
    this->poisson_param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::CG;
    this->poisson_param.multigrid_data.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;
    this->poisson_param.multigrid_data.coarse_problem.solver_data.rel_tol = 1.e-3;
  }

  void
  set_boundary_descriptor_poisson() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    std::shared_ptr<dealii::Function<dim>> bc = this->create_mesh_movement_function();
    this->poisson_boundary_descriptor->dirichlet_bc.insert(pair(0, bc));
    this->poisson_boundary_descriptor->dirichlet_bc.insert(pair(1, bc));
  }

  void
  set_field_functions_poisson() final
  {
    this->poisson_field_functions->initial_solution.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->poisson_field_functions->right_hand_side.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output              = this->output_parameters.write;
    pp_data.output_data.directory                 = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename                  = this->output_parameters.filename;
    pp_data.output_data.start_time                = start_time;
    pp_data.output_data.interval_time             = (end_time - start_time) / 20;
    pp_data.output_data.write_vorticity           = true;
    pp_data.output_data.write_divergence          = true;
    pp_data.output_data.write_velocity_magnitude  = true;
    pp_data.output_data.write_vorticity_magnitude = true;
    pp_data.output_data.write_processor_id        = true;
    pp_data.output_data.mean_velocity.calculate   = true;
    pp_data.output_data.mean_velocity.sample_start_time      = start_time;
    pp_data.output_data.mean_velocity.sample_end_time        = end_time;
    pp_data.output_data.mean_velocity.sample_every_timesteps = 1;
    pp_data.output_data.write_higher_order                   = true;
    pp_data.output_data.degree                               = this->param.degree_u;

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
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

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
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_VORTEX_H_ */
