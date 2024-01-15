/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef APPLICATIONS_AERO_ACOUSTIC_SANTA_IN_CROSSFLOW_H_
#define APPLICATIONS_AERO_ACOUSTIC_SANTA_IN_CROSSFLOW_H_

#include <deal.II/base/function.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>

#include <exadg/grid/grid_utilities.h>

// Air at -10°C
// density:        1.34
// speed_od_sound: 325.47
// viscosity:      1.247e-5

namespace ExaDG
{
namespace AcousticsAeroAcoustic
{
using namespace Acoustics;

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    prm.enter_subsection("Application");
    {
      // PHYSICAL QUANTITIES
      prm.add_parameter("StartTimeAcoustic", this->param.start_time);
      prm.add_parameter("EndTime", this->param.end_time);
      prm.add_parameter("SpeedOfSound", this->param.speed_of_sound);

      // TEMPORAL DISCRETIZATION
      prm.add_parameter("CFLAcoustics", this->param.cfl);
    }
    prm.leave_subsection();
  }

private:
  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.formulation = Formulation::SkewSymmetric;

    // TEMPORAL DISCRETIZATION
    this->param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    this->param.order_time_integrator         = 2;
    this->param.start_with_low_order          = true;
    this->param.adaptive_time_stepping        = false;

    // output of solver information
    this->param.solver_info_data.interval_time = (this->param.end_time - this->param.start_time);

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.mapping_degree          = 1;
    this->param.degree_p                = this->param.degree_u;
    this->param.degree_u                = this->param.degree_p;
  }

  void
  create_grid(Grid<dim> & grid, std::shared_ptr<dealii::Mapping<dim>> & mapping) final
  {
    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> & tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & /*periodic_face_pairs*/,
          unsigned int const global_refinements,
          std::vector<unsigned int> const & /* vector_local_refinements*/) {
        dealii::GridIn<dim> mesh(tria);
        std::ifstream input_file("../applications/aero_acoustic/santa_in_crossflow/santa.msh");
        mesh.read_msh(input_file);

        for(const auto & face : tria.active_face_iterators())
          if(face->at_boundary())
          {
            if(face->center().norm() < 20.0)
              face->set_boundary_id(0); // santa
            else
              face->set_boundary_id(1); // absorbing
          }

        tria.refine_global(global_refinements);
      };

    GridUtilities::create_triangulation<dim>(
      grid, this->mpi_comm, this->param.grid, lambda_create_triangulation, {});

    GridUtilities::create_mapping(mapping,
                                  this->param.grid.element_type,
                                  this->param.mapping_degree);
  }

  void
  set_boundary_descriptor() final
  {
    {
      double const Y = 0.1; // Santa and deers are minimally absorbing
      this->boundary_descriptor->admittance_bc.insert(
        std::make_pair(0, std::make_shared<dealii::Functions::ConstantFunction<dim>>(Y, 1)));
    }
    {
      double const Y = 1.0; // ABC
      this->boundary_descriptor->admittance_bc.insert(
        std::make_pair(1, std::make_shared<dealii::Functions::ConstantFunction<dim>>(Y, 1)));
    }
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity =
      std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim);

    this->field_functions->initial_solution_pressure =
      std::make_shared<dealii::Functions::ZeroFunction<dim>>(1);
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active  = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time = this->param.start_time;
    pp_data.output_data.time_control_data.trigger_interval =
      (this->param.end_time - this->param.start_time) / 20.0;
    pp_data.output_data.directory          = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = this->output_parameters.filename + "_acoustic";
    pp_data.output_data.write_pressure     = true;
    pp_data.output_data.write_velocity     = true;
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.degree             = this->param.degree_u;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

} // namespace AcousticsAeroAcoustic


namespace FluidAeroAcoustic
{
using namespace IncNS;

template<int dim>
class InflowBC : public dealii::Function<dim>
{
public:
  InflowBC(double const inlet_velocity_in)
    : dealii::Function<dim>(dim, 0.0), inlet_velocity(inlet_velocity_in)
  {
  }

  double
  value(dealii::Point<dim> const &, unsigned int const component) const final
  {
    if(component == 0)
      return inlet_velocity;
    return 0.0;
  }

private:
  double const inlet_velocity;
};


template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    prm.enter_subsection("Application");
    {
      // PHYSICAL QUANTITIES
      prm.add_parameter("EndTime", this->param.end_time);
      prm.add_parameter("KinematicViscosity", this->param.viscosity);

      // TEMPORAL DISCRETIZATION
      prm.add_parameter("CFLFluid", this->param.cfl);

      // Testcase
      prm.add_parameter("TravelSpeed",
                        inlet_velocity,
                        "Speed at which Santa travels.",
                        dealii::Patterns::Double(1.0e-12));
    }
    prm.leave_subsection();
  }

  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type                   = ProblemType::Unsteady;
    this->param.equation_type                  = EquationType::NavierStokes;
    this->param.formulation_viscous_term       = FormulationViscousTerm::LaplaceFormulation;
    this->param.formulation_convective_term    = FormulationConvectiveTerm::DivergenceFormulation;
    this->param.right_hand_side                = false;
    this->param.use_outflow_bc_convective_term = true;

    // PHYSICAL QUANTITIES
    this->param.start_time = 0.0;

    // TEMPORAL DISCRETIZATION
    this->param.solver_type                     = SolverType::Unsteady;
    this->param.temporal_discretization         = TemporalDiscretization::BDFDualSplittingScheme;
    this->param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    this->param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    this->param.order_time_integrator           = 2;
    this->param.start_with_low_order            = false;
    this->param.adaptive_time_stepping          = false;
    this->param.cfl_exponent_fe_degree_velocity = 1.5;
    this->param.max_velocity                    = inlet_velocity;

    // output of solver information
    this->param.solver_info_data.interval_time = (this->param.end_time - this->param.start_time);

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.mapping_degree          = this->param.degree_u;
    this->param.degree_p                = DegreePressure::MixedOrder;

    // convective term
    if(this->param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      this->param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

    // variant Direct allows to use larger time step
    // sizes due to CFL condition at inflow boundary
    this->param.type_dirichlet_bc_convective = TypeDirichletBCs::Mirror;

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

    // PROJECTION METHODS

    // pressure Poisson equation
    this->param.solver_pressure_poisson              = SolverPressurePoisson::CG;
    this->param.solver_data_pressure_poisson         = SolverData(1e4, 1.e-12, 1.e-6, 100);
    this->param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    this->param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
    this->param.multigrid_data_pressure_poisson.smoother_data.smoother =
      MultigridSmoother::Chebyshev;
    this->param.multigrid_data_pressure_poisson.smoother_data.preconditioner =
      PreconditionerSmoother::PointJacobi;
    this->param.multigrid_data_pressure_poisson.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev;
    this->param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::PointJacobi;

    // pressure Poisson equation
    this->param.solver_pressure_poisson              = SolverPressurePoisson::CG;
    this->param.solver_data_pressure_poisson         = SolverData(1000, ABS_TOL, REL_TOL, 100);
    this->param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    this->param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
    this->param.multigrid_data_pressure_poisson.coarse_problem.solver =
      MultigridCoarseGridSolver::CG;
    this->param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;

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
    this->param.multigrid_data_pressure_block.coarse_problem.solver = MultigridCoarseGridSolver::CG;
    this->param.multigrid_data_pressure_block.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;
  }

  void
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) final
  {
    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> &                        tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & periodic_face_pairs,
          unsigned int const                                       global_refinements,
          std::vector<unsigned int> const &                        vector_local_refinements) {
        (void)periodic_face_pairs;
        (void)vector_local_refinements;

        dealii::GridIn<dim> mesh(tria);
        std::ifstream input_file("../applications/aero_acoustic/santa_in_crossflow/santa.msh");
        mesh.read_msh(input_file);

        for(const auto & face : tria.active_face_iterators())
          if(face->at_boundary())
          {
            if(face->center().norm() < 20.0)
              face->set_boundary_id(0); // santa
            else if(face->center()[0] < -50.0 + 1e-6)
              face->set_boundary_id(1); // inlet
            else if(face->center()[0] > 50.0 - 1e-6)
              face->set_boundary_id(3); // outlet
            else
              face->set_boundary_id(2); // symmetry
          }

        tria.refine_global(global_refinements);
      };

    GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
                                                            {});
    // mappings
    GridUtilities::create_mapping_with_multigrid(mapping,
                                                 multigrid_mappings,
                                                 this->param.grid.element_type,
                                                 this->param.mapping_degree,
                                                 this->param.mapping_degree_coarse_grids,
                                                 this->param.involves_h_multigrid());
  }

  void
  set_boundary_descriptor() final
  {
    // santa
    this->boundary_descriptor->pressure->neumann_bc.insert(0);
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      std::make_pair(0, std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim)));

    // inflow
    this->boundary_descriptor->pressure->neumann_bc.insert(1);
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      std::make_pair(1, std::make_shared<InflowBC<dim>>(inlet_velocity)));

    // symmetry
    this->boundary_descriptor->velocity->symmetry_bc.insert(
      std::make_pair(2, std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim)));
    this->boundary_descriptor->pressure->neumann_bc.insert(2);

    // outflow
    this->boundary_descriptor->pressure->dirichlet_bc.insert(
      std::make_pair(3, std::make_shared<dealii::Functions::ZeroFunction<dim>>(1)));
    this->boundary_descriptor->velocity->neumann_bc.insert(
      std::make_pair(3, std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim)));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity =
      std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim);

    this->field_functions->initial_solution_pressure =
      std::make_shared<dealii::Functions::ZeroFunction<dim>>(1);

    this->field_functions->analytical_solution_pressure =
      std::make_shared<dealii::Functions::ZeroFunction<dim>>(1);

    this->field_functions->right_hand_side =
      std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim);
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active  = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time = this->param.start_time;
    pp_data.output_data.time_control_data.trigger_interval =
      (this->param.end_time - this->param.start_time) / 50.0;
    pp_data.output_data.directory          = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = this->output_parameters.filename + "_fluid";
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.write_processor_id = true;
    pp_data.output_data.write_boundary_IDs = true;
    pp_data.output_data.degree             = this->param.degree_u;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  // solver tolerances
  double const ABS_TOL = 1.e-12;
  double const REL_TOL = 1.e-4;

  double const ABS_TOL_LINEAR = 1.e-12;
  double const REL_TOL_LINEAR = 1.e-2;

  // testcase specific
  // For a travel speed of 0.08 a Karman vortex street emerges in the wake of Santa
  double inlet_velocity = 0.08;
};
} // namespace FluidAeroAcoustic

namespace AeroAcoustic
{
template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

  void
  set_single_field_solvers(std::string input_file, MPI_Comm const & comm) final
  {
    this->acoustic =
      std::make_shared<AcousticsAeroAcoustic::Application<dim, Number>>(input_file, comm);
    this->fluid = std::make_shared<FluidAeroAcoustic::Application<dim, Number>>(input_file, comm);
  }
};
} // namespace AeroAcoustic

} // namespace ExaDG

#include <exadg/aero_acoustic/user_interface/implement_get_application.h>

#endif /*APPLICATIONS_AERO_ACOUSTIC_SANTA_IN_CROSSFLOW_H_*/
