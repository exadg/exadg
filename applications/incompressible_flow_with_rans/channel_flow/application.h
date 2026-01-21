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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_WITH_RANS_TEST_CASES_TEMPLATE_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_WITH_RANS_TEST_CASES_TEMPLATE_H_

namespace ExaDG
{
namespace NSRans
{
double channel_height      = 1.0;
double channel_length      = 4.0;
double bulk_velocity       = 1.0;
double kinematic_viscosity = 1e-5;

double start_time = 0.0;
double end_time   = 10.0;

bool const   write_restart         = false;
double const restart_interval_time = 10.0;

double const CFL                    = 0.3;
double const max_velocity           = 2.0 * bulk_velocity;
bool const   adaptive_time_stepping = true;

// vtu output
double const output_interval_time = (end_time - start_time) / 10.0;

// moving mesh (ALE)
bool const ALE = false;

// solver tolerances
double const ABS_TOL = 1.e-12;
double const REL_TOL = 1.e-6;

double const ABS_TOL_LINEAR = 1.e-12;
double const REL_TOL_LINEAR = 1.e-2;

template<int dim, typename Number>
class Fluid : public FluidBase<dim, Number>
{
public:
  Fluid(std::string parameter_file, MPI_Comm const & comm)
    : FluidBase<dim, Number>(parameter_file, comm)
  {
  }

private:
  void
  set_parameters() final
  {
    using namespace IncRANS;

    // MATHEMATICAL MODEL
    this->param.problem_type                = ProblemType::Unsteady;
    this->param.equation_type               = EquationType::NavierStokes;
    this->param.formulation_viscous_term    = FormulationViscousTerm::LaplaceFormulation;
    this->param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
    this->param.ale_formulation             = ALE;
    this->param.mesh_movement_type          = MeshMovementType::Function;
    this->param.right_hand_side             = true;
    this->param.boussinesq_term             = false;


    // PHYSICAL QUANTITIES
    this->param.start_time = start_time;
    this->param.end_time   = end_time;
    this->param.viscosity  = kinematic_viscosity;


    // TEMPORAL DISCRETIZATION
    this->param.solver_type                     = SolverType::Unsteady;
    this->param.temporal_discretization         = TemporalDiscretization::BDFCoupledSolution;
    this->param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    this->param.adaptive_time_stepping          = adaptive_time_stepping;
    this->param.order_time_integrator           = 2;
    this->param.start_with_low_order            = true;
    this->param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    this->param.max_velocity                    = max_velocity;
    this->param.cfl_exponent_fe_degree_velocity = 1.5;
    this->param.cfl                             = CFL;
    this->param.time_step_size                  = 1.0e-1;

    // output of solver information
    this->param.solver_info_data.interval_time = (end_time - start_time) / 10.;

    // restart
    this->param.restart_data.write_restart = write_restart;
    this->param.restart_data.interval_time = restart_interval_time;
    this->param.restart_data.filename =
      this->output_parameters.directory + this->output_parameters.filename + "_fluid";

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type     = TriangulationType::Distributed;
    this->param.mapping_degree              = 1;
    this->param.mapping_degree_coarse_grids = this->param.mapping_degree;
    this->param.degree_p                    = DegreePressure::MixedOrder;

    // convective term
    if(this->param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      this->param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

    // viscous term
    this->param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // div-div and continuity penalty
    this->param.use_divergence_penalty                     = true;
    this->param.divergence_penalty_factor                  = 1.0e0;
    this->param.use_continuity_penalty                     = true;
    this->param.continuity_penalty_factor                  = this->param.divergence_penalty_factor;
    this->param.continuity_penalty_components              = ContinuityPenaltyComponents::Normal;
    this->param.continuity_penalty_use_boundary_data       = true;
    this->param.apply_penalty_terms_in_postprocessing_step = true;
    this->param.type_penalty_parameter                     = TypePenaltyParameter::ConvectiveTerm;

    // NUMERICAL PARAMETERS
    this->param.implement_block_diagonal_preconditioner_matrix_free = false;
    this->param.use_cell_based_face_loops                           = false;

    // PROJECTION METHODS

    // pressure Poisson equation
    this->param.solver_pressure_poisson              = SolverPressurePoisson::CG;
    this->param.solver_data_pressure_poisson         = SolverData(1000, ABS_TOL, REL_TOL, 100);
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
    this->param.solver_data_projection    = SolverData(1000, ABS_TOL, REL_TOL);
    this->param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulation
    this->param.order_extrapolation_pressure_nbc =
      this->param.order_time_integrator <= 2 ? this->param.order_time_integrator : 2;

    if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
    {
      this->param.solver_momentum         = SolverMomentum::CG;
      this->param.solver_data_momentum    = SolverData(1000, ABS_TOL, REL_TOL);
      this->param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
    }


    // PRESSURE-CORRECTION SCHEME

    // formulation
    this->param.order_pressure_extrapolation = 1;
    this->param.rotational_formulation       = true;

    // momentum step
    if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
    {
      // Newton solver
      this->param.newton_solver_data_momentum = Newton::SolverData(100, ABS_TOL, REL_TOL);

      // linear solver
      this->param.solver_momentum = SolverMomentum::GMRES;
      if(this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
        this->param.solver_data_momentum = SolverData(1e4, ABS_TOL_LINEAR, REL_TOL_LINEAR, 100);
      else
        this->param.solver_data_momentum = SolverData(1e4, ABS_TOL, REL_TOL, 100);

      this->param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
    }


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

    // preconditioner linear solver
    this->param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    this->param.update_preconditioner_coupled = false;

    // preconditioner velocity/momentum block
    this->param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

    // preconditioner Schur-complement block
    this->param.preconditioner_pressure_block =
      SchurComplementPreconditioner::PressureConvectionDiffusion;
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
        // create triangulation and perform local/global refinements
        /*(void)tria;*/
        (void)periodic_face_pairs;
        (void)global_refinements;
        (void)vector_local_refinements;

        unsigned int x_subdivisions = static_cast<unsigned int>(channel_length / channel_height);
        std::vector<unsigned int> repetitions = {x_subdivisions, 1};
        dealii::Point<dim>        bottom_left, top_right;

        bottom_left[0] = 0.0;
        bottom_left[1] = 0.0;

        top_right[0] = channel_length;
        top_right[1] = channel_height;

        if(dim == 2)
        {
          repetitions = {x_subdivisions, 1};
        }
        else if(dim == 3)
        {
          AssertThrow(false, dealii::ExcMessage("Not implemented!"));
        }

        dealii::GridGenerator::subdivided_hyper_rectangle(tria,
                                                          repetitions,
                                                          bottom_left,
                                                          top_right);

        for(auto cell : tria.cell_iterators())
        {
          for(auto const & f : cell->face_indices())
          {
            if(cell->face(f)->at_boundary())
            {
              if(std::fabs(cell->face(f)->center()(0) - 0.0) < 1e-12)
              {
                cell->face(f)->set_boundary_id(1); // inlet
              }
              else if(std::fabs(cell->face(f)->center()(0) - channel_length) < 1e-12)
              {
                cell->face(f)->set_boundary_id(2); // outlet
              }
              else if(std::fabs(cell->face(f)->center()(1) - 0.0) < 1e-12)
              {
                cell->face(f)->set_boundary_id(3); // bottom wall
              }
              else if(std::fabs(cell->face(f)->center()(1) - channel_height) < 1e-12)
              {
                cell->face(f)->set_boundary_id(3); // top_wall
              }
            }
          }
        }
        tria.refine_global(global_refinements);
      };

    GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
                                                            {} /* no local refinements */);

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
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    std::vector<double> inlet_velocity = std::vector<double>(dim, 0.0);
    inlet_velocity[0]                  = bulk_velocity;

    // velocity
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(1, new dealii::Functions::ConstantFunction<dim>(inlet_velocity)));
    this->boundary_descriptor->velocity->neumann_bc.insert(
      pair(2, new dealii::Functions::ZeroFunction<dim>(dim)));
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(3, new dealii::Functions::ZeroFunction<dim>(dim)));

    // pressure
    this->boundary_descriptor->pressure->neumann_bc.insert(1);
    this->boundary_descriptor->pressure->dirichlet_bc.insert(
      pair(2, new dealii::Functions::ZeroFunction<dim>(1)));
    this->boundary_descriptor->pressure->neumann_bc.insert(3);
  }

  void
  set_field_functions() final
  {
    std::vector<double> inlet_velocity = std::vector<double>(dim, 0.0);
    inlet_velocity[0]                  = bulk_velocity;

    this->field_functions->initial_solution_velocity.reset(
      new dealii::Functions::ConstantFunction<dim>(inlet_velocity));
    this->field_functions->initial_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->analytical_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<IncRANS::PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    IncRANS::PostProcessorData<dim> pp_data;

    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = start_time;
    pp_data.output_data.time_control_data.trigger_interval = output_interval_time;
    pp_data.output_data.directory          = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = this->output_parameters.filename;
    pp_data.output_data.write_processor_id = true;
    pp_data.output_data.degree             = this->param.degree_u;
    pp_data.output_data.write_higher_order = true;

    std::shared_ptr<IncRANS::PostProcessorBase<dim, Number>> pp;
    pp.reset(new IncRANS::PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

template<int dim, typename Number>
class Scalar : public ScalarBase<dim, Number>
{
public:
  Scalar(std::string parameter_file, MPI_Comm const & comm)
    : ScalarBase<dim, Number>(parameter_file, comm)
  {
  }

private:
  void
  set_parameters() final
  {
    using namespace RANS;
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    this->boundary_descriptor->dirichlet_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(1)));
    this->boundary_descriptor->neumann_bc.insert(
      pair(1, new dealii::Functions::ZeroFunction<dim>(1)));
  }


  void
  set_field_functions() final
  {
    this->field_functions->initial_solution.reset(new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->velocity.reset(new dealii::Functions::ZeroFunction<dim>(1));
  }

  std::shared_ptr<RANS::PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    RANS::PostProcessorData<dim> pp_data;

    std::shared_ptr<RANS::PostProcessorBase<dim, Number>> pp;
    pp.reset(new RANS::PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
    this->fluid = std::make_shared<Fluid<dim, Number>>(input_file, comm);

    // create one (or even more) scalar fields
    this->scalars.resize(0);
    /*this->scalars[0] = std::make_shared<Scalar<dim, Number>>(input_file, comm);*/
  }
};

} // namespace NSRans

} // namespace ExaDG

#include <exadg/incompressible_flow_with_rans/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_WITH_RANS_TEST_CASES_TEMPLATE_H_ */
