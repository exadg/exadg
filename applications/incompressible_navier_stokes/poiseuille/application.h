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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_POISEUILLE_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_POISEUILLE_H_

namespace ExaDG
{
namespace IncNS
{
enum class BoundaryCondition
{
  ParabolicInflow,
  PressureInflow,
  Periodic
};

template<int dim>
class AnalyticalSolutionVelocity : public dealii::Function<dim>
{
public:
  AnalyticalSolutionVelocity(double const max_velocity, double const H)
    : dealii::Function<dim>(dim, 0.0), max_velocity(max_velocity), H(H)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    double result = 0.0;

    if(component == 0)
      result = -max_velocity * (pow(p[1] / (H / 2.), 2.0) - 1.0);

    return result;
  }

private:
  double const max_velocity, H;
};

template<int dim>
class AnalyticalSolutionPressure : public dealii::Function<dim>
{
public:
  AnalyticalSolutionPressure(double const viscosity,
                             double const max_velocity,
                             double const L,
                             double const H)
    : dealii::Function<dim>(1 /*n_components*/, 0.0),
      viscosity(viscosity),
      max_velocity(max_velocity),
      L(L),
      H(H)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const /*component*/) const final
  {
    // pressure decreases linearly in flow direction
    double pressure_gradient = -2. * viscosity * max_velocity / std::pow(H / 2., 2.0);

    double const result = (p[0] - L) * pressure_gradient;

    return result;
  }

private:
  double const viscosity, max_velocity, L, H;
};

template<int dim>
class NeumannBoundaryVelocity : public dealii::Function<dim>
{
public:
  NeumannBoundaryVelocity(FormulationViscousTerm const & formulation,
                          double const                   max_velocity,
                          double const                   H,
                          double const                   normal)
    : dealii::Function<dim>(dim, 0.0),
      formulation(formulation),
      max_velocity(max_velocity),
      H(H),
      normal(normal)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    (void)p;
    (void)component;

    double result = 0.0;

    // The Neumann velocity boundary condition that is consistent with the analytical solution
    // (in case of a parabolic inflow profile) is (grad U)*n = 0.

    // Hence:
    // If the viscous term is written in Laplace formulation, prescribe result = 0 as Neumann BC
    // If the viscous term is written in Divergence formulation, the following boundary condition
    // has to be used to ensure that (grad U)*n = 0:
    // (grad U + (grad U)^T)*n = (grad U)^T * n

    if(formulation == FormulationViscousTerm::DivergenceFormulation)
    {
      if(component == 1)
        result = -max_velocity * 2.0 * p[1] / std::pow(H / 2., 2.0) * normal;
    }

    return result;
  }

private:
  FormulationViscousTerm const formulation;
  double const                 max_velocity;
  double const                 H;
  double const                 normal;
};

template<int dim>
class RightHandSide : public dealii::Function<dim>
{
public:
  RightHandSide(double const viscosity, double const max_velocity, double const H)
    : dealii::Function<dim>(dim, 0.0), viscosity(viscosity), max_velocity(max_velocity), H(H)
  {
  }

  double
  value(dealii::Point<dim> const & /*p*/, unsigned int const component = 0) const final
  {
    double pressure_gradient = 0.0;

    if(component == 0)
      pressure_gradient = -2. * viscosity * max_velocity / std::pow(H / 2., 2.0);

    return -pressure_gradient;
  }

private:
  double const viscosity, max_velocity, H;
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
      prm.add_parameter("BoundaryConditionType", boundary_condition, "Type of boundary condition.");
      prm.add_parameter("ApplySymmetryBC",
                        apply_symmetry_bc,
                        "Apply symmetry boundary condition.",
                        dealii::Patterns::Bool());
    }
    prm.leave_subsection();
  }

private:
  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type                   = ProblemType::Unsteady;
    this->param.equation_type                  = EquationType::NavierStokes;
    this->param.formulation_viscous_term       = formulation_viscous_term;
    this->param.formulation_convective_term    = FormulationConvectiveTerm::ConvectiveFormulation;
    this->param.use_outflow_bc_convective_term = false;
    // prescribe body force in x-direction in case of periodic BC's
    this->param.right_hand_side = (boundary_condition == BoundaryCondition::Periodic);


    // PHYSICAL QUANTITIES
    this->param.start_time = start_time;
    this->param.end_time   = end_time;
    this->param.viscosity  = viscosity;


    // TEMPORAL DISCRETIZATION
    this->param.solver_type                     = SolverType::Unsteady;
    this->param.temporal_discretization         = TemporalDiscretization::BDFDualSplittingScheme;
    this->param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    this->param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    this->param.adaptive_time_stepping          = true;
    this->param.max_velocity                    = max_velocity;
    this->param.cfl                             = 2.0e-1;
    this->param.cfl_exponent_fe_degree_velocity = 1.5;
    this->param.time_step_size                  = 1.0e-1;
    this->param.order_time_integrator           = 2;    // 1; // 2; // 3;
    this->param.start_with_low_order            = true; // true; // false;

    this->param.convergence_criterion_steady_problem =
      ConvergenceCriterionSteadyProblem::SolutionIncrement; // ResidualSteadyNavierStokes;
    this->param.abs_tol_steady = 1.e-12;
    this->param.rel_tol_steady = 1.e-6;

    // output of solver information
    this->param.solver_info_data.interval_time =
      (this->param.end_time - this->param.start_time) / 10;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.mapping_degree          = this->param.degree_u;
    this->param.degree_p                = DegreePressure::MixedOrder;

    // convective term
    if(this->param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      this->param.upwind_factor = 0.5;

    // viscous term
    this->param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // PROJECTION METHODS

    // pressure Poisson equation
    this->param.solver_pressure_poisson         = SolverPressurePoisson::CG;
    this->param.solver_data_pressure_poisson    = SolverData(1000, 1.e-20, 1.e-6, 100);
    this->param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

    // projection step
    this->param.solver_projection         = SolverProjection::CG;
    this->param.solver_data_projection    = SolverData(1000, 1.e-20, 1.e-12);
    this->param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;


    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    this->param.order_extrapolation_pressure_nbc =
      this->param.order_time_integrator <= 2 ? this->param.order_time_integrator : 2;

    // viscous step
    this->param.solver_viscous         = SolverViscous::CG;
    this->param.solver_data_viscous    = SolverData(1000, 1.e-20, 1.e-6);
    this->param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; // Multigrid;

    // PRESSURE-CORRECTION SCHEME

    // formulation
    this->param.order_pressure_extrapolation = 1;
    this->param.rotational_formulation       = true;

    // momentum step

    // Newton solver
    this->param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-14, 1.e-6);

    // linear solver
    this->param.solver_momentum                = SolverMomentum::GMRES;
    this->param.solver_data_momentum           = SolverData(1e4, 1.e-20, 1.e-6, 100);
    this->param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
    this->param.update_preconditioner_momentum = false;


    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    this->param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-10, 1.e-6);

    // linear solver
    this->param.solver_coupled      = SolverCoupled::FGMRES; // GMRES;
    this->param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-2, 200);

    // preconditioning linear solver
    this->param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    this->param.update_preconditioner_coupled = true;

    // preconditioner velocity/momentum block
    this->param.preconditioner_velocity_block          = MomentumPreconditioner::Multigrid;
    this->param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;
    this->param.multigrid_data_velocity_block.smoother_data.smoother =
      MultigridSmoother::Chebyshev; // Jacobi; //Chebyshev; //GMRES;
    this->param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::BlockJacobi; // PointJacobi; //BlockJacobi;
    this->param.multigrid_data_velocity_block.smoother_data.iterations        = 5;
    this->param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
    this->param.multigrid_data_velocity_block.coarse_problem.solver =
      MultigridCoarseGridSolver::GMRES;

    // preconditioner Schur-complement block
    this->param.preconditioner_pressure_block =
      SchurComplementPreconditioner::PressureConvectionDiffusion;
  }

  void
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) final
  {
    (void)mapping;
    (void)multigrid_mappings;

    auto const lambda_create_triangulation = [&](dealii::Triangulation<dim, dim> & tria,
                                                 std::vector<dealii::GridTools::PeriodicFacePair<
                                                   typename dealii::Triangulation<
                                                     dim>::cell_iterator>> & periodic_face_pairs,
                                                 unsigned int const          global_refinements,
                                                 std::vector<unsigned int> const &
                                                   vector_local_refinements) {
      (void)vector_local_refinements;

      double const              y_upper = apply_symmetry_bc ? 0.0 : H / 2.;
      dealii::Point<dim>        point1(0.0, -H / 2.), point2(L, y_upper);
      std::vector<unsigned int> repetitions({2, 1});
      dealii::GridGenerator::subdivided_hyper_rectangle(tria, repetitions, point1, point2);

      // set boundary indicator
      for(auto cell : tria.cell_iterators())
      {
        for(auto const & face : cell->face_indices())
        {
          if((std::fabs(cell->face(face)->center()(0) - 0.0) < 1e-12))
            cell->face(face)->set_boundary_id(1);
          if((std::fabs(cell->face(face)->center()(0) - L) < 1e-12))
            cell->face(face)->set_boundary_id(2);

          if(apply_symmetry_bc) // upper wall
            if((std::fabs(cell->face(face)->center()(1) - y_upper) < 1e-12))
              cell->face(face)->set_boundary_id(3);
        }
      }

      if(boundary_condition == BoundaryCondition::Periodic)
      {
        AssertThrow(
          this->param.grid.triangulation_type != TriangulationType::FullyDistributed,
          dealii::ExcMessage(
            "Periodic faces might not be applied correctly for TriangulationType::FullyDistributed. "
            "Try to use another triangulation type, or try to fix these limitations in ExaDG or deal.II."));

        dealii::GridTools::collect_periodic_faces(tria, 1, 2, 0, periodic_face_pairs);
        tria.add_periodicity(periodic_face_pairs);
      }

      tria.refine_global(global_refinements);
    };

    GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
                                                            {} /* no local refinements */);
  }

  void
  set_boundary_descriptor(Grid<dim> const &                             grid,
                          std::shared_ptr<dealii::Mapping<dim>> const & mapping) final
  {
    (void)grid;
    (void)mapping;

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    // fill boundary descriptor velocity

    // no-slip walls
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));

    if(boundary_condition != BoundaryCondition::Periodic)
    {
      // inflow
      if(boundary_condition == BoundaryCondition::ParabolicInflow)
      {
        this->boundary_descriptor->velocity->dirichlet_bc.insert(
          pair(1, new AnalyticalSolutionVelocity<dim>(max_velocity, H)));
      }
      else if(boundary_condition == BoundaryCondition::PressureInflow)
      {
        this->boundary_descriptor->velocity->neumann_bc.insert(
          pair(1,
               new NeumannBoundaryVelocity<dim>(formulation_viscous_term, max_velocity, H, -1.0)));
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("not implemented."));
      }

      // outflow
      this->boundary_descriptor->velocity->neumann_bc.insert(
        pair(2, new NeumannBoundaryVelocity<dim>(formulation_viscous_term, max_velocity, H, 1.0)));
    }

    if(apply_symmetry_bc)
    {
      // slip boundary condition: always u*n=0
      // function will not be used -> use ZeroFunction
      this->boundary_descriptor->velocity->symmetry_bc.insert(
        pair(3, new dealii::Functions::ZeroFunction<dim>(dim)));
    }

    // fill boundary descriptor pressure

    // no-slip walls
    this->boundary_descriptor->pressure->neumann_bc.insert(0);

    if(boundary_condition != BoundaryCondition::Periodic)
    {
      // inflow
      if(boundary_condition == BoundaryCondition::ParabolicInflow)
      {
        this->boundary_descriptor->pressure->neumann_bc.insert(1);
      }
      else if(boundary_condition == BoundaryCondition::PressureInflow)
      {
        this->boundary_descriptor->pressure->dirichlet_bc.insert(
          pair(1, new AnalyticalSolutionPressure<dim>(viscosity, max_velocity, L, H)));
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("not implemented."));
      }

      // outflow
      this->boundary_descriptor->pressure->dirichlet_bc.insert(
        pair(2, new AnalyticalSolutionPressure<dim>(viscosity, max_velocity, L, H)));
    }

    if(apply_symmetry_bc)
    {
      // On symmetry boundaries, a Neumann BC is prescribed for the pressure.
      // -> prescribe dudt for dual-splitting scheme, which is equal to zero since
      // (du/dt)*n = d(u*n)/dt = d(0)/dt = 0, i.e., the time derivative term is multiplied by the
      // normal vector and the normal velocity is zero (= symmetry boundary condition).
      this->boundary_descriptor->pressure->neumann_bc.insert(3);
    }
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(
      new dealii::Functions::ZeroFunction<dim>(dim));
    this->field_functions->initial_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    if(boundary_condition == BoundaryCondition::Periodic)
      this->field_functions->analytical_solution_pressure.reset(
        new dealii::Functions::ZeroFunction<dim>(1));
    else
      this->field_functions->analytical_solution_pressure.reset(
        new AnalyticalSolutionPressure<dim>(viscosity, max_velocity, L, H));
    this->field_functions->right_hand_side.reset(
      new RightHandSide<dim>(viscosity, max_velocity, H));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = start_time;
    pp_data.output_data.time_control_data.trigger_interval = (end_time - start_time) / 10.0;
    pp_data.output_data.directory                 = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename                  = this->output_parameters.filename;
    pp_data.output_data.write_vorticity           = true;
    pp_data.output_data.write_divergence          = true;
    pp_data.output_data.write_shear_rate          = true;
    pp_data.output_data.write_velocity_magnitude  = true;
    pp_data.output_data.write_vorticity_magnitude = true;
    pp_data.output_data.write_processor_id        = true;
    pp_data.output_data.write_q_criterion         = true;
    pp_data.output_data.degree                    = this->param.degree_u;
    pp_data.output_data.write_higher_order        = true;

    // calculation of error
    // calculation of velocity error
    pp_data.error_data_u.time_control_data.is_active        = true;
    pp_data.error_data_u.time_control_data.start_time       = start_time;
    pp_data.error_data_u.time_control_data.trigger_interval = (end_time - start_time) / 10.0;
    pp_data.error_data_u.analytical_solution.reset(
      new AnalyticalSolutionVelocity<dim>(max_velocity, H));
    pp_data.error_data_u.calculate_relative_errors = false;
    pp_data.error_data_u.name                      = "velocity";

    // ... pressure error
    pp_data.error_data_p.time_control_data.is_active        = true;
    pp_data.error_data_p.time_control_data.start_time       = start_time;
    pp_data.error_data_p.time_control_data.trigger_interval = (end_time - start_time) / 10.0;
    if(boundary_condition == BoundaryCondition::Periodic)
      pp_data.error_data_p.analytical_solution.reset(new dealii::Functions::ZeroFunction<dim>(1));
    else
      pp_data.error_data_p.analytical_solution.reset(
        new AnalyticalSolutionPressure<dim>(viscosity, max_velocity, L, H));
    pp_data.error_data_p.calculate_relative_errors = false;
    pp_data.error_data_p.name                      = "pressure";

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  BoundaryCondition boundary_condition = BoundaryCondition::ParabolicInflow;

  bool apply_symmetry_bc = false;

  FormulationViscousTerm const formulation_viscous_term =
    FormulationViscousTerm::LaplaceFormulation;

  double const max_velocity = 1.0;
  double const viscosity    = 1.0e-1;

  double const H = 2.0;
  double const L = 4.0;

  double const start_time = 0.0;
  double const end_time   = 100.0;
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_POISEUILLE_H_ */
