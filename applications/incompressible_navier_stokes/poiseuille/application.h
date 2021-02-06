/*
 * poiseuille.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_POISEUILLE_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_POISEUILLE_H_

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

enum class BoundaryCondition
{
  ParabolicInflow,
  PressureInflow,
  Periodic
};

inline void
string_to_enum(BoundaryCondition & enum_type, std::string const string_type)
{
  // clang-format off
  if     (string_type == "ParabolicInflow") enum_type = BoundaryCondition::ParabolicInflow;
  else if(string_type == "PressureInflow")  enum_type = BoundaryCondition::PressureInflow;
  else if(string_type == "Periodic")        enum_type = BoundaryCondition::Periodic;
  else AssertThrow(false, ExcMessage("Unknown operator type. Not implemented."));
  // clang-format on
}

template<int dim>
class AnalyticalSolutionVelocity : public Function<dim>
{
public:
  AnalyticalSolutionVelocity(double const max_velocity, double const H)
    : Function<dim>(dim, 0.0), max_velocity(max_velocity), H(H)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
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
class AnalyticalSolutionPressure : public Function<dim>
{
public:
  AnalyticalSolutionPressure(double const viscosity,
                             double const max_velocity,
                             double const L,
                             double const H)
    : Function<dim>(1 /*n_components*/, 0.0),
      viscosity(viscosity),
      max_velocity(max_velocity),
      L(L),
      H(H)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int /*component*/) const
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
class NeumannBoundaryVelocity : public Function<dim>
{
public:
  NeumannBoundaryVelocity(FormulationViscousTerm const & formulation,
                          double const                   max_velocity,
                          double const                   H,
                          double const                   normal)
    : Function<dim>(dim, 0.0),
      formulation(formulation),
      max_velocity(max_velocity),
      H(H),
      normal(normal)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
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
class RightHandSide : public Function<dim>
{
public:
  RightHandSide(double const viscosity, double const max_velocity, double const H)
    : Function<dim>(dim, 0.0), viscosity(viscosity), max_velocity(max_velocity), H(H)
  {
  }

  double
  value(const Point<dim> & /*p*/, const unsigned int component = 0) const
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
  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(input_file, "", true, true);

    string_to_enum(boundary_condition, boundary_condition_string);
  }

  void
  add_parameters(ParameterHandler & prm)
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    // clang-format off
    prm.enter_subsection("Application");
      prm.add_parameter("BoundaryConditionType",
                        boundary_condition_string,
                        "Type of boundary condition.",
                        Patterns::Selection("ParabolicInflow|PressureInflow|Periodic"));
      prm.add_parameter("ApplySymmetryBC",
                        apply_symmetry_bc,
                        "Apply symmetry boundary condition.",
                        Patterns::Bool());
    prm.leave_subsection();
    // clang-format on
  }

  std::string       boundary_condition_string = "ParabolicInflow";
  BoundaryCondition boundary_condition        = BoundaryCondition::ParabolicInflow;

  bool apply_symmetry_bc = false;

  FormulationViscousTerm const formulation_viscous_term =
    FormulationViscousTerm::LaplaceFormulation;

  double const max_velocity = 1.0;
  double const viscosity    = 1.0e-1;

  double const H = 2.0;
  double const L = 4.0;

  double const start_time = 0.0;
  double const end_time   = 100.0;

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.problem_type                   = ProblemType::Unsteady;
    param.equation_type                  = EquationType::NavierStokes;
    param.formulation_viscous_term       = formulation_viscous_term;
    param.formulation_convective_term    = FormulationConvectiveTerm::ConvectiveFormulation;
    param.use_outflow_bc_convective_term = false;
    // prescribe body force in x-direction in case of periodic BC's
    param.right_hand_side = (boundary_condition == BoundaryCondition::Periodic);


    // PHYSICAL QUANTITIES
    param.start_time = start_time;
    param.end_time   = end_time;
    param.viscosity  = viscosity;


    // TEMPORAL DISCRETIZATION
    param.solver_type                   = SolverType::Unsteady;
    param.temporal_discretization       = TemporalDiscretization::BDFDualSplittingScheme;
    param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Explicit;
    param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    param.adaptive_time_stepping        = true;
    param.max_velocity                  = max_velocity;
    param.cfl                           = 2.0e-1;
    param.time_step_size                = 1.0e-1;
    param.order_time_integrator         = 2;    // 1; // 2; // 3;
    param.start_with_low_order          = true; // true; // false;

    param.convergence_criterion_steady_problem =
      ConvergenceCriterionSteadyProblem::SolutionIncrement; // ResidualSteadyNavierStokes;
    param.abs_tol_steady = 1.e-12;
    param.rel_tol_steady = 1.e-6;

    // output of solver information
    param.solver_info_data.interval_time = (param.end_time - param.start_time) / 10;

    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_p           = DegreePressure::MixedOrder;
    param.mapping            = MappingType::Isoparametric;

    // convective term
    if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      param.upwind_factor = 0.5;

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // PROJECTION METHODS

    // formulation
    param.store_previous_boundary_values = true;

    // pressure Poisson equation
    param.solver_pressure_poisson         = SolverPressurePoisson::CG;
    param.solver_data_pressure_poisson    = SolverData(1000, 1.e-20, 1.e-6, 100);
    param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

    // projection step
    param.solver_projection         = SolverProjection::CG;
    param.solver_data_projection    = SolverData(1000, 1.e-20, 1.e-12);
    param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;


    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // viscous step
    param.solver_viscous         = SolverViscous::CG;
    param.solver_data_viscous    = SolverData(1000, 1.e-20, 1.e-6);
    param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; // Multigrid;

    // PRESSURE-CORRECTION SCHEME

    // formulation
    param.order_pressure_extrapolation = 1;
    param.rotational_formulation       = true;

    // momentum step

    // Newton solver
    param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-14, 1.e-6);

    // linear solver
    param.solver_momentum                = SolverMomentum::GMRES;
    param.solver_data_momentum           = SolverData(1e4, 1.e-20, 1.e-6, 100);
    param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
    param.update_preconditioner_momentum = false;


    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-10, 1.e-6);

    // linear solver
    param.solver_coupled      = SolverCoupled::FGMRES; // GMRES;
    param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-2, 200);

    // preconditioning linear solver
    param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    param.update_preconditioner_coupled = true;

    // preconditioner velocity/momentum block
    param.preconditioner_velocity_block          = MomentumPreconditioner::Multigrid;
    param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;
    param.multigrid_data_velocity_block.smoother_data.smoother =
      MultigridSmoother::Chebyshev; // Jacobi; //Chebyshev; //GMRES;
    param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::BlockJacobi; // PointJacobi; //BlockJacobi;
    param.multigrid_data_velocity_block.smoother_data.iterations        = 5;
    param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
    param.multigrid_data_velocity_block.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block =
      SchurComplementPreconditioner::PressureConvectionDiffusion;
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    double const              y_upper = apply_symmetry_bc ? 0.0 : H / 2.;
    Point<dim>                point1(0.0, -H / 2.), point2(L, y_upper);
    std::vector<unsigned int> repetitions({2, 1});
    GridGenerator::subdivided_hyper_rectangle(*triangulation, repetitions, point1, point2);

    // set boundary indicator
    for(auto cell : triangulation->active_cell_iterators())
    {
      for(unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell;
          ++face_number)
      {
        if((std::fabs(cell->face(face_number)->center()(0) - 0.0) < 1e-12))
          cell->face(face_number)->set_boundary_id(1);
        if((std::fabs(cell->face(face_number)->center()(0) - L) < 1e-12))
          cell->face(face_number)->set_boundary_id(2);

        if(apply_symmetry_bc) // upper wall
          if((std::fabs(cell->face(face_number)->center()(1) - y_upper) < 1e-12))
            cell->face(face_number)->set_boundary_id(3);
      }
    }

    if(boundary_condition == BoundaryCondition::Periodic)
    {
      auto tria = dynamic_cast<Triangulation<dim> *>(&*triangulation);
      GridTools::collect_periodic_faces(*tria, 1, 2, 0, periodic_faces);
      triangulation->add_periodicity(periodic_faces);
    }

    triangulation->refine_global(n_refine_space);
  }

  void
  set_boundary_conditions(std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
                          std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // fill boundary descriptor velocity

    // no-slip walls
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));

    if(boundary_condition != BoundaryCondition::Periodic)
    {
      // inflow
      if(boundary_condition == BoundaryCondition::ParabolicInflow)
      {
        boundary_descriptor_velocity->dirichlet_bc.insert(
          pair(1, new AnalyticalSolutionVelocity<dim>(max_velocity, H)));
      }
      else if(boundary_condition == BoundaryCondition::PressureInflow)
      {
        boundary_descriptor_velocity->neumann_bc.insert(
          pair(1,
               new NeumannBoundaryVelocity<dim>(formulation_viscous_term, max_velocity, H, -1.0)));
      }
      else
      {
        AssertThrow(false, ExcMessage("not implemented."));
      }

      // outflow
      boundary_descriptor_velocity->neumann_bc.insert(
        pair(2, new NeumannBoundaryVelocity<dim>(formulation_viscous_term, max_velocity, H, 1.0)));
    }

    if(apply_symmetry_bc)
    {
      // slip boundary condition: always u*n=0
      // function will not be used -> use ZeroFunction
      boundary_descriptor_velocity->symmetry_bc.insert(
        pair(3, new Functions::ZeroFunction<dim>(dim)));
    }

    // fill boundary descriptor pressure

    // no-slip walls
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));

    if(boundary_condition != BoundaryCondition::Periodic)
    {
      // inflow
      if(boundary_condition == BoundaryCondition::ParabolicInflow)
      {
        boundary_descriptor_pressure->neumann_bc.insert(
          pair(1, new Functions::ZeroFunction<dim>(dim)));
      }
      else if(boundary_condition == BoundaryCondition::PressureInflow)
      {
        boundary_descriptor_pressure->dirichlet_bc.insert(
          pair(1, new AnalyticalSolutionPressure<dim>(viscosity, max_velocity, L, H)));
      }
      else
      {
        AssertThrow(false, ExcMessage("not implemented."));
      }

      // outflow
      boundary_descriptor_pressure->dirichlet_bc.insert(
        pair(2, new AnalyticalSolutionPressure<dim>(viscosity, max_velocity, L, H)));
    }

    if(apply_symmetry_bc)
    {
      // On symmetry boundaries, a Neumann BC is prescribed for the pressure.
      // -> prescribe dudt for dual-splitting scheme, which is equal to zero since
      // (du/dt)*n = d(u*n)/dt = d(0)/dt = 0, i.e., the time derivative term is multiplied by the
      // normal vector and the normal velocity is zero (= symmetry boundary condition).
      boundary_descriptor_pressure->neumann_bc.insert(
        pair(3, new Functions::ZeroFunction<dim>(dim)));
    }
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    if(boundary_condition == BoundaryCondition::Periodic)
      field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    else
      field_functions->analytical_solution_pressure.reset(
        new AnalyticalSolutionPressure<dim>(viscosity, max_velocity, L, H));
    field_functions->right_hand_side.reset(new RightHandSide<dim>(viscosity, max_velocity, H));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output              = this->write_output;
    pp_data.output_data.output_folder             = this->output_directory + "vtu/";
    pp_data.output_data.output_name               = this->output_name;
    pp_data.output_data.output_start_time         = start_time;
    pp_data.output_data.output_interval_time      = (end_time - start_time) / 10;
    pp_data.output_data.write_vorticity           = true;
    pp_data.output_data.write_divergence          = true;
    pp_data.output_data.write_velocity_magnitude  = true;
    pp_data.output_data.write_vorticity_magnitude = true;
    pp_data.output_data.write_processor_id        = true;
    pp_data.output_data.write_q_criterion         = true;
    pp_data.output_data.degree                    = degree;
    pp_data.output_data.write_higher_order        = true;

    // calculation of error
    // calculation of velocity error
    pp_data.error_data_u.analytical_solution_available = true;
    pp_data.error_data_u.analytical_solution.reset(
      new AnalyticalSolutionVelocity<dim>(max_velocity, H));
    pp_data.error_data_u.calculate_relative_errors = false;
    pp_data.error_data_u.error_calc_start_time     = start_time;
    pp_data.error_data_u.error_calc_interval_time  = (end_time - start_time) / 10;
    pp_data.error_data_u.name                      = "velocity";

    // ... pressure error
    pp_data.error_data_p.analytical_solution_available = true;
    if(boundary_condition == BoundaryCondition::Periodic)
      pp_data.error_data_p.analytical_solution.reset(new Functions::ZeroFunction<dim>(1));
    else
      pp_data.error_data_p.analytical_solution.reset(
        new AnalyticalSolutionPressure<dim>(viscosity, max_velocity, L, H));
    pp_data.error_data_p.calculate_relative_errors = false;
    pp_data.error_data_p.error_calc_start_time     = start_time;
    pp_data.error_data_p.error_calc_interval_time  = (end_time - start_time) / 10;
    pp_data.error_data_p.name                      = "pressure";

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace IncNS

template<int dim, typename Number>
std::shared_ptr<IncNS::ApplicationBase<dim, Number>>
get_application(std::string input_file)
{
  return std::make_shared<IncNS::Application<dim, Number>>(input_file);
}

} // namespace ExaDG


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_POISEUILLE_H_ */
