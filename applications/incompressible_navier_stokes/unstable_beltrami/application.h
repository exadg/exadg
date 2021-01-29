/*
 * unstable_beltrami.h
 *
 *  Created on: July, 2018
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_UNSTABLE_BELTRAMI_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_UNSTABLE_BELTRAMI_H_

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim>
class AnalyticalSolutionVelocity : public Function<dim>
{
public:
  AnalyticalSolutionVelocity(double const viscosity) : Function<dim>(dim, 0.0), nu(viscosity)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double t      = this->get_time();
    double result = 0.0;

    const double pi = numbers::PI;

    if(component == 0)
      result = std::sin(2.0 * pi * p[0]) * std::sin(2.0 * pi * p[1]);
    else if(component == 1)
      result = std::cos(2.0 * pi * p[0]) * std::cos(2.0 * pi * p[1]);
    else if(component == 2)
      result = std::sqrt(2.0) * std::sin(2.0 * pi * p[0]) * std::cos(2.0 * pi * p[1]);

    result *= std::exp(-8.0 * pi * pi * nu * t);

    return result;
  }

  Tensor<1, dim, double>
  gradient(const Point<dim> & p, const unsigned int component = 0) const
  {
    double                 t = this->get_time();
    Tensor<1, dim, double> result;

    const double pi = numbers::PI;

    AssertThrow(dim == 3, ExcMessage("not implemented."));

    if(component == 0)
    {
      result[0] = 2.0 * pi * std::cos(2.0 * pi * p[0]) * std::sin(2.0 * pi * p[1]);
      result[1] = 2.0 * pi * std::sin(2.0 * pi * p[0]) * std::cos(2.0 * pi * p[1]);
      result[2] = 0.0;
    }
    else if(component == 1)
    {
      result[0] = -2.0 * pi * std::sin(2.0 * pi * p[0]) * std::cos(2.0 * pi * p[1]);
      result[1] = -2.0 * pi * std::cos(2.0 * pi * p[0]) * std::sin(2.0 * pi * p[1]);
      result[2] = 0.0;
    }
    else if(component == 2)
    {
      result[0] = 2.0 * pi * std::sqrt(2.0) * std::cos(2.0 * pi * p[0]) * std::cos(2.0 * pi * p[1]);
      result[1] =
        -2.0 * pi * std::sqrt(2.0) * std::sin(2.0 * pi * p[0]) * std::sin(2.0 * pi * p[1]);
      result[2] = 0.0;
    }

    result *= std::exp(-8.0 * pi * pi * nu * t);

    return result;
  }

private:
  double const nu;
};


template<int dim>
class AnalyticalSolutionPressure : public Function<dim>
{
public:
  AnalyticalSolutionPressure(double const viscosity)
    : Function<dim>(1 /*n_components*/, 0.0), nu(viscosity)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int /*component*/) const
  {
    double       t  = this->get_time();
    const double pi = numbers::PI;

    double result = -0.5 * (+std::sin(2.0 * pi * p[0]) * std::sin(2.0 * pi * p[0]) +
                            std::cos(2.0 * pi * p[1]) * std::cos(2.0 * pi * p[1]) - 1.0);

    result *= std::exp(-16.0 * pi * pi * nu * t);

    return result;
  }

private:
  double const nu;
};

template<int dim>
class PressureBC_dudt : public Function<dim>
{
public:
  PressureBC_dudt(double const viscosity) : Function<dim>(dim, 0.0), nu(viscosity)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double t      = this->get_time();
    double result = 0.0;

    const double pi = numbers::PI;

    if(component == 0)
      result = std::sin(2.0 * pi * p[0]) * std::sin(2.0 * pi * p[1]);
    else if(component == 1)
      result = std::cos(2.0 * pi * p[0]) * std::cos(2.0 * pi * p[1]);
    else if(component == 2)
      result = std::sqrt(2.0) * std::sin(2.0 * pi * p[0]) * std::cos(2.0 * pi * p[1]);

    result *= -8.0 * pi * pi * nu * std::exp(-8.0 * pi * pi * nu * t);

    return result;
  }

private:
  double const nu;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  double const viscosity = 1.0 / (2.0e3);

  double const start_time = 0.0;
  double const end_time   = 20.0;

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.problem_type                = ProblemType::Unsteady;
    param.equation_type               = EquationType::NavierStokes;
    param.formulation_viscous_term    = FormulationViscousTerm::LaplaceFormulation;
    param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
    param.right_hand_side             = false;

    // PHYSICAL QUANTITIES
    param.start_time = start_time;
    param.end_time   = end_time;
    param.viscosity  = viscosity;


    // TEMPORAL DISCRETIZATION
    param.solver_type                     = SolverType::Unsteady;
    param.temporal_discretization         = TemporalDiscretization::BDFDualSplittingScheme;
    param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    param.max_velocity                    = std::sqrt(2.0);
    param.cfl                             = 0.1;
    param.cfl_exponent_fe_degree_velocity = 1.5;
    param.time_step_size                  = 1.0e-3;
    param.order_time_integrator           = 2;     // 1; // 2; // 3;
    param.start_with_low_order            = false; // true; // false;

    // output of solver information
    param.solver_info_data.interval_time = (param.end_time - param.start_time) / 10;

    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_p           = DegreePressure::MixedOrder;
    param.mapping            = MappingType::Isoparametric;

    // convective term

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // pressure level is undefined
    param.adjust_pressure_level = AdjustPressureLevel::ApplyAnalyticalMeanValue;

    // div-div and continuity penalty
    param.use_divergence_penalty                     = true;
    param.divergence_penalty_factor                  = 1.0e0;
    param.use_continuity_penalty                     = true;
    param.continuity_penalty_factor                  = param.divergence_penalty_factor;
    param.apply_penalty_terms_in_postprocessing_step = true;


    // PROJECTION METHODS

    // pressure Poisson equation
    param.solver_pressure_poisson         = SolverPressurePoisson::CG;
    param.solver_data_pressure_poisson    = SolverData(1000, 1.e-12, 1.e-8, 100);
    param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

    // projection step
    param.solver_projection         = SolverProjection::CG;
    param.solver_data_projection    = SolverData(1000, 1.e-12, 1.e-12);
    param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // viscous step
    param.solver_viscous         = SolverViscous::CG;
    param.solver_data_viscous    = SolverData(1000, 1.e-12, 1.e-8);
    param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; // Multigrid;

    // PRESSURE-CORRECTION SCHEME

    // momentum step

    // Newton solver
    param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-12, 1.e-8);

    // linear solver
    param.solver_momentum                = SolverMomentum::GMRES;
    param.solver_data_momentum           = SolverData(1e4, 1.e-12, 1.e-8, 100);
    param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
    param.update_preconditioner_momentum = false;

    // formulation
    param.order_pressure_extrapolation = 1;
    param.rotational_formulation       = true;


    // COUPLED NAVIER-STOKES SOLVER
    param.use_scaling_continuity = false;

    // nonlinear solver (Newton solver)
    param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-12, 1.e-8);

    // linear solver
    param.solver_coupled      = SolverCoupled::GMRES;
    param.solver_data_coupled = SolverData(1e3, 1.e-12, 1.e-8, 100);

    // preconditioning linear solver
    param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

    // preconditioner velocity/momentum block
    param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

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
    AssertThrow(dim == 3, ExcMessage("This test case can only be used for dim==3!"));

    const double left = 0.0, right = 1.0;
    GridGenerator::hyper_cube(*triangulation, left, right);

    // periodicity in x-,y-, and z-direction

    // x-direction
    triangulation->begin()->face(0)->set_all_boundary_ids(0);
    triangulation->begin()->face(1)->set_all_boundary_ids(1);
    // y-direction
    triangulation->begin()->face(2)->set_all_boundary_ids(2);
    triangulation->begin()->face(3)->set_all_boundary_ids(3);
    // z-direction
    triangulation->begin()->face(4)->set_all_boundary_ids(4);
    triangulation->begin()->face(5)->set_all_boundary_ids(5);

    auto tria = dynamic_cast<Triangulation<dim> *>(&*triangulation);
    GridTools::collect_periodic_faces(*tria, 0, 1, 0 /*x-direction*/, periodic_faces);
    GridTools::collect_periodic_faces(*tria, 2, 3, 1 /*y-direction*/, periodic_faces);
    GridTools::collect_periodic_faces(*tria, 4, 5, 2 /*z-direction*/, periodic_faces);

    triangulation->add_periodicity(periodic_faces);

    // global refinements
    triangulation->refine_global(n_refine_space);
  }

  void
  set_boundary_conditions(std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
                          std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure)
  {
    (void)boundary_descriptor_velocity;
    (void)boundary_descriptor_pressure;
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution_velocity.reset(
      new AnalyticalSolutionVelocity<dim>(viscosity));
    field_functions->initial_solution_pressure.reset(
      new AnalyticalSolutionPressure<dim>(viscosity));
    field_functions->analytical_solution_pressure.reset(
      new AnalyticalSolutionPressure<dim>(viscosity));
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
    pp_data.output_data.output_interval_time = (end_time - start_time) / 100;
    pp_data.output_data.write_divergence     = false;
    pp_data.output_data.degree               = degree;

    // calculation of velocity error
    pp_data.error_data_u.analytical_solution_available = true;
    pp_data.error_data_u.analytical_solution.reset(new AnalyticalSolutionVelocity<dim>(viscosity));
    pp_data.error_data_u.calculate_H1_seminorm_error = true;
    pp_data.error_data_u.calculate_relative_errors   = true;
    pp_data.error_data_u.error_calc_start_time       = start_time;
    pp_data.error_data_u.error_calc_interval_time    = 10;
    pp_data.error_data_u.write_errors_to_file        = true;
    pp_data.error_data_u.folder                      = this->output_directory;
    pp_data.error_data_u.name                        = this->output_name + "_velocity";

    // ... pressure error
    pp_data.error_data_p.analytical_solution_available = true;
    pp_data.error_data_p.analytical_solution.reset(new AnalyticalSolutionPressure<dim>(viscosity));
    pp_data.error_data_p.calculate_relative_errors = true;
    pp_data.error_data_p.error_calc_start_time     = start_time;
    pp_data.error_data_p.error_calc_interval_time  = 10;
    pp_data.error_data_p.write_errors_to_file      = true;
    pp_data.error_data_p.folder                    = this->output_directory;
    pp_data.error_data_p.name                      = this->output_name + "_pressure";

    // kinetic energy
    pp_data.kinetic_energy_data.calculate                  = true;
    pp_data.kinetic_energy_data.calculate_every_time_steps = 10;
    pp_data.kinetic_energy_data.viscosity                  = viscosity;
    pp_data.kinetic_energy_data.filename = this->output_directory + this->output_name + "_energy";

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


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_UNSTABLE_BELTRAMI_H_ */
