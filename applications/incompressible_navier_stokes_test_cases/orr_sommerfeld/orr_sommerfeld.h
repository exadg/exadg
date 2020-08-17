/*
 * orr_sommerfeld.h
 *
 *  Created on: Aug 31, 2017
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_ORR_SOMMERFELD_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_ORR_SOMMERFELD_H_

#include "orr_sommerfeld_equation.h"
#include "postprocessor.h"

namespace ExaDG
{
namespace IncNS
{
namespace OrrSommerfeld
{
using namespace dealii;

template<int dim>
class AnalyticalSolutionVelocity : public Function<dim>
{
public:
  AnalyticalSolutionVelocity(double const                              H,
                             double const                              MAX_VELOCITY,
                             double const                              ALPHA,
                             double const                              EPSILON,
                             FE_DGQ<1> const &                         FE,
                             std::complex<double> const &              OMEGA,
                             std::vector<std::complex<double>> const & EIG_VEC)
    : Function<dim>(dim, 0.0),
      H(H),
      MAX_VELOCITY(MAX_VELOCITY),
      ALPHA(ALPHA),
      EPSILON(EPSILON),
      FE(FE),
      OMEGA(OMEGA),
      EIG_VEC(EIG_VEC)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double t      = this->get_time();
    double result = 0.0;

    double const x = p[0] / H;
    // transform from interval [-H,H] (-> y) to unit interval [0,1] (-> eta)
    double const eta = 0.5 * (p[1] / H + 1.0);

    Assert(eta <= 1.0 + 1.e-12 and eta >= 0.0 - 1.e-12,
           ExcMessage("Point in reference coordinates is invalid."));

    double cos = std::cos(ALPHA * x - OMEGA.real() * t);
    double sin = std::sin(ALPHA * x - OMEGA.real() * t);

    double amplification = std::exp(OMEGA.imag() * t);

    std::complex<double> exp(cos, sin);

    if(component == 0)
    {
      double base = MAX_VELOCITY * (1.0 - pow(p[1] / H, 2.0));

      // d(psi)/dy = d(psi)/d(eta) * d(eta)/dy
      // evaluate derivative d(psi)/d(eta) in eta(y)
      std::complex<double> dpsi = 0;
      for(unsigned int i = 0; i < FE.get_degree() + 1; ++i)
        dpsi += EIG_VEC[i] * FE.shape_grad(i, Point<1>(eta))[0];

      // multiply by d(eta)/dy to obtain derivative d(psi)/dy in physical space
      dpsi *= 0.5 / H;

      std::complex<double> perturbation_complex = dpsi * exp * amplification;
      double               perturbation         = perturbation_complex.real();

      result = base + EPSILON * perturbation;
    }
    else if(component == 1)
    {
      // evaluate function psi in y
      std::complex<double> psi = 0;
      for(unsigned int i = 0; i < FE.get_degree() + 1; ++i)
        psi += EIG_VEC[i] * FE.shape_value(i, Point<1>(eta));

      std::complex<double> i(0, 1);
      std::complex<double> perturbation_complex = -i * ALPHA * psi * exp * amplification;
      double               perturbation         = perturbation_complex.real();

      result = EPSILON * perturbation;
    }
    else
    {
      Assert(false, ExcMessage("Orr-Sommerfeld problem is a 2-dimensional problem."));
    }

    return result;
  }

private:
  const double H            = 1.0;
  const double MAX_VELOCITY = 1.0;
  const double ALPHA        = 1.0;
  const double EPSILON      = 1.0e-5;

  FE_DGQ<1> const &                         FE;
  std::complex<double> const &              OMEGA;
  std::vector<std::complex<double>> const & EIG_VEC;
};


/*
 *  Right-hand side function: Implements the body force vector occuring on the
 *  right-hand side of the momentum equation of the Navier-Stokes equations
 */
template<int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide(double const viscosity, double const max_velocity, double const H)
    : Function<dim>(dim, 0.0), nu(viscosity), U_max(max_velocity), H(H)
  {
  }

  double
  value(const Point<dim> & /*p*/, const unsigned int component = 0) const
  {
    double result = 0.0;

    // mean flow is driven by a constant body force since we use
    // periodic BC's in streamwise direction.
    // Body force is derived by a balance of forces in streamwise direction
    //   f * L * 2H = tau * 2 * L (2H = height, L = length, factor 2 = upper and lower wall)
    // with tau = nu du/dy|_{y=-H} = nu * U_max * (-2y/H^2)|_{y=-H} = 2 * nu * U_max / H
    if(component == 0)
      result = 2. * nu * U_max / (H * H);

    return result;
  }

private:
  double const nu, U_max, H;
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

    // solve Orr-Sommerfeld equation
    compute_eigenvector(EIG_VEC, OMEGA, Re, ALPHA, FE);
    // calculate characteristic time interval
    t0       = 2.0 * PI * ALPHA / OMEGA.real();
    end_time = 2.0 * t0;
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

  std::string output_directory = "output/orr_sommerfeld/", output_name = "Re7500_l5_k2_div_conti";

  const double Re = 7500.0;

  const double H  = 1.0;
  const double PI = numbers::PI;
  const double L  = 2.0 * PI * H;

  const double MAX_VELOCITY = 1.0;
  const double VISCOSITY    = MAX_VELOCITY * H / Re;
  const double ALPHA        = 1.0;
  const double EPSILON      = 1.0e-5; // perturbations are small (<< 1, linearization)

  // Orr-Sommerfeld solver: calculates unstable eigenvalue (OMEGA) of
  // Orr-Sommerfeld equation for Poiseuille flow and corresponding eigenvector (EIG_VEC).
  // do not use more than 300 due to conditioning of polynomials
  unsigned int const DEGREE_OS_SOLVER = 200;

  FE_DGQ<1> FE = FE_DGQ<1>(DEGREE_OS_SOLVER);

  std::complex<double> OMEGA;

  std::vector<std::complex<double>> EIG_VEC =
    std::vector<std::complex<double>>(DEGREE_OS_SOLVER + 1);

  // the time the Tollmien-Schlichting-waves need to travel through the domain
  // (depends on solution of Orr-Sommerfeld equation)
  double       t0         = 0.0;
  double const start_time = 0.0;
  double       end_time   = 2.0 * t0;

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.problem_type             = ProblemType::Unsteady;
    param.equation_type            = EquationType::NavierStokes;
    param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
    param.right_hand_side          = true; // prescribe body force in x-direction


    // PHYSICAL QUANTITIES
    param.start_time = start_time;
    param.end_time   = end_time;
    param.viscosity  = VISCOSITY;


    // TEMPORAL DISCRETIZATION
    param.solver_type = SolverType::Unsteady;
    param.temporal_discretization =
      TemporalDiscretization::BDFCoupledSolution; // BDFDualSplittingScheme;
    param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Explicit;
    param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    param.max_velocity = MAX_VELOCITY; // MAX_VELOCITY is also needed somewhere else
    // approx. cfl_crit = 0.9 for BDF2 (and exponent = 1.5)
    // approx. cfl_crit = 0.4 for BDF3 (and exponent = 1.5)
    param.cfl                             = 0.4;
    param.cfl_exponent_fe_degree_velocity = 1.5;
    param.time_step_size                  = 1.0e-2;
    param.max_number_of_time_steps        = 1e8;
    param.order_time_integrator           = 2;    // 1; // 2; // 3;
    param.start_with_low_order            = true; // true; // false;

    // output of solver information
    param.solver_info_data.interval_time = (param.end_time - param.start_time) / 20;


    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_p           = DegreePressure::MixedOrder;
    param.mapping            = MappingType::Isoparametric;

    // convective term

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // divergence and continuity penalty terms
    param.use_divergence_penalty                     = true;
    param.divergence_penalty_factor                  = 1.0e0;
    param.use_continuity_penalty                     = true;
    param.continuity_penalty_factor                  = param.divergence_penalty_factor;
    param.apply_penalty_terms_in_postprocessing_step = true;

    // PROJECTION METHODS

    // pressure Poisson equation
    param.solver_pressure_poisson         = SolverPressurePoisson::CG;
    param.solver_data_pressure_poisson    = SolverData(1000, 1.e-12, 1.e-6, 100);
    param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

    // projection step
    param.solver_projection         = SolverProjection::CG;
    param.solver_data_projection    = SolverData(1000, 1.e-12, 1.e-6);
    param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // viscous step
    param.solver_viscous         = SolverViscous::CG;
    param.solver_data_viscous    = SolverData(1000, 1.e-12, 1.e-6);
    param.preconditioner_viscous = PreconditionerViscous::Multigrid;

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
    param.solver_coupled      = SolverCoupled::GMRES;
    param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-6, 200);

    // preconditioning linear solver
    param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    param.update_preconditioner_coupled = true;

    // preconditioner velocity/momentum block
    param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix; // Multigrid;
    param.multigrid_data_velocity_block.smoother_data.smoother =
      MultigridSmoother::Jacobi; // Jacobi; //Chebyshev; //GMRES;
    param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::BlockJacobi; // PointJacobi; //BlockJacobi;
    param.multigrid_data_velocity_block.smoother_data.iterations        = 5;
    param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
    param.multigrid_data_velocity_block.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block =
      SchurComplementPreconditioner::PressureConvectionDiffusion;
    param.discretization_of_laplacian = DiscretizationOfLaplacian::Classical;
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    std::vector<unsigned int> repetitions({1, 1});
    Point<dim>                point1(0.0, -H), point2(L, H);
    GridGenerator::subdivided_hyper_rectangle(*triangulation, repetitions, point1, point2);

    // periodicity in x-direction
    for(auto cell : triangulation->active_cell_iterators())
    {
      for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      {
        if((std::fabs(cell->face(f)->center()(0) - 0.0) < 1e-12))
          cell->face(f)->set_boundary_id(0 + 10);
        if((std::fabs(cell->face(f)->center()(0) - L) < 1e-12))
          cell->face(f)->set_boundary_id(1 + 10);
      }
    }

    auto tria = dynamic_cast<Triangulation<dim> *>(&*triangulation);
    GridTools::collect_periodic_faces(*tria, 0 + 10, 1 + 10, 0, periodic_faces);
    triangulation->add_periodicity(periodic_faces);

    triangulation->refine_global(n_refine_space);
  }

  void
  set_boundary_conditions(std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
                          std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    boundary_descriptor_velocity->dirichlet_bc.insert(pair(
      0, new AnalyticalSolutionVelocity<dim>(H, MAX_VELOCITY, ALPHA, EPSILON, FE, OMEGA, EIG_VEC)));

    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution_velocity.reset(
      new AnalyticalSolutionVelocity<dim>(H, MAX_VELOCITY, ALPHA, EPSILON, FE, OMEGA, EIG_VEC));
    field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new RightHandSide<dim>(VISCOSITY, MAX_VELOCITY, H));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output         = false;
    pp_data.output_data.output_folder        = output_directory;
    pp_data.output_data.output_name          = output_name;
    pp_data.output_data.output_start_time    = start_time;
    pp_data.output_data.output_interval_time = (end_time - start_time) / 20;
    pp_data.output_data.write_divergence     = true;
    pp_data.output_data.degree               = degree;

    MyPostProcessorData<dim> pp_data_os;
    pp_data_os.pp_data = pp_data;

    // perturbation energy
    pp_data_os.energy_data.calculate                  = true;
    pp_data_os.energy_data.calculate_every_time_steps = 1;
    pp_data_os.energy_data.filename_prefix =
      output_directory + output_name + "_perturbation_energy";
    pp_data_os.energy_data.U_max   = MAX_VELOCITY;
    pp_data_os.energy_data.h       = H;
    pp_data_os.energy_data.omega_i = OMEGA.imag();

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new MyPostProcessor<dim, Number>(pp_data_os, mpi_comm));

    return pp;
  }
};

} // namespace OrrSommerfeld
} // namespace IncNS
} // namespace ExaDG

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_ORR_SOMMERFELD_H_ */
