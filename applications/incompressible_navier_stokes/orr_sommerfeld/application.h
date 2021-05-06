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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_ORR_SOMMERFELD_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_ORR_SOMMERFELD_H_

// Orr-Sommerfeld application
#include "include/orr_sommerfeld_equation.h"
#include "include/postprocessor.h"

namespace ExaDG
{
namespace IncNS
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
  value(Point<dim> const & p, unsigned int const component = 0) const
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
  double const H            = 1.0;
  double const MAX_VELOCITY = 1.0;
  double const ALPHA        = 1.0;
  double const EPSILON      = 1.0e-5;

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
  value(Point<dim> const & /*p*/, unsigned int const component = 0) const
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
  typedef typename ApplicationBase<dim, Number>::PeriodicFaces PeriodicFaces;

  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);

    // solve Orr-Sommerfeld equation
    compute_eigenvector(EIG_VEC, OMEGA, Re, ALPHA, FE);
    // calculate characteristic time interval
    t0       = 2.0 * PI * ALPHA / OMEGA.real();
    end_time = 2.0 * t0;
  }

  double const Re = 7500.0;

  double const H  = 1.0;
  double const PI = numbers::PI;
  double const L  = 2.0 * PI * H;

  double const MAX_VELOCITY = 1.0;
  double const VISCOSITY    = MAX_VELOCITY * H / Re;
  double const ALPHA        = 1.0;
  double const EPSILON      = 1.0e-5; // perturbations are small (<< 1, linearization)

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
    param.problem_type                = ProblemType::Unsteady;
    param.equation_type               = EquationType::NavierStokes;
    param.formulation_viscous_term    = FormulationViscousTerm::LaplaceFormulation;
    param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
    param.right_hand_side             = true; // prescribe body force in x-direction


    // PHYSICAL QUANTITIES
    param.start_time = start_time;
    param.end_time   = end_time;
    param.viscosity  = VISCOSITY;


    // TEMPORAL DISCRETIZATION
    param.solver_type                     = SolverType::Unsteady;
    param.temporal_discretization         = TemporalDiscretization::BDFCoupledSolution;
    param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    param.order_time_integrator           = 2;
    param.start_with_low_order            = true;
    param.adaptive_time_stepping          = false;
    param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    param.max_velocity                    = MAX_VELOCITY;
    param.cfl                             = 0.2;
    param.cfl_exponent_fe_degree_velocity = 1.5;
    param.time_step_size                  = 1.0e-2;
    param.max_number_of_time_steps        = 1e8;

    // output of solver information
    param.solver_info_data.interval_time = (param.end_time - param.start_time) / 20;


    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_p           = DegreePressure::MixedOrder;
    param.mapping            = MappingType::Isoparametric;

    // convective term
    if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      param.upwind_factor = 0.5;

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // divergence and continuity penalty terms
    param.use_divergence_penalty                     = true;
    param.divergence_penalty_factor                  = 1.0e0;
    param.use_continuity_penalty                     = true;
    param.continuity_penalty_factor                  = param.divergence_penalty_factor;
    param.continuity_penalty_use_boundary_data       = true;
    param.apply_penalty_terms_in_postprocessing_step = true;

    // PROJECTION METHODS

    // formulation
    param.store_previous_boundary_values = true;

    // pressure Poisson equation
    param.solver_pressure_poisson              = SolverPressurePoisson::CG;
    param.solver_data_pressure_poisson         = SolverData(1000, 1.e-14, 1.e-14, 100);
    param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;

    // projection step
    param.solver_projection         = SolverProjection::CG;
    param.solver_data_projection    = SolverData(1000, 1.e-14, 1.e-14);
    param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // viscous step
    param.solver_viscous         = SolverViscous::CG;
    param.solver_data_viscous    = SolverData(1000, 1.e-14, 1.e-14);
    param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;

    // PRESSURE-CORRECTION SCHEME

    // formulation
    param.order_pressure_extrapolation = 1;
    param.rotational_formulation       = true;

    // momentum step

    // Newton solver
    param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-14, 1.e-14);

    // linear solver
    param.solver_momentum         = SolverMomentum::GMRES;
    param.solver_data_momentum    = SolverData(1e4, 1.e-14, 1.e-14, 100);
    param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;

    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-14, 1.e-14);

    // linear solver
    param.solver_coupled      = SolverCoupled::GMRES;
    param.solver_data_coupled = SolverData(1e4, 1.e-14, 1.e-14, 200);

    // preconditioning linear solver
    param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

    // preconditioner velocity/momentum block
    param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block      = SchurComplementPreconditioner::CahouetChabard;
    param.multigrid_data_pressure_block.type = MultigridType::cphMG;
  }

  void
  create_grid(std::shared_ptr<Triangulation<dim>> triangulation,
              PeriodicFaces &                     periodic_faces,
              unsigned int const                  n_refine_space,
              std::shared_ptr<Mapping<dim>> &     mapping,
              unsigned int const                  mapping_degree)
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

    mapping.reset(new MappingQGeneric<dim>(mapping_degree));
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
    pp_data.output_data.write_output         = this->write_output;
    pp_data.output_data.output_folder        = this->output_directory + "vtu/";
    pp_data.output_data.output_name          = this->output_name + "_k" + std::to_string(degree);
    pp_data.output_data.output_start_time    = start_time;
    pp_data.output_data.output_interval_time = (end_time - start_time) / 20;
    pp_data.output_data.write_divergence     = true;
    pp_data.output_data.degree               = degree;

    MyPostProcessorData<dim> pp_data_os;
    pp_data_os.pp_data = pp_data;

    // perturbation energy
    pp_data_os.energy_data.calculate                  = true;
    pp_data_os.energy_data.calculate_every_time_steps = 1;
    pp_data_os.energy_data.directory                  = this->output_directory;
    pp_data_os.energy_data.filename =
      this->output_name + "_perturbation_energy" + "_k" + std::to_string(degree);
    pp_data_os.energy_data.U_max   = MAX_VELOCITY;
    pp_data_os.energy_data.h       = H;
    pp_data_os.energy_data.omega_i = OMEGA.imag();

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new MyPostProcessor<dim, Number>(pp_data_os, mpi_comm));

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

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_ORR_SOMMERFELD_H_ */
