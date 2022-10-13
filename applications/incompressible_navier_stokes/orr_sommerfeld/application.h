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
template<int dim>
class AnalyticalSolutionVelocity : public dealii::Function<dim>
{
public:
  AnalyticalSolutionVelocity(double const                              H,
                             double const                              MAX_VELOCITY,
                             double const                              ALPHA,
                             double const                              EPSILON,
                             dealii::FE_DGQ<1> const &                 FE,
                             std::complex<double> const &              OMEGA,
                             std::vector<std::complex<double>> const & EIG_VEC)
    : dealii::Function<dim>(dim, 0.0),
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
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const
  {
    double t      = this->get_time();
    double result = 0.0;

    double const x = p[0] / H;
    // transform from interval [-H,H] (-> y) to unit interval [0,1] (-> eta)
    double const eta = 0.5 * (p[1] / H + 1.0);

    Assert(eta <= 1.0 + 1.e-12 and eta >= 0.0 - 1.e-12,
           dealii::ExcMessage("Point in reference coordinates is invalid."));

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
        dpsi += EIG_VEC[i] * FE.shape_grad(i, dealii::Point<1>(eta))[0];

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
        psi += EIG_VEC[i] * FE.shape_value(i, dealii::Point<1>(eta));

      std::complex<double> i(0, 1);
      std::complex<double> perturbation_complex = -i * ALPHA * psi * exp * amplification;
      double               perturbation         = perturbation_complex.real();

      result = EPSILON * perturbation;
    }
    else
    {
      Assert(false, dealii::ExcMessage("Orr-Sommerfeld problem is a 2-dimensional problem."));
    }

    return result;
  }

private:
  double const H            = 1.0;
  double const MAX_VELOCITY = 1.0;
  double const ALPHA        = 1.0;
  double const EPSILON      = 1.0e-5;

  dealii::FE_DGQ<1> const &                 FE;
  std::complex<double> const &              OMEGA;
  std::vector<std::complex<double>> const & EIG_VEC;
};


/*
 *  Right-hand side function: Implements the body force vector occuring on the
 *  right-hand side of the momentum equation of the Navier-Stokes equations
 */
template<int dim>
class RightHandSide : public dealii::Function<dim>
{
public:
  RightHandSide(double const viscosity, double const max_velocity, double const H)
    : dealii::Function<dim>(dim, 0.0), nu(viscosity), U_max(max_velocity), H(H)
  {
  }

  double
  value(dealii::Point<dim> const & /*p*/, unsigned int const component = 0) const
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
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
    // solve Orr-Sommerfeld equation
    compute_eigenvector(EIG_VEC, OMEGA, Re, ALPHA, FE);
    // calculate characteristic time interval
    t0       = 2.0 * PI * ALPHA / OMEGA.real();
    end_time = 2.0 * t0;
  }

private:
  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type                = ProblemType::Unsteady;
    this->param.equation_type               = EquationType::NavierStokes;
    this->param.formulation_viscous_term    = FormulationViscousTerm::LaplaceFormulation;
    this->param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
    this->param.right_hand_side             = true; // prescribe body force in x-direction


    // PHYSICAL QUANTITIES
    this->param.start_time = start_time;
    this->param.end_time   = end_time;
    this->param.viscosity  = VISCOSITY;


    // TEMPORAL DISCRETIZATION
    this->param.solver_type                     = SolverType::Unsteady;
    this->param.temporal_discretization         = TemporalDiscretization::BDFCoupledSolution;
    this->param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    this->param.order_time_integrator           = 2;
    this->param.start_with_low_order            = true;
    this->param.adaptive_time_stepping          = false;
    this->param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    this->param.max_velocity                    = MAX_VELOCITY;
    this->param.cfl                             = 0.2;
    this->param.cfl_exponent_fe_degree_velocity = 1.5;
    this->param.time_step_size                  = 1.0e-2;
    this->param.max_number_of_time_steps        = 1e8;

    // output of solver information
    this->param.solver_info_data.interval_time =
      (this->param.end_time - this->param.start_time) / 20;


    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.grid.mapping_degree     = this->param.degree_u;
    this->param.degree_p                = DegreePressure::MixedOrder;

    // convective term
    if(this->param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      this->param.upwind_factor = 0.5;

    // viscous term
    this->param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // divergence and continuity penalty terms
    this->param.use_divergence_penalty                     = true;
    this->param.divergence_penalty_factor                  = 1.0e0;
    this->param.use_continuity_penalty                     = true;
    this->param.continuity_penalty_factor                  = this->param.divergence_penalty_factor;
    this->param.continuity_penalty_use_boundary_data       = true;
    this->param.apply_penalty_terms_in_postprocessing_step = true;

    // PROJECTION METHODS

    // pressure Poisson equation
    this->param.solver_pressure_poisson              = SolverPressurePoisson::CG;
    this->param.solver_data_pressure_poisson         = SolverData(1000, 1.e-14, 1.e-14, 100);
    this->param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    this->param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;

    // projection step
    this->param.solver_projection         = SolverProjection::CG;
    this->param.solver_data_projection    = SolverData(1000, 1.e-14, 1.e-14);
    this->param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    this->param.order_extrapolation_pressure_nbc =
      this->param.order_time_integrator <= 2 ? this->param.order_time_integrator : 2;

    // viscous step
    this->param.solver_viscous         = SolverViscous::CG;
    this->param.solver_data_viscous    = SolverData(1000, 1.e-14, 1.e-14);
    this->param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;

    // PRESSURE-CORRECTION SCHEME

    // formulation
    this->param.order_pressure_extrapolation = 1;
    this->param.rotational_formulation       = true;

    // momentum step

    // Newton solver
    this->param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-14, 1.e-14);

    // linear solver
    this->param.solver_momentum         = SolverMomentum::GMRES;
    this->param.solver_data_momentum    = SolverData(1e4, 1.e-14, 1.e-14, 100);
    this->param.preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;

    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    this->param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-14, 1.e-14);

    // linear solver
    this->param.solver_coupled      = SolverCoupled::GMRES;
    this->param.solver_data_coupled = SolverData(1e4, 1.e-14, 1.e-14, 200);

    // preconditioning linear solver
    this->param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

    // preconditioner velocity/momentum block
    this->param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

    // preconditioner Schur-complement block
    this->param.preconditioner_pressure_block      = SchurComplementPreconditioner::CahouetChabard;
    this->param.multigrid_data_pressure_block.type = MultigridType::cphMG;
  }

  void
  create_grid() final
  {
    std::vector<unsigned int> repetitions({1, 1});
    dealii::Point<dim>        point1(0.0, -H), point2(L, H);
    dealii::GridGenerator::subdivided_hyper_rectangle(*this->grid->triangulation,
                                                      repetitions,
                                                      point1,
                                                      point2);

    // periodicity in x-direction
    for(auto cell : this->grid->triangulation->cell_iterators())
    {
      for(auto const & f : cell->face_indices())
      {
        if((std::fabs(cell->face(f)->center()(0) - 0.0) < 1e-12))
          cell->face(f)->set_boundary_id(0 + 10);
        if((std::fabs(cell->face(f)->center()(0) - L) < 1e-12))
          cell->face(f)->set_boundary_id(1 + 10);
      }
    }

    dealii::GridTools::collect_periodic_faces(
      *this->grid->triangulation, 0 + 10, 1 + 10, 0, this->grid->periodic_faces);
    this->grid->triangulation->add_periodicity(this->grid->periodic_faces);

    this->grid->triangulation->refine_global(this->param.grid.n_refine_global);
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    this->boundary_descriptor->velocity->dirichlet_bc.insert(pair(
      0, new AnalyticalSolutionVelocity<dim>(H, MAX_VELOCITY, ALPHA, EPSILON, FE, OMEGA, EIG_VEC)));

    this->boundary_descriptor->pressure->neumann_bc.insert(0);
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(
      new AnalyticalSolutionVelocity<dim>(H, MAX_VELOCITY, ALPHA, EPSILON, FE, OMEGA, EIG_VEC));
    this->field_functions->initial_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->analytical_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(
      new RightHandSide<dim>(VISCOSITY, MAX_VELOCITY, H));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = start_time;
    pp_data.output_data.time_control_data.trigger_interval = (end_time - start_time) / 20.0;
    pp_data.output_data.directory = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename =
      this->output_parameters.filename + "_k" + std::to_string(this->param.degree_u);
    pp_data.output_data.write_divergence = true;
    pp_data.output_data.degree           = this->param.degree_u;

    MyPostProcessorData<dim> pp_data_os;
    pp_data_os.pp_data = pp_data;

    // perturbation energy
    pp_data_os.energy_data.time_control_data.is_active                = true;
    pp_data_os.energy_data.time_control_data.start_time               = start_time;
    pp_data_os.energy_data.time_control_data.trigger_every_time_steps = 1;
    pp_data_os.energy_data.directory = this->output_parameters.directory;
    pp_data_os.energy_data.filename  = this->output_parameters.filename + "_perturbation_energy" +
                                      "_k" + std::to_string(this->param.degree_u);
    pp_data_os.energy_data.U_max   = MAX_VELOCITY;
    pp_data_os.energy_data.h       = H;
    pp_data_os.energy_data.omega_i = OMEGA.imag();

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new MyPostProcessor<dim, Number>(pp_data_os, this->mpi_comm));

    return pp;
  }

  double const Re = 7500.0;

  double const H  = 1.0;
  double const PI = dealii::numbers::PI;
  double const L  = 2.0 * PI * H;

  double const MAX_VELOCITY = 1.0;
  double const VISCOSITY    = MAX_VELOCITY * H / Re;
  double const ALPHA        = 1.0;
  double const EPSILON      = 1.0e-5; // perturbations are small (<< 1, linearization)

  // Orr-Sommerfeld solver: calculates unstable eigenvalue (OMEGA) of
  // Orr-Sommerfeld equation for Poiseuille flow and corresponding eigenvector (EIG_VEC).
  // do not use more than 300 due to conditioning of polynomials
  unsigned int const DEGREE_OS_SOLVER = 200;

  dealii::FE_DGQ<1> FE = dealii::FE_DGQ<1>(DEGREE_OS_SOLVER);

  std::complex<double> OMEGA;

  std::vector<std::complex<double>> EIG_VEC =
    std::vector<std::complex<double>>(DEGREE_OS_SOLVER + 1);

  // the time the Tollmien-Schlichting-waves need to travel through the domain
  // (depends on solution of Orr-Sommerfeld equation)
  double       t0         = 0.0;
  double const start_time = 0.0;
  double       end_time   = 2.0 * t0;
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_ORR_SOMMERFELD_H_ */
