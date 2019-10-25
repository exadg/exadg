/*
 * time_int_bdf_pressure_correction.cpp
 *
 *  Created on: Nov 15, 2018
 *      Author: fehn
 */

#include "time_int_bdf_pressure_correction.h"

#include "../spatial_discretization/interface.h"
#include "../user_interface/input_parameters.h"
#include "functionalities/set_zero_mean_value.h"
#include "time_integration/push_back_vectors.h"

namespace IncNS
{
template<typename Number>
TimeIntBDFPressureCorrection<Number>::TimeIntBDFPressureCorrection(
  std::shared_ptr<InterfaceBase> operator_base_in,
  std::shared_ptr<InterfacePDE>  pde_operator_in,
  InputParameters const &        param_in)
  : TimeIntBDF<Number>(operator_base_in, param_in),
    pde_operator(pde_operator_in),
    velocity(param_in.order_time_integrator),
    pressure(param_in.order_time_integrator),
    vec_convective_term(param_in.order_time_integrator),
    order_pressure_extrapolation(param_in.order_pressure_extrapolation),
    extra_pressure_gradient(param_in.order_pressure_extrapolation, param_in.start_with_low_order),
    vec_pressure_gradient_term(param_in.order_pressure_extrapolation),
    computing_times(3),
    iterations(3),
    N_iter_nonlinear_momentum(0)
{
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::update_time_integrator_constants()
{
  // call function of base class to update the standard time integrator constants
  TimeIntBDF<Number>::update_time_integrator_constants();

  // update time integrator constants for extrapolation scheme of pressure gradient term

  // if start_with_low_order == true (no analytical solution available)
  // the pressure is unknown at t = t_0:
  // -> use no extrapolation (order=0, non-incremental) in first time step (the pressure solution is
  // calculated in the second sub step)
  // -> use first order extrapolation in second time step, second order extrapolation in third time
  // step, etc.
  if(this->adaptive_time_stepping == false)
  {
    extra_pressure_gradient.update(this->get_time_step_number() - 1);
  }
  else // adaptive time stepping
  {
    extra_pressure_gradient.update(this->get_time_step_number() - 1, this->get_time_step_vector());
  }

  // use this function to check the correctness of the time integrator constants
  //  std::cout << "Coefficients extrapolation scheme pressure: Time step = " <<
  //  this->get_time_step_number() << std::endl; extra_pressure_gradient.print();
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::setup_derived()
{
  if(this->param.convective_problem() && this->start_with_low_order == false &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    initialize_vec_convective_term();
  }

  if(extra_pressure_gradient.get_order() > 0)
    initialize_vec_pressure_gradient_term();
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::allocate_vectors()
{
  // velocity
  for(unsigned int i = 0; i < velocity.size(); ++i)
    this->operator_base->initialize_vector_velocity(velocity[i]);
  this->operator_base->initialize_vector_velocity(velocity_np);

  // pressure
  for(unsigned int i = 0; i < pressure.size(); ++i)
    this->operator_base->initialize_vector_pressure(pressure[i]);
  this->operator_base->initialize_vector_pressure(pressure_np);
  this->operator_base->initialize_vector_pressure(pressure_increment);

  // vec_convective_term
  if(this->param.convective_problem() &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    for(unsigned int i = 0; i < vec_convective_term.size(); ++i)
      this->operator_base->initialize_vector_velocity(vec_convective_term[i]);
  }

  // vec_pressure_gradient_term
  for(unsigned int i = 0; i < vec_pressure_gradient_term.size(); ++i)
    this->operator_base->initialize_vector_velocity(vec_pressure_gradient_term[i]);

  // Sum_i (alpha_i/dt * u_i)
  this->operator_base->initialize_vector_velocity(this->sum_alphai_ui);
}


template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::initialize_current_solution()
{
  this->operator_base->prescribe_initial_conditions(velocity[0], pressure[0], this->get_time());
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::initialize_former_solutions()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i = 1; i < velocity.size(); ++i)
  {
    this->operator_base->prescribe_initial_conditions(velocity[i],
                                                      pressure[i],
                                                      this->get_previous_time(i));
  }
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::initialize_vec_convective_term()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i = 1; i < vec_convective_term.size(); ++i)
  {
    this->operator_base->evaluate_convective_term(vec_convective_term[i],
                                                  velocity[i],
                                                  this->get_previous_time(i));
  }
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::initialize_vec_pressure_gradient_term()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i = 1; i < vec_pressure_gradient_term.size(); ++i)
  {
    this->operator_base->evaluate_pressure_gradient_term(vec_pressure_gradient_term[i],
                                                         pressure[i],
                                                         this->get_previous_time(i));
  }
}

template<typename Number>
LinearAlgebra::distributed::Vector<Number> const &
TimeIntBDFPressureCorrection<Number>::get_velocity() const
{
  return velocity[0];
}

template<typename Number>
LinearAlgebra::distributed::Vector<Number> const &
TimeIntBDFPressureCorrection<Number>::get_velocity(unsigned int i) const
{
  return velocity[i];
}

template<typename Number>
LinearAlgebra::distributed::Vector<Number> const &
TimeIntBDFPressureCorrection<Number>::get_pressure(unsigned int i) const
{
  return pressure[i];
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::set_velocity(VectorType const & velocity_in,
                                                   unsigned int const i)
{
  velocity[i] = velocity_in;
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::set_pressure(VectorType const & pressure_in,
                                                   unsigned int const i)
{
  pressure[i] = pressure_in;
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::postprocessing() const
{
  pde_operator->do_postprocessing(velocity[0],
                                  pressure[0],
                                  this->get_time(),
                                  this->get_time_step_number());
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::postprocessing_steady_problem() const
{
  pde_operator->do_postprocessing_steady_problem(velocity[0], pressure[0]);
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::postprocessing_stability_analysis()
{
  AssertThrow(this->order == 1,
              ExcMessage("Order of BDF scheme has to be 1 for this stability analysis."));

  AssertThrow(velocity[0].l2_norm() < 1.e-15 && pressure[0].l2_norm() < 1.e-15,
              ExcMessage("Solution vector has to be zero for this stability analysis."));

  AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
              ExcMessage("Number of MPI processes has to be 1."));

  std::cout << std::endl << "Analysis of eigenvalue spectrum:" << std::endl;

  const unsigned int size = velocity[0].local_size();

  LAPACKFullMatrix<Number> propagation_matrix(size, size);

  // loop over all columns of propagation matrix
  for(unsigned int j = 0; j < size; ++j)
  {
    // set j-th element to 1
    velocity[0].local_element(j) = 1.0;

    // solve time step
    solve_timestep();

    // dst-vector velocity_np is j-th column of propagation matrix
    for(unsigned int i = 0; i < size; ++i)
    {
      propagation_matrix(i, j) = velocity_np.local_element(i);
    }

    // reset j-th element to 0
    velocity[0].local_element(j) = 0.0;
  }

  // compute eigenvalues
  propagation_matrix.compute_eigenvalues();

  double norm_max = 0.0;

  std::cout << "List of all eigenvalues:" << std::endl;

  for(unsigned int i = 0; i < size; ++i)
  {
    double norm = std::abs(propagation_matrix.eigenvalue(i));
    if(norm > norm_max)
      norm_max = norm;

    // print eigenvalues
    std::cout << std::scientific << std::setprecision(5) << propagation_matrix.eigenvalue(i)
              << std::endl;
  }

  std::cout << std::endl << std::endl << "Maximum eigenvalue = " << norm_max << std::endl;
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::solve_timestep()
{
  // perform the substeps of the pressure-correction scheme
  momentum_step();

  pressure_step();

  projection_step();
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::momentum_step()
{
  Timer timer;
  timer.restart();

  /*
   *  if a turbulence model is used:
   *  update turbulence model before calculating rhs_momentum
   */
  if(this->param.use_turbulence_model == true)
  {
    Timer timer_turbulence;
    timer_turbulence.restart();

    velocity_np = 0.0;
    for(unsigned int i = 0; i < velocity.size(); ++i)
    {
      velocity_np.add(this->extra.get_beta(i), velocity[i]);
    }

    this->operator_base->update_turbulence_model(velocity_np);

    if(this->print_solver_info())
    {
      this->pcout << std::endl
                  << "Update of turbulent viscosity:   Wall time [s]: " << std::scientific
                  << timer_turbulence.wall_time() << std::endl;
    }
  }


  /*
   *  Calculate the right-hand side of the linear system of equations
   *  or the vector that is constant when solving the nonlinear momentum equation
   *  (where constant means that the vector does not change from one Newton iteration
   *  to the next, i.e., it does not depend on the current solution of the nonlinear solver)
   */
  VectorType rhs(velocity_np);
  rhs_momentum(rhs);

  /*
   *  Solve the linear or nonlinear problem.
   */

  bool const update_preconditioner =
    this->param.update_preconditioner_momentum &&
    ((this->time_step_number - 1) % this->param.update_preconditioner_momentum_every_time_steps ==
     0);

  if(this->param.linear_problem_has_to_be_solved())
  {
    if(this->param.viscous_problem())
    {
      /*
       *  Extrapolate old solution to get a good initial estimate for the solver.
       */
      velocity_np = 0.0;
      for(unsigned int i = 0; i < velocity.size(); ++i)
      {
        velocity_np.add(this->extra.get_beta(i), velocity[i]);
      }

      // solve linear system of equations
      unsigned int linear_iterations_momentum = 0;

      pde_operator->solve_linear_momentum_equation(velocity_np,
                                                   rhs,
                                                   update_preconditioner,
                                                   this->get_scaling_factor_time_derivative_term(),
                                                   linear_iterations_momentum);

      // write output explicit case
      if(this->print_solver_info())
      {
        this->pcout << std::endl
                    << "Solve linear momentum equation for intermediate velocity:" << std::endl
                    << "  Iterations:        " << std::setw(6) << std::right
                    << linear_iterations_momentum << "\t Wall time [s]: " << std::scientific
                    << timer.wall_time() << std::endl;
      }

      iterations[0] += linear_iterations_momentum;
    }
    else // Euler equations
    {
      this->operator_base->apply_inverse_mass_matrix(velocity_np, rhs);
      velocity_np *= this->get_time_step_size() / this->bdf.get_gamma0();

      // write output explicit case
      if(this->print_solver_info())
      {
        this->pcout << std::endl
                    << "Solve linear momentum equation for intermediate velocity:" << std::endl
                    << "  Iterations:        " << std::setw(6) << std::right << 0
                    << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
      }
    }
  }
  else // nonlinear problem
  {
    AssertThrow(this->param.nonlinear_problem_has_to_be_solved(), ExcMessage("Logical error."));

    /*
     *  Extrapolate old solution to get a good initial estimate for the solver.
     */
    velocity_np = 0.0;
    for(unsigned int i = 0; i < velocity.size(); ++i)
    {
      velocity_np.add(this->extra.get_beta(i), velocity[i]);
    }

    unsigned int linear_iterations_momentum;
    unsigned int nonlinear_iterations_momentum;

    pde_operator->solve_nonlinear_momentum_equation(velocity_np,
                                                    rhs,
                                                    this->get_next_time(),
                                                    update_preconditioner,
                                                    this->get_scaling_factor_time_derivative_term(),
                                                    nonlinear_iterations_momentum,
                                                    linear_iterations_momentum);

    // write output implicit case
    if(this->print_solver_info())
    {
      this->pcout << std::endl
                  << "Solve nonlinear momentum equation for intermediate velocity:" << std::endl
                  << "  Newton iterations: " << std::setw(6) << std::right
                  << nonlinear_iterations_momentum << "\t Wall time [s]: " << std::scientific
                  << timer.wall_time() << std::endl
                  << "  Linear iterations: " << std::setw(6) << std::right << std::fixed
                  << std::setprecision(2)
                  << (nonlinear_iterations_momentum > 0 ?
                        (double)linear_iterations_momentum / (double)nonlinear_iterations_momentum :
                        linear_iterations_momentum)
                  << " (avg)" << std::endl
                  << "  Linear iterations: " << std::setw(6) << std::right << std::fixed
                  << std::setprecision(2) << linear_iterations_momentum << " (tot)" << std::endl;
    }

    iterations[0] += linear_iterations_momentum;
    N_iter_nonlinear_momentum += nonlinear_iterations_momentum;
  }

  computing_times[0] += timer.wall_time();
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::rhs_momentum(VectorType & rhs)
{
  rhs = 0.0;

  /*
   *  Add extrapolation of pressure gradient term to the rhs in case of incremental formulation
   */
  if(extra_pressure_gradient.get_order() > 0)
  {
    this->operator_base->evaluate_pressure_gradient_term(vec_pressure_gradient_term[0],
                                                         pressure[0],
                                                         this->get_time());

    for(unsigned int i = 0; i < extra_pressure_gradient.get_order(); ++i)
      rhs.add(-extra_pressure_gradient.get_beta(i), vec_pressure_gradient_term[i]);
  }

  /*
   *  Body force term
   */
  if(this->param.right_hand_side == true)
  {
    this->operator_base->evaluate_add_body_force_term(rhs, this->get_next_time());
  }

  /*
   *  Convective term formulated explicitly (additive decomposition):
   *  Evaluate convective term and add extrapolation of convective term to the rhs
   */
  if(this->param.convective_problem() &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    this->operator_base->evaluate_convective_term(vec_convective_term[0],
                                                  velocity[0],
                                                  this->get_time());

    for(unsigned int i = 0; i < vec_convective_term.size(); ++i)
      rhs.add(-this->extra.get_beta(i), vec_convective_term[i]);
  }

  /*
   *  calculate sum (alpha_i/dt * u_i): This term is relevant for both the explicit
   *  and the implicit formulation of the convective term
   */
  // calculate sum (alpha_i/dt * u_tilde_i) in case of explicit treatment of convective term
  // and operator-integration-factor (OIF) splitting
  if(this->param.convective_problem() &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    this->calculate_sum_alphai_ui_oif_substepping(this->cfl, this->cfl_oif);
  }
  // calculate sum (alpha_i/dt * u_i) for standard BDF discretization
  else
  {
    this->sum_alphai_ui.equ(this->bdf.get_alpha(0) / this->get_time_step_size(), velocity[0]);
    for(unsigned int i = 1; i < velocity.size(); ++i)
    {
      this->sum_alphai_ui.add(this->bdf.get_alpha(i) / this->get_time_step_size(), velocity[i]);
    }
  }

  this->operator_base->apply_mass_matrix_add(rhs, this->sum_alphai_ui);

  /*
   *  Right-hand side viscous term:
   *  If a linear system of equations has to be solved,
   *  inhomogeneous parts of boundary face integrals of the viscous operator
   *  have to be shifted to the right-hand side of the equation.
   */
  if(this->param.viscous_problem() && this->param.linear_problem_has_to_be_solved())
  {
    pde_operator->rhs_add_viscous_term(rhs, this->get_next_time());
  }
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::pressure_step()
{
  Timer timer;
  timer.restart();

  // compute right-hand side vector
  VectorType rhs(pressure_np);
  rhs_pressure(rhs);

  // extrapolate old solution to get a good initial estimate for the solver

  // calculate initial guess for pressure solve
  pressure_increment = 0.0;

  // extrapolate old solution to get a good initial estimate for the
  // pressure solution p_{n+1} at time t^{n+1}
  for(unsigned int i = 0; i < pressure.size(); ++i)
  {
    pressure_increment.add(this->extra.get_beta(i), pressure[i]);
  }

  // incremental formulation
  if(extra_pressure_gradient.get_order() > 0)
  {
    // Subtract extrapolation of pressure since the PPE is solved for the
    // pressure increment phi = p_{n+1} - sum_i (beta_pressure_extra_i * pressure_i),
    // where p_{n+1} is approximated by an extrapolation of order J (=order of BDF scheme).
    // Note that the divergence correction term in case of the rotational formulation is not
    // considered when calculating a good initial guess for the solution of the PPE,
    // which will slightly increase the number of iterations compared to the standard
    // formulation of the pressure-correction scheme.
    for(unsigned int i = 0; i < extra_pressure_gradient.get_order(); ++i)
    {
      pressure_increment.add(-extra_pressure_gradient.get_beta(i), pressure[i]);
    }
  }

  // solve linear system of equations
  unsigned int iterations_pressure = pde_operator->solve_pressure(pressure_increment, rhs);

  // calculate pressure p^{n+1} from pressure increment
  pressure_update();

  // Special case: pure Dirichlet BC's
  // Adjust the pressure level in order to allow a calculation of the pressure error.
  // This is necessary because otherwise the pressure solution moves away from the exact solution.
  // For some test cases it was found that ApplyZeroMeanValue works better than
  // ApplyAnalyticalSolutionInPoint
  if(this->param.pure_dirichlet_bc)
  {
    if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalSolutionInPoint)
    {
      this->operator_base->shift_pressure(pressure_np, this->get_next_time());
    }
    else if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyZeroMeanValue)
    {
      set_zero_mean_value(pressure_np);
    }
    else if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalMeanValue)
    {
      this->operator_base->shift_pressure_mean_value(pressure_np, this->get_next_time());
    }
    else
    {
      AssertThrow(false,
                  ExcMessage("Specified method to adjust pressure level is not implemented."));
    }
  }

  // write output
  if(this->print_solver_info())
  {
    this->pcout << std::endl
                << "Solve Poisson equation for pressure p:" << std::endl
                << "  Iterations:        " << std::setw(6) << std::right << iterations_pressure
                << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
  }

  computing_times[1] += timer.wall_time();
  iterations[1] += iterations_pressure;
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::calculate_chi(double & chi) const
{
  if(this->param.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
  {
    chi = 1.0;
  }
  else if(this->param.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
  {
    chi = 2.0;
  }
  else
  {
    AssertThrow(this->param.formulation_viscous_term ==
                    FormulationViscousTerm::LaplaceFormulation &&
                  this->param.formulation_viscous_term ==
                    FormulationViscousTerm::DivergenceFormulation,
                ExcMessage("Not implemented!"));
  }
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::rhs_pressure(VectorType & rhs) const
{
  /*
   *  I. calculate divergence term
   */
  VectorType temp(pressure_np);
  this->operator_base->evaluate_velocity_divergence_term(temp, velocity_np, this->get_next_time());

  rhs.equ(-this->bdf.get_gamma0() / this->get_time_step_size(), temp);


  /*
   *  II. calculate terms originating from inhomogeneous parts of boundary face integrals,
   *  i.e., pressure Dirichlet boundary conditions on Gamma_N and
   *  pressure Neumann boundary conditions on Gamma_D (always h=0 for pressure-correction scheme!)
   */
  pde_operator->rhs_ppe_laplace_add(rhs, this->get_next_time());

  // incremental formulation of pressure-correction scheme
  for(unsigned int i = 0; i < extra_pressure_gradient.get_order(); ++i)
  {
    // set temp to zero since rhs_ppe_laplace_add() adds into the vector
    temp           = 0.0;
    double const t = this->get_previous_time(i);
    pde_operator->rhs_ppe_laplace_add(temp, t);
    rhs.add(-extra_pressure_gradient.get_beta(i), temp);
  }

  // special case: pure Dirichlet BC's
  // Unclear if this is really necessary, because from a theoretical
  // point of view one would expect that the mean value of the rhs of the
  // presssure Poisson equation is zero if consistent Dirichlet boundary
  // conditions are prescribed.
  // In principle, it works (since the linear system of equations is consistent)
  // but we detected no convergence for some test cases and specific parameters.
  // Hence, for reasons of robustness we also solve a transformed linear system of equations
  // in case of the pressure-correction scheme.

  if(this->param.pure_dirichlet_bc)
    set_zero_mean_value(rhs);
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::pressure_update()
{
  // First set pressure solution to zero.
  pressure_np = 0.0;

  // Rotational formulation only (this step is performed first in order
  // to avoid the storage of another temporary variable).
  if(this->param.rotational_formulation == true)
  {
    // Automatically sets pressure_np to zero before operator evaluation.
    this->operator_base->evaluate_velocity_divergence_term(pressure_np,
                                                           velocity_np,
                                                           this->get_next_time());

    pde_operator->apply_inverse_pressure_mass_matrix(pressure_np, pressure_np);

    double chi = 0.0;
    calculate_chi(chi);

    pressure_np *= -chi * this->param.viscosity;
  }

  // This is done for both the incremental and the non-incremental formulation,
  // the standard and the rotational formulation.
  pressure_np.add(1.0, pressure_increment);

  // Incremental formulation only.

  // add extrapolation of pressure to the pressure-increment solution in order to obtain
  // the pressure solution at the end of the time step, i.e.,
  // p^{n+1} = (pressure_increment)^{n+1} + sum_i (beta_pressure_extrapolation_i * p^{n-i});
  for(unsigned int i = 0; i < extra_pressure_gradient.get_order(); ++i)
  {
    pressure_np.add(extra_pressure_gradient.get_beta(i), pressure[i]);
  }
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::rhs_projection(VectorType & rhs) const
{
  /*
   *  I. calculate mass matrix term
   */
  this->operator_base->apply_mass_matrix(rhs, velocity_np);

  /*
   *  II. calculate pressure gradient term including boundary condition g_p(t_{n+1})
   */
  VectorType temp(rhs);
  this->operator_base->evaluate_pressure_gradient_term(temp,
                                                       pressure_increment,
                                                       this->get_next_time());

  rhs.add(-this->get_time_step_size() / this->bdf.get_gamma0(), temp);

  /*
   *  III. pressure gradient term: boundary conditions g_p(t_{n-i})
   *       in case of incremental formulation of pressure-correction scheme
   */
  for(unsigned int i = 0; i < extra_pressure_gradient.get_order(); ++i)
  {
    // evaluate inhomogeneous parts of boundary face integrals
    double const current_time = this->get_previous_time(i);
    // note that the function rhs_...() already includes a factor of -1.0
    pde_operator->rhs_pressure_gradient_term(temp, current_time);

    rhs.add(-extra_pressure_gradient.get_beta(i) * this->get_time_step_size() /
              this->bdf.get_gamma0(),
            temp);
  }
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::projection_step()
{
  Timer timer;
  timer.restart();

  // compute right-hand-side vector
  VectorType rhs(velocity_np);
  rhs_projection(rhs);

  // apply inverse mass matrix: this is the solution if no penalty terms are applied
  // and serves as a good initial guess for the case with penalty terms
  this->operator_base->apply_inverse_mass_matrix(velocity_np, rhs);

  unsigned int iterations_projection = 0;

  // extrapolate velocity to time t_n+1 and use this velocity field to
  // calculate the penalty parameter for the divergence and continuity penalty term
  if(this->param.use_divergence_penalty == true || this->param.use_continuity_penalty == true)
  {
    VectorType velocity_extrapolated;
    velocity_extrapolated.reinit(velocity[0]);
    for(unsigned int i = 0; i < velocity.size(); ++i)
      velocity_extrapolated.add(this->extra.get_beta(i), velocity[i]);

    this->operator_base->update_projection_operator(velocity_extrapolated,
                                                    this->get_time_step_size());

    // solve linear system of equations
    bool const update_preconditioner =
      this->param.update_preconditioner_projection &&
      ((this->time_step_number - 1) %
         this->param.update_preconditioner_projection_every_time_steps ==
       0);

    iterations_projection =
      this->operator_base->solve_projection(velocity_np, rhs, update_preconditioner);
  }

  // write output
  if(this->print_solver_info())
  {
    this->pcout << std::endl
                << "Solve projection step for intermediate velocity:" << std::endl
                << "  Iterations:        " << std::setw(6) << std::right << iterations_projection
                << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
  }

  computing_times[2] += timer.wall_time();
  iterations[2] += iterations_projection;
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::prepare_vectors_for_next_timestep()
{
  push_back(velocity);
  velocity[0].swap(velocity_np);

  push_back(pressure);
  pressure[0].swap(pressure_np);

  if(this->param.convective_problem() &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    push_back(vec_convective_term);
  }

  if(extra_pressure_gradient.get_order() > 0)
  {
    push_back(vec_pressure_gradient_term);
  }
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::solve_steady_problem()
{
  this->pcout << std::endl << "Starting time loop ..." << std::endl;

  // pseudo-time integration in order to solve steady-state problem
  bool converged = false;

  if(this->param.convergence_criterion_steady_problem ==
     ConvergenceCriterionSteadyProblem::SolutionIncrement)
  {
    VectorType velocity_tmp;
    VectorType pressure_tmp;

    while(!converged && this->time < (this->end_time - this->eps) &&
          this->get_time_step_number() <= this->param.max_number_of_time_steps)
    {
      // save solution from previous time step
      velocity_tmp = velocity[0];
      pressure_tmp = pressure[0];

      // calculate normm of solution
      double const norm_u = velocity_tmp.l2_norm();
      double const norm_p = pressure_tmp.l2_norm();
      double const norm   = std::sqrt(norm_u * norm_u + norm_p * norm_p);

      // solve time step
      this->do_timestep(false);

      // calculate increment:
      // increment = solution_{n+1} - solution_{n}
      //           = solution[0] - solution_tmp
      velocity_tmp *= -1.0;
      pressure_tmp *= -1.0;
      velocity_tmp.add(1.0, velocity[0]);
      pressure_tmp.add(1.0, pressure[0]);

      double const incr_u   = velocity_tmp.l2_norm();
      double const incr_p   = pressure_tmp.l2_norm();
      double const incr     = std::sqrt(incr_u * incr_u + incr_p * incr_p);
      double       incr_rel = 1.0;
      if(norm > 1.0e-10)
        incr_rel = incr / norm;

      // write output
      if(this->print_solver_info())
      {
        this->pcout << std::endl
                    << "Norm of solution increment:" << std::endl
                    << "  ||incr_abs|| = " << std::scientific << std::setprecision(10) << incr
                    << std::endl
                    << "  ||incr_rel|| = " << std::scientific << std::setprecision(10) << incr_rel
                    << std::endl;
      }

      // check convergence
      if(incr < this->param.abs_tol_steady || incr_rel < this->param.rel_tol_steady)
      {
        converged = true;
      }
    }
  }
  else if(this->param.convergence_criterion_steady_problem ==
          ConvergenceCriterionSteadyProblem::ResidualSteadyNavierStokes)
  {
    double const initial_residual = evaluate_residual();

    while(!converged && this->time < (this->end_time - this->eps) &&
          this->get_time_step_number() <= this->param.max_number_of_time_steps)
    {
      this->do_timestep(false);

      // check convergence by evaluating the residual of
      // the steady-state incompressible Navier-Stokes equations
      double const residual = evaluate_residual();

      if(residual < this->param.abs_tol_steady ||
         residual / initial_residual < this->param.rel_tol_steady)
      {
        converged = true;
      }
    }
  }
  else
  {
    AssertThrow(false, ExcMessage("not implemented."));
  }

  AssertThrow(
    converged == true,
    ExcMessage(
      "Maximum number of time steps or end time exceeded! This might be due to the fact that "
      "(i) the maximum number of time steps is simply too small to reach a steady solution, "
      "(ii) the problem is unsteady so that the applied solution approach is inappropriate, "
      "(iii) some of the solver tolerances are in conflict."));

  this->pcout << std::endl << "... done!" << std::endl;
}

template<typename Number>
double
TimeIntBDFPressureCorrection<Number>::evaluate_residual()
{
  pde_operator->evaluate_nonlinear_residual_steady(
    velocity_np, pressure_np, velocity[0], pressure[0], this->get_time());

  double const norm_u = velocity_np.l2_norm();
  double const norm_p = pressure_np.l2_norm();

  double residual = std::sqrt(norm_u * norm_u + norm_p * norm_p);

  // write output
  if(this->print_solver_info())
  {
    this->pcout << std::endl
                << "Norm of residual of steady Navier-Stokes equations:" << std::endl
                << "  ||r|| = " << std::scientific << std::setprecision(10) << residual
                << std::endl;
  }

  return residual;
}


template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::get_iterations(std::vector<std::string> & name,
                                                     std::vector<double> &      iteration) const
{
  unsigned int N_time_steps = this->get_time_step_number() - 1;

  if(this->param.linear_problem_has_to_be_solved())
  {
    name.resize(3);
    std::vector<std::string> names = {"Momentum", "Pressure", "Projection"};
    name                           = names;

    unsigned int N_time_steps = this->get_time_step_number() - 1;

    iteration.resize(3);
    for(unsigned int i = 0; i < this->iterations.size(); ++i)
    {
      iteration[i] = (double)this->iterations[i] / (double)N_time_steps;
    }
  }
  else // nonlinear system of equations in momentum step
  {
    name.resize(5);
    std::vector<std::string> names = {"Momentum (nonlinear)",
                                      "Momentum (linear)",
                                      "Momentum (linear-accumulated)",
                                      "Pressure",
                                      "Projection"};

    name = names;

    double n_iter_nonlinear          = (double)N_iter_nonlinear_momentum / (double)N_time_steps;
    double n_iter_linear_accumulated = (double)iterations[0] / (double)N_time_steps;
    double n_iter_pressure           = (double)iterations[1] / (double)N_time_steps;
    double n_iter_projection         = (double)iterations[2] / (double)N_time_steps;

    iteration.resize(5);
    iteration[0] = n_iter_nonlinear;
    iteration[1] = n_iter_linear_accumulated / n_iter_nonlinear;
    iteration[2] = n_iter_linear_accumulated;
    iteration[3] = n_iter_pressure;
    iteration[4] = n_iter_projection;
  }
}

template<typename Number>
void
TimeIntBDFPressureCorrection<Number>::get_wall_times(std::vector<std::string> & name,
                                                     std::vector<double> &      wall_time) const
{
  name.resize(3);
  std::vector<std::string> names = {"Momentum", "Pressure", "Projection"};
  name                           = names;

  wall_time.resize(3);
  for(unsigned int i = 0; i < this->computing_times.size(); ++i)
  {
    wall_time[i] = this->computing_times[i];
  }
}

// instantiations

template class TimeIntBDFPressureCorrection<float>;
template class TimeIntBDFPressureCorrection<double>;

} // namespace IncNS
