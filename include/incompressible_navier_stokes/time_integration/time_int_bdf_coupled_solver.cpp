/*
 * time_int_bdf_coupled_solver.cpp
 *
 *  Created on: Nov 15, 2018
 *      Author: fehn
 */

#include "time_int_bdf_coupled_solver.h"

#include "../spatial_discretization/interface.h"
#include "../user_interface/input_parameters.h"
#include "functionalities/set_zero_mean_value.h"
#include "time_integration/push_back_vectors.h"
#include "time_integration/time_step_calculation.h"

namespace IncNS
{
template<typename Number>
TimeIntBDFCoupled<Number>::TimeIntBDFCoupled(std::shared_ptr<InterfaceBase> operator_base_in,
                                             std::shared_ptr<InterfacePDE>  pde_operator_in,
                                             InputParameters const &        param_in)
  : TimeIntBDF<Number>(operator_base_in, param_in),
    pde_operator(pde_operator_in),
    solution(this->order),
    vec_convective_term(this->order),
    computing_times(2),
    iterations(2),
    N_iter_nonlinear(0.0),
    scaling_factor_continuity(1.0),
    characteristic_element_length(1.0)
{
}

template<typename Number>
void
TimeIntBDFCoupled<Number>::allocate_vectors()
{
  // solution
  for(unsigned int i = 0; i < solution.size(); ++i)
    pde_operator->initialize_block_vector_velocity_pressure(solution[i]);
  pde_operator->initialize_block_vector_velocity_pressure(solution_np);

  // convective term
  if(this->param.convective_problem() &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    for(unsigned int i = 0; i < vec_convective_term.size(); ++i)
      this->operator_base->initialize_vector_velocity(vec_convective_term[i]);
  }

  // temporal derivative term: sum_i (alpha_i * u_i)
  this->operator_base->initialize_vector_velocity(this->sum_alphai_ui);

  // rhs_vector
  if(this->param.linear_problem_has_to_be_solved())
  {
    pde_operator->initialize_block_vector_velocity_pressure(rhs_vector);
  }
}

template<typename Number>
void
TimeIntBDFCoupled<Number>::initialize_current_solution()
{
  this->operator_base->prescribe_initial_conditions(solution[0].block(0),
                                                    solution[0].block(1),
                                                    this->get_time());
}

template<typename Number>
void
TimeIntBDFCoupled<Number>::initialize_former_solutions()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i = 1; i < solution.size(); ++i)
  {
    this->operator_base->prescribe_initial_conditions(solution[i].block(0),
                                                      solution[i].block(1),
                                                      this->get_previous_time(i));
  }
}

template<typename Number>
void
TimeIntBDFCoupled<Number>::setup_derived()
{
  // scaling factor continuity equation:
  // Calculate characteristic element length h
  double characteristic_element_length = this->operator_base->calculate_minimum_element_length();

  unsigned int const degree_u = this->operator_base->get_polynomial_degree();

  characteristic_element_length =
    calculate_characteristic_element_length(characteristic_element_length, degree_u);

  // convective term treated explicitly (additive decomposition)
  if(this->param.convective_problem() &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit &&
     this->param.start_with_low_order == false)
  {
    initialize_vec_convective_term();
  }
}

template<typename Number>
void
TimeIntBDFCoupled<Number>::initialize_vec_convective_term()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i = 1; i < vec_convective_term.size(); ++i)
  {
    this->operator_base->evaluate_convective_term(vec_convective_term[i],
                                                  solution[i].block(0),
                                                  this->get_previous_time(i));
  }
}

template<typename Number>
LinearAlgebra::distributed::Vector<Number> const &
TimeIntBDFCoupled<Number>::get_velocity() const
{
  return solution[0].block(0);
}

template<typename Number>
LinearAlgebra::distributed::Vector<Number> const &
TimeIntBDFCoupled<Number>::get_velocity(unsigned int i) const
{
  return solution[i].block(0);
}

template<typename Number>
LinearAlgebra::distributed::Vector<Number> const &
TimeIntBDFCoupled<Number>::get_pressure(unsigned int i) const
{
  return solution[i].block(1);
}

template<typename Number>
void
TimeIntBDFCoupled<Number>::set_velocity(VectorType const & velocity_in, unsigned int const i)
{
  solution[i].block(0) = velocity_in;
}

template<typename Number>
void
TimeIntBDFCoupled<Number>::set_pressure(VectorType const & pressure_in, unsigned int const i)
{
  solution[i].block(1) = pressure_in;
}

template<typename Number>
void
TimeIntBDFCoupled<Number>::solve_timestep()
{
  Timer timer;
  timer.restart();

  // update scaling factor of continuity equation
  if(this->param.use_scaling_continuity == true)
  {
    scaling_factor_continuity = this->param.scaling_factor_continuity *
                                characteristic_element_length / this->get_time_step_size();
    pde_operator->set_scaling_factor_continuity(scaling_factor_continuity);
  }
  else // use_scaling_continuity == false
  {
    scaling_factor_continuity = 1.0;
  }

  // extrapolate old solution to obtain a good initial guess for the solver
  solution_np.equ(this->extra.get_beta(0), solution[0]);
  for(unsigned int i = 1; i < solution.size(); ++i)
    solution_np.add(this->extra.get_beta(i), solution[i]);

  // update of turbulence model
  if(this->param.use_turbulence_model == true)
  {
    Timer timer_turbulence;
    timer_turbulence.restart();

    this->operator_base->update_turbulence_model(solution_np.block(0));

    if(this->print_solver_info())
    {
      this->pcout << std::endl
                  << "Update of turbulent viscosity:   Wall time [s]: " << std::scientific
                  << timer_turbulence.wall_time() << std::endl;
    }
  }

  // calculate auxiliary variable p^{*} = 1/scaling_factor * p
  solution_np.block(1) *= 1.0 / scaling_factor_continuity;

  // Update divergence and continuity penalty operator in case
  // that these terms are added to the monolithic system of equations
  // instead of applying these terms in a postprocessing step.
  if(this->param.add_penalty_terms_to_monolithic_system == true)
  {
    if(this->param.use_divergence_penalty == true || this->param.use_continuity_penalty == true)
    {
      // extrapolate velocity to time t_n+1 and use this velocity field to
      // calculate the penalty parameter for the divergence and continuity penalty term
      VectorType velocity_extrapolated(solution[0].block(0));
      velocity_extrapolated = 0;
      for(unsigned int i = 0; i < solution.size(); ++i)
        velocity_extrapolated.add(this->extra.get_beta(i), solution[i].block(0));

      if(this->param.use_divergence_penalty == true)
      {
        pde_operator->update_divergence_penalty_operator(velocity_extrapolated);
      }
      if(this->param.use_continuity_penalty == true)
      {
        pde_operator->update_continuity_penalty_operator(velocity_extrapolated);
      }
    }
  }

  bool const update_preconditioner =
    this->param.update_preconditioner_coupled &&
    ((this->time_step_number - 1) % this->param.update_preconditioner_coupled_every_time_steps ==
     0);

  if(this->param.linear_problem_has_to_be_solved())
  {
    // calculate rhs vector for the Stokes problem, i.e., the convective term is neglected in this
    // step
    pde_operator->rhs_stokes_problem(rhs_vector, this->get_next_time());

    // Add the convective term to the right-hand side of the equations
    // if the convective term is treated explicitly (additive decomposition):
    // evaluate convective term and add extrapolation of convective term to the rhs (-> minus sign!)
    if(this->param.convective_problem() &&
       this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
    {
      this->operator_base->evaluate_convective_term(vec_convective_term[0],
                                                    solution[0].block(0),
                                                    this->get_time());

      for(unsigned int i = 0; i < vec_convective_term.size(); ++i)
        rhs_vector.block(0).add(-this->extra.get_beta(i), vec_convective_term[i]);
    }

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
      this->sum_alphai_ui.equ(this->bdf.get_alpha(0) / this->get_time_step_size(),
                              solution[0].block(0));
      for(unsigned int i = 1; i < solution.size(); ++i)
      {
        this->sum_alphai_ui.add(this->bdf.get_alpha(i) / this->get_time_step_size(),
                                solution[i].block(0));
      }
    }

    // apply mass matrix to sum_alphai_ui and add to rhs vector
    this->operator_base->apply_mass_matrix_add(rhs_vector.block(0), this->sum_alphai_ui);

    unsigned int linear_iterations =
      pde_operator->solve_linear_stokes_problem(solution_np,
                                                rhs_vector,
                                                update_preconditioner,
                                                this->get_next_time(),
                                                this->get_scaling_factor_time_derivative_term());

    iterations[0] += linear_iterations;
    computing_times[0] += timer.wall_time();

    // write output
    if(this->print_solver_info())
    {
      ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
      pcout << "Solve linear problem:" << std::endl
            << "  Iterations: " << std::setw(6) << std::right << linear_iterations
            << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
    }
  }
  else // a nonlinear system of equations has to be solved
  {
    // calculate Sum_i (alpha_i/dt * u_i)
    this->sum_alphai_ui.equ(this->bdf.get_alpha(0) / this->get_time_step_size(),
                            solution[0].block(0));
    for(unsigned int i = 1; i < solution.size(); ++i)
    {
      this->sum_alphai_ui.add(this->bdf.get_alpha(i) / this->get_time_step_size(),
                              solution[i].block(0));
    }

    VectorType rhs(this->sum_alphai_ui);
    this->operator_base->apply_mass_matrix(rhs, this->sum_alphai_ui);
    if(this->param.right_hand_side)
      this->operator_base->evaluate_add_body_force_term(rhs, this->get_next_time());

    // Newton solver
    unsigned int newton_iterations = 0;
    unsigned int linear_iterations = 0;

    pde_operator->solve_nonlinear_problem(solution_np,
                                          rhs,
                                          this->get_next_time(),
                                          update_preconditioner,
                                          this->get_scaling_factor_time_derivative_term(),
                                          newton_iterations,
                                          linear_iterations);

    N_iter_nonlinear += newton_iterations;
    iterations[0] += linear_iterations;
    computing_times[0] += timer.wall_time();

    // write output
    if(this->print_solver_info())
    {
      ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

      pcout << "Solve nonlinear problem:" << std::endl
            << "  Newton iterations: " << std::setw(6) << std::right << newton_iterations
            << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl
            << "  Linear iterations: " << std::setw(6) << std::fixed << std::setprecision(2)
            << std::right
            << ((newton_iterations > 0) ? (double(linear_iterations) / (double)newton_iterations) :
                                          linear_iterations)
            << " (avg)" << std::endl
            << "  Linear iterations: " << std::setw(6) << std::fixed << std::setprecision(2)
            << std::right << linear_iterations << " (tot)" << std::endl;
    }
  }

  // reconstruct pressure solution p from auxiliary variable p^{*}: p = scaling_factor * p^{*}
  solution_np.block(1) *= scaling_factor_continuity;

  // special case: pure Dirichlet BC's
  // Adjust the pressure level in order to allow a calculation of the pressure error.
  // This is necessary because otherwise the pressure solution moves away from the exact solution.
  if(this->param.pure_dirichlet_bc)
  {
    if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalSolutionInPoint)
    {
      this->operator_base->shift_pressure(solution_np.block(1), this->get_next_time());
    }
    else if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyZeroMeanValue)
    {
      set_zero_mean_value(solution_np.block(1));
    }
    else if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalMeanValue)
    {
      this->operator_base->shift_pressure_mean_value(solution_np.block(1), this->get_next_time());
    }
    else
    {
      AssertThrow(false,
                  ExcMessage("Specified method to adjust pressure level is not implemented."));
    }
  }

  // If the penalty terms are applied in a postprocessing step
  if(this->param.add_penalty_terms_to_monolithic_system == false)
  {
    // projection of velocity field using divergence and/or continuity penalty terms
    if(this->param.use_divergence_penalty == true || this->param.use_continuity_penalty == true)
    {
      timer.restart();

      projection_step();

      computing_times[1] += timer.wall_time();
    }
  }
}

template<typename Number>
void
TimeIntBDFCoupled<Number>::projection_step()
{
  Timer timer;
  timer.restart();

  // right-hand side term: apply mass matrix
  VectorType rhs(solution_np.block(0));
  this->operator_base->apply_mass_matrix(rhs, solution_np.block(0));

  // extrapolate velocity to time t_n+1 and use this velocity field to
  // calculate the penalty parameter for the divergence and continuity penalty term
  VectorType velocity_extrapolated(solution[0].block(0));
  velocity_extrapolated = 0;
  for(unsigned int i = 0; i < solution.size(); ++i)
    velocity_extrapolated.add(this->extra.get_beta(i), solution[i].block(0));

  // update projection operator
  this->operator_base->update_projection_operator(velocity_extrapolated,
                                                  this->get_time_step_size());

  bool const update_preconditioner =
    this->param.update_preconditioner_projection &&
    ((this->time_step_number - 1) % this->param.update_preconditioner_projection_every_time_steps ==
     0);

  // solve projection (where also the preconditioner is updated)
  unsigned int iterations_postprocessing =
    this->operator_base->solve_projection(solution_np.block(0), rhs, update_preconditioner);

  iterations[1] += iterations_postprocessing;

  // write output
  if(this->print_solver_info())
  {
    this->pcout << std::endl
                << "Solve projection step:" << std::endl
                << "  Iterations: " << std::setw(6) << std::right << iterations_postprocessing
                << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
  }
}

template<typename Number>
void
TimeIntBDFCoupled<Number>::postprocessing() const
{
  pde_operator->do_postprocessing(solution[0].block(0),
                                  solution[0].block(1),
                                  this->get_time(),
                                  this->get_time_step_number());
}

template<typename Number>
void
TimeIntBDFCoupled<Number>::postprocessing_steady_problem() const
{
  pde_operator->do_postprocessing_steady_problem(solution[0].block(0), solution[0].block(1));
}


template<typename Number>
void
TimeIntBDFCoupled<Number>::postprocessing_stability_analysis()
{
  AssertThrow(this->order == 1,
              ExcMessage("Order of BDF scheme has to be 1 for this stability analysis"));

  AssertThrow(solution[0].block(0).l2_norm() < 1.e-15 && solution[0].block(1).l2_norm() < 1.e-15,
              ExcMessage("Solution vector has to be zero for this stability analysis."));

  AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
              ExcMessage("Number of MPI processes has to be 1."));

  std::cout << std::endl << "Analysis of eigenvalue spectrum:" << std::endl;

  const unsigned int size = solution[0].block(0).local_size();

  LAPACKFullMatrix<Number> propagation_matrix(size, size);

  // loop over all columns of propagation matrix
  for(unsigned int j = 0; j < size; ++j)
  {
    // set j-th element to 1
    solution[0].block(0).local_element(j) = 1.0;

    // solve time step
    solve_timestep();

    // dst-vector velocity_np is j-th column of propagation matrix
    for(unsigned int i = 0; i < size; ++i)
    {
      propagation_matrix(i, j) = solution_np.block(0).local_element(i);
    }

    // reset j-th element to 0
    solution[0].block(0).local_element(j) = 0.0;
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
TimeIntBDFCoupled<Number>::prepare_vectors_for_next_timestep()
{
  push_back(solution);
  solution[0].swap(solution_np);

  if(this->param.convective_problem() &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    push_back(vec_convective_term);
  }
}

template<typename Number>
void
TimeIntBDFCoupled<Number>::solve_steady_problem()
{
  this->pcout << std::endl << "Starting time loop ..." << std::endl;

  double const initial_residual = evaluate_residual();

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
      velocity_tmp = this->solution[0].block(0);
      pressure_tmp = this->solution[0].block(1);

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
      velocity_tmp.add(1.0, this->solution[0].block(0));
      pressure_tmp.add(1.0, this->solution[0].block(1));

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
TimeIntBDFCoupled<Number>::evaluate_residual()
{
  this->pde_operator->evaluate_nonlinear_residual_steady(this->solution_np,
                                                         this->solution[0],
                                                         this->get_time());

  double residual = this->solution_np.l2_norm();

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
TimeIntBDFCoupled<Number>::get_iterations(std::vector<std::string> & name,
                                          std::vector<double> &      iteration) const
{
  unsigned int N_time_steps = this->get_time_step_number() - 1;

  if(this->param.linear_problem_has_to_be_solved())
  {
    name.resize(2);
    std::vector<std::string> names = {"Coupled system", "Projection"};
    name                           = names;

    unsigned int N_time_steps = this->get_time_step_number() - 1;

    iteration.resize(2);
    for(unsigned int i = 0; i < this->iterations.size(); ++i)
    {
      iteration[i] = (double)this->iterations[i] / (double)N_time_steps;
    }
  }
  else // nonlinear system of equations in momentum step
  {
    name.resize(4);
    std::vector<std::string> names = {"Coupled system (nonlinear)",
                                      "Coupled system (linear)",
                                      "Coupled system (linear-accumulated)",
                                      "Projection"};

    name = names;

    double n_iter_nonlinear          = (double)N_iter_nonlinear / (double)N_time_steps;
    double n_iter_linear_accumulated = (double)iterations[0] / (double)N_time_steps;
    double n_iter_projection         = (double)iterations[1] / (double)N_time_steps;

    iteration.resize(4);
    iteration[0] = n_iter_nonlinear;
    iteration[1] = n_iter_linear_accumulated / n_iter_nonlinear;
    iteration[2] = n_iter_linear_accumulated;
    iteration[3] = n_iter_projection;
  }
}

template<typename Number>
void
TimeIntBDFCoupled<Number>::get_wall_times(std::vector<std::string> & name,
                                          std::vector<double> &      wall_time) const
{
  name.resize(2);
  std::vector<std::string> names = {"Coupled system", "Projection"};
  name                           = names;

  wall_time.resize(2);
  for(unsigned int i = 0; i < this->computing_times.size(); ++i)
  {
    wall_time[i] = this->computing_times[i];
  }
}

// instantiations

template class TimeIntBDFCoupled<float>;
template class TimeIntBDFCoupled<double>;
} // namespace IncNS
