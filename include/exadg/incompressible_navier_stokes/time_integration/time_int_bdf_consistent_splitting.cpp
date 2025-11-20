/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by the ExaDG authors
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

#include <deal.II/numerics/vector_tools_mean_value.h>

#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_consistent_splitting.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_consistent_splitting.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>
#include <exadg/time_integration/restart.h>
#include <exadg/time_integration/time_step_calculation.h>
#include <exadg/time_integration/vector_handling.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
TimeIntBDFConsistentSplitting<dim, Number>::TimeIntBDFConsistentSplitting(
  std::shared_ptr<Operator>                       operator_in,
  std::shared_ptr<HelpersALE<dim, Number> const>  helpers_ale_in,
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in,
  Parameters const &                              param_in,
  MPI_Comm const &                                mpi_comm_in,
  bool const                                      is_test_in)
  : Base(operator_in, helpers_ale_in, postprocessor_in, param_in, mpi_comm_in, is_test_in),
    pde_operator(operator_in),
    velocity(this->order),
    pressure(this->order),
    velocity_divergence(this->order),
    vec_convective_term_div(this->order),
    iterations_pressure({0, 0}),
    iterations_projection({0, 0}),
    iterations_viscous({0, {0, 0}}),
    iterations_penalty({0, 0}),
    iterations_mass({0, 0}),
    extra_pressure_nbc(this->param.order_extrapolation_pressure_nbc,
                       this->param.start_with_low_order),
    extra_pressure_rhs(this->param.order_extrapolation_pressure_rhs,
                       this->param.start_with_low_order)
{
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplitting<dim, Number>::update_time_integrator_constants()
{
  // call function of base class to update the standard time integrator constants
  Base::update_time_integrator_constants();

  // update time integrator constants for extrapolation scheme of pressure Neumann bc
  extra_pressure_nbc.update(this->get_time_step_number(),
                            this->adaptive_time_stepping,
                            this->get_time_step_vector());

  extra_pressure_rhs.update(this->get_time_step_number(),
                            this->adaptive_time_stepping,
                            this->get_time_step_vector());

  // use this function to check the correctness of the time integrator constants
  //    std::cout << "Coefficients extrapolation scheme pressure NBC:" << std::endl;
  //    extra_pressure_nbc.print(this->pcout);
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplitting<dim, Number>::setup_derived()
{
  Base::setup_derived();
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplitting<dim, Number>::allocate_vectors()
{
  Base::allocate_vectors();

  // velocity
  pde_operator->initialize_vector_velocity(velocity_np);
  for(unsigned int i = 0; i < velocity.size(); ++i)
    pde_operator->initialize_vector_velocity(velocity[i]);

  // pressure
  pde_operator->initialize_vector_pressure(pressure_np);
  for(unsigned int i = 0; i < pressure.size(); ++i)
    pde_operator->initialize_vector_pressure(pressure[i]);

  // velocity divergence
  for(unsigned int i = 0; i < velocity_divergence.size(); ++i)
    pde_operator->initialize_vector_pressure(velocity_divergence[i]);

  // convective term divergence
  for(unsigned int i = 0; i < vec_convective_term_div.size(); ++i)
    pde_operator->initialize_vector_pressure(vec_convective_term_div[i]);
}


template<int dim, typename Number>
void
TimeIntBDFConsistentSplitting<dim, Number>::initialize_current_solution()
{
  pde_operator->prescribe_initial_conditions(velocity[0], pressure[0], this->get_time());

  // Now compute divergence of velocity and convective term
  pde_operator->apply_velocity_divergence_term(velocity_divergence[0], velocity[0]);

  pde_operator->apply_convective_divergence_term(vec_convective_term_div[0], velocity[0]);
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplitting<dim, Number>::initialize_former_multistep_dof_vectors()
{
  // Note that the loop begins with i=1.
  for(unsigned int i = 1; i < velocity.size(); ++i)
  {
    pde_operator->prescribe_initial_conditions(velocity[i],
                                               pressure[i],
                                               this->get_previous_time(i));

    pde_operator->apply_velocity_divergence_term(velocity_divergence[i], velocity[i]);

    // We need to compute this
    pde_operator->apply_convective_divergence_term(vec_convective_term_div[i], velocity[i]);
  }
}

template<int dim, typename Number>
typename TimeIntBDFConsistentSplitting<dim, Number>::VectorType const &
TimeIntBDFConsistentSplitting<dim, Number>::get_velocity() const
{
  return velocity[0];
}

template<int dim, typename Number>
typename TimeIntBDFConsistentSplitting<dim, Number>::VectorType const &
TimeIntBDFConsistentSplitting<dim, Number>::get_velocity_np() const
{
  return velocity_np;
}

template<int dim, typename Number>
typename TimeIntBDFConsistentSplitting<dim, Number>::VectorType const &
TimeIntBDFConsistentSplitting<dim, Number>::get_pressure() const
{
  return pressure[0];
}

template<int dim, typename Number>
typename TimeIntBDFConsistentSplitting<dim, Number>::VectorType const &
TimeIntBDFConsistentSplitting<dim, Number>::get_pressure_np() const
{
  return pressure_np;
}

template<int dim, typename Number>
typename TimeIntBDFConsistentSplitting<dim, Number>::VectorType const &
TimeIntBDFConsistentSplitting<dim, Number>::get_velocity(unsigned int i) const
{
  return velocity[i];
}

template<int dim, typename Number>
typename TimeIntBDFConsistentSplitting<dim, Number>::VectorType const &
TimeIntBDFConsistentSplitting<dim, Number>::get_pressure(unsigned int i) const
{
  return pressure[i];
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplitting<dim, Number>::set_velocity(VectorType const & velocity_in,
                                                         unsigned int const i)
{
  velocity[i] = velocity_in;
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplitting<dim, Number>::set_pressure(VectorType const & pressure_in,
                                                         unsigned int const i)
{
  pressure[i] = pressure_in;
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplitting<dim, Number>::postprocessing_stability_analysis()
{
  AssertThrow(this->order == 1,
              dealii::ExcMessage("Order of BDF scheme has to be 1 for this stability analysis."));

  AssertThrow(this->param.convective_problem() == false,
              dealii::ExcMessage(
                "Stability analysis can not be performed for nonlinear convective problems."));

  AssertThrow(velocity[0].l2_norm() < 1.e-15 and pressure[0].l2_norm() < 1.e-15,
              dealii::ExcMessage("Solution vector has to be zero for this stability analysis."));

  AssertThrow(dealii::Utilities::MPI::n_mpi_processes(this->mpi_comm) == 1,
              dealii::ExcMessage("Number of MPI processes has to be 1."));

  std::cout << std::endl << "Analysis of eigenvalue spectrum:" << std::endl;

  unsigned int const size = velocity[0].locally_owned_size();

  dealii::LAPACKFullMatrix<Number> propagation_matrix(size, size);

  // loop over all columns of propagation matrix
  for(unsigned int j = 0; j < size; ++j)
  {
    // set j-th element to 1
    velocity[0].local_element(j) = 1.0;

    // solve time step
    this->do_timestep_solve();

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

template<int dim, typename Number>
void
TimeIntBDFConsistentSplitting<dim, Number>::do_timestep_solve()
{
  pressure_step();

  momentum_step();

  if(this->param.apply_penalty_terms_in_postprocessing_step)
    penalty_step();

  // evaluate convective term once the final solution at time
  // t_{n+1} is known in the explicit case
  if(this->param.convective_problem() and
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
    evaluate_convective_term();
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplitting<dim, Number>::evaluate_convective_term()
{
  dealii::Timer timer;
  timer.restart();

  pde_operator->evaluate_convective_term(this->convective_term_np,
                                         velocity_np,
                                         this->get_next_time());

  this->timer_tree->insert({"Timeloop", "Convective step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplitting<dim, Number>::pressure_step()
{
  dealii::Timer timer;
  timer.restart();

  // compute right-hand-side vector
  VectorType rhs;
  rhs.reinit(pressure_np, true);
  rhs_pressure(rhs);

  // extrapolate old solution to get a good initial estimate for the solver
  if(this->use_extrapolation)
  {
    pressure_np.equ(this->extra.get_beta(0), pressure[0]);
    for(unsigned int i = 1; i < pressure.size(); ++i)
    {
      pressure_np.add(this->extra.get_beta(i), pressure[i]);
    }
  }
  else
  {
    pressure_np = 0;
  }

  // solve linear system of equations
  bool const update_preconditioner =
    this->param.update_preconditioner_pressure_poisson and
    ((this->time_step_number - 1) %
       this->param.update_preconditioner_pressure_poisson_every_time_steps ==
     0);

  unsigned int const n_iter =
    pde_operator->do_solve_pressure(pressure_np, rhs, update_preconditioner);

  iterations_pressure.first += 1;
  iterations_pressure.second += n_iter;

  // special case: pressure level is undefined
  // Adjust the pressure level in order to allow a calculation of the pressure error.
  // This is necessary because otherwise the pressure solution moves away from the exact solution.
  pde_operator->adjust_pressure_level_if_undefined(pressure_np, this->get_next_time());

  // write output
  if(this->print_solver_info() and not(this->is_test))
  {
    this->pcout << std::endl << "Solve pressure step:";
    print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
  }

  this->timer_tree->insert({"Timeloop", "Pressure step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplitting<dim, Number>::rhs_pressure(VectorType & rhs) const
{
  /*
   *  I. convective extrapolation
   */
  rhs.equ(extra_pressure_rhs.get_beta(0), this->vec_convective_term_div[0]);
  for(unsigned int i = 1; i < extra_pressure_rhs.get_order(); ++i)
    rhs.add(extra_pressure_rhs.get_beta(i), this->vec_convective_term_div[i]);

  /*
   *  II. forcing term
   */
  if(this->param.right_hand_side)
    pde_operator->rhs_ppe_div_term_body_forces_add(rhs, this->get_next_time());

  /*
   *  III. calculate Leray projection
   */
  if(this->param.apply_leray_projection)
    for(unsigned int i = 0; i < velocity_divergence.size(); ++i)
      rhs.add(-this->bdf.get_alpha(i) / this->get_time_step_size(), velocity_divergence[i]);

  /*
   *  IV. handle consistent boundary condition
   */
  /*
   *  IV.1 extrapolate speed and compute curl-curl term and time derivative term
   */
  VectorType velocity_extra;
  velocity_extra.reinit(velocity_np, true);
  velocity_extra.equ(this->extra_pressure_nbc.get_beta(0), velocity[0]);
  for(unsigned int i = 1; i < extra_pressure_nbc.get_order(); ++i)
  {
    velocity_extra.add(this->extra_pressure_nbc.get_beta(i), velocity[i]);
  }

  VectorType vorticity;
  vorticity.reinit(velocity_extra);
  pde_operator->compute_vorticity(vorticity, velocity_extra);

  // Now actually add the curl-curl term and the time derivative to the rhs
  pde_operator->rhs_ppe_nbc_add(rhs,
                                vorticity,
                                this->get_next_time(),
                                this->bdf.get_gamma0() / this->get_time_step_size());
  // If Leray projection is not applied then the contributions from the old time derivative do not
  // cancel, so they have to be added
  if(!this->param.apply_leray_projection)
  {
    // Set curl to 0 as it was already added
    vorticity = 0.0;
    for(unsigned int i = 0; i < this->bdf.get_order(); ++i)
    {
      pde_operator->rhs_ppe_nbc_add(rhs,
                                    vorticity,
                                    this->get_previous_time(i),
                                    -this->bdf.get_alpha(i) / this->get_time_step_size());
    }
  }

  // IV.2. pressure Dirichlet boundary conditions
  pde_operator->do_rhs_ppe_laplace_add(rhs, this->get_next_time());


  // special case: pressure level is undefined
  // Set mean value of rhs to zero in order to obtain a consistent linear system of equations.
  // This is really necessary for the Consistent-splitting scheme in contrast to the
  // pressure-correction scheme and coupled solution approach due to the Dirichlet BC prescribed for
  // the intermediate velocity field and the pressure Neumann BC in case of the Consistent-splitting
  // scheme.
  if(pde_operator->is_pressure_level_undefined())
    dealii::VectorTools::subtract_mean_value(rhs);
}


template<int dim, typename Number>
void
TimeIntBDFConsistentSplitting<dim, Number>::momentum_step()
{
  dealii::Timer timer;
  timer.restart();

  // in case we need to iteratively solve a linear or nonlinear system of equations
  if(this->param.viscous_problem() or this->param.non_explicit_convective_problem())
  {
    // Extrapolate old solutions to get a good initial estimate for the solver.
    velocity_np.equ(this->extra.get_beta(0), velocity[0]);
    for(unsigned int i = 1; i < velocity.size(); ++i)
    {
      velocity_np.add(this->extra.get_beta(i), velocity[i]);
    }

    bool const update_preconditioner =
      this->param.update_preconditioner_momentum and
      ((this->time_step_number - 1) % this->param.update_preconditioner_momentum_every_time_steps ==
       0);

    if(this->param.nonlinear_problem_has_to_be_solved())
    {
      /*
       *  Calculate the vector that is constant when solving the nonlinear momentum equation
       *  (where constant means that the vector does not change from one Newton iteration
       *  to the next, i.e., it does not depend on the current solution of the nonlinear solver)
       */
      VectorType rhs;
      velocity_np.reinit(velocity_np, true);
      VectorType transport_velocity_dummy;
      rhs_momentum(rhs, transport_velocity_dummy);

      // solve non-linear system of equations
      auto const iter = pde_operator->solve_nonlinear_momentum_equation(
        velocity_np,
        rhs,
        this->get_next_time(),
        update_preconditioner,
        this->get_scaling_factor_time_derivative_term());

      iterations_viscous.first += 1;
      std::get<0>(iterations_viscous.second) += std::get<0>(iter);
      std::get<1>(iterations_viscous.second) += std::get<1>(iter);

      if(this->print_solver_info() and not(this->is_test))
      {
        this->pcout << std::endl << "Solve momentum step:";
        print_solver_info_nonlinear(this->pcout,
                                    std::get<0>(iter),
                                    std::get<1>(iter),
                                    timer.wall_time());
      }
    }
    else // linear problem
    {
      // linearly implicit convective term: use extrapolated/stored velocity as transport velocity
      VectorType transport_velocity;
      if(this->param.convective_problem() and
         this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::LinearlyImplicit)
      {
        transport_velocity = velocity_np;
      }

      /*
       *  Calculate the right-hand side of the linear system of equations.
       */
      VectorType rhs;
      rhs.reinit(velocity_np, true);
      rhs_momentum(rhs, transport_velocity);

      // solve linear system of equations
      unsigned int n_iter = pde_operator->solve_linear_momentum_equation(
        velocity_np,
        rhs,
        transport_velocity,
        update_preconditioner,
        this->get_scaling_factor_time_derivative_term());

      iterations_viscous.first += 1;
      std::get<1>(iterations_viscous.second) += n_iter;

      if(this->print_solver_info() and not(this->is_test))
      {
        this->pcout << std::endl << "Solve momentum step:";
        print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
      }
    }
  }
  else // no viscous term and no (linearly) implicit convective term, i.e. we only need to invert
       // the mass matrix
  {
    /*
     *  Calculate the right-hand side vector.
     */
    VectorType rhs;
    rhs.reinit(velocity_np, true);
    VectorType transport_velocity_dummy;
    rhs_momentum(rhs, transport_velocity_dummy);

    pde_operator->apply_inverse_mass_operator(velocity_np, rhs);
    velocity_np *= this->get_time_step_size() / this->bdf.get_gamma0();

    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl << "Explicit momentum step:";
      print_wall_time(this->pcout, timer.wall_time());
    }
  }

  this->timer_tree->insert({"Timeloop", "Momentum step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplitting<dim, Number>::rhs_momentum(
  VectorType &       rhs,
  VectorType const & transport_velocity) const
{
  /*
   *  Pressure gradient term
   */
  pde_operator->evaluate_pressure_gradient_term(rhs, pressure_np, this->get_next_time());

  rhs *= -1.0;

  /*
   *  Body force term
   */
  if(this->param.right_hand_side == true)
  {
    pde_operator->evaluate_add_body_force_term(rhs, this->get_next_time());
  }

  /*
   *  Convective term formulated explicitly (additive decomposition):
   *  Evaluate convective term and add extrapolation of convective term to the rhs
   */
  if(this->param.convective_problem())
  {
    if(this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
    {
      for(unsigned int i = 0; i < this->vec_convective_term.size(); ++i)
        rhs.add(-this->extra.get_beta(i), this->vec_convective_term[i]);
    }

    if(this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::LinearlyImplicit)
    {
      pde_operator->rhs_add_convective_term(rhs, transport_velocity, this->get_next_time());
    }
  }

  /*
   *  calculate sum (alpha_i/dt * u_i) and apply mass operator to this vector
   */
  VectorType sum_alphai_ui(velocity[0]);

  // calculate sum (alpha_i/dt * u_i)
  sum_alphai_ui.equ(this->bdf.get_alpha(0) / this->get_time_step_size(), velocity[0]);
  for(unsigned int i = 1; i < velocity.size(); ++i)
  {
    sum_alphai_ui.add(this->bdf.get_alpha(i) / this->get_time_step_size(), velocity[i]);
  }

  pde_operator->apply_mass_operator_add(rhs, sum_alphai_ui);

  /*
   *  Right-hand side viscous term:
   *  If there is no nonlinearity, we solve a linear system of equations, where
   *  inhomogeneous parts of boundary face integrals of the viscous operator
   *  have to be shifted to the right-hand side of the equation.
   */
  if(this->param.viscous_problem() and not(this->param.nonlinear_problem_has_to_be_solved()))
  {
    pde_operator->rhs_add_viscous_term(rhs, this->get_next_time());
  }
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplitting<dim, Number>::penalty_step()
{
  if(this->param.use_divergence_penalty == true or this->param.use_continuity_penalty == true)
  {
    dealii::Timer timer;
    timer.restart();

    // compute right-hand-side vector
    VectorType rhs;
    rhs.reinit(velocity_np, true /* omit_zeroing_entries */);
    pde_operator->apply_mass_operator(rhs, velocity_np);

    // extrapolate velocity to time t_n+1 and use this velocity field to
    // calculate the penalty parameter for the divergence and continuity penalty term
    VectorType velocity_extrapolated;
    velocity_extrapolated.reinit(velocity_np, true /* omit_zeroing_entries */);
    velocity_extrapolated.equ(this->extra.get_beta(0), velocity[0]);
    for(unsigned int i = 1; i < velocity.size(); ++i)
      velocity_extrapolated.add(this->extra.get_beta(i), velocity[i]);

    pde_operator->update_projection_operator(velocity_extrapolated, this->get_time_step_size());

    // right-hand side term: add inhomogeneous contributions of continuity penalty operator to
    // rhs-vector if desired
    if(this->param.use_continuity_penalty and this->param.continuity_penalty_use_boundary_data)
      pde_operator->rhs_add_projection_operator(rhs, this->get_next_time());

    // solve linear system of equations
    bool const update_preconditioner =
      this->param.update_preconditioner_projection and
      ((this->time_step_number - 1) %
         this->param.update_preconditioner_projection_every_time_steps ==
       0);

    if(this->use_extrapolation == false)
      velocity_np = 0;

    unsigned int const n_iter =
      pde_operator->solve_projection(velocity_np, rhs, update_preconditioner);

    iterations_penalty.first += 1;
    iterations_penalty.second += n_iter;

    // write output
    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl << "Solve penalty step:";
      print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
    }

    this->timer_tree->insert({"Timeloop", "Penalty step"}, timer.wall_time());
  }
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplitting<dim, Number>::prepare_vectors_for_next_timestep()
{
  Base::prepare_vectors_for_next_timestep();

  swap_back_one_step(velocity);
  velocity[0].swap(velocity_np);

  swap_back_one_step(pressure);
  pressure[0].swap(pressure_np);

  // Compute the divergence of the velocity for the next timestep
  if(this->param.apply_leray_projection)
  {
    swap_back_one_step(velocity_divergence);
    pde_operator->apply_velocity_divergence_term(velocity_divergence[0], velocity[0]);
  }

  // Compute divergence of convective term
  swap_back_one_step(vec_convective_term_div);
  pde_operator->apply_convective_divergence_term(vec_convective_term_div[0], velocity[0]);
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplitting<dim, Number>::solve_steady_problem()
{
  this->pcout << std::endl << "Starting time loop ..." << std::endl;

  // pseudo-time integration in order to solve steady-state problem
  bool converged = false;

  if(this->param.convergence_criterion_steady_problem ==
     ConvergenceCriterionSteadyProblem::SolutionIncrement)
  {
    VectorType velocity_tmp;
    VectorType pressure_tmp;

    while(not(converged) and this->time < (this->end_time - this->eps) and
          this->get_time_step_number() <= this->param.max_number_of_time_steps)
    {
      // save solution from previous time step
      velocity_tmp = this->velocity[0];
      pressure_tmp = this->pressure[0];

      // calculate norm of solution
      double const norm_u = velocity_tmp.l2_norm();
      double const norm_p = pressure_tmp.l2_norm();
      double const norm   = std::sqrt(norm_u * norm_u + norm_p * norm_p);

      // solve time step
      this->do_timestep();

      // calculate increment:
      // increment = solution_{n+1} - solution_{n}
      //           = solution[0] - solution_tmp
      velocity_tmp *= -1.0;
      pressure_tmp *= -1.0;
      velocity_tmp.add(1.0, this->velocity[0]);
      pressure_tmp.add(1.0, this->pressure[0]);

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
      if(incr < this->param.abs_tol_steady or incr_rel < this->param.rel_tol_steady)
      {
        converged = true;
      }
    }
  }
  else if(this->param.convergence_criterion_steady_problem ==
          ConvergenceCriterionSteadyProblem::ResidualSteadyNavierStokes)
  {
    AssertThrow(this->param.convergence_criterion_steady_problem !=
                  ConvergenceCriterionSteadyProblem::ResidualSteadyNavierStokes,
                dealii::ExcMessage(
                  "This option is not available for the consistent splitting scheme. "
                  "Due to splitting errors the solution does not fulfill the "
                  "residual of the steady, incompressible Navier-Stokes equations."));
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("not implemented."));
  }

  AssertThrow(
    converged == true,
    dealii::ExcMessage(
      "Maximum number of time steps or end time exceeded! This might be due to the fact that "
      "(i) the maximum number of time steps is simply too small to reach a steady solution, "
      "(ii) the problem is unsteady so that the applied solution approach is inappropriate, "
      "(iii) some of the solver tolerances are in conflict."));

  this->pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplitting<dim, Number>::print_iterations() const
{
  std::vector<std::string> names;
  std::vector<double>      iterations_avg;

  if(this->param.nonlinear_problem_has_to_be_solved())
  {
    names = {"Pressure step",
             "Momentum step (nonlinear)",
             "Momentum step (accumulated)",
             "Momentum step (linear per nonlinear)"};

    iterations_avg.resize(4);
    iterations_avg[1] =
      (double)iterations_pressure.second / std::max(1., (double)iterations_pressure.first);
    iterations_avg[2] = (double)std::get<0>(iterations_viscous.second) /
                        std::max(1., (double)iterations_viscous.first);
    iterations_avg[3] = (double)std::get<1>(iterations_viscous.second) /
                        std::max(1., (double)iterations_viscous.first);

    if(iterations_avg[2] > std::numeric_limits<double>::min())
      iterations_avg[4] = iterations_avg[3] / iterations_avg[2];
    else
      iterations_avg[4] = iterations_avg[3];
  }
  else
  {
    names = {"Pressure step", "Momentum step"};

    iterations_avg.resize(2);
    iterations_avg[0] =
      (double)iterations_pressure.second / std::max(1., (double)iterations_pressure.first);
    iterations_avg[1] = (double)std::get<1>(iterations_viscous.second) /
                        std::max(1., (double)iterations_viscous.first);
  }

  if(this->param.spatial_discretization == SpatialDiscretization::HDIV)
  {
    names.push_back("Mass solver");
    iterations_avg.push_back(iterations_mass.second / std::max(1., (double)iterations_mass.first));
  }

  if(this->param.apply_penalty_terms_in_postprocessing_step)
  {
    names.push_back("Penalty step");
    iterations_avg.push_back((double)iterations_penalty.second /
                             std::max(1., (double)iterations_penalty.first));
  }

  print_list_of_iterations(this->pcout, names, iterations_avg);
}

// instantiations

template class TimeIntBDFConsistentSplitting<2, float>;
template class TimeIntBDFConsistentSplitting<2, double>;

template class TimeIntBDFConsistentSplitting<3, float>;
template class TimeIntBDFConsistentSplitting<3, double>;

} // namespace IncNS
} // namespace ExaDG
