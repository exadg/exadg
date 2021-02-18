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

#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_coupled.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h>
#include <exadg/incompressible_navier_stokes/user_interface/input_parameters.h>
#include <exadg/time_integration/push_back_vectors.h>
#include <exadg/time_integration/time_step_calculation.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim, typename Number>
TimeIntBDFCoupled<dim, Number>::TimeIntBDFCoupled(
  std::shared_ptr<Operator>                       operator_in,
  InputParameters const &                         param_in,
  unsigned int const                              refine_steps_time_in,
  MPI_Comm const &                                mpi_comm_in,
  bool const                                      print_wall_times_in,
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in,
  std::shared_ptr<MovingMeshBase<dim, Number>>    moving_mesh_in,
  std::shared_ptr<MatrixFree<dim, Number>>        matrix_free_in)
  : Base(operator_in,
         param_in,
         refine_steps_time_in,
         mpi_comm_in,
         print_wall_times_in,
         postprocessor_in,
         moving_mesh_in,
         matrix_free_in),
    pde_operator(operator_in),
    solution(this->order),
    iterations({0, {0, 0}}),
    iterations_penalty({0, 0}),
    scaling_factor_continuity(1.0),
    characteristic_element_length(1.0)
{
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::allocate_vectors()
{
  TimeIntBDF<dim, Number>::allocate_vectors();

  // solution
  for(unsigned int i = 0; i < solution.size(); ++i)
    pde_operator->initialize_block_vector_velocity_pressure(solution[i]);
  pde_operator->initialize_block_vector_velocity_pressure(solution_np);
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::initialize_current_solution()
{
  if(this->param.ale_formulation)
    this->move_mesh(this->get_time());

  pde_operator->prescribe_initial_conditions(solution[0].block(0),
                                             solution[0].block(1),
                                             this->get_time());
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::initialize_former_solutions()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i = 1; i < solution.size(); ++i)
  {
    if(this->param.ale_formulation)
      this->move_mesh(this->get_previous_time(i));

    pde_operator->prescribe_initial_conditions(solution[i].block(0),
                                               solution[i].block(1),
                                               this->get_previous_time(i));
  }
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::setup_derived()
{
  Base::setup_derived();

  // scaling factor continuity equation:
  // Calculate characteristic element length h
  double minimum_element_length = pde_operator->calculate_minimum_element_length();

  unsigned int const degree_u = pde_operator->get_polynomial_degree();

  characteristic_element_length =
    calculate_characteristic_element_length(minimum_element_length, degree_u);
}

template<int dim, typename Number>
typename TimeIntBDFCoupled<dim, Number>::VectorType const &
TimeIntBDFCoupled<dim, Number>::get_velocity() const
{
  return solution[0].block(0);
}

template<int dim, typename Number>
typename TimeIntBDFCoupled<dim, Number>::VectorType const &
TimeIntBDFCoupled<dim, Number>::get_velocity(unsigned int i) const
{
  return solution[i].block(0);
}

template<int dim, typename Number>
typename TimeIntBDFCoupled<dim, Number>::VectorType const &
TimeIntBDFCoupled<dim, Number>::get_velocity_np() const
{
  return solution_np.block(0);
}

template<int dim, typename Number>
typename TimeIntBDFCoupled<dim, Number>::VectorType const &
TimeIntBDFCoupled<dim, Number>::get_pressure_np() const
{
  return solution_np.block(1);
}

template<int dim, typename Number>
typename TimeIntBDFCoupled<dim, Number>::VectorType const &
TimeIntBDFCoupled<dim, Number>::get_pressure(unsigned int i) const
{
  return solution[i].block(1);
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::set_velocity(VectorType const & velocity_in, unsigned int const i)
{
  solution[i].block(0) = velocity_in;
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::set_pressure(VectorType const & pressure_in, unsigned int const i)
{
  solution[i].block(1) = pressure_in;
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::solve_timestep()
{
  Timer timer;
  timer.restart();

  // extrapolate old solutions to obtain a good initial guess for the solver, or
  // to update the turbulence model or the penalty parameters based on this
  // extrapolated solution
  if(this->use_extrapolation)
  {
    solution_np.equ(this->extra.get_beta(0), solution[0]);
    for(unsigned int i = 1; i < solution.size(); ++i)
      solution_np.add(this->extra.get_beta(i), solution[i]);
  }
  else if(this->param.apply_penalty_terms_in_postprocessing_step == true)
  {
    solution_np = solution_last_iter;
  }

  // update of turbulence model
  if(this->param.use_turbulence_model == true)
  {
    Timer timer_turbulence;
    timer_turbulence.restart();

    pde_operator->update_turbulence_model(solution_np.block(0));

    if(this->print_solver_info())
    {
      this->pcout << std::endl << "Update of turbulent viscosity:";
      if(this->print_wall_times)
        print_wall_time(this->pcout, timer_turbulence.wall_time());
    }
  }

  // Update divergence and continuity penalty operator in case
  // that these terms are added to the monolithic system of equations.
  if(this->param.apply_penalty_terms_in_postprocessing_step == false)
  {
    if(this->param.use_divergence_penalty == true)
    {
      pde_operator->update_divergence_penalty_operator(solution_np.block(0));
    }
    if(this->param.use_continuity_penalty == true)
    {
      pde_operator->update_continuity_penalty_operator(solution_np.block(0));
    }
  }

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

  // calculate auxiliary variable p^{*} = 1/scaling_factor * p
  solution_np.block(1) *= 1.0 / scaling_factor_continuity;

  bool const update_preconditioner =
    this->param.update_preconditioner_coupled &&
    ((this->time_step_number - 1) % this->param.update_preconditioner_coupled_every_time_steps ==
     0);

  if(this->param.linear_problem_has_to_be_solved())
  {
    BlockVectorType rhs_vector;
    pde_operator->initialize_block_vector_velocity_pressure(rhs_vector);

    // calculate rhs vector for the Stokes problem, i.e., the convective term is neglected in this
    // step
    pde_operator->rhs_stokes_problem(rhs_vector, this->get_next_time());

    // Add the convective term to the right-hand side of the equations
    // if the convective term is treated explicitly (additive decomposition):
    // evaluate convective term and add extrapolation of convective term to the rhs (-> minus sign!)
    if(this->param.convective_problem() &&
       this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
    {
      if(this->param.ale_formulation)
      {
        // evaluate convective term for all previous times since the mesh has been updated
        for(unsigned int i = 0; i < this->vec_convective_term.size(); ++i)
        {
          // in a general setting, we only know the boundary conditions at time t_{n+1}
          pde_operator->evaluate_convective_term(this->vec_convective_term[i],
                                                 solution[i].block(0),
                                                 this->get_next_time());
        }
      }

      for(unsigned int i = 0; i < this->vec_convective_term.size(); ++i)
        rhs_vector.block(0).add(-this->extra.get_beta(i), this->vec_convective_term[i]);
    }

    VectorType sum_alphai_ui(solution[0].block(0));

    // calculate sum (alpha_i/dt * u_tilde_i) in case of explicit treatment of convective term
    // and operator-integration-factor (OIF) splitting
    if(this->param.convective_problem() &&
       this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
    {
      this->calculate_sum_alphai_ui_oif_substepping(sum_alphai_ui, this->cfl, this->cfl_oif);
    }
    // calculate sum (alpha_i/dt * u_i) for standard BDF discretization
    else
    {
      sum_alphai_ui.equ(this->bdf.get_alpha(0) / this->get_time_step_size(), solution[0].block(0));
      for(unsigned int i = 1; i < solution.size(); ++i)
      {
        sum_alphai_ui.add(this->bdf.get_alpha(i) / this->get_time_step_size(),
                          solution[i].block(0));
      }
    }

    // apply mass operator to sum_alphai_ui and add to rhs vector
    pde_operator->apply_mass_operator_add(rhs_vector.block(0), sum_alphai_ui);

    unsigned int const n_iter =
      pde_operator->solve_linear_stokes_problem(solution_np,
                                                rhs_vector,
                                                update_preconditioner,
                                                this->get_next_time(),
                                                this->get_scaling_factor_time_derivative_term());

    iterations.first += 1;
    std::get<1>(iterations.second) += n_iter;

    // write output
    if(this->print_solver_info())
    {
      this->pcout << std::endl << "Solve linear problem:";
      print_solver_info_linear(this->pcout, n_iter, timer.wall_time(), this->print_wall_times);
    }
  }
  else // a nonlinear system of equations has to be solved
  {
    VectorType sum_alphai_ui(solution[0].block(0));

    // calculate Sum_i (alpha_i/dt * u_i)
    sum_alphai_ui.equ(this->bdf.get_alpha(0) / this->get_time_step_size(), solution[0].block(0));
    for(unsigned int i = 1; i < solution.size(); ++i)
    {
      sum_alphai_ui.add(this->bdf.get_alpha(i) / this->get_time_step_size(), solution[i].block(0));
    }

    VectorType rhs(sum_alphai_ui);
    pde_operator->apply_mass_operator(rhs, sum_alphai_ui);
    if(this->param.right_hand_side)
      pde_operator->evaluate_add_body_force_term(rhs, this->get_next_time());

    // Newton solver
    auto const iter =
      pde_operator->solve_nonlinear_problem(solution_np,
                                            rhs,
                                            update_preconditioner,
                                            this->get_next_time(),
                                            this->get_scaling_factor_time_derivative_term());

    iterations.first += 1;
    std::get<0>(iterations.second) += std::get<0>(iter);
    std::get<1>(iterations.second) += std::get<1>(iter);

    // write output
    if(this->print_solver_info())
    {
      this->pcout << std::endl << "Solve nonlinear problem:";
      print_solver_info_nonlinear(this->pcout,
                                  std::get<0>(iter),
                                  std::get<1>(iter),
                                  timer.wall_time(),
                                  this->print_wall_times);
    }
  }

  // reconstruct pressure solution p from auxiliary variable p^{*}: p = scaling_factor * p^{*}
  solution_np.block(1) *= scaling_factor_continuity;

  // special case: pressure level is undefined
  // Adjust the pressure level in order to allow a calculation of the pressure error.
  // This is necessary because otherwise the pressure solution moves away from the exact solution.
  pde_operator->adjust_pressure_level_if_undefined(solution_np.block(1), this->get_next_time());

  if(this->store_solution && this->param.apply_penalty_terms_in_postprocessing_step == true)
  {
    solution_last_iter = solution_np;
  }

  this->timer_tree->insert({"Timeloop", "Coupled system"}, timer.wall_time());

  // If the penalty terms are applied in a postprocessing step
  if(this->param.apply_penalty_terms_in_postprocessing_step == true)
  {
    if(this->param.use_divergence_penalty == true || this->param.use_continuity_penalty == true)
    {
      timer.restart();

      penalty_step();

      this->timer_tree->insert({"Timeloop", "Penalty step"}, timer.wall_time());
    }
  }

  timer.restart();

  // evaluate convective term once solution_np is known
  if(this->param.convective_problem() &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    if(this->param.ale_formulation == false) // Eulerian case
    {
      pde_operator->evaluate_convective_term(this->convective_term_np,
                                             solution_np.block(0),
                                             this->get_next_time());
    }
  }

  this->timer_tree->insert({"Timeloop", "Coupled system"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::penalty_step()
{
  Timer timer;
  timer.restart();

  // right-hand side term: apply mass operator
  VectorType rhs(solution_np.block(0));
  pde_operator->apply_mass_operator(rhs, solution_np.block(0));

  // extrapolate velocity to time t_n+1 and use this velocity field to
  // calculate the penalty parameter for the divergence and continuity penalty term
  VectorType velocity_extrapolated(solution_np.block(0));
  if(this->use_extrapolation)
  {
    velocity_extrapolated = 0.0;
    for(unsigned int i = 0; i < solution.size(); ++i)
      velocity_extrapolated.add(this->extra.get_beta(i), solution[i].block(0));
  }
  else
  {
    velocity_extrapolated = velocity_penalty_last_iter;
  }

  // update projection operator
  pde_operator->update_projection_operator(velocity_extrapolated, this->get_time_step_size());

  // right-hand side term: add inhomogeneous contributions of continuity penalty operator to
  // rhs-vector if desired
  if(this->param.use_continuity_penalty && this->param.continuity_penalty_use_boundary_data)
    pde_operator->rhs_add_projection_operator(rhs, this->get_next_time());

  bool const update_preconditioner =
    this->param.update_preconditioner_projection &&
    ((this->time_step_number - 1) % this->param.update_preconditioner_projection_every_time_steps ==
     0);

  // solve projection step
  if(this->use_extrapolation == false)
    solution_np.block(0) = velocity_penalty_last_iter;

  unsigned int n_iter =
    pde_operator->solve_projection(solution_np.block(0), rhs, update_preconditioner);

  if(this->store_solution)
    velocity_penalty_last_iter = solution_np.block(0);

  iterations_penalty.first += 1;
  iterations_penalty.second += n_iter;

  // write output
  if(this->print_solver_info())
  {
    this->pcout << std::endl << "Solve penalty step:";
    print_solver_info_linear(this->pcout, n_iter, timer.wall_time(), this->print_wall_times);
  }
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::postprocessing_stability_analysis()
{
  AssertThrow(this->order == 1,
              ExcMessage("Order of BDF scheme has to be 1 for this stability analysis"));

  AssertThrow(this->param.convective_problem() == false,
              ExcMessage(
                "Stability analysis can not be performed for nonlinear convective problems."));

  AssertThrow(solution[0].block(0).l2_norm() < 1.e-15 && solution[0].block(1).l2_norm() < 1.e-15,
              ExcMessage("Solution vector has to be zero for this stability analysis."));

  AssertThrow(Utilities::MPI::n_mpi_processes(this->mpi_comm) == 1,
              ExcMessage("Number of MPI processes has to be 1."));

  std::cout << std::endl << "Analysis of eigenvalue spectrum:" << std::endl;

  unsigned int const size = solution[0].block(0).local_size();

  LAPACKFullMatrix<Number> propagation_matrix(size, size);

  // loop over all columns of propagation matrix
  for(unsigned int j = 0; j < size; ++j)
  {
    // set j-th element to 1
    solution[0].block(0).local_element(j) = 1.0;

    // solve time step
    this->solve_timestep();

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

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::prepare_vectors_for_next_timestep()
{
  Base::prepare_vectors_for_next_timestep();

  push_back(solution);
  solution[0].swap(solution_np);
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::solve_steady_problem()
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
      this->do_timestep();

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
      this->do_timestep();

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

template<int dim, typename Number>
double
TimeIntBDFCoupled<dim, Number>::evaluate_residual()
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

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::print_iterations() const
{
  std::vector<std::string> names;
  std::vector<double>      iterations_avg;

  if(this->param.linear_problem_has_to_be_solved())
  {
    names = {"Coupled system"};
    iterations_avg.resize(1);
    iterations_avg[0] =
      (double)std::get<1>(iterations.second) / std::max(1., (double)iterations.first);
  }
  else // nonlinear system of equations in momentum step
  {
    names = {"Coupled system (nonlinear)",
             "Coupled system (linear accumulated)",
             "Coupled system (linear per nonlinear)"};

    iterations_avg.resize(3);
    iterations_avg[0] =
      (double)std::get<0>(iterations.second) / std::max(1., (double)iterations.first);
    iterations_avg[1] =
      (double)std::get<1>(iterations.second) / std::max(1., (double)iterations.first);
    if(iterations_avg[0] > std::numeric_limits<double>::min())
      iterations_avg[2] = iterations_avg[1] / iterations_avg[0];
    else
      iterations_avg[2] = iterations_avg[1];
  }

  if(this->param.apply_penalty_terms_in_postprocessing_step)
  {
    names.push_back("Penalty terms");
    iterations_avg.push_back(iterations_penalty.second /
                             std::max(1., (double)iterations_penalty.first));
  }

  print_list_of_iterations(this->pcout, names, iterations_avg);
}

// instantiations

template class TimeIntBDFCoupled<2, float>;
template class TimeIntBDFCoupled<2, double>;

template class TimeIntBDFCoupled<3, float>;
template class TimeIntBDFCoupled<3, double>;

} // namespace IncNS
} // namespace ExaDG
