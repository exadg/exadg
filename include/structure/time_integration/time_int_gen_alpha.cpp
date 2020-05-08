/*
 * time_int_gen_alpha.cpp
 *
 *  Created on: 20.04.2020
 *      Author: fehn
 */

#include "time_int_gen_alpha.h"

#include "../../utilities/print_throughput.h"
#include "../postprocessor/postprocessor_base.h"
#include "../spatial_discretization/interface.h"
#include "../user_interface/input_parameters.h"

namespace Structure
{
template<int dim, typename Number>
TimeIntGenAlpha<dim, Number>::TimeIntGenAlpha(
  std::shared_ptr<Interface::Operator<Number>> operator_,
  std::shared_ptr<PostProcessorBase<Number>>   postprocessor_,
  unsigned int const                           refine_steps_time_,
  InputParameters const &                      param_,
  MPI_Comm const &                             mpi_comm_)
  : TimeIntGenAlphaBase<Number>(param_.start_time,
                                param_.end_time,
                                param_.max_number_of_time_steps,
                                param_.spectral_radius,
                                param_.gen_alpha_type,
                                param_.restart_data,
                                mpi_comm_),
    pde_operator(operator_),
    postprocessor(postprocessor_),
    refine_steps_time(refine_steps_time_),
    param(param_),
    mpi_comm(mpi_comm_),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm_) == 0),
    iterations_linear(0),
    iterations_nonlinear(0)
{
}

template<int dim, typename Number>
void
TimeIntGenAlpha<dim, Number>::setup(bool const do_restart)
{
  this->pcout << std::endl << "Setup time integrator ..." << std::endl << std::endl << std::flush;

  // allocate vectors
  pde_operator->initialize_dof_vector(displacement_n);
  pde_operator->initialize_dof_vector(displacement_np);

  pde_operator->initialize_dof_vector(velocity_n);
  pde_operator->initialize_dof_vector(velocity_np);

  pde_operator->initialize_dof_vector(acceleration_n);
  pde_operator->initialize_dof_vector(acceleration_np);

  // initialize solution and time step size
  if(do_restart)
  {
    // The solution vectors, the time step size, etc. have to be read from
    // restart files.
    this->read_restart();
  }
  else
  {
    this->set_current_time_step_size(param.time_step_size / std::pow(2.0, refine_steps_time));

    pde_operator->prescribe_initial_displacement(displacement_n, this->get_time());
    pde_operator->prescribe_initial_velocity(velocity_n, this->get_time());
    // solve momentum equation to obtain initial acceleration
    pde_operator->compute_initial_acceleration(acceleration_n, displacement_n, this->get_time());
  }

  this->pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
TimeIntGenAlpha<dim, Number>::solve_timestep()
{
  if(this->print_solver_info())
    this->output_solver_info_header();

  // compute right-hand side in case of linear problems or "constant vector"
  // in case of nonlinear problems
  Timer timer;
  timer.restart();

  // compute const_vector
  VectorType const_vector, rhs;
  const_vector.reinit(displacement_n);
  rhs.reinit(displacement_n);
  this->compute_const_vector(rhs, displacement_n, velocity_n, acceleration_n);
  pde_operator->apply_mass_matrix(const_vector, rhs);
  // set entries of constant vector corresponding to Dirichlet degrees of freedom
  // to zero in order to allow convergence of solvers (especially for Newton solver,
  // linear solver should converge without this line because the linear operator
  // has values of 1 on the diagonal for constrained degrees of freedom).
  pde_operator->set_constrained_values_to_zero(const_vector);

  if(param.large_deformation == false) // linear case
  {
    // calculate right-hand side vector
    pde_operator->compute_rhs_linear(rhs, this->get_mid_time());
    // shift const_vector to right-hand side
    rhs.add(-1.0, const_vector);
  }

  this->timer_tree->insert({"Timeloop", "Compute rhs"}, timer.wall_time());

  // solve system of equations for displacement d_{n+1-alpha_f}
  timer.restart();

  // initial guess
  displacement_np = displacement_n;

  if(param.large_deformation) // nonlinear case
  {
    bool const update_preconditioner =
      this->param.update_preconditioner &&
      ((this->time_step_number - 1) % this->param.update_preconditioner_every_time_steps == 0);

    auto const iter = pde_operator->solve_nonlinear(displacement_np,
                                                    const_vector,
                                                    this->get_scaling_factor_mass(),
                                                    this->get_mid_time(),
                                                    update_preconditioner);

    iterations_nonlinear += std::get<0>(iter);
    iterations_linear += std::get<1>(iter);

    if(this->print_solver_info())
      print_solver_info_nonlinear(pcout, std::get<0>(iter), std::get<1>(iter), timer.wall_time());
  }
  else // linear case
  {
    // solve linear system of equations
    unsigned int const N_iter_linear = pde_operator->solve_linear(displacement_np,
                                                                  rhs,
                                                                  this->get_scaling_factor_mass(),
                                                                  this->get_mid_time());

    iterations_linear += N_iter_linear;

    if(this->print_solver_info())
      print_solver_info_linear(pcout, N_iter_linear, timer.wall_time());
  }

  this->timer_tree->insert({"Timeloop", "Solve"}, timer.wall_time());

  // compute vectors at time t_{n+1}
  timer.restart();

  this->update_displacement(displacement_np, displacement_n);
  this->update_velocity(velocity_np, displacement_np, displacement_n, velocity_n, acceleration_n);
  this->update_acceleration(
    acceleration_np, displacement_np, displacement_n, velocity_n, acceleration_n);

  this->timer_tree->insert({"Timeloop", "Update vectors"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntGenAlpha<dim, Number>::extrapolate_displacement_to_np(VectorType & displacement)
{
  // D_np = D_n + dt * V_n + 1/2 dt^2 * A_n
  displacement = displacement_n;
  displacement.add(this->get_time_step_size(), velocity_n);
  displacement.add(std::pow(this->get_time_step_size(), 2.0) / 2.0, acceleration_n);
}

template<int dim, typename Number>
typename TimeIntGenAlpha<dim, Number>::VectorType const &
TimeIntGenAlpha<dim, Number>::get_displacement_np()
{
  return displacement_np;
}

template<int dim, typename Number>
void
TimeIntGenAlpha<dim, Number>::extrapolate_velocity_to_np(VectorType & velocity)
{
  // V_np = V_n + dt * A_n
  velocity = velocity_n;
  velocity.add(this->get_time_step_size(), acceleration_n);
}

template<int dim, typename Number>
typename TimeIntGenAlpha<dim, Number>::VectorType const &
TimeIntGenAlpha<dim, Number>::get_velocity_n()
{
  return velocity_n;
}

template<int dim, typename Number>
typename TimeIntGenAlpha<dim, Number>::VectorType const &
TimeIntGenAlpha<dim, Number>::get_velocity_np()
{
  return velocity_np;
}

template<int dim, typename Number>
void
TimeIntGenAlpha<dim, Number>::prepare_vectors_for_next_timestep()
{
  displacement_n.swap(displacement_np);
  velocity_n.swap(velocity_np);
  acceleration_n.swap(acceleration_np);
}

template<int dim, typename Number>
void
TimeIntGenAlpha<dim, Number>::do_write_restart(std::string const & filename) const
{
  (void)filename;
  AssertThrow(false, ExcMessage("Restart has not been implemented for Structure."));
}

template<int dim, typename Number>
void
TimeIntGenAlpha<dim, Number>::do_read_restart(std::ifstream & in)
{
  (void)in;
  AssertThrow(false, ExcMessage("Restart has not been implemented for Structure."));
}

template<int dim, typename Number>
void
TimeIntGenAlpha<dim, Number>::postprocessing() const
{
  Timer timer;
  timer.restart();

  postprocessor->do_postprocessing(displacement_n, this->get_time(), this->get_time_step_number());

  this->timer_tree->insert({"Timeloop", "Postprocessing"}, timer.wall_time());
}

template<int dim, typename Number>
bool
TimeIntGenAlpha<dim, Number>::print_solver_info() const
{
  return param.solver_info_data.write(this->global_timer.wall_time(),
                                      this->time - this->start_time,
                                      this->time_step_number);
}

template<int dim, typename Number>
void
TimeIntGenAlpha<dim, Number>::print_iterations() const
{
  unsigned int const N_time_steps = std::max(1, int(this->get_time_step_number()) - 1);

  std::vector<std::string> names;
  std::vector<double>      iterations_avg;

  if(param.large_deformation)
  {
    names = {"Nonlinear iterations",
             "Linear iterations (accumulated)",
             "Linear iterations (per nonlinear it.)"};

    iterations_avg.resize(3);
    iterations_avg[0] = (double)iterations_nonlinear / (double)N_time_steps;
    iterations_avg[1] = (double)iterations_linear / (double)N_time_steps;
    if(iterations_avg[0] > std::numeric_limits<double>::min())
      iterations_avg[2] = iterations_avg[1] / iterations_avg[0];
    else
      iterations_avg[2] = iterations_avg[1];
  }
  else // linear
  {
    names = {"Linear iterations"};
    iterations_avg.resize(1);
    iterations_avg[0] = (double)iterations_linear / (double)N_time_steps;
  }

  unsigned int length = 1;
  for(unsigned int i = 0; i < names.size(); ++i)
  {
    length = length > names[i].length() ? length : names[i].length();
  }

  // print
  for(unsigned int i = 0; i < iterations_avg.size(); ++i)
  {
    this->pcout << "  " << std::setw(length + 2) << std::left << names[i] << std::fixed
                << std::setprecision(2) << std::right << std::setw(6) << iterations_avg[i]
                << std::endl;
  }
}

template class TimeIntGenAlpha<2, float>;
template class TimeIntGenAlpha<3, float>;

template class TimeIntGenAlpha<2, double>;
template class TimeIntGenAlpha<3, double>;
} // namespace Structure
