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

#include <exadg/structure/postprocessor/postprocessor_base.h>
#include <exadg/structure/spatial_discretization/interface.h>
#include <exadg/structure/time_integration/time_int_gen_alpha.h>
#include <exadg/structure/user_interface/parameters.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

template<int dim, typename Number>
TimeIntGenAlpha<dim, Number>::TimeIntGenAlpha(
  std::shared_ptr<Interface::Operator<Number>> operator_,
  std::shared_ptr<PostProcessorBase<Number>>   postprocessor_,
  Parameters const &                           param_,
  MPI_Comm const &                             mpi_comm_,
  bool const                                   is_test_)
  : TimeIntGenAlphaBase<Number>(param_.start_time,
                                param_.end_time,
                                param_.max_number_of_time_steps,
                                param_.spectral_radius,
                                param_.gen_alpha_type,
                                param_.restart_data,
                                mpi_comm_,
                                is_test_),
    pde_operator(operator_),
    postprocessor(postprocessor_),
    refine_steps_time(param_.n_refine_time),
    param(param_),
    mpi_comm(mpi_comm_),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm_) == 0),
    use_extrapolation(true),
    store_solution(false),
    iterations({0, {0, 0}})
{
}

template<int dim, typename Number>
void
TimeIntGenAlpha<dim, Number>::setup(bool const do_restart)
{
  this->pcout << std::endl << "Setup elasticity time integrator ..." << std::endl << std::flush;

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
TimeIntGenAlpha<dim, Number>::advance_one_timestep_partitioned_solve(bool const use_extrapolation,
                                                                     bool const store_solution)
{
  if(this->use_extrapolation == false)
    AssertThrow(this->store_solution == true, ExcMessage("Invalid parameters."));

  this->use_extrapolation = use_extrapolation;
  this->store_solution    = store_solution;

  this->advance_one_timestep_solve();
}

template<int dim, typename Number>
void
TimeIntGenAlpha<dim, Number>::solve_timestep()
{
  // compute right-hand side in case of linear problems or "constant vector"
  // in case of nonlinear problems
  Timer timer;
  timer.restart();

  // compute const_vector
  VectorType const_vector, rhs;
  const_vector.reinit(displacement_n);
  rhs.reinit(displacement_n);
  this->compute_const_vector(rhs, displacement_n, velocity_n, acceleration_n);
  pde_operator->apply_mass_operator(const_vector, rhs);
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
  if(use_extrapolation)
    displacement_np = displacement_n;
  else
    displacement_np = displacement_last_iter;

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

    iterations.first += 1;
    std::get<0>(iterations.second) += std::get<0>(iter);
    std::get<1>(iterations.second) += std::get<1>(iter);

    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl << "Solve nonlinear elasticity problem:";
      print_solver_info_nonlinear(pcout, std::get<0>(iter), std::get<1>(iter), timer.wall_time());
    }
  }
  else // linear case
  {
    // solve linear system of equations
    unsigned int const iter = pde_operator->solve_linear(displacement_np,
                                                         rhs,
                                                         this->get_scaling_factor_mass(),
                                                         this->get_mid_time());

    iterations.first += 1;
    std::get<1>(iterations.second) += iter;

    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl << "Solve linear elasticity problem:";
      print_solver_info_linear(pcout, iter, timer.wall_time());
    }
  }

  this->timer_tree->insert({"Timeloop", "Solve"}, timer.wall_time());

  // compute vectors at time t_{n+1}
  timer.restart();

  if(this->store_solution)
    displacement_last_iter = displacement_np;

  this->update_displacement(displacement_np, displacement_n);
  this->update_velocity(velocity_np, displacement_np, displacement_n, velocity_n, acceleration_n);
  this->update_acceleration(
    acceleration_np, displacement_np, displacement_n, velocity_n, acceleration_n);

  this->timer_tree->insert({"Timeloop", "Update vectors"}, timer.wall_time());
}

template<int dim, typename Number>
typename TimeIntGenAlpha<dim, Number>::VectorType const &
TimeIntGenAlpha<dim, Number>::get_displacement_np()
{
  return displacement_np;
}

template<int dim, typename Number>
void
TimeIntGenAlpha<dim, Number>::extrapolate_displacement_to_np(VectorType & displacement)
{
  // D_np = D_n + dt * V_n
  displacement = displacement_n;
  displacement.add(this->get_time_step_size(), velocity_n);
}

template<int dim, typename Number>
void
TimeIntGenAlpha<dim, Number>::extrapolate_velocity_to_np(VectorType & velocity)
{
  // use old velocity solution as guess for velocity_np
  velocity = velocity_n;
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
TimeIntGenAlpha<dim, Number>::set_displacement(VectorType const & displacement)
{
  displacement_np = displacement;

  // velocity_np, acceleration_np depend on displacement_np, so we need to
  // update these vectors as well
  this->update_velocity(velocity_np, displacement_np, displacement_n, velocity_n, acceleration_n);
  this->update_acceleration(
    acceleration_np, displacement_np, displacement_n, velocity_n, acceleration_n);
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
  std::vector<std::string> names;
  std::vector<double>      iterations_avg;

  if(param.large_deformation)
  {
    names = {"Nonlinear iterations",
             "Linear iterations (accumulated)",
             "Linear iterations (per nonlinear it.)"};

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
  else // linear
  {
    names = {"Linear iterations"};
    iterations_avg.resize(1);
    iterations_avg[0] =
      (double)std::get<1>(iterations.second) / std::max(1., (double)iterations.first);
  }

  print_list_of_iterations(pcout, names, iterations_avg);
}

template class TimeIntGenAlpha<2, float>;
template class TimeIntGenAlpha<3, float>;

template class TimeIntGenAlpha<2, double>;
template class TimeIntGenAlpha<3, double>;

} // namespace Structure
} // namespace ExaDG
