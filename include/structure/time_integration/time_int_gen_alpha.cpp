/*
 * time_int_gen_alpha.cpp
 *
 *  Created on: 20.04.2020
 *      Author: fehn
 */

#include "time_int_gen_alpha.h"

namespace Structure
{
template<int dim, typename Number>
TimeIntGenAlpha<dim, Number>::TimeIntGenAlpha(
  std::shared_ptr<Operator<dim, Number>>      operator_,
  std::shared_ptr<PostProcessor<dim, Number>> postprocessor_,
  InputParameters const &                     param_,
  MPI_Comm const &                            mpi_comm_)
  : TimeIntGenAlphaBase<Number>(param_.start_time,
                                param_.end_time,
                                param_.max_number_of_time_steps,
                                param_.spectral_radius,
                                param_.gen_alpha_type,
                                param_.restart_data,
                                mpi_comm_),
    pde_operator(operator_),
    postprocessor(postprocessor_),
    param(param_),
    mpi_comm(mpi_comm_),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm_) == 0)
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
    this->set_current_time_step_size(param.time_step_size);

    pde_operator->prescribe_initial_displacement(displacement_n, this->get_time());
    pde_operator->prescribe_initial_velocity(velocity_n, this->get_time());
    // solve momentum equation to obtain initial accelerations
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
  pde_operator->apply_dirichlet_bc_homogeneous(const_vector);

  // solve system of equations for displacement d_{n+1-alpha_f}
  unsigned int N_iter_nonlinear = 0;
  unsigned int N_iter_linear    = 0;

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

    N_iter_nonlinear = std::get<0>(iter);
    N_iter_linear    = std::get<1>(iter);
  }
  else // linear case
  {
    // calculate right-hand side vector
    pde_operator->compute_rhs_linear(rhs, this->get_mid_time());
    // shift const_vector to right-hand side
    rhs.add(-1.0, const_vector);

    // solve linear system of equations
    N_iter_linear = pde_operator->solve_linear(displacement_np,
                                               rhs,
                                               this->get_scaling_factor_mass(),
                                               this->get_mid_time());
  }

  // compute vectors at time t_{n+1}
  this->update_displacement(displacement_np, displacement_n);
  this->update_velocity(velocity_np, displacement_np, displacement_n, velocity_n, acceleration_n);
  this->update_acceleration(
    acceleration_np, displacement_np, displacement_n, velocity_n, acceleration_n);

  // solver info output
  if(this->print_solver_info())
  {
    if(param.large_deformation) // nonlinear case
    {
      double N_iter_linear_avg =
        (N_iter_nonlinear > 0) ? double(N_iter_linear) / double(N_iter_nonlinear) : N_iter_linear;

      // clang-format off
        pcout << std::endl
              << "  Newton iterations:      " << std::setw(12) << std::right << N_iter_nonlinear << std::endl
              << "  Linear iterations (avg):" << std::setw(12) << std::scientific << std::setprecision(4) << std::right << N_iter_linear_avg << std::endl
              << "  Linear iterations (tot):" << std::setw(12) << std::scientific << std::setprecision(4) << std::right << N_iter_linear << std::endl
              << "  Wall time [s]:          " << std::setw(12) << std::scientific << std::setprecision(4) << timer.wall_time() << std::endl;
      // clang-format on
    }
    else // linear case
    {
      // clang-format off
        pcout << std::endl
              << "  Iterations:   " << std::setw(12) << std::right << N_iter_linear << std::endl
              << "  Wall time [s]:" << std::setw(12) << std::scientific << std::setprecision(4) << timer.wall_time() << std::endl;
      // clang-format on
    }
  }
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
  postprocessor->do_postprocessing(displacement_n, this->get_time(), this->get_time_step_number());
}

template<int dim, typename Number>
bool
TimeIntGenAlpha<dim, Number>::print_solver_info() const
{
  return param.solver_info_data.write(this->global_timer.wall_time(),
                                      this->time - this->start_time,
                                      this->time_step_number);
}

template class TimeIntGenAlpha<2, float>;
template class TimeIntGenAlpha<3, float>;

template class TimeIntGenAlpha<2, double>;
template class TimeIntGenAlpha<3, double>;
} // namespace Structure
