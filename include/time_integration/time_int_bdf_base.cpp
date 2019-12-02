/*
 * time_int_bdf_base.cpp
 *
 *  Created on: Nov 14, 2018
 *      Author: fehn
 */

#include "time_int_bdf_base.h"

template<typename Number>
TimeIntBDFBase<Number>::TimeIntBDFBase(double const        start_time_,
                                       double const        end_time_,
                                       unsigned int const  max_number_of_time_steps_,
                                       double const        order_,
                                       bool const          start_with_low_order_,
                                       bool const          adaptive_time_stepping_,
                                       RestartData const & restart_data_)
  : TimeIntBase(start_time_, end_time_, max_number_of_time_steps_, restart_data_),
    order(order_),
    bdf(order_, start_with_low_order_),
    extra(order_, start_with_low_order_),
    start_with_low_order(start_with_low_order_),
    adaptive_time_stepping(adaptive_time_stepping_),
    time_steps(order_, -1.0)
{
}

template<typename Number>
void
TimeIntBDFBase<Number>::setup(bool const do_restart)
{
  this->pcout << std::endl << "Setup time integrator ..." << std::endl << std::endl;

  // operator-integration-factor splitting
  initialize_oif();

  // allocate global solution vectors
  allocate_vectors();

  // initializes the solution and calculates the time step size
  initialize_solution_and_calculate_timestep(do_restart);

  // this is where the setup of derived classes is performed
  setup_derived();

  this->pcout << std::endl << "... done!" << std::endl;
}

template<typename Number>
void
TimeIntBDFBase<Number>::initialize_solution_and_calculate_timestep(bool do_restart)
{
  if(do_restart)
  {
    // The solution vectors and the current time, the time step size, etc. have to be read from
    // restart files.
    read_restart();
  }
  else
  {
    // The time step size might depend on the current solution.
    initialize_current_solution();

    // The time step size has to be computed before the solution can be initialized at times
    // t - dt[1], t - dt[1] - dt[2], etc.
    calculate_time_step_size();

    // Finally, prescribe initial conditions at former instants of time. This is only necessary if
    // the time integrator starts with a high-order scheme in the first time step.
    if(start_with_low_order == false)
      initialize_former_solutions();
  }
}

template<typename Number>
void
TimeIntBDFBase<Number>::timeloop_steady_problem()
{
  this->global_timer.restart();

  postprocessing_steady_problem();

  solve_steady_problem();

  postprocessing_steady_problem();
}

template<typename Number>
double
TimeIntBDFBase<Number>::get_scaling_factor_time_derivative_term() const
{
  return bdf.get_gamma0() / time_steps[0];
}

template<typename Number>
double
TimeIntBDFBase<Number>::get_next_time() const
{
  /*
   *   time t
   *  -------->   t_{n-2}   t_{n-1}   t_{n}      t_{n+1}
   *  _______________|_________|________|___________|___________\
   *                 |         |        |           |           /
   *  time_steps-vec:   dt[2]     dt[1]    dt[0]
   */
  return this->time + this->time_steps[0];
}

template<typename Number>
double
TimeIntBDFBase<Number>::get_previous_time(int const i /* t_{n-i} */) const
{
  /*
   *   time t
   *  -------->   t_{n-2}   t_{n-1}   t_{n}     t_{n+1}
   *  _______________|_________|________|___________|___________\
   *                 |         |        |           |           /
   *  time_steps-vec:   dt[2]     dt[1]    dt[0]
   */
  AssertThrow(i >= 0, ExcMessage("Invalid parameter."));

  double t = this->time;

  for(int k = 1; k <= i; ++k)
    t -= this->time_steps[k];

  return t;
}

template<typename Number>
void
TimeIntBDFBase<Number>::reset_time(double const & current_time)
{
  // Only allow overwriting the time to a value smaller than start_time (which is needed when
  // coupling different solvers, different domains, etc.).
  if(current_time <= start_time + eps)
    this->time = current_time;
  else
    AssertThrow(false, ExcMessage("The variable time may not be overwritten via public access."));
}

template<typename Number>
double
TimeIntBDFBase<Number>::get_time_step_size() const
{
  return get_time_step_size(0);
}

template<typename Number>
double
TimeIntBDFBase<Number>::get_time_step_size(int const i /* dt[i] */) const
{
  /*
   *   time t
   *  -------->   t_{n-2}   t_{n-1}   t_{n}      t_{n+1}
   *  _______________|_________|________|___________|___________\
   *                 |         |        |           |           /
   *  time_steps-vec:   dt[2]     dt[1]    dt[0]
   */
  AssertThrow(i >= 0 && i <= int(order) - 1, ExcMessage("Invalid access."));

  AssertThrow(time_steps[i] > 0.0, ExcMessage("Invalid or uninitialized time step size."));

  return time_steps[i];
}

template<typename Number>
std::vector<double>
TimeIntBDFBase<Number>::get_time_step_vector() const
{
  return time_steps;
}

template<typename Number>
void
TimeIntBDFBase<Number>::push_back_time_step_sizes()
{
  /*
   * push back time steps
   *
   *   time t
   *  -------->   t_{n-2}   t_{n-1}   t_{n}     t_{n+1}
   *  _______________|_________|________|___________|___________\
   *                 |         |        |           |           /
   *  time_steps-vec:   dt[2]     dt[1]    dt[0]
   *
   *                    dt[1]  <- dt[0] <- new_dt
   *
   */
  for(unsigned int i = order - 1; i > 0; --i)
    time_steps[i] = time_steps[i - 1];
}

template<typename Number>
void
TimeIntBDFBase<Number>::set_time_step_size(double const & time_step_size)
{
  // constant time step sizes
  if(adaptive_time_stepping == false)
  {
    AssertThrow(time_step_number == 1,
                ExcMessage("For time integration with constant time step sizes this "
                           "function can only be called in the very first time step."));
  }

  time_steps[0] = time_step_size;

  // Fill time_steps array in the first time step
  if(time_step_number == 1)
  {
    for(unsigned int i = 1; i < order; ++i)
      time_steps[i] = time_steps[0];
  }
}

template<typename Number>
void
TimeIntBDFBase<Number>::do_timestep(bool const do_write_output)
{
  update_time_integrator_constants();

  output_solver_info_header();

  solve_timestep();

  output_remaining_time(do_write_output);

  prepare_vectors_for_next_timestep();

  time += time_steps[0];
  ++time_step_number;

  if(adaptive_time_stepping == true)
  {
    push_back_time_step_sizes();
    double dt = recalculate_time_step_size();
    // make sure that the end time is hit exactly
    dt = std::min(end_time - time, dt);
    set_time_step_size(dt);
  }

  if(restart_data.write_restart == true)
  {
    write_restart();
  }
}

template<typename Number>
void
TimeIntBDFBase<Number>::update_time_integrator_constants()
{
  if(adaptive_time_stepping == false) // constant time steps
  {
    bdf.update(time_step_number);
    extra.update(time_step_number);
  }
  else // adaptive time stepping
  {
    bdf.update(time_step_number, time_steps);
    extra.update(time_step_number, time_steps);
  }

  // use this function to check the correctness of the time integrator constants
  //  std::cout << std::endl << "Time step " << time_step_number << std::endl << std::endl;
  //  std::cout << "Coefficients BDF time integration scheme:" << std::endl;
  //  bdf.print();
  //  std::cout << "Coefficients extrapolation scheme:" << std::endl;
  //  extra.print();
}

template<typename Number>
void
TimeIntBDFBase<Number>::calculate_sum_alphai_ui_oif_substepping(VectorType & sum_alphai_ui,
                                                                double const cfl,
                                                                double const cfl_oif)
{
  VectorType solution_tilde_mp(sum_alphai_ui), solution_tilde_m(sum_alphai_ui);

  /*
   * Loop over all previous time instants required by the BDF scheme and calculate u_tilde by
   * substepping algorithm, i.e., integrate over time interval t_{n-i} <= t <= t_{n+1} for all 0 <=
   * i <= order using explicit Runge-Kutta methods.
   *
   *   time t
   *  -------->   t_{n-2}   t_{n-1}   t_{n}     t_{n+1}
   *  _______________|_________|________|___________|___________\
   *                 |         |        |           |           /
   *  time_steps-vec:   dt[2]     dt[1]    dt[0]
   *
   *  i=0:                                   k=0
   *                                    +---------->+
   *
   *  i=1:                         k=1       k=0
   *                           +--------+---------->+
   *
   *  i=2:               k=2       k=1       k=0
   *                 +-------- +--------+---------->+
   */
  for(unsigned int i = 0; i < order; ++i)
  {
    // initialize solution: u_tilde(s=0) = u(t_{n-i})
    initialize_solution_oif_substepping(solution_tilde_m, i);

    // integrate over time interval t_{n-i} <= t <= t_{n+1}
    // which are i+1 "macro" time steps
    for(int k = i; k >= 0; --k)
    {
      // integrate over interval: t_{n-k} <= t <= t_{n-k+1}

      // calculate start time t_{n-k}
      double const time_n_k = this->get_previous_time(k);

      // number of sub-steps per "macro" time step
      int M = (int)(cfl / (cfl_oif - eps));

      AssertThrow(M >= 1, ExcMessage("Invalid parameters cfl and cfl_oif."));

      // make sure that cfl_oif is not violated
      if(cfl_oif < cfl / double(M) - eps)
        M += 1;

      // calculate sub-stepping time step size delta_s
      double const delta_s = this->get_time_step_size(k) / (double)M;

      for(int m = 0; m < M; ++m)
      {
        do_timestep_oif_substepping(solution_tilde_mp,
                                    solution_tilde_m,
                                    time_n_k + delta_s * m,
                                    delta_s);

        solution_tilde_mp.swap(solution_tilde_m);
      }
    }

    // note that solution_tilde_m contains the solution since swap() has been done last
    update_sum_alphai_ui_oif_substepping(sum_alphai_ui, solution_tilde_m, i);
  }
}

template<typename Number>
void
TimeIntBDFBase<Number>::do_read_restart(std::ifstream & in)
{
  boost::archive::binary_iarchive ia(in);
  read_restart_preamble(ia);
  read_restart_vectors(ia);

  // In order to change the CFL number (or the time step calculation criterion in general),
  // start_with_low_order = true has to be used. Otherwise, the old solutions would not fit the
  // time step increments.
  if(start_with_low_order == true)
    calculate_time_step_size();
}

template<typename Number>
void
TimeIntBDFBase<Number>::read_restart_preamble(boost::archive::binary_iarchive & ia)
{
  // Note that the operations done here must be in sync with the output.

  // 1. ranks
  unsigned int n_old_ranks = 1;
  ia &         n_old_ranks;

  unsigned int n_ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  AssertThrow(n_old_ranks == n_ranks,
              ExcMessage("Tried to restart with " + Utilities::to_string(n_ranks) +
                         " processes, "
                         "but restart was written on " +
                         Utilities::to_string(n_old_ranks) + " processes."));

  // 2. time
  ia & time;

  // Note that start_time has to be set to the new start_time (since param.start_time might still be
  // the original start time).
  this->start_time = time;

  // 3. order
  unsigned int old_order = 1;
  ia &         old_order;

  AssertThrow(old_order == order, ExcMessage("Order of time integrator may not change."));

  // 4. time step sizes
  for(unsigned int i = 0; i < order; i++)
    ia & time_steps[i];
}

template<typename Number>
void
TimeIntBDFBase<Number>::do_write_restart(std::string const & filename) const
{
  std::ostringstream oss;

  boost::archive::binary_oarchive oa(oss);

  write_restart_preamble(oa);
  write_restart_vectors(oa);
  write_restart_file(oss, filename);
}

template<typename Number>
void
TimeIntBDFBase<Number>::write_restart_preamble(boost::archive::binary_oarchive & oa) const
{
  unsigned int n_ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  // 1. ranks
  oa & n_ranks;

  // 2. time
  oa & time;

  // 3. order
  oa & order;

  // 4. time step sizes
  for(unsigned int i = 0; i < order; i++)
    oa & time_steps[i];
}

template<typename Number>
void
TimeIntBDFBase<Number>::postprocessing_steady_problem() const
{
  AssertThrow(false, ExcMessage("This function has to be implemented by derived classes."));
}

template<typename Number>
void
TimeIntBDFBase<Number>::solve_steady_problem()
{
  AssertThrow(false, ExcMessage("This function has to be implemented by derived classes."));
}

template<typename Number>
void
TimeIntBDFBase<Number>::initialize_solution_oif_substepping(VectorType &, unsigned int)
{
  AssertThrow(false, ExcMessage("This function has to be implemented by derived classes."));
}

template<typename Number>
void
TimeIntBDFBase<Number>::update_sum_alphai_ui_oif_substepping(VectorType &,
                                                             VectorType const &,
                                                             unsigned int)
{
  AssertThrow(false, ExcMessage("This function has to be implemented by derived classes."));
}

template<typename Number>
void
TimeIntBDFBase<Number>::do_timestep_oif_substepping(VectorType &,
                                                    VectorType &,
                                                    double const,
                                                    double const)
{
  AssertThrow(false, ExcMessage("This function has to be implemented by derived classes."));
}

template class TimeIntBDFBase<float>;
template class TimeIntBDFBase<double>;
