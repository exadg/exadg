/*
 * time_int_explicit_runge_kutta.cpp
 *
 *  Created on: Nov 14, 2018
 *      Author: fehn
 */

#include "time_int_explicit_runge_kutta.h"

#include "../interface_space_time/operator.h"
#include "../user_interface/input_parameters.h"
#include "time_integration/time_step_calculation.h"

namespace CompNS
{
template<int dim, typename Number>
TimeIntExplRK<dim, Number>::TimeIntExplRK(std::shared_ptr<Operator>    operator_in,
                                          InputParameters<dim> const & param_in,
                                          unsigned int const           n_refine_time_in)
  : TimeIntExplRKBase<Number>(param_in.start_time,
                              param_in.end_time,
                              param_in.max_number_of_time_steps,
                              param_in.restart_data,
                              false), // currently no adaptive time stepping implemented
    pde_operator(operator_in),
    param(param_in),
    time_postprocessing(0.0),
    l2_norm(0.0),
    n_refine_time(n_refine_time_in),
    cfl_number(param.cfl_number / std::pow(2.0, n_refine_time)),
    diffusion_number(param.diffusion_number / std::pow(2.0, n_refine_time))
{
}

template<int dim, typename Number>
void
TimeIntExplRK<dim, Number>::initialize_time_integrator()
{
  // initialize Runge-Kutta time integrator
  if(this->param.temporal_discretization == TemporalDiscretization::ExplRK)
  {
    rk_time_integrator.reset(
      new ExplicitRungeKuttaTimeIntegrator<Operator, VectorType>(param.order_time_integrator,
                                                                 pde_operator));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::ExplRK3Stage4Reg2C)
  {
    rk_time_integrator.reset(new LowStorageRK3Stage4Reg2C<Operator, VectorType>(pde_operator));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::ExplRK4Stage5Reg2C)
  {
    rk_time_integrator.reset(new LowStorageRK4Stage5Reg2C<Operator, VectorType>(pde_operator));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::ExplRK4Stage5Reg3C)
  {
    rk_time_integrator.reset(new LowStorageRK4Stage5Reg3C<Operator, VectorType>(pde_operator));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::ExplRK5Stage9Reg2S)
  {
    rk_time_integrator.reset(new LowStorageRK5Stage9Reg2S<Operator, VectorType>(pde_operator));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::ExplRK3Stage7Reg2)
  {
    rk_time_integrator.reset(new LowStorageRKTD<Operator, VectorType>(pde_operator, 3, 7));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::ExplRK4Stage8Reg2)
  {
    rk_time_integrator.reset(new LowStorageRKTD<Operator, VectorType>(pde_operator, 4, 8));
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::SSPRK)
  {
    rk_time_integrator.reset(
      new SSPRK<Operator, VectorType>(pde_operator, param.order_time_integrator, param.stages));
  }
}

/*
 *  initialize global solution vectors (allocation)
 */
template<int dim, typename Number>
void
TimeIntExplRK<dim, Number>::initialize_vectors()
{
  pde_operator->initialize_dof_vector(this->solution_n);
  pde_operator->initialize_dof_vector(this->solution_np);
}

/*
 *  initializes the solution by interpolation of analytical solution
 */
template<int dim, typename Number>
void
TimeIntExplRK<dim, Number>::initialize_solution()
{
  pde_operator->prescribe_initial_conditions(this->solution_n, this->time);
}

/*
 *  calculate time step size
 */
template<int dim, typename Number>
void
TimeIntExplRK<dim, Number>::calculate_time_step_size()
{
  this->pcout << std::endl << "Calculation of time step size:" << std::endl << std::endl;

  unsigned int degree = pde_operator->get_polynomial_degree();

  if(param.calculation_of_time_step_size == TimeStepCalculation::UserSpecified)
  {
    this->time_step = calculate_const_time_step(param.time_step_size, n_refine_time);

    print_parameter(this->pcout, "time step size", this->time_step);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::CFL)
  {
    // calculate minimum vertex distance
    double const h_min = pde_operator->calculate_minimum_element_length();

    print_parameter(this->pcout, "h_min", h_min);

    // calculate time step according to CFL condition

    // speed of sound a = sqrt(gamma * R * T)
    double const speed_of_sound =
      sqrt(param.heat_capacity_ratio * param.specific_gas_constant * param.max_temperature);
    double const acoustic_wave_speed = param.max_velocity + speed_of_sound;

    this->time_step = calculate_time_step_cfl_global(
      cfl_number, acoustic_wave_speed, h_min, degree, param.exponent_fe_degree_cfl);

    this->time_step =
      adjust_time_step_to_hit_end_time(this->start_time, this->end_time, this->time_step);

    print_parameter(this->pcout, "U_max", param.max_velocity);
    print_parameter(this->pcout, "speed of sound", speed_of_sound);
    print_parameter(this->pcout, "CFL", cfl_number);
    print_parameter(this->pcout, "Time step size (convection)", this->time_step);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::Diffusion)
  {
    // calculate minimum vertex distance
    double const h_min = pde_operator->calculate_minimum_element_length();

    print_parameter(this->pcout, "h_min", h_min);

    // calculate time step size according to diffusion number condition
    this->time_step =
      calculate_const_time_step_diff(diffusion_number,
                                     param.dynamic_viscosity / param.reference_density,
                                     h_min,
                                     degree,
                                     param.exponent_fe_degree_viscous);

    this->time_step =
      adjust_time_step_to_hit_end_time(this->start_time, this->end_time, this->time_step);

    print_parameter(this->pcout, "Diffusion number", diffusion_number);
    print_parameter(this->pcout, "Time step size (diffusion)", this->time_step);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::CFLAndDiffusion)
  {
    // calculate minimum vertex distance
    double const h_min = pde_operator->calculate_minimum_element_length();

    print_parameter(this->pcout, "h_min", h_min);

    double time_step_conv = std::numeric_limits<double>::max();
    double time_step_diff = std::numeric_limits<double>::max();

    // calculate time step according to CFL condition

    // speed of sound a = sqrt(gamma * R * T)
    double const speed_of_sound =
      sqrt(param.heat_capacity_ratio * param.specific_gas_constant * param.max_temperature);
    double const acoustic_wave_speed = param.max_velocity + speed_of_sound;

    time_step_conv = calculate_time_step_cfl_global(
      cfl_number, acoustic_wave_speed, h_min, degree, param.exponent_fe_degree_cfl);

    print_parameter(this->pcout, "U_max", param.max_velocity);
    print_parameter(this->pcout, "speed of sound", speed_of_sound);
    print_parameter(this->pcout, "CFL", cfl_number);
    print_parameter(this->pcout, "Time step size (convection)", time_step_conv);

    // calculate time step size according to diffusion number condition
    time_step_diff =
      calculate_const_time_step_diff(diffusion_number,
                                     param.dynamic_viscosity / param.reference_density,
                                     h_min,
                                     degree,
                                     param.exponent_fe_degree_viscous);

    print_parameter(this->pcout, "Diffusion number", diffusion_number);
    print_parameter(this->pcout, "Time step size (diffusion)", time_step_diff);

    this->time_step = std::min(time_step_conv, time_step_diff);

    this->time_step =
      adjust_time_step_to_hit_end_time(this->start_time, this->end_time, this->time_step);

    print_parameter(this->pcout, "Time step size (combined)", this->time_step);
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified type of time step calculation is not implemented."));
  }
}

template<int dim, typename Number>
double
TimeIntExplRK<dim, Number>::recalculate_time_step_size() const
{
  AssertThrow(false, ExcMessage("Currently no adaptive time stepping implemented."));

  return 1.0;
}

template<int dim, typename Number>
void
TimeIntExplRK<dim, Number>::detect_instabilities() const
{
  if(this->param.detect_instabilities == true)
  {
    double const l2_norm_new = this->solution_n.l2_norm();
    if(l2_norm > 1.e-12)
    {
      AssertThrow(l2_norm_new < 10. * l2_norm,
                  ExcMessage("Instabilities detected. Norm of solution vector seems to explode."));
    }

    l2_norm = l2_norm_new;
  }
}

template<int dim, typename Number>
void
TimeIntExplRK<dim, Number>::postprocessing() const
{
  timer_postprocessing.restart();

  detect_instabilities();

  pde_operator->do_postprocessing(this->solution_n, this->time, this->time_step_number);

  time_postprocessing += timer_postprocessing.wall_time();
}

template<int dim, typename Number>
void
TimeIntExplRK<dim, Number>::solve_timestep()
{
  Timer timer;
  timer.restart();

  rk_time_integrator->solve_timestep(this->solution_np,
                                     this->solution_n,
                                     this->time,
                                     this->time_step);

  double const wall_time = timer.wall_time();

  // output
  if(print_solver_info())
  {
    this->pcout << std::endl
                << "Solve time step explicitly: Wall time in [s] = " << std::scientific << wall_time
                << std::endl;
  }
}

template<int dim, typename Number>
bool
TimeIntExplRK<dim, Number>::print_solver_info() const
{
  return this->get_time_step_number() % param.output_solver_info_every_timesteps == 0;
}

template<int dim, typename Number>
void
TimeIntExplRK<dim, Number>::analyze_computing_times() const
{
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl
              << "Computing times:                                                                 "
              << std::endl;

  // overall wall time including postprocessing
  Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(this->total_time, MPI_COMM_WORLD);
  this->pcout << std::scientific << std::setprecision(4)
              << " Total wall time in [s] =          " << data.avg << std::endl;

  // time spent with operator evaluation
  double time_operator_evaluation = pde_operator->get_wall_time_operator_evaluation();
  Utilities::MPI::MinMaxAvg data_op =
    Utilities::MPI::min_max_avg(time_operator_evaluation, MPI_COMM_WORLD);
  this->pcout << std::scientific << std::setprecision(4)
              << " Wall time operator eval. in [s] = " << data_op.avg << " -> " << std::fixed
              << std::setprecision(2) << data_op.avg / data.avg * 100. << " %" << std::endl;

  // time spent with postprocessing
  Utilities::MPI::MinMaxAvg data_post =
    Utilities::MPI::min_max_avg(this->time_postprocessing, MPI_COMM_WORLD);
  this->pcout << std::scientific << std::setprecision(4)
              << " Wall time postprocessing in [s] = " << data_post.avg << " -> " << std::fixed
              << std::setprecision(2) << data_post.avg / data.avg * 100. << " %" << std::endl;

  // Wall time per time step
  unsigned int N_time_steps = this->time_step_number - 1;
  this->pcout << std::endl
              << "Time steps:" << std::endl
              << " Number of time steps =            " << std::left << N_time_steps << std::endl
              << " Wall time per time step in [s] =  " << std::scientific << std::setprecision(4)
              << data.avg / (double)N_time_steps << std::endl;

  // Computational costs
  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  this->pcout << std::scientific << std::setprecision(4) << std::endl
              << "Computational costs = t_wall * N_cores:" << std::endl
              << " Number of MPI processes =         " << N_mpi_processes << std::endl
              << " Computational costs in [CPUs] =   " << data.avg * (double)N_mpi_processes
              << std::endl
              << " Computational costs in [CPUh] =   "
              << data.avg * (double)N_mpi_processes / 3600.0 << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl;
}

// instantiations
#include <navierstokes/config.h>

#if DIM_2 && OP_FLOAT
template class TimeIntExplRK<2, float>;
#endif
#if DIM_3 && OP_FLOAT
template class TimeIntExplRK<3, float>;
#endif

#if DIM_2 && OP_DOUBLE
template class TimeIntExplRK<2, double>;
#endif
#if DIM_3 && OP_DOUBLE
template class TimeIntExplRK<3, double>;
#endif

} // namespace CompNS
