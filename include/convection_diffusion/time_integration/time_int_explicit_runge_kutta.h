/*
 * TimeIntExplRKConvDiff.h
 *
 *  Created on: Aug 2, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_TIME_INT_EXPLICIT_RUNGE_KUTTA_H_
#define INCLUDE_CONVECTION_DIFFUSION_TIME_INT_EXPLICIT_RUNGE_KUTTA_H_

#include <deal.II/lac/parallel_vector.h>

#include <deal.II/base/timer.h>

#include "convection_diffusion/spatial_discretization/dg_convection_diffusion_operation.h"
#include "convection_diffusion/user_interface/input_parameters.h"
#include "functionalities/print_functions.h"
#include "time_integration/time_step_calculation.h"

namespace ConvDiff
{
template<int dim, int fe_degree, typename value_type>
class TimeIntExplRK
{
public:
  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

  TimeIntExplRK(
    std::shared_ptr<ConvDiff::DGOperation<dim, fe_degree, value_type>> conv_diff_operation_in,
    ConvDiff::InputParameters const &                                  param_in,
    std::shared_ptr<Function<dim>>                                     velocity_in,
    unsigned int const                                                 n_refine_time_in)
    : conv_diff_operation(conv_diff_operation_in),
      param(param_in),
      velocity(velocity_in),
      total_time(0.0),
      pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
      time(param.start_time),
      time_step(1.0),
      time_step_number(1),
      adaptive_time_stepping(false),
      n_refine_time(n_refine_time_in),
      cfl_number(param.cfl_number / std::pow(2.0, n_refine_time)),
      diffusion_number(param.diffusion_number / std::pow(2.0, n_refine_time))
  {
  }

  void
  setup();

  void
  timeloop();

  bool
  advance_one_timestep(bool write_final_output);

  void
  set_time(double const & current_time)
  {
    this->time = current_time;
  }

  double
  get_time_step_size() const
  {
    if(adaptive_time_stepping == true)
    {
      double const EPSILON = 1.e-10;
      if(time > param.start_time - EPSILON)
      {
        return time_step;
      }
      else // time integrator has not yet started
      {
        // return a large value because we take the minimum time step size when coupling this time
        // integrator to others. This way, this time integrator does not pose a restriction on the
        // time step size.
        return std::numeric_limits<double>::max();
      }
    }
    else // constant time step size
    {
      return time_step;
    }
  }

  void
  set_time_step_size(double const time_step_size)
  {
    time_step = time_step_size;
  }

private:
  void
  initialize_vectors();

  void
  initialize_solution();

  void
  postprocessing() const;

  void
  output_solver_info_header();

  void
  output_remaining_time();

  void
  do_timestep();

  void
  solve_timestep();

  void
  prepare_vectors_for_next_timestep();

  void
  calculate_timestep();

  void
  initialize_time_integrator();

  void
  analyze_computing_times() const;

  std::shared_ptr<ConvDiff::DGOperation<dim, fe_degree, value_type>> conv_diff_operation;

  std::shared_ptr<
    ExplicitTimeIntegrator<ConvDiff::DGOperation<dim, fe_degree, value_type>, VectorType>>
    rk_time_integrator;

  ConvDiff::InputParameters const & param;

  std::shared_ptr<Function<dim>> velocity;

  // timer
  Timer  global_timer;
  double total_time;

  // screen output
  ConditionalOStream pcout;

  // solution vectors
  VectorType solution_n, solution_np;

  // current time and time step size
  double time, time_step;

  // the number of the current time step starting with time_step_number = 1
  unsigned int time_step_number;

  // use adaptive time stepping?
  bool const adaptive_time_stepping;

  unsigned int const n_refine_time;
  double const       cfl_number;
  double const       diffusion_number;
};

template<int dim, int fe_degree, typename value_type>
void
TimeIntExplRK<dim, fe_degree, value_type>::setup()
{
  pcout << std::endl << "Setup time integrator ..." << std::endl;

  // initialize global solution vectors (allocation)
  initialize_vectors();

  // initializes the solution by interpolation of analytical solution
  initialize_solution();

  // initialize time integrator
  initialize_time_integrator();

  // calculate time step size
  calculate_timestep();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntExplRK<dim, fe_degree, value_type>::initialize_vectors()
{
  conv_diff_operation->initialize_dof_vector(solution_n);
  conv_diff_operation->initialize_dof_vector(solution_np);
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntExplRK<dim, fe_degree, value_type>::initialize_solution()
{
  conv_diff_operation->prescribe_initial_conditions(solution_n, time);
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntExplRK<dim, fe_degree, value_type>::calculate_timestep()
{
  pcout << std::endl << "Calculation of time step size:" << std::endl << std::endl;

  if(param.calculation_of_time_step_size ==
     ConvDiff::TimeStepCalculation::ConstTimeStepUserSpecified)
  {
    time_step = calculate_const_time_step(param.time_step_size, n_refine_time);

    print_parameter(pcout, "time step size", time_step);
  }
  else if(param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepCFL)
  {
    AssertThrow(param.equation_type == ConvDiff::EquationType::Convection ||
                  param.equation_type == ConvDiff::EquationType::ConvectionDiffusion,
                ExcMessage("Time step calculation ConstTimeStepCFL does not make sense!"));

    // calculate minimum vertex distance
    const double global_min_cell_diameter = calculate_minimum_vertex_distance(
      conv_diff_operation->get_data().get_dof_handler().get_triangulation());

    print_parameter(pcout, "h_min", global_min_cell_diameter);

    double time_step_conv = 1.0;

    const double max_velocity =
      calculate_max_velocity(conv_diff_operation->get_data().get_dof_handler().get_triangulation(),
                             velocity,
                             time);

    print_parameter(pcout, "U_max", max_velocity);
    print_parameter(pcout, "CFL", cfl_number);
    print_parameter(pcout, "Exponent fe_degree (convection)", param.exponent_fe_degree_convection);

    time_step_conv = calculate_const_time_step_cfl(cfl_number,
                                                   max_velocity,
                                                   global_min_cell_diameter,
                                                   fe_degree,
                                                   param.exponent_fe_degree_convection);

    // decrease time_step in order to exactly hit end_time
    time_step = (param.end_time - param.start_time) /
                (1 + int((param.end_time - param.start_time) / time_step_conv));

    print_parameter(pcout, "Time step size (convection)", time_step);
  }
  else if(param.calculation_of_time_step_size ==
          ConvDiff::TimeStepCalculation::ConstTimeStepDiffusion)
  {
    AssertThrow(param.equation_type == ConvDiff::EquationType::Diffusion ||
                  param.equation_type == ConvDiff::EquationType::ConvectionDiffusion,
                ExcMessage("Time step calculation ConstTimeStepDiffusion does not make sense!"));

    // calculate minimum vertex distance
    const double global_min_cell_diameter = calculate_minimum_vertex_distance(
      conv_diff_operation->get_data().get_dof_handler().get_triangulation());

    print_parameter(pcout, "h_min", global_min_cell_diameter);

    print_parameter(pcout, "Diffusion number", diffusion_number);
    print_parameter(pcout, "Exponent fe_degree (diffusion)", param.exponent_fe_degree_diffusion);

    double time_step_diff = 1.0;
    // calculate time step according to Diffusion number condition
    time_step_diff = calculate_const_time_step_diff(diffusion_number,
                                                    param.diffusivity,
                                                    global_min_cell_diameter,
                                                    fe_degree,
                                                    param.exponent_fe_degree_diffusion);

    // decrease time_step in order to exactly hit end_time
    time_step = (param.end_time - param.start_time) /
                (1 + int((param.end_time - param.start_time) / time_step_diff));

    print_parameter(pcout, "Time step size (diffusion)", time_step);
  }
  else if(param.calculation_of_time_step_size ==
          ConvDiff::TimeStepCalculation::ConstTimeStepCFLAndDiffusion)
  {
    AssertThrow(param.equation_type == ConvDiff::EquationType::ConvectionDiffusion,
                ExcMessage(
                  "Time step calculation ConstTimeStepCFLAndDiffusion does not make sense!"));

    // calculate minimum vertex distance
    const double global_min_cell_diameter = calculate_minimum_vertex_distance(
      conv_diff_operation->get_data().get_dof_handler().get_triangulation());

    print_parameter(pcout, "h_min", global_min_cell_diameter);

    double time_step_conv = std::numeric_limits<double>::max();
    double time_step_diff = std::numeric_limits<double>::max();

    // calculate time step according to CFL condition
    const double max_velocity =
      calculate_max_velocity(conv_diff_operation->get_data().get_dof_handler().get_triangulation(),
                             velocity,
                             time);

    print_parameter(pcout, "U_max", max_velocity);
    print_parameter(pcout, "CFL", cfl_number);
    print_parameter(pcout, "Exponent fe_degree (convection)", param.exponent_fe_degree_convection);

    time_step_conv = calculate_const_time_step_cfl(cfl_number,
                                                   max_velocity,
                                                   global_min_cell_diameter,
                                                   fe_degree,
                                                   param.exponent_fe_degree_convection);

    print_parameter(pcout, "Time step size (convection)", time_step_conv);


    // calculate time step according to Diffusion number condition
    time_step_diff = calculate_const_time_step_diff(diffusion_number,
                                                    param.diffusivity,
                                                    global_min_cell_diameter,
                                                    fe_degree,
                                                    param.exponent_fe_degree_diffusion);

    print_parameter(pcout, "Diffusion number", diffusion_number);
    print_parameter(pcout, "Exponent fe_degree (diffusion)", param.exponent_fe_degree_diffusion);
    print_parameter(pcout, "Time step size (diffusion)", time_step_diff);

    // adopt minimum time step size
    time_step = time_step_diff < time_step_conv ? time_step_diff : time_step_conv;

    // decrease time_step in order to exactly hit end_time
    time_step = (param.end_time - param.start_time) /
                (1 + int((param.end_time - param.start_time) / time_step));

    print_parameter(pcout, "Time step size (combined)", time_step);
  }
  else if(param.calculation_of_time_step_size ==
          ConvDiff::TimeStepCalculation::ConstTimeStepMaxEfficiency)
  {
    // calculate minimum vertex distance
    const double global_min_cell_diameter = calculate_minimum_vertex_distance(
      conv_diff_operation->get_data().get_dof_handler().get_triangulation());

    unsigned int const order = rk_time_integrator->get_order();

    double time_step_tmp = calculate_time_step_max_efficiency(
      param.c_eff, global_min_cell_diameter, fe_degree, order, n_refine_time);

    // decrease time_step in order to exactly hit end_time
    time_step = (param.end_time - param.start_time) /
                (1 + int((param.end_time - param.start_time) / time_step_tmp));

    pcout << "Calculation of time step size (max efficiency):" << std::endl << std::endl;
    print_parameter(pcout, "C_eff", param.c_eff / std::pow(2, n_refine_time));
    print_parameter(pcout, "Time step size", time_step);
  }
  else
  {
    AssertThrow(param.calculation_of_time_step_size ==
                    ConvDiff::TimeStepCalculation::ConstTimeStepUserSpecified ||
                  param.calculation_of_time_step_size ==
                    ConvDiff::TimeStepCalculation::ConstTimeStepCFL ||
                  param.calculation_of_time_step_size ==
                    ConvDiff::TimeStepCalculation::ConstTimeStepDiffusion ||
                  param.calculation_of_time_step_size ==
                    ConvDiff::TimeStepCalculation::ConstTimeStepCFLAndDiffusion ||
                  param.calculation_of_time_step_size ==
                    ConvDiff::TimeStepCalculation::ConstTimeStepMaxEfficiency,
                ExcMessage("Specified calculation of time step size is not implemented!"));
  }
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntExplRK<dim, fe_degree, value_type>::initialize_time_integrator()
{
  if(this->param.time_integrator_rk == ConvDiff::TimeIntegratorRK::ExplRK1Stage1)
  {
    rk_time_integrator.reset(
      new ExplicitRungeKuttaTimeIntegrator<ConvDiff::DGOperation<dim, fe_degree, value_type>,
                                           VectorType>(1, conv_diff_operation));
  }
  else if(this->param.time_integrator_rk == ConvDiff::TimeIntegratorRK::ExplRK2Stage2)
  {
    rk_time_integrator.reset(
      new ExplicitRungeKuttaTimeIntegrator<ConvDiff::DGOperation<dim, fe_degree, value_type>,
                                           VectorType>(2, conv_diff_operation));
  }
  else if(this->param.time_integrator_rk == ConvDiff::TimeIntegratorRK::ExplRK3Stage3)
  {
    rk_time_integrator.reset(
      new ExplicitRungeKuttaTimeIntegrator<ConvDiff::DGOperation<dim, fe_degree, value_type>,
                                           VectorType>(3, conv_diff_operation));
  }
  else if(this->param.time_integrator_rk == ConvDiff::TimeIntegratorRK::ExplRK4Stage4)
  {
    rk_time_integrator.reset(
      new ExplicitRungeKuttaTimeIntegrator<ConvDiff::DGOperation<dim, fe_degree, value_type>,
                                           VectorType>(4, conv_diff_operation));
  }
  else if(this->param.time_integrator_rk == ConvDiff::TimeIntegratorRK::ExplRK3Stage4Reg2C)
  {
    rk_time_integrator.reset(
      new LowStorageRK3Stage4Reg2C<ConvDiff::DGOperation<dim, fe_degree, value_type>, VectorType>(
        conv_diff_operation));
  }
  else if(this->param.time_integrator_rk == ConvDiff::TimeIntegratorRK::ExplRK4Stage5Reg2C)
  {
    rk_time_integrator.reset(
      new LowStorageRK4Stage5Reg2C<ConvDiff::DGOperation<dim, fe_degree, value_type>, VectorType>(
        conv_diff_operation));
  }
  else if(this->param.time_integrator_rk == ConvDiff::TimeIntegratorRK::ExplRK4Stage5Reg3C)
  {
    rk_time_integrator.reset(
      new LowStorageRK4Stage5Reg3C<ConvDiff::DGOperation<dim, fe_degree, value_type>, VectorType>(
        conv_diff_operation));
  }
  else if(this->param.time_integrator_rk == ConvDiff::TimeIntegratorRK::ExplRK5Stage9Reg2S)
  {
    rk_time_integrator.reset(
      new LowStorageRK5Stage9Reg2S<ConvDiff::DGOperation<dim, fe_degree, value_type>, VectorType>(
        conv_diff_operation));
  }
  else if(this->param.time_integrator_rk == ConvDiff::TimeIntegratorRK::ExplRK3Stage7Reg2)
  {
    rk_time_integrator.reset(
      new LowStorageRKTD<ConvDiff::DGOperation<dim, fe_degree, value_type>, VectorType>(
        conv_diff_operation, 3, 7));
  }
  else if(this->param.time_integrator_rk == ConvDiff::TimeIntegratorRK::ExplRK4Stage8Reg2)
  {
    rk_time_integrator.reset(
      new LowStorageRKTD<ConvDiff::DGOperation<dim, fe_degree, value_type>, VectorType>(
        conv_diff_operation, 4, 8));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntExplRK<dim, fe_degree, value_type>::timeloop()
{
  pcout << std::endl << "Starting time loop ..." << std::endl;

  global_timer.restart();

  postprocessing();

  const double EPSILON = 1.0e-10;
  while(time < (param.end_time - EPSILON))
  {
    do_timestep();

    postprocessing();
  }

  total_time += global_timer.wall_time();

  pcout << std::endl << "... done!" << std::endl;

  analyze_computing_times();
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntExplRK<dim, fe_degree, value_type>::do_timestep()
{
  output_solver_info_header();

  solve_timestep();

  output_remaining_time();

  prepare_vectors_for_next_timestep();

  time += time_step;
  ++time_step_number;

  // currently no write_restart implemented

  // currently no adaptive time stepping implemented
}

template<int dim, int fe_degree, typename value_type>
bool
TimeIntExplRK<dim, fe_degree, value_type>::advance_one_timestep(bool write_final_output)
{
  // a small number which is much smaller than the time step size
  const value_type EPSILON = 1.0e-10;

  bool started = time > (param.start_time - EPSILON);

  // If the time integrator has not yet started, simply increment physical time without solving the
  // current time step.
  if(!started)
  {
    time += time_step;
  }

  if(started && time_step_number == 1)
  {
    pcout << std::endl
          << "Starting time loop for scalar convection-diffusion equation ..." << std::endl;

    global_timer.restart();

    postprocessing();
  }

  // check if we have reached the end of the time loop
  bool finished =
    !(time < (param.end_time - EPSILON) && time_step_number <= param.max_number_of_time_steps);

  if(started && !finished)
  {
    // advance one time step
    do_timestep();

    postprocessing();
  }

  if(finished && write_final_output)
  {
    total_time += global_timer.wall_time();

    pcout << std::endl << "... done!" << std::endl;

    analyze_computing_times();
  }

  return finished;
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntExplRK<dim, fe_degree, value_type>::prepare_vectors_for_next_timestep()
{
  // solution at t_n+1 -> solution at t_n
  solution_n.swap(solution_np);
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntExplRK<dim, fe_degree, value_type>::output_solver_info_header()
{
  // write output
  if(this->time_step_number % this->param.output_solver_info_every_timesteps == 0)
  {
    pcout << std::endl
          << "______________________________________________________________________" << std::endl
          << std::endl
          << " Number of TIME STEPS: " << std::left << std::setw(8) << this->time_step_number
          << "t_n = " << std::scientific << std::setprecision(4) << this->time
          << " -> t_n+1 = " << this->time + this->time_step << std::endl
          << "______________________________________________________________________" << std::endl
          << std::endl;
  }
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntExplRK<dim, fe_degree, value_type>::output_remaining_time()
{
  // write output
  if(this->time_step_number % this->param.output_solver_info_every_timesteps == 0)
  {
    if(time > param.start_time)
    {
      double const remaining_time =
        global_timer.wall_time() * (param.end_time - time) / (time - param.start_time);
      pcout << std::endl
            << "Estimated time until completion is " << remaining_time << " s / "
            << remaining_time / 3600. << " h." << std::endl;
    }
  }
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntExplRK<dim, fe_degree, value_type>::solve_timestep()
{
  Timer timer;
  timer.restart();

  rk_time_integrator->solve_timestep(solution_np, solution_n, time, time_step);

  // write output
  if(this->time_step_number % this->param.output_solver_info_every_timesteps == 0)
  {
    pcout << std::endl
          << "Solve time step explicitly: Wall time in [s] = " << std::scientific
          << timer.wall_time() << std::endl;
  }
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntExplRK<dim, fe_degree, value_type>::postprocessing() const
{
  conv_diff_operation->do_postprocessing(solution_n, time, time_step_number);
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntExplRK<dim, fe_degree, value_type>::analyze_computing_times() const
{
  pcout << std::endl
        << "_________________________________________________________________________________"
        << std::endl
        << std::endl
        << "Computing times:          min        avg        max        rel      p_min  p_max "
        << std::endl;

  Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(this->total_time, MPI_COMM_WORLD);
  pcout << "  Global time:         " << std::scientific << std::setprecision(4) << std::setw(10)
        << data.min << " " << std::setprecision(4) << std::setw(10) << data.avg << " "
        << std::setprecision(4) << std::setw(10) << data.max << " "
        << "          "
        << "  " << std::setw(6) << std::left << data.min_index << " " << std::setw(6) << std::left
        << data.max_index << std::endl
        << "_________________________________________________________________________________"
        << std::endl
        << std::endl;
}

} // namespace ConvDiff

#endif /* INCLUDE_CONVECTION_DIFFUSION_TIME_INT_EXPLICIT_RUNGE_KUTTA_H_ */
