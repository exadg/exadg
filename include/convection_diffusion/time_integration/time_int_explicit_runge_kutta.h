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

template<int dim, int fe_degree> class PostProcessor;

template<int dim, int fe_degree, typename value_type>
class TimeIntExplRK
{
public:
  TimeIntExplRK(std::shared_ptr<ConvDiff::DGOperation<dim, fe_degree, value_type> > conv_diff_operation_in,
                std::shared_ptr<ConvDiff::PostProcessor<dim, fe_degree> >           postprocessor_in,
                ConvDiff::InputParameters const                                     &param_in,
                std::shared_ptr<Function<dim> >                                     velocity_in,
                unsigned int const                                                  n_refine_time_in)
    :
    conv_diff_operation(conv_diff_operation_in),
    postprocessor(postprocessor_in),
    param(param_in),
    velocity(velocity_in),
    total_time(0.0),
    pcout(std::cout,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    time(param.start_time),
    time_step(1.0),
    time_step_number(1),
    order(param.order_time_integrator),
    n_refine_time(n_refine_time_in),
    cfl_number(param.cfl_number/std::pow(2.0,n_refine_time)),
    diffusion_number(param.diffusion_number/std::pow(2.0,n_refine_time))
  {}

  void timeloop();

  void setup();

private:
  void initialize_vectors();
  void initialize_solution();
  void postprocessing() const;
  void solve_timestep();
  void prepare_vectors_for_next_timestep();
  void calculate_timestep();
  void analyze_computing_times() const;

  std::shared_ptr<ConvDiff::DGOperation<dim, fe_degree, value_type> > conv_diff_operation;

  std::shared_ptr<ExplicitRungeKuttaTimeIntegrator<
    ConvDiff::DGOperation<dim, fe_degree, value_type>,
    parallel::distributed::Vector<value_type> > > rk_time_integrator;

  std::shared_ptr<ConvDiff::PostProcessor<dim, fe_degree> > postprocessor;
  ConvDiff::InputParameters const & param;
  std::shared_ptr<Function<dim> > velocity;

  // timer
  Timer global_timer;
  double total_time;

  // screen output
  ConditionalOStream pcout;

  // solution vectors
  parallel::distributed::Vector<value_type> solution_n, solution_np;

  // current time and time step size
  double time, time_step;

  // the number of the current time step starting with time_step_number = 1
  unsigned int time_step_number;

  unsigned int const order;
  unsigned int const n_refine_time;
  double const cfl_number;
  double const diffusion_number;

};

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRK<dim,fe_degree,value_type>::setup()
{
  pcout << std::endl << "Setup time integrator ..." << std::endl;

  // initialize global solution vectors (allocation)
  initialize_vectors();

  // initializes the solution by interpolation of analytical solution
  initialize_solution();

  // calculate time step size
  calculate_timestep();

  // initialize Runge-Kutta time integrator
  rk_time_integrator.reset(new ExplicitRungeKuttaTimeIntegrator<
      ConvDiff::DGOperation<dim, fe_degree, value_type>,
      parallel::distributed::Vector<value_type> >(order,conv_diff_operation));

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRK<dim,fe_degree,value_type>::initialize_vectors()
{
  conv_diff_operation->initialize_dof_vector(solution_n);
  conv_diff_operation->initialize_dof_vector(solution_np);
}

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRK<dim,fe_degree,value_type>::initialize_solution()
{
  conv_diff_operation->prescribe_initial_conditions(solution_n,time);
}

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRK<dim,fe_degree,value_type>::calculate_timestep()
{
  pcout << std::endl << "Calculation of time step size:" << std::endl << std::endl;

  if(param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepUserSpecified)
  {
    time_step = calculate_const_time_step(param.time_step_size,n_refine_time);

    print_parameter(pcout,"time step size",time_step);
  }
  else if(param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepCFL)
  {
    AssertThrow(param.equation_type == ConvDiff::EquationType::Convection ||
                param.equation_type == ConvDiff::EquationType::ConvectionDiffusion,
                ExcMessage("Time step calculation ConstTimeStepCFL does not make sense!"));

    // calculate minimum vertex distance
    const double global_min_cell_diameter =
        calculate_minimum_vertex_distance(conv_diff_operation->get_data().get_dof_handler().get_triangulation());

    print_parameter(pcout,"h_min",global_min_cell_diameter);

    double time_step_conv = 1.0;

    const double max_velocity =
        calculate_max_velocity(conv_diff_operation->get_data().get_dof_handler().get_triangulation(),
                               velocity,
                               time);

    print_parameter(pcout,"U_max",max_velocity);
    print_parameter(pcout,"CFL",cfl_number);

    time_step_conv = calculate_const_time_step_cfl(cfl_number,
                                                   max_velocity,
                                                   global_min_cell_diameter,
                                                   fe_degree);

    // decrease time_step in order to exactly hit end_time
    time_step = (param.end_time-param.start_time)/(1+int((param.end_time-param.start_time)/time_step_conv));

    print_parameter(pcout,"Time step size (convection)",time_step);
  }
  else if(param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepDiffusion)
  {
    AssertThrow(param.equation_type == ConvDiff::EquationType::Diffusion ||
                param.equation_type == ConvDiff::EquationType::ConvectionDiffusion,
                ExcMessage("Time step calculation ConstTimeStepDiffusion does not make sense!"));

    // calculate minimum vertex distance
    const double global_min_cell_diameter =
        calculate_minimum_vertex_distance(conv_diff_operation->get_data().get_dof_handler().get_triangulation());

    print_parameter(pcout,"h_min",global_min_cell_diameter);

    print_parameter(pcout,"Diffusion number",diffusion_number);

    double time_step_diff = 1.0;
    // calculate time step according to Diffusion number condition
    time_step_diff = calculate_const_time_step_diff(diffusion_number,
                                                    param.diffusivity,
                                                    global_min_cell_diameter,
                                                    fe_degree);

    // decrease time_step in order to exactly hit end_time
    time_step = (param.end_time-param.start_time)/(1+int((param.end_time-param.start_time)/time_step_diff));

    print_parameter(pcout,"Time step size (diffusion)",time_step);
  }
  else if(param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepCFLAndDiffusion)
  {
    AssertThrow(param.equation_type == ConvDiff::EquationType::ConvectionDiffusion,
                ExcMessage("Time step calculation ConstTimeStepCFLAndDiffusion does not make sense!"));

    // calculate minimum vertex distance
    const double global_min_cell_diameter =
        calculate_minimum_vertex_distance(conv_diff_operation->get_data().get_dof_handler().get_triangulation());

    print_parameter(pcout,"h_min",global_min_cell_diameter);

    double time_step_conv = std::numeric_limits<double>::max();
    double time_step_diff = std::numeric_limits<double>::max();

    // calculate time step according to CFL condition
    const double max_velocity =
        calculate_max_velocity(conv_diff_operation->get_data().get_dof_handler().get_triangulation(),
                               velocity,
                               time);

    print_parameter(pcout,"U_max",max_velocity);
    print_parameter(pcout,"CFL",cfl_number);

    time_step_conv = calculate_const_time_step_cfl(cfl_number,
                                                   max_velocity,
                                                   global_min_cell_diameter,
                                                   fe_degree);

    print_parameter(pcout,"Time step size (convection)",time_step_conv);


    // calculate time step according to Diffusion number condition
    time_step_diff = calculate_const_time_step_diff(diffusion_number,
                                                    param.diffusivity,
                                                    global_min_cell_diameter,
                                                    fe_degree);

    print_parameter(pcout,"Diffusion number",diffusion_number);
    print_parameter(pcout,"Time step size (diffusion)",time_step_diff);

    //adopt minimum time step size
    time_step = time_step_diff < time_step_conv ? time_step_diff : time_step_conv;

    // decrease time_step in order to exactly hit end_time
    time_step = (param.end_time-param.start_time)/(1+int((param.end_time-param.start_time)/time_step));

    print_parameter(pcout,"Time step size (combined)",time_step);
  }
  else if(param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepMaxEfficiency)
  {
    // calculate minimum vertex distance
    const double global_min_cell_diameter =
        calculate_minimum_vertex_distance(conv_diff_operation->get_data().get_dof_handler().get_triangulation());

    double time_step_tmp = calculate_time_step_max_efficiency(param.c_eff,
                                                              global_min_cell_diameter,
                                                              fe_degree,
                                                              order,
                                                              n_refine_time);

    // decrease time_step in order to exactly hit end_time
    time_step = (param.end_time-param.start_time)/(1+int((param.end_time-param.start_time)/time_step_tmp));

    pcout << "Calculation of time step size (max efficiency):" << std::endl << std::endl;
    print_parameter(pcout,"C_eff",param.c_eff/std::pow(2,n_refine_time));
    print_parameter(pcout,"Time step size",time_step);
  }
  else
  {
    AssertThrow(param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepUserSpecified ||
                param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepCFL ||
                param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepDiffusion ||
                param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepCFLAndDiffusion ||
                param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepMaxEfficiency ,
                ExcMessage("Specified calculation of time step size is not implemented!"));
  }
}


template<int dim, int fe_degree, typename value_type>
void TimeIntExplRK<dim,fe_degree,value_type>::timeloop()
{
  pcout << std::endl << "Starting time loop ..." << std::endl;

  global_timer.restart();

  postprocessing();

  const double EPSILON = 1.0e-10;
  while(time<(param.end_time-EPSILON))
  {
    solve_timestep();

    prepare_vectors_for_next_timestep();

    time += time_step;
    ++time_step_number;

    postprocessing();
  }

  total_time += global_timer.wall_time();

  pcout << std::endl << "... done!" << std::endl;

  analyze_computing_times();
}

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRK<dim,fe_degree,value_type>::postprocessing() const
{
  postprocessor->do_postprocessing(solution_n,time,0 /*use a value >=0 for unsteady problems*/);
}

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRK<dim,fe_degree,value_type>::
prepare_vectors_for_next_timestep()
{
  // solution at t_n+1 -> solution at t_n
  solution_n.swap(solution_np);
}

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRK<dim,fe_degree,value_type>::
solve_timestep()
{
  // write output
  if(this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
  {
    pcout << std::endl
          << "______________________________________________________________________"
          << std::endl
          << std::endl
          << " Number of TIME STEPS: " << std::left << std::setw(8) << this->time_step_number
          << "t_n = " << std::scientific << std::setprecision(4) << this->time
          << " -> t_n+1 = " << this->time + this->time_step << std::endl
          << "______________________________________________________________________"
          << std::endl << std::endl;
  }

  Timer timer;
  timer.restart();

  rk_time_integrator->solve_timestep(solution_np,
                                     solution_n,
                                     time,
                                     time_step);

  // write output
  if(this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
  {
    pcout << std::endl
          << "Solve time step explicitly: Wall time in [s] = " << std::scientific << timer.wall_time() << std::endl;

    if(time > param.start_time)
    {
      double const remaining_time = global_timer.wall_time() * (param.end_time-time)/(time-param.start_time);
      pcout << std::endl
            << "Estimated time until completion is " << remaining_time << " s / " << remaining_time/3600. << " h."
            << std::endl;
    }
  }
}

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRK<dim,fe_degree,value_type>::
analyze_computing_times() const
{
  pcout << std::endl
        << "_________________________________________________________________________________" << std::endl << std::endl
        << "Computing times:          min        avg        max        rel      p_min  p_max " << std::endl;

  Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg (this->total_time, MPI_COMM_WORLD);
  pcout  << "  Global time:         " << std::scientific
         << std::setprecision(4) << std::setw(10) << data.min << " "
         << std::setprecision(4) << std::setw(10) << data.avg << " "
         << std::setprecision(4) << std::setw(10) << data.max << " "
         << "          " << "  "
         << std::setw(6) << std::left << data.min_index << " "
         << std::setw(6) << std::left << data.max_index << std::endl
         << "_________________________________________________________________________________"
         << std::endl << std::endl;
}

} // namespace ConvDiff

#endif /* INCLUDE_CONVECTION_DIFFUSION_TIME_INT_EXPLICIT_RUNGE_KUTTA_H_ */
