/*
 * TimeIntExplRKConvDiff.h
 *
 *  Created on: Aug 2, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_TIMEINTEXPLRKCONVDIFF_H_
#define INCLUDE_TIMEINTEXPLRKCONVDIFF_H_

#include "../include/TimeStepCalculation.h"
#include "../include/InputParametersConvDiff.h"
#include "../include/PrintFunctions.h"

template<int dim, int fe_degree> class PostProcessor;

template<int dim, int fe_degree, typename value_type>
class TimeIntExplRKConvDiff
{
public:
  TimeIntExplRKConvDiff(std_cxx11::shared_ptr<DGConvDiffOperation<dim, fe_degree, value_type> > conv_diff_operation_in,
                        std_cxx11::shared_ptr<PostProcessor<dim, fe_degree> >                   postprocessor_in,
                        ConvDiff::InputParametersConvDiff const                                 &param_in,
                        std_cxx11::shared_ptr<Function<dim> >                                   velocity_in,
                        unsigned int const                                                      n_refine_time_in)
    :
    conv_diff_operation(conv_diff_operation_in),
    postprocessor(postprocessor_in),
    param(param_in),
    velocity(velocity_in),
    total_time(0.0),
    time(param.start_time),
    time_step(1.0),
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

  std_cxx11::shared_ptr<DGConvDiffOperation<dim, fe_degree, value_type> > conv_diff_operation;
  std_cxx11::shared_ptr<PostProcessor<dim, fe_degree> > postprocessor;
  ConvDiff::InputParametersConvDiff const & param;
  std_cxx11::shared_ptr<Function<dim> > velocity;

  Timer global_timer;
  double total_time;

  parallel::distributed::Vector<value_type> solution_n, solution_np;

  parallel::distributed::Vector<value_type> vec_rhs, vec_temp;

  double time, time_step;
  unsigned int const order;
  unsigned int const n_refine_time;
  double const cfl_number;
  double const diffusion_number;

};

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRKConvDiff<dim,fe_degree,value_type>::setup()
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Setup time integrator ..." << std::endl;

  // initialize global solution vectors (allocation)
  initialize_vectors();

  // initializes the solution by interpolation of analytical solution
  initialize_solution();

  // calculate time step size
  calculate_timestep();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRKConvDiff<dim,fe_degree,value_type>::initialize_vectors()
{
  conv_diff_operation->initialize_dof_vector(solution_n);
  conv_diff_operation->initialize_dof_vector(solution_np);

  if(order >= 2)
    conv_diff_operation->initialize_dof_vector(vec_rhs);
  if(order >= 3)
    conv_diff_operation->initialize_dof_vector(vec_temp);
}

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRKConvDiff<dim,fe_degree,value_type>::initialize_solution()
{
  conv_diff_operation->prescribe_initial_conditions(solution_n,time);
}

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRKConvDiff<dim,fe_degree,value_type>::calculate_timestep()
{
  ConditionalOStream pcout(std::cout,
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
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
        calculate_min_cell_diameter(conv_diff_operation->get_data().get_dof_handler().get_triangulation());

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
  else if(param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepCFLAndDiffusion)
  {
    AssertThrow(param.equation_type == ConvDiff::EquationType::ConvectionDiffusion,
                ExcMessage("Time step calculation ConstTimeStepCFLAndDiffusion does not make sense!"));

    // calculate minimum vertex distance
    const double global_min_cell_diameter =
        calculate_min_cell_diameter(conv_diff_operation->get_data().get_dof_handler().get_triangulation());

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
  else
  {
    AssertThrow(param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepUserSpecified ||
                param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepCFL ||
                param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepCFLAndDiffusion,
                ExcMessage("Specified calculation of time step size is not implemented!"));
  }
}


template<int dim, int fe_degree, typename value_type>
void TimeIntExplRKConvDiff<dim,fe_degree,value_type>::timeloop()
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Starting time loop ..." << std::endl;

  global_timer.restart();

  postprocessing();

  const double EPSILON = 1.0e-10;
  while(time<(param.end_time-EPSILON))
  {
    solve_timestep();

    prepare_vectors_for_next_timestep();

    time += time_step;

    postprocessing();
  }

  total_time += global_timer.wall_time();

  pcout << std::endl << "... done!" << std::endl;

  analyze_computing_times();
}

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRKConvDiff<dim,fe_degree,value_type>::postprocessing() const
{
  postprocessor->do_postprocessing(solution_n,time);
}

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRKConvDiff<dim,fe_degree,value_type>::
prepare_vectors_for_next_timestep()
{
  // solution at t_n+1 -> solution at t_n
  solution_n.swap(solution_np);
}

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRKConvDiff<dim,fe_degree,value_type>::
solve_timestep()
{
  if(order == 1) // explicit Euler method
  {
    if(true)
    {
      conv_diff_operation->evaluate(solution_np,solution_n,time);
      solution_np *= time_step;
      solution_np.add(1.0,solution_n);
    }
  }
  else if(order == 2) // Runge-Kutta method of order 2
  {
    if(true)
    {
      // stage 1
      conv_diff_operation->evaluate(vec_rhs,solution_n,time);

      // stage 2
      vec_rhs *= time_step/2.;
      vec_rhs.add(1.0,solution_n);
      conv_diff_operation->evaluate(solution_np,vec_rhs,time + time_step/2.);
      solution_np *= time_step;
      solution_np.add(1.0,solution_n);
    }
  }
  else if(order == 3) //Heun's method of order 3
  {
    solution_np = solution_n;

    // stage 1
    conv_diff_operation->evaluate(vec_temp,solution_n,time);
    solution_np.add(1.*time_step/4.,vec_temp);

    // stage 2
    vec_rhs.equ(1.,solution_n);
    vec_rhs.add(time_step/3.,vec_temp);
    conv_diff_operation->evaluate(vec_temp,vec_rhs,time+time_step/3.);

    // stage 3
    vec_rhs.equ(1.,solution_n);
    vec_rhs.add(2.0*time_step/3.0,vec_temp);
    conv_diff_operation->evaluate(vec_temp,vec_rhs,time+2.*time_step/3.);
    solution_np.add(3.*time_step/4.,vec_temp);
  }
  else if(order == 4) //classical 4th order Runge-Kutta method
  {
    solution_np = solution_n;

    // stage 1
    conv_diff_operation->evaluate(vec_temp,solution_n,time);
    solution_np.add(time_step/6., vec_temp);

    // stage 2
    vec_rhs.equ(1.,solution_n);
    vec_rhs.add(time_step/2., vec_temp);
    conv_diff_operation->evaluate(vec_temp,vec_rhs,time+time_step/2.);
    solution_np.add(time_step/3., vec_temp);

    // stage 3
    vec_rhs.equ(1., solution_n);
    vec_rhs.add(time_step/2., vec_temp);
    conv_diff_operation->evaluate(vec_temp,vec_rhs,time+time_step/2.);
    solution_np.add(time_step/3., vec_temp);

    // stage 4
    vec_rhs.equ(1., solution_n);
    vec_rhs.add(time_step, vec_temp);
    conv_diff_operation->evaluate(vec_temp,vec_rhs,time+time_step);
    solution_np.add(time_step/6., vec_temp);
  }
  else
  {
    AssertThrow(order <= 4,ExcMessage("Explicit Runge-Kutta method only implemented for order <= 4!"));
  }
}

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRKConvDiff<dim,fe_degree,value_type>::
analyze_computing_times() const
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "_________________________________________________________________________________" << std::endl
        << std::endl << "Computing times:          min        avg        max        rel      p_min  p_max" << std::endl;

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

#endif /* INCLUDE_TIMEINTEXPLRKCONVDIFF_H_ */
