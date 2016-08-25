/*
 * TimeIntBDFConvDiff.h
 *
 *  Created on: Aug 22, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_TIMEINTBDFCONVDIFF_H_
#define INCLUDE_TIMEINTBDFCONVDIFF_H_

#include "../include/TimeIntBDFBase.h"
#include "../include/PushBackVectors.h"

template<int dim, int fe_degree, typename value_type>
class TimeIntBDFConvDiff : public TimeIntBDFBase
{
public:
  TimeIntBDFConvDiff(std_cxx11::shared_ptr<DGConvDiffOperation<dim, fe_degree, value_type> > conv_diff_operation_in,
                     std_cxx11::shared_ptr<PostProcessor<dim, fe_degree> >                   postprocessor_in,
                     ConvDiff::InputParametersConvDiff const                                 &param_in,
                     std_cxx11::shared_ptr<Function<dim> >                                   velocity_in,
                     unsigned int const                                                      n_refine_time_in)
    :
    TimeIntBDFBase(param_in.order_time_integrator,
                   param_in.start_with_low_order,
                   false), // false: currently no adaptive time stepping implemented
    conv_diff_operation(conv_diff_operation_in),
    postprocessor(postprocessor_in),
    param(param_in),
    velocity(velocity_in),
    n_refine_time(n_refine_time_in),
    total_time(0.0),
    time(param.start_time),
    time_steps(this->order),
    solution(this->order),
    N_iter_average(0.0),
    solver_time_average(.0)
  {}

  virtual ~TimeIntBDFConvDiff(){}

  virtual void setup(bool do_restart=0);

  virtual void timeloop();

private:
  void initialize_vectors();
  void initialize_solution();
  void calculate_timestep();
  void prepare_vectors_for_next_timestep();
  void solve_timestep();
  void postprocessing() const;
  void analyze_computing_times() const;

  std_cxx11::shared_ptr<DGConvDiffOperation<dim, fe_degree, value_type> > conv_diff_operation;
  std_cxx11::shared_ptr<PostProcessor<dim, fe_degree> > postprocessor;
  ConvDiff::InputParametersConvDiff const & param;
  std_cxx11::shared_ptr<Function<dim> > velocity;

  unsigned int const n_refine_time;

  Timer global_timer;
  double total_time;

  double time;
  std::vector<double> time_steps;

  parallel::distributed::Vector<value_type> solution_np;
  std::vector<parallel::distributed::Vector<value_type> > solution;

  parallel::distributed::Vector<value_type> sum_alphai_ui;
  parallel::distributed::Vector<value_type> rhs_vector;

  double N_iter_average;
  double solver_time_average;
};

template<int dim, int fe_degree, typename value_type>
void TimeIntBDFConvDiff<dim,fe_degree,value_type>::
setup(bool /*do_restart*/)
{
  ConditionalOStream pcout(std::cout,
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Setup time integrator ..." << std::endl;

  // call function of base class to initialize the time integrator constants
  this->initialize_time_integrator_constants();

  // initialize global solution vectors (allocation)
  initialize_vectors();

  // calculate time step size before initializing the solution because
  // initialization of solution depends on the time step size
  calculate_timestep();

  // initializes the solution by interpolation of analytical solution
  initialize_solution();

  // set the parameters that DGConvDiffOperation depends on
  conv_diff_operation->set_scaling_factor_time_derivative_term(gamma0/time_steps[0]);

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int fe_degree, typename value_type>
void TimeIntBDFConvDiff<dim,fe_degree,value_type>::
initialize_vectors()
{
  for(unsigned int i=0;i<solution.size();++i)
    conv_diff_operation->initialize_dof_vector(solution[i]);

  conv_diff_operation->initialize_dof_vector(solution_np);

  conv_diff_operation->initialize_dof_vector(sum_alphai_ui);
  conv_diff_operation->initialize_dof_vector(rhs_vector);
}

template<int dim, int fe_degree, typename value_type>
void TimeIntBDFConvDiff<dim,fe_degree,value_type>::
initialize_solution()
{
  for(unsigned int i=0;i<solution.size();++i)
    conv_diff_operation->prescribe_initial_conditions(solution[i],time - double(i)*time_steps[0]);
}

template<int dim, int fe_degree, typename value_type>
void TimeIntBDFConvDiff<dim,fe_degree,value_type>::
calculate_timestep()
{
  ConditionalOStream pcout(std::cout,
       Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Calculation of time step size:" << std::endl << std::endl;

  if(param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepUserSpecified)
  {
    time_steps[0] = calculate_const_time_step(param.time_step_size,n_refine_time);

    print_parameter(pcout,"time step size",time_steps[0]);
  }
  else
  {
    AssertThrow(param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepUserSpecified,
                ExcMessage("Specified calculation of time step size not implemented for BDF time integrator!"));
  }

  // fill time_steps array
  for(unsigned int i=1;i<order;++i)
    time_steps[i] = time_steps[0];
}


template<int dim, int fe_degree, typename value_type>
void TimeIntBDFConvDiff<dim,fe_degree,value_type>::
timeloop()
{
  ConditionalOStream pcout(std::cout,
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Starting time loop ..." << std::endl;

  global_timer.restart();

  postprocessing();

  const double EPSILON = 1.0e-10;
  while(time<(param.end_time-EPSILON))
  {
    this->update_time_integrator_constants();

    solve_timestep();

    prepare_vectors_for_next_timestep();

    time += time_steps[0];
    ++time_step_number;

    postprocessing();

    // currently no write_restart implemented

    // currently no adaptive time stepping implemented
  }

  total_time += global_timer.wall_time();

  pcout << std::endl << "... done!" << std::endl;

  analyze_computing_times();
}

template<int dim, int fe_degree, typename value_type>
void TimeIntBDFConvDiff<dim,fe_degree,value_type>::
postprocessing() const
{
  postprocessor->do_postprocessing(solution[0],time);
}

template<int dim, int fe_degree, typename value_type>
void TimeIntBDFConvDiff<dim,fe_degree,value_type>::
prepare_vectors_for_next_timestep()
{
  push_back(solution);

  solution[0].swap(solution_np);
}

template<int dim, int fe_degree, typename value_type>
void TimeIntBDFConvDiff<dim,fe_degree,value_type>::
solve_timestep()
{
  // write output
  if(this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << std::endl
          << "______________________________________________________________________"
          << std::endl
          << std::endl
          << " Number of TIME STEPS: " << std::left << std::setw(8) << this->time_step_number
          << "t_n = " << std::scientific << std::setprecision(4) << this->time
          << " -> t_n+1 = " << this->time + this->time_steps[0] << std::endl
          << "______________________________________________________________________"
          << std::endl << std::endl;
  }

  Timer timer;
  timer.restart();

  // calculate sum (alpha_i/dt * u_i)
  sum_alphai_ui.equ(this->alpha[0]/this->time_steps[0],solution[0]);
  for (unsigned int i=1;i<solution.size();++i)
    sum_alphai_ui.add(this->alpha[i]/this->time_steps[0],solution[i]);

  // calculate rhs
  conv_diff_operation->rhs(rhs_vector,&sum_alphai_ui,this->time+this->time_steps[0]);

  // extrapolate old solution to obtain a good initial guess for the solver
  solution_np.equ(this->beta[0],solution[0]);
  for(unsigned int i=1;i<solution.size();++i)
    solution_np.add(this->beta[i],solution[i]);

  unsigned int iterations = conv_diff_operation->solve(solution_np,
                                                       rhs_vector,
                                                       this->gamma0/this->time_steps[0],
                                                       this->time + this->time_steps[0]);

  N_iter_average += iterations;
  solver_time_average += timer.wall_time();

  // write output
  if(this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << "Solve linear convection-diffusion problem:" << std::endl
          << "  Iterations: " << std::setw(6) << std::right << iterations
          << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
  }
}

template<int dim, int fe_degree, typename value_type>
void TimeIntBDFConvDiff<dim,fe_degree,value_type>::
analyze_computing_times() const
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl
        << "_________________________________________________________________________________"
        << std::endl
        << std::endl
        << "Computing times:          min        avg        max        rel      p_min  p_max"
        << std::endl;

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


#endif /* INCLUDE_TIMEINTBDFCONVDIFF_H_ */
