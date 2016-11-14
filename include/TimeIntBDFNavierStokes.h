/*
 * TimeIntBDF.h
 *
 *  Created on: Jun 10, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_TIMEINTBDFNAVIERSTOKES_H_
#define INCLUDE_TIMEINTBDFNAVIERSTOKES_H_

#include "../include/TimeIntBDFBase.h"

#include "TimeStepCalculation.h"
#include "Restart.h"

template<int dim> class PostProcessorBase;

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
class TimeIntBDFNavierStokes : public TimeIntBDFBase
{
public:
  TimeIntBDFNavierStokes(std_cxx11::shared_ptr<NavierStokesOperation>   navier_stokes_operation_in,
                         std_cxx11::shared_ptr<PostProcessorBase<dim> > postprocessor_in,
                         InputParametersNavierStokes<dim> const         &param_in,
                         unsigned int const                             n_refine_time_in,
                         bool const                                     use_adaptive_time_stepping)
    :
    TimeIntBDFBase(param_in.order_time_integrator,
                   param_in.start_with_low_order,
                   use_adaptive_time_stepping),
    postprocessor(postprocessor_in),
    param(param_in),
    total_time(0.0),
    time(param.start_time),
    cfl(param.cfl/std::pow(2.0,n_refine_time_in)),
    pcout(std::cout,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    n_refine_time(n_refine_time_in),
    navier_stokes_operation(navier_stokes_operation_in)
  {}

  virtual ~TimeIntBDFNavierStokes(){}

  void setup(bool do_restart);

  void timeloop();

  virtual void analyze_computing_times() const = 0;

protected:
  std_cxx11::shared_ptr<PostProcessorBase<dim> > postprocessor;

  virtual void calculate_time_step();

  virtual void read_restart_vectors(boost::archive::binary_iarchive & ia) = 0;
  virtual void write_restart_vectors(boost::archive::binary_oarchive & oa) const = 0;
  virtual void resume_from_restart();
  void write_restart() const;

  InputParametersNavierStokes<dim> const & param;

  Timer global_timer;
  double total_time;

  double time;

  double const cfl;

  ConditionalOStream pcout;

private:
  virtual void setup_derived() = 0;

  virtual void initialize_vectors() = 0;
  virtual void initialize_current_solution() = 0;
  virtual void initialize_former_solution() = 0;
  void initialize_solution_and_calculate_timestep(bool do_restart);


  virtual void recalculate_adaptive_time_step();

  virtual void solve_timestep() = 0;
  virtual void postprocessing() const = 0;

  virtual void prepare_vectors_for_next_timestep() = 0;

  virtual parallel::distributed::Vector<value_type> const & get_velocity() = 0;

  unsigned int const n_refine_time;

  std_cxx11::shared_ptr<NavierStokesOperation> navier_stokes_operation;
};

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
setup(bool do_restart)
{
  pcout << std::endl << "Setup time integrator ..."
        << std::endl << std::endl;

  AssertThrow(param.problem_type == ProblemType::Unsteady,
              ExcMessage("In order to apply the BDF time integration scheme "
                         "the problem_type has to be ProblemType::Unsteady !"));

  // initialize time integrator constants assuming that the time integrator
  // uses a high-order method in first time step, i.e., the default case is
  // start_with_low_order = false. This is reasonable since DGNavierStokes
  // uses these time integrator constants for the setup of solvers.
  // in case of start_with_low_order == true the time integrator constants
  // have to be adjusted in timeloop().
  initialize_time_integrator_constants();

  // initialize global solution vectors (allocation)
  initialize_vectors();

  // initializes the solution and calculates the time step size!
  initialize_solution_and_calculate_timestep(do_restart);

  // set the parameters that NavierStokesOperation depends on
  navier_stokes_operation->set_time_step(time_steps[0]);
  navier_stokes_operation->set_scaling_factor_time_derivative_term(gamma0/time_steps[0]);

  // this is where the setup of deriving classes is performed
  setup_derived();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
initialize_solution_and_calculate_timestep(bool do_restart)
{
  if(do_restart)
  {
    resume_from_restart();

    // if anything in the temporal discretization is changed, start_with_low_order has to be set to true
    // otherwise the old solutions would not fit the time step increments, etc.
    if(param.start_with_low_order)
      calculate_time_step();
  }
  else
  {
    // when using time step adaptivity the time_step depends on the velocity field. Therefore, first prescribe
    // initial conditions before calculating the time step size
    initialize_current_solution();

    // initializing the solution at former time instants, e.g. t = start_time - time_step, requires the time step size.
    // Therefore, first calculate the time step size
    calculate_time_step();

    // now: prescribe initial conditions at former time instants t = time - time_step, time - 2.0*time_step, etc.
    if(param.start_with_low_order == false)
      initialize_former_solution();
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
resume_from_restart()
{

  const std::string filename = restart_filename<dim>(param);
  std::ifstream in (filename.c_str());
  check_file(in, filename);
  boost::archive::binary_iarchive ia (in);
  resume_restart<dim,value_type>
      (ia, param, time, postprocessor, time_steps, order);

  read_restart_vectors(ia);

  finished_reading_restart_output();
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
write_restart() const
{
  const double EPSILON = 1.0e-10; // small number which is much smaller than the time step size

  const double wall_time = global_timer.wall_time();
  if( (std::fmod(time ,param.restart_interval_time) < time_steps[0] + EPSILON && time > param.restart_interval_time - EPSILON)
      || (std::fmod(wall_time, param.restart_interval_wall_time) < wall_time-total_time)
      || (time_step_number % param.restart_every_timesteps == 0))
  {
    std::ostringstream oss;

    boost::archive::binary_oarchive oa(oss);
    // TODO
//    write_restart_preamble<dim, value_type>(oa, param, time_steps, time, postprocessor->get_output_counter(), order);
    write_restart_vectors(oa);
    write_restart_file<dim>(oss, param);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
calculate_time_step()
{
  if(param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepUserSpecified)
  {
    time_steps[0] = calculate_const_time_step(param.time_step_size,n_refine_time);

    pcout << "User specified time step size:" << std::endl << std::endl;
    print_parameter(pcout,"time step size",time_steps[0]);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepCFL)
  {
    const double global_min_cell_diameter = calculate_min_cell_diameter(
        navier_stokes_operation->get_dof_handler_u().get_triangulation());

    double time_step = calculate_const_time_step_cfl(cfl,
                                                     param.max_velocity,
                                                     global_min_cell_diameter,
                                                     fe_degree_u);

    // decrease time_step in order to exactly hit end_time
    time_steps[0] = (param.end_time-param.start_time)/(1+int((param.end_time-param.start_time)/time_step));

    pcout << "Calculation of time step size according to CFL condition:" << std::endl << std::endl;

    print_parameter(pcout,"h_min",global_min_cell_diameter);
    print_parameter(pcout,"U_max",param.max_velocity);
    print_parameter(pcout,"CFL",cfl);
    print_parameter(pcout,"Time step size",time_steps[0]);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::AdaptiveTimeStepCFL)
  {
    time_steps[0] = calculate_adaptive_time_step_cfl<dim, fe_degree_u, value_type>(
        navier_stokes_operation->get_data(),
        navier_stokes_operation->get_dof_index_velocity(),
        navier_stokes_operation->get_quad_index_velocity_linear(),
        get_velocity(),
        cfl,
        time_steps[0],
        false);

    pcout << "Calculation of time step size according to adaptive CFL condition:" << std::endl << std::endl;

    print_parameter(pcout,"CFL",cfl);
    print_parameter(pcout,"Time step size",time_steps[0]);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepMaxEfficiency)
  {
    const double global_min_cell_diameter = calculate_min_cell_diameter(
        navier_stokes_operation->get_dof_handler_u().get_triangulation());

    double time_step = calculate_time_step_max_efficiency(param.c_eff,
                                                          global_min_cell_diameter,
                                                          fe_degree_u,
                                                          order,
                                                          n_refine_time);

    // decrease time_step in order to exactly hit end_time
    time_steps[0] = (param.end_time-param.start_time)/(1+int((param.end_time-param.start_time)/time_step));

    pcout << "Calculation of time step size (max efficiency):" << std::endl << std::endl;
    print_parameter(pcout,"C_eff",param.c_eff/std::pow(2,n_refine_time));
    print_parameter(pcout,"Time step size",time_steps[0]);
  }
  else
  {
    AssertThrow(param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepUserSpecified ||
                param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepCFL ||
                param.calculation_of_time_step_size == TimeStepCalculation::AdaptiveTimeStepCFL ||
                param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepMaxEfficiency,
                ExcMessage("User did not specify how to calculate time step size - "
                    "possibilities are ConstTimeStepUserSpecified, ConstTimeStepCFL  and AdaptiveTimeStepCFL."));
  }

  // fill time_steps array
  for(unsigned int i=1;i<order;++i)
    time_steps[i] = time_steps[0];
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
recalculate_adaptive_time_step()
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
   *                    dt[1]  <- dt[0] <- recalculate dt[0]
   *
   */

  for(unsigned int i=order-1;i>0;--i)
    time_steps[i] = time_steps[i-1];

  time_steps[0] = calculate_adaptive_time_step_cfl<dim, fe_degree_u, value_type>(
      navier_stokes_operation->get_data(),
      navier_stokes_operation->get_dof_index_velocity(),
      navier_stokes_operation->get_quad_index_velocity_linear(),
      get_velocity(),
      cfl,
      time_steps[0]);

}


template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
timeloop()
{
  pcout << std::endl << "Starting time loop ..." << std::endl;

  global_timer.restart();

  postprocessing();

  const value_type EPSILON = 1.0e-10; // epsilon is a small number which is much smaller than the time step size
  while(time<(param.end_time-EPSILON) && time_step_number<=param.max_number_of_time_steps)
  {
    update_time_integrator_constants();

    solve_timestep();

    prepare_vectors_for_next_timestep();

    time += time_steps[0];
    ++time_step_number;

    postprocessing();

    if(param.write_restart == true)
      write_restart();

    if(adaptive_time_stepping == true)
      recalculate_adaptive_time_step();
  }

  total_time += global_timer.wall_time();

  pcout << std::endl << "... done!" << std::endl;

  analyze_computing_times();
}

#endif /* INCLUDE_TIMEINTBDFNAVIERSTOKES_H_ */
