/*
 * TimeIntBDF.h
 *
 *  Created on: Jun 10, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_TIMEINTBDF_H_
#define INCLUDE_TIMEINTBDF_H_


#include "TimeStepCalculation.h"
#include "Restart.h"

template<int dim> class PostProcessor;

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
class TimeIntBDF
{
public:
  TimeIntBDF(std_cxx11::shared_ptr<DGNavierStokesBase<dim, fe_degree,
               fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> > ns_operation_in,
             std_cxx11::shared_ptr<PostProcessor<dim> >             postprocessor_in,
             InputParameters const                                  &param_in,
             unsigned int const                                     n_refine_time_in)
    :
    n_refine_time(n_refine_time_in),
    ns_operation(ns_operation_in),
    postprocessor(postprocessor_in),
    param(param_in),
    total_time(0.0),
    time(param.start_time),
    time_step_number(1),
    order(param.order_time_integrator),
    time_steps(order),
    cfl(param.cfl/std::pow(2.0,n_refine_time)),
    gamma0(1.0),
    alpha(order),
    beta(order)
  {}

  virtual ~TimeIntBDF(){}

  void setup(bool do_restart);

  void timeloop();

  virtual void analyze_computing_times() const = 0;

private:
  virtual void setup_derived() = 0;

  virtual void initialize_vectors() = 0;
  virtual void initialize_current_solution() = 0;
  virtual void initialize_former_solution() = 0;
  void initialize_solution(bool do_restart);

  void resume_from_restart();
  void write_restart();
  virtual void read_restart_vectors(boost::archive::binary_iarchive & ia) = 0;
  virtual void write_restart_vectors(boost::archive::binary_oarchive & oa) = 0;

  void initialize_time_integrator_constants();
  void update_time_integrator_constants();
  void set_time_integrator_constants (unsigned int const current_order);
  void set_adaptive_time_integrator_constants(unsigned int const current_order);
  void set_alpha_and_beta (std::vector<value_type> const &alpha_local,
                           std::vector<value_type> const &beta_local);
  void check_time_integrator_constants (unsigned int const number) const;

  void calculate_time_step();
  void recalculate_adaptive_time_step();

  virtual void solve_timestep() = 0;
  virtual void postprocessing() const = 0;

  virtual void prepare_vectors_for_next_timestep() = 0;

  virtual parallel::distributed::Vector<value_type> const & get_velocity() = 0;

  unsigned int const n_refine_time;

protected:
  std_cxx11::shared_ptr<DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> > ns_operation;

  std_cxx11::shared_ptr<PostProcessor<dim> > postprocessor;
  InputParameters const & param;

  Timer global_timer;
  value_type total_time;

  value_type time;
  unsigned int time_step_number;
  unsigned int const order;
  std::vector<value_type> time_steps;

  value_type const cfl;
  value_type gamma0;
  std::vector<value_type> alpha, beta;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDF<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
setup(bool do_restart)
{
  // initialize time integrator constants assuming that the time integrator uses a high-order method in first time step,
  // i.e., the default case is start_with_low_order = false
  initialize_time_integrator_constants();

  // initialize global solution vectors (allocation)
  initialize_vectors();

  initialize_solution(do_restart);

  // set the parameters that NavierStokesOperation depends on
  ns_operation->set_time(time);
  ns_operation->set_time_step(time_steps[0]);
  ns_operation->set_gamma0(gamma0);

  // this is where the setup of deriving classes is performed
  setup_derived();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDF<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
initialize_solution(bool do_restart)
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

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDF<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
resume_from_restart()
{

  const std::string filename = restart_filename(param);
  std::ifstream in (filename.c_str());
  check_file(in, filename);
  boost::archive::binary_iarchive ia (in);
  resume_restart<dim,value_type>(ia, param, time, postprocessor, time_steps, order);

  read_restart_vectors(ia);

  finished_reading_restart_output();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDF<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
write_restart()
{
  const double EPSILON = 1.0e-10; // small number which is much smaller than the time step size

  const double wall_time = global_timer.wall_time();
  if( std::fmod(time ,param.restart_interval_time) < time_steps[0] + EPSILON
   || std::fmod(wall_time, param.restart_interval_wall_time) < wall_time-total_time
   || time_step_number % param.restart_interval_step == 0)
  {
    std::ostringstream oss;

    boost::archive::binary_oarchive oa(oss);
    write_restart_preamble<value_type>(oa, param, time_steps, time, postprocessor->get_output_counter(), order);
    write_restart_vectors(oa);
    write_restart_file(oss, param);

  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDF<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
initialize_time_integrator_constants()
{
  AssertThrow(order == 1 || order == 2 || order == 3,
      ExcMessage("Specified order of time integration scheme is not implemented."));

  // the default case is start_with_low_order == false
  // in case of start_with_low_order == true the time integrator constants have to be adjusted in do_timestep
  set_time_integrator_constants(order);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDF<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
calculate_time_step()
{
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << std::endl
              << "Temporal discretization:" << std::endl << std::endl
              << "  High order dual splitting scheme of temporal order " << param.order_time_integrator << std::endl << std::endl;


  if(param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepUserSpecified)
  {
    time_steps[0] = calculate_const_time_step(param.time_step_size,n_refine_time);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepCFL)
  {
    time_steps[0] = calculate_const_time_step_cfl<dim, fe_degree>(ns_operation->get_dof_handler_u().get_triangulation(),
                                                                  cfl,
                                                                  param.max_velocity,
                                                                  param.start_time,
                                                                  param.end_time);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::AdaptiveTimeStepCFL)
  {
    time_steps[0] = calculate_const_time_step_cfl<dim, fe_degree>(ns_operation->get_dof_handler_u().get_triangulation(),
                                                                  cfl,
                                                                  param.max_velocity,
                                                                  param.start_time,
                                                                  param.end_time);

    value_type adaptive_time_step = calculate_adaptive_time_step_cfl<dim, fe_degree, value_type>(ns_operation->get_data(),
                                                                                                 get_velocity(),
                                                                                                 cfl,
                                                                                                 time_steps[0],
                                                                                                 false);

    if(adaptive_time_step < time_steps[0])
      time_steps[0] = adaptive_time_step;
  }
  // fill time_steps array
  for(unsigned int i=1;i<order;++i)
    time_steps[i] = time_steps[0];

  AssertThrow(param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepUserSpecified ||
              param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepCFL ||
              param.calculation_of_time_step_size == TimeStepCalculation::AdaptiveTimeStepCFL,
              ExcMessage("User did not specify how to calculate time step size - possibilities are ConstTimeStepUserSpecified, ConstTimeStepCFL  and AdaptiveTimeStepCFL."));

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDF<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
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

  time_steps[0] = calculate_adaptive_time_step_cfl<dim, fe_degree, value_type>(ns_operation->get_data(),
                                                                               get_velocity(),
                                                                               cfl,
                                                                               time_steps[0]);
}


template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDF<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
timeloop()
{
  global_timer.restart();

  postprocessing();

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << std::endl << "Starting time loop ..." << std::endl;

  const value_type EPSILON = 1.0e-10; // epsilon is a small number which is much smaller than the time step size
  while(time<(param.end_time-EPSILON) && time_step_number<=param.max_number_of_steps)
  {
    update_time_integrator_constants();

    solve_timestep();

    prepare_vectors_for_next_timestep();

    time += time_steps[0];
    ++time_step_number;

    postprocessing();
    write_restart();

    if(param.calculation_of_time_step_size == TimeStepCalculation::AdaptiveTimeStepCFL)
       recalculate_adaptive_time_step();
  }

  total_time += global_timer.wall_time();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDF<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
set_time_integrator_constants (unsigned int const current_order)
{
  AssertThrow(current_order <= order,
      ExcMessage("There is a logical error when updating the time integrator constants."));

  unsigned int const MAX_ORDER = 3;
  std::vector<value_type> alpha_local(MAX_ORDER);
  std::vector<value_type> beta_local(MAX_ORDER);

  if(current_order == 1)   //BDF 1
  {
    gamma0 = 1.0;

    alpha_local[0] = 1.0;
    alpha_local[1] = 0.0;
    alpha_local[2] = 0.0;

    beta_local[0] = 1.0;
    beta_local[1] = 0.0;
    beta_local[2] = 0.0;
  }
  else if(current_order == 2) //BDF 2
  {
    gamma0 = 3.0/2.0;

    alpha_local[0] = 2.0;
    alpha_local[1] = -0.5;
    alpha_local[2] = 0.0;

    beta_local[0] = 2.0;
    beta_local[1] = -1.0;
    beta_local[2] = 0.0;
  }
  else if(current_order == 3) //BDF 3
  {
    gamma0 = 11./6.;

    alpha_local[0] = 3.;
    alpha_local[1] = -1.5;
    alpha_local[2] = 1./3.;

    beta_local[0] = 3.0;
    beta_local[1] = -3.0;
    beta_local[2] = 1.0;
  }

  set_alpha_and_beta(alpha_local,beta_local);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDF<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
set_adaptive_time_integrator_constants (unsigned int const current_order)
{
  AssertThrow(current_order <= order,
    ExcMessage("There is a logical error when updating the time integrator constants."));

  unsigned int const MAX_ORDER = 3;
  std::vector<value_type> alpha_local(MAX_ORDER);
  std::vector<value_type> beta_local(MAX_ORDER);

  if(current_order == 1)   // BDF 1
  {
    gamma0 = 1.0;

    alpha_local[0] = 1.0;
    alpha_local[1] = 0.0;
    alpha_local[2] = 0.0;

    beta_local[0] = 1.0;
    beta_local[1] = 0.0;
    beta_local[2] = 0.0;
  }
  else if(current_order == 2) // BDF 2
  {
    FullMatrix<value_type> coeffM_alpha(3);
    coeffM_alpha(0,0) = 1;
    coeffM_alpha(0,1) = -1;
    coeffM_alpha(0,2) = -1;
    coeffM_alpha(1,0) = 0;
    coeffM_alpha(1,1) = time_steps[0];
    coeffM_alpha(1,2) = time_steps[1] + time_steps[0];
    coeffM_alpha(2,0) = 0;
    coeffM_alpha(2,1) = time_steps[0]*time_steps[0];
    coeffM_alpha(2,2) = (time_steps[1] + time_steps[0])*(time_steps[1] + time_steps[0]);

    Vector<value_type> alphas(3);
    Vector<value_type> b_alpha(3);
    b_alpha(0) = 0;
    b_alpha(1) = time_steps[0];
    b_alpha(2) = 0;
    coeffM_alpha.gauss_jordan();
    coeffM_alpha.vmult(alphas,b_alpha);

    FullMatrix<value_type> coeffM_betha(2);
    coeffM_betha(0,0) = 1;
    coeffM_betha(0,1) = 1;
    coeffM_betha(1,0) = time_steps[0];
    coeffM_betha(1,1) = time_steps[1] + time_steps[0];

    Vector<value_type> bethas(2);
    Vector<value_type> b_bethas(2);
    b_bethas(0) = 1;
    b_bethas(1) = 0;
    coeffM_betha.gauss_jordan();
    coeffM_betha.vmult(bethas,b_bethas);

    gamma0 = alphas(0);
    alpha_local[0] = alphas(1);
    alpha_local[1] = alphas(2);
    alpha_local[2] = 0.0;
    beta_local[0] = bethas(0);
    beta_local[1] = bethas(1);
    beta_local[2] = 0.0;
  }
  else if(current_order == 3) // BDF 3
  {
    FullMatrix<value_type> coeffM_alpha(4);
    coeffM_alpha(0,0) = 1;
    coeffM_alpha(0,1) = -1;
    coeffM_alpha(0,2) = -1;
    coeffM_alpha(0,3) = -1;
    coeffM_alpha(1,0) = 0;
    coeffM_alpha(1,1) = time_steps[0];
    coeffM_alpha(1,2) = time_steps[1] + time_steps[0];
    coeffM_alpha(1,3) = time_steps[2] + time_steps[1] + time_steps[0];
    coeffM_alpha(2,0) = 0;
    coeffM_alpha(2,1) = time_steps[0]*time_steps[0];
    coeffM_alpha(2,2) = (time_steps[1] + time_steps[0])*(time_steps[1] + time_steps[0]);
    coeffM_alpha(2,3) = (time_steps[2] + time_steps[1] + time_steps[0])*(time_steps[2] + time_steps[1] + time_steps[0]);
    coeffM_alpha(3,0) = 0;
    coeffM_alpha(3,1) = time_steps[0]*time_steps[0]*time_steps[0];
    coeffM_alpha(3,2) = (time_steps[1] + time_steps[0])*(time_steps[1] + time_steps[0])*(time_steps[1] + time_steps[0]);
    coeffM_alpha(3,3) = (time_steps[2] + time_steps[1] + time_steps[0])*(time_steps[2] + time_steps[1] + time_steps[0])*(time_steps[2] + time_steps[1] + time_steps[0]);

    Vector<value_type> alphas(4);
    Vector<value_type> b_alpha(4);
    b_alpha(0) = 0;
    b_alpha(1) = time_steps[0];
    b_alpha(2) = 0;
    b_alpha(3) = 0;
    coeffM_alpha.gauss_jordan();
    coeffM_alpha.vmult(alphas,b_alpha);

    FullMatrix<value_type> coeffM_betha(3);
    coeffM_betha(0,0) = 1;
    coeffM_betha(0,1) = 1;
    coeffM_betha(0,2) = 1;
    coeffM_betha(1,0) = time_steps[0];
    coeffM_betha(1,1) = time_steps[1] + time_steps[0];
    coeffM_betha(1,2) = time_steps[2] + time_steps[1] + time_steps[0];
    coeffM_betha(2,0) = time_steps[0]*time_steps[0];
    coeffM_betha(2,1) = (time_steps[1] + time_steps[0])*(time_steps[1] + time_steps[0]);
    coeffM_betha(2,2) = (time_steps[2] + time_steps[1] + time_steps[0])*(time_steps[2] + time_steps[1] + time_steps[0]);

    Vector<value_type> bethas(3);
    Vector<value_type> b_bethas(3);
    b_bethas(0) = 1;
    b_bethas(1) = 0;
    b_bethas(2) = 0;
    coeffM_betha.gauss_jordan();
    coeffM_betha.vmult(bethas,b_bethas);

    gamma0 = alphas(0);
    alpha_local[0] = alphas(1);
    alpha_local[1] = alphas(2);
    alpha_local[2] = alphas(3);
    beta_local[0] = bethas(0);
    beta_local[1] = bethas(1);
    beta_local[2] = bethas(2);
  }
  set_alpha_and_beta(alpha_local,beta_local);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDF<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
set_alpha_and_beta (std::vector<value_type> const &alpha_local,
                    std::vector<value_type> const &beta_local)
{
  AssertThrow((alpha.size() <= alpha_local.size()) && (beta.size() <= beta_local.size()),
      ExcMessage("There is a logical error when setting the time integrator constants. Probably, the specified order of the time integration scheme is not implemented."));

  for(unsigned int i=0;i<alpha.size();++i)
    alpha[i] = alpha_local[i];

  for(unsigned int i=0;i<beta.size();++i)
    beta[i] = beta_local[i];
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDF<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
update_time_integrator_constants()
{
  if(param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepUserSpecified ||
     param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepCFL)
  {
    // when starting the time integrator with a low order method, ensure that
    // the time integrator constants are set properly
    if(time_step_number <= order && param.start_with_low_order == true)
    {
      set_time_integrator_constants(time_step_number);
    }
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::AdaptiveTimeStepCFL)
  {
    // when starting the time integrator with a low order method, ensure that
    // the time integrator constants are set properly
    if(time_step_number <= order && param.start_with_low_order == true)
    {
      set_adaptive_time_integrator_constants(time_step_number);
    }
    else // otherwise, adjust time integrator constants since this is adaptive time stepping
    {
      set_adaptive_time_integrator_constants(order);
    }
  }
  // check_time_integrator_constants(time_step_number);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDF<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
check_time_integrator_constants(unsigned int current_time_step_number) const
{
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << "Time integrator constants: time step " << current_time_step_number << std::endl;

    std::cout << "Gamma0 = " << gamma0   << std::endl;

    for(unsigned int i=0;i<order;++i)
      std::cout << "Alpha[" << i <<"] = " << alpha[i] << std::endl;

    for(unsigned int i=0;i<order;++i)
      std::cout << "Beta[" << i <<"] = " << beta[i] << std::endl;
  }
}

#endif /* INCLUDE_TIMEINTBDF_H_ */
