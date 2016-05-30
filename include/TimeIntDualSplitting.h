/*
 * TimeIntDualSplitting.h
 *
 *  Created on: May 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_TIMEINTDUALSPLITTING_H_
#define INCLUDE_TIMEINTDUALSPLITTING_H_

#include "TimeStepCalculation.h"

template<int dim> class PostProcessor;

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
class TimeIntDualSplitting
{
public:
  static const unsigned int number_vorticity_components = (dim==2) ? 1 : dim;

  TimeIntDualSplitting(NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &ns_operation_in,
                       PostProcessor<dim> &postprocessor_in,
                       InputParameters const & param_in,
                       unsigned int const n_refine_time_in)
    :
    ns_operation(ns_operation_in),
    postprocessor(postprocessor_in),
    param(param_in),
    computing_times(5),
    total_time(0.0),
    order(param.order_time_integrator),
    time(param.start_time),
    time_steps(order),
    n_refine_time(n_refine_time_in),
    cfl(param.cfl/std::pow(2.0,n_refine_time)),
    time_step_number(1),
    gamma0(1.0),
    alpha(order),
    beta(order),
    velocity(order),
    pressure(order),
    vorticity(order),
    rhs_vec_convection(order)
  {}

  void setup();

  void timeloop();

  void analyze_computing_times() const;

private:
  void do_timestep();

  void initialize_time_integrator_constants();
  void update_time_integrator_constants();
  void set_time_integrator_constants (unsigned int const current_order);
  void set_adaptive_time_integrator_constants(unsigned int const current_order);
  void set_alpha_and_beta (std::vector<double> const &alpha_local,
                           std::vector<double> const &beta_local);
  void check_time_integrator_constants (unsigned int const number) const;

  void calculate_time_step();
  void recalculate_adaptive_time_step();

  void initialize_vectors();
  void initialize_current_solution();
  void initialize_former_solution();
  void initialize_vorticity();
  void initialize_rhs_vec_convection();

  void convective_step();
  void pressure_step();
  void projection_step();
  void viscous_step();
  void rhs_pressure (const parallel::distributed::BlockVector<value_type>  &src,
                     parallel::distributed::Vector<value_type>             &dst);

  void push_back_vectors();
  void push_back_solution();
  void push_back_vorticity();
  void push_back_rhs_vec_convection();

  NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &ns_operation;
  PostProcessor<dim> &postprocessor;
  InputParameters const & param;

  std::vector<double> computing_times;
  Timer global_timer;
  double total_time;

  unsigned int const order;
  double time;
  std::vector<double> time_steps;
  unsigned int const n_refine_time;
  double cfl;
  unsigned int time_step_number;

  double gamma0;
  std::vector<double> alpha, beta;

  parallel::distributed::BlockVector<value_type> velocity_np;
  std::vector<parallel::distributed::BlockVector<value_type> > velocity;

  parallel::distributed::Vector<value_type> pressure_np;
  std::vector<parallel::distributed::Vector<value_type> > pressure;

  parallel::distributed::BlockVector<value_type> vorticity_extrapolated;
  std::vector<parallel::distributed::BlockVector<value_type> > vorticity;

  std::vector<parallel::distributed::BlockVector<value_type> > rhs_vec_convection;

  parallel::distributed::BlockVector<value_type> rhs_vec_body_force;

  // solve convective step implicitly
  parallel::distributed::BlockVector<value_type> res_convection;
  parallel::distributed::BlockVector<value_type> delta_u_tilde;
  parallel::distributed::BlockVector<value_type> temp;

  parallel::distributed::Vector<value_type> rhs_vec_pressure;
  parallel::distributed::Vector<value_type> rhs_vec_pressure_temp;
  parallel::distributed::BlockVector<value_type> dummy;

  parallel::distributed::BlockVector<value_type> rhs_vec_projection;
  parallel::distributed::BlockVector<value_type> rhs_vec_viscous;

  // postprocessing: divergence of intermediate velocity u_hathat
  parallel::distributed::Vector<value_type> divergence;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
setup()
{
  initialize_time_integrator_constants();

  initialize_vectors();

  // when using time step adaptivity the time_step depends on the velocity field. Therefore, first prescribe
  // initial conditions before calculating the time step size
  initialize_current_solution();

  // initializing the solution at former time instants, e.g. t = start_time - time_step, requires the time step size.
  // Therefore, first calculate the time step size
  calculate_time_step();

  // now: prescribe initial conditions at former time instants t = time - time_step, time - 2.0*time_step, etc.
  if(param.start_with_low_order == false)
    initialize_former_solution();

  initialize_vorticity();

  if(param.solve_stokes_equations == false && param.start_with_low_order == false)
    initialize_rhs_vec_convection();

  // set the parameters that NavierStokesOperation depends on
  ns_operation.set_time(time);
  ns_operation.set_time_step(time_steps[0]);
  ns_operation.set_gamma0(gamma0);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
initialize_time_integrator_constants()
{
  AssertThrow(order == 1 || order == 2 || order == 3,
      ExcMessage("Specified order of time integration scheme is not implemented."));

  // the default case is start_with_low_order == false
  // in case of start_with_low_order == true the time integrator constants have to be adjusted in do_timestep
  set_time_integrator_constants(order);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
initialize_vectors()
{
  // velocity
  for(unsigned int i=0;i<velocity.size();++i)
    ns_operation.initialize_block_vector_velocity(velocity[i]);
  ns_operation.initialize_block_vector_velocity(velocity_np);

  // pressure
  for(unsigned int i=0;i<pressure.size();++i)
    ns_operation.initialize_vector_pressure(pressure[i]);
  ns_operation.initialize_vector_pressure(pressure_np);

  // vorticity
  for(unsigned int i=0;i<vorticity.size();++i)
    ns_operation.initialize_block_vector_vorticity(vorticity[i]);
  ns_operation.initialize_block_vector_vorticity(vorticity_extrapolated);

  // rhs_vec_convection
  if(param.solve_stokes_equations == false)
  {
    for(unsigned int i=0;i<rhs_vec_convection.size();++i)
      ns_operation.initialize_block_vector_velocity(rhs_vec_convection[i]);
  }

  // body force vector
  ns_operation.initialize_block_vector_velocity(rhs_vec_body_force);

  // implicit convective step
  if(param.convective_step_implicit == true)
  {
    ns_operation.initialize_block_vector_velocity(res_convection);
    ns_operation.initialize_block_vector_velocity(delta_u_tilde);
    ns_operation.initialize_block_vector_velocity(temp);
  }

  // rhs vector pressure
  ns_operation.initialize_vector_pressure(rhs_vec_pressure);
  ns_operation.initialize_vector_pressure(rhs_vec_pressure_temp);

  // rhs vector projection, viscous
  ns_operation.initialize_block_vector_velocity(rhs_vec_projection);
  ns_operation.initialize_block_vector_velocity(rhs_vec_viscous);

  // divergence
  if(param.compute_divergence == true)
  {
    ns_operation.initialize_scalar_vector_velocity(divergence);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
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
    time_steps[0] = calculate_const_time_step_cfl<dim, fe_degree>(ns_operation.get_dof_handler_u().get_triangulation(),
                                                                  cfl,
                                                                  param.max_velocity,
                                                                  param.start_time,
                                                                  param.end_time);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::AdaptiveTimeStepCFL)
  {
    time_steps[0] = calculate_const_time_step_cfl<dim, fe_degree>(ns_operation.get_dof_handler_u().get_triangulation(),
                                                                  cfl,
                                                                  param.max_velocity,
                                                                  param.start_time,
                                                                  param.end_time);

    double adaptive_time_step = calculate_adaptive_time_step_cfl<dim, fe_degree, value_type>(ns_operation.get_data(),
                                                                                             velocity[0],
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
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
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

  time_steps[0] = calculate_adaptive_time_step_cfl<dim, fe_degree, value_type>(ns_operation.get_data(),
                                                                               velocity[0],
                                                                               cfl,
                                                                               time_steps[0]);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
initialize_current_solution()
{
  ns_operation.prescribe_initial_conditions(velocity[0],pressure[0],time);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
initialize_former_solution()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i=1;i<velocity.size();++i)
    ns_operation.prescribe_initial_conditions(velocity[i],pressure[i],time - double(i)*time_steps[0]);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
initialize_vorticity()
{
  ns_operation.compute_vorticity(velocity[0],vorticity[0]);

  if(param.start_with_low_order == false)
  {
    for(unsigned int i=1;i<vorticity.size();++i)
      ns_operation.compute_vorticity(velocity[i],vorticity[i]);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
initialize_rhs_vec_convection()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i=1;i<rhs_vec_convection.size();++i)
  {
    ns_operation.set_time(time - double(i)*time_steps[0]);
    ns_operation.rhs_convection(velocity[i],rhs_vec_convection[i]);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
timeloop()
{
  global_timer.restart();

  postprocessor.do_postprocessing(velocity[0],pressure[0],vorticity[0],divergence,time,time_step_number);

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << std::endl << "Starting time loop ..." << std::endl;

  const double EPSILON = 1.0e-10; // epsilon is a small number which is much smaller than the time step size
  while(time<(param.end_time-EPSILON) && time_step_number<=param.max_number_of_steps)
  {
    do_timestep();

    time += time_steps[0];
    ++time_step_number;

    postprocessor.do_postprocessing(velocity[0],pressure[0],vorticity[0],divergence,time,time_step_number);

    // if adaptive time step control: calculate new time step size
    if(param.calculation_of_time_step_size == TimeStepCalculation::AdaptiveTimeStepCFL)
       recalculate_adaptive_time_step();
  }

  total_time += global_timer.wall_time();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
set_time_integrator_constants (unsigned int const current_order)
{
  AssertThrow(current_order <= order,
      ExcMessage("There is a logical error when updating the time integrator constants."));

  unsigned int const MAX_ORDER = 3;
  std::vector<double> alpha_local(MAX_ORDER);
  std::vector<double> beta_local(MAX_ORDER);

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
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
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

    set_alpha_and_beta(alpha_local,beta_local);
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
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
set_alpha_and_beta (std::vector<double> const &alpha_local,
                    std::vector<double> const &beta_local)
{
  AssertThrow((alpha.size() <= alpha_local.size()) && (beta.size() <= beta_local.size()),
      ExcMessage("There is a logical error when setting the time integrator constants. Probably, the specified order of the time integration schemes is not implemented."));

  for(unsigned int i=0;i<alpha.size();++i)
    alpha[i] = alpha_local[i];

  for(unsigned int i=0;i<beta.size();++i)
    beta[i] = beta_local[i];
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
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
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
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

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
do_timestep()
{
  // set the new time integrator constants gamma0, alpha_i, beta_i
  update_time_integrator_constants();

  // set the parameters that NavierStokesOperation depends on
  ns_operation.set_time(time);
  ns_operation.set_time_step(time_steps[0]);
  ns_operation.set_gamma0(gamma0);

  // perform the four substeps of the dual-splitting method
  convective_step();

  pressure_step();

  projection_step();

  viscous_step();

  // prepare global vectors for the next time step
  push_back_vectors();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
convective_step()
{
  Timer timer;
  timer.restart();

  // write output
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && time_step_number%param.output_solver_info_every_timesteps == 0)
  {
    std::cout << std::endl << "______________________________________________________________________" << std::endl
              << std::endl << " Number of TIME STEPS: " << std::left << std::setw(8) << time_step_number
                           << "t_n = " << std::scientific << std::setprecision(4) << time << " -> t_n+1 = " << time + time_steps[0] << std::endl
                           << "______________________________________________________________________" << std::endl;
  }

  // compute body force vector
  ns_operation.compute_rhs(rhs_vec_body_force);
  for (unsigned int d=0; d<velocity_np.n_blocks(); ++d)
  {
    velocity_np.block(d) = rhs_vec_body_force.block(d);
  }

  // compute convective term and extrapolate convective term (if not Stokes equations)
  if(param.solve_stokes_equations == false)
  {
    ns_operation.rhs_convection(velocity[0],rhs_vec_convection[0]);
    for(unsigned int i=0;i<rhs_vec_convection.size();++i)
    {
      for (unsigned int d=0; d<velocity_np.n_blocks(); ++d)
      {
        velocity_np.block(d).add(beta[i],rhs_vec_convection[i].block(d));
      }
    }
  }

  // solve discrete temporal derivative term for intermediate velocity u_hat (if not STS approach)
  if(param.small_time_steps_stability == false)
  {
    for (unsigned int d=0; d<velocity_np.n_blocks(); ++d)
    {
      velocity_np.block(d).equ(time_steps[0]/gamma0,velocity_np.block(d));
    }
    for (unsigned int i=0;i<velocity.size();++i)
    {
      for (unsigned int d=0; d<velocity_np.n_blocks(); ++d)
      {
        velocity_np.block(d).add(alpha[i]/gamma0,velocity[i].block(d));
      }
    }
  }

  if(param.convective_step_implicit == false)
  {
    // write output explicit case
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && time_step_number%param.output_solver_info_every_timesteps == 0)
    {
      std::cout << std::endl << "Solve nonlinear convective problem explicitly:" << std::endl
                << "  iterations:        " << std::setw(4) << std::right << "-" << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
    }
  }
  else // param.convective_step_implicit == true
  {
    AssertThrow(param.convective_step_implicit && !(param.solve_stokes_equations || param.small_time_steps_stability),
        ExcMessage("Use CONVECTIVE_STEP_IMPLICIT = false when solving the Stokes equations or when using the STS approach."));

    // compute temporary vector: temp = Sum_i (alpha_i*u_i)
    temp = 0;
    for(unsigned int i=0;i<velocity.size();++i)
    {
      for(unsigned int d=0;d<temp.n_blocks();++d)
      {
        temp.block(d).add(alpha[i]/time_steps[0],velocity[i].block(d));
      }
    }

    // solve nonlinear convective problem
    unsigned int iterations_implicit_convection = ns_operation.solve_implicit_convective_step(velocity_np,
                                                                                              res_convection,
                                                                                              delta_u_tilde,
                                                                                              temp);

    // write output implicit case
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && time_step_number%param.output_solver_info_every_timesteps == 0)
    {
      std::cout << std::endl << "Solve nonlinear convective problem for intermediate velocity:" << std::endl
                << "  Newton iterations: " << std::setw(4) << std::right << iterations_implicit_convection
                << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
    }
  }

  computing_times[0] += timer.wall_time();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
pressure_step()
{
  Timer timer;
  timer.restart();

  // compute right-hand-side vector
  rhs_pressure(velocity_np,rhs_vec_pressure);

  // extrapolate old solution to get a good initial estimate for the solver
  pressure_np = 0;
  for(unsigned int i=0;i<pressure.size();++i)
  {
    pressure_np.add(beta[i],pressure[i]);
  }

  // solve linear system of equations
  unsigned int pres_niter = ns_operation.solve_pressure(pressure_np, rhs_vec_pressure);

  if(param.pure_dirichlet_bc)
  {
    ns_operation.shift_pressure(pressure_np);
  }

  // write output
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && time_step_number%param.output_solver_info_every_timesteps == 0)
  {
    std::cout << std::endl << "Solve Poisson equation for pressure p:" << std::endl
              << "  PCG iterations:    " << std::setw(4) << std::right << pres_niter << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
  }
  /*
  if(time_step_number%param.output_solver_info_every_timesteps == 0)
  {
    Utilities::System::MemoryStats stats;
    Utilities::System::get_memory_stats(stats);
    Utilities::MPI::MinMaxAvg memory = Utilities::MPI::min_max_avg (stats.VmRSS/1024., MPI_COMM_WORLD);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << "  Memory [MB]: " << memory.min << " (proc" << memory.min_index << ") <= "
                << memory.avg << " (avg)" << " <= " << memory.max << " (proc" << memory.max_index << ")" << std::endl;
    }
  }
  */
  computing_times[1] += timer.wall_time();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
rhs_pressure (const parallel::distributed::BlockVector<value_type>  &src,
              parallel::distributed::Vector<value_type>             &dst)
{
  dst = 0;

  /******************************** I. calculate divergence term ********************************/
  ns_operation.rhs_pressure_divergence_term(src,dst);
  /**********************************************************************************************/

  /***** II. calculate terms originating from inhomogeneous parts of boundary face integrals ****/

  // II.1. BC terms depending on prescribed boundary data,
  //       i.e. pressure Dirichlet boundary conditions on Gamma_N and
  //       body force vector, temporal derivative of velocity on Gamma_D
  ns_operation.rhs_pressure_BC_term(dummy,dst);

  // II.2. viscous term of pressure Neumann boundary condition on Gamma_D
  //       extrapolate vorticity and subsequently evaluate boundary face integral
  //       (this is possible since pressure Neumann BC is linear in vorticity)
  vorticity_extrapolated = 0;
  for(unsigned int i=0;i<vorticity.size();++i)
  {
    for(unsigned int d=0;d<vorticity_extrapolated.n_blocks();++d)
    {
      vorticity_extrapolated.block(d).add(beta[i],vorticity[i].block(d));
    }
  }
  ns_operation.rhs_pressure_viscous_term(vorticity_extrapolated, dst);

  // II.3. convective term of pressure Neumann boundary condition on Gamma_D
  //       (only if we do not solve the Stokes equations)
  //       evaluate convective term and subsequently extrapolate rhs vectors
  //       (the convective term is nonlinear!)
  if(!param.solve_stokes_equations)
  {
    for(unsigned int i=0;i<velocity.size();++i)
    {
      rhs_vec_pressure_temp = 0;
      ns_operation.rhs_pressure_convective_term(velocity[i], rhs_vec_pressure_temp);
      dst.add(beta[i],rhs_vec_pressure_temp);
    }
  }
  /**********************************************************************************************/

  if(param.pure_dirichlet_bc)
    ns_operation.apply_nullspace_projection(dst);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
projection_step()
{
  Timer timer;
  timer.restart();

  // when using the STS stability approach vector updates have to be performed to obtain the
  // intermediate velocity u_hat which is used to calculate the rhs of the projection step
  if(param.small_time_steps_stability == true)
  {
    for (unsigned int d=0; d<velocity_np.n_blocks(); ++d)
    {
      velocity_np.block(d).equ(time_steps[0]/gamma0,velocity_np.block(d));
    }
    for (unsigned int i=0;i<velocity.size();++i)
    {
      for (unsigned int d=0; d<velocity_np.n_blocks(); ++d)
      {
        velocity_np.block(d).add(alpha[i]/gamma0,velocity[i].block(d));
      }
    }
  }

  // compute right-hand-side vector
  ns_operation.rhs_projection(velocity_np,pressure_np,rhs_vec_projection);

  // solve linear system of equations
  unsigned int iterations_projection = ns_operation.solve_projection(velocity_np,rhs_vec_projection,velocity[0],cfl);

  // write output
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && time_step_number%param.output_solver_info_every_timesteps == 0)
  {
    std::cout << std::endl << "Solve projection step for intermediate velocity:" << std::endl
              << "  PCG iterations:    " << std::setw(4) << std::right << iterations_projection << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
  }

  // postprocessing related to the analysis of different projection algorithms
  if(param.compute_divergence == true)
  {
    postprocessor.analyze_divergence_error(velocity_np,time,time_step_number);
    ns_operation.compute_divergence(velocity_np,divergence,false);
  }

  computing_times[2] += timer.wall_time();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
viscous_step()
{
  Timer timer;
  timer.restart();

  // compute right-hand-side vector
  ns_operation.rhs_viscous(velocity_np,rhs_vec_viscous);

  // extrapolate old solution to get a good initial estimate for the solver
  velocity_np = 0;
  for (unsigned int i=0; i<velocity.size(); ++i)
  {
    for (unsigned int d=0; d<velocity_np.n_blocks(); ++d)
    {
      velocity_np.block(d).add(beta[i],velocity[i].block(d));
    }
  }

  // solve linear system of equations
  unsigned int iterations_viscous = ns_operation.solve_viscous(velocity_np, rhs_vec_viscous);

  // write output
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && time_step_number%param.output_solver_info_every_timesteps == 0)
  {
    std::cout << std::endl << "Solve viscous step for velocity u:" << std::endl
              << "  PCG iterations:    " << std::setw(4) << std::right << iterations_viscous << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
  }

  computing_times[3] += timer.wall_time();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
push_back_vectors()
{
  Timer timer;
  timer.restart();

  push_back_solution();

  push_back_vorticity();

  if(param.solve_stokes_equations == false)
    push_back_rhs_vec_convection();

  computing_times[4] += timer.wall_time();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
push_back_solution()
{
  /*
   *   time t
   *  -------->   t_{n-2}   t_{n-1}   t_{n}    t_{n+1}
   *  _______________|_________|________|_________|___________\
   *                 |         |        |         |           /
   *
   *  sol-vec:    sol[2]    sol[1]    sol[0]    sol_np
   *
   * <- sol[2] <- sol[1] <- sol[0] <- sol_np <- sol[2] <--
   * |___________________________________________________|
   *
   */

  // solution at t_{n-i} <-- solution at t_{n-i+1}
  for(unsigned int i=velocity.size()-1; i>0; --i)
  {
    velocity[i].swap(velocity[i-1]);
    pressure[i].swap(pressure[i-1]);
  }
  velocity[0].swap(velocity_np);
  pressure[0].swap(pressure_np);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
push_back_vorticity()
{
  // solution at t_{n-i} <-- solution at t_{n-i+1}
  for(unsigned int i=vorticity.size()-1; i>0; --i)
  {
    vorticity[i].swap(vorticity[i-1]);
  }
  ns_operation.compute_vorticity(velocity[0],vorticity[0]);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
push_back_rhs_vec_convection()
{
  // solution at t_{n-i} <-- solution at t_{n-i+1}
  for(unsigned int i=rhs_vec_convection.size()-1; i>0; --i)
  {
    rhs_vec_convection[i].swap(rhs_vec_convection[i-1]);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
analyze_computing_times() const
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  std::string names[5] = {"Convection   ","Pressure     ","Projection   ","Viscous      ","Other        "};
  pcout << std::endl << "_________________________________________________________________________________" << std::endl
        << std::endl << "Computing times:          min        avg        max        rel      p_min  p_max" << std::endl;
  double total_avg_time = 0.0;
  for (unsigned int i=0; i<computing_times.size(); ++i)
  {
    Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg (computing_times[i], MPI_COMM_WORLD);
        total_avg_time += data.avg;
  }
  for (unsigned int i=0; i<computing_times.size(); ++i)
  {
    Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg (computing_times[i], MPI_COMM_WORLD);
    pcout << "  Step " << i+1 <<  ": " << names[i]  << std::scientific
          << std::setprecision(4) << std::setw(10) << data.min << " "
          << std::setprecision(4) << std::setw(10) << data.avg << " "
          << std::setprecision(4) << std::setw(10) << data.max << " "
          << std::setprecision(4) << std::setw(10) << data.avg/total_avg_time << "  "
          << std::setw(6) << std::left << data.min_index << " "
          << std::setw(6) << std::left << data.max_index << std::endl;
  }
  pcout  << "  Time in steps 1-" << computing_times.size() << ":              "
         << std::setprecision(4) << std::setw(10) << total_avg_time
         << "            "
         << std::setprecision(4) << std::setw(10) << total_avg_time/total_avg_time << std::endl;
  Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg (total_time, MPI_COMM_WORLD);
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


#endif /* INCLUDE_TIMEINTDUALSPLITTING_H_ */
