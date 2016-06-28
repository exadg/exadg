/*
 * TimeIntNavierStokesCoupled.h
 *
 *  Created on: Jun 8, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_TIMEINTNAVIERSTOKESCOUPLED_H_
#define INCLUDE_TIMEINTNAVIERSTOKESCOUPLED_H_


#include "TimeStepCalculation.h"

template<int dim> class PostProcessor;

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
class TimeIntNavierStokesCoupled
{
public:
  static const unsigned int number_vorticity_components = (dim==2) ? 1 : dim;

  TimeIntNavierStokesCoupled(NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &ns_operation_in,
                             PostProcessor<dim> &postprocessor_in,
                             InputParameters const & param_in,
                             unsigned int const n_refine_time_in)
    :
    ns_operation(ns_operation_in),
    postprocessor(postprocessor_in),
    param(param_in),
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
//    velocity(order),
//    pressure(order),
    solution(order),
//    vorticity(order),
    vec_convective_term(order)
  {}

  void setup();

  void timeloop();

  void analyze_computing_times() const;

private:
//  void do_timestep();
  void solve_timestep();

  void initialize_time_integrator_constants();
  void update_time_integrator_constants();
  void set_time_integrator_constants (unsigned int const current_order);
  void set_adaptive_time_integrator_constants(unsigned int const current_order);
  void set_alpha_and_beta (std::vector<value_type> const &alpha_local,
                           std::vector<value_type> const &beta_local);
  void check_time_integrator_constants (unsigned int const number) const;

  void calculate_time_step();
  void recalculate_adaptive_time_step();

  void postprocessing();

  void initialize_vectors();
  void initialize_current_solution();
  void initialize_former_solution();
  void calculate_vorticity();
//  void initialize_vorticity();
  void initialize_vec_convective_term();


  void prepare_vectors_for_next_timestep();
  void push_back_solution();
//  void push_back_vorticity();
//  void push_back_rhs_vec_convection();
  void push_back_vec_convective_term();

  NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &ns_operation;
  PostProcessor<dim> &postprocessor;
  InputParameters const & param;

  Timer global_timer;
  value_type total_time;

  unsigned int const order;
  value_type time;
  std::vector<value_type> time_steps;
  unsigned int const n_refine_time;
  value_type cfl;
  unsigned int time_step_number;

  value_type gamma0;
  std::vector<value_type> alpha, beta;

//  parallel::distributed::BlockVector<value_type> velocity_np;
//  std::vector<parallel::distributed::BlockVector<value_type> > velocity;
//
//  parallel::distributed::Vector<value_type> pressure_np;
//  std::vector<parallel::distributed::Vector<value_type> > pressure;

  parallel::distributed::BlockVector<value_type> solution_np;
  std::vector<parallel::distributed::BlockVector<value_type> > solution;

  parallel::distributed::BlockVector<value_type> sum_alphai_ui;
  parallel::distributed::BlockVector<value_type> rhs_vector;

  std::vector<parallel::distributed::BlockVector<value_type> > vec_convective_term;

  parallel::distributed::BlockVector<value_type> vorticity;

//  std::vector<parallel::distributed::BlockVector<value_type> > rhs_vec_convection;

  // postprocessing: divergence of intermediate velocity u_hathat
  parallel::distributed::Vector<value_type> divergence;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
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

  // set the parameters that NavierStokesOperation depends on
  ns_operation.set_time(time);
  ns_operation.set_time_step(time_steps[0]);
  ns_operation.set_gamma0(gamma0);

  calculate_vorticity();

  if(param.solve_stokes_equations == false && param.convective_step_implicit == false &&
     param.start_with_low_order == false)
    initialize_vec_convective_term();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
initialize_time_integrator_constants()
{
  AssertThrow(order == 1 || order == 2 || order == 3,
      ExcMessage("Specified order of time integration scheme is not implemented."));

  // the default case is start_with_low_order == false
  // in case of start_with_low_order == true the time integrator constants have to be adjusted in do_timestep
  set_time_integrator_constants(order);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
initialize_vectors()
{
  // solution
  for(unsigned int i=0;i<solution.size();++i)
    ns_operation.initialize_block_vector_velocity_pressure(solution[i]);
  ns_operation.initialize_block_vector_velocity_pressure(solution_np);

  // convective term
  if(param.solve_stokes_equations == false && param.convective_step_implicit == false)
  {
    for(unsigned int i=0;i<vec_convective_term.size();++i)
      ns_operation.initialize_block_vector_velocity(vec_convective_term[i]);
  }

  // temporal derivative term: sum (alpha_i * u_i)
  ns_operation.initialize_block_vector_velocity(sum_alphai_ui);
  // rhs_vector
  ns_operation.initialize_block_vector_velocity_pressure(rhs_vector);

//  // velocity
//  for(unsigned int i=0;i<velocity.size();++i)
//    ns_operation.initialize_block_vector_velocity(velocity[i]);
//  ns_operation.initialize_block_vector_velocity(velocity_np);
//  // pressure
//  for(unsigned int i=0;i<pressure.size();++i)
//    ns_operation.initialize_vector_pressure(pressure[i]);
//  ns_operation.initialize_vector_pressure(pressure_np);

  // vorticity
  ns_operation.initialize_block_vector_vorticity(vorticity);

  // divergence
  if(param.compute_divergence == true)
  {
    ns_operation.initialize_scalar_vector_velocity(divergence);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
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

    value_type adaptive_time_step = calculate_adaptive_time_step_cfl<dim, fe_degree, value_type>(ns_operation.get_data(),
                                                                                                 solution[0],
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
void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
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
                                                                               solution[0],
                                                                               cfl,
                                                                               time_steps[0]);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
initialize_current_solution()
{
  ns_operation.prescribe_initial_conditions(solution[0],solution[0].block(dim),time);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
initialize_former_solution()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i=1;i<solution.size();++i)
    ns_operation.prescribe_initial_conditions(solution[i],solution[i].block(dim),time - value_type(i)*time_steps[0]);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
calculate_vorticity()
{
  ns_operation.compute_vorticity(solution[0],vorticity);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
initialize_vec_convective_term()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i=1;i<vec_convective_term.size();++i)
  {
    ns_operation.evaluate_convective_term(vec_convective_term[i],solution[i],time - value_type(i)*time_steps[0]);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
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

    if(param.calculation_of_time_step_size == TimeStepCalculation::AdaptiveTimeStepCFL)
       recalculate_adaptive_time_step();
  }

  total_time += global_timer.wall_time();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
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
void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
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
void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
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
void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
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
void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
check_time_integrator_constants(unsigned int current_time_step_number) const
{
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << "Time integrator constants: time step " << current_time_step_number << std::endl;

    std::cout << "Gamma0   = " << gamma0   << std::endl;

    for(unsigned int i=0;i<order;++i)
      std::cout << "Alpha[" << i <<"] = " << alpha[i] << std::endl;

    for(unsigned int i=0;i<order;++i)
      std::cout << "Beta[" << i <<"]  = " << beta[i] << std::endl;
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
postprocessing()
{
  postprocessor.do_postprocessing(solution[0],solution[0].block(dim),vorticity,divergence,time,time_step_number);
}

//template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
//void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
//do_timestep()
//{
//  // set the new time integrator constants gamma0, alpha_i, beta_i
//  update_time_integrator_constants();
//
//  // solve coupled Navier-Stokes equations
//  solve_timestep();
//
//  // prepare global vectors for the next time step
//  prepare_vectors_for_next_timestep();
//}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
solve_timestep()
{
  Timer timer;
  timer.restart();

  // set the parameters that NavierStokesOperation depends on
  ns_operation.set_time(time);
  ns_operation.set_time_step(time_steps[0]);
  ns_operation.set_gamma0(gamma0);

  // write output
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && time_step_number%param.output_solver_info_every_timesteps == 0)
  {
    std::cout << std::endl << "______________________________________________________________________" << std::endl
              << std::endl << " Number of TIME STEPS: " << std::left << std::setw(8) << time_step_number
                           << "t_n = " << std::scientific << std::setprecision(4) << time << " -> t_n+1 = " << time + time_steps[0] << std::endl
                           << "______________________________________________________________________" << std::endl;
  }

  // extrapolate old solution to obtain a good initial guess for the solver
  // 1st order extrapolation
  solution_np.equ(1.0,solution[0]);

  // higher order extrapolation
//  solution_np.equ(beta[0],solution[0]);
//  for(unsigned int i=1;i<solution.size();++i)
//    solution_np.add(beta[i],solution[i]);

  // calculate sum (alpha_i/dt * u_i)
  for(unsigned int d=0;d<dim;++d)
    sum_alphai_ui.block(d).equ(alpha[0]/time_steps[0],solution[0].block(d));
  for (unsigned int i=1;i<solution.size();++i)
  {
    for(unsigned int d=0;d<dim;++d)
      sum_alphai_ui.block(d).add(alpha[i]/time_steps[0],solution[i].block(d));
  }

  if(param.solve_stokes_equations == true || param.convective_step_implicit == false)
  {
    // calculate rhs vector
    ns_operation.rhs_stokes_problem(rhs_vector,sum_alphai_ui);

    // evaluate convective term and add extrapolation of convective term to the rhs (-> minus sign!)
    if(param.solve_stokes_equations == false)
    {
      ns_operation.evaluate_convective_term(vec_convective_term[0],solution[0],time);

      for(unsigned int i=0;i<vec_convective_term.size();++i)
      {
        for(unsigned int d=0;d<dim;++d)
          rhs_vector.block(d).add(-beta[i],vec_convective_term[i].block(d));
      }
    }

    // solve coupled system of equations
    unsigned int iterations = ns_operation.solve_linearized_problem(solution_np,rhs_vector);

    // write output
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && time_step_number%param.output_solver_info_every_timesteps == 0)
    {
      std::cout << std::endl << "Solve linear Navier-Stokes problem:" << std::endl
                             << "  Iterations: " << std::setw(6) << std::right << iterations
                             << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
    }
  }
  else
  {
    // Newton solver
    unsigned int iterations = ns_operation.solve_nonlinear_problem(solution_np,sum_alphai_ui);

    // write output
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && time_step_number%param.output_solver_info_every_timesteps == 0)
    {
      std::cout << std::endl << "Solve nonlinear Navier-Stokes problem:" << std::endl
                             << "  Newton iterations: " << std::setw(6) << std::right << iterations
                             << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
    }
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
prepare_vectors_for_next_timestep()
{
  push_back_solution();

  calculate_vorticity();

  if(param.solve_stokes_equations == false && param.convective_step_implicit == false)
    push_back_vec_convective_term();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
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
  for(unsigned int i=solution.size()-1; i>0; --i)
  {
    solution[i].swap(solution[i-1]);
  }
  solution[0].swap(solution_np);
}

//template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
//void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
//push_back_vorticity()
//{
//  // solution at t_{n-i} <-- solution at t_{n-i+1}
//  for(unsigned int i=vorticity.size()-1; i>0; --i)
//  {
//    vorticity[i].swap(vorticity[i-1]);
//  }
//  ns_operation.compute_vorticity(solution[0],vorticity[0]);
//}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
push_back_vec_convective_term()
{
  // solution at t_{n-i} <-- solution at t_{n-i+1}
  for(unsigned int i=vec_convective_term.size()-1; i>0; --i)
  {
    vec_convective_term[i].swap(vec_convective_term[i-1]);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
analyze_computing_times() const
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "_________________________________________________________________________________" << std::endl
        << std::endl << "Computing times:          min        avg        max        rel      p_min  p_max" << std::endl;

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


#endif /* INCLUDE_TIMEINTNAVIERSTOKESCOUPLED_H_ */
