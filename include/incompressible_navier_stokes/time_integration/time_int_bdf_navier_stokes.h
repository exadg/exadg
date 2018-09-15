/*
 * TimeIntBDF.h
 *
 *  Created on: Jun 10, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_NAVIER_STOKES_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_NAVIER_STOKES_H_

#include <deal.II/base/timer.h>

#include "../../incompressible_navier_stokes/time_integration/restart.h"
#include "time_integration/bdf_time_integration.h"
#include "time_integration/extrapolation_scheme.h"
#include "time_integration/time_step_calculation.h"

template<typename Operator, typename Vector>
class ExplicitRungeKuttaTimeIntegratorOIF
{
public:
  // Constructor
  ExplicitRungeKuttaTimeIntegratorOIF(unsigned int              order_time_integrator,
                                      std::shared_ptr<Operator> underlying_operator_in)
    : order(order_time_integrator), underlying_operator(underlying_operator_in)
  {
    underlying_operator->initialize_dof_vector(solution_interpolated);

    // initialize vectors
    if(order >= 2)
      underlying_operator->initialize_dof_vector(vec_rhs);
    if(order >= 3)
      underlying_operator->initialize_dof_vector(vec_temp);
  }

  void
  interpolate(Vector &                    dst,
              double const                evaluation_time,
              std::vector<Vector *> const solutions,
              std::vector<double> const   times) const
  {
    dst = 0;

    // loop over all interpolation points
    for(unsigned int k = 0; k < solutions.size(); ++k)
    {
      // evaluate lagrange polynomial l_k
      double l_k = 1.0;

      for(unsigned int j = 0; j < solutions.size(); ++j)
      {
        if(j != k)
        {
          l_k *= (evaluation_time - times[j]) / (times[k] - times[j]);
        }
      }

      dst.add(l_k, *solutions[k]);
    }
  }

  void
  solve_timestep(Vector &                    dst,
                 Vector const &              src,
                 double                      time_n,
                 double                      time_step,
                 std::vector<Vector *> const solutions,
                 std::vector<double> const   times)
  {
    if(order == 1) // explicit Euler method
    {
      interpolate(solution_interpolated, time_n, solutions, times);
      underlying_operator->evaluate(dst, src, time_n, solution_interpolated);
      dst *= time_step;
      dst.add(1.0, src);
    }
    else if(order == 2) // Runge-Kutta method of order 2
    {
      // stage 1
      interpolate(solution_interpolated, time_n, solutions, times);
      underlying_operator->evaluate(vec_rhs, src, time_n, solution_interpolated);

      // stage 2
      vec_rhs *= time_step / 2.;
      vec_rhs.add(1.0, src);
      interpolate(solution_interpolated, time_n + time_step / 2., solutions, times);
      underlying_operator->evaluate(dst, vec_rhs, time_n + time_step / 2., solution_interpolated);
      dst *= time_step;
      dst.add(1.0, src);
    }
    else if(order == 3) // Heun's method of order 3
    {
      dst = src;

      // stage 1
      interpolate(solution_interpolated, time_n, solutions, times);
      underlying_operator->evaluate(vec_temp, src, time_n, solution_interpolated);
      dst.add(1. * time_step / 4., vec_temp);

      // stage 2
      vec_rhs.equ(1., src);
      vec_rhs.add(time_step / 3., vec_temp);
      interpolate(solution_interpolated, time_n + time_step / 3., solutions, times);
      underlying_operator->evaluate(vec_temp,
                                    vec_rhs,
                                    time_n + time_step / 3.,
                                    solution_interpolated);

      // stage 3
      vec_rhs.equ(1., src);
      vec_rhs.add(2.0 * time_step / 3.0, vec_temp);
      interpolate(solution_interpolated, time_n + 2. * time_step / 3., solutions, times);
      underlying_operator->evaluate(vec_temp,
                                    vec_rhs,
                                    time_n + 2. * time_step / 3.,
                                    solution_interpolated);
      dst.add(3. * time_step / 4., vec_temp);
    }
    else if(order == 4) // classical 4th order Runge-Kutta method
    {
      dst = src;

      // stage 1
      interpolate(solution_interpolated, time_n, solutions, times);
      underlying_operator->evaluate(vec_temp, src, time_n, solution_interpolated);
      dst.add(time_step / 6., vec_temp);

      // stage 2
      vec_rhs.equ(1., src);
      vec_rhs.add(time_step / 2., vec_temp);
      interpolate(solution_interpolated, time_n + time_step / 2., solutions, times);
      underlying_operator->evaluate(vec_temp,
                                    vec_rhs,
                                    time_n + time_step / 2.,
                                    solution_interpolated);
      dst.add(time_step / 3., vec_temp);

      // stage 3
      vec_rhs.equ(1., src);
      vec_rhs.add(time_step / 2., vec_temp);
      interpolate(solution_interpolated, time_n + time_step / 2., solutions, times);
      underlying_operator->evaluate(vec_temp,
                                    vec_rhs,
                                    time_n + time_step / 2.,
                                    solution_interpolated);
      dst.add(time_step / 3., vec_temp);

      // stage 4
      vec_rhs.equ(1., src);
      vec_rhs.add(time_step, vec_temp);
      interpolate(solution_interpolated, time_n + time_step, solutions, times);
      underlying_operator->evaluate(vec_temp, vec_rhs, time_n + time_step, solution_interpolated);
      dst.add(time_step / 6., vec_temp);
    }
    else
    {
      AssertThrow(order <= 1,
                  ExcMessage("Explicit Runge-Kutta method only implemented for order <= 4!"));
    }
  }

private:
  unsigned int order;

  std::shared_ptr<Operator> underlying_operator;

  Vector vec_rhs, vec_temp;
  Vector solution_interpolated;
};

namespace IncNS
{
template<int dim, typename Number>
class PostProcessorBase;

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
class TimeIntBDFNavierStokes
{
public:
  TimeIntBDFNavierStokes(std::shared_ptr<NavierStokesOperation> navier_stokes_operation_in,
                         std::shared_ptr<PostProcessorBase<dim, value_type>> postprocessor_in,
                         InputParameters<dim> const &                        param_in,
                         unsigned int const                                  n_refine_time_in,
                         bool const use_adaptive_time_stepping)
    : postprocessor(postprocessor_in),
      param(param_in),
      total_time(0.0),
      time(param.start_time),
      time_step_number(1),
      order(param_in.order_time_integrator),
      time_steps(param_in.order_time_integrator),
      bdf(param_in.order_time_integrator, param_in.start_with_low_order),
      extra(param_in.order_time_integrator, param_in.start_with_low_order),
      adaptive_time_stepping(use_adaptive_time_stepping),
      cfl(param.cfl / std::pow(2.0, n_refine_time_in)),
      cfl_oif(param_in.cfl_oif / std::pow(2.0, n_refine_time_in)),
      M(1),
      delta_s(1.0),
      pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
      counter_mean_velocity(0),
      n_refine_time(n_refine_time_in),
      navier_stokes_operation(navier_stokes_operation_in)
  {
  }

  virtual ~TimeIntBDFNavierStokes()
  {
  }

  void
  setup(bool do_restart);

  void
  timeloop();

  bool
  advance_one_timestep(bool write_final_output);

  void
  timeloop_steady_problem();

  virtual void
  analyze_computing_times() const = 0;

  double
  get_time() const
  {
    return this->time;
  }

  void
  set_time(double const & current_time)
  {
    this->time = current_time;
  }

  double
  get_time_step_size() const
  {
    return time_steps[0];
  }

  void
  set_time_step_size(double const & time_step)
  {
    // constant time step sizes
    if(adaptive_time_stepping == false)
    {
      AssertThrow(time_step_number == 1,
                  ExcMessage("For time integration with constant time step sizes this "
                             "function can only be called in the very first time step."));
    }

    time_steps[0] = time_step;

    // fill time_steps array
    if(time_step_number == 1)
    {
      for(unsigned int i = 1; i < order; ++i)
        time_steps[i] = time_steps[0];
    }
  }

  double
  get_scaling_factor_time_derivative_term()
  {
    return bdf.get_gamma0() / time_steps[0];
  }

protected:
  std::shared_ptr<PostProcessorBase<dim, value_type>> postprocessor;

  void
  do_timestep();

  virtual void
  initialize_vectors();

  virtual void
  initialize_time_integrator_constants();

  virtual void
  update_time_integrator_constants();

  void
  calculate_vorticity(parallel::distributed::Vector<value_type> &       dst,
                      parallel::distributed::Vector<value_type> const & src) const;

  void
  calculate_divergence(parallel::distributed::Vector<value_type> &       dst,
                       parallel::distributed::Vector<value_type> const & src) const;

  void
  calculate_velocity_magnitude(parallel::distributed::Vector<value_type> &       dst,
                               parallel::distributed::Vector<value_type> const & src) const;

  void
  calculate_vorticity_magnitude(parallel::distributed::Vector<value_type> &       dst,
                                parallel::distributed::Vector<value_type> const & src) const;

  void
  calculate_streamfunction(parallel::distributed::Vector<value_type> &       dst,
                           parallel::distributed::Vector<value_type> const & src) const;

  void
  calculate_q_criterion(parallel::distributed::Vector<value_type> &       dst,
                        parallel::distributed::Vector<value_type> const & src) const;

  void
  calculate_processor_id(parallel::distributed::Vector<value_type> & dst) const;

  void
  calculate_mean_velocity(parallel::distributed::Vector<value_type> &       dst,
                          parallel::distributed::Vector<value_type> const & src) const;

  void
  initialize_oif();

  virtual void
  calculate_time_step();

  virtual void
  recalculate_adaptive_time_step();

  // output of solver information regarding iteration counts, wall times,
  // and remaining time until completion of the simulation
  void
  output_solver_info_header() const;

  void
  output_remaining_time() const;

  virtual void
  read_restart_vectors(boost::archive::binary_iarchive & ia) = 0;

  virtual void
  write_restart_vectors(boost::archive::binary_oarchive & oa) const = 0;

  virtual void
  resume_from_restart();

  void
  write_restart() const;

  InputParameters<dim> const & param;

  // computation time
  Timer  global_timer;
  double total_time;

  // physical time
  double time;

  // the number of the current time step starting with time_step_number = 1
  unsigned int time_step_number;

  // order of time integration scheme
  unsigned int const order;

  // Vector that stores time step sizes. This vector is necessary
  // if adaptive_time_stepping = true. For constant time step sizes
  // one double for the time step size would be sufficient.
  std::vector<double> time_steps;

  // Time integrator constants of BDF and extrapolation schemes
  BDFTimeIntegratorConstants bdf;
  ExtrapolationConstants     extra;

  // use adaptive time stepping?
  bool const adaptive_time_stepping;

  // gobal cfl number
  double const cfl;

  // Operator-integration-factor splitting for convective term
  std::shared_ptr<ConvectiveOperatorNavierStokes<NavierStokesOperation, value_type>>
    convective_operator_OIF;

  std::shared_ptr<ExplicitRungeKuttaTimeIntegratorOIF<
    ConvectiveOperatorNavierStokes<NavierStokesOperation, value_type>,
    parallel::distributed::Vector<value_type>>>
    rk_time_integrator_OIF;

  // cfl number cfl_oif for operator-integration-factor splitting
  double const cfl_oif;

  // number of substeps for operator-integration-factor splitting per macro time step dt
  unsigned int M;

  // substepping time step size delta_s for operator-integration-factor splitting
  double delta_s;

  // solution vectors needed for OIF substepping of convective term
  parallel::distributed::Vector<value_type> solution_tilde_m;
  parallel::distributed::Vector<value_type> solution_tilde_mp;

  ConditionalOStream pcout;

  // postprocessing: additional fields
  mutable parallel::distributed::Vector<value_type> divergence;
  mutable parallel::distributed::Vector<value_type> velocity_magnitude;
  mutable parallel::distributed::Vector<value_type> vorticity_magnitude;
  mutable parallel::distributed::Vector<value_type> streamfunction;
  mutable parallel::distributed::Vector<value_type> q_criterion;
  mutable parallel::distributed::Vector<value_type> processor_id;
  // mean velocity, i.e., velocity field averaged over time
  mutable parallel::distributed::Vector<value_type> mean_velocity;
  mutable unsigned int                              counter_mean_velocity;

  std::vector<SolutionField<dim, value_type>> additional_fields;

private:
  virtual void
  setup_derived() = 0;

  virtual void
  initialize_current_solution() = 0;

  virtual void
  initialize_former_solution() = 0;

  void
  initialize_solution_and_calculate_timestep(bool do_restart);

  virtual void
  solve_timestep() = 0;

  virtual void
  solve_steady_problem() = 0;

  virtual void
  postprocessing() const = 0;

  virtual void
  postprocessing_steady_problem() const = 0;

  // TODO
  virtual void
  postprocessing_stability_analysis() = 0;

  virtual void
  prepare_vectors_for_next_timestep() = 0;

  virtual parallel::distributed::Vector<value_type> const &
  get_velocity() = 0;

  unsigned int const n_refine_time;

  std::shared_ptr<NavierStokesOperation> navier_stokes_operation;
};

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::setup(bool do_restart)
{
  pcout << std::endl << "Setup time integrator ..." << std::endl << std::endl;

  // initialize time integrator constants assuming that the time integrator
  // uses a high-order method in first time step, i.e., the default case is
  // start_with_low_order = false. This is reasonable since DGNavierStokes
  // uses these time integrator constants for the setup of solvers.
  // in case of start_with_low_order == true the time integrator constants
  // have to be adjusted in timeloop().
  initialize_time_integrator_constants();

  // operator-integration-factor splitting
  initialize_oif();

  // initialize global solution vectors (allocation)
  initialize_vectors();

  // initializes the solution and calculates the time step size!
  initialize_solution_and_calculate_timestep(do_restart);

  // this is where the setup of deriving classes is performed
  setup_derived();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
  initialize_time_integrator_constants()
{
  bdf.initialize();
  extra.initialize();
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
  update_time_integrator_constants()
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

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::initialize_vectors()
{
  // divergence
  if(this->param.output_data.write_divergence == true)
  {
    navier_stokes_operation->initialize_vector_velocity_scalar(this->divergence);

    SolutionField<dim, value_type> sol;
    sol.type        = SolutionFieldType::scalar;
    sol.name        = "div_u";
    sol.dof_handler = &navier_stokes_operation->get_dof_handler_u_scalar();
    sol.vector      = &divergence;
    this->additional_fields.push_back(sol);
  }

  // velocity magnitude
  if(this->param.output_data.write_velocity_magnitude == true)
  {
    navier_stokes_operation->initialize_vector_velocity_scalar(this->velocity_magnitude);

    SolutionField<dim, value_type> sol;
    sol.type        = SolutionFieldType::scalar;
    sol.name        = "velocity_magnitude";
    sol.dof_handler = &navier_stokes_operation->get_dof_handler_u_scalar();
    sol.vector      = &velocity_magnitude;
    this->additional_fields.push_back(sol);
  }

  // vorticity magnitude
  if(this->param.output_data.write_vorticity_magnitude == true)
  {
    navier_stokes_operation->initialize_vector_velocity_scalar(this->vorticity_magnitude);

    SolutionField<dim, value_type> sol;
    sol.type        = SolutionFieldType::scalar;
    sol.name        = "vorticity_magnitude";
    sol.dof_handler = &navier_stokes_operation->get_dof_handler_u_scalar();
    sol.vector      = &vorticity_magnitude;
    this->additional_fields.push_back(sol);
  }


  // streamfunction
  if(this->param.output_data.write_streamfunction == true)
  {
    navier_stokes_operation->initialize_vector_velocity_scalar(this->streamfunction);

    SolutionField<dim, value_type> sol;
    sol.type        = SolutionFieldType::scalar;
    sol.name        = "streamfunction";
    sol.dof_handler = &navier_stokes_operation->get_dof_handler_u_scalar();
    sol.vector      = &streamfunction;
    this->additional_fields.push_back(sol);
  }

  // q criterion
  if(this->param.output_data.write_q_criterion == true)
  {
    navier_stokes_operation->initialize_vector_velocity_scalar(this->q_criterion);

    SolutionField<dim, value_type> sol;
    sol.type        = SolutionFieldType::scalar;
    sol.name        = "q_criterion";
    sol.dof_handler = &navier_stokes_operation->get_dof_handler_u_scalar();
    sol.vector      = &q_criterion;
    this->additional_fields.push_back(sol);
  }

  // processor id
  if(this->param.output_data.write_processor_id == true)
  {
    navier_stokes_operation->initialize_vector_velocity_scalar(this->processor_id);

    SolutionField<dim, value_type> sol;
    sol.type        = SolutionFieldType::scalar;
    sol.name        = "processor_id";
    sol.dof_handler = &navier_stokes_operation->get_dof_handler_u_scalar();
    sol.vector      = &processor_id;
    this->additional_fields.push_back(sol);
  }

  // mean velocity
  if(this->param.output_data.mean_velocity.calculate == true)
  {
    navier_stokes_operation->initialize_vector_velocity(this->mean_velocity);

    SolutionField<dim, value_type> sol;
    sol.type        = SolutionFieldType::vector;
    sol.name        = "mean_velocity";
    sol.dof_handler = &navier_stokes_operation->get_dof_handler_u();
    sol.vector      = &mean_velocity;
    this->additional_fields.push_back(sol);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
  initialize_solution_and_calculate_timestep(bool do_restart)
{
  if(do_restart)
  {
    resume_from_restart();

    // if anything in the temporal discretization is changed, start_with_low_order has to be set to
    // true otherwise the old solutions would not fit the time step increments, etc.
    if(param.start_with_low_order)
      calculate_time_step();

    if(adaptive_time_stepping == true)
      recalculate_adaptive_time_step();
  }
  else
  {
    // when using time step adaptivity the time_step depends on the velocity field. Therefore, first
    // prescribe initial conditions before calculating the time step size
    initialize_current_solution();

    // initializing the solution at former time instants, e.g. t = start_time - time_step, requires
    // the time step size. Therefore, first calculate the time step size
    calculate_time_step();

    // now: prescribe initial conditions at former time instants t = time - time_step, time
    // - 2.0*time_step, etc.
    if(param.start_with_low_order == false)
      initialize_former_solution();
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::initialize_oif()
{
  // Operator-integration-factor splitting
  if(this->param.equation_type == EquationType::NavierStokes &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    convective_operator_OIF.reset(
      new ConvectiveOperatorNavierStokes<NavierStokesOperation, value_type>(
        navier_stokes_operation));

    rk_time_integrator_OIF.reset(
      new ExplicitRungeKuttaTimeIntegratorOIF<
        ConvectiveOperatorNavierStokes<NavierStokesOperation, value_type>,
        parallel::distributed::Vector<value_type>>(this->order, convective_operator_OIF));

    // temporary vectors required for operator-integration-factor splitting of convective term
    navier_stokes_operation->initialize_vector_velocity(solution_tilde_m);
    navier_stokes_operation->initialize_vector_velocity(solution_tilde_mp);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::resume_from_restart()
{
  const std::string filename = restart_filename<dim>(param);
  std::ifstream     in(filename.c_str());
  check_file(in, filename);
  boost::archive::binary_iarchive ia(in);
  resume_restart<dim, value_type>(ia, param, time, time_steps, order);

  read_restart_vectors(ia);

  finished_reading_restart_output();
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::write_restart() const
{
  const double EPSILON = 1.0e-10; // small number which is much smaller than the time step size

  const double wall_time = global_timer.wall_time();
  if((std::fmod(time, param.restart_interval_time) < time_steps[0] + EPSILON &&
      time > param.restart_interval_time - EPSILON) ||
     (std::fmod(wall_time, param.restart_interval_wall_time) < wall_time - total_time) ||
     (time_step_number % param.restart_every_timesteps == 0))
  {
    std::ostringstream oss;

    boost::archive::binary_oarchive oa(oss);
    write_restart_preamble<dim, value_type>(oa, param, time_steps, time, order);
    write_restart_vectors(oa);
    write_restart_file<dim>(oss, param);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::calculate_vorticity(
  parallel::distributed::Vector<value_type> &       dst,
  parallel::distributed::Vector<value_type> const & src) const
{
  navier_stokes_operation->compute_vorticity(dst, src);
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::calculate_divergence(
  parallel::distributed::Vector<value_type> &       dst,
  parallel::distributed::Vector<value_type> const & src) const
{
  if(this->param.output_data.write_output == true &&
     this->param.output_data.write_divergence == true)
  {
    navier_stokes_operation->compute_divergence(dst, src);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
  calculate_velocity_magnitude(parallel::distributed::Vector<value_type> &       dst,
                               parallel::distributed::Vector<value_type> const & src) const
{
  if(this->param.output_data.write_output == true &&
     this->param.output_data.write_velocity_magnitude == true)
  {
    navier_stokes_operation->compute_velocity_magnitude(dst, src);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
  calculate_vorticity_magnitude(parallel::distributed::Vector<value_type> &       dst,
                                parallel::distributed::Vector<value_type> const & src) const
{
  if(this->param.output_data.write_output == true &&
     this->param.output_data.write_vorticity_magnitude == true)
  {
    // use the same implementation as for velocity_magnitude
    navier_stokes_operation->compute_velocity_magnitude(dst, src);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
  calculate_streamfunction(parallel::distributed::Vector<value_type> &       dst,
                           parallel::distributed::Vector<value_type> const & src) const
{
  if(this->param.output_data.write_output == true &&
     this->param.output_data.write_streamfunction == true)
  {
    navier_stokes_operation->compute_streamfunction(dst, src);
  }
}


template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::calculate_q_criterion(
  parallel::distributed::Vector<value_type> &       dst,
  parallel::distributed::Vector<value_type> const & src) const
{
  if(this->param.output_data.write_output == true &&
     this->param.output_data.write_q_criterion == true)
  {
    navier_stokes_operation->compute_q_criterion(dst, src);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::calculate_processor_id(
  parallel::distributed::Vector<value_type> & dst) const
{
  if(this->param.output_data.write_output == true &&
     this->param.output_data.write_processor_id == true)
  {
    dst = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
  calculate_mean_velocity(parallel::distributed::Vector<value_type> &       dst,
                          parallel::distributed::Vector<value_type> const & src) const
{
  if(this->param.output_data.write_output == true &&
     this->param.output_data.mean_velocity.calculate == true)
  {
    if(this->time >= this->param.output_data.mean_velocity.sample_start_time &&
       this->time <= this->param.output_data.mean_velocity.sample_end_time &&
       this->time_step_number % this->param.output_data.mean_velocity.sample_every_timesteps == 0)
    {
      dst.sadd((double)counter_mean_velocity, 1.0, src);
      ++counter_mean_velocity;
      dst *= 1. / (double)counter_mean_velocity;
    }
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::calculate_time_step()
{
  if(param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepUserSpecified)
  {
    time_steps[0] = calculate_const_time_step(param.time_step_size, n_refine_time);

    pcout << "User specified time step size:" << std::endl << std::endl;
    print_parameter(pcout, "time step size", time_steps[0]);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepCFL)
  {
    const double global_min_cell_diameter = calculate_minimum_vertex_distance(
      navier_stokes_operation->get_dof_handler_u().get_triangulation());

    double time_step = calculate_const_time_step_cfl(cfl,
                                                     param.max_velocity,
                                                     global_min_cell_diameter,
                                                     fe_degree_u,
                                                     param.cfl_exponent_fe_degree_velocity);

    // decrease time_step in order to exactly hit end_time
    time_steps[0] = (param.end_time - param.start_time) /
                    (1 + int((param.end_time - param.start_time) / time_step));

    pcout << "Calculation of time step size according to CFL condition:" << std::endl << std::endl;

    print_parameter(pcout, "h_min", global_min_cell_diameter);
    print_parameter(pcout, "U_max", param.max_velocity);
    print_parameter(pcout, "CFL", cfl);
    print_parameter(pcout, "exponent fe_degree_velocity", param.cfl_exponent_fe_degree_velocity);
    print_parameter(pcout, "Time step size", time_steps[0]);
  }
  else if(adaptive_time_stepping == true)
  {
    const double global_min_cell_diameter = calculate_minimum_vertex_distance(
      navier_stokes_operation->get_dof_handler_u().get_triangulation());

    // calculate a temporary time step size using a  guess for the maximum velocity
    double time_step_tmp = calculate_const_time_step_cfl(cfl,
                                                         param.max_velocity,
                                                         global_min_cell_diameter,
                                                         fe_degree_u,
                                                         param.cfl_exponent_fe_degree_velocity);

    pcout << "Calculation of time step size according to CFL condition:" << std::endl << std::endl;

    print_parameter(pcout, "h_min", global_min_cell_diameter);
    print_parameter(pcout, "U_max", param.max_velocity);
    print_parameter(pcout, "CFL", cfl);
    print_parameter(pcout, "exponent fe_degree_velocity", param.cfl_exponent_fe_degree_velocity);
    print_parameter(pcout, "Time step size", time_step_tmp);

    // if u(x,t=0)=0, this time step size will tend to infinity
    time_steps[0] = calculate_adaptive_time_step_cfl<dim, fe_degree_u, value_type>(
      navier_stokes_operation->get_data(),
      navier_stokes_operation->get_dof_index_velocity(),
      navier_stokes_operation->get_quad_index_velocity_linear(),
      get_velocity(),
      cfl,
      param.cfl_exponent_fe_degree_velocity);

    // use adaptive time step size only if it is smaller, otherwise use temporary time step size
    time_steps[0] = std::min(time_steps[0], time_step_tmp);

    pcout << std::endl
          << "Calculation of time step size according to adaptive CFL condition:" << std::endl
          << std::endl;

    print_parameter(pcout, "CFL", cfl);
    print_parameter(pcout, "exponent fe_degree_velocity", param.cfl_exponent_fe_degree_velocity);
    print_parameter(pcout, "Time step size", time_steps[0]);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepMaxEfficiency)
  {
    const double global_min_cell_diameter = calculate_minimum_vertex_distance(
      navier_stokes_operation->get_dof_handler_u().get_triangulation());

    double time_step = calculate_time_step_max_efficiency(
      param.c_eff, global_min_cell_diameter, fe_degree_u, order, n_refine_time);

    // decrease time_step in order to exactly hit end_time
    time_steps[0] = (param.end_time - param.start_time) /
                    (1 + int((param.end_time - param.start_time) / time_step));

    pcout << "Calculation of time step size (max efficiency):" << std::endl << std::endl;
    print_parameter(pcout, "C_eff", param.c_eff / std::pow(2, n_refine_time));
    print_parameter(pcout, "Time step size", time_steps[0]);
  }
  else
  {
    AssertThrow(
      param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepUserSpecified ||
        param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepCFL ||
        param.calculation_of_time_step_size == TimeStepCalculation::AdaptiveTimeStepCFL ||
        param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepMaxEfficiency,
      ExcMessage(
        "User did not specify how to calculate time step size - "
        "possibilities are ConstTimeStepUserSpecified, ConstTimeStepCFL  and AdaptiveTimeStepCFL."));
  }

  // fill time_steps array
  for(unsigned int i = 1; i < order; ++i)
    time_steps[i] = time_steps[0];

  if(this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    // make sure that CFL condition is used for the calculation of the time step size (the aim
    // of the OIF splitting approach is to overcome limitations of the CFL condition)
    AssertThrow(
      param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepCFL,
      ExcMessage(
        "Specified calculation of time step size not compatible with OIF splitting approach!"));

    // calculate number of substeps M
    double tol = 1.0e-6;
    M          = (int)(this->cfl / (cfl_oif - tol));

    if(cfl_oif < this->cfl / double(M) - tol)
      M += 1;

    // calculate substepping time step size delta_s
    delta_s = this->time_steps[0] / (double)M;

    pcout << std::endl
          << "Calculation of OIF substepping time step size:" << std::endl
          << std::endl;

    print_parameter(pcout, "CFL (OIF)", cfl_oif);
    print_parameter(pcout, "Number of substeps", M);
    print_parameter(pcout, "Substepping time step size", delta_s);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
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

  for(unsigned int i = order - 1; i > 0; --i)
    time_steps[i] = time_steps[i - 1];

  value_type last_time_step = time_steps[0];

  time_steps[0] = calculate_adaptive_time_step_cfl<dim, fe_degree_u, value_type>(
    navier_stokes_operation->get_data(),
    navier_stokes_operation->get_dof_index_velocity(),
    navier_stokes_operation->get_quad_index_velocity_linear(),
    get_velocity(),
    cfl,
    param.cfl_exponent_fe_degree_velocity);

  bool use_limiter = true;
  if(use_limiter == true)
  {
    double factor = param.adaptive_time_stepping_limiting_factor;
    limit_time_step_change(time_steps[0], last_time_step, factor);
  }
}


template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::timeloop()
{
  pcout << std::endl << "Starting time loop ..." << std::endl;

  global_timer.restart();

  postprocessing();

  // TODO
  //  postprocessing_stability_analysis();

  // a small number which is much smaller than the time step size
  const value_type EPSILON = 1.0e-10;

  while(time < (param.end_time - EPSILON) && time_step_number <= param.max_number_of_time_steps)
  {
    do_timestep();

    postprocessing();
  }

  total_time += global_timer.wall_time();

  pcout << std::endl << "... done!" << std::endl;

  analyze_computing_times();
}

/*
 *  For the two-domain solver we only want to advance one time step because
 *  the solvers for the two domains have to communicate between the time steps.
 */
template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
bool
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::advance_one_timestep(
  bool write_final_output)
{
  // a small number which is much smaller than the time step size
  const value_type EPSILON = 1.0e-10;

  bool started = time > (param.start_time - EPSILON);

  // If the time integrator has not yet started, simply increment physical
  // time without solving the current time step.
  if(!started)
  {
    time += time_steps[0];
  }

  if(started && this->time_step_number == 1)
  {
    pcout << std::endl << "Starting time loop ..." << std::endl;

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

/*
 *  Implementation of pseudo-timestepping to solve steady-state problems
 *  applying an unsteady solution approach.
 *  The aim/motivation is to obtain a solution algorithm that allows to
 *  solve the steady Navier-Stokes equations more efficiently for large
 *  Reynolds numbers as compared to a steady-state solver for which preconditioning
 *  of the linearized, coupled system of equations becomes more difficult for
 *  large Re numbers.
 *  This solver differs from the unsteady solver only in the way that the simulation
 *  is terminated in case a convergence criterion is fulfilled.
 */
template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
  timeloop_steady_problem()
{
  global_timer.restart();

  postprocessing_steady_problem();

  solve_steady_problem();

  postprocessing_steady_problem();

  total_time += this->global_timer.wall_time();

  analyze_computing_times();
}

/*
 *  Solve on time step including the update of time integrator constants, counters,
 *  the time step size (in case of adaptive time stepping) and the update of solution
 *  vectors so that everything is pepared for the next time step.
 */
template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::do_timestep()
{
  update_time_integrator_constants();

  output_solver_info_header();

  solve_timestep();

  output_remaining_time();

  prepare_vectors_for_next_timestep();

  time += time_steps[0];
  ++time_step_number;

  if(param.write_restart == true)
    write_restart();

  if(adaptive_time_stepping == true)
    recalculate_adaptive_time_step();
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
  output_solver_info_header() const
{
  if(this->time_step_number % this->param.output_solver_info_every_timesteps == 0)
  {
    this->pcout << std::endl
                << "______________________________________________________________________"
                << std::endl
                << std::endl
                << " Number of TIME STEPS: " << std::left << std::setw(8) << this->time_step_number
                << "t_n = " << std::scientific << std::setprecision(4) << this->time
                << " -> t_n+1 = " << this->time + this->time_steps[0] << std::endl
                << "______________________________________________________________________"
                << std::endl;
  }
}

/*
 *  This function estimates the remaining wall time based on the overall time interval to be
 * simulated and the measured wall time already needed to simulate from the start time until the
 * current time.
 */
template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::output_remaining_time()
  const
{
  if(this->time_step_number % this->param.output_solver_info_every_timesteps == 0)
  {
    if(this->time > this->param.start_time)
    {
      double const remaining_time = this->global_timer.wall_time() *
                                    (this->param.end_time - this->time) /
                                    (this->time - this->param.start_time);

      this->pcout << std::endl
                  << "Estimated time until completion is " << remaining_time << " s / "
                  << remaining_time / 3600. << " h." << std::endl;
    }
  }
}

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_NAVIER_STOKES_H_ */
