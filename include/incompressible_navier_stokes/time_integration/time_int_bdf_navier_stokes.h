/*
 * time_int_bdf_navier_stokes.h
 *
 *  Created on: Jun 10, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_NAVIER_STOKES_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_NAVIER_STOKES_H_

#include "time_integration/time_int_bdf_base.h"

#include "../../incompressible_navier_stokes/time_integration/restart.h"
#include "time_integration/explicit_runge_kutta.h"
#include "time_integration/time_step_calculation.h"

namespace IncNS
{
template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
class TimeIntBDFNavierStokes : public TimeIntBDFBase
{
public:
  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

  TimeIntBDFNavierStokes(std::shared_ptr<NavierStokesOperation> navier_stokes_operation_in,
                         InputParameters<dim> const &           param_in,
                         unsigned int const                     n_refine_time_in,
                         bool const                             use_adaptive_time_stepping_in)
    : TimeIntBDFBase(param_in.start_time,
                     param_in.end_time,
                     param_in.max_number_of_time_steps,
                     param_in.order_time_integrator,
                     param_in.start_with_low_order,
                     use_adaptive_time_stepping_in,
                     param_in.write_restart),
      param(param_in),
      cfl(param.cfl / std::pow(2.0, n_refine_time_in)),
      cfl_oif(param_in.cfl_oif / std::pow(2.0, n_refine_time_in)),
      n_refine_time(n_refine_time_in),
      navier_stokes_operation(navier_stokes_operation_in)
  {
  }

  virtual ~TimeIntBDFNavierStokes()
  {
  }

  void
  setup(bool do_restart);

protected:
  virtual void
  update_time_integrator_constants();

  virtual void
  initialize_vectors() = 0;

  void
  initialize_oif();

  void
  initialize_solution_oif_substepping(unsigned int i);

  void
  update_sum_alphai_ui_oif_substepping(unsigned int i);

  void
  do_timestep_oif_substepping_and_update_vectors(double const start_time,
                                                 double const time_step_size);

  virtual void
  calculate_time_step();

  virtual void
  recalculate_adaptive_time_step();

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

  // BDF time integration: Sum_i (alpha_i/dt * u_i)
  VectorType sum_alphai_ui;

  // gobal cfl number
  double const cfl;

  // Operator-integration-factor splitting for convective term
  std::shared_ptr<ConvectiveOperatorNavierStokes<NavierStokesOperation, value_type>>
    convective_operator_OIF;

  // OIF splitting
  std::shared_ptr<
    ExplicitTimeIntegrator<ConvectiveOperatorNavierStokes<NavierStokesOperation, value_type>,
                           VectorType>>
    time_integrator_OIF;

  // cfl number cfl_oif for operator-integration-factor splitting
  double const cfl_oif;

  // solution vectors needed for OIF substepping of convective term
  VectorType solution_tilde_m;
  VectorType solution_tilde_mp;

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
  solve_steady_problem() = 0;

  virtual void
  postprocessing_steady_problem() const = 0;

  virtual void
  prepare_vectors_for_next_timestep() = 0;

  virtual VectorType const &
  get_velocity() = 0;

  virtual VectorType const &
  get_velocity(unsigned int i /* t_{n-i} */) = 0;

  unsigned int const n_refine_time;

  std::shared_ptr<NavierStokesOperation> navier_stokes_operation;
};

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::setup(bool do_restart)
{
  pcout << std::endl << "Setup time integrator ..." << std::endl << std::endl;

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
  initialize_solution_and_calculate_timestep(bool do_restart)
{
  if(do_restart)
  {
    resume_from_restart();

    // if anything in the temporal discretization is changed, start_with_low_order has to be set to
    // true otherwise the old solutions would not fit the time step increments, etc.
    if(start_with_low_order == true)
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
    if(start_with_low_order == false)
      initialize_former_solution();
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
  update_time_integrator_constants()
{
  // call function of base class to update the standard time integrator constants
  TimeIntBDFBase::update_time_integrator_constants();
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::initialize_oif()
{
  // Operator-integration-factor splitting
  if(param.equation_type == EquationType::NavierStokes &&
     param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    convective_operator_OIF.reset(
      new ConvectiveOperatorNavierStokes<NavierStokesOperation, value_type>(
        navier_stokes_operation));

    // initialize OIF time integrator
    if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK1Stage1)
    {
      time_integrator_OIF.reset(new ExplicitRungeKuttaTimeIntegrator<
                                ConvectiveOperatorNavierStokes<NavierStokesOperation, value_type>,
                                VectorType>(1, convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK2Stage2)
    {
      time_integrator_OIF.reset(new ExplicitRungeKuttaTimeIntegrator<
                                ConvectiveOperatorNavierStokes<NavierStokesOperation, value_type>,
                                VectorType>(2, convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK3Stage3)
    {
      time_integrator_OIF.reset(new ExplicitRungeKuttaTimeIntegrator<
                                ConvectiveOperatorNavierStokes<NavierStokesOperation, value_type>,
                                VectorType>(3, convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK4Stage4)
    {
      time_integrator_OIF.reset(new ExplicitRungeKuttaTimeIntegrator<
                                ConvectiveOperatorNavierStokes<NavierStokesOperation, value_type>,
                                VectorType>(4, convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK3Stage4Reg2C)
    {
      time_integrator_OIF.reset(new LowStorageRK3Stage4Reg2C<
                                ConvectiveOperatorNavierStokes<NavierStokesOperation, value_type>,
                                VectorType>(convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK4Stage5Reg2C)
    {
      time_integrator_OIF.reset(new LowStorageRK4Stage5Reg2C<
                                ConvectiveOperatorNavierStokes<NavierStokesOperation, value_type>,
                                VectorType>(convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK4Stage5Reg3C)
    {
      time_integrator_OIF.reset(new LowStorageRK4Stage5Reg3C<
                                ConvectiveOperatorNavierStokes<NavierStokesOperation, value_type>,
                                VectorType>(convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK5Stage9Reg2S)
    {
      time_integrator_OIF.reset(new LowStorageRK5Stage9Reg2S<
                                ConvectiveOperatorNavierStokes<NavierStokesOperation, value_type>,
                                VectorType>(convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK3Stage7Reg2)
    {
      time_integrator_OIF.reset(
        new LowStorageRKTD<ConvectiveOperatorNavierStokes<NavierStokesOperation, value_type>,
                           VectorType>(convective_operator_OIF, 3, 7));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK4Stage8Reg2)
    {
      time_integrator_OIF.reset(
        new LowStorageRKTD<ConvectiveOperatorNavierStokes<NavierStokesOperation, value_type>,
                           VectorType>(convective_operator_OIF, 4, 8));
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

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

  double              time_local = 0.0;
  std::vector<double> time_steps_local(order, -1.0);
  resume_restart<dim, value_type>(ia, param, time_local, time_steps_local, order);
  this->reset_time(time_local);
  this->reset_time_step_vector(time_steps_local);

  read_restart_vectors(ia);

  finished_reading_restart_output();
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::write_restart() const
{
  const double EPSILON = 1.0e-10; // small number which is much smaller than the time step size

  const double wall_time = global_timer.wall_time();
  if((std::fmod(this->get_time(), param.restart_interval_time) <
        this->get_time_step_size() + EPSILON &&
      this->get_time() > param.restart_interval_time - EPSILON) ||
     (std::fmod(wall_time, param.restart_interval_wall_time) < wall_time - total_time) ||
     (get_time_step_number() % param.restart_every_timesteps == 0))
  {
    std::ostringstream oss;

    boost::archive::binary_oarchive oa(oss);
    write_restart_preamble<dim, value_type>(
      oa, param, this->get_time_step_vector(), this->get_time(), order);
    write_restart_vectors(oa);
    write_restart_file<dim>(oss, param);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::calculate_time_step()
{
  if(param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepUserSpecified)
  {
    double const time_step = calculate_const_time_step(param.time_step_size, n_refine_time);

    this->set_time_step_size(time_step);

    pcout << "User specified time step size:" << std::endl << std::endl;
    print_parameter(pcout, "time step size", time_step);
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
    time_step = (param.end_time - param.start_time) /
                (1 + int((param.end_time - param.start_time) / time_step));

    this->set_time_step_size(time_step);

    pcout << "Calculation of time step size according to CFL condition:" << std::endl << std::endl;

    print_parameter(pcout, "h_min", global_min_cell_diameter);
    print_parameter(pcout, "U_max", param.max_velocity);
    print_parameter(pcout, "CFL", cfl);
    print_parameter(pcout, "exponent fe_degree_velocity", param.cfl_exponent_fe_degree_velocity);
    print_parameter(pcout, "Time step size", time_step);
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
    double time_step_adap = calculate_adaptive_time_step_cfl<dim, fe_degree_u, value_type>(
      navier_stokes_operation->get_data(),
      navier_stokes_operation->get_dof_index_velocity(),
      navier_stokes_operation->get_quad_index_velocity_linear(),
      get_velocity(),
      cfl,
      param.cfl_exponent_fe_degree_velocity);

    // use adaptive time step size only if it is smaller, otherwise use temporary time step size
    time_step_adap = std::min(time_step_adap, time_step_tmp);

    this->set_time_step_size(time_step_adap);

    pcout << std::endl
          << "Calculation of time step size according to adaptive CFL condition:" << std::endl
          << std::endl;

    print_parameter(pcout, "CFL", cfl);
    print_parameter(pcout, "exponent fe_degree_velocity", param.cfl_exponent_fe_degree_velocity);
    print_parameter(pcout, "Time step size", time_step_adap);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepMaxEfficiency)
  {
    const double global_min_cell_diameter = calculate_minimum_vertex_distance(
      navier_stokes_operation->get_dof_handler_u().get_triangulation());

    double time_step = calculate_time_step_max_efficiency(
      param.c_eff, global_min_cell_diameter, fe_degree_u, order, n_refine_time);

    // decrease time_step in order to exactly hit end_time
    time_step = (end_time - start_time) / (1 + int((end_time - start_time) / time_step));

    this->set_time_step_size(time_step);

    pcout << "Calculation of time step size (max efficiency):" << std::endl << std::endl;
    print_parameter(pcout, "C_eff", param.c_eff / std::pow(2, n_refine_time));
    print_parameter(pcout, "Time step size", time_step);
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

  if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    // make sure that CFL condition is used for the calculation of the time step size (the aim
    // of the OIF splitting approach is to overcome limitations of the CFL condition)
    AssertThrow(
      param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepCFL,
      ExcMessage(
        "Specified calculation of time step size not compatible with OIF splitting approach!"));

    pcout << std::endl << "OIF substepping for convective term:" << std::endl << std::endl;

    print_parameter(pcout, "CFL (OIF)", cfl_oif);
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
   *                    dt[1]  <- dt[0] <- new_dt
   *
   */
  this->push_back_time_step_sizes();

  double last_time_step = this->get_time_step_size();

  double new_time_step = calculate_adaptive_time_step_cfl<dim, fe_degree_u, value_type>(
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
    limit_time_step_change(new_time_step, last_time_step, factor);
  }

  this->set_time_step_size(new_time_step);
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
  output_solver_info_header() const
{
  if(get_time_step_number() % param.output_solver_info_every_timesteps == 0)
  {
    pcout << std::endl
          << "______________________________________________________________________" << std::endl
          << std::endl
          << " Number of TIME STEPS: " << std::left << std::setw(8) << get_time_step_number()
          << "t_n = " << std::scientific << std::setprecision(4) << this->get_time()
          << " -> t_n+1 = " << this->get_next_time() << std::endl
          << "______________________________________________________________________" << std::endl;
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
  if(get_time_step_number() % param.output_solver_info_every_timesteps == 0)
  {
    if(this->get_time() > start_time)
    {
      double const remaining_time =
        global_timer.wall_time() * (end_time - this->get_time()) / (this->get_time() - start_time);

      pcout << std::endl
            << "Estimated time until completion is " << remaining_time << " s / "
            << remaining_time / 3600. << " h." << std::endl;
    }
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
  initialize_solution_oif_substepping(unsigned int i)
{
  // initialize solution: u_tilde(s=0) = u(t_{n-i})
  solution_tilde_m = get_velocity(i);
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
  update_sum_alphai_ui_oif_substepping(unsigned int i)
{
  // calculate sum (alpha_i/dt * u_tilde_i)
  if(i == 0)
    sum_alphai_ui.equ(bdf.get_alpha(i) / this->get_time_step_size(), solution_tilde_m);
  else // i>0
    sum_alphai_ui.add(bdf.get_alpha(i) / this->get_time_step_size(), solution_tilde_m);
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
  do_timestep_oif_substepping_and_update_vectors(double const start_time,
                                                 double const time_step_size)
{
  // solve sub-step
  time_integrator_OIF->solve_timestep(solution_tilde_mp,
                                      solution_tilde_m,
                                      start_time,
                                      time_step_size);

  solution_tilde_mp.swap(solution_tilde_m);
}


} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_NAVIER_STOKES_H_ */
