/*
 * time_int_bdf_navier_stokes.cpp
 *
 *  Created on: Nov 15, 2018
 *      Author: fehn
 */

#include <deal.II/lac/vector_view.h>

#include "../interface_space_time/operator.h"
#include "../user_interface/input_parameters.h"
#include "time_int_bdf_navier_stokes.h"
#include "time_integration/time_step_calculation.h"

namespace IncNS
{
template<int dim, typename Number>
TimeIntBDF<dim, Number>::TimeIntBDF(std::shared_ptr<InterfaceBase> operator_in,
                                    InputParameters<dim> const &   param_in,
                                    unsigned int const             n_refine_time_in)
  : TimeIntBDFBase(param_in.start_time,
                   param_in.end_time,
                   param_in.max_number_of_time_steps,
                   param_in.order_time_integrator,
                   param_in.start_with_low_order,
                   param_in.adaptive_time_stepping,
                   param_in.restart_data),
    param(param_in),
    cfl(param.cfl / std::pow(2.0, n_refine_time_in)),
    cfl_oif(param_in.cfl_oif / std::pow(2.0, n_refine_time_in)),
    operator_base(operator_in),
    n_refine_time(n_refine_time_in)
{
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::update_time_integrator_constants()
{
  // call function of base class to update the standard time integrator constants
  TimeIntBDFBase::update_time_integrator_constants();
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::initialize_oif()
{
  // Operator-integration-factor splitting
  if(param.equation_type == EquationType::NavierStokes &&
     param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    convective_operator_OIF.reset(new Interface::OperatorOIF<Number>(operator_base));

    // initialize OIF time integrator
    if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK1Stage1)
    {
      time_integrator_OIF.reset(
        new ExplicitRungeKuttaTimeIntegrator<Interface::OperatorOIF<Number>, VectorType>(
          1, convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK2Stage2)
    {
      time_integrator_OIF.reset(
        new ExplicitRungeKuttaTimeIntegrator<Interface::OperatorOIF<Number>, VectorType>(
          2, convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK3Stage3)
    {
      time_integrator_OIF.reset(
        new ExplicitRungeKuttaTimeIntegrator<Interface::OperatorOIF<Number>, VectorType>(
          3, convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK4Stage4)
    {
      time_integrator_OIF.reset(
        new ExplicitRungeKuttaTimeIntegrator<Interface::OperatorOIF<Number>, VectorType>(
          4, convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK3Stage4Reg2C)
    {
      time_integrator_OIF.reset(
        new LowStorageRK3Stage4Reg2C<Interface::OperatorOIF<Number>, VectorType>(
          convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK4Stage5Reg2C)
    {
      time_integrator_OIF.reset(
        new LowStorageRK4Stage5Reg2C<Interface::OperatorOIF<Number>, VectorType>(
          convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK4Stage5Reg3C)
    {
      time_integrator_OIF.reset(
        new LowStorageRK4Stage5Reg3C<Interface::OperatorOIF<Number>, VectorType>(
          convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK5Stage9Reg2S)
    {
      time_integrator_OIF.reset(
        new LowStorageRK5Stage9Reg2S<Interface::OperatorOIF<Number>, VectorType>(
          convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK3Stage7Reg2)
    {
      time_integrator_OIF.reset(new LowStorageRKTD<Interface::OperatorOIF<Number>, VectorType>(
        convective_operator_OIF, 3, 7));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK4Stage8Reg2)
    {
      time_integrator_OIF.reset(new LowStorageRKTD<Interface::OperatorOIF<Number>, VectorType>(
        convective_operator_OIF, 4, 8));
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    // temporary vectors required for operator-integration-factor splitting of convective term
    operator_base->initialize_vector_velocity(solution_tilde_m);
    operator_base->initialize_vector_velocity(solution_tilde_mp);
  }
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::read_restart_vectors(boost::archive::binary_iarchive & ia)
{
  Vector<double> tmp;
  for(unsigned int i = 0; i < this->order; i++)
  {
    ia >> tmp;
    VectorType velocity_copy = get_velocity(i);
    std::copy(tmp.begin(), tmp.end(), velocity_copy.begin());
    set_velocity(velocity_copy, i);
  }
  for(unsigned int i = 0; i < this->order; i++)
  {
    ia >> tmp;
    VectorType pressure_copy = get_pressure(i);
    std::copy(tmp.begin(), tmp.end(), pressure_copy.begin());
    set_pressure(pressure_copy, i);
  }
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::write_restart_vectors(boost::archive::binary_oarchive & oa) const
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  for(unsigned int i = 0; i < this->order; i++)
  {
    VectorView<Number> vector_view(get_velocity(i).local_size(), get_velocity(i).begin());
    oa << vector_view;
  }
  for(unsigned int i = 0; i < this->order; i++)
  {
    VectorView<Number> vector_view(get_pressure(i).local_size(), get_pressure(i).begin());
    oa << vector_view;
  }

#pragma GCC diagnostic pop
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::calculate_time_step_size()
{
  unsigned int const degree_u = operator_base->get_polynomial_degree();

  if(param.calculation_of_time_step_size == TimeStepCalculation::UserSpecified)
  {
    double const time_step = calculate_const_time_step(param.time_step_size, n_refine_time);

    this->set_time_step_size(time_step);

    pcout << "User specified time step size:" << std::endl << std::endl;
    print_parameter(pcout, "time step size", time_step);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::CFL)
  {
    double time_step = 1.0;

    double const h_min = operator_base->calculate_minimum_element_length();

    double time_step_global = calculate_time_step_cfl_global(
      cfl, param.max_velocity, h_min, degree_u, param.cfl_exponent_fe_degree_velocity);

    pcout << "Calculation of time step size according to CFL condition:" << std::endl << std::endl;
    print_parameter(pcout, "h_min", h_min);
    print_parameter(pcout, "U_max", param.max_velocity);
    print_parameter(pcout, "CFL", cfl);
    print_parameter(pcout, "exponent fe_degree", param.cfl_exponent_fe_degree_velocity);
    print_parameter(pcout, "Time step size (global)", time_step_global);

    if(adaptive_time_stepping == true)
    {
      // if u(x,t=0)=0, this time step size will tend to infinity
      double time_step_adap =
        operator_base->calculate_time_step_cfl(get_velocity(),
                                               cfl,
                                               param.cfl_exponent_fe_degree_velocity);

      // use adaptive time step size only if it is smaller, otherwise use temporary time step size
      time_step = std::min(time_step_adap, time_step_global);

      print_parameter(pcout, "Time step size (adaptive)", time_step);
    }
    else
    {
      time_step =
        adjust_time_step_to_hit_end_time(param.start_time, param.end_time, time_step_global);

      pcout << std::endl << "Adjust time step size to hit end time:" << std::endl << std::endl;
      print_parameter(pcout, "Time step size", time_step);
    }

    this->set_time_step_size(time_step);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::MaxEfficiency)
  {
    double const h_min = operator_base->calculate_minimum_element_length();

    double time_step =
      calculate_time_step_max_efficiency(param.c_eff, h_min, degree_u, order, n_refine_time);

    time_step = adjust_time_step_to_hit_end_time(param.start_time, param.end_time, time_step);

    this->set_time_step_size(time_step);

    pcout << "Calculation of time step size (max efficiency):" << std::endl << std::endl;
    print_parameter(pcout, "C_eff", param.c_eff / std::pow(2, n_refine_time));
    print_parameter(pcout, "Time step size", time_step);
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified type of time step calculation is not implemented."));
  }

  if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    // make sure that CFL condition is used for the calculation of the time step size (the aim
    // of the OIF splitting approach is to overcome limitations of the CFL condition)
    AssertThrow(param.calculation_of_time_step_size == TimeStepCalculation::CFL,
                ExcMessage(
                  "Specified type of time step calculation is not compatible with OIF approach!"));

    pcout << std::endl << "OIF sub-stepping for convective term:" << std::endl << std::endl;
    print_parameter(pcout, "CFL (OIF)", cfl_oif);
  }
}

template<int dim, typename Number>
double
TimeIntBDF<dim, Number>::recalculate_time_step_size() const
{
  double new_time_step_size =
    operator_base->calculate_time_step_cfl(get_velocity(),
                                           cfl,
                                           param.cfl_exponent_fe_degree_velocity);

  bool use_limiter = true;
  if(use_limiter)
  {
    double last_time_step_size = this->get_time_step_size();
    double factor              = param.adaptive_time_stepping_limiting_factor;
    limit_time_step_change(new_time_step_size, last_time_step_size, factor);
  }

  return new_time_step_size;
}

template<int dim, typename Number>
bool
TimeIntBDF<dim, Number>::print_solver_info() const
{
  return get_time_step_number() % param.output_solver_info_every_timesteps == 0;
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::initialize_solution_oif_substepping(unsigned int i)
{
  // initialize solution: u_tilde(s=0) = u(t_{n-i})
  solution_tilde_m = get_velocity(i);
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::calculate_sum_alphai_ui_oif_substepping(double const cfl,
                                                                 double const cfl_oif)
{
  /*
   * the convective term is nonlinear, so we have to initialize the transport velocity
   * and the discrete time instants that can be used for interpolation
   *
   *   time t
   *  -------->   t_{n-2}   t_{n-1}   t_{n}     t_{n+1}
   *  _______________|_________|________|___________|___________\
   *                 |         |        |           |           /
   *               sol[2]    sol[1]   sol[0]
   */
  unsigned int current_order = this->order;
  if(this->time_step_number <= this->order && this->param.start_with_low_order == true)
  {
    current_order = this->time_step_number;
  }

  AssertThrow(current_order > 0 && current_order <= this->order,
              ExcMessage("Invalid parameter current_order"));

  // fill vectors with previous velocity solutions and previous time instants
  std::vector<VectorType const *> solutions(current_order);
  std::vector<double>             times(current_order);

  for(unsigned int i = 0; i < current_order; ++i)
  {
    solutions.at(i) = &get_velocity(i);
    times.at(i)     = get_previous_time(i);
  }

  // this is only needed for transport with interpolated/extrapolated velocity
  // as opposed to the standard nonlinear transport
  this->convective_operator_OIF->set_solutions_and_times(solutions, times);

  // call function implemented in base class for the actual OIF sub-stepping
  TimeIntBDFBase::calculate_sum_alphai_ui_oif_substepping(cfl, cfl_oif);
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::update_sum_alphai_ui_oif_substepping(unsigned int i)
{
  // calculate sum (alpha_i/dt * u_tilde_i)
  if(i == 0)
    sum_alphai_ui.equ(bdf.get_alpha(i) / this->get_time_step_size(), solution_tilde_m);
  else // i>0
    sum_alphai_ui.add(bdf.get_alpha(i) / this->get_time_step_size(), solution_tilde_m);
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::do_timestep_oif_substepping_and_update_vectors(double const start_time,
                                                                        double const time_step_size)
{
  // solve sub-step
  time_integrator_OIF->solve_timestep(solution_tilde_mp,
                                      solution_tilde_m,
                                      start_time,
                                      time_step_size);

  solution_tilde_mp.swap(solution_tilde_m);
}

// instantiations
#include <navierstokes/config.h>

// float
#if DIM_2 && OP_FLOAT
template class TimeIntBDF<2, float>;
#endif
#if DIM_3 && OP_FLOAT
template class TimeIntBDF<3, float>;
#endif

// double
#if DIM_2 && OP_DOUBLE
template class TimeIntBDF<2, double>;
#endif
#if DIM_3 && OP_DOUBLE
template class TimeIntBDF<3, double>;
#endif

} // namespace IncNS
