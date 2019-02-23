/*
 * time_int_bdf.cpp
 *
 *  Created on: Nov 14, 2018
 *      Author: fehn
 */

#include "time_int_bdf.h"

#include "time_integration/push_back_vectors.h"
#include "time_integration/time_step_calculation.h"

#include "../interface_space_time/operator.h"
#include "../user_interface/input_parameters.h"

namespace ConvDiff
{
template<typename Number>
TimeIntBDF<Number>::TimeIntBDF(std::shared_ptr<Operator> operator_in,
                               InputParameters const &   param_in,
                               unsigned int const        n_refine_time_in)
  : TimeIntBDFBase(param_in.start_time,
                   param_in.end_time,
                   param_in.max_number_of_time_steps,
                   param_in.order_time_integrator,
                   param_in.start_with_low_order,
                   param_in.adaptive_time_stepping,
                   param_in.restart_data),
    pde_operator(operator_in),
    param(param_in),
    n_refine_time(n_refine_time_in),
    cfl(param.cfl / std::pow(2.0, n_refine_time_in)),
    solution(param_in.order_time_integrator),
    vec_convective_term(param_in.order_time_integrator),
    iterations(0.0),
    wall_time(0.0),
    cfl_oif(param.cfl_oif / std::pow(2.0, n_refine_time_in))
{
}

template<typename Number>
void
TimeIntBDF<Number>::setup_derived()
{
  // Initialize vec_convective_term: Note that this function has to be called
  // after the solution has been initialized because the solution is evaluated in this function.
  if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit &&
     (param.equation_type == EquationType::Convection ||
      param.equation_type == EquationType::ConvectionDiffusion) &&
     start_with_low_order == false)
  {
    initialize_vec_convective_term();
  }
}

template<typename Number>
void
TimeIntBDF<Number>::initialize_oif()
{
  // Operator-integration-factor splitting
  if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    convective_operator_OIF.reset(new Interface::OperatorOIF<Number>(pde_operator));

    if(param.time_integrator_oif == TimeIntegratorRK::ExplRK1Stage1)
    {
      time_integrator_OIF.reset(
        new ExplicitRungeKuttaTimeIntegrator<Interface::OperatorOIF<Number>, VectorType>(
          1, convective_operator_OIF));
    }
    else if(param.time_integrator_oif == TimeIntegratorRK::ExplRK2Stage2)
    {
      time_integrator_OIF.reset(
        new ExplicitRungeKuttaTimeIntegrator<Interface::OperatorOIF<Number>, VectorType>(
          2, convective_operator_OIF));
    }
    else if(param.time_integrator_oif == TimeIntegratorRK::ExplRK3Stage3)
    {
      time_integrator_OIF.reset(
        new ExplicitRungeKuttaTimeIntegrator<Interface::OperatorOIF<Number>, VectorType>(
          3, convective_operator_OIF));
    }
    else if(param.time_integrator_oif == TimeIntegratorRK::ExplRK4Stage4)
    {
      time_integrator_OIF.reset(
        new ExplicitRungeKuttaTimeIntegrator<Interface::OperatorOIF<Number>, VectorType>(
          4, convective_operator_OIF));
    }
    else if(param.time_integrator_oif == TimeIntegratorRK::ExplRK3Stage4Reg2C)
    {
      time_integrator_OIF.reset(
        new LowStorageRK3Stage4Reg2C<Interface::OperatorOIF<Number>, VectorType>(
          convective_operator_OIF));
    }
    else if(param.time_integrator_oif == TimeIntegratorRK::ExplRK4Stage5Reg2C)
    {
      time_integrator_OIF.reset(
        new LowStorageRK4Stage5Reg2C<Interface::OperatorOIF<Number>, VectorType>(
          convective_operator_OIF));
    }
    else if(param.time_integrator_oif == TimeIntegratorRK::ExplRK4Stage5Reg3C)
    {
      time_integrator_OIF.reset(
        new LowStorageRK4Stage5Reg3C<Interface::OperatorOIF<Number>, VectorType>(
          convective_operator_OIF));
    }
    else if(param.time_integrator_oif == TimeIntegratorRK::ExplRK5Stage9Reg2S)
    {
      time_integrator_OIF.reset(
        new LowStorageRK5Stage9Reg2S<Interface::OperatorOIF<Number>, VectorType>(
          convective_operator_OIF));
    }
    else if(param.time_integrator_oif == TimeIntegratorRK::ExplRK3Stage7Reg2)
    {
      time_integrator_OIF.reset(new LowStorageRKTD<Interface::OperatorOIF<Number>, VectorType>(
        convective_operator_OIF, 3, 7));
    }
    else if(param.time_integrator_oif == TimeIntegratorRK::ExplRK4Stage8Reg2)
    {
      time_integrator_OIF.reset(new LowStorageRKTD<Interface::OperatorOIF<Number>, VectorType>(
        convective_operator_OIF, 4, 8));
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }
}

template<typename Number>
void
TimeIntBDF<Number>::allocate_vectors()
{
  for(unsigned int i = 0; i < solution.size(); ++i)
    pde_operator->initialize_dof_vector(solution[i]);

  pde_operator->initialize_dof_vector(solution_np);

  pde_operator->initialize_dof_vector(sum_alphai_ui);
  pde_operator->initialize_dof_vector(rhs_vector);

  if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit &&
     (param.equation_type == EquationType::Convection ||
      param.equation_type == EquationType::ConvectionDiffusion))
  {
    for(unsigned int i = 0; i < vec_convective_term.size(); ++i)
      pde_operator->initialize_dof_vector(vec_convective_term[i]);
  }

  if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    pde_operator->initialize_dof_vector(solution_tilde_m);
    pde_operator->initialize_dof_vector(solution_tilde_mp);
  }
}

template<typename Number>
void
TimeIntBDF<Number>::initialize_current_solution()
{
  pde_operator->prescribe_initial_conditions(solution[0], this->get_time());
}

template<typename Number>
void
TimeIntBDF<Number>::initialize_former_solutions()
{
  // Start with i=1 since we only want to initialize the solution at former instants of time.
  for(unsigned int i = 1; i < solution.size(); ++i)
  {
    pde_operator->prescribe_initial_conditions(solution[i], this->get_previous_time(i));
  }
}

template<typename Number>
void
TimeIntBDF<Number>::initialize_vec_convective_term()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i = 1; i < vec_convective_term.size(); ++i)
  {
    pde_operator->evaluate_convective_term(vec_convective_term[i],
                                           solution[i],
                                           this->get_previous_time(i));
  }
}

template<typename Number>
void
TimeIntBDF<Number>::calculate_time_step_size()
{
  unsigned int const degree = pde_operator->get_polynomial_degree();

  if(param.calculation_of_time_step_size == TimeStepCalculation::UserSpecified)
  {
    double const time_step = calculate_const_time_step(param.time_step_size, n_refine_time);
    this->set_time_step_size(time_step);

    pcout << "Calculation of time step size (user-specified):" << std::endl << std::endl;
    print_parameter(pcout, "time step size", time_step);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::CFL)
  {
    AssertThrow(param.equation_type == EquationType::Convection ||
                  param.equation_type == EquationType::ConvectionDiffusion,
                ExcMessage("Specified type of time step calculation does not make sense!"));

    double time_step = 1.0;

    double const h_min = pde_operator->calculate_minimum_element_length();

    double max_velocity = 0.0;
    if(param.type_velocity_field == TypeVelocityField::Analytical)
    {
      max_velocity = pde_operator->calculate_maximum_velocity(this->get_time());
    }

    // max_velocity computed above might be zero depending on the initial velocity field -> dt would
    // tend to infinity
    max_velocity = std::max(max_velocity, param.max_velocity);

    double time_step_global = calculate_time_step_cfl_global(
      cfl, max_velocity, h_min, degree, param.exponent_fe_degree_convection);

    pcout << "Calculation of time step size according to CFL condition:" << std::endl << std::endl;
    print_parameter(pcout, "h_min", h_min);
    print_parameter(pcout, "U_max", max_velocity);
    print_parameter(pcout, "CFL", cfl);
    print_parameter(pcout, "Exponent fe_degree", param.exponent_fe_degree_convection);
    print_parameter(pcout, "Time step size (global)", time_step_global);

    if(adaptive_time_stepping == true)
    {
      double time_step_adap = std::numeric_limits<double>::max();

      if(param.type_velocity_field == TypeVelocityField::Analytical)
      {
        time_step_adap = pde_operator->calculate_time_step_cfl(this->get_time(),
                                                               cfl,
                                                               param.exponent_fe_degree_convection);
      }
      else if(param.type_velocity_field == TypeVelocityField::Numerical)
      {
        time_step_adap =
          pde_operator->calculate_time_step_cfl(cfl, param.exponent_fe_degree_convection);
      }
      else
      {
        AssertThrow(false, ExcMessage("Not implemented."));
      }

      // use adaptive time step size only if it is smaller, otherwise use global time step size
      time_step = std::min(time_step_adap, time_step_global);

      print_parameter(pcout, "Time step size (adaptive)", time_step);
    }
    else // constant time step size
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
    // calculate minimum vertex distance
    double const h_min = pde_operator->calculate_minimum_element_length();

    double time_step =
      calculate_time_step_max_efficiency(param.c_eff, h_min, degree, order, n_refine_time);

    time_step = adjust_time_step_to_hit_end_time(param.start_time, param.end_time, time_step);

    this->set_time_step_size(time_step);

    pcout << std::endl
          << "Calculation of time step size (max efficiency):" << std::endl
          << std::endl;
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
    AssertThrow(
      param.calculation_of_time_step_size == TimeStepCalculation::CFL,
      ExcMessage(
        "Specified type of time step calculation is not compatible with OIF splitting approach!"));

    pcout << std::endl << "OIF substepping for convective term:" << std::endl << std::endl;
    print_parameter(pcout, "CFL (OIF)", cfl_oif);
  }
}

template<typename Number>
double
TimeIntBDF<Number>::recalculate_time_step_size() const
{
  double new_time_step_size =
    pde_operator->calculate_time_step_cfl(this->get_time(),
                                          cfl,
                                          param.exponent_fe_degree_convection);

  bool use_limiter = true;
  if(use_limiter)
  {
    double last_time_step_size = this->get_time_step_size();
    double factor              = param.adaptive_time_stepping_limiting_factor;
    limit_time_step_change(new_time_step_size, last_time_step_size, factor);
  }

  return new_time_step_size;
}

template<typename Number>
void
TimeIntBDF<Number>::prepare_vectors_for_next_timestep()
{
  push_back(solution);

  solution[0].swap(solution_np);

  if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit &&
     (param.equation_type == EquationType::Convection ||
      param.equation_type == EquationType::ConvectionDiffusion))
  {
    push_back(vec_convective_term);
  }
}

template<typename Number>
bool
TimeIntBDF<Number>::print_solver_info() const
{
  return this->get_time_step_number() % param.output_solver_info_every_timesteps == 0;
}

template<typename Number>
void
TimeIntBDF<Number>::read_restart_vectors(boost::archive::binary_iarchive & ia)
{
  Vector<double> tmp;
  for(unsigned int i = 0; i < this->order; i++)
  {
    ia >> tmp;
    std::copy(tmp.begin(), tmp.end(), solution[i].begin());
  }
}

template<typename Number>
void
TimeIntBDF<Number>::write_restart_vectors(boost::archive::binary_oarchive & oa) const
{
  for(unsigned int i = 0; i < this->order; i++)
  {
    oa << solution[i];
  }
}

template<typename Number>
void
TimeIntBDF<Number>::solve_timestep()
{
  Timer timer;
  timer.restart();

  // calculate rhs (rhs-vector f and inhomogeneous boundary face integrals)
  pde_operator->rhs(rhs_vector, this->get_next_time());

  // add the convective term to the right-hand side of the equations
  // if the convective term is treated explicitly (additive decomposition)
  if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    // only if convective term is really involved
    if(param.equation_type == EquationType::Convection ||
       param.equation_type == EquationType::ConvectionDiffusion)
    {
      pde_operator->evaluate_convective_term(vec_convective_term[0], solution[0], this->get_time());

      for(unsigned int i = 0; i < vec_convective_term.size(); ++i)
        rhs_vector.add(-extra.get_beta(i), vec_convective_term[i]);
    }
  }

  // calculate sum (alpha_i/dt * u_tilde_i) in case of explicit treatment of convective term
  // and operator-integration-factor splitting
  if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    calculate_sum_alphai_ui_oif_substepping(cfl, cfl_oif);
  }
  // calculate sum (alpha_i/dt * u_i) for standard BDF discretization
  else
  {
    sum_alphai_ui.equ(bdf.get_alpha(0) / this->get_time_step_size(), solution[0]);
    for(unsigned int i = 1; i < solution.size(); ++i)
      sum_alphai_ui.add(bdf.get_alpha(i) / this->get_time_step_size(), solution[i]);
  }

  // apply mass matrix to sum_alphai_ui and add to rhs vector
  pde_operator->apply_mass_matrix_add(rhs_vector, sum_alphai_ui);

  // extrapolate old solution to obtain a good initial guess for the solver
  solution_np.equ(extra.get_beta(0), solution[0]);
  for(unsigned int i = 1; i < solution.size(); ++i)
    solution_np.add(extra.get_beta(i), solution[i]);

  // solve the linear system of equations
  bool const update_preconditioner =
    this->param.update_preconditioner &&
    (this->time_step_number % this->param.update_preconditioner_every_time_steps == 0);

  iterations += pde_operator->solve(solution_np,
                                    rhs_vector,
                                    update_preconditioner,
                                    bdf.get_gamma0() / this->get_time_step_size(),
                                    this->get_next_time());

  wall_time += timer.wall_time();

  // write output
  if(print_solver_info())
  {
    pcout << "Solve scalar convection-diffusion problem:" << std::endl
          << "  Iterations: " << std::setw(6) << std::right << iterations
          << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
  }
}

template<typename Number>
void
TimeIntBDF<Number>::initialize_solution_oif_substepping(unsigned int i)
{
  // initialize solution: u_tilde(s=0) = u(t_{n-i})
  solution_tilde_m = solution[i];
}

template<typename Number>
void
TimeIntBDF<Number>::update_sum_alphai_ui_oif_substepping(unsigned int i)
{
  // calculate sum (alpha_i/dt * u_tilde_i)
  if(i == 0)
    sum_alphai_ui.equ(bdf.get_alpha(i) / this->get_time_step_size(), solution_tilde_m);
  else // i>0
    sum_alphai_ui.add(bdf.get_alpha(i) / this->get_time_step_size(), solution_tilde_m);
}

template<typename Number>
void
TimeIntBDF<Number>::do_timestep_oif_substepping_and_update_vectors(double const start_time,
                                                                   double const time_step_size)
{
  // solve sub-step
  time_integrator_OIF->solve_timestep(solution_tilde_mp,
                                      solution_tilde_m,
                                      start_time,
                                      time_step_size);

  solution_tilde_mp.swap(solution_tilde_m);
}


template<typename Number>
void
TimeIntBDF<Number>::postprocessing() const
{
  pde_operator->do_postprocessing(solution[0], this->get_time(), this->get_time_step_number());
}

template<typename Number>
void
TimeIntBDF<Number>::get_iterations(std::vector<std::string> & name,
                                   std::vector<double> &      iteration) const
{
  unsigned int N_time_steps = this->get_time_step_number() - 1;

  name.resize(1);
  std::vector<std::string> names = {"Linear system"};
  name                           = names;

  iteration.resize(1);
  iteration[0] = (double)iterations / (double)N_time_steps;
}

template<typename Number>
void
TimeIntBDF<Number>::get_wall_times(std::vector<std::string> & name,
                                   std::vector<double> &      wall_time_vector) const
{
  name.resize(1);
  std::vector<std::string> names = {"Linear system"};
  name                           = names;

  wall_time_vector.resize(1);
  wall_time_vector[0] = this->wall_time;
}

// instantiations
#include <navierstokes/config.h>

#if OP_FLOAT
template class TimeIntBDF<float>;
#endif
#if OP_DOUBLE
template class TimeIntBDF<double>;
#endif

} // namespace ConvDiff
