/*
 * time_int_bdf.cpp
 *
 *  Created on: Nov 14, 2018
 *      Author: fehn
 */

#include "time_int_bdf.h"

#include "../spatial_discretization/interface.h"
#include "time_integration/push_back_vectors.h"
#include "time_integration/time_step_calculation.h"

#include "../user_interface/input_parameters.h"

namespace ConvDiff
{
template<typename Number>
TimeIntBDF<Number>::TimeIntBDF(std::shared_ptr<Operator> operator_in,
                               InputParameters const &   param_in,
                               MPI_Comm const &          mpi_comm_in)
  : TimeIntBDFBase<Number>(param_in.start_time,
                           param_in.end_time,
                           param_in.max_number_of_time_steps,
                           param_in.order_time_integrator,
                           param_in.start_with_low_order,
                           param_in.adaptive_time_stepping,
                           param_in.restart_data,
                           mpi_comm_in),
    pde_operator(operator_in),
    param(param_in),
    cfl(param.cfl / std::pow(2.0, param.dt_refinements)),
    solution(param_in.order_time_integrator),
    vec_convective_term(param_in.order_time_integrator),
    iterations(0.0),
    wall_time(0.0),
    cfl_oif(param.cfl_oif / std::pow(2.0, param.dt_refinements))
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
     this->start_with_low_order == false)
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
    AssertThrow(param.equation_type == EquationType::Convection ||
                  param.equation_type == EquationType::ConvectionDiffusion,
                ExcMessage("Invalid parameters"));

    bool numerical_velocity_field =
      (param.get_type_velocity_field() == TypeVelocityField::DoFVector);

    convective_operator_OIF.reset(
      new Interface::OperatorOIF<Number>(pde_operator, numerical_velocity_field));

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

  pde_operator->initialize_dof_vector(rhs_vector);

  if(param.equation_type == EquationType::Convection ||
     param.equation_type == EquationType::ConvectionDiffusion)
  {
    if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
    {
      for(unsigned int i = 0; i < vec_convective_term.size(); ++i)
        pde_operator->initialize_dof_vector(vec_convective_term[i]);
    }
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
double
TimeIntBDF<Number>::calculate_time_step_size()
{
  double time_step = 1.0;

  unsigned int const degree = pde_operator->get_polynomial_degree();

  if(param.calculation_of_time_step_size == TimeStepCalculation::UserSpecified)
  {
    time_step = calculate_const_time_step(param.time_step_size, param.dt_refinements);

    this->pcout << "Calculation of time step size (user-specified):" << std::endl << std::endl;
    print_parameter(this->pcout, "time step size", time_step);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::CFL)
  {
    AssertThrow(param.equation_type == EquationType::Convection ||
                  param.equation_type == EquationType::ConvectionDiffusion,
                ExcMessage("Specified type of time step calculation does not make sense!"));

    double const h_min = pde_operator->calculate_minimum_element_length();

    double max_velocity = 0.0;
    if(param.analytical_velocity_field)
    {
      max_velocity = pde_operator->calculate_maximum_velocity(this->get_time());
    }

    // max_velocity computed above might be zero depending on the initial velocity field -> dt would
    // tend to infinity
    max_velocity = std::max(max_velocity, param.max_velocity);

    double time_step_global = calculate_time_step_cfl_global(
      cfl, max_velocity, h_min, degree, param.exponent_fe_degree_convection);

    this->pcout << "Calculation of time step size according to CFL condition:" << std::endl
                << std::endl;
    print_parameter(this->pcout, "h_min", h_min);
    print_parameter(this->pcout, "U_max", max_velocity);
    print_parameter(this->pcout, "CFL", cfl);
    print_parameter(this->pcout, "Exponent fe_degree", param.exponent_fe_degree_convection);
    print_parameter(this->pcout, "Time step size (CFL global)", time_step_global);

    if(this->adaptive_time_stepping == true)
    {
      double time_step_adap = std::numeric_limits<double>::max();

      if(param.analytical_velocity_field)
      {
        time_step_adap = pde_operator->calculate_time_step_cfl_analytical_velocity(
          this->get_time(), cfl, param.exponent_fe_degree_convection);
      }
      else
      {
        // do nothing (the velocity field is not known at this point)
      }

      // use adaptive time step size only if it is smaller, otherwise use global time step size
      time_step = std::min(time_step_adap, time_step_global);

      // make sure that the maximum allowable time step size is not exceeded
      time_step = std::min(time_step, param.time_step_size_max);

      print_parameter(this->pcout, "Time step size (CFL adaptive)", time_step);
    }
    else // constant time step size
    {
      time_step =
        adjust_time_step_to_hit_end_time(param.start_time, param.end_time, time_step_global);

      this->pcout << std::endl
                  << "Adjust time step size to hit end time:" << std::endl
                  << std::endl;
      print_parameter(this->pcout, "Time step size", time_step);
    }
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::MaxEfficiency)
  {
    // calculate minimum vertex distance
    double const h_min = pde_operator->calculate_minimum_element_length();

    time_step = calculate_time_step_max_efficiency(
      param.c_eff, h_min, degree, this->order, param.dt_refinements);

    time_step = adjust_time_step_to_hit_end_time(param.start_time, param.end_time, time_step);

    this->pcout << std::endl
                << "Calculation of time step size (max efficiency):" << std::endl
                << std::endl;
    print_parameter(this->pcout, "C_eff", param.c_eff / std::pow(2, param.dt_refinements));
    print_parameter(this->pcout, "Time step size", time_step);
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

    this->pcout << std::endl << "OIF substepping for convective term:" << std::endl << std::endl;
    print_parameter(this->pcout, "CFL (OIF)", cfl_oif);
  }

  return time_step;
}

template<typename Number>
double
TimeIntBDF<Number>::recalculate_time_step_size() const
{
  AssertThrow(param.calculation_of_time_step_size == TimeStepCalculation::CFL,
              ExcMessage(
                "Adaptive time step is not implemented for this type of time step calculation."));

  double new_time_step_size = std::numeric_limits<double>::max();
  if(param.analytical_velocity_field)
  {
    new_time_step_size = pde_operator->calculate_time_step_cfl_analytical_velocity(
      this->get_time(), cfl, param.exponent_fe_degree_convection);
  }
  else
  {
    AssertThrow(velocities[0] != nullptr, ExcMessage("Pointer velocities[0] is not initialized."));

    new_time_step_size =
      pde_operator->calculate_time_step_cfl_numerical_velocity(*velocities[0],
                                                               cfl,
                                                               param.exponent_fe_degree_convection);
  }

  // make sure that time step size does not exceed maximum allowable time step size
  new_time_step_size = std::min(new_time_step_size, param.time_step_size_max);

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
  return param.solver_info_data.write(this->global_timer.wall_time(),
                                      this->time,
                                      this->time_step_number);
}

template<typename Number>
void
TimeIntBDF<Number>::read_restart_vectors(boost::archive::binary_iarchive & ia)
{
  for(unsigned int i = 0; i < this->order; i++)
  {
    ia >> solution[i];
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
  this->output_solver_info_header();

  Timer timer;
  timer.restart();

  // prepare pointer for velocity field, but only if necessary
  VectorType const * velocity_ptr = nullptr;
  VectorType         velocity_vector;

  if(param.linear_system_including_convective_term_has_to_be_solved())
  {
    if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
    {
      if(param.analytical_velocity_field)
      {
        pde_operator->initialize_dof_vector_velocity(velocity_vector);
        pde_operator->project_velocity(velocity_vector, this->get_next_time());

        velocity_ptr = &velocity_vector;
      }
      else
      {
        AssertThrow(std::abs(times[0] - this->get_next_time()) < 1.e-12 * param.end_time,
                    ExcMessage("Invalid assumption."));

        velocity_ptr = velocities[0];
      }
    }
  }

  // calculate rhs (rhs-vector f and inhomogeneous boundary face integrals)
  pde_operator->rhs(rhs_vector, this->get_next_time(), velocity_ptr);

  // if the convective term is involved in the equations:
  // add the convective term to the right-hand side of the equations
  // if this term is treated explicitly (additive decomposition)
  if((param.equation_type == EquationType::Convection ||
      param.equation_type == EquationType::ConvectionDiffusion) &&
     param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
    {
      AssertThrow(std::abs(times[1] - this->get_time()) < 1.e-12 * param.end_time,
                  ExcMessage("Invalid assumption."));

      pde_operator->evaluate_convective_term(vec_convective_term[0],
                                             solution[0],
                                             this->get_time(),
                                             velocities[1]);
    }
    else
    {
      pde_operator->evaluate_convective_term(vec_convective_term[0], solution[0], this->get_time());
    }

    for(unsigned int i = 0; i < vec_convective_term.size(); ++i)
      rhs_vector.add(-this->extra.get_beta(i), vec_convective_term[i]);
  }

  VectorType sum_alphai_ui(solution[0]);

  // calculate sum (alpha_i/dt * u_tilde_i) if operator-integration-factor splitting
  // is used to integrate the convective term
  if((param.equation_type == EquationType::Convection ||
      param.equation_type == EquationType::ConvectionDiffusion) &&
     param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    calculate_sum_alphai_ui_oif_substepping(sum_alphai_ui, cfl, cfl_oif);
  }
  // calculate sum (alpha_i/dt * u_i) for standard BDF discretization
  else
  {
    sum_alphai_ui.equ(this->bdf.get_alpha(0) / this->get_time_step_size(), solution[0]);
    for(unsigned int i = 1; i < solution.size(); ++i)
      sum_alphai_ui.add(this->bdf.get_alpha(i) / this->get_time_step_size(), solution[i]);
  }

  // apply mass matrix to sum_alphai_ui and add to rhs_vector
  pde_operator->apply_mass_matrix_add(rhs_vector, sum_alphai_ui);

  // extrapolate old solution to obtain a good initial guess for the solver
  solution_np.equ(this->extra.get_beta(0), solution[0]);
  for(unsigned int i = 1; i < solution.size(); ++i)
    solution_np.add(this->extra.get_beta(i), solution[i]);

  // solve the linear system of equations
  bool const update_preconditioner =
    this->param.update_preconditioner &&
    (this->time_step_number % this->param.update_preconditioner_every_time_steps == 0);

  unsigned int const N_iter =
    pde_operator->solve(solution_np,
                        rhs_vector,
                        update_preconditioner,
                        this->bdf.get_gamma0() / this->get_time_step_size(),
                        this->get_next_time(),
                        velocity_ptr);

  iterations += N_iter;
  wall_time += timer.wall_time();

  // TODO: implement filtering as a separate module
  // filtering is currently based on multigrid implementation and can therefore
  // only be used in combination with semi-implicit BDF time integration and
  // multigrid preconditioner
  if(param.filter_solution)
    pde_operator->filter_solution(solution_np);

  // write output
  if(print_solver_info())
  {
    this->pcout << "Solve scalar convection-diffusion problem:" << std::endl
                << "  Iterations: " << std::setw(6) << std::right << N_iter
                << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
  }
}

template<typename Number>
void
TimeIntBDF<Number>::calculate_sum_alphai_ui_oif_substepping(VectorType & sum_alphai_ui,
                                                            double const cfl,
                                                            double const cfl_oif)
{
  if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
  {
    this->convective_operator_OIF->set_velocities_and_times(velocities, times);
  }

  // call function implemented in base class for the actual OIF sub-stepping
  TimeIntBDFBase<Number>::calculate_sum_alphai_ui_oif_substepping(sum_alphai_ui, cfl, cfl_oif);
}

template<typename Number>
void
TimeIntBDF<Number>::initialize_solution_oif_substepping(VectorType & solution_tilde_m,
                                                        unsigned int i)
{
  // initialize solution: u_tilde(s=0) = u(t_{n-i})
  solution_tilde_m = solution[i];
}

template<typename Number>
void
TimeIntBDF<Number>::update_sum_alphai_ui_oif_substepping(VectorType &       sum_alphai_ui,
                                                         VectorType const & u_tilde_i,
                                                         unsigned int       i)
{
  // calculate sum (alpha_i/dt * u_tilde_i)
  if(i == 0)
    sum_alphai_ui.equ(this->bdf.get_alpha(i) / this->get_time_step_size(), u_tilde_i);
  else // i>0
    sum_alphai_ui.add(this->bdf.get_alpha(i) / this->get_time_step_size(), u_tilde_i);
}

template<typename Number>
void
TimeIntBDF<Number>::do_timestep_oif_substepping(VectorType & solution_tilde_mp,
                                                VectorType & solution_tilde_m,
                                                double const start_time,
                                                double const time_step_size)
{
  // solve sub-step
  time_integrator_OIF->solve_timestep(solution_tilde_mp,
                                      solution_tilde_m,
                                      start_time,
                                      time_step_size);
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

template<typename Number>
void
TimeIntBDF<Number>::set_velocities_and_times(std::vector<VectorType const *> const & velocities_in,
                                             std::vector<double> const &             times_in)
{
  velocities = velocities_in;
  times      = times_in;
}

template<typename Number>
void
TimeIntBDF<Number>::extrapolate_solution(VectorType & vector)
{
  // make sure that the time integrator constants are up-to-date
  this->update_time_integrator_constants();

  vector.equ(this->extra.get_beta(0), this->solution[0]);
  for(unsigned int i = 1; i < solution.size(); ++i)
    vector.add(this->extra.get_beta(i), this->solution[i]);
}

// instantiations

template class TimeIntBDF<float>;
template class TimeIntBDF<double>;

} // namespace ConvDiff
