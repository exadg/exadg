/*
 * time_int_bdf.cpp
 *
 *  Created on: Nov 14, 2018
 *      Author: fehn
 */

#include <exadg/convection_diffusion/postprocessor/postprocessor_base.h>
#include <exadg/convection_diffusion/spatial_discretization/dg_operator.h>
#include <exadg/convection_diffusion/time_integration/time_int_bdf.h>
#include <exadg/convection_diffusion/user_interface/input_parameters.h>
#include <exadg/grid/moving_mesh_base.h>
#include <exadg/time_integration/push_back_vectors.h>
#include <exadg/time_integration/time_step_calculation.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace ConvDiff
{
using namespace dealii;

template<int dim, typename Number>
TimeIntBDF<dim, Number>::TimeIntBDF(
  std::shared_ptr<Operator>                       operator_in,
  InputParameters const &                         param_in,
  unsigned int const                              refine_steps_time_in,
  MPI_Comm const &                                mpi_comm_in,
  bool const                                      print_wall_times_in,
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in,
  std::shared_ptr<MovingMeshBase<dim, Number>>    moving_mesh_in,
  std::shared_ptr<MatrixFree<dim, Number>>        matrix_free_in)
  : TimeIntBDFBase<Number>(param_in.start_time,
                           param_in.end_time,
                           param_in.max_number_of_time_steps,
                           param_in.order_time_integrator,
                           param_in.start_with_low_order,
                           param_in.adaptive_time_stepping,
                           param_in.restart_data,
                           mpi_comm_in,
                           print_wall_times_in),
    pde_operator(operator_in),
    param(param_in),
    refine_steps_time(refine_steps_time_in),
    cfl(param.cfl / std::pow(2.0, refine_steps_time_in)),
    solution(param_in.order_time_integrator),
    vec_convective_term(param_in.order_time_integrator),
    iterations({0, 0}),
    cfl_oif(param.cfl_oif / std::pow(2.0, refine_steps_time_in)),
    postprocessor(postprocessor_in),
    vec_grid_coordinates(param_in.order_time_integrator),
    moving_mesh(moving_mesh_in),
    matrix_free(matrix_free_in)
{
  if(param.ale_formulation)
  {
    AssertThrow(moving_mesh != nullptr,
                ExcMessage("Shared pointer moving_mesh is not correctly initialized."));
    AssertThrow(matrix_free != nullptr,
                ExcMessage("Shared pointer matrix_free_wrapper is not correctly initialized."));
  }
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::setup_derived()
{
  // In the case of an arbitrary Lagrangian-Eulerian formulation:
  if(param.ale_formulation && param.restarted_simulation == false)
  {
    // compute the grid coordinates at start time (and at previous times in case of
    // start_with_low_order == false)

    moving_mesh->move_mesh(this->get_time());
    moving_mesh->fill_grid_coordinates_vector(vec_grid_coordinates[0],
                                              pde_operator->get_dof_handler_velocity());

    if(this->start_with_low_order == false)
    {
      // compute grid coordinates at previous times (start with 1!)
      for(unsigned int i = 1; i < this->order; ++i)
      {
        moving_mesh->move_mesh(this->get_previous_time(i));
        moving_mesh->fill_grid_coordinates_vector(vec_grid_coordinates[i],
                                                  pde_operator->get_dof_handler_velocity());
      }
    }
  }

  // Initialize vec_convective_term: Note that this function has to be called
  // after the solution has been initialized because the solution is evaluated in this function.
  if(param.convective_problem() &&
     param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    // vec_convective_term does not have to be initialized in ALE case (the convective
    // term is recomputed in each time step for all previous times on the new mesh).
    // vec_convective_term does not have to be initialized in case of a restart, where
    // the vectors are read from memory.
    if(param.ale_formulation == false && param.restarted_simulation == false)
    {
      initialize_vec_convective_term();
    }
  }
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::initialize_oif()
{
  // Operator-integration-factor splitting
  if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    AssertThrow(param.convective_problem(), ExcMessage("Invalid parameters"));

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

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::allocate_vectors()
{
  for(unsigned int i = 0; i < solution.size(); ++i)
    pde_operator->initialize_dof_vector(solution[i]);

  pde_operator->initialize_dof_vector(solution_np);

  pde_operator->initialize_dof_vector(rhs_vector);

  if(param.convective_problem())
  {
    if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
    {
      for(unsigned int i = 0; i < vec_convective_term.size(); ++i)
        pde_operator->initialize_dof_vector(vec_convective_term[i]);

      if(param.ale_formulation == false)
        pde_operator->initialize_dof_vector(convective_term_np);
    }
  }

  if(param.ale_formulation == true)
  {
    pde_operator->initialize_dof_vector_velocity(grid_velocity);

    pde_operator->initialize_dof_vector_velocity(grid_coordinates_np);

    for(unsigned int i = 0; i < vec_grid_coordinates.size(); ++i)
      pde_operator->initialize_dof_vector_velocity(vec_grid_coordinates[i]);
  }
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::initialize_current_solution()
{
  if(this->param.ale_formulation)
    this->move_mesh(this->get_time());

  pde_operator->prescribe_initial_conditions(solution[0], this->get_time());
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::initialize_former_solutions()
{
  // Start with i=1 since we only want to initialize the solution at former instants of time.
  for(unsigned int i = 1; i < solution.size(); ++i)
  {
    if(this->param.ale_formulation)
      this->move_mesh(this->get_previous_time(i));

    pde_operator->prescribe_initial_conditions(solution[i], this->get_previous_time(i));
  }
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::initialize_vec_convective_term()
{
  if(this->param.get_type_velocity_field() != TypeVelocityField::DoFVector)
  {
    pde_operator->evaluate_convective_term(vec_convective_term[0], solution[0], this->get_time());

    if(this->param.start_with_low_order == false)
    {
      for(unsigned int i = 1; i < vec_convective_term.size(); ++i)
      {
        pde_operator->evaluate_convective_term(vec_convective_term[i],
                                               solution[i],
                                               this->get_previous_time(i));
      }
    }
  }
}

template<int dim, typename Number>
double
TimeIntBDF<dim, Number>::calculate_time_step_size()
{
  double time_step = 1.0;

  unsigned int const degree = pde_operator->get_polynomial_degree();

  if(param.calculation_of_time_step_size == TimeStepCalculation::UserSpecified)
  {
    time_step = calculate_const_time_step(param.time_step_size, refine_steps_time);

    this->pcout << std::endl
                << "Calculation of time step size (user-specified):" << std::endl
                << std::endl;
    print_parameter(this->pcout, "time step size", time_step);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::CFL)
  {
    AssertThrow(param.convective_problem(),
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

    this->pcout << std::endl
                << "Calculation of time step size according to CFL condition:" << std::endl
                << std::endl;
    print_parameter(this->pcout, "h_min", h_min);
    print_parameter(this->pcout, "U_max", max_velocity);
    print_parameter(this->pcout, "CFL", cfl);
    print_parameter(this->pcout, "Exponent fe_degree", param.exponent_fe_degree_convection);
    print_parameter(this->pcout, "Time step size (CFL global)", time_step_global);

    if(this->adaptive_time_stepping == true)
    {
      double time_step_adap = std::numeric_limits<double>::max();

      // Note that in the ALE case there is no possibility to know the grid velocity at this point
      // and to use it for the calculation of the time step size.

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
      param.c_eff, h_min, degree, this->order, refine_steps_time);

    time_step = adjust_time_step_to_hit_end_time(param.start_time, param.end_time, time_step);

    this->pcout << std::endl
                << "Calculation of time step size (max efficiency):" << std::endl
                << std::endl;
    print_parameter(this->pcout, "C_eff", param.c_eff / std::pow(2, refine_steps_time));
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

template<int dim, typename Number>
double
TimeIntBDF<dim, Number>::recalculate_time_step_size() const
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
  else // numerical velocity field
  {
    AssertThrow(velocities[0] != nullptr, ExcMessage("Pointer velocities[0] is not initialized."));

    VectorType u_relative = *velocities[0];
    if(param.ale_formulation == true)
      u_relative -= grid_velocity;

    new_time_step_size =
      pde_operator->calculate_time_step_cfl_numerical_velocity(u_relative,
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

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::prepare_vectors_for_next_timestep()
{
  push_back(solution);

  solution[0].swap(solution_np);

  if(param.convective_problem() &&
     param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    if(param.ale_formulation == false)
    {
      push_back(vec_convective_term);
      vec_convective_term[0].swap(convective_term_np);
    }
  }

  if(param.ale_formulation)
  {
    push_back(vec_grid_coordinates);
    vec_grid_coordinates[0].swap(grid_coordinates_np);
  }
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::ale_update()
{
  // and compute grid coordinates at the end of the current time step t_{n+1}
  moving_mesh->fill_grid_coordinates_vector(grid_coordinates_np,
                                            pde_operator->get_dof_handler_velocity());

  // and update grid velocity using BDF time derivative
  compute_bdf_time_derivative(grid_velocity,
                              grid_coordinates_np,
                              vec_grid_coordinates,
                              this->bdf,
                              this->get_time_step_size());
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::move_mesh(double const time) const
{
  moving_mesh->move_mesh(time);
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::move_mesh_and_update_dependent_data_structures(double const time) const
{
  moving_mesh->move_mesh(time);
  matrix_free->update_mapping(moving_mesh->get_mapping());
  pde_operator->update_after_mesh_movement();
}

template<int dim, typename Number>
bool
TimeIntBDF<dim, Number>::print_solver_info() const
{
  return param.solver_info_data.write(this->global_timer.wall_time(),
                                      this->time,
                                      this->time_step_number);
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::read_restart_vectors(boost::archive::binary_iarchive & ia)
{
  for(unsigned int i = 0; i < this->order; i++)
  {
    ia >> solution[i];
  }

  if(param.convective_problem() &&
     param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    if(this->param.ale_formulation == false)
    {
      for(unsigned int i = 0; i < this->order; i++)
      {
        ia >> vec_convective_term[i];
      }
    }
  }

  if(this->param.ale_formulation)
  {
    for(unsigned int i = 0; i < vec_grid_coordinates.size(); i++)
    {
      ia >> vec_grid_coordinates[i];
    }
  }
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::write_restart_vectors(boost::archive::binary_oarchive & oa) const
{
  for(unsigned int i = 0; i < this->order; i++)
  {
    oa << solution[i];
  }

  if(param.convective_problem() &&
     param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    if(this->param.ale_formulation == false)
    {
      for(unsigned int i = 0; i < this->order; i++)
      {
        oa << vec_convective_term[i];
      }
    }
  }

  if(this->param.ale_formulation)
  {
    for(unsigned int i = 0; i < vec_grid_coordinates.size(); i++)
    {
      oa << vec_grid_coordinates[i];
    }
  }
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::solve_timestep()
{
  Timer timer;
  timer.restart();

  // transport velocity
  VectorType velocity_np;

  if(param.convective_problem())
  {
    if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
    {
      pde_operator->initialize_dof_vector_velocity(velocity_np);

      if(param.analytical_velocity_field)
      {
        pde_operator->project_velocity(velocity_np, this->get_next_time());
      }
      else
      {
        AssertThrow(std::abs(times[0] - this->get_next_time()) < 1.e-12 * param.end_time,
                    ExcMessage("Invalid assumption."));
        AssertThrow(velocities[0] != nullptr,
                    ExcMessage("Pointer velocities[0] is not correctly initialized."));

        velocity_np = *velocities[0];
      }

      if(param.ale_formulation)
      {
        velocity_np -= grid_velocity;
      }
    }
    else
    {
      AssertThrow(param.ale_formulation == false, ExcMessage("not implemented."));
    }
  }

  // calculate rhs (rhs-vector f and inhomogeneous boundary face integrals)
  pde_operator->rhs(rhs_vector, this->get_next_time(), &velocity_np);

  // if the convective term is involved in the equations:
  // add the convective term to the right-hand side of the equations
  // if this term is treated explicitly (additive decomposition)
  if(param.convective_problem() &&
     param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    // recompute convective term on new mesh for all previous time instants in case of
    // ALE formulation
    if(param.ale_formulation == true)
    {
      for(unsigned int i = 0; i < vec_convective_term.size(); ++i)
      {
        pde_operator->evaluate_convective_term(vec_convective_term[i],
                                               solution[i],
                                               this->get_previous_time(i),
                                               &velocity_np);
      }
    }

    for(unsigned int i = 0; i < vec_convective_term.size(); ++i)
      rhs_vector.add(-this->extra.get_beta(i), vec_convective_term[i]);
  }

  VectorType sum_alphai_ui(solution[0]);

  // calculate sum (alpha_i/dt * u_tilde_i) if operator-integration-factor splitting
  // is used to integrate the convective term
  if(param.convective_problem() &&
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

  // apply mass operator to sum_alphai_ui and add to rhs_vector
  pde_operator->apply_mass_operator_add(rhs_vector, sum_alphai_ui);

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
                        &velocity_np);

  iterations.first += 1;
  iterations.second += N_iter;

  // evaluate convective term at end time t_{n+1} at which we know the boundary condition
  // g_u(t_{n+1})
  if(param.convective_problem() &&
     param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    if(param.ale_formulation == false)
    {
      if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
      {
        pde_operator->evaluate_convective_term(convective_term_np,
                                               solution_np,
                                               this->get_next_time(),
                                               &velocity_np);
      }
      else
      {
        pde_operator->evaluate_convective_term(convective_term_np,
                                               solution_np,
                                               this->get_next_time());
      }
    }
  }

  if(print_solver_info())
  {
    this->pcout << std::endl << "Solve scalar convection-diffusion equation:";
    print_solver_info_linear(this->pcout, N_iter, timer.wall_time(), this->print_wall_times);
  }

  this->timer_tree->insert({"Timeloop", "Solve"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::calculate_sum_alphai_ui_oif_substepping(VectorType & sum_alphai_ui,
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

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::initialize_solution_oif_substepping(VectorType & solution_tilde_m,
                                                             unsigned int i)
{
  // initialize solution: u_tilde(s=0) = u(t_{n-i})
  solution_tilde_m = solution[i];
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::update_sum_alphai_ui_oif_substepping(VectorType &       sum_alphai_ui,
                                                              VectorType const & u_tilde_i,
                                                              unsigned int       i)
{
  // calculate sum (alpha_i/dt * u_tilde_i)
  if(i == 0)
    sum_alphai_ui.equ(this->bdf.get_alpha(i) / this->get_time_step_size(), u_tilde_i);
  else // i>0
    sum_alphai_ui.add(this->bdf.get_alpha(i) / this->get_time_step_size(), u_tilde_i);
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::do_timestep_oif_substepping(VectorType & solution_tilde_mp,
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


template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::postprocessing() const
{
  Timer timer;
  timer.restart();

  // the mesh has to be at the correct position to allow a computation of
  // errors at start_time
  if(this->param.ale_formulation && this->get_time_step_number() == 1)
  {
    move_mesh_and_update_dependent_data_structures(this->get_time());
  }

  postprocessor->do_postprocessing(solution[0], this->get_time(), this->get_time_step_number());

  this->timer_tree->insert({"Timeloop", "Postprocessing"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::print_iterations() const
{
  std::vector<std::string> names = {"Linear system"};

  std::vector<double> iterations_avg;
  iterations_avg.resize(1);
  iterations_avg[0] = (double)iterations.second / std::max(1., (double)iterations.first);

  print_list_of_iterations(this->pcout, names, iterations_avg);
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::set_velocities_and_times(
  std::vector<VectorType const *> const & velocities_in,
  std::vector<double> const &             times_in)
{
  velocities = velocities_in;
  times      = times_in;
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::extrapolate_solution(VectorType & vector)
{
  // make sure that the time integrator constants are up-to-date
  this->update_time_integrator_constants();

  vector.equ(this->extra.get_beta(0), this->solution[0]);
  for(unsigned int i = 1; i < solution.size(); ++i)
    vector.add(this->extra.get_beta(i), this->solution[i]);
}

// instantiations

template class TimeIntBDF<2, float>;
template class TimeIntBDF<2, double>;

template class TimeIntBDF<3, float>;
template class TimeIntBDF<3, double>;

} // namespace ConvDiff
} // namespace ExaDG
