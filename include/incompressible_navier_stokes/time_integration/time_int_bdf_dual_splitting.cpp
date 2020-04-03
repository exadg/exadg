/*
 * time_int_bdf_dual_splitting.cpp
 *
 *  Created on: Nov 15, 2018
 *      Author: fehn
 */

#include "time_int_bdf_dual_splitting.h"

#include "../../time_integration/push_back_vectors.h"
#include "../spatial_discretization/interface.h"
#include "../user_interface/input_parameters.h"

namespace IncNS
{
template<int dim, typename Number>
TimeIntBDFDualSplitting<dim, Number>::TimeIntBDFDualSplitting(
  std::shared_ptr<Operator>                       operator_in,
  InputParameters const &                         param_in,
  unsigned int const                              refine_steps_time_in,
  MPI_Comm const &                                mpi_comm_in,
  std::shared_ptr<PostProcessorBase<dim, Number>> postprocessor_in,
  std::shared_ptr<MovingMeshBase<dim, Number>>    moving_mesh_in,
  std::shared_ptr<MatrixFreeWrapper<dim, Number>> matrix_free_wrapper_in)
  : Base(operator_in,
         param_in,
         refine_steps_time_in,
         mpi_comm_in,
         postprocessor_in,
         moving_mesh_in,
         matrix_free_wrapper_in),
    pde_operator(operator_in),
    velocity(this->order),
    pressure(this->order),
#ifdef EXTRAPOLATE_ACCELERATION
    acceleration(this->param.order_extrapolation_pressure_nbc),
#endif
    velocity_dbc(this->order),
    computing_times(6),
    computing_time_convective(0.0),
    iterations(5),
    extra_pressure_nbc(this->param.order_extrapolation_pressure_nbc,
                       this->param.start_with_low_order)
{
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::update_time_integrator_constants()
{
  // call function of base class to update the standard time integrator constants
  Base::update_time_integrator_constants();

  // update time integrator constants for extrapolation scheme of pressure Neumann bc
  if(this->adaptive_time_stepping == false)
  {
    extra_pressure_nbc.update(this->get_time_step_number());
  }
  else // adaptive time stepping
  {
    extra_pressure_nbc.update(this->get_time_step_number(), this->get_time_step_vector());
  }

  // use this function to check the correctness of the time integrator constants
  //    std::cout << "Coefficients extrapolation scheme pressure NBC:" << std::endl;
  //    extra_pressure_nbc.print(this->pcout);
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::setup_derived()
{
  Base::setup_derived();

  // accelaration, velocity_dbc do not have to be initialized in case of a restart, where
  // the vectors are read from memory.
  if(this->param.store_previous_boundary_values && this->param.restarted_simulation == false)
  {
    initialize_acceleration_and_velocity_on_boundary();
  }
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::read_restart_vectors(boost::archive::binary_iarchive & ia)
{
  Base::read_restart_vectors(ia);

  if(this->param.store_previous_boundary_values)
  {
#ifdef EXTRAPOLATE_ACCELERATION
    for(unsigned int i = 0; i < acceleration.size(); i++)
    {
      ia >> acceleration[i];
    }
#endif

    for(unsigned int i = 0; i < velocity_dbc.size(); i++)
    {
      ia >> velocity_dbc[i];
    }
  }
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::write_restart_vectors(
  boost::archive::binary_oarchive & oa) const
{
  Base::write_restart_vectors(oa);

  if(this->param.store_previous_boundary_values)
  {
#ifdef EXTRAPOLATE_ACCELERATION
    for(unsigned int i = 0; i < acceleration.size(); i++)
    {
      oa << acceleration[i];
    }
#endif

    for(unsigned int i = 0; i < velocity_dbc.size(); i++)
    {
      oa << velocity_dbc[i];
    }
  }
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::allocate_vectors()
{
  Base::allocate_vectors();

  // velocity
  for(unsigned int i = 0; i < velocity.size(); ++i)
  {
    pde_operator->initialize_vector_velocity(velocity[i]);
  }
  pde_operator->initialize_vector_velocity(velocity_np);

  // pressure
  for(unsigned int i = 0; i < pressure.size(); ++i)
    pde_operator->initialize_vector_pressure(pressure[i]);
  pde_operator->initialize_vector_pressure(pressure_np);

  // acceleration
  if(this->param.store_previous_boundary_values)
  {
#ifdef EXTRAPOLATE_ACCELERATION
    for(unsigned int i = 0; i < acceleration.size(); ++i)
      pde_operator->initialize_vector_velocity(acceleration[i]);
#endif

    for(unsigned int i = 0; i < velocity_dbc.size(); ++i)
      pde_operator->initialize_vector_velocity(velocity_dbc[i]);

    pde_operator->initialize_vector_velocity(velocity_dbc_np);
  }
}


template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::initialize_current_solution()
{
  if(this->param.ale_formulation)
    this->move_mesh(this->get_time());

  pde_operator->prescribe_initial_conditions(velocity[0], pressure[0], this->get_time());
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::initialize_former_solutions()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i = 1; i < velocity.size(); ++i)
  {
    if(this->param.ale_formulation)
      this->move_mesh(this->get_previous_time(i));

    pde_operator->prescribe_initial_conditions(velocity[i],
                                               pressure[i],
                                               this->get_previous_time(i));
  }
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::initialize_acceleration_and_velocity_on_boundary()
{
#ifdef EXTRAPOLATE_ACCELERATION
  // accelerations will only be accessed if the order of extrapolation in the pressure
  // Neumann boundary condition is larger than 0.
  if(this->param.order_extrapolation_pressure_nbc > 0)
  {
    // temporary vectors used to store the velocity at different instants of time and
    // used to calculate the acceleration via BDF time derivative
    VectorType vel_np;
    pde_operator->initialize_vector_velocity(vel_np);

    // Note that for BDF1 it can not be guaranteed that the results are the same for
    // start_with_low_order = true and false if the time derivative of the velocity
    // in the pressure Neumann boundary condition is computed numerically. This is due to
    // the fact that computing the time derivative requires knowledge about the velocity
    // at previous times, which is only available for start_with_low_order = false.
    // Hence, the acceleration at start_time will be zero for start_with_low_order = true
    // and this acceleration will be used in the first time step.
    if(this->start_with_low_order == false)
    {
      // compute acceleration at start_time
      if(this->param.ale_formulation)
        this->move_mesh_and_update_dependent_data_structures(this->get_time());
      pde_operator->interpolate_velocity_dirichlet_bc(vel_np, this->get_time());

      for(unsigned int i = 0; i < velocity_dbc.size(); ++i)
      {
        double const time = this->get_time() - double(i + 1) * this->get_time_step_size();
        if(this->param.ale_formulation)
          this->move_mesh_and_update_dependent_data_structures(time);
        pde_operator->interpolate_velocity_dirichlet_bc(velocity_dbc[i], time);
      }

      compute_bdf_time_derivative(
        acceleration[0], vel_np, velocity_dbc, this->bdf, this->get_time_step_size());
    }

    // compute accelerations at previous times t_j = start_time - j * dt, j = 1, ...
    if(this->start_with_low_order == false)
    {
      for(unsigned int j = 1; j < acceleration.size(); ++j)
      {
        double const time = this->get_time() - double(j) * this->get_time_step_size();
        if(this->param.ale_formulation)
          this->move_mesh_and_update_dependent_data_structures(time);
        pde_operator->interpolate_velocity_dirichlet_bc(vel_np, time);

        for(unsigned int i = 0; i < velocity_dbc.size(); ++i)
        {
          double const time = this->get_time() - double(j + i + 1) * this->get_time_step_size();
          if(this->param.ale_formulation)
            this->move_mesh_and_update_dependent_data_structures(time);
          pde_operator->interpolate_velocity_dirichlet_bc(velocity_dbc[i], time);
        }

        compute_bdf_time_derivative(
          acceleration[j], vel_np, velocity_dbc, this->bdf, this->get_time_step_size());
      }
    }
  }
#endif

  // fill vector velocity_dbc: The first entry [0] is already needed if start_with_low_order == true
  if(this->param.ale_formulation)
    this->move_mesh_and_update_dependent_data_structures(this->get_time());
  pde_operator->interpolate_velocity_dirichlet_bc(velocity_dbc[0], this->get_time());
  // ... and previous times if start_with_low_order == false
  if(this->start_with_low_order == false)
  {
    for(unsigned int i = 1; i < velocity_dbc.size(); ++i)
    {
      double const time = this->get_time() - double(i) * this->get_time_step_size();
      if(this->param.ale_formulation)
        this->move_mesh_and_update_dependent_data_structures(time);
      pde_operator->interpolate_velocity_dirichlet_bc(velocity_dbc[i], time);
    }
  }
}

template<int dim, typename Number>
LinearAlgebra::distributed::Vector<Number> const &
TimeIntBDFDualSplitting<dim, Number>::get_velocity() const
{
  return velocity[0];
}

template<int dim, typename Number>
LinearAlgebra::distributed::Vector<Number> const &
TimeIntBDFDualSplitting<dim, Number>::get_velocity_np() const
{
  return velocity_np;
}

template<int dim, typename Number>
LinearAlgebra::distributed::Vector<Number> const &
TimeIntBDFDualSplitting<dim, Number>::get_velocity(unsigned int i) const
{
  return velocity[i];
}

template<int dim, typename Number>
LinearAlgebra::distributed::Vector<Number> const &
TimeIntBDFDualSplitting<dim, Number>::get_pressure(unsigned int i) const
{
  return pressure[i];
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::set_velocity(VectorType const & velocity_in,
                                                   unsigned int const i)
{
  velocity[i] = velocity_in;
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::set_pressure(VectorType const & pressure_in,
                                                   unsigned int const i)
{
  pressure[i] = pressure_in;
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::postprocessing_stability_analysis()
{
  AssertThrow(this->order == 1,
              ExcMessage("Order of BDF scheme has to be 1 for this stability analysis."));

  AssertThrow(this->param.convective_problem() == false,
              ExcMessage(
                "Stability analysis can not be performed for nonlinear convective problems."));

  AssertThrow(velocity[0].l2_norm() < 1.e-15 && pressure[0].l2_norm() < 1.e-15,
              ExcMessage("Solution vector has to be zero for this stability analysis."));

  AssertThrow(Utilities::MPI::n_mpi_processes(this->mpi_comm) == 1,
              ExcMessage("Number of MPI processes has to be 1."));

  std::cout << std::endl << "Analysis of eigenvalue spectrum:" << std::endl;

  const unsigned int size = velocity[0].local_size();

  LAPACKFullMatrix<Number> propagation_matrix(size, size);

  // loop over all columns of propagation matrix
  for(unsigned int j = 0; j < size; ++j)
  {
    // set j-th element to 1
    velocity[0].local_element(j) = 1.0;

    // solve time step
    this->solve_timestep();

    // dst-vector velocity_np is j-th column of propagation matrix
    for(unsigned int i = 0; i < size; ++i)
    {
      propagation_matrix(i, j) = velocity_np.local_element(i);
    }

    // reset j-th element to 0
    velocity[0].local_element(j) = 0.0;
  }

  // compute eigenvalues
  propagation_matrix.compute_eigenvalues();

  double norm_max = 0.0;

  std::cout << "List of all eigenvalues:" << std::endl;

  for(unsigned int i = 0; i < size; ++i)
  {
    double norm = std::abs(propagation_matrix.eigenvalue(i));
    if(norm > norm_max)
      norm_max = norm;

    // print eigenvalues
    std::cout << std::scientific << std::setprecision(5) << propagation_matrix.eigenvalue(i)
              << std::endl;
  }

  std::cout << std::endl << std::endl << "Maximum eigenvalue = " << norm_max << std::endl;
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::solve_timestep()
{
  this->output_solver_info_header();

#ifndef EXTRAPOLATE_ACCELERATION
  // pre-computations
  if(this->param.store_previous_boundary_values)
    update_velocity_dbc();
#endif

  // perform the four sub-steps of the dual-splitting method
  convective_step();

  pressure_step();

  projection_step();

  viscous_step();

  if(this->param.apply_penalty_terms_in_postprocessing_step)
    penalty_step();

  // evaluate convective term once the final solution at time
  // t_{n+1} is known
  evaluate_convective_term();
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::update_velocity_dbc()
{
  pde_operator->interpolate_velocity_dirichlet_bc(velocity_dbc_np, this->get_next_time());
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::convective_step()
{
  Timer timer;
  timer.restart();

  velocity_np = 0.0;

  // compute convective term and extrapolate convective term (if not Stokes equations)
  if(this->param.convective_problem() &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    if(this->param.ale_formulation)
    {
      for(unsigned int i = 0; i < this->vec_convective_term.size(); ++i)
      {
        // in a general setting, we only know the boundary conditions at time t_{n+1}
        pde_operator->evaluate_convective_term(this->vec_convective_term[i],
                                               velocity[i],
                                               this->get_next_time());
      }
    }

    for(unsigned int i = 0; i < this->vec_convective_term.size(); ++i)
      velocity_np.add(-this->extra.get_beta(i), this->vec_convective_term[i]);
  }

  // compute body force vector
  if(this->param.right_hand_side == true)
  {
    pde_operator->evaluate_add_body_force_term(velocity_np, this->get_next_time());
  }

  // apply inverse mass matrix
  pde_operator->apply_inverse_mass_matrix(velocity_np, velocity_np);


  // calculate sum (alpha_i/dt * u_tilde_i) in case of explicit treatment of convective term
  // and operator-integration-factor (OIF) splitting
  if(this->param.convective_problem() &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    VectorType sum_alphai_ui(velocity[0]);
    this->calculate_sum_alphai_ui_oif_substepping(sum_alphai_ui, this->cfl, this->cfl_oif);
    velocity_np.add(1.0, sum_alphai_ui);
  }
  // calculate sum (alpha_i/dt * u_i) for standard BDF discretization
  else
  {
    for(unsigned int i = 0; i < velocity.size(); ++i)
    {
      velocity_np.add(this->bdf.get_alpha(i) / this->get_time_step_size(), velocity[i]);
    }
  }

  // solve discrete temporal derivative term for intermediate velocity u_hat
  velocity_np *= this->get_time_step_size() / this->bdf.get_gamma0();

  if(this->print_solver_info())
  {
    this->pcout << std::endl
                << "Solve convective step explicitly:" << std::endl
                << "  Iterations:        " << std::setw(6) << std::right << "-"
                << "\t Wall time [s]: " << std::scientific
                << timer.wall_time() + computing_time_convective << std::endl;
  }

  computing_times[0] += timer.wall_time() + computing_time_convective;
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::evaluate_convective_term()
{
  if(this->param.convective_problem() &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    if(this->param.ale_formulation == false) // Eulerian case
    {
      Timer timer;
      timer.restart();

      pde_operator->evaluate_convective_term(this->convective_term_np,
                                             velocity_np,
                                             this->get_next_time());

      computing_time_convective = timer.wall_time();
    }
  }
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::pressure_step()
{
  Timer timer;
  timer.restart();

  // compute right-hand-side vector
  VectorType rhs(pressure_np);
  rhs_pressure(rhs);

  // extrapolate old solution to get a good initial estimate for the solver
  pressure_np = 0;
  for(unsigned int i = 0; i < pressure.size(); ++i)
  {
    pressure_np.add(this->extra.get_beta(i), pressure[i]);
  }

  // solve linear system of equations
  bool const update_preconditioner =
    this->param.update_preconditioner_pressure_poisson &&
    ((this->time_step_number - 1) %
       this->param.update_preconditioner_pressure_poisson_every_time_steps ==
     0);

  unsigned int iterations_pressure =
    pde_operator->solve_pressure(pressure_np, rhs, update_preconditioner);

  // special case: pressure level is undefined
  // Adjust the pressure level in order to allow a calculation of the pressure error.
  // This is necessary because otherwise the pressure solution moves away from the exact solution.
  pde_operator->adjust_pressure_level_if_undefined(pressure_np, this->get_next_time());

  // write output
  if(this->print_solver_info())
  {
    this->pcout << std::endl
                << "Solve Poisson equation for pressure p:" << std::endl
                << "  Iterations:        " << std::setw(6) << std::right << iterations_pressure
                << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
  }

  computing_times[1] += timer.wall_time();
  iterations[1] += iterations_pressure;
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::rhs_pressure(VectorType & rhs) const
{
  /*
   *  I. calculate divergence term
   */
  // homogeneous part of velocity divergence operator
  pde_operator->apply_velocity_divergence_term(rhs, velocity_np);

  rhs *= -this->bdf.get_gamma0() / this->get_time_step_size();

  // inhomogeneous parts of boundary face integrals of velocity divergence operator
  if(this->param.divu_integrated_by_parts == true && this->param.divu_use_boundary_data == true)
  {
    VectorType temp(rhs);

    // sum alpha_i * u_i term
    for(unsigned int i = 0; i < velocity.size(); ++i)
    {
      if(this->param.store_previous_boundary_values)
        pde_operator->rhs_velocity_divergence_term_dirichlet_bc_from_dof_vector(temp,
                                                                                velocity_dbc[i]);
      else
        pde_operator->rhs_velocity_divergence_term(temp, this->get_previous_time(i));

      // note that the minus sign related to this term is already taken into account
      // in the function rhs() of the divergence operator
      rhs.add(this->bdf.get_alpha(i) / this->get_time_step_size(), temp);
    }

    // convective term
    if(this->param.convective_problem())
    {
      for(unsigned int i = 0; i < velocity.size(); ++i)
      {
        temp = 0.0;
        pde_operator->rhs_ppe_div_term_convective_term_add(temp, velocity[i]);
        rhs.add(this->extra.get_beta(i), temp);
      }
    }

    // body force term
    if(this->param.right_hand_side)
      pde_operator->rhs_ppe_div_term_body_forces_add(rhs, this->get_next_time());
  }

  /*
   *  II. calculate terms originating from inhomogeneous parts of boundary face integrals of Laplace
   * operator
   */

  // II.1. pressure Dirichlet boundary conditions
  pde_operator->rhs_ppe_laplace_add(rhs, this->get_next_time());

  // II.2. pressure Neumann boundary condition: body force vector
  if(this->param.right_hand_side)
  {
    pde_operator->rhs_ppe_nbc_body_force_term_add(rhs, this->get_next_time());
  }

  // II.3. pressure Neumann boundary condition: temporal derivative of velocity
  if(this->param.store_previous_boundary_values)
  {
#ifdef EXTRAPOLATE_ACCELERATION
    VectorType temp(rhs);
    for(unsigned int i = 0; i < extra_pressure_nbc.get_order(); ++i)
    {
      temp = 0.0;
      pde_operator->rhs_ppe_nbc_numerical_time_derivative_add(temp, acceleration[i]);
      rhs.add(this->extra_pressure_nbc.get_beta(i), temp);
    }
#else
    VectorType acceleration(velocity_dbc_np);
    compute_bdf_time_derivative(
      acceleration, velocity_dbc_np, velocity_dbc, this->bdf, this->get_time_step_size());
    pde_operator->rhs_ppe_nbc_numerical_time_derivative_add(rhs, acceleration);
#endif
  }
  else
  {
    pde_operator->rhs_ppe_nbc_analytical_time_derivative_add(rhs, this->get_next_time());
  }

  // II.4. viscous term of pressure Neumann boundary condition on Gamma_D:
  //       extrapolate velocity, evaluate vorticity, and subsequently evaluate boundary
  //       face integral (this is possible since pressure Neumann BC is linear in vorticity)
  if(this->param.viscous_problem())
  {
    if(this->param.order_extrapolation_pressure_nbc > 0)
    {
      VectorType velocity_extra(velocity[0]);
      velocity_extra = 0.0;
      for(unsigned int i = 0; i < extra_pressure_nbc.get_order(); ++i)
      {
        velocity_extra.add(this->extra_pressure_nbc.get_beta(i), velocity[i]);
      }

      VectorType vorticity(velocity_extra);
      pde_operator->compute_vorticity(vorticity, velocity_extra);

      pde_operator->rhs_ppe_viscous_add(rhs, vorticity);
    }
  }

  // II.5. convective term of pressure Neumann boundary condition on Gamma_D:
  //       evaluate convective term and subsequently extrapolate rhs vectors
  //       (the convective term is nonlinear!)
  if(this->param.convective_problem())
  {
    if(this->param.order_extrapolation_pressure_nbc > 0)
    {
      VectorType temp(rhs);
      for(unsigned int i = 0; i < extra_pressure_nbc.get_order(); ++i)
      {
        temp = 0.0;
        pde_operator->rhs_ppe_convective_add(temp, velocity[i]);
        rhs.add(this->extra_pressure_nbc.get_beta(i), temp);
      }
    }
  }

  // special case: pressure level is undefined
  // Set mean value of rhs to zero in order to obtain a consistent linear system of equations.
  // This is really necessary for the dual-splitting scheme in contrast to the pressure-correction
  // scheme and coupled solution approach due to the Dirichlet BC prescribed for the intermediate
  // velocity field and the pressure Neumann BC in case of the dual-splitting scheme.
  if(pde_operator->is_pressure_level_undefined())
    set_zero_mean_value(rhs);
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::projection_step()
{
  Timer timer;
  timer.restart();

  // extrapolate velocity to time t_n+1 and use this velocity field to
  // calculate the penalty parameter for the divergence and continuity penalty term
  if(this->param.apply_penalty_terms_in_postprocessing_step == false)
  {
    if(this->param.use_divergence_penalty == true || this->param.use_continuity_penalty == true)
    {
      VectorType velocity_extrapolated;
      velocity_extrapolated.reinit(velocity[0]);
      for(unsigned int i = 0; i < velocity.size(); ++i)
        velocity_extrapolated.add(this->extra.get_beta(i), velocity[i]);

      pde_operator->update_projection_operator(velocity_extrapolated, this->get_time_step_size());
    }
  }

  // compute right-hand-side vector
  VectorType rhs(velocity_np);
  rhs_projection(rhs);

  // apply inverse mass matrix: this is the solution if no penalty terms are applied
  // and serves as a good initial guess for the case with penalty terms
  pde_operator->apply_inverse_mass_matrix(velocity_np, rhs);

  // penalty terms
  unsigned int iterations_projection = 0;

  if(this->param.apply_penalty_terms_in_postprocessing_step == false)
  {
    if(this->param.use_divergence_penalty == true || this->param.use_continuity_penalty == true)
    {
      // solve linear system of equations
      bool const update_preconditioner =
        this->param.update_preconditioner_projection &&
        ((this->time_step_number - 1) %
           this->param.update_preconditioner_projection_every_time_steps ==
         0);

      iterations_projection =
        pde_operator->solve_projection(velocity_np, rhs, update_preconditioner);
    }
  }

  // write output
  if(this->print_solver_info())
  {
    this->pcout << std::endl
                << "Solve projection step for intermediate velocity:" << std::endl
                << "  Iterations:        " << std::setw(6) << std::right << iterations_projection
                << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
  }

  computing_times[2] += timer.wall_time();
  iterations[2] += iterations_projection;
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::rhs_projection(VectorType & rhs) const
{
  /*
   *  I. calculate pressure gradient term
   */
  pde_operator->evaluate_pressure_gradient_term(rhs, pressure_np, this->get_next_time());

  rhs *= -this->get_time_step_size() / this->bdf.get_gamma0();

  /*
   *  II. add mass matrix term
   */
  pde_operator->apply_mass_matrix_add(rhs, velocity_np);
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::viscous_step()
{
  Timer timer;
  timer.restart();

  // if a turbulence model is used:
  // update turbulence model before calculating rhs_viscous
  if(this->param.use_turbulence_model == true)
  {
    Timer timer_turbulence;
    timer_turbulence.restart();

    // extrapolate velocity to time t_n+1 and use this velocity field to
    // update the turbulence model (to recalculate the turbulent viscosity)
    VectorType velocity_extrapolated(velocity[0]);
    velocity_extrapolated = 0;
    for(unsigned int i = 0; i < velocity.size(); ++i)
      velocity_extrapolated.add(this->extra.get_beta(i), velocity[i]);

    pde_operator->update_turbulence_model(velocity_extrapolated);

    if(this->print_solver_info())
    {
      this->pcout << std::endl
                  << "Update of turbulent viscosity:   Wall time [s]: " << std::scientific
                  << timer_turbulence.wall_time() << std::endl;
    }
  }

  if(this->param.viscous_problem())
  {
    VectorType rhs(velocity_np);
    // compute right-hand-side vector
    rhs_viscous(rhs);

    // Extrapolate old solution to get a good initial estimate for the solver.
    // Note that this has to be done after calling rhs_viscous()!
    velocity_np = 0;
    for(unsigned int i = 0; i < velocity.size(); ++i)
      velocity_np.add(this->extra.get_beta(i), velocity[i]);

    // solve linear system of equations
    bool const update_preconditioner =
      this->param.update_preconditioner_viscous &&
      ((this->time_step_number - 1) % this->param.update_preconditioner_viscous_every_time_steps ==
       0);

    unsigned int iterations_viscous = pde_operator->solve_viscous(
      velocity_np, rhs, update_preconditioner, this->get_scaling_factor_time_derivative_term());

    // write output
    if(this->print_solver_info())
    {
      this->pcout << std::endl
                  << "Solve viscous step for velocity u:" << std::endl
                  << "  Iterations:        " << std::setw(6) << std::right << iterations_viscous
                  << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
    }

    iterations[3] += iterations_viscous;
  }
  else // inviscid
  {
    // nothing to do
    AssertThrow(this->param.equation_type == EquationType::Euler, ExcMessage("Logical error."));
  }

  computing_times[3] += timer.wall_time();
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::rhs_viscous(VectorType & rhs) const
{
  /*
   *  I. calculate mass matrix term
   */
  pde_operator->apply_mass_matrix(rhs, velocity_np);
  rhs *= this->bdf.get_gamma0() / this->get_time_step_size();

  /*
   *  II. inhomogeneous parts of boundary face integrals of viscous operator
   */
  pde_operator->rhs_add_viscous_term(rhs, this->get_next_time());
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::penalty_step()
{
  Timer timer;
  timer.restart();

  if(this->param.use_divergence_penalty == true || this->param.use_continuity_penalty == true)
  {
    // extrapolate velocity to time t_n+1 and use this velocity field to
    // calculate the penalty parameter for the divergence and continuity penalty term
    VectorType velocity_extrapolated;
    velocity_extrapolated.reinit(velocity[0]);
    for(unsigned int i = 0; i < velocity.size(); ++i)
      velocity_extrapolated.add(this->extra.get_beta(i), velocity[i]);

    pde_operator->update_projection_operator(velocity_extrapolated, this->get_time_step_size());

    // compute right-hand-side vector
    VectorType rhs(velocity_np);
    pde_operator->apply_mass_matrix(rhs, velocity_np);

    // right-hand side term: add inhomogeneous contributions of continuity penalty operator to
    // rhs-vector if desired
    if(this->param.use_continuity_penalty && this->param.continuity_penalty_use_boundary_data)
      pde_operator->rhs_add_projection_operator(rhs, this->get_next_time());

    // solve linear system of equations
    bool const update_preconditioner =
      this->param.update_preconditioner_projection &&
      ((this->time_step_number - 1) %
         this->param.update_preconditioner_projection_every_time_steps ==
       0);

    // use solution of previous step as initial guess
    unsigned int iterations_projection =
      pde_operator->solve_projection(velocity_np, rhs, update_preconditioner);

    // write output
    if(this->print_solver_info())
    {
      this->pcout << std::endl
                  << "Solve penalty step:" << std::endl
                  << "  Iterations:        " << std::setw(6) << std::right << iterations_projection
                  << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
    }

    iterations[4] += iterations_projection;
  }

  computing_times[4] += timer.wall_time();
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::prepare_vectors_for_next_timestep()
{
  Base::prepare_vectors_for_next_timestep();

  if(this->param.store_previous_boundary_values)
  {
#ifdef EXTRAPOLATE_ACCELERATION
    // push back accelerations and compute new acceleration at the end
    // of the current time step before velocity_dbc is pushed back
    // no need to move the mesh here since we still have the mesh Omega_{n+1} at this point!

    pde_operator->interpolate_velocity_dirichlet_bc(velocity_dbc_np, this->get_next_time());

    push_back(acceleration);

    if(this->param.order_extrapolation_pressure_nbc)
    {
      compute_bdf_time_derivative(
        acceleration[0], velocity_dbc_np, velocity_dbc, this->bdf, this->get_time_step_size());
    }

    push_back(velocity_dbc);
    velocity_dbc[0].swap(velocity_dbc_np);


    // Variant 3:
    // Compute acceleration and velocity vectors used for inhomogeneous boundary condition terms
    // on the rhs of the pressure Poisson equation as a function of the numerical solution for the
    // velocity. Note: This variant leads to instabilities for small time step sizes.
    /*
    push_back(acceleration);

    if(this->param.order_extrapolation_pressure_nbc)
    {
      compute_bdf_time_derivative(
        acceleration[0], velocity_np, velocity, this->bdf, this->get_time_step_size());
    }

    push_back(velocity_dbc);
    velocity_dbc[0].swap(velocity_np);
    */

    // Variant 3b:
    // Compute acceleration used for inhomogeneous boundary condition terms
    // on the rhs of the pressure Poisson equation as a function of the numerical solution for the
    // velocity. But use boundary data g_u for velocity_dbc for intermediate velocity u_hat.
    /*
    push_back(acceleration);

    // use interior solution u only for computation of time derivative (acceleration)
    if(this->param.order_extrapolation_pressure_nbc)
    {
      compute_bdf_time_derivative(
        acceleration[0], velocity_np, velocity, this->bdf, this->get_time_step_size());
    }

    // use boundary condition g_u for velocity_dbc!
    // no need to move the mesh here since we still have the mesh Omega_{n+1} at this point!
    pde_operator->interpolate_velocity_dirichlet_bc(velocity_dbc_np,
    this->get_next_time());

    push_back(velocity_dbc);
    velocity_dbc[0].swap(velocity_dbc_np);
    */
#else
    // If we do not extrapolate the acceleration, we only have to care about the history of
    // velocity Dirichlet boundary conditions, where velocity_dbc_np has already been updated.
    push_back(velocity_dbc);
    velocity_dbc[0].swap(velocity_dbc_np);
#endif
  }

  push_back(velocity);
  velocity[0].swap(velocity_np);

  push_back(pressure);
  pressure[0].swap(pressure_np);
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::solve_steady_problem()
{
  this->pcout << std::endl << "Starting time loop ..." << std::endl;

  // pseudo-time integration in order to solve steady-state problem
  bool converged = false;

  if(this->param.convergence_criterion_steady_problem ==
     ConvergenceCriterionSteadyProblem::SolutionIncrement)
  {
    VectorType velocity_tmp;
    VectorType pressure_tmp;

    while(!converged && this->time < (this->end_time - this->eps) &&
          this->get_time_step_number() <= this->param.max_number_of_time_steps)
    {
      // save solution from previous time step
      velocity_tmp = this->velocity[0];
      pressure_tmp = this->pressure[0];

      // calculate norm of solution
      double const norm_u = velocity_tmp.l2_norm();
      double const norm_p = pressure_tmp.l2_norm();
      double const norm   = std::sqrt(norm_u * norm_u + norm_p * norm_p);

      // solve time step
      this->do_timestep(false);

      // calculate increment:
      // increment = solution_{n+1} - solution_{n}
      //           = solution[0] - solution_tmp
      velocity_tmp *= -1.0;
      pressure_tmp *= -1.0;
      velocity_tmp.add(1.0, this->velocity[0]);
      pressure_tmp.add(1.0, this->pressure[0]);

      double const incr_u   = velocity_tmp.l2_norm();
      double const incr_p   = pressure_tmp.l2_norm();
      double const incr     = std::sqrt(incr_u * incr_u + incr_p * incr_p);
      double       incr_rel = 1.0;
      if(norm > 1.0e-10)
        incr_rel = incr / norm;

      // write output
      if(this->print_solver_info())
      {
        this->pcout << std::endl
                    << "Norm of solution increment:" << std::endl
                    << "  ||incr_abs|| = " << std::scientific << std::setprecision(10) << incr
                    << std::endl
                    << "  ||incr_rel|| = " << std::scientific << std::setprecision(10) << incr_rel
                    << std::endl;
      }

      // check convergence
      if(incr < this->param.abs_tol_steady || incr_rel < this->param.rel_tol_steady)
      {
        converged = true;
      }
    }
  }
  else if(this->param.convergence_criterion_steady_problem ==
          ConvergenceCriterionSteadyProblem::ResidualSteadyNavierStokes)
  {
    AssertThrow(this->param.convergence_criterion_steady_problem !=
                  ConvergenceCriterionSteadyProblem::ResidualSteadyNavierStokes,
                ExcMessage("This option is not available for the dual splitting scheme. "
                           "Due to splitting errors the solution does not fulfill the "
                           "residual of the steady, incompressible Navier-Stokes equations."));
  }
  else
  {
    AssertThrow(false, ExcMessage("not implemented."));
  }

  AssertThrow(
    converged == true,
    ExcMessage(
      "Maximum number of time steps or end time exceeded! This might be due to the fact that "
      "(i) the maximum number of time steps is simply too small to reach a steady solution, "
      "(ii) the problem is unsteady so that the applied solution approach is inappropriate, "
      "(iii) some of the solver tolerances are in conflict."));

  this->pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::get_iterations(std::vector<std::string> & name,
                                                     std::vector<double> &      iteration) const
{
  unsigned int             size  = 4;
  std::vector<std::string> names = {"Convection", "Pressure", "Projection", "Viscous"};

  if(this->param.apply_penalty_terms_in_postprocessing_step)
  {
    names.push_back("Penalty terms");
    size++;
  }

  unsigned int N_time_steps = this->get_time_step_number() - 1;

  name.resize(size);
  iteration.resize(size);
  for(unsigned int i = 0; i < size; ++i)
  {
    name[i]      = names[i];
    iteration[i] = (double)this->iterations[i] / (double)N_time_steps;
  }
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::get_wall_times(std::vector<std::string> & name,
                                                     std::vector<double> &      wall_time) const
{
  unsigned int             size  = 4;
  std::vector<std::string> names = {"Convection", "Pressure", "Projection", "Viscous"};

  if(this->param.apply_penalty_terms_in_postprocessing_step)
  {
    names.push_back("Penalty terms");
    size++;
  }

  name.resize(size);
  wall_time.resize(size);
  for(unsigned int i = 0; i < size; ++i)
  {
    name[i]      = names[i];
    wall_time[i] = this->computing_times[i];
  }
}

// instantiations

template class TimeIntBDFDualSplitting<2, float>;
template class TimeIntBDFDualSplitting<2, double>;

template class TimeIntBDFDualSplitting<3, float>;
template class TimeIntBDFDualSplitting<3, double>;

} // namespace IncNS
