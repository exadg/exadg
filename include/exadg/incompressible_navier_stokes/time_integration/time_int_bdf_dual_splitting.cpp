/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>
#include <exadg/time_integration/push_back_vectors.h>
#include <exadg/time_integration/time_step_calculation.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
TimeIntBDFDualSplitting<dim, Number>::TimeIntBDFDualSplitting(
  std::shared_ptr<Operator>                       operator_in,
  Parameters const &                              param_in,
  MPI_Comm const &                                mpi_comm_in,
  bool const                                      is_test_in,
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in)
  : Base(operator_in, param_in, mpi_comm_in, is_test_in, postprocessor_in),
    pde_operator(operator_in),
    velocity(this->order),
    pressure(this->order),
    velocity_dbc(this->order),
    iterations_pressure({0, 0}),
    iterations_projection({0, 0}),
    iterations_viscous({0, 0}),
    iterations_penalty({0, 0}),
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

  // velocity_dbc vectors do not have to be initialized in case of a restart, where
  // the vectors are read from restart files.
  if(this->param.store_previous_boundary_values && this->param.restarted_simulation == false)
  {
    initialize_velocity_dbc();
  }
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::read_restart_vectors(boost::archive::binary_iarchive & ia)
{
  Base::read_restart_vectors(ia);

  if(this->param.store_previous_boundary_values)
  {
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

  // velocity_dbc
  if(this->param.store_previous_boundary_values)
  {
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
    pde_operator->move_grid(this->get_time());

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
      pde_operator->move_grid(this->get_previous_time(i));

    pde_operator->prescribe_initial_conditions(velocity[i],
                                               pressure[i],
                                               this->get_previous_time(i));
  }
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::initialize_velocity_dbc()
{
  // fill vector velocity_dbc: The first entry [0] is already needed if start_with_low_order == true
  if(this->param.ale_formulation)
    pde_operator->move_grid_and_update_dependent_data_structures(this->get_time());
  pde_operator->interpolate_velocity_dirichlet_bc(velocity_dbc[0], this->get_time());
  // ... and previous times if start_with_low_order == false
  if(this->start_with_low_order == false)
  {
    for(unsigned int i = 1; i < velocity_dbc.size(); ++i)
    {
      double const time = this->get_time() - double(i) * this->get_time_step_size();
      if(this->param.ale_formulation)
        pde_operator->move_grid_and_update_dependent_data_structures(time);
      pde_operator->interpolate_velocity_dirichlet_bc(velocity_dbc[i], time);
    }
  }
}

template<int dim, typename Number>
typename TimeIntBDFDualSplitting<dim, Number>::VectorType const &
TimeIntBDFDualSplitting<dim, Number>::get_velocity() const
{
  return velocity[0];
}

template<int dim, typename Number>
typename TimeIntBDFDualSplitting<dim, Number>::VectorType const &
TimeIntBDFDualSplitting<dim, Number>::get_velocity_np() const
{
  return velocity_np;
}

template<int dim, typename Number>
typename TimeIntBDFDualSplitting<dim, Number>::VectorType const &
TimeIntBDFDualSplitting<dim, Number>::get_pressure_np() const
{
  return pressure_np;
}

template<int dim, typename Number>
typename TimeIntBDFDualSplitting<dim, Number>::VectorType const &
TimeIntBDFDualSplitting<dim, Number>::get_velocity(unsigned int i) const
{
  return velocity[i];
}

template<int dim, typename Number>
typename TimeIntBDFDualSplitting<dim, Number>::VectorType const &
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
              dealii::ExcMessage("Order of BDF scheme has to be 1 for this stability analysis."));

  AssertThrow(this->param.convective_problem() == false,
              dealii::ExcMessage(
                "Stability analysis can not be performed for nonlinear convective problems."));

  AssertThrow(velocity[0].l2_norm() < 1.e-15 && pressure[0].l2_norm() < 1.e-15,
              dealii::ExcMessage("Solution vector has to be zero for this stability analysis."));

  AssertThrow(dealii::Utilities::MPI::n_mpi_processes(this->mpi_comm) == 1,
              dealii::ExcMessage("Number of MPI processes has to be 1."));

  std::cout << std::endl << "Analysis of eigenvalue spectrum:" << std::endl;

  unsigned int const size = velocity[0].locally_owned_size();

  dealii::LAPACKFullMatrix<Number> propagation_matrix(size, size);

  // loop over all columns of propagation matrix
  for(unsigned int j = 0; j < size; ++j)
  {
    // set j-th element to 1
    velocity[0].local_element(j) = 1.0;

    // solve time step
    this->do_timestep_solve();

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
TimeIntBDFDualSplitting<dim, Number>::do_timestep_solve()
{
  // pre-computations
  if(this->param.store_previous_boundary_values)
    pde_operator->interpolate_velocity_dirichlet_bc(velocity_dbc_np, this->get_next_time());

  // perform the sub-steps of the dual-splitting method
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
TimeIntBDFDualSplitting<dim, Number>::convective_step()
{
  dealii::Timer timer;
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

  // apply inverse mass operator
  pde_operator->apply_inverse_mass_operator(velocity_np, velocity_np);


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

  if(this->print_solver_info() and not(this->is_test))
  {
    this->pcout << std::endl << "Explicit convective step:";
    print_wall_time(this->pcout, timer.wall_time());
  }

  this->timer_tree->insert({"Timeloop", "Convective step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::evaluate_convective_term()
{
  dealii::Timer timer;
  timer.restart();

  if(this->param.convective_problem() &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    if(this->param.ale_formulation == false) // Eulerian case
    {
      pde_operator->evaluate_convective_term(this->convective_term_np,
                                             velocity_np,
                                             this->get_next_time());
    }
  }

  this->timer_tree->insert({"Timeloop", "Convective step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::pressure_step()
{
  dealii::Timer timer;
  timer.restart();

  // compute right-hand-side vector
  VectorType rhs(pressure_np);
  rhs_pressure(rhs);

  // extrapolate old solution to get a good initial estimate for the solver
  if(this->use_extrapolation)
  {
    pressure_np = 0;
    for(unsigned int i = 0; i < pressure.size(); ++i)
    {
      pressure_np.add(this->extra.get_beta(i), pressure[i]);
    }
  }
  else
  {
    pressure_np = pressure_last_iter;
  }

  // solve linear system of equations
  bool const update_preconditioner =
    this->param.update_preconditioner_pressure_poisson &&
    ((this->time_step_number - 1) %
       this->param.update_preconditioner_pressure_poisson_every_time_steps ==
     0);

  unsigned int const n_iter = pde_operator->solve_pressure(pressure_np, rhs, update_preconditioner);
  iterations_pressure.first += 1;
  iterations_pressure.second += n_iter;

  // special case: pressure level is undefined
  // Adjust the pressure level in order to allow a calculation of the pressure error.
  // This is necessary because otherwise the pressure solution moves away from the exact solution.
  pde_operator->adjust_pressure_level_if_undefined(pressure_np, this->get_next_time());

  if(this->store_solution)
    pressure_last_iter = pressure_np;

  // write output
  if(this->print_solver_info() and not(this->is_test))
  {
    this->pcout << std::endl << "Solve pressure step:";
    print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
  }

  this->timer_tree->insert({"Timeloop", "Pressure step"}, timer.wall_time());
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
      pde_operator->rhs_velocity_divergence_term_dirichlet_bc_from_dof_vector(temp,
                                                                              velocity_dbc[i]);

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
  VectorType acceleration(velocity_dbc_np);
  compute_bdf_time_derivative(
    acceleration, velocity_dbc_np, velocity_dbc, this->bdf, this->get_time_step_size());
  pde_operator->rhs_ppe_nbc_numerical_time_derivative_add(rhs, acceleration);

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

      pde_operator->rhs_ppe_nbc_viscous_add(rhs, vorticity);
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
        pde_operator->rhs_ppe_nbc_convective_add(temp, velocity[i]);
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
  dealii::Timer timer;
  timer.restart();

  // compute right-hand-side vector
  VectorType rhs(velocity_np);
  rhs_projection(rhs);

  // apply inverse mass operator: this is the solution if no penalty terms are applied
  // and serves as a good initial guess for the case with penalty terms
  pde_operator->apply_inverse_mass_operator(velocity_np, rhs);

  // penalty terms
  if(this->param.apply_penalty_terms_in_postprocessing_step == false &&
     (this->param.use_divergence_penalty == true || this->param.use_continuity_penalty == true))
  {
    // extrapolate velocity to time t_n+1 and use this velocity field to
    // calculate the penalty parameter for the divergence and continuity penalty term
    VectorType velocity_extrapolated;
    if(this->use_extrapolation)
    {
      velocity_extrapolated.reinit(velocity[0]);
      for(unsigned int i = 0; i < velocity.size(); ++i)
        velocity_extrapolated.add(this->extra.get_beta(i), velocity[i]);
    }
    else
    {
      velocity_extrapolated = velocity_projection_last_iter;
    }

    pde_operator->update_projection_operator(velocity_extrapolated, this->get_time_step_size());

    // solve linear system of equations
    bool const update_preconditioner =
      this->param.update_preconditioner_projection &&
      ((this->time_step_number - 1) %
         this->param.update_preconditioner_projection_every_time_steps ==
       0);

    if(this->use_extrapolation == false)
      velocity_np = velocity_projection_last_iter;

    unsigned int n_iter = pde_operator->solve_projection(velocity_np, rhs, update_preconditioner);
    iterations_projection.first += 1;
    iterations_projection.second += n_iter;

    if(this->store_solution)
      velocity_projection_last_iter = velocity_np;

    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl << "Solve projection step:";
      print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
    }
  }
  else // no penalty terms
  {
    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl << "Explicit projection step:";
      print_wall_time(this->pcout, timer.wall_time());
    }
  }

  this->timer_tree->insert({"Timeloop", "Pojection step"}, timer.wall_time());
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
   *  II. add mass operator term
   */
  pde_operator->apply_mass_operator_add(rhs, velocity_np);
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::viscous_step()
{
  dealii::Timer timer;
  timer.restart();

  if(this->param.viscous_problem())
  {
    // if a turbulence model is used:
    // update turbulence model before calculating rhs_viscous
    if(this->param.use_turbulence_model == true)
    {
      dealii::Timer timer_turbulence;
      timer_turbulence.restart();

      // extrapolate velocity to time t_n+1 and use this velocity field to
      // update the turbulence model (to recalculate the turbulent viscosity)
      VectorType velocity_extrapolated(velocity[0]);
      velocity_extrapolated = 0;
      for(unsigned int i = 0; i < velocity.size(); ++i)
        velocity_extrapolated.add(this->extra.get_beta(i), velocity[i]);

      pde_operator->update_turbulence_model(velocity_extrapolated);

      if(this->print_solver_info() and not(this->is_test))
      {
        this->pcout << std::endl << "Update of turbulent viscosity:";
        print_wall_time(this->pcout, timer_turbulence.wall_time());
      }
    }

    VectorType rhs(velocity_np);
    // compute right-hand-side vector
    rhs_viscous(rhs);

    // Extrapolate old solution to get a good initial estimate for the solver.
    // Note that this has to be done after calling rhs_viscous()!
    if(this->use_extrapolation)
    {
      velocity_np = 0;
      for(unsigned int i = 0; i < velocity.size(); ++i)
        velocity_np.add(this->extra.get_beta(i), velocity[i]);
    }
    else
    {
      velocity_np = velocity_viscous_last_iter;
    }

    // solve linear system of equations
    bool const update_preconditioner =
      this->param.update_preconditioner_viscous &&
      ((this->time_step_number - 1) % this->param.update_preconditioner_viscous_every_time_steps ==
       0);

    unsigned int const n_iter = pde_operator->solve_viscous(
      velocity_np, rhs, update_preconditioner, this->get_scaling_factor_time_derivative_term());
    iterations_viscous.first += 1;
    iterations_viscous.second += n_iter;

    if(this->store_solution)
      velocity_viscous_last_iter = velocity_np;

    // write output
    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl << "Solve viscous step:";
      print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
    }
  }
  else // inviscid
  {
    // nothing to do
    AssertThrow(this->param.equation_type == EquationType::Euler,
                dealii::ExcMessage("Logical error."));
  }

  this->timer_tree->insert({"Timeloop", "Viscous step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::rhs_viscous(VectorType & rhs) const
{
  /*
   *  I. apply mass operator
   */
  pde_operator->apply_mass_operator(rhs, velocity_np);
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
  if(this->param.use_divergence_penalty == true || this->param.use_continuity_penalty == true)
  {
    dealii::Timer timer;
    timer.restart();

    // compute right-hand-side vector
    VectorType rhs(velocity_np);
    pde_operator->apply_mass_operator(rhs, velocity_np);

    // extrapolate velocity to time t_n+1 and use this velocity field to
    // calculate the penalty parameter for the divergence and continuity penalty term
    VectorType velocity_extrapolated(velocity_np);
    velocity_extrapolated = 0.0;
    for(unsigned int i = 0; i < velocity.size(); ++i)
      velocity_extrapolated.add(this->extra.get_beta(i), velocity[i]);

    pde_operator->update_projection_operator(velocity_extrapolated, this->get_time_step_size());

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

    if(this->use_extrapolation == false)
      velocity_np = velocity_projection_last_iter;

    unsigned int const n_iter =
      pde_operator->solve_projection(velocity_np, rhs, update_preconditioner);

    iterations_penalty.first += 1;
    iterations_penalty.second += n_iter;

    if(this->store_solution)
      velocity_projection_last_iter = velocity_np;

    // write output
    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl << "Solve penalty step:";
      print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
    }

    this->timer_tree->insert({"Timeloop", "Penalty step"}, timer.wall_time());
  }
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::prepare_vectors_for_next_timestep()
{
  Base::prepare_vectors_for_next_timestep();

  if(this->param.store_previous_boundary_values)
  {
    // We have to care about the history of velocity Dirichlet boundary conditions,
    // where velocity_dbc_np has already been updated.
    push_back(velocity_dbc);
    velocity_dbc[0].swap(velocity_dbc_np);
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
      this->do_timestep();

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
                dealii::ExcMessage(
                  "This option is not available for the dual splitting scheme. "
                  "Due to splitting errors the solution does not fulfill the "
                  "residual of the steady, incompressible Navier-Stokes equations."));
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("not implemented."));
  }

  AssertThrow(
    converged == true,
    dealii::ExcMessage(
      "Maximum number of time steps or end time exceeded! This might be due to the fact that "
      "(i) the maximum number of time steps is simply too small to reach a steady solution, "
      "(ii) the problem is unsteady so that the applied solution approach is inappropriate, "
      "(iii) some of the solver tolerances are in conflict."));

  this->pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::print_iterations() const
{
  std::vector<std::string> names = {"Convective step",
                                    "Pressure step",
                                    "Projection step",
                                    "Viscous step"};

  std::vector<double> iterations_avg;
  iterations_avg.resize(4);
  iterations_avg[0] = 0.0;
  iterations_avg[1] =
    (double)iterations_pressure.second / std::max(1., (double)iterations_pressure.first);
  iterations_avg[2] =
    (double)iterations_projection.second / std::max(1., (double)iterations_projection.first);
  iterations_avg[3] =
    (double)iterations_viscous.second / std::max(1., (double)iterations_viscous.first);

  if(this->param.apply_penalty_terms_in_postprocessing_step)
  {
    names.push_back("Penalty step");
    iterations_avg.push_back((double)iterations_penalty.second /
                             std::max(1., (double)iterations_penalty.first));
  }

  print_list_of_iterations(this->pcout, names, iterations_avg);
}

// instantiations

template class TimeIntBDFDualSplitting<2, float>;
template class TimeIntBDFDualSplitting<2, double>;

template class TimeIntBDFDualSplitting<3, float>;
template class TimeIntBDFDualSplitting<3, double>;

} // namespace IncNS
} // namespace ExaDG
