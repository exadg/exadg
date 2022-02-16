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

// ExaDG
#include <exadg/fluid_structure_interaction/acceleration_schemes/linear_algebra.h>
#include <exadg/fluid_structure_interaction/driver.h>
#include <exadg/utilities/print_general_infos.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace FSI
{
template<int dim, typename Number>
Driver<dim, Number>::Driver(std::string const &                           input_file,
                            MPI_Comm const &                              comm,
                            std::shared_ptr<ApplicationBase<dim, Number>> app,
                            bool const                                    is_test)
  : mpi_comm(comm),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0),
    is_test(is_test),
    application(app),
    partitioned_iterations({0, 0})
{
  print_general_info<Number>(pcout, mpi_comm, is_test);

  dealii::ParameterHandler prm;
  parameters.add_parameters(prm);
  prm.parse_input(input_file, "", true, true);

  structure = std::make_shared<WrapperStructure<dim, Number>>();
  fluid     = std::make_shared<WrapperFluid<dim, Number>>();
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup()
{
  dealii::Timer timer;
  timer.restart();

  pcout << std::endl << "Setting up fluid-structure interaction solver:" << std::endl;

  // setup application
  {
    dealii::Timer timer_local;

    application->setup();

    timer_tree.insert({"FSI", "Setup", "Application"}, timer_local.wall_time());
  }

  // setup structure
  {
    dealii::Timer timer_local;

    structure->setup(application->structure, mpi_comm, is_test);

    timer_tree.insert({"FSI", "Setup", "Structure"}, timer_local.wall_time());
  }

  // setup fluid
  {
    dealii::Timer timer_local;

    fluid->setup(application->fluid, mpi_comm, is_test);

    timer_tree.insert({"FSI", "Setup", "Fluid"}, timer_local.wall_time());
  }

  // setup interface coupling
  setup_interface_coupling();

  timer_tree.insert({"FSI", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup_interface_coupling()
{
  // structure to ALE
  {
    dealii::Timer timer_local;
    timer_local.restart();

    pcout << std::endl << "Setup interface coupling structure -> ALE ..." << std::endl;

    if(application->fluid->get_parameters().mesh_movement_type == IncNS::MeshMovementType::Poisson)
    {
      structure_to_ale = std::make_shared<InterfaceCoupling<dim, dim, Number>>();
      structure_to_ale->setup(fluid->ale_poisson_operator->get_container_interface_data(),
                              application->structure->get_boundary_descriptor()->neumann_cached_bc,
                              structure->pde_operator->get_dof_handler(),
                              *application->structure->get_grid()->mapping,
                              parameters.geometric_tolerance);
    }
    else if(application->fluid->get_parameters().mesh_movement_type ==
            IncNS::MeshMovementType::Elasticity)
    {
      structure_to_ale = std::make_shared<InterfaceCoupling<dim, dim, Number>>();
      structure_to_ale->setup(
        fluid->ale_elasticity_operator->get_container_interface_data_dirichlet(),
        application->structure->get_boundary_descriptor()->neumann_cached_bc,
        structure->pde_operator->get_dof_handler(),
        *application->structure->get_grid()->mapping,
        parameters.geometric_tolerance);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("not implemented."));
    }

    pcout << std::endl << "... done!" << std::endl;

    timer_tree.insert({"FSI", "Setup", "Coupling structure -> ALE"}, timer_local.wall_time());
  }

  // structure to fluid
  {
    dealii::Timer timer_local;
    timer_local.restart();

    pcout << std::endl << "Setup interface coupling structure -> fluid ..." << std::endl;

    structure_to_fluid = std::make_shared<InterfaceCoupling<dim, dim, Number>>();
    structure_to_fluid->setup(fluid->pde_operator->get_container_interface_data(),
                              application->structure->get_boundary_descriptor()->neumann_cached_bc,
                              structure->pde_operator->get_dof_handler(),
                              *application->structure->get_grid()->mapping,
                              parameters.geometric_tolerance);

    pcout << std::endl << "... done!" << std::endl;

    timer_tree.insert({"FSI", "Setup", "Coupling structure -> fluid"}, timer_local.wall_time());
  }

  // fluid to structure
  {
    dealii::Timer timer_local;
    timer_local.restart();

    pcout << std::endl << "Setup interface coupling fluid -> structure ..." << std::endl;

    fluid_to_structure = std::make_shared<InterfaceCoupling<dim, dim, Number>>();
    std::shared_ptr<dealii::Mapping<dim> const> mapping_fluid =
      get_dynamic_mapping<dim, Number>(application->fluid->get_grid(), fluid->ale_grid_motion);
    fluid_to_structure->setup(
      structure->pde_operator->get_container_interface_data_neumann(),
      application->fluid->get_boundary_descriptor()->velocity->dirichlet_cached_bc,
      fluid->pde_operator->get_dof_handler_u(),
      *mapping_fluid,
      parameters.geometric_tolerance);

    pcout << std::endl << "... done!" << std::endl;

    timer_tree.insert({"FSI", "Setup", "Coupling fluid -> structure"}, timer_local.wall_time());
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::set_start_time() const
{
  // The fluid domain is the master that dictates the start time
  structure->time_integrator->reset_time(fluid->time_integrator->get_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::synchronize_time_step_size() const
{
  // The fluid domain is the master that dictates the time step size
  structure->time_integrator->set_current_time_step_size(
    fluid->time_integrator->get_time_step_size());
}

template<int dim, typename Number>
void
Driver<dim, Number>::coupling_structure_to_ale(VectorType const & displacement_structure) const
{
  dealii::Timer sub_timer;
  sub_timer.restart();

  structure_to_ale->update_data(displacement_structure);
  timer_tree.insert({"FSI", "Coupling structure -> ALE"}, sub_timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::coupling_structure_to_fluid(bool const extrapolate) const
{
  dealii::Timer sub_timer;
  sub_timer.restart();

  VectorType velocity_structure;
  structure->pde_operator->initialize_dof_vector(velocity_structure);
  if(extrapolate)
    structure->time_integrator->extrapolate_velocity_to_np(velocity_structure);
  else
    velocity_structure = structure->time_integrator->get_velocity_np();

  structure_to_fluid->update_data(velocity_structure);

  timer_tree.insert({"FSI", "Coupling structure -> fluid"}, sub_timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::coupling_fluid_to_structure(bool const end_of_time_step) const
{
  dealii::Timer sub_timer;
  sub_timer.restart();

  VectorType stress_fluid;
  fluid->pde_operator->initialize_vector_velocity(stress_fluid);
  // calculate fluid stress at fluid-structure interface
  if(end_of_time_step)
  {
    fluid->pde_operator->interpolate_stress_bc(stress_fluid,
                                               fluid->time_integrator->get_velocity_np(),
                                               fluid->time_integrator->get_pressure_np());
  }
  else
  {
    fluid->pde_operator->interpolate_stress_bc(stress_fluid,
                                               fluid->time_integrator->get_velocity(),
                                               fluid->time_integrator->get_pressure());
  }

  stress_fluid *= -1.0;
  fluid_to_structure->update_data(stress_fluid);

  timer_tree.insert({"FSI", "Coupling fluid -> structure"}, sub_timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::apply_dirichlet_neumann_scheme(VectorType &       d_tilde,
                                                    VectorType const & d,
                                                    unsigned int       iteration) const
{
  coupling_structure_to_ale(d);

  // move the fluid mesh and update dependent data structures
  fluid->solve_ale(application->fluid, is_test);

  // update velocity boundary condition for fluid
  coupling_structure_to_fluid(iteration == 0);

  // solve fluid problem
  fluid->time_integrator->advance_one_timestep_partitioned_solve(iteration == 0);

  // update stress boundary condition for solid
  coupling_fluid_to_structure(/* end_of_time_step = */ true);

  // solve structural problem
  structure->time_integrator->advance_one_timestep_partitioned_solve(iteration == 0);

  d_tilde = structure->time_integrator->get_displacement_np();
}

template<int dim, typename Number>
bool
Driver<dim, Number>::check_convergence(VectorType const & residual) const
{
  double const residual_norm = residual.l2_norm();
  double const ref_norm_abs  = std::sqrt(structure->pde_operator->get_number_of_dofs());
  double const ref_norm_rel  = structure->time_integrator->get_velocity_np().l2_norm() *
                              structure->time_integrator->get_time_step_size();

  bool const converged = (residual_norm < parameters.abs_tol * ref_norm_abs) ||
                         (residual_norm < parameters.rel_tol * ref_norm_rel);

  return converged;
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_solver_info_header(unsigned int const iteration) const
{
  if(fluid->time_integrator->print_solver_info())
  {
    pcout << std::endl
          << "======================================================================" << std::endl
          << " Partitioned FSI: iteration counter = " << std::left << std::setw(8) << iteration
          << std::endl
          << "======================================================================" << std::endl;
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_solver_info_converged(unsigned int const iteration) const
{
  if(fluid->time_integrator->print_solver_info())
  {
    pcout << std::endl
          << "Partitioned FSI iteration converged in " << iteration << " iterations." << std::endl;
  }
}

template<int dim, typename Number>
unsigned int
Driver<dim, Number>::solve_partitioned_problem() const
{
  // iteration counter
  unsigned int k = 0;

  // fixed-point iteration with dynamic relaxation (Aitken relaxation)
  if(parameters.method == "Aitken")
  {
    VectorType r_old, d;
    structure->pde_operator->initialize_dof_vector(r_old);
    structure->pde_operator->initialize_dof_vector(d);

    bool   converged = false;
    double omega     = 1.0;
    while(not converged and k < parameters.partitioned_iter_max)
    {
      print_solver_info_header(k);

      if(k == 0)
        structure->time_integrator->extrapolate_displacement_to_np(d);
      else
        d = structure->time_integrator->get_displacement_np();

      VectorType d_tilde(d);
      apply_dirichlet_neumann_scheme(d_tilde, d, k);

      // compute residual and check convergence
      VectorType r = d_tilde;
      r.add(-1.0, d);
      converged = check_convergence(r);

      // relaxation
      if(not(converged))
      {
        dealii::Timer timer;
        timer.restart();

        if(k == 0)
        {
          omega = parameters.omega_init;
        }
        else
        {
          VectorType delta_r = r;
          delta_r.add(-1.0, r_old);
          omega *= -(r_old * delta_r) / delta_r.norm_sqr();
        }

        r_old = r;

        d.add(omega, r);
        structure->time_integrator->set_displacement(d);

        timer_tree.insert({"FSI", "Aitken"}, timer.wall_time());
      }

      // increment counter of partitioned iteration
      ++k;
    }
  }
  else if(parameters.method == "IQN-ILS")
  {
    std::shared_ptr<std::vector<VectorType>> D, R;
    D = std::make_shared<std::vector<VectorType>>();
    R = std::make_shared<std::vector<VectorType>>();

    VectorType d, d_tilde, d_tilde_old, r, r_old;
    structure->pde_operator->initialize_dof_vector(d);
    structure->pde_operator->initialize_dof_vector(d_tilde);
    structure->pde_operator->initialize_dof_vector(d_tilde_old);
    structure->pde_operator->initialize_dof_vector(r);
    structure->pde_operator->initialize_dof_vector(r_old);

    unsigned int const q = parameters.reused_time_steps;
    unsigned int const n = fluid->time_integrator->get_number_of_time_steps();

    bool converged = false;
    while(not(converged) and k < parameters.partitioned_iter_max)
    {
      print_solver_info_header(k);

      if(k == 0)
        structure->time_integrator->extrapolate_displacement_to_np(d);
      else
        d = structure->time_integrator->get_displacement_np();

      apply_dirichlet_neumann_scheme(d_tilde, d, k);

      // compute residual and check convergence
      r = d_tilde;
      r.add(-1.0, d);
      converged = check_convergence(r);

      // relaxation
      if(not(converged))
      {
        dealii::Timer timer;
        timer.restart();

        if(k == 0 and (q == 0 or n == 0))
        {
          d.add(parameters.omega_init, r);
        }
        else
        {
          if(k >= 1)
          {
            // append D, R matrices
            VectorType delta_d_tilde = d_tilde;
            delta_d_tilde.add(-1.0, d_tilde_old);
            D->push_back(delta_d_tilde);

            VectorType delta_r = r;
            delta_r.add(-1.0, r_old);
            R->push_back(delta_r);
          }

          // fill vectors (including reuse)
          std::vector<VectorType> Q = *R;
          for(auto R_q : R_history)
            for(auto delta_r : *R_q)
              Q.push_back(delta_r);
          std::vector<VectorType> D_all = *D;
          for(auto D_q : D_history)
            for(auto delta_d : *D_q)
              D_all.push_back(delta_d);

          AssertThrow(D_all.size() == Q.size(),
                      dealii::ExcMessage("D, Q vectors must have same size."));

          unsigned int const k_all = Q.size();
          if(k_all >= 1)
          {
            // compute QR-decomposition
            Matrix<Number> U(k_all);
            compute_QR_decomposition(Q, U);

            std::vector<Number> rhs(k_all, 0.0);
            for(unsigned int i = 0; i < k_all; ++i)
              rhs[i] = -Number(Q[i] * r);

            // alpha = U^{-1} rhs
            std::vector<Number> alpha(k_all, 0.0);
            backward_substitution(U, alpha, rhs);

            // d_{k+1} = d_tilde_{k} + delta d_tilde
            d = d_tilde;
            for(unsigned int i = 0; i < k_all; ++i)
              d.add(alpha[i], D_all[i]);
          }
          else // despite reuse, the vectors might be empty
          {
            d.add(parameters.omega_init, r);
          }
        }

        d_tilde_old = d_tilde;
        r_old       = r;

        structure->time_integrator->set_displacement(d);

        timer_tree.insert({"FSI", "IQN-ILS"}, timer.wall_time());
      }

      // increment counter of partitioned iteration
      ++k;
    }

    dealii::Timer timer;
    timer.restart();

    // Update history
    D_history.push_back(D);
    R_history.push_back(R);
    if(D_history.size() > q)
      D_history.erase(D_history.begin());
    if(R_history.size() > q)
      R_history.erase(R_history.begin());

    timer_tree.insert({"FSI", "IQN-ILS"}, timer.wall_time());
  }
  else if(parameters.method == "IQN-IMVLS")
  {
    std::shared_ptr<std::vector<VectorType>> D, R;
    D = std::make_shared<std::vector<VectorType>>();
    R = std::make_shared<std::vector<VectorType>>();

    std::vector<VectorType> B;

    VectorType d, d_tilde, d_tilde_old, r, r_old, b, b_old;
    structure->pde_operator->initialize_dof_vector(d);
    structure->pde_operator->initialize_dof_vector(d_tilde);
    structure->pde_operator->initialize_dof_vector(d_tilde_old);
    structure->pde_operator->initialize_dof_vector(r);
    structure->pde_operator->initialize_dof_vector(r_old);
    structure->pde_operator->initialize_dof_vector(b);
    structure->pde_operator->initialize_dof_vector(b_old);

    std::shared_ptr<Matrix<Number>> U;
    std::vector<VectorType>         Q;

    unsigned int const q = parameters.reused_time_steps;
    unsigned int const n = fluid->time_integrator->get_number_of_time_steps();

    bool converged = false;
    while(not converged and k < parameters.partitioned_iter_max)
    {
      print_solver_info_header(k);

      if(k == 0)
        structure->time_integrator->extrapolate_displacement_to_np(d);
      else
        d = structure->time_integrator->get_displacement_np();

      apply_dirichlet_neumann_scheme(d_tilde, d, k);

      // compute residual and check convergence
      r = d_tilde;
      r.add(-1.0, d);
      converged = check_convergence(r);

      // relaxation
      if(not(converged))
      {
        dealii::Timer timer;
        timer.restart();

        // compute b vector
        inv_jacobian_times_residual(b, D_history, R_history, Z_history, r);

        if(k == 0 and (q == 0 or n == 0))
        {
          d.add(parameters.omega_init, r);
        }
        else
        {
          d = d_tilde;
          d.add(-1.0, b);

          if(k >= 1)
          {
            // append D, R, B matrices
            VectorType delta_d_tilde = d_tilde;
            delta_d_tilde.add(-1.0, d_tilde_old);
            D->push_back(delta_d_tilde);

            VectorType delta_r = r;
            delta_r.add(-1.0, r_old);
            R->push_back(delta_r);

            VectorType delta_b = delta_d_tilde;
            delta_b.add(1.0, b_old);
            delta_b.add(-1.0, b);
            B.push_back(delta_b);

            // compute QR-decomposition
            U = std::make_shared<Matrix<Number>>(k);
            Q = *R;
            compute_QR_decomposition(Q, *U);

            std::vector<Number> rhs(k, 0.0);
            for(unsigned int i = 0; i < k; ++i)
              rhs[i] = -Number(Q[i] * r);

            // alpha = U^{-1} rhs
            std::vector<Number> alpha(k, 0.0);
            backward_substitution(*U, alpha, rhs);

            for(unsigned int i = 0; i < k; ++i)
              d.add(alpha[i], B[i]);
          }
        }

        d_tilde_old = d_tilde;
        r_old       = r;
        b_old       = b;

        structure->time_integrator->set_displacement(d);

        timer_tree.insert({"FSI", "IQN-IMVLS"}, timer.wall_time());
      }

      // increment counter of partitioned iteration
      ++k;
    }

    dealii::Timer timer;
    timer.restart();

    // Update history
    D_history.push_back(D);
    R_history.push_back(R);
    if(D_history.size() > q)
      D_history.erase(D_history.begin());
    if(R_history.size() > q)
      R_history.erase(R_history.begin());

    // compute Z and add to Z_history
    std::shared_ptr<std::vector<VectorType>> Z;
    Z  = std::make_shared<std::vector<VectorType>>();
    *Z = Q; // make sure that Z has correct size
    backward_substitution_multiple_rhs(*U, *Z, Q);
    Z_history.push_back(Z);
    if(Z_history.size() > q)
      Z_history.erase(Z_history.begin());

    timer_tree.insert({"FSI", "IQN-IMVLS"}, timer.wall_time());
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("This method is not implemented."));
  }

  return k;
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve() const
{
  set_start_time();

  synchronize_time_step_size();

  // compute initial acceleration for structural problem
  {
    // update stress boundary condition for solid at time t_n (not t_{n+1})
    coupling_fluid_to_structure(/* end_of_time_step = */ false);
    structure->time_integrator->compute_initial_acceleration(
      application->structure->get_parameters().restarted_simulation);
  }

  // The fluid domain is the master that dictates when the time loop is finished
  while(!fluid->time_integrator->finished())
  {
    // pre-solve
    fluid->time_integrator->advance_one_timestep_pre_solve(true);
    structure->time_integrator->advance_one_timestep_pre_solve(false);

    // solve (using strongly-coupled partitioned scheme)
    unsigned int iterations = solve_partitioned_problem();

    partitioned_iterations.first += 1;
    partitioned_iterations.second += iterations;

    print_solver_info_converged(iterations);

    // post-solve
    fluid->time_integrator->advance_one_timestep_post_solve();
    structure->time_integrator->advance_one_timestep_post_solve();

    if(application->fluid->get_parameters().adaptive_time_stepping)
      synchronize_time_step_size();
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_partitioned_iterations() const
{
  std::vector<std::string> names;
  std::vector<double>      iterations_avg;

  names = {"Partitioned iterations"};
  iterations_avg.resize(1);
  iterations_avg[0] =
    (double)partitioned_iterations.second / std::max(1.0, (double)partitioned_iterations.first);

  print_list_of_iterations(pcout, names, iterations_avg);
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_performance_results(double const total_time) const
{
  pcout << std::endl
        << "_________________________________________________________________________________"
        << std::endl
        << std::endl;

  pcout << "Performance results for fluid-structure interaction solver:" << std::endl;

  // iterations
  pcout << std::endl << "Average number of iterations:" << std::endl;

  pcout << std::endl << "FSI:" << std::endl;
  print_partitioned_iterations();

  pcout << std::endl << "Fluid:" << std::endl;
  fluid->time_integrator->print_iterations();

  pcout << std::endl << "ALE:" << std::endl;
  fluid->ale_grid_motion->print_iterations();

  pcout << std::endl << "Structure:" << std::endl;
  structure->time_integrator->print_iterations();

  // wall times
  pcout << std::endl << "Wall times:" << std::endl;

  timer_tree.insert({"FSI"}, total_time);

  timer_tree.insert({"FSI"}, fluid->time_integrator->get_timings(), "Fluid");
  timer_tree.insert({"FSI"}, fluid->get_timings_ale(), "Fluid-ALE");
  timer_tree.insert({"FSI"}, structure->time_integrator->get_timings(), "Structure");

  pcout << std::endl << "Timings for level 1:" << std::endl;
  timer_tree.print_level(pcout, 1);

  pcout << std::endl << "Timings for level 2:" << std::endl;
  timer_tree.print_level(pcout, 2);

  // Throughput in DoFs/s per time step per core
  dealii::types::global_dof_index DoFs =
    fluid->pde_operator->get_number_of_dofs() + structure->pde_operator->get_number_of_dofs();

  if(application->fluid->get_parameters().mesh_movement_type == IncNS::MeshMovementType::Poisson)
  {
    DoFs += fluid->pde_operator->get_number_of_dofs();
  }
  else if(application->fluid->get_parameters().mesh_movement_type ==
          IncNS::MeshMovementType::Elasticity)
  {
    DoFs += fluid->ale_elasticity_operator->get_number_of_dofs();
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("not implemented."));
  }

  unsigned int const N_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

  dealii::Utilities::MPI::MinMaxAvg total_time_data =
    dealii::Utilities::MPI::min_max_avg(total_time, mpi_comm);
  double const total_time_avg = total_time_data.avg;

  unsigned int N_time_steps = fluid->time_integrator->get_number_of_time_steps();

  print_throughput_unsteady(pcout, DoFs, total_time_avg, N_time_steps, N_mpi_processes);

  // computational costs in CPUh
  print_costs(pcout, total_time_avg, N_mpi_processes);

  pcout << "_________________________________________________________________________________"
        << std::endl
        << std::endl;
}

template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace FSI
} // namespace ExaDG
