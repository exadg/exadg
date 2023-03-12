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


#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARTITIONED_SOLVER_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARTITIONED_SOLVER_H_

// FSI
#include <exadg/fluid_structure_interaction/acceleration_schemes/linear_algebra.h>
#include <exadg/fluid_structure_interaction/acceleration_schemes/parameters.h>
#include <exadg/fluid_structure_interaction/single_field_solvers/fluid.h>
#include <exadg/fluid_structure_interaction/single_field_solvers/structure.h>

// utilities
#include <exadg/utilities/print_solver_results.h>
#include <exadg/utilities/timer_tree.h>

namespace ExaDG
{
namespace FSI
{
template<int dim, typename Number>
class PartitionedSolver
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  PartitionedSolver(Parameters const & parameters, MPI_Comm const & comm);

  void
  setup(std::shared_ptr<SolverFluid<dim, Number>>     fluid_,
        std::shared_ptr<SolverStructure<dim, Number>> structure_);

  void
  solve(std::function<void(VectorType &, VectorType const &, unsigned int)> const &
          apply_dirichlet_neumann_scheme);

  void
  print_iterations(dealii::ConditionalOStream const & pcout) const;

  std::shared_ptr<TimerTree>
  get_timings() const;

private:
  bool
  check_convergence(VectorType const & residual) const;

  void
  print_solver_info_header(unsigned int const iteration) const;

  void
  print_solver_info_converged(unsigned int const iteration) const;

  Parameters parameters;

  // output to std::cout
  dealii::ConditionalOStream pcout;

  std::shared_ptr<SolverFluid<dim, Number>>     fluid;
  std::shared_ptr<SolverStructure<dim, Number>> structure;

  // required for quasi-Newton methods
  std::vector<std::shared_ptr<std::vector<VectorType>>> D_history, R_history, Z_history;

  // Computation time (wall clock time).
  std::shared_ptr<TimerTree> timer_tree;

  /*
   * The first number counts the number of time steps, the second number the total number
   * (accumulated over all time steps) of iterations of the partitioned FSI scheme.
   */
  std::pair<unsigned int, unsigned long long> partitioned_iterations;
};

template<int dim, typename Number>
PartitionedSolver<dim, Number>::PartitionedSolver(Parameters const & parameters,
                                                  MPI_Comm const &   comm)
  : parameters(parameters),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0),
    partitioned_iterations({0, 0})
{
  timer_tree = std::make_shared<TimerTree>();
}

template<int dim, typename Number>
void
PartitionedSolver<dim, Number>::setup(std::shared_ptr<SolverFluid<dim, Number>>     fluid_,
                                      std::shared_ptr<SolverStructure<dim, Number>> structure_)
{
  fluid     = fluid_;
  structure = structure_;
}

template<int dim, typename Number>
bool
PartitionedSolver<dim, Number>::check_convergence(VectorType const & residual) const
{
  double const residual_norm = residual.l2_norm();
  double const ref_norm_abs  = std::sqrt(structure->pde_operator->get_number_of_dofs());
  double const ref_norm_rel  = structure->time_integrator->get_velocity_np().l2_norm() *
                              structure->time_integrator->get_time_step_size();

  bool const converged = (residual_norm < parameters.abs_tol * ref_norm_abs) or
                         (residual_norm < parameters.rel_tol * ref_norm_rel);

  return converged;
}

template<int dim, typename Number>
void
PartitionedSolver<dim, Number>::print_solver_info_header(unsigned int const iteration) const
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
PartitionedSolver<dim, Number>::print_solver_info_converged(unsigned int const iteration) const
{
  if(fluid->time_integrator->print_solver_info())
  {
    pcout << std::endl
          << "Partitioned FSI iteration converged in " << iteration << " iterations." << std::endl;
  }
}

template<int dim, typename Number>
void
PartitionedSolver<dim, Number>::print_iterations(dealii::ConditionalOStream const & pcout) const
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
std::shared_ptr<TimerTree>
PartitionedSolver<dim, Number>::get_timings() const
{
  return timer_tree;
}

template<int dim, typename Number>
void
PartitionedSolver<dim, Number>::solve(
  std::function<void(VectorType &, VectorType const &, unsigned int)> const &
    apply_dirichlet_neumann_scheme)
{
  // iteration counter
  unsigned int k = 0;

  // fixed-point iteration with dynamic relaxation (Aitken relaxation)
  if(parameters.method == AccelerationMethod::Aitken)
  {
    VectorType r_old, d;
    structure->pde_operator->initialize_dof_vector(r_old);
    structure->pde_operator->initialize_dof_vector(d);

    bool   converged = false;
    double omega     = 1.0;
    while(not(converged) and k < parameters.partitioned_iter_max)
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

        timer_tree->insert({"Aitken"}, timer.wall_time());
      }

      // increment counter of partitioned iteration
      ++k;
    }
  }
  else if(parameters.method == AccelerationMethod::IQN_ILS)
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

        timer_tree->insert({"IQN-ILS"}, timer.wall_time());
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

    timer_tree->insert({"IQN-ILS"}, timer.wall_time());
  }
  else if(parameters.method == AccelerationMethod::IQN_IMVLS)
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

        timer_tree->insert({"IQN-IMVLS"}, timer.wall_time());
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

    timer_tree->insert({"IQN-IMVLS"}, timer.wall_time());
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("This method is not implemented."));
  }

  partitioned_iterations.first += 1;
  partitioned_iterations.second += k;

  print_solver_info_converged(k);
}

} // namespace FSI
} // namespace ExaDG

#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARTITIONED_SOLVER_H_ */
