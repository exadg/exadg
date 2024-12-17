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

#include <deal.II/numerics/vector_tools_mean_value.h>

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
  std::shared_ptr<HelpersALE<dim, Number> const>  helpers_ale_in,
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in,
  Parameters const &                              param_in,
  MPI_Comm const &                                mpi_comm_in,
  bool const                                      is_test_in)
  : Base(operator_in, helpers_ale_in, postprocessor_in, param_in, mpi_comm_in, is_test_in),
    pde_operator(operator_in),
    velocity(this->order),
    pressure(this->order),
    velocity_dbc(this->order),
    iterations_pressure({0, 0}),
    iterations_projection({0, 0}),
    iterations_viscous({0, {0, 0}}),
    iterations_penalty({0, 0}),
    iterations_mass({0, 0}),
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
  extra_pressure_nbc.update(this->get_time_step_number(),
                            this->adaptive_time_stepping,
                            this->get_time_step_vector());

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
  if(this->param.restarted_simulation == false)
  {
    initialize_velocity_dbc();
  }
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::read_restart_vectors(boost::archive::binary_iarchive & ia)
{
  Base::read_restart_vectors(ia);

  for(unsigned int i = 0; i < velocity_dbc.size(); i++)
  {
    ia >> velocity_dbc[i];
  }
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::write_restart_vectors(
  boost::archive::binary_oarchive & oa) const
{
  Base::write_restart_vectors(oa);

  for(unsigned int i = 0; i < velocity_dbc.size(); i++)
  {
    oa << velocity_dbc[i];
  }
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::allocate_vectors()
{
  Base::allocate_vectors();

  // velocity
  for(unsigned int i = 0; i < velocity.size(); ++i)
    pde_operator->initialize_vector_velocity(velocity[i]);
  pde_operator->initialize_vector_velocity(velocity_np);

  // pressure
  for(unsigned int i = 0; i < pressure.size(); ++i)
    pde_operator->initialize_vector_pressure(pressure[i]);
  pde_operator->initialize_vector_pressure(pressure_np);

  // velocity_dbc
  for(unsigned int i = 0; i < velocity_dbc.size(); ++i)
    pde_operator->initialize_vector_velocity(velocity_dbc[i]);
  pde_operator->initialize_vector_velocity(velocity_dbc_np);
}


template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::initialize_current_solution()
{
  if(this->param.ale_formulation)
    this->helpers_ale->move_grid(this->get_time());

  pde_operator->prescribe_initial_conditions(velocity[0], pressure[0], this->get_time());
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::initialize_former_multistep_dof_vectors()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i = 1; i < velocity.size(); ++i)
  {
    if(this->param.ale_formulation)
      this->helpers_ale->move_grid(this->get_previous_time(i));

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
  {
    this->helpers_ale->move_grid(this->get_time());
    this->helpers_ale->update_pde_operator_after_grid_motion();
  }
  pde_operator->interpolate_velocity_dirichlet_bc(velocity_dbc[0], this->get_time());
  // ... and previous times if start_with_low_order == false
  if(this->start_with_low_order == false)
  {
    for(unsigned int i = 1; i < velocity_dbc.size(); ++i)
    {
      double const time = this->get_time() - double(i) * this->get_time_step_size();
      if(this->param.ale_formulation)
      {
        this->helpers_ale->move_grid(time);
        this->helpers_ale->update_pde_operator_after_grid_motion();
      }
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
TimeIntBDFDualSplitting<dim, Number>::get_pressure() const
{
  return pressure[0];
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

  AssertThrow(velocity[0].l2_norm() < 1.e-15 and pressure[0].l2_norm() < 1.e-15,
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
  if(this->param.convective_problem())
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
  unsigned int const n_iter_mass =
    pde_operator->apply_inverse_mass_operator(velocity_np, velocity_np);
  iterations_mass.first += 1;
  iterations_mass.second += n_iter_mass;

  // calculate sum (alpha_i/dt * u_i) and add to velocity_np
  for(unsigned int i = 0; i < velocity.size(); ++i)
  {
    velocity_np.add(this->bdf.get_alpha(i) / this->get_time_step_size(), velocity[i]);
  }

  // solve discrete temporal derivative term for intermediate velocity u_hat
  velocity_np *= this->get_time_step_size() / this->bdf.get_gamma0();

  if(this->print_solver_info() and not(this->is_test))
  {
    if(this->param.spatial_discretization == SpatialDiscretization::HDIV)
    {
      this->pcout << std::endl << "Convective step:";
      print_solver_info_linear(this->pcout, n_iter_mass, timer.wall_time());
    }
    else if(this->param.spatial_discretization == SpatialDiscretization::L2)
    {
      this->pcout << std::endl << "Explicit convective step:";
      print_wall_time(this->pcout, timer.wall_time());
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }

  this->timer_tree->insert({"Timeloop", "Convective step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::evaluate_convective_term()
{
  dealii::Timer timer;
  timer.restart();

  if(this->param.convective_problem())
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

template<typename Number>
void
extrapolate_vectors(std::vector<Number> const &                                             factors,
                    std::vector<dealii::LinearAlgebra::distributed::Vector<Number>> const & vectors,
                    dealii::LinearAlgebra::distributed::Vector<Number> &                    result)
{
  unsigned int const locally_owned_size = result.locally_owned_size();
  if(factors.size() == 1)
    result.equ(factors[0], vectors[0]);
  else if(factors.size() == 2)
  {
    Number const * vec_0  = vectors[0].begin();
    Number const * vec_1  = vectors[1].begin();
    Number *       res    = result.begin();
    Number const   beta_0 = factors[0];
    Number const   beta_1 = factors[1];

    DEAL_II_OPENMP_SIMD_PRAGMA
    for(unsigned int i = 0; i < locally_owned_size; ++i)
      res[i] = beta_0 * vec_0[i] + beta_1 * vec_1[i];
  }
  else if(factors.size() == 3)
  {
    Number const * vec_0  = vectors[0].begin();
    Number const * vec_1  = vectors[1].begin();
    Number const * vec_2  = vectors[2].begin();
    Number *       res    = result.begin();
    Number const   beta_0 = factors[0];
    Number const   beta_1 = factors[1];
    Number const   beta_2 = factors[2];

    DEAL_II_OPENMP_SIMD_PRAGMA
    for(unsigned int i = 0; i < locally_owned_size; ++i)
      res[i] = beta_0 * vec_0[i] + beta_1 * vec_1[i] + beta_2 * vec_2[i];
  }
  else
    for(unsigned int i = 0; i < locally_owned_size; ++i)
    {
      Number entry = factors[0] * vectors[0].local_element(i);
      for(unsigned int j = 1; j < factors.size(); ++j)
        entry += factors[j] * vectors[j].local_element(i);
      result.local_element(i) = entry;
    }
}



template<typename OperatorType, typename VectorType>
void
compute_least_squares_fit(OperatorType const &            op,
                          std::vector<VectorType> const & vectors,
                          VectorType const &              rhs,
                          VectorType &                    result)
{
  using Number = typename VectorType::value_type;
  std::vector<VectorType>    tmp(vectors.size());
  dealii::FullMatrix<double> matrix(vectors.size(), vectors.size());
  std::vector<Number>        small_vector(vectors.size());

  // This algorithm performs a Cholesky (LDLT) factorization of the normal
  // equations for the minimization problem
  // min_{alpha_i} | sum(alpha_i A x_i) - b |
  // which eventually gives the linear combination sum (alpha_i x_i)
  // minimizing the residual among the given search vectors
  unsigned int i = 0;
  for(; i < vectors.size(); ++i)
  {
    tmp[i].reinit(vectors[0], true);
    op.vmult(tmp[i], vectors[i]);

    std::array<Number *, 11> vec_ptrs = {};
    for(unsigned int j = 0; j <= i; ++j)
      vec_ptrs[j] = tmp[j].begin();
    Number const * rhs_ptr = rhs.begin();

    unsigned int constexpr n_lanes    = dealii::VectorizedArray<Number>::size();
    unsigned int constexpr n_lanes_4  = 4 * n_lanes;
    unsigned int const regular_size_4 = (vectors[0].locally_owned_size()) / n_lanes_4 * n_lanes_4;
    unsigned int const regular_size   = (vectors[0].locally_owned_size()) / n_lanes * n_lanes;

    // compute inner products in normal equations (all at once)
    std::array<dealii::VectorizedArray<Number>, 12> local_sums = {};

    unsigned int k = 0;
    for(; k < regular_size_4; k += n_lanes_4)
    {
      dealii::VectorizedArray<Number> v_k_0, v_k_1, v_k_2, v_k_3;
      v_k_0.load(vec_ptrs[i] + k);
      v_k_1.load(vec_ptrs[i] + k + n_lanes);
      v_k_2.load(vec_ptrs[i] + k + 2 * n_lanes);
      v_k_3.load(vec_ptrs[i] + k + 3 * n_lanes);
      for(unsigned int j = 0; j < i; ++j)
      {
        dealii::VectorizedArray<Number> v_j_k, tmp0;
        v_j_k.load(vec_ptrs[j] + k);
        tmp0 = v_k_0 * v_j_k;
        v_j_k.load(vec_ptrs[j] + k + n_lanes);
        tmp0 += v_k_1 * v_j_k;
        v_j_k.load(vec_ptrs[j] + k + 2 * n_lanes);
        tmp0 += v_k_2 * v_j_k;
        v_j_k.load(vec_ptrs[j] + k + 3 * n_lanes);
        tmp0 += v_k_3 * v_j_k;
        local_sums[j] += tmp0;
      }
      local_sums[i] += v_k_0 * v_k_0 + v_k_1 * v_k_1 + v_k_2 * v_k_2 + v_k_3 * v_k_3;

      dealii::VectorizedArray<Number> rhs_k, tmp0;
      rhs_k.load(rhs_ptr + k);
      tmp0 = rhs_k * v_k_0;
      rhs_k.load(rhs_ptr + k + n_lanes);
      tmp0 += rhs_k * v_k_1;
      rhs_k.load(rhs_ptr + k + 2 * n_lanes);
      tmp0 += rhs_k * v_k_2;
      rhs_k.load(rhs_ptr + k + 3 * n_lanes);
      tmp0 += rhs_k * v_k_3;
      local_sums[i + 1] += tmp0;
    }
    for(; k < regular_size; k += n_lanes)
    {
      dealii::VectorizedArray<Number> v_k;
      v_k.load(vec_ptrs[i] + k);
      for(unsigned int j = 0; j < i; ++j)
      {
        dealii::VectorizedArray<Number> v_j_k;
        v_j_k.load(vec_ptrs[j] + k);
        local_sums[j] += v_k * v_j_k;
      }
      local_sums[i] += v_k * v_k;
      dealii::VectorizedArray<Number> rhs_k;
      rhs_k.load(rhs_ptr + k);
      local_sums[i + 1] += v_k * rhs_k;
    }
    for(; k < vectors[0].locally_owned_size(); ++k)
    {
      for(unsigned int j = 0; j <= i; ++j)
        local_sums[j][k - regular_size] += vec_ptrs[i][k] * vec_ptrs[j][k];
      local_sums[i + 1][k - regular_size] += vec_ptrs[i][k] * rhs_ptr[k];
    }
    std::array<Number, 12> scalar_sums;
    for(unsigned int j = 0; j < i + 2; ++j)
      scalar_sums[j] = local_sums[j].sum();

    dealii::Utilities::MPI::sum(dealii::ArrayView<Number const>(scalar_sums.data(), i + 2),
                                vectors[0].get_mpi_communicator(),
                                dealii::ArrayView<Number>(scalar_sums.data(), i + 2));

    for(unsigned int j = 0; j <= i; ++j)
      matrix(i, j) = scalar_sums[j];

    // update row in Cholesky factorization associated to matrix of normal
    // equations using the diagonal entry D
    for(unsigned int j = 0; j < i; ++j)
    {
      double const inv_entry = matrix(i, j) / matrix(j, j);
      for(unsigned int k = j + 1; k <= i; ++k)
        matrix(i, k) -= matrix(k, j) * inv_entry;
    }
    if(matrix(i, i) < 1e-12 * matrix(0, 0) or matrix(0, 0) < 1e-30)
      break;

    // update for the right hand side (forward substitution)
    small_vector[i] = scalar_sums[i + 1];
    for(unsigned int j = 0; j < i; ++j)
      small_vector[i] -= matrix(i, j) / matrix(j, j) * small_vector[j];
  }

  // backward substitution of Cholesky factorization
  for(unsigned int s = i; s < small_vector.size(); ++s)
    small_vector[s] = 0.;
  for(int s = i - 1; s >= 0; --s)
  {
    double sum = small_vector[s];
    for(unsigned int j = s + 1; j < i; ++j)
      sum -= small_vector[j] * matrix(j, s);
    small_vector[s] = sum / matrix(s, s);
  }
  if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << "e ";
    for(const double a : small_vector)
      std::cout << a << " ";
    if (i > 0)
      std::cout << "i=" << i << " " << matrix(i - 1, i - 1) / matrix(0, 0) << "   ";
  }
  extrapolate_vectors(small_vector, vectors, result);
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
    compute_least_squares_fit(pde_operator->laplace_operator, pressure, rhs, pressure_np);
  }
  else
  {
    pressure_np = pressure_last_iter;
  }

  // solve linear system of equations
  bool const update_preconditioner =
    this->param.update_preconditioner_pressure_poisson and
    ((this->time_step_number - 1) %
       this->param.update_preconditioner_pressure_poisson_every_time_steps ==
     0);

  const double rhs_norm = rhs.l2_norm();
  VectorType tmp, tmp2;
  tmp.reinit(pressure_np, true);
  tmp2.reinit(pressure_np, true);
  pde_operator->laplace_operator.vmult(tmp, pressure_np);
  const double res_norm = std::sqrt(tmp.add_and_dot(-1.0, rhs, tmp));
  if (pressure.size() > 0)
   pde_operator->laplace_operator.vmult(tmp, pressure[0]);
  const double res2_norm = std::sqrt(tmp.add_and_dot(-1.0, rhs, tmp));

  unsigned int const n_iter = pde_operator->solve_pressure(pressure_np, rhs, update_preconditioner);
  iterations_pressure.first += 1;
  iterations_pressure.second += n_iter;

  pde_operator->laplace_operator.vmult(tmp, pressure_np);
  const double res_norm4 = std::sqrt(tmp.add_and_dot(-1.0, rhs, tmp));
  this->pcout << "Residual norms pressure:   " << rhs_norm << " " << res_norm << " " << res2_norm
              << " " << res_norm4 << " (" << n_iter << ")" << std::endl;

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
  if(this->param.divu_integrated_by_parts == true and this->param.divu_use_boundary_data == true)
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
    dealii::VectorTools::subtract_mean_value(rhs);
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
  unsigned int const n_iter_mass = pde_operator->apply_inverse_mass_operator(velocity_np, rhs);
  iterations_mass.first += 1;
  iterations_mass.second += n_iter_mass;

  // penalty terms
  if(this->param.apply_penalty_terms_in_postprocessing_step == false and
     (this->param.use_divergence_penalty == true or this->param.use_continuity_penalty == true))
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
      this->param.update_preconditioner_projection and
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
      if(this->param.spatial_discretization == SpatialDiscretization::HDIV)
      {
        this->pcout << std::endl << "Projection step:";
        print_solver_info_linear(this->pcout, n_iter_mass, timer.wall_time());
      }
      else if(this->param.spatial_discretization == SpatialDiscretization::L2)
      {
        this->pcout << std::endl << "Explicit projection step:";
        print_wall_time(this->pcout, timer.wall_time());
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("Not implemented."));
      }
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

  // in case we need to iteratively solve a linear or nonlinear system of equations
  if(this->param.viscous_problem() or this->param.non_explicit_convective_problem())
  {
    // if a variable viscosity is used: update
    // viscosity model before calculating rhs_viscous
    if(this->param.viscosity_is_variable())
    {
      dealii::Timer timer_viscosity_update;
      timer_viscosity_update.restart();

      // extrapolate velocity to time t_n+1 and use this velocity field to
      // update the viscosity model (to recalculate the variable viscosity)
      VectorType velocity_extrapolated;
      velocity_extrapolated.reinit(velocity_np, true);
      std::vector<Number> beta(velocity.size());
      for(unsigned int i = 0; i < velocity.size(); ++i)
        beta[i] = this->extra.get_beta(i);
      extrapolate_vectors(beta, velocity, velocity_extrapolated);

      pde_operator->update_viscosity(velocity_extrapolated);

      if(this->print_solver_info() and not(this->is_test))
      {
        this->pcout << std::endl << "Update of variable viscosity:";
        print_wall_time(this->pcout, timer_viscosity_update.wall_time());
      }
    }

    // Extrapolate old solution to get a good initial estimate for the solver.
    if(this->use_extrapolation)
    {
      std::vector<Number> beta(velocity.size());
      for(unsigned int i = 0; i < velocity.size(); ++i)
        beta[i] = this->extra.get_beta(i);
      extrapolate_vectors(beta, velocity, velocity_np);
    }
    else
    {
      velocity_np = velocity_viscous_last_iter;
    }

    /*
     *  update variable viscosity
     */
    if(this->param.viscous_problem() and this->param.viscosity_is_variable() and
       this->param.treatment_of_variable_viscosity == TreatmentOfVariableViscosity::Explicit)
    {
      dealii::Timer timer_viscosity_update;
      timer_viscosity_update.restart();

      pde_operator->update_viscosity(velocity_np);

      if(this->print_solver_info() and not(this->is_test))
      {
        this->pcout << std::endl << "Update of variable viscosity:";
        print_wall_time(this->pcout, timer_viscosity_update.wall_time());
      }
    }

    bool const update_preconditioner =
      this->param.update_preconditioner_momentum and
      ((this->time_step_number - 1) % this->param.update_preconditioner_momentum_every_time_steps ==
       0);

    if(this->param.nonlinear_problem_has_to_be_solved())
    {
      /*
       *  Calculate the vector that is constant when solving the nonlinear momentum equation
       *  (where constant means that the vector does not change from one Newton iteration
       *  to the next, i.e., it does not depend on the current solution of the nonlinear solver)
       */
      VectorType rhs;
      rhs.reinit(velocity_np, true);
      VectorType transport_velocity_dummy;
      rhs_viscous(rhs, transport_velocity_dummy, transport_velocity_dummy);

      // solve non-linear system of equations
      auto const iter = pde_operator->solve_nonlinear_momentum_equation(
        velocity_np,
        rhs,
        this->get_next_time(),
        update_preconditioner,
        this->get_scaling_factor_time_derivative_term());

      iterations_viscous.first += 1;
      std::get<0>(iterations_viscous.second) += std::get<0>(iter);
      std::get<1>(iterations_viscous.second) += std::get<1>(iter);

      if(this->print_solver_info() and not(this->is_test))
      {
        this->pcout << std::endl << "Solve momentum step:";
        print_solver_info_nonlinear(this->pcout,
                                    std::get<0>(iter),
                                    std::get<1>(iter),
                                    timer.wall_time());
      }
    }
    else // linear problem
    {
      // linearly implicit convective term: use extrapolated/stored velocity as transport velocity
      VectorType transport_velocity;
      if(this->param.convective_problem() and
         this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::LinearlyImplicit)
      {
        transport_velocity = velocity_np;
      }

      /*
       *  Calculate the right-hand side of the linear system of equations.
       */
      VectorType rhs;
      rhs.reinit(velocity_np, true);
      rhs_viscous(rhs, transport_velocity, transport_velocity);

      // solve linear system of equations
      unsigned int const n_iter = pde_operator->solve_linear_momentum_equation(
        velocity_np,
        rhs,
        transport_velocity,
        update_preconditioner,
        this->get_scaling_factor_time_derivative_term());
      iterations_viscous.first += 1;
      std::get<1>(iterations_viscous.second) += n_iter;

      if(this->print_solver_info() and not(this->is_test))
      {
        this->pcout << std::endl << "Solve viscous step:";
        print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
      }

    if(this->store_solution)
      velocity_viscous_last_iter = velocity_np;
    // const double rhs_norm = rhs.l2_norm();
    // VectorType tmp, tmp2;
    // tmp.reinit(velocity_np, true);
    // tmp2.reinit(velocity_np, true);
    // pde_operator->momentum_operator.vmult(tmp, velocity_np);
    // const double res_norm = std::sqrt(tmp.add_and_dot(-1.0, rhs, tmp));
    // if (velocity.size() > 0)
    //  pde_operator->momentum_operator.vmult(tmp, velocity[0]);
    // const double res2_norm = std::sqrt(tmp.add_and_dot(-1.0, rhs, tmp));
    // compute_least_squares_fit(pde_operator->momentum_operator, velocity, rhs, tmp2);
    // pde_operator->momentum_operator.vmult(tmp, tmp2);
    // const double res_norm3 = std::sqrt(tmp.add_and_dot(-1.0, rhs, tmp));
    // if (velocity_viscous_last_iter.size() > 0)
    //  pde_operator->momentum_operator.vmult(tmp, velocity_viscous_last_iter);
    // const double res_norm5 = std::sqrt(tmp.add_and_dot(-1.0, rhs, tmp));
    // this->pcout << std::setprecision(6) << "Residual norms momentum:   " << rhs_norm << " " <<
    // res_norm << " " << res2_norm << " " << res_norm3 << " " << res_norm5;

    //unsigned int const n_iter = pde_operator->solve_viscous(
    //  velocity_np, rhs, update_preconditioner, this->get_scaling_factor_time_derivative_term());
    // pde_operator->momentum_operator.vmult(tmp, velocity_np);
    // const double res_norm4 = std::sqrt(tmp.add_and_dot(-1.0, rhs, tmp));
    // this->pcout << " " << res_norm4 << " (" << n_iter << ")" << std::endl;
    //iterations_viscous.first += 1;
    //iterations_viscous.second += n_iter;

    //velocity_viscous_last_iter = velocity_np;

    // write output
    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl << "Solve viscous step:";
      print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
    }
    }
  }
  else // no viscous term and no (linearly) implicit convective term, i.e. there is nothing to do in
       // this step of the dual splitting scheme
  {
    // nothing to do
    AssertThrow(this->param.equation_type == EquationType::Euler and
                  this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit,
                dealii::ExcMessage("Logical error."));
  }

  this->timer_tree->insert({"Timeloop", "Viscous step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::rhs_viscous(VectorType &       rhs,
                                                  VectorType const & velocity_mass_operator,
                                                  VectorType const & transport_velocity) const
{
  /*
   *  apply mass operator
   */
  pde_operator->apply_scaled_mass_operator(rhs,
                                           this->bdf.get_gamma0() / this->get_time_step_size(),
                                           velocity_np);

  // compensate for explicit convective term taken into account in the first sub-step of the
  // dual-splitting scheme
  if(this->param.non_explicit_convective_problem())
  {
    for(unsigned int i = 0; i < this->vec_convective_term.size(); ++i)
      rhs.add(this->extra.get_beta(i), this->vec_convective_term[i]);
  }

  if(this->param.nonlinear_problem_has_to_be_solved())
  {
    // for a nonlinear problem, inhomogeneous contributions are taken into account when evaluating
    // the nonlinear residual
  }
  else // linear problem
  {
    // compute inhomogeneous contributions of linearly implicit convective term
    if(this->param.convective_problem() and
       this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::LinearlyImplicit)
    {
      pde_operator->rhs_add_convective_term(rhs, transport_velocity, this->get_next_time());
    }

    // inhomogeneous parts of boundary face integrals of viscous operator
    pde_operator->rhs_add_viscous_term(rhs, this->get_next_time());
  }
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::penalty_step()
{
  if(this->param.use_divergence_penalty == true or this->param.use_continuity_penalty == true)
  {
    dealii::Timer timer;
    timer.restart();

    // compute right-hand-side vector
    VectorType rhs;
    rhs.reinit(velocity_np, true);
    pde_operator->apply_mass_operator(rhs, velocity_np);

    // extrapolate velocity to time t_n+1 and use this velocity field to
    // calculate the penalty parameter for the divergence and continuity penalty term
    VectorType velocity_extrapolated;
    velocity_extrapolated.reinit(velocity_np, true);
    std::vector<Number> beta(velocity.size());
    for(unsigned int i = 0; i < velocity.size(); ++i)
      beta[i] = this->extra.get_beta(i);
    extrapolate_vectors(beta, velocity, velocity_extrapolated);

    pde_operator->update_projection_operator(velocity_extrapolated, this->get_time_step_size());

    // right-hand side term: add inhomogeneous contributions of continuity penalty operator to
    // rhs-vector if desired
    if(this->param.use_continuity_penalty and this->param.continuity_penalty_use_boundary_data)
      pde_operator->rhs_add_projection_operator(rhs, this->get_next_time());

    // solve linear system of equations
    bool const update_preconditioner =
      this->param.update_preconditioner_projection and
      ((this->time_step_number - 1) %
         this->param.update_preconditioner_projection_every_time_steps ==
       0);

    if(this->use_extrapolation == false)
      velocity_np = velocity_projection_last_iter;
    else
      compute_least_squares_fit(*pde_operator->projection_operator, velocity, rhs, velocity_np);

    VectorType tmp;
    tmp.reinit(velocity_np, true);
    pde_operator->projection_operator->vmult(tmp, velocity_np);
    const double res_norm3 = std::sqrt(tmp.add_and_dot(-1.0, rhs, tmp));
    this->pcout << "Residual norms projection: " << rhs.l2_norm() << " " << res_norm3;

    unsigned int const n_iter =
      pde_operator->solve_projection(velocity_np, rhs, update_preconditioner);
    pde_operator->projection_operator->vmult(tmp, velocity_np);
    const double res_norm4 = std::sqrt(tmp.add_and_dot(-1.0, rhs, tmp));
    this->pcout << " " << res_norm4 << " (" << n_iter << ")" << std::endl;

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

  push_back(velocity);
  velocity[0].swap(velocity_np);

  push_back(pressure);
  pressure[0].swap(pressure_np);

  // We also have to care about the history of velocity Dirichlet boundary conditions.
  // Note that velocity_dbc_np has already been updated.
  push_back(velocity_dbc);
  velocity_dbc[0].swap(velocity_dbc_np);
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

    while(not(converged) and this->time < (this->end_time - this->eps) and
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
      if(incr < this->param.abs_tol_steady or incr_rel < this->param.rel_tol_steady)
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
  std::vector<std::string> names;
  std::vector<double>      iterations_avg;

  if(this->param.nonlinear_problem_has_to_be_solved())
  {
    names = {"Convective step",
             "Pressure step",
             "Projection step",
             "Viscous step (nonlinear)",
             "Viscous step (accumulated)",
             "Viscous step (linear per nonlinear)"};

    iterations_avg.resize(6);
    iterations_avg[0] = 0.0; // explicit convective step
    iterations_avg[1] =
      (double)iterations_pressure.second / std::max(1., (double)iterations_pressure.first);
    iterations_avg[2] =
      (double)iterations_projection.second / std::max(1., (double)iterations_projection.first);
    iterations_avg[3] = (double)std::get<0>(iterations_viscous.second) /
                        std::max(1., (double)iterations_viscous.first);
    iterations_avg[4] = (double)std::get<1>(iterations_viscous.second) /
                        std::max(1., (double)iterations_viscous.first);

    if(iterations_avg[3] > std::numeric_limits<double>::min())
      iterations_avg[5] = iterations_avg[4] / iterations_avg[3];
    else
      iterations_avg[5] = iterations_avg[4];
  }
  else
  {
    names = {"Convective step", "Pressure step", "Projection step", "Viscous step"};

    iterations_avg.resize(4);
    iterations_avg[0] = 0.0; // explicit convective step
    iterations_avg[1] =
      (double)iterations_pressure.second / std::max(1., (double)iterations_pressure.first);
    iterations_avg[2] =
      (double)iterations_projection.second / std::max(1., (double)iterations_projection.first);
    iterations_avg[3] = (double)std::get<1>(iterations_viscous.second) /
                        std::max(1., (double)iterations_viscous.first);
  }

  if(this->param.spatial_discretization == SpatialDiscretization::HDIV)
  {
    names.push_back("Mass solver");
    iterations_avg.push_back(iterations_mass.second / std::max(1., (double)iterations_mass.first));
  }

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
