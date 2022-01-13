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

#ifndef INCLUDE_CONVECTION_DIFFUSION_TIME_INT_BDF_H_
#define INCLUDE_CONVECTION_DIFFUSION_TIME_INT_BDF_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/time_integration/explicit_runge_kutta.h>
#include <exadg/time_integration/time_int_bdf_base.h>

namespace ExaDG
{
using namespace dealii;

namespace ConvDiff
{
class Parameters;

template<int dim, typename Number>
class Operator;

template<typename Number>
class PostProcessorInterface;

template<typename Number>
class OperatorOIF;
} // namespace ConvDiff


namespace ConvDiff
{
template<int dim, typename Number>
class TimeIntBDF : public TimeIntBDFBase<Number>
{
public:
  typedef typename TimeIntBDFBase<Number>::VectorType VectorType;

  TimeIntBDF(std::shared_ptr<Operator<dim, Number>>          operator_in,
             Parameters const &                              param_in,
             MPI_Comm const &                                mpi_comm_in,
             bool const                                      is_test_in,
             std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in);

  void
  set_velocities_and_times(std::vector<VectorType const *> const & velocities_in,
                           std::vector<double> const &             times_in);

  void
  extrapolate_solution(VectorType & vector);

  void
  ale_update();

  void
  print_iterations() const;

private:
  void
  allocate_vectors();

  void
  initialize_current_solution();

  void
  initialize_former_solutions();

  void
  initialize_vec_convective_term();

  double
  calculate_time_step_size();

  double
  recalculate_time_step_size() const;

  void
  prepare_vectors_for_next_timestep();

  void
  solve_timestep();

  void
  initialize_oif();

  void
  setup_derived();

  void
  calculate_sum_alphai_ui_oif_substepping(VectorType & sum_alphai_ui,
                                          double const cfl,
                                          double const cfl_oif);

  void
  initialize_solution_oif_substepping(VectorType & solution_tilde_m, unsigned int i);

  void
  update_sum_alphai_ui_oif_substepping(VectorType &       sum_alphai_ui,
                                       VectorType const & u_tilde_i,
                                       unsigned int       i);

  void
  do_timestep_oif_substepping(VectorType & solution_tilde_mp,
                              VectorType & solution_tilde_m,
                              double const start_time,
                              double const time_step_size);

  bool
  print_solver_info() const;

  void
  read_restart_vectors(boost::archive::binary_iarchive & ia);

  void
  write_restart_vectors(boost::archive::binary_oarchive & oa) const;

  void
  postprocessing() const;

  std::shared_ptr<Operator<dim, Number>> pde_operator;

  Parameters const & param;

  unsigned int const refine_steps_time;

  double const cfl;

  // solution vectors
  VectorType              solution_np;
  std::vector<VectorType> solution;
  std::vector<VectorType> vec_convective_term;
  VectorType              convective_term_np;

  VectorType rhs_vector;

  // numerical velocity field
  std::vector<VectorType const *> velocities;
  std::vector<double>             times;

  // iteration counts
  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */> iterations;

  // Operator-integration-factor (OIF) splitting

  // cfl number for OIF splitting
  double const cfl_oif;

  std::shared_ptr<OperatorOIF<Number>> convective_operator_OIF;

  std::shared_ptr<ExplicitTimeIntegrator<OperatorOIF<Number>, VectorType>> time_integrator_OIF;

  std::shared_ptr<PostProcessorInterface<Number>> postprocessor;

  // ALE
  VectorType              grid_velocity;
  std::vector<VectorType> vec_grid_coordinates;
  VectorType              grid_coordinates_np;
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_CONVECTION_DIFFUSION_TIME_INT_BDF_H_ */
