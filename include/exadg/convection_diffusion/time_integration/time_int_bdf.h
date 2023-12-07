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
#include <exadg/time_integration/lambda_functions_ale.h>
#include <exadg/time_integration/time_int_bdf_base.h>

namespace ExaDG
{
namespace ConvDiff
{
class Parameters;

template<int dim, typename Number>
class Operator;

template<typename Number>
class PostProcessorInterface;
} // namespace ConvDiff


namespace ConvDiff
{
template<int dim, typename Number>
class TimeIntBDF : public TimeIntBDFBase
{
public:
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  TimeIntBDF(std::shared_ptr<Operator<dim, Number>>          operator_in,
             std::shared_ptr<HelpersALE<Number> const>       helpers_ale_in,
             std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in,
             Parameters const &                              param_in,
             MPI_Comm const &                                mpi_comm_in,
             bool const                                      is_test_in);

  void
  set_velocities_and_times(std::vector<VectorType const *> const & velocities_in,
                           std::vector<double> const &             times_in);

  void
  extrapolate_solution(VectorType & vector);

  VectorType const &
  get_solution_np() const;

  void
  ale_update();

  void
  print_iterations() const;

  void
  prepare_coarsening_and_refinement() final;

  void
  interpolate_after_coarsening_and_refinement() final;

private:
  void
  allocate_vectors() final;

  std::shared_ptr<std::vector<VectorType *>>
  get_vectors();

  void
  initialize_current_solution() final;

  void
  initialize_former_multistep_dof_vectors() final;

  void
  initialize_vec_convective_term();

  double
  calculate_time_step_size() final;

  double
  recalculate_time_step_size() const final;

  void
  prepare_vectors_for_next_timestep() final;

  void
  do_timestep_solve() final;

  void
  setup_derived() final;

  bool
  print_solver_info() const final;

  void
  read_restart_vectors(boost::archive::binary_iarchive & ia) final;

  void
  write_restart_vectors(boost::archive::binary_oarchive & oa) const final;

  void
  postprocessing() const final;

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

  // postprocessor
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor;

  // This object allows to access utility functions needed for ALE
  std::shared_ptr<HelpersALE<Number> const> helpers_ale;

  // ALE
  VectorType              grid_velocity;
  std::vector<VectorType> vec_grid_coordinates;
  VectorType              grid_coordinates_np;
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_CONVECTION_DIFFUSION_TIME_INT_BDF_H_ */
