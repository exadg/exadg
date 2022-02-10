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

#ifndef INCLUDE_EXADG_STRUCTURE_TIME_INTEGRATION_TIME_INT_GEN_ALPHA_H_
#define INCLUDE_EXADG_STRUCTURE_TIME_INTEGRATION_TIME_INT_GEN_ALPHA_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/time_integration/time_int_gen_alpha_base.h>
#include <exadg/utilities/timer_tree.h>

namespace ExaDG
{
namespace Structure
{
// forward declarations
class Parameters;

template<typename Number>
class PostProcessorBase;

namespace Interface
{
template<typename Number>
class Operator;

class Parameters;
} // namespace Interface

template<int dim, typename Number>
class TimeIntGenAlpha : public TimeIntGenAlphaBase<Number>
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  TimeIntGenAlpha(std::shared_ptr<Interface::Operator<Number>> operator_,
                  std::shared_ptr<PostProcessorBase<Number>>   postprocessor_,
                  Parameters const &                           param_,
                  MPI_Comm const &                             mpi_comm_,
                  bool const                                   is_test_);

  void
  setup(bool const do_restart) final;

  void
  print_iterations() const;

  void
  extrapolate_displacement_to_np(VectorType & displacement);

  VectorType const &
  get_displacement_np();

  void
  extrapolate_velocity_to_np(VectorType & velocity);

  VectorType const &
  get_velocity_n();

  VectorType const &
  get_velocity_np();

  void
  set_displacement(VectorType const & displacement);

  void
  advance_one_timestep_partitioned_solve(bool const use_extrapolation);

private:
  void
  do_timestep_solve() final;

  void
  prepare_vectors_for_next_timestep() final;

  void
  do_write_restart(std::string const & filename) const final;

  void
  do_read_restart(std::ifstream & in) final;

  void
  postprocessing() const final;

  bool
  print_solver_info() const final;

  std::shared_ptr<Interface::Operator<Number>> pde_operator;

  std::shared_ptr<PostProcessorBase<Number>> postprocessor;

  // number of refinement steps, where the time step size is reduced in
  // factors of 2 with each refinement
  unsigned int const refine_steps_time;

  Parameters const & param;

  MPI_Comm const mpi_comm;

  dealii::ConditionalOStream pcout;

  // DoF vectors
  VectorType displacement_n, displacement_np;
  VectorType velocity_n, velocity_np;
  VectorType acceleration_n, acceleration_np;

  // required for strongly-coupled partitioned FSI
  bool       use_extrapolation;
  bool       store_solution;
  VectorType displacement_last_iter;

  std::pair<
    unsigned int /* calls */,
    std::tuple<unsigned long long, unsigned long long> /* iteration counts {Newton, linear}*/>
    iterations;
};

} // namespace Structure
} // namespace ExaDG


#endif /* INCLUDE_EXADG_STRUCTURE_TIME_INTEGRATION_TIME_INT_GEN_ALPHA_H_ */
