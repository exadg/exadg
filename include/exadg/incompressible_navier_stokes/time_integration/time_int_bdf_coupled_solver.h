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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_COUPLED_SOLVER_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_COUPLED_SOLVER_H_

// deal.II
#include <deal.II/lac/la_parallel_block_vector.h>

// ExaDG
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf.h>

namespace ExaDG
{
namespace IncNS
{
// forward declarations
template<int dim, typename Number>
class OperatorCoupled;

template<int dim, typename Number>
class TimeIntBDFCoupled : public TimeIntBDF<dim, Number>
{
private:
  typedef TimeIntBDF<dim, Number> Base;

  typedef typename Base::VectorType VectorType;

  typedef dealii::LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef OperatorCoupled<dim, Number> Operator;

public:
  TimeIntBDFCoupled(std::shared_ptr<Operator>                       operator_in,
                    std::shared_ptr<HelpersALE<Number> const>       helpers_ale_in,
                    std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in,
                    Parameters const &                              param_in,
                    MPI_Comm const &                                mpi_comm_in,
                    bool const                                      is_test_in);

  void
  postprocessing_stability_analysis();

  void
  print_iterations() const final;

  VectorType const &
  get_velocity() const final;

  VectorType const &
  get_velocity_np() const final;

  VectorType const &
  get_pressure() const final;

  VectorType const &
  get_pressure_np() const final;

private:
  void
  allocate_vectors() final;

  void
  setup_derived() final;

  void
  initialize_current_solution() final;

  void
  initialize_former_solutions() final;

  void
  do_timestep_solve() final;

  void
  solve_steady_problem() final;

  double
  evaluate_residual();

  void
  penalty_step();

  void
  prepare_vectors_for_next_timestep() final;

  VectorType const &
  get_velocity(unsigned int i /* t_{n-i} */) const final;

  VectorType const &
  get_pressure(unsigned int i /* t_{n-i} */) const final;

  void
  set_velocity(VectorType const & velocity, unsigned int const i /* t_{n-i} */) final;

  void
  set_pressure(VectorType const & pressure, unsigned int const i /* t_{n-i} */) final;

  std::shared_ptr<Operator> pde_operator;

  std::vector<BlockVectorType> solution;
  BlockVectorType              solution_np;

  // required for strongly-coupled partitioned FSI
  BlockVectorType solution_last_iter;
  VectorType      velocity_penalty_last_iter;

  // iteration counts
  std::pair<
    unsigned int /* calls */,
    std::tuple<unsigned long long, unsigned long long> /* iteration counts {Newton, linear} */>
                                                                                 iterations;
  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */> iterations_penalty;

  // scaling factor continuity equation
  double scaling_factor_continuity;
  double characteristic_element_length;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_COUPLED_SOLVER_H_ \
        */
