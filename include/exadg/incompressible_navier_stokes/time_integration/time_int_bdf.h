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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/time_integration/lambda_functions_ale.h>
#include <exadg/time_integration/time_int_multistep_base.h>

namespace ExaDG
{
namespace IncNS
{
class Parameters;

template<int dim, typename Number>
class SpatialOperatorBase;

template<typename Number>
class PostProcessorInterface;

template<int dim, typename Number>
class TimeIntBDF : public TimeIntBDFBase
{
public:
  using Base            = TimeIntBDFBase;
  using VectorType      = dealii::LinearAlgebra::distributed::Vector<Number>;
  using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<Number>;

  TimeIntBDF(std::shared_ptr<SpatialOperatorBase<dim, Number>> operator_in,
             std::shared_ptr<HelpersALE<Number> const>         helpers_ale_in,
             std::shared_ptr<PostProcessorInterface<Number>>   postprocessor_in,
             Parameters const &                                param_in,
             MPI_Comm const &                                  mpi_comm_in,
             bool const                                        is_test_in);

  virtual ~TimeIntBDF()
  {
  }

  virtual VectorType const &
  get_velocity() const = 0;

  virtual VectorType const &
  get_velocity_np() const = 0;

  virtual VectorType const &
  get_pressure() const = 0;

  virtual VectorType const &
  get_pressure_np() const = 0;

  void
  get_velocities_and_times(std::vector<VectorType const *> & velocities,
                           std::vector<double> &             times) const;

  void
  get_velocities_and_times_np(std::vector<VectorType const *> & velocities,
                              std::vector<double> &             times) const;

  void
  get_pressures_and_times(std::vector<VectorType const *> & pressures,
                          std::vector<double> &             times) const;

  void
  get_pressures_and_times_np(std::vector<VectorType const *> & pressures,
                             std::vector<double> &             times) const;

  void
  ale_update();

  void
  advance_one_timestep_partitioned_solve(bool const use_extrapolation,
                                         bool const update_velocity,
                                         bool const update_pressure);

  virtual void
  print_iterations() const = 0;

  bool
  print_solver_info() const final;

protected:
  void
  allocate_vectors() override;

  void
  setup_derived() override;

  void
  read_restart_vectors(boost::archive::binary_iarchive & ia) override;

  void
  write_restart_vectors(boost::archive::binary_oarchive & oa) const override;

  void
  prepare_vectors_for_next_timestep() override;

  Parameters const & param;

  // number of refinement steps, where the time step size is reduced in
  // factors of 2 with each refinement
  unsigned int const refine_steps_time;

  // global cfl number
  double const cfl;

  // spatial discretization operator
  std::shared_ptr<SpatialOperatorBase<dim, Number>> operator_base;

  // convective term formulated explicitly
  std::vector<VectorType> vec_convective_term;
  VectorType              convective_term_np;

  // required for strongly-coupled partitioned iteration
  bool use_extrapolation;
  bool store_solution;
  bool update_velocity;
  bool update_pressure;

  // This object allows to access utility functions needed for ALE
  std::shared_ptr<HelpersALE<Number> const> helpers_ale;

private:
  void
  get_quantities_and_times(
    std::vector<VectorType const *> &                             quantities,
    std::vector<double> &                                         times,
    std::function<VectorType const *(unsigned int const)> const & get_quantity) const;

  void
  get_quantities_and_times_np(
    std::vector<VectorType const *> &                             quantities,
    std::vector<double> &                                         times,
    std::function<VectorType const *(unsigned int const)> const & get_quantity,
    std::function<VectorType const *()> const &                   get_quantity_np) const;

  void
  initialize_vec_convective_term();

  double
  calculate_time_step_size() final;

  double
  recalculate_time_step_size() const final;

  virtual VectorType const &
  get_velocity(unsigned int i /* t_{n-i} */) const = 0;

  virtual VectorType const &
  get_pressure(unsigned int i /* t_{n-i} */) const = 0;

  virtual void
  set_velocity(VectorType const & velocity, unsigned int const i /* t_{n-i} */) = 0;

  virtual void
  set_pressure(VectorType const & pressure, unsigned int const i /* t_{n-i} */) = 0;

  void
  postprocessing() const final;

  // postprocessor
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor;

  // ALE
  VectorType              grid_velocity;
  std::vector<VectorType> vec_grid_coordinates;
  VectorType              grid_coordinates_np;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_H_ */
