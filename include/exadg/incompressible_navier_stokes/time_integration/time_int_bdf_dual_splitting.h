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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_DUAL_SPLITTING_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_DUAL_SPLITTING_H_

#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

// forward declarations
template<int dim, typename Number>
class OperatorDualSplitting;

template<int dim, typename Number>
class TimeIntBDFDualSplitting : public TimeIntBDF<dim, Number>
{
private:
  typedef TimeIntBDF<dim, Number> Base;

  typedef typename Base::VectorType VectorType;

  typedef OperatorDualSplitting<dim, Number> Operator;

public:
  TimeIntBDFDualSplitting(std::shared_ptr<Operator>                       pde_operator_in,
                          InputParameters const &                         param_in,
                          unsigned int const                              refine_steps_time_in,
                          MPI_Comm const &                                mpi_comm_in,
                          bool const                                      is_test_in,
                          std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in);

  virtual ~TimeIntBDFDualSplitting()
  {
  }

  void
  postprocessing_stability_analysis();

  void
  print_iterations() const final;

  VectorType const &
  get_velocity_np() const final;

  VectorType const &
  get_pressure_np() const final;

private:
  void
  setup_derived() final;

  void
  read_restart_vectors(boost::archive::binary_iarchive & ia) final;

  void
  write_restart_vectors(boost::archive::binary_oarchive & oa) const final;

  void
  solve_timestep() final;

  void
  allocate_vectors() final;

  void
  prepare_vectors_for_next_timestep() final;

  void
  update_velocity_dbc();

  void
  convective_step();

  void
  evaluate_convective_term();

  void
  update_time_integrator_constants() final;

  void
  initialize_current_solution() final;

  void
  initialize_former_solutions() final;

  void
  initialize_acceleration_and_velocity_on_boundary();

  void
  pressure_step();

  void
  rhs_pressure(VectorType & rhs) const;

  void
  projection_step();

  void
  rhs_projection(VectorType & rhs) const;

  void
  penalty_step();

  void
  viscous_step();

  void
  rhs_viscous(VectorType & rhs) const;

  void
  solve_steady_problem() final;

  double
  evaluate_residual();

  VectorType const &
  get_velocity() const final;

  VectorType const &
  get_velocity(unsigned int i /* t_{n-i} */) const final;

  VectorType const &
  get_pressure(unsigned int i /* t_{n-i} */) const final;

  void
  set_velocity(VectorType const & velocity, unsigned int const i /* t_{n-i} */) final;

  void
  set_pressure(VectorType const & pressure, unsigned int const i /* t_{n-i} */) final;

  std::shared_ptr<Operator> pde_operator;

  std::vector<VectorType> velocity;

  VectorType velocity_np;

  std::vector<VectorType> pressure;

  VectorType pressure_np;

  std::vector<VectorType> velocity_dbc;
  VectorType              velocity_dbc_np;

  // required for strongly-coupled partitioned FSI
  VectorType pressure_last_iter;
  VectorType velocity_projection_last_iter;
  VectorType velocity_viscous_last_iter;

  // iteration counts
  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */>
    iterations_pressure;
  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */>
                                                                                 iterations_projection;
  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */> iterations_viscous;
  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */> iterations_penalty;

  // time integrator constants: extrapolation scheme
  ExtrapolationConstants extra_pressure_nbc;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_DUAL_SPLITTING_H_ \
        */
