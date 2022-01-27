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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_PRESSURE_CORRECTION_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_PRESSURE_CORRECTION_H_

#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf.h>

namespace ExaDG
{
namespace IncNS
{
// forward declarations
template<int dim, typename Number>
class OperatorPressureCorrection;

template<int dim, typename Number>
class TimeIntBDFPressureCorrection : public TimeIntBDF<dim, Number>
{
private:
  typedef TimeIntBDF<dim, Number> Base;

  typedef typename Base::VectorType VectorType;

  typedef OperatorPressureCorrection<dim, Number> Operator;

public:
  TimeIntBDFPressureCorrection(std::shared_ptr<Operator>                       operator_in,
                               Parameters const &                              param_in,
                               MPI_Comm const &                                mpi_comm_in,
                               bool const                                      is_test_in,
                               std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in);

  virtual ~TimeIntBDFPressureCorrection()
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
  allocate_vectors() final;

  void
  setup_derived() final;

  void
  update_time_integrator_constants() final;

  void
  initialize_current_solution() final;

  void
  initialize_former_solutions() final;

  void
  read_restart_vectors(boost::archive::binary_iarchive & ia) final;

  void
  write_restart_vectors(boost::archive::binary_oarchive & oa) const final;

  void
  initialize_pressure_on_boundary();

  void
  do_timestep_solve() final;

  void
  solve_steady_problem() final;

  double
  evaluate_residual();

  void
  momentum_step();

  void
  rhs_momentum(VectorType & rhs);

  void
  pressure_step(VectorType & pressure_increment);

  void
  projection_step(VectorType const & pressure_increment);

  void
  evaluate_convective_term();

  void
  rhs_projection(VectorType & rhs, VectorType const & pressure_increment) const;

  void
  pressure_update(VectorType const & pressure_increment);

  void
  calculate_chi(double & chi) const;

  void
  rhs_pressure(VectorType & rhs) const;

  void
  prepare_vectors_for_next_timestep() final;

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

  VectorType              velocity_np;
  std::vector<VectorType> velocity;

  VectorType              pressure_np;
  std::vector<VectorType> pressure;

  // incremental formulation of pressure-correction scheme
  unsigned int order_pressure_extrapolation;

  // time integrator constants: extrapolation scheme
  ExtrapolationConstants extra_pressure_gradient;

  // stores pressure Dirichlet boundary values at previous times
  std::vector<VectorType> pressure_dbc;

  // required for strongly-coupled partitioned FSI
  VectorType pressure_increment_last_iter;
  VectorType velocity_momentum_last_iter;
  VectorType velocity_projection_last_iter;

  // iteration counts
  std::pair<
    unsigned int /* calls */,
    std::tuple<unsigned long long, unsigned long long> /* iteration counts {Newton, linear} */>
    iterations_momentum;
  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */>
    iterations_pressure;
  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */>
    iterations_projection;
};

} // namespace IncNS
} // namespace ExaDG


#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_PRESSURE_CORRECTION_H_ \
        */
