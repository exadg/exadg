/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef INCLUDE_EXADG_AERO_ACOUSTIC_DRIVER_H_
#define INCLUDE_EXADG_AERO_ACOUSTIC_DRIVER_H_

// application
#include <exadg/aero_acoustic/user_interface/application_base.h>

// AeroAcoustic
#include <exadg/aero_acoustic/single_field_solvers/acoustics.h>
#include <exadg/aero_acoustic/single_field_solvers/fluid.h>
#include <exadg/aero_acoustic/volume_coupling.h>

// utilities
#include <exadg/utilities/timer_tree.h>

namespace ExaDG
{
namespace AeroAcoustic
{
template<int dim, typename Number>
class Driver
{
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

public:
  Driver(MPI_Comm const &                              comm,
         std::shared_ptr<ApplicationBase<dim, Number>> application,
         bool const                                    is_test);

  void
  setup();

  void
  solve();

  void
  print_performance_results(double const total_time) const;

private:
  void
  setup_volume_coupling();

  void
  set_start_time() const;

  void
  synchronize_time_step_size() const;

  void
  couple_fluid_to_acoustic();

  MPI_Comm const mpi_comm;

  dealii::ConditionalOStream pcout;

  bool const is_test;

  std::shared_ptr<ApplicationBase<dim, Number>> application;

  // single field solvers
  std::shared_ptr<SolverAcoustic<dim, Number>> acoustic;
  std::shared_ptr<SolverFluid<dim, Number>>    fluid;

  // class that manages volume coupling
  VolumeCoupling<dim, Number> volume_coupling;

  // computation time
  mutable TimerTree timer_tree;
};

} // namespace AeroAcoustic
} // namespace ExaDG


#endif /* INCLUDE_EXADG_AERO_ACOUSTIC_DRIVER_H_ */
