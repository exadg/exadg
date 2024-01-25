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

#ifndef INCLUDE_EXADG_AERO_ACOUSTIC_SINGLE_FIELD_SOLVERS_ACOUSTICS_H_
#define INCLUDE_EXADG_AERO_ACOUSTIC_SINGLE_FIELD_SOLVERS_ACOUSTICS_H_

// Acoustics
#include <exadg/acoustic_conservation_equations/time_integration/time_int_abm.h>
#include <exadg/acoustic_conservation_equations/user_interface/application_base.h>

// application
#include <exadg/aero_acoustic/user_interface/application_base.h>

namespace ExaDG
{
namespace AeroAcoustic
{
template<int dim, typename Number>
class SolverAcoustic
{
public:
  void
  setup(std::shared_ptr<AcousticsAeroAcoustic::ApplicationBase<dim, Number>> application,
        MPI_Comm const                                                       mpi_comm,
        bool const                                                           is_test)
  {
    // setup application
    application->setup(grid, mapping);

    // setup spatial operator
    pde_operator = std::make_shared<Acoustics::SpatialOperator<dim, Number>>(
      grid,
      mapping,
      application->get_boundary_descriptor(),
      application->get_field_functions(),
      application->get_parameters(),
      "acoustic",
      mpi_comm);

    pde_operator->setup();

    // initialize postprocessor
    postprocessor = application->create_postprocessor();
    postprocessor->setup(*pde_operator);

    // initialize time integrator
    time_integrator = std::make_shared<Acoustics::TimeIntAdamsBashforthMoulton<Number>>(
      pde_operator, application->get_parameters(), postprocessor, mpi_comm, is_test);

    time_integrator->setup(application->get_parameters().restarted_simulation);
  }

  void
  advance_multiple_timesteps(double const global_dt)
  {
    double const dt = compute_sub_time_step_size(global_dt);

    // define epsilon dependent on time-step size to avoid
    // wrong terminations of the sub-timestepping loop.
    double eps = 0.01 * dt;

    // perform time-steps until we advanced the global time step
    unsigned int n_sub_time_steps = 0;
    while(n_sub_time_steps * dt + eps < global_dt)
    {
      time_integrator->set_current_time_step_size(dt);
      time_integrator->advance_one_timestep();
      ++n_sub_time_steps;
    }

    if(time_integrator->started())
    {
      ++sub_time_steps.first;
      sub_time_steps.second += n_sub_time_steps;
    }
  }

  double
  get_average_sub_time_steps() const
  {
    return sub_time_steps.second / sub_time_steps.first;
  }

  unsigned int
  get_number_of_global_time_steps() const
  {
    return sub_time_steps.first;
  }

  // grid and mapping
  std::shared_ptr<Grid<dim>>            grid;
  std::shared_ptr<dealii::Mapping<dim>> mapping;

  // spatial discretization
  std::shared_ptr<Acoustics::SpatialOperator<dim, Number>> pde_operator;

  // temporal discretization
  std::shared_ptr<Acoustics::TimeIntAdamsBashforthMoulton<Number>> time_integrator;

  // postprocessor
  std::shared_ptr<Acoustics::PostProcessorBase<dim, Number>> postprocessor;

private:
  double
  compute_sub_time_step_size(double const global_dt)
  {
    double const current_dt = time_integrator->get_time_step_size();
    if(global_dt < current_dt)
    {
      // in case the global time step is smaller than the acoustic timestep, simply run
      // with global time step
      return global_dt;
    }
    else
    {
      // otherwise, ensure that sub-timestepping ends exactly at global_dt
      return adjust_time_step_to_hit_end_time(0.0, global_dt, current_dt);
    }
  }

  std::pair<unsigned int /* calls */, unsigned long long /* sub_dt */> sub_time_steps;
};


} // namespace AeroAcoustic
} // namespace ExaDG

#endif /* INCLUDE_EXADG_AERO_ACOUSTIC_SINGLE_FIELD_SOLVERS_ACOUSTICS_H_ */
