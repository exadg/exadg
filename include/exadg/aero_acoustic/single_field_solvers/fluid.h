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

#ifndef INCLUDE_EXADG_AERO_ACOUSTIC_SINGLE_FIELD_SOLVERS_FLUID_H_
#define INCLUDE_EXADG_AERO_ACOUSTIC_SINGLE_FIELD_SOLVERS_FLUID_H_

// IncNS
#include <exadg/incompressible_navier_stokes/postprocessor/postprocessor.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/create_operator.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_coupled.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_pressure_correction.h>
#include <exadg/incompressible_navier_stokes/time_integration/create_time_integrator.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.h>

// utilities
#include <exadg/utilities/timer_tree.h>

// application
#include <exadg/aero_acoustic/user_interface/application_base.h>

namespace ExaDG
{
namespace AeroAcoustic
{
template<int dim, typename Number>
class SolverFluid
{
public:
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  SolverFluid()
    : timer_tree(std::make_shared<TimerTree>()),
      adaptive_time_stepping_limiting_factor(std::numeric_limits<double>::min())
  {
  }

  void
  setup(std::shared_ptr<FluidAeroAcoustic::ApplicationBase<dim, Number>> application,
        MPI_Comm const                                                   mpi_comm,
        bool const                                                       is_test)
  {
    // setup application
    application->setup(grid, mapping, multigrid_mappings);

    // initialize pde_operator
    pde_operator = IncNS::create_operator<dim, Number>(grid,
                                                       mapping,
                                                       multigrid_mappings,
                                                       application->get_boundary_descriptor(),
                                                       application->get_field_functions(),
                                                       application->get_parameters(),
                                                       "fluid",
                                                       mpi_comm);

    // setup Navier-Stokes operator
    pde_operator->setup();

    // setup postprocessor
    postprocessor = application->create_postprocessor();
    postprocessor->setup(*pde_operator);

    // setup time integrator before calling setup_solvers (this is necessary since the setup
    // of the solvers depends on quantities such as the time_step_size or gamma0!)
    AssertThrow(application->get_parameters().solver_type == IncNS::SolverType::Unsteady,
                dealii::ExcMessage("Invalid parameter in context of fluid-structure interaction."));

    // initialize time_integrator
    time_integrator = IncNS::create_time_integrator<dim, Number>(pde_operator,
                                                                 nullptr /*no ALE*/,
                                                                 postprocessor,
                                                                 application->get_parameters(),
                                                                 mpi_comm,
                                                                 is_test);

    time_integrator->setup(application->get_parameters().restarted_simulation);

    // initialize vector that stores the pressure time derivative
    pde_operator->initialize_vector_pressure(pressure_time_derivative);

    // store the adaptive time-stepping limiting factor
    adaptive_time_stepping_limiting_factor =
      application->get_parameters().adaptive_time_stepping_limiting_factor;
  }

  double
  max_next_time_step_size() const
  {
    const auto dt = time_integrator->get_time_step_size();
    return dt * adaptive_time_stepping_limiting_factor;
  }

  void
  advance_one_timestep_and_compute_pressure_time_derivative(bool const update_dpdt)
  {
    time_integrator->advance_one_timestep_pre_solve(true);
    time_integrator->advance_one_timestep_solve();

    // The pressure time derivative has to be computed before the push back
    // of pressure vectors that is triggered in advance_one_timestep_post_solve()
    if(update_dpdt)
    {
      std::vector<VectorType const *> pressures;
      std::vector<double>             times;
      time_integrator->get_pressures_and_times_np(pressures, times);

      compute_bdf_time_derivative(pressure_time_derivative, pressures, times);
    }

    time_integrator->advance_one_timestep_post_solve();
  }

  VectorType const &
  get_pressure_time_derivative() const
  {
    return pressure_time_derivative;
  }

  // grid and mapping
  std::shared_ptr<Grid<dim>>            grid;
  std::shared_ptr<dealii::Mapping<dim>> mapping;

  std::shared_ptr<MultigridMappings<dim, Number>> multigrid_mappings;

  // spatial discretization
  std::shared_ptr<IncNS::SpatialOperatorBase<dim, Number>> pde_operator;

  // temporal discretization
  std::shared_ptr<IncNS::TimeIntBDF<dim, Number>> time_integrator;

  // Postprocessor
  std::shared_ptr<IncNS::PostProcessorBase<dim, Number>> postprocessor;

  // Computation time (wall clock time).
  std::shared_ptr<TimerTree> timer_tree;

private:
  // The aeroacoustic source term needs the pressure time derivative.
  // The update of the vector is performed using
  // advance_one_timestep_and_compute_pressure_time_derivative(true).
  VectorType pressure_time_derivative;

  // To be able to estimate if dp/dt has to be evaluated we have to
  // estimate the maximum time-step size that can be performed after
  // the current time step. For that we need the limiting factor.
  double adaptive_time_stepping_limiting_factor;
};

} // namespace AeroAcoustic
} // namespace ExaDG



#endif /* INCLUDE_EXADG_AERO_ACOUSTIC_SINGLE_FIELD_SOLVERS_FLUID_H_ */
