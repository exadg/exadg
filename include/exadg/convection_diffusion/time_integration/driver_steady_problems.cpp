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

#include <exadg/convection_diffusion/postprocessor/postprocessor_base.h>
#include <exadg/convection_diffusion/spatial_discretization/interface.h>
#include <exadg/convection_diffusion/time_integration/driver_steady_problems.h>
#include <exadg/convection_diffusion/user_interface/parameters.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace ConvDiff
{
using namespace dealii;

template<typename Number>
DriverSteadyProblems<Number>::DriverSteadyProblems(
  std::shared_ptr<Operator>                       operator_,
  Parameters const &                              param_,
  MPI_Comm const &                                mpi_comm_,
  bool const                                      is_test_,
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor_)
  : pde_operator(operator_),
    param(param_),
    mpi_comm(mpi_comm_),
    is_test(is_test_),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm_) == 0),
    timer_tree(new TimerTree()),
    postprocessor(postprocessor_)
{
}

template<typename Number>
void
DriverSteadyProblems<Number>::setup()
{
  // initialize global solution vectors (allocation)
  initialize_vectors();

  // initialize solution by interpolation of initial data
  initialize_solution();
}

template<typename Number>
void
DriverSteadyProblems<Number>::solve_problem()
{
  Timer timer;
  timer.restart();

  postprocessing();

  solve();

  postprocessing();

  timer_tree->insert({"DriverSteady"}, timer.wall_time());
}

template<typename Number>
std::shared_ptr<TimerTree>
DriverSteadyProblems<Number>::get_timings() const
{
  return timer_tree;
}

template<typename Number>
void
DriverSteadyProblems<Number>::initialize_vectors()
{
  // solution
  pde_operator->initialize_dof_vector(solution);

  // rhs_vector
  pde_operator->initialize_dof_vector(rhs_vector);
}

template<typename Number>
void
DriverSteadyProblems<Number>::initialize_solution()
{
  double time = 0.0;
  pde_operator->prescribe_initial_conditions(solution, time);
}

template<typename Number>
void
DriverSteadyProblems<Number>::solve()
{
  Timer timer;
  timer.restart();

  pcout << std::endl << "Solving steady state problem ..." << std::endl;

  // prepare pointer for velocity field, but only if necessary
  VectorType const * velocity_ptr = nullptr;
  VectorType         velocity_vector;

  if(param.linear_system_including_convective_term_has_to_be_solved())
  {
    if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
    {
      if(param.analytical_velocity_field)
      {
        pde_operator->initialize_dof_vector_velocity(velocity_vector);
        pde_operator->project_velocity(velocity_vector, 0.0 /* time */);

        velocity_ptr = &velocity_vector;
      }
      else
      {
        AssertThrow(false, ExcMessage("Not implemented."));
      }
    }
  }

  // calculate rhs vector
  pde_operator->rhs(rhs_vector, 0.0 /* time */, velocity_ptr);

  // solve linear system of equations
  unsigned int iterations = pde_operator->solve(solution,
                                                rhs_vector,
                                                param.update_preconditioner,
                                                1.0 /* scaling_factor */,
                                                0.0 /* time */,
                                                velocity_ptr);

  if(not(is_test))
    print_solver_info_linear(pcout, iterations, timer.wall_time());

  pcout << std::endl << "... done!" << std::endl;

  timer_tree->insert({"DriverSteady", "Solve"}, timer.wall_time());
}

template<typename Number>
void
DriverSteadyProblems<Number>::postprocessing() const
{
  Timer timer;
  timer.restart();

  postprocessor->do_postprocessing(solution);

  timer_tree->insert({"DriverSteady", "Postprocessing"}, timer.wall_time());
}

// instantiations
template class DriverSteadyProblems<float>;
template class DriverSteadyProblems<double>;

} // namespace ConvDiff
} // namespace ExaDG
