/*
 * driver_steady_problems.cpp
 *
 *  Created on: Nov 14, 2018
 *      Author: fehn
 */

#include "driver_steady_problems.h"

#include "../spatial_discretization/interface.h"
#include "../user_interface/input_parameters.h"

namespace ConvDiff
{
template<typename Number>
DriverSteadyProblems<Number>::DriverSteadyProblems(
  std::shared_ptr<Operator>                       operator_in,
  InputParameters const &                         param_in,
  MPI_Comm const &                                mpi_comm_in,
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in)
  : pde_operator(operator_in),
    param(param_in),
    mpi_comm(mpi_comm_in),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm_in) == 0),
    timer_tree(new TimerTree()),
    postprocessor(postprocessor_in)
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

  // write output
  pcout << std::endl
        << "Solve linear system of equations:" << std::endl
        << "  Iterations: " << std::setw(6) << std::right << iterations
        << "\t Wall time [s]: " << std::scientific << std::setprecision(4) << timer.wall_time()
        << std::endl;

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
