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
DriverSteadyProblems<Number>::DriverSteadyProblems(std::shared_ptr<Operator> operator_in,
                                                   InputParameters const &   param_in)
  : pde_operator(operator_in),
    param(param_in),
    pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    computing_times(1)
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
  postprocessing();

  solve();

  postprocessing();
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
  pcout << std::endl << "Solving steady state problem ..." << std::endl;

  Timer timer;
  timer.restart();

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

  computing_times[0] += timer.wall_time();

  // write output
  pcout << std::endl
        << "Solve linear system of equations:" << std::endl
        << "  Iterations: " << std::setw(6) << std::right << iterations
        << "\t Wall time [s]: " << std::scientific << std::setprecision(4) << computing_times[0]
        << std::endl;

  pcout << std::endl << "... done!" << std::endl;
}

template<typename Number>
void
DriverSteadyProblems<Number>::postprocessing() const
{
  pde_operator->do_postprocessing(solution);
}

template<typename Number>
void
DriverSteadyProblems<Number>::get_wall_times(std::vector<std::string> & name,
                                             std::vector<double> &      wall_time) const
{
  name.resize(1);
  std::vector<std::string> names = {"Linear system"};
  name                           = names;

  wall_time.resize(1);
  wall_time[0] = computing_times[0];
}

// instantiations

template class DriverSteadyProblems<float>;
template class DriverSteadyProblems<double>;

} // namespace ConvDiff
