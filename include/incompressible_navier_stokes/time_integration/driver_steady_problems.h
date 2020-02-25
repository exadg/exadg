/*
 * driver_steady_problems.h
 *
 *  Created on: Jul 4, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_DRIVER_STEADY_PROBLEMS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_DRIVER_STEADY_PROBLEMS_H_

#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "../postprocessor/postprocessor_base.h"
#include "../spatial_discretization/dg_coupled_solver.h"

using namespace dealii;

namespace IncNS
{
// forward declarations
class InputParameters;

template<int dim, typename Number>
class DriverSteadyProblems
{
public:
  typedef LinearAlgebra::distributed::Vector<Number>      VectorType;
  typedef LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef DGNavierStokesCoupled<dim, Number> Operator;

  DriverSteadyProblems(std::shared_ptr<Operator>                       operator_in,
                       InputParameters const &                         param_in,
                       MPI_Comm const &                                mpi_comm_in,
                       std::shared_ptr<PostProcessorBase<dim, Number>> postprocessor_in);

  void
  setup();

  void
  solve_steady_problem();

  void
  get_wall_times(std::vector<std::string> & name, std::vector<double> & wall_time) const;

  VectorType const &
  get_velocity() const;

private:
  void
  initialize_vectors();

  void
  initialize_solution();

  void
  solve();

  std::shared_ptr<Operator> pde_operator;

  InputParameters const & param;

  MPI_Comm const & mpi_comm;

  std::vector<double> computing_times;

  ConditionalOStream pcout;

  BlockVectorType solution;
  BlockVectorType rhs_vector;

  std::shared_ptr<PostProcessorBase<dim, Number>> postprocessor;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_DRIVER_STEADY_PROBLEMS_H_ */
