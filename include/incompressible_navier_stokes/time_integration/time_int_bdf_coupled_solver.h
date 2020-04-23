/*
 * time_int_bdf_coupled.h
 *
 *  Created on: Jun 13, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_COUPLED_SOLVER_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_COUPLED_SOLVER_H_

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "../spatial_discretization/dg_coupled_solver.h"
#include "time_int_bdf.h"

namespace IncNS
{
// forward declarations
class InputParameters;

template<int dim, typename Number>
class TimeIntBDFCoupled : public TimeIntBDF<dim, Number>
{
public:
  typedef TimeIntBDF<dim, Number> Base;

  typedef typename Base::VectorType VectorType;

  typedef LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef DGNavierStokesCoupled<dim, Number> Operator;

  TimeIntBDFCoupled(
    std::shared_ptr<Operator>                       operator_in,
    InputParameters const &                         param_in,
    unsigned int const                              refine_steps_time_in,
    MPI_Comm const &                                mpi_comm_in,
    std::shared_ptr<PostProcessorBase<dim, Number>> postprocessor_in,
    std::shared_ptr<MovingMeshBase<dim, Number>>    moving_mesh_in         = nullptr,
    std::shared_ptr<MatrixFreeWrapper<dim, Number>> matrix_free_wrapper_in = nullptr);

  void
  postprocessing_stability_analysis();

  void
  get_iterations(std::vector<std::string> & name, std::vector<double> & iteration) const;

  void
  get_wall_times(std::vector<std::string> & name, std::vector<double> & wall_time) const;

private:
  void
  setup_derived() override;

  void
  allocate_vectors() override;

  void
  initialize_current_solution();

  void
  initialize_former_solutions();

  void
  solve_timestep() override;

  void
  solve_steady_problem();

  double
  evaluate_residual();

  void
  projection_step();

  void
  prepare_vectors_for_next_timestep() override;

  LinearAlgebra::distributed::Vector<Number> const &
  get_velocity() const;

  LinearAlgebra::distributed::Vector<Number> const &
  get_velocity_np() const;

  LinearAlgebra::distributed::Vector<Number> const &
  get_velocity(unsigned int i /* t_{n-i} */) const;

  LinearAlgebra::distributed::Vector<Number> const &
  get_pressure(unsigned int i /* t_{n-i} */) const;

  void
  set_velocity(VectorType const & velocity, unsigned int const i /* t_{n-i} */);

  void
  set_pressure(VectorType const & pressure, unsigned int const i /* t_{n-i} */);

  std::shared_ptr<Operator> pde_operator;

  std::vector<BlockVectorType> solution;
  BlockVectorType              solution_np;

  // performance analysis: average number of iterations and solver time
  mutable std::vector<double> computing_times;
  double                      computing_time_convective;
  std::vector<unsigned int>   iterations;
  unsigned int                N_iter_nonlinear;

  // scaling factor continuity equation
  double scaling_factor_continuity;
  double characteristic_element_length;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_COUPLED_SOLVER_H_ */
