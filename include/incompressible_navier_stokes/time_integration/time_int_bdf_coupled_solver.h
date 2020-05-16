/*
 * time_int_bdf_coupled.h
 *
 *  Created on: Jun 13, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_COUPLED_SOLVER_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_COUPLED_SOLVER_H_

// deal.II
#include <deal.II/lac/la_parallel_block_vector.h>

#include "time_int_bdf.h"

namespace IncNS
{
// forward declarations
template<int dim, typename Number>
class DGNavierStokesCoupled;

template<int dim, typename Number>
class TimeIntBDFCoupled : public TimeIntBDF<dim, Number>
{
private:
  typedef TimeIntBDF<dim, Number> Base;

  typedef typename Base::VectorType VectorType;

  typedef LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef DGNavierStokesCoupled<dim, Number> Operator;

public:
  TimeIntBDFCoupled(std::shared_ptr<Operator>                       operator_in,
                    InputParameters const &                         param_in,
                    unsigned int const                              refine_steps_time_in,
                    MPI_Comm const &                                mpi_comm_in,
                    std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in,
                    std::shared_ptr<MovingMeshBase<dim, Number>>    moving_mesh_in = nullptr,
                    std::shared_ptr<MatrixFree<dim, Number>>        matrix_free_in = nullptr);

  void
  postprocessing_stability_analysis();

  void
  print_iterations() const;

  VectorType const &
  get_velocity_np() const;

  VectorType const &
  get_pressure_np() const;

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
  penalty_step();

  void
  prepare_vectors_for_next_timestep() override;

  VectorType const &
  get_velocity() const;

  VectorType const &
  get_velocity(unsigned int i /* t_{n-i} */) const;

  VectorType const &
  get_pressure(unsigned int i /* t_{n-i} */) const;

  void
  set_velocity(VectorType const & velocity, unsigned int const i /* t_{n-i} */);

  void
  set_pressure(VectorType const & pressure, unsigned int const i /* t_{n-i} */);

  std::shared_ptr<Operator> pde_operator;

  std::vector<BlockVectorType> solution;
  BlockVectorType              solution_np;

  // required for strongly-coupled partitioned FSI
  BlockVectorType solution_last_iter;
  VectorType      velocity_penalty_last_iter;

  // iteration counts
  std::pair<
    unsigned int /* calls */,
    std::tuple<unsigned long long, unsigned long long> /* iteration counts {Newton, linear} */>
                                                                                 iterations;
  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */> iterations_penalty;

  // scaling factor continuity equation
  double scaling_factor_continuity;
  double characteristic_element_length;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_COUPLED_SOLVER_H_ */
