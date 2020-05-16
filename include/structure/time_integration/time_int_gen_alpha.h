/*
 * time_int_gen_alpha.h
 *
 *  Created on: 20.04.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_TIME_INTEGRATION_TIME_INT_GEN_ALPHA_H_
#define INCLUDE_STRUCTURE_TIME_INTEGRATION_TIME_INT_GEN_ALPHA_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

#include "../../time_integration/time_int_gen_alpha_base.h"

#include "../../utilities/timings_hierarchical.h"

namespace Structure
{
// forward declarations
class InputParameters;

template<typename Number>
class PostProcessorBase;

namespace Interface
{
template<typename Number>
class Operator;

class InputParameters;
} // namespace Interface

template<int dim, typename Number>
class TimeIntGenAlpha : public TimeIntGenAlphaBase<Number>
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  TimeIntGenAlpha(std::shared_ptr<Interface::Operator<Number>> operator_,
                  std::shared_ptr<PostProcessorBase<Number>>   postprocessor_,
                  unsigned int const                           refine_time_,
                  InputParameters const &                      param_,
                  MPI_Comm const &                             mpi_comm_);

  void
  setup(bool const do_restart) override;

  void
  print_iterations() const;

  void
  extrapolate_displacement_to_np(VectorType & displacement);

  VectorType const &
  get_displacement_np();

  void
  extrapolate_velocity_to_np(VectorType & velocity);

  VectorType const &
  get_velocity_n();

  VectorType const &
  get_velocity_np();

  void
  relax_displacement(double const omega, VectorType const & displacement_previous);

  void
  advance_one_timestep_partitioned_solve(bool const use_extrapolation, bool const store_solution);

private:
  void
  solve_timestep() override;

  void
  prepare_vectors_for_next_timestep() override;

  void
  do_write_restart(std::string const & filename) const override;

  void
  do_read_restart(std::ifstream & in) override;

  void
  postprocessing() const override;

  bool
  print_solver_info() const;

  std::shared_ptr<Interface::Operator<Number>> pde_operator;

  std::shared_ptr<PostProcessorBase<Number>> postprocessor;

  // number of refinement steps, where the time step size is reduced in
  // factors of 2 with each refinement
  unsigned int const refine_steps_time;

  InputParameters const & param;

  MPI_Comm const & mpi_comm;

  ConditionalOStream pcout;

  // DoF vectors
  VectorType displacement_n, displacement_np;
  VectorType velocity_n, velocity_np;
  VectorType acceleration_n, acceleration_np;

  // required for strongly-coupled partitioned FSI
  bool       use_extrapolation;
  bool       store_solution;
  VectorType displacement_last_iter;

  std::pair<
    unsigned int /* calls */,
    std::tuple<unsigned long long, unsigned long long> /* iteration counts {Newton, linear}*/>
    iterations;
};

} // namespace Structure



#endif /* INCLUDE_STRUCTURE_TIME_INTEGRATION_TIME_INT_GEN_ALPHA_H_ */
