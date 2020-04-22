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

#include "../postprocessor/postprocessor.h"
#include "../spatial_discretization/operator.h"

namespace Structure
{
template<int dim, typename Number>
class TimeIntGenAlpha : public TimeIntGenAlphaBase<Number>
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  TimeIntGenAlpha(std::shared_ptr<Operator<dim, Number>>      operator_,
                  std::shared_ptr<PostProcessor<dim, Number>> postprocessor_,
                  InputParameters const &                     param_,
                  MPI_Comm const &                            mpi_comm_);

  void
  setup(bool const do_restart) override;

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

  std::shared_ptr<Operator<dim, Number>> pde_operator;

  std::shared_ptr<PostProcessor<dim, Number>> postprocessor;

  InputParameters const & param;

  MPI_Comm const & mpi_comm;

  ConditionalOStream pcout;

  // DoF vectors
  VectorType displacement_n, displacement_np;
  VectorType velocity_n, velocity_np;
  VectorType acceleration_n, acceleration_np;
};

} // namespace Structure



#endif /* INCLUDE_STRUCTURE_TIME_INTEGRATION_TIME_INT_GEN_ALPHA_H_ */
