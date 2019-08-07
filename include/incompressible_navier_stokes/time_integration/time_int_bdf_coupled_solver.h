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

#include "time_int_bdf_navier_stokes.h"

namespace IncNS
{
// forward declarations
class InputParameters;

namespace Interface
{
template<typename Number>
class OperatorBase;
template<typename Number>
class OperatorCoupled;

} // namespace Interface

template<typename Number>
class TimeIntBDFCoupled : public TimeIntBDF<Number>
{
public:
  typedef TimeIntBDF<Number> Base;

  typedef typename Base::VectorType VectorType;

  typedef LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef Interface::OperatorBase<Number>    InterfaceBase;
  typedef Interface::OperatorCoupled<Number> InterfacePDE;

  TimeIntBDFCoupled(std::shared_ptr<InterfaceBase> operator_base_in,
                    std::shared_ptr<InterfacePDE>  pde_operator_in,
                    InputParameters const &        param_in);

  void
  postprocessing_stability_analysis();

  void
  get_iterations(std::vector<std::string> & name, std::vector<double> & iteration) const;

  void
  get_wall_times(std::vector<std::string> & name, std::vector<double> & wall_time) const;

  //ALE

  void
  reinit_former_solution_with_former_mesh_ALE(std::vector<BlockVectorType> solution_in) override;

  void
  reinit_convective_term_with_former_mesh_ALE(std::vector<VectorType> vec_convective_term_in) override;

private:
  void
  setup_derived();

  void
  allocate_vectors();

  void
  initialize_current_solution();

  void
  initialize_former_solutions();

  void
  initialize_vec_convective_term();

  void
  solve_timestep();

  void
  solve_steady_problem();

  double
  evaluate_residual();

  void
  projection_step();

  void
  postprocessing() const;

  void
  postprocessing_steady_problem() const;

  void
  prepare_vectors_for_next_timestep();

  LinearAlgebra::distributed::Vector<Number> const &
  get_velocity() const;

  LinearAlgebra::distributed::Vector<Number> const &
  get_velocity(unsigned int i /* t_{n-i} */) const;

  LinearAlgebra::distributed::Vector<Number> const &
  get_pressure(unsigned int i /* t_{n-i} */) const;

  void
  set_velocity(VectorType const & velocity, unsigned int const i /* t_{n-i} */);

  void
  set_pressure(VectorType const & pressure, unsigned int const i /* t_{n-i} */);

  std::shared_ptr<InterfacePDE> pde_operator;

  std::vector<BlockVectorType> solution;
  BlockVectorType              solution_np;

  BlockVectorType rhs_vector;

  std::vector<VectorType> vec_convective_term;

  //ALE:
  VectorType vec_convective_np;

  // performance analysis: average number of iterations and solver time
  std::vector<Number>       computing_times;
  std::vector<unsigned int> iterations;
  unsigned int              N_iter_nonlinear;

  // scaling factor continuity equation
  double scaling_factor_continuity;
  double characteristic_element_length;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_COUPLED_SOLVER_H_ */
