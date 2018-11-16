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
template<int dim>
class InputParameters;

namespace Interface
{
template<typename Number>
class OperatorBase;
template<typename Number>
class OperatorCoupled;

} // namespace Interface

template<int dim, typename Number>
class TimeIntBDFCoupled : public TimeIntBDF<dim, Number>
{
public:
  typedef TimeIntBDF<dim, Number> Base;

  typedef typename Base::VectorType VectorType;

  typedef LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef Interface::OperatorBase<Number>    InterfaceBase;
  typedef Interface::OperatorCoupled<Number> InterfacePDE;

  TimeIntBDFCoupled(std::shared_ptr<InterfaceBase> operator_base_in,
                    std::shared_ptr<InterfacePDE>  pde_operator_in,
                    InputParameters<dim> const &   param_in,
                    unsigned int const             n_refine_time_in,
                    bool const                     use_adaptive_time_stepping_in);

  void
  analyze_computing_times() const;

  void
  postprocessing_stability_analysis();

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
  postprocess_velocity();

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

  // performance analysis: average number of iterations and solver time
  std::vector<Number>       computing_times;
  std::vector<unsigned int> iterations;
  unsigned int              N_iter_nonlinear;

  // scaling factor continuity equation
  double scaling_factor_continuity;
  double characteristic_element_length;

  // temporary vectors needed for pseudo-timestepping algorithm
  VectorType velocity_tmp;
  VectorType pressure_tmp;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_COUPLED_SOLVER_H_ */
