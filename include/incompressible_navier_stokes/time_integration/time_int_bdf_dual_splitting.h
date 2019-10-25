/*
 * time_int_bdf_dual_splitting.h
 *
 *  Created on: May 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_DUAL_SPLITTING_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_DUAL_SPLITTING_H_

#include <deal.II/lac/la_parallel_block_vector.h>

#include "time_int_bdf_navier_stokes.h"

using namespace dealii;

namespace IncNS
{
// forward declarations
class InputParameters;

namespace Interface
{
template<typename Number>
class OperatorBase;
template<typename Number>
class OperatorDualSplitting;

} // namespace Interface

template<typename Number>
class TimeIntBDFDualSplitting : public TimeIntBDF<Number>
{
public:
  typedef TimeIntBDF<Number> Base;

  typedef typename Base::VectorType      VectorType;
  typedef typename Base::BlockVectorType BlockVectorType;

  typedef Interface::OperatorBase<Number>          InterfaceBase;
  typedef Interface::OperatorDualSplitting<Number> InterfacePDE;

  TimeIntBDFDualSplitting(std::shared_ptr<InterfaceBase> operator_base_in,
                          std::shared_ptr<InterfacePDE>  pde_operator_in,
                          InputParameters const &        param_in);

  void
  postprocessing_stability_analysis();

  void
  get_iterations(std::vector<std::string> & name, std::vector<double> & iteration) const;

  void
  get_wall_times(std::vector<std::string> & name, std::vector<double> & wall_time) const;

  void
  reinit_former_solution_considering_former_mesh_instances(
    std::vector<BlockVectorType> solution_in) override;

  void
  reinit_convective_term_considering_former_mesh_instances(
    std::vector<VectorType> convective_term_in) override;

  void
  reinit_vec_rhs_ppe_div_term_convective_term_considering_former_mesh_instances(
    std::vector<VectorType> vec_rhs_ppe_div_term_convective_term_in);

  void
  reinit_vec_rhs_ppe_convective_considering_former_mesh_instances(
    std::vector<VectorType> vec_rhs_ppe_convective_in);

  void
  reinit_vec_rhs_ppe_viscous_considering_former_mesh_instances(
    std::vector<VectorType> vec_rhs_ppe_viscous_in);

private:
  void
  setup_derived();

  void
  solve_timestep();

  void
  allocate_vectors();

  void
  prepare_vectors_for_next_timestep();

  void
  convective_step();

  void
  postprocessing() const;

  void
  postprocessing_steady_problem() const;

  void
  update_time_integrator_constants();

  void
  initialize_current_solution();

  void
  initialize_former_solutions();

  void
  initialize_vorticity();

  void
  initialize_vec_convective_term();

  void
  initialize_vec_rhs_ppe_div_term_convective_term();

  void
  initialize_vec_rhs_ppe_convective();

  void
  initialize_vec_rhs_ppe_viscous();

  void
  pressure_step();

  void
  rhs_pressure(VectorType & rhs) const;

  void
  projection_step();

  void
  rhs_projection(VectorType & rhs) const;

  void
  viscous_step();

  void
  rhs_viscous(VectorType & rhs) const;

  void
  solve_steady_problem();

  double
  evaluate_residual();

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

  std::vector<VectorType> velocity;

  VectorType velocity_np;

  std::vector<VectorType> pressure;

  VectorType pressure_np;

  std::vector<VectorType> vorticity;
  VectorType              vorticity_np;

  std::vector<VectorType> vec_convective_term;
  VectorType              convective_term_np;

  std::vector<VectorType> vec_rhs_ppe_div_term_convective_term;
  VectorType              rhs_ppe_div_term_convective_term_np;

  std::vector<VectorType> vec_rhs_ppe_convective;
  VectorType              rhs_ppe_convective_np;

  std::vector<VectorType> vec_rhs_ppe_viscous;
  VectorType              rhs_ppe_viscous_np;


  std::vector<double>       computing_times;
  std::vector<unsigned int> iterations;

  // time integrator constants: extrapolation scheme
  ExtrapolationConstants extra_pressure_nbc;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_DUAL_SPLITTING_H_ */
