/*
 * time_int_bdf.h
 *
 *  Created on: Aug 22, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_TIME_INT_BDF_H_
#define INCLUDE_CONVECTION_DIFFUSION_TIME_INT_BDF_H_

#include <deal.II/lac/la_parallel_vector.h>

#include "time_integration/explicit_runge_kutta.h"
#include "time_integration/time_int_bdf_base.h"

using namespace dealii;

namespace ConvDiff
{
// forward declarations
class InputParameters;

namespace Interface
{
template<typename Number>
class Operator;
template<typename Number>
class OperatorOIF;
} // namespace Interface

template<typename Number>
class TimeIntBDF : public TimeIntBDFBase
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef Interface::Operator<Number> Operator;

  TimeIntBDF(std::shared_ptr<Operator> operator_in,
             InputParameters const &   param_in,
             unsigned int const        n_refine_time_in);

  void
  get_iterations(std::vector<std::string> & name, std::vector<double> & iteration) const;

  void
  get_wall_times(std::vector<std::string> & name, std::vector<double> & wall_time) const;

private:
  void
  allocate_vectors();

  void
  initialize_current_solution();

  void
  initialize_former_solutions();

  void
  initialize_vec_convective_term();

  void
  calculate_time_step_size();

  double
  recalculate_time_step_size() const;

  void
  prepare_vectors_for_next_timestep();

  void
  solve_timestep();

  void
  initialize_oif();

  void
  setup_derived();

  void
  initialize_solution_oif_substepping(unsigned int i);

  void
  update_sum_alphai_ui_oif_substepping(unsigned int i);

  void
  do_timestep_oif_substepping_and_update_vectors(double const start_time,
                                                 double const time_step_size);

  bool
  print_solver_info() const;

  void
  read_restart_vectors(boost::archive::binary_iarchive & ia);

  void
  write_restart_vectors(boost::archive::binary_oarchive & oa) const;

  void
  postprocessing() const;

  std::shared_ptr<Operator> pde_operator;

  InputParameters const & param;

  unsigned int const n_refine_time;
  double const       cfl;

  // solution vectors
  VectorType              solution_np;
  std::vector<VectorType> solution;
  std::vector<VectorType> vec_convective_term;

  VectorType sum_alphai_ui;
  VectorType rhs_vector;

  // iteration counts and solver time
  double iterations;
  double wall_time;

  // Operator-integration-factor (OIF) splitting

  // cfl number for OIF splitting
  double const cfl_oif;

  std::shared_ptr<Interface::OperatorOIF<Number>> convective_operator_OIF;

  std::shared_ptr<ExplicitTimeIntegrator<Interface::OperatorOIF<Number>, VectorType>>
    time_integrator_OIF;

  // solution vectors needed for OIF substepping of convective term
  VectorType solution_tilde_m;
  VectorType solution_tilde_mp;
};

} // namespace ConvDiff

#endif /* INCLUDE_CONVECTION_DIFFUSION_TIME_INT_BDF_H_ */
