/*
 * time_int_bdf.h
 *
 *  Created on: Aug 22, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_TIME_INT_BDF_H_
#define INCLUDE_CONVECTION_DIFFUSION_TIME_INT_BDF_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>

#include "time_integration/explicit_runge_kutta.h"
#include "time_integration/time_int_bdf_base.h"

using namespace dealii;

// forward declarations
template<int dim, typename Number>
class MovingMeshBase;

namespace ConvDiff
{
class InputParameters;

template<int dim, typename Number>
class DGOperator;

template<typename Number>
class PostProcessorInterface;

namespace Interface
{
template<typename Number>
class OperatorOIF;
} // namespace Interface
} // namespace ConvDiff


namespace ConvDiff
{
template<int dim, typename Number>
class TimeIntBDF : public TimeIntBDFBase<Number>
{
public:
  typedef typename TimeIntBDFBase<Number>::VectorType VectorType;

  typedef DGOperator<dim, Number> Operator;

  TimeIntBDF(std::shared_ptr<Operator>                       operator_in,
             InputParameters const &                         param_in,
             unsigned int const                              refine_steps_time_in,
             MPI_Comm const &                                mpi_comm_in,
             std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in,
             std::shared_ptr<MovingMeshBase<dim, Number>>    moving_mesh_in = nullptr,
             std::shared_ptr<MatrixFree<dim, Number>>        matrix_free_in = nullptr);

  void
  set_velocities_and_times(std::vector<VectorType const *> const & velocities_in,
                           std::vector<double> const &             times_in);

  void
  extrapolate_solution(VectorType & vector);

  void
  ale_update();

  void
  print_iterations() const;

private:
  void
  allocate_vectors();

  void
  initialize_current_solution();

  void
  initialize_former_solutions();

  void
  initialize_vec_convective_term();

  double
  calculate_time_step_size();

  double
  recalculate_time_step_size() const;

  void
  prepare_vectors_for_next_timestep();

  void
  move_mesh(double const time) const;

  void
  move_mesh_and_update_dependent_data_structures(double const time) const;

  void
  solve_timestep();

  void
  initialize_oif();

  void
  setup_derived();

  void
  calculate_sum_alphai_ui_oif_substepping(VectorType & sum_alphai_ui,
                                          double const cfl,
                                          double const cfl_oif);

  void
  initialize_solution_oif_substepping(VectorType & solution_tilde_m, unsigned int i);

  void
  update_sum_alphai_ui_oif_substepping(VectorType &       sum_alphai_ui,
                                       VectorType const & u_tilde_i,
                                       unsigned int       i);

  void
  do_timestep_oif_substepping(VectorType & solution_tilde_mp,
                              VectorType & solution_tilde_m,
                              double const start_time,
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

  unsigned int const refine_steps_time;

  double const cfl;

  // solution vectors
  VectorType              solution_np;
  std::vector<VectorType> solution;
  std::vector<VectorType> vec_convective_term;
  VectorType              convective_term_np;

  VectorType rhs_vector;

  // numerical velocity field
  std::vector<VectorType const *> velocities;
  std::vector<double>             times;

  // iteration counts and solver time
  double iterations;

  // Operator-integration-factor (OIF) splitting

  // cfl number for OIF splitting
  double const cfl_oif;

  std::shared_ptr<Interface::OperatorOIF<Number>> convective_operator_OIF;

  std::shared_ptr<ExplicitTimeIntegrator<Interface::OperatorOIF<Number>, VectorType>>
    time_integrator_OIF;

  std::shared_ptr<PostProcessorInterface<Number>> postprocessor;

  // ALE
  VectorType              grid_velocity;
  std::vector<VectorType> vec_grid_coordinates;
  VectorType              grid_coordinates_np;

  std::shared_ptr<MovingMeshBase<dim, Number>> moving_mesh;
  std::shared_ptr<MatrixFree<dim, Number>>     matrix_free;
};

} // namespace ConvDiff

#endif /* INCLUDE_CONVECTION_DIFFUSION_TIME_INT_BDF_H_ */
