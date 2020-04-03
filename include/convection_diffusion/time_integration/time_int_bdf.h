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

#include "../../functionalities/matrix_free_wrapper.h"
#include "../../functionalities/moving_mesh.h"
#include "../postprocessor/postprocessor_base.h"

#include "../spatial_discretization/dg_operator.h"

using namespace dealii;

namespace ConvDiff
{
// forward declarations
class InputParameters;

namespace Interface
{
template<typename Number>
class OperatorOIF;
} // namespace Interface

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
             std::shared_ptr<MovingMeshBase<dim, Number>>    moving_mesh_in         = nullptr,
             std::shared_ptr<MatrixFreeWrapper<dim, Number>> matrix_free_wrapper_in = nullptr);

  void
  get_iterations(std::vector<std::string> & name, std::vector<double> & iteration) const;

  void
  get_wall_times(std::vector<std::string> & name, std::vector<double> & wall_time) const;

  void
  set_velocities_and_times(std::vector<VectorType const *> const & velocities_in,
                           std::vector<double> const &             times_in);

  void
  extrapolate_solution(VectorType & vector);

  void
  ale_update();

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
  double wall_time;

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

  std::shared_ptr<MovingMeshBase<dim, Number>>    moving_mesh;
  std::shared_ptr<MatrixFreeWrapper<dim, Number>> matrix_free_wrapper;
};

} // namespace ConvDiff

#endif /* INCLUDE_CONVECTION_DIFFUSION_TIME_INT_BDF_H_ */
