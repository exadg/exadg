#ifndef INCLUDE_MOVING_MESH_H_
#define INCLUDE_MOVING_MESH_H_

#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_block_vector.h>

#include "dg_dual_splitting.h"
#include "dg_navier_stokes_base.h"
#include "dg_pressure_correction.h"

using namespace dealii;

namespace IncNS
{
template<int dim, typename Number>
class MovingMesh
{
public:
  typedef MappingFEField<dim, dim, LinearAlgebra::distributed::Vector<Number>> MappingField;
  typedef LinearAlgebra::distributed::Vector<Number>                           VectorType;
  typedef LinearAlgebra::distributed::BlockVector<Number>                      BlockVectorType;

  MovingMesh(InputParameters const &                           param_in,
             std::shared_ptr<parallel::TriangulationBase<dim>> triangulation_in,
             std::shared_ptr<MeshMovementFunctions<dim>> const mesh_movement_function_in,
             std::shared_ptr<DGNavierStokesBase<dim, Number>>  navier_stokes_operation_in);

  void
  setup();

  void
  advance_grid_coordinates(double const time);

  void
  update_grid_velocities(const double              time_in,
                         const double              time_step_size,
                         const std::vector<Number> time_integrator_constants);

  void
  initialize_grid_coordinates_on_former_mesh_instances(std::vector<double> eval_times);

  std::vector<BlockVectorType>
  get_former_solution_on_former_mesh_instances(std::vector<double> eval_times);

  std::vector<VectorType>
  get_convective_term_on_former_mesh_instances(std::vector<double> eval_times);

  // Dual splitting
  std::vector<VectorType>
  get_vec_rhs_ppe_div_term_convective_term_on_former_mesh_instances(std::vector<double> eval_times);

  std::vector<VectorType>
  get_vec_rhs_ppe_convective_on_former_mesh_instances(std::vector<double> eval_times);

  std::vector<VectorType>
  get_vec_rhs_ppe_viscous_on_former_mesh_instances(std::vector<double> eval_times);

  // Pressure-correction
  std::vector<VectorType>
  get_vec_pressure_gradient_term_on_former_mesh_instances(std::vector<double> eval_times);

  std::vector<VectorType>
  get_pressure_mass_matrix_term_on_former_mesh_instances(std::vector<double> eval_times);

  VectorType
  get_grid_velocity() const;

private:
  void
  initialize_dof_handler();

  void
  initialize_vectors();

  void
  initialize_mapping_ale();

  Mapping<dim> &
  get_mapping() const;

  void
  compute_grid_velocity_analytical(double const time);

  void
  compute_grid_velocity_from_grid_coordinates(std::vector<Number> time_integrator_constants,
                                              double              time_step_size);

  void
  compute_bdf_time_derivative(VectorType &            dst,
                              std::vector<VectorType> src,
                              std::vector<Number>     time_integrator_constants,
                              double                  time_step_size);

  void
  fill_grid_coordinates_vector(int component = 0);


  InputParameters                                  param;
  std::shared_ptr<MeshMovementFunctions<dim>>      mesh_movement_function;
  std::shared_ptr<DGNavierStokesBase<dim, Number>> navier_stokes_operation;

  // fe systems
  std::shared_ptr<FESystem<dim>> fe_x_grid_continuous;
  std::shared_ptr<FESystem<dim>> fe_u_grid;

  // dof handlers
  DoFHandler<dim> dof_handler_x_grid_continuous;
  DoFHandler<dim> dof_handler_u_grid;
  DoFHandler<dim> dof_handler_x_grid_discontinuous;

  // vectors
  std::vector<VectorType>                                 vec_position_grid_new;
  LinearAlgebra::distributed::Vector<Number>              grid_velocity;
  std::vector<LinearAlgebra::distributed::Vector<Number>> vec_x_grid_discontinuous;

  // mappings
  std::shared_ptr<MappingQGeneric<dim>> mapping;
  std::shared_ptr<MappingField>         mapping_ale;
};

} // namespace IncNS
#endif /*INCLUDE_MOVING_MESH_H_*/
