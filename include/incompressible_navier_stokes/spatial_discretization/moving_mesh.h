#ifndef INCLUDE_MOVING_MESH_H_
#define INCLUDE_MOVING_MESH_H_

#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_block_vector.h>

#include "../../../applications/grid_tools/mesh_movement_functions.h"
#include "../user_interface/field_functions.h"
#include "dg_navier_stokes_base.h"

using namespace dealii;

namespace IncNS
{
template<int dim, typename Number>
class MovingMesh
{
public:
  typedef MappingFEField<dim, dim, LinearAlgebra::distributed::Vector<Number>> MappingField;
  typedef MappingQGeneric<dim>                                                 MappingQ;
  typedef LinearAlgebra::distributed::Vector<Number>                           VectorType;
  typedef LinearAlgebra::distributed::BlockVector<Number>                      BlockVectorType;

  MovingMesh(InputParameters const &                          param_in,
             std::shared_ptr<parallel::Triangulation<dim>>    triangulation_in,
             std::shared_ptr<FieldFunctions<dim>> const       field_functions_in,
             std::shared_ptr<DGNavierStokesBase<dim, Number>> navier_stokes_operation_in);

  void
  setup();

  void
  move_mesh(const double time_in);

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

  VectorType
  get_grid_velocity() const;

  double
  get_wall_time_ale_update() const;

  double
  get_wall_time_advance_mesh() const;

  double
  get_wall_time_compute_and_set_mesh_velocity() const;


private:
  void
  initialize_dof_handler();

  void
  initialize_vectors();

  void
  initialize_mapping_ale();

  Mapping<dim> const &
  get_mapping() const;

  void
  get_analytical_grid_velocity(double const evaluation_time);

  void
  advance_mesh(double time_in);

  /*
   * Two cases can occur:
   * The user wants to advance the mesh from the one of the last timestep:
   *  This is the case if a mesh moving algorithm is applied
   * The user wants to advance the mesh with an analytical mesh movement function and wants to use
   * the initial mesh as a reference Hence: MappingTypeIn can be MappingQ(Case2) or
   * MappingField(Case1)
   */
  template<class MappingTypeIn>
  void
  advance_position_grid_new_multigrid(MappingTypeIn & mapping_in);

  void
  compute_grid_velocity(std::vector<Number> time_integrator_constants, double time_step_size);

  void
  compute_bdf_time_derivative(VectorType &            dst,
                              std::vector<VectorType> src,
                              std::vector<Number>     time_integrator_constants,
                              double                  time_step_size);

  void
  fill_grid_coordinates_vector(int component = 0);


  InputParameters                                  param;
  std::shared_ptr<FieldFunctions<dim>>             field_functions;
  std::shared_ptr<DGNavierStokesBase<dim, Number>> navier_stokes_operation;

  // fe systems
  std::shared_ptr<FESystem<dim>> fe_grid;
  std::shared_ptr<FESystem<dim>> fe_u_grid;

  // dof handlers
  DoFHandler<dim> dof_handler_grid;
  DoFHandler<dim> dof_handler_u_grid;
  DoFHandler<dim> dof_handler_x_grid;

  // vectors
  std::vector<VectorType>                                 position_grid_new_multigrid;
  LinearAlgebra::distributed::Vector<Number>              u_grid_np;
  std::vector<LinearAlgebra::distributed::Vector<Number>> x_grid;

  // mappings
  std::shared_ptr<MappingQ>     mapping;
  std::shared_ptr<MappingField> mapping_ale;

  // timer
  Timer  timer_ale;
  double ale_update_timer;
  double advance_mesh_timer;
  double compute_and_set_mesh_velocity_timer;
  double help_timer;
};

} // namespace IncNS
#endif /*INCLUDE_MOVING_MESH_H_*/
