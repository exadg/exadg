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
  move_mesh(const double              time_in,
            const double              time_step_size,
            const std::vector<Number> time_integrator_constants);

  void
  init_d_grid_on_former_mesh(std::vector<double> eval_times);

  std::vector<BlockVectorType>
  init_former_solution_on_former_mesh(std::vector<double> eval_times);

  std::vector<VectorType>
  init_convective_term_on_former_mesh(std::vector<double> eval_times);

  VectorType
  get_grid_velocity();

  double
  get_wall_time_ALE_update();

  double
  get_wall_time_advance_mesh();

  double
  get_wall_time_compute_and_set_mesh_velocity();


private:
  void
  initialize_dof_handler();

  void
  initialize_vectors();

  Mapping<dim> const &
  get_mapping() const;

  void
  get_analytical_grid_velocity(double const evaluation_time);

  void
  advance_mesh(double time_in);

  void
  initialize_mapping_field();

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
  interpolate_mg(MappingTypeIn & mapping_in);

  void
  compute_grid_velocity(std::vector<Number> time_integrator_constants, double time_step_size);

  void
  compute_BDF_time_derivative(VectorType &            dst,
                              std::vector<VectorType> src,
                              std::vector<Number>     time_integrator_constants,
                              double                  time_step_size);

  void
  fill_d_grid(int component = 0);

  void
  initialize_ALE_update_data();


  InputParameters                                  param;
  std::shared_ptr<FieldFunctions<dim>>             field_functions;
  std::shared_ptr<DGNavierStokesBase<dim, Number>> navier_stokes_operation;

  // fe systems
  std::shared_ptr<FESystem<dim>> fe_grid;
  std::shared_ptr<FESystem<dim>> fe_u_grid;

  // dof handlers
  DoFHandler<dim> dof_handler_grid;
  DoFHandler<dim> dof_handler_u_grid;
  DoFHandler<dim> dof_handler_d_grid;

  // vectors
  std::vector<VectorType>                                 position_grid_new_multigrid;
  LinearAlgebra::distributed::Vector<Number>              u_grid_np;
  std::vector<LinearAlgebra::distributed::Vector<Number>> d_grid;

  // mappings
  std::shared_ptr<MappingField> mapping;
  std::shared_ptr<MappingQ>     mapping_init;


  // matrix_free update data:
  std::vector<Quadrature<1>> quadratures_ALE;
  AffineConstraints<double>  constraint_u_ALE, constraint_p_ALE, constraint_u_scalar_ALE;
  std::vector<const AffineConstraints<double> *>   constraint_matrix_vec_ALE;
  std::vector<const DoFHandler<dim> *>             dof_handler_vec_ALE;
  typename MatrixFree<dim, Number>::AdditionalData additional_data_ALE;
  UpdateFlags                                      ale_update_flags =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
     update_values | update_inverse_jacobians /*CFL*/);

  // timer
  Timer  timer_ale;
  double ALE_update_timer;
  double advance_mesh_timer;
  double compute_and_set_mesh_velocity_timer;
  double help_timer;
};

} // namespace IncNS
#endif /*INCLUDE_MOVING_MESH_H_*/
