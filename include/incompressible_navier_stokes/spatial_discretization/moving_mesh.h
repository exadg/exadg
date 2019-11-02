#ifndef INCLUDE_MOVING_MESH_H_
#define INCLUDE_MOVING_MESH_H_

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/la_parallel_block_vector.h>

#include "../../../applications/grid_tools/mesh_movement_functions.h"

using namespace dealii;

namespace IncNS
{
template<int dim, typename Number>
class MovingMesh
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  MovingMesh(parallel::TriangulationBase<dim> const &          triangulation,
             unsigned int const                                polynomial_degree,
             std::shared_ptr<MeshMovementFunctions<dim>> const mesh_movement_function);

  /*
   * This function initialized the MappingFEField object that is used to describe
   * a moving mesh.
   */
  std::shared_ptr<MappingFEField<dim, dim, VectorType>>
  initialize_mapping_fe_field(double const time, Mapping<dim> const & mapping);

  /*
   * This function is formulated w.r.t. reference coordinates, i.e., the mapping describing
   * the initial mesh position has to be used for this function.
   */
  void
  move_mesh_analytical(double const time, Mapping<dim> const & mapping);

  /*
   * This function is formulated w.r.t. reference coordinates, i.e., the mapping describing
   * the initial mesh position has to be used for this function.
   */
  void
  compute_grid_velocity_analytical(VectorType &            velocity,
                                   double const            time,
                                   DoFHandler<dim> const & dof_handler,
                                   Mapping<dim> const &    mapping);

  /*
   * This function extracts the grid coordinates of the current mesh configuration, i.e.,
   * a mapping describing the mesh displacement has to be used here.
   */
  void
  fill_grid_coordinates_vector(VectorType &            vector,
                               DoFHandler<dim> const & dof_handler,
                               Mapping<dim> const &    mapping);

private:
  // An analytical function that describes the mesh movement
  std::shared_ptr<MeshMovementFunctions<dim>> mesh_movement_function;

  // Finite Element (use a continuous finite element space to describe the mesh movement)
  std::shared_ptr<FESystem<dim>> fe;

  // DoFHandler
  DoFHandler<dim> dof_handler;

  // vectors with grid coordinates for all multigrid levels
  std::vector<VectorType> grid_coordinates;
};

} // namespace IncNS
#endif /*INCLUDE_MOVING_MESH_H_*/
