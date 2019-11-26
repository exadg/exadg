#ifndef INCLUDE_MOVING_MESH_H_
#define INCLUDE_MOVING_MESH_H_

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/la_parallel_block_vector.h>

using namespace dealii;

#define MAPPING_Q_CACHE

template<int dim, typename Number>
class MovingMesh
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  MovingMesh(parallel::TriangulationBase<dim> const & triangulation,
             unsigned int const                       polynomial_degree,
             std::shared_ptr<Function<dim>> const     mesh_movement_function);

  /*
   * This function initialized the MappingFEField object that is used to describe
   * a moving mesh.
   */
  std::shared_ptr<Mapping<dim>>
  initialize_mapping_ale(double const time, Mapping<dim> const & mapping);

  /*
   * This function is formulated w.r.t. reference coordinates, i.e., the mapping describing
   * the initial mesh position has to be used for this function.
   */
  void
  move_mesh_analytical(double const                  time,
                       Mapping<dim> const &          mapping,
                       std::shared_ptr<Mapping<dim>> mapping_ale);

  /*
   * This function extracts the grid coordinates of the current mesh configuration, i.e.,
   * a mapping describing the mesh displacement has to be used here.
   */
  void
  fill_grid_coordinates_vector(VectorType &            vector,
                               DoFHandler<dim> const & dof_handler,
                               Mapping<dim> const &    mapping);

private:
  unsigned int polynomial_degree;

  // needed for re-initialization of MappingQCache
  parallel::TriangulationBase<dim> const & triangulation;

  // An analytical function that describes the mesh movement
  std::shared_ptr<Function<dim>> mesh_movement_function;

  // Finite Element (use a continuous finite element space to describe the mesh movement)
  std::shared_ptr<FESystem<dim>> fe;

#ifndef MAPPING_Q_CACHE
  // vectors with grid coordinates for all multigrid levels
  std::vector<VectorType> grid_coordinates;
#endif

  // DoFHandler
  DoFHandler<dim> dof_handler;
};

#endif /*INCLUDE_MOVING_MESH_H_*/
