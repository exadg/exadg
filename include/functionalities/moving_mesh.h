#ifndef INCLUDE_MOVING_MESH_H_
#define INCLUDE_MOVING_MESH_H_

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/lac/la_parallel_block_vector.h>

#include "mesh.h"

using namespace dealii;

#define MAPPING_Q_CACHE

template<int dim, typename Number>
class MovingMesh : public Mesh<dim>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  MovingMesh(unsigned int const                       mapping_degree_in,
             parallel::TriangulationBase<dim> const & triangulation,
             unsigned int const                       polynomial_degree,
             std::shared_ptr<Function<dim>> const     mesh_movement_function,
             double const                             start_time,
             MPI_Comm const &                         mpi_comm);

  Mapping<dim> const &
  get_mapping() const override;

  /*
   * This function initialized the MappingFEField object that is used to describe
   * a moving mesh.
   */
  void
  initialize_mapping_ale(double const time);

  /*
   * This function is formulated w.r.t. reference coordinates, i.e., the mapping describing
   * the initial mesh position has to be used for this function.
   */
  void
  move_mesh_analytical(double const time);

  /*
   * This function extracts the grid coordinates of the current mesh configuration, i.e.,
   * a mapping describing the mesh displacement has to be used here.
   */
  void
  fill_grid_coordinates_vector(VectorType & vector, DoFHandler<dim> const & dof_handler);

private:
  std::shared_ptr<Mapping<dim>> mapping_ale;

  // needed for re-initialization of MappingQCache
  parallel::TriangulationBase<dim> const & triangulation;

  unsigned int polynomial_degree;

  // An analytical function that describes the mesh movement
  std::shared_ptr<Function<dim>> mesh_movement_function;

  // MPI communciator
  MPI_Comm const & mpi_comm;

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
