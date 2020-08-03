#ifndef INCLUDE_MOVING_MESH_H_
#define INCLUDE_MOVING_MESH_H_

#include "moving_mesh_base.h"

using namespace dealii;

template<int dim, typename Number>
class MovingMeshFunction : public MovingMeshBase<dim, Number>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  MovingMeshFunction(parallel::TriangulationBase<dim> const & triangulation_in,
                     unsigned int const                       mapping_degree_static_in,
                     unsigned int const                       mapping_degree_moving_in,
                     MPI_Comm const &                         mpi_comm_in,
                     std::shared_ptr<Function<dim>> const     mesh_movement_function_in,
                     double const                             start_time)
    : MovingMeshBase<dim, Number>(mapping_degree_static_in, mapping_degree_moving_in, mpi_comm_in),
      mesh_movement_function(mesh_movement_function_in),
      triangulation(triangulation_in)
  {
    move_mesh(start_time);
  }

  /*
   * This function is formulated w.r.t. reference coordinates, i.e., the mapping describing
   * the initial mesh position has to be used for this function.
   */
  void
  move_mesh(double const time, bool const print_solver_info = false)
  {
    (void)print_solver_info;

    mesh_movement_function->set_time(time);

    this->initialize_mapping_q_cache(*this->mapping, triangulation, mesh_movement_function);
  }

private:
  std::shared_ptr<Function<dim>> mesh_movement_function;

  Triangulation<dim> const & triangulation;
};

#endif /*INCLUDE_MOVING_MESH_H_*/
