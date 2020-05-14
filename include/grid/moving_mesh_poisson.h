/*
 * moving_mesh_poisson.h
 *
 *  Created on: 13.05.2020
 *      Author: fehn
 */

#ifndef INCLUDE_GRID_MOVING_MESH_POISSON_H_
#define INCLUDE_GRID_MOVING_MESH_POISSON_H_

#include "moving_mesh_base.h"

#include "../poisson/spatial_discretization/operator.h"

template<int dim, typename Number>
class MovingMeshPoisson : public MovingMeshBase<dim, Number>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  MovingMeshPoisson(unsigned int const                                   mapping_degree_static,
                    MPI_Comm const &                                     mpi_comm,
                    std::shared_ptr<Poisson::Operator<dim, Number, dim>> poisson_operator,
                    double const &                                       start_time)
    : MovingMeshBase<dim, Number>(mapping_degree_static,
                                  // extract mapping_degree_moving from Poisson operator
                                  poisson_operator->get_dof_handler().get_fe().degree,
                                  mpi_comm),
      poisson(poisson_operator)
  {
    // make sure that the mapping is initialized
    move_mesh(start_time);
  }

  void
  move_mesh(double const time)
  {
    VectorType displacement, rhs;
    poisson->initialize_dof_vector(displacement);
    poisson->initialize_dof_vector(rhs);

    // compute rhs and solve mesh deformation problem
    poisson->rhs(rhs, time);
    poisson->solve(displacement, rhs, time);

    this->initialize_mapping_q_cache(*this->mapping, poisson->get_dof_handler(), displacement);
  }

private:
  std::shared_ptr<Poisson::Operator<dim, Number, dim>> poisson;
};


#endif /* INCLUDE_GRID_MOVING_MESH_POISSON_H_ */
