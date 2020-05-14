/*
 * moving_mesh_elasticity.h
 *
 *  Created on: 13.05.2020
 *      Author: fehn
 */

#ifndef INCLUDE_GRID_MOVING_MESH_ELASTICITY_H_
#define INCLUDE_GRID_MOVING_MESH_ELASTICITY_H_

#include "moving_mesh_base.h"

#include "../structure/spatial_discretization/operator.h"

template<int dim, typename Number>
class MovingMeshElasticity : public MovingMeshBase<dim, Number>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  MovingMeshElasticity(unsigned int const                                mapping_degree_static,
                       MPI_Comm const &                                  mpi_comm,
                       std::shared_ptr<Structure::Operator<dim, Number>> structure_operator,
                       Structure::InputParameters const &                structure_parameters,
                       double const &                                    start_time)
    : MovingMeshBase<dim, Number>(mapping_degree_static,
                                  // extract mapping_degree_moving from elasticity operator
                                  structure_operator->get_dof_handler().get_fe().degree,
                                  mpi_comm),
      pde_operator(structure_operator),
      param(structure_parameters)
  {
    // make sure that the mapping is initialized
    move_mesh(start_time);
  }

  void
  move_mesh(double const time)
  {
    VectorType displacement;
    pde_operator->initialize_dof_vector(displacement);

    if(param.large_deformation) // nonlinear problem
    {
      VectorType const_vector;
      pde_operator->solve_nonlinear(
        displacement, const_vector, 0.0 /* no mass term */, time, param.update_preconditioner);
    }
    else // linear problem
    {
      // calculate right-hand side vector
      VectorType rhs;
      pde_operator->initialize_dof_vector(rhs);
      pde_operator->compute_rhs_linear(rhs, time);

      pde_operator->solve_linear(displacement, rhs, 0.0 /* no mass term */, time);
    }

    this->initialize_mapping_q_cache(*this->mapping, pde_operator->get_dof_handler(), displacement);
  }

private:
  std::shared_ptr<Structure::Operator<dim, Number>> pde_operator;

  Structure::InputParameters const & param;
};


#endif /* INCLUDE_GRID_MOVING_MESH_ELASTICITY_H_ */
