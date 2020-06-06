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
#include "../utilities/print_throughput.h"

template<int dim, typename Number>
class MovingMeshPoisson : public MovingMeshBase<dim, Number>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  MovingMeshPoisson(unsigned int const                                   mapping_degree_static,
                    MPI_Comm const &                                     mpi_comm,
                    std::shared_ptr<Poisson::Operator<dim, Number, dim>> poisson_operator)
    : MovingMeshBase<dim, Number>(mapping_degree_static,
                                  // extract mapping_degree_moving from Poisson operator
                                  poisson_operator->get_dof_handler().get_fe().degree,
                                  mpi_comm),
      poisson(poisson_operator),
      pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0),
      iterations({0, 0})
  {
    poisson->initialize_dof_vector(displacement);

    // make sure that the mapping is initialized
    this->initialize_mapping_q_cache(*this->mapping, poisson->get_dof_handler(), displacement);
  }

  void
  move_mesh(double const time, bool const print_solver_info = false)
  {
    Timer timer;
    timer.restart();

    VectorType rhs;
    poisson->initialize_dof_vector(rhs);

    // compute rhs and solve mesh deformation problem
    poisson->rhs(rhs, time);

    auto const n_iter = poisson->solve(displacement, rhs, time);
    iterations.first += 1;
    iterations.second += n_iter;

    if(print_solver_info)
    {
      this->pcout << std::endl << "Solve moving mesh problem (Poisson):";
      print_solver_info_linear(pcout, n_iter, timer.wall_time());
    }

    this->initialize_mapping_q_cache(*this->mapping, poisson->get_dof_handler(), displacement);
  }

  void
  print_iterations() const
  {
    std::vector<std::string> names;
    std::vector<double>      iterations_avg;

    names = {"Linear iterations"};
    iterations_avg.resize(1);
    iterations_avg[0] = (double)iterations.second / std::max(1., (double)iterations.first);

    print_list_of_iterations(pcout, names, iterations_avg);
  }

private:
  std::shared_ptr<Poisson::Operator<dim, Number, dim>> poisson;

  // store solution of previous time step / iteration so that a good initial
  // guess is available in the next step, easing convergence or reducing computational
  // costs by allowing larger tolerances
  VectorType displacement;

  ConditionalOStream pcout;

  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */> iterations;
};


#endif /* INCLUDE_GRID_MOVING_MESH_POISSON_H_ */
