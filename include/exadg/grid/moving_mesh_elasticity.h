/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_EXADG_GRID_MOVING_MESH_ELASTICITY_H_
#define INCLUDE_EXADG_GRID_MOVING_MESH_ELASTICITY_H_

// deal.II
#include <deal.II/base/timer.h>

// ExaDG
#include <exadg/grid/moving_mesh_base.h>
#include <exadg/structure/spatial_discretization/operator.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, typename Number>
class MovingMeshElasticity : public MovingMeshBase<dim, Number>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  MovingMeshElasticity(std::shared_ptr<Mapping<dim>>                     mapping,
                       MPI_Comm const &                                  mpi_comm,
                       bool const                                        print_wall_times,
                       std::shared_ptr<Structure::Operator<dim, Number>> structure_operator,
                       Structure::InputParameters const &                structure_parameters)
    : MovingMeshBase<dim, Number>(mapping,
                                  // extract mapping_degree_moving from elasticity operator
                                  structure_operator->get_dof_handler().get_fe().degree,
                                  structure_operator->get_dof_handler().get_triangulation(),
                                  mpi_comm),
      pde_operator(structure_operator),
      param(structure_parameters),
      pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0),
      iterations({0, {0, 0}}),
      print_wall_times(print_wall_times)
  {
    // make sure that the mapping is initialized
    pde_operator->initialize_dof_vector(displacement);
    this->initialize(pde_operator->get_dof_handler(), displacement);
  }

  void
  update(double const time, bool const print_solver_info = false)
  {
    Timer timer;
    timer.restart();

    if(param.large_deformation) // nonlinear problem
    {
      VectorType const_vector;

      auto const iter = pde_operator->solve_nonlinear(
        displacement, const_vector, 0.0 /* no mass term */, time, param.update_preconditioner);

      iterations.first += 1;
      std::get<0>(iterations.second) += std::get<0>(iter);
      std::get<1>(iterations.second) += std::get<1>(iter);

      if(print_solver_info)
      {
        this->pcout << std::endl << "Solve moving mesh problem (nonlinear elasticity):";
        print_solver_info_nonlinear(
          pcout, std::get<0>(iter), std::get<1>(iter), timer.wall_time(), print_wall_times);
      }
    }
    else // linear problem
    {
      // calculate right-hand side vector
      VectorType rhs;
      pde_operator->initialize_dof_vector(rhs);
      pde_operator->compute_rhs_linear(rhs, time);

      auto const iter = pde_operator->solve_linear(displacement, rhs, 0.0 /* no mass term */, time);

      iterations.first += 1;
      std::get<1>(iterations.second) += iter;

      if(print_solver_info)
      {
        this->pcout << std::endl << "Solve moving mesh problem (linear elasticity):";
        print_solver_info_linear(pcout, iter, timer.wall_time(), print_wall_times);
      }
    }

    this->initialize(pde_operator->get_dof_handler(), displacement);
  }

  void
  print_iterations() const
  {
    std::vector<std::string> names;
    std::vector<double>      iterations_avg;

    if(param.large_deformation)
    {
      names = {"Nonlinear iterations",
               "Linear iterations (accumulated)",
               "Linear iterations (per nonlinear it.)"};

      iterations_avg.resize(3);
      iterations_avg[0] =
        (double)std::get<0>(iterations.second) / std::max(1., (double)iterations.first);
      iterations_avg[1] =
        (double)std::get<1>(iterations.second) / std::max(1., (double)iterations.first);
      if(iterations_avg[0] > std::numeric_limits<double>::min())
        iterations_avg[2] = iterations_avg[1] / iterations_avg[0];
      else
        iterations_avg[2] = iterations_avg[1];
    }
    else // linear
    {
      names = {"Linear iterations"};
      iterations_avg.resize(1);
      iterations_avg[0] =
        (double)std::get<1>(iterations.second) / std::max(1., (double)iterations.first);
    }

    print_list_of_iterations(pcout, names, iterations_avg);
  }

private:
  std::shared_ptr<Structure::Operator<dim, Number>> pde_operator;

  Structure::InputParameters const & param;

  // store solution of previous time step / iteration so that a good initial
  // guess is available in the next step, easing convergence or reducing computational
  // costs by allowing larger tolerances
  VectorType displacement;

  ConditionalOStream pcout;

  std::pair<
    unsigned int /* calls */,
    std::tuple<unsigned long long, unsigned long long> /* iteration counts {Newton, linear}*/>
    iterations;

  bool print_wall_times;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_GRID_MOVING_MESH_ELASTICITY_H_ */
