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

#ifndef INCLUDE_EXADG_GRID_GRID_MOTION_POISSON_H_
#define INCLUDE_EXADG_GRID_GRID_MOTION_POISSON_H_

// deal.II
#include <deal.II/base/timer.h>

// ExaDG
#include <exadg/grid/grid_motion_base.h>
#include <exadg/poisson/spatial_discretization/operator.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
/**
 * Class for moving grid problems based on a Poisson-type grid motion technique.
 */
template<int dim, typename Number>
class GridMotionPoisson : public GridMotionBase<dim, Number>
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  /**
   * Constructor.
   */
  GridMotionPoisson(std::shared_ptr<Grid<dim> const>                           grid,
                    std::shared_ptr<dealii::Mapping<dim> const>                mapping_undeformed,
                    std::shared_ptr<Poisson::BoundaryDescriptor<1, dim> const> boundary_descriptor,
                    std::shared_ptr<Poisson::FieldFunctions<dim> const>        field_functions,
                    Poisson::Parameters const &                                param,
                    std::string const &                                        field,
                    MPI_Comm const &                                           mpi_comm)
    : GridMotionBase<dim, Number>(mapping_undeformed, param.degree, *grid->triangulation),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
      iterations({0, 0})
  {
    // initialize PDE operator
    pde_operator = std::make_shared<Poisson::Operator<dim, dim, Number>>(
      grid, mapping_undeformed, boundary_descriptor, field_functions, param, field, mpi_comm);

    // initialize matrix_free_data
    matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();

    if(param.enable_cell_based_face_loops)
      Categorization::do_cell_based_loops(*grid->triangulation, matrix_free_data->data);

    matrix_free_data->append(pde_operator);

    // initialize matrix_free
    matrix_free = std::make_shared<dealii::MatrixFree<dim, Number>>();
    matrix_free->reinit(*mapping_undeformed,
                        matrix_free_data->get_dof_handler_vector(),
                        matrix_free_data->get_constraint_vector(),
                        matrix_free_data->get_quadrature_vector(),
                        matrix_free_data->data);

    // setup PDE operator and solver
    pde_operator->setup(matrix_free, matrix_free_data);
    pde_operator->setup_solver();

    // finally, initialize dof vector
    pde_operator->initialize_dof_vector(displacement);
  }

  std::shared_ptr<Poisson::Operator<dim, dim, Number> const>
  get_pde_operator() const
  {
    return pde_operator;
  }

  std::shared_ptr<dealii::MatrixFree<dim, Number> const>
  get_matrix_free() const
  {
    return matrix_free;
  }

  /**
   * Updates the mapping, i.e., moves the mesh by solving a Poisson-type problem.
   */
  void
  update(double const     time,
         bool const       print_solver_info,
         types::time_step time_step_number) override
  {
    // preconditioner update has no effect
    (void)time_step_number;

    dealii::Timer timer;
    timer.restart();

    VectorType rhs;
    pde_operator->initialize_dof_vector(rhs);

    // compute rhs and solve mesh deformation problem
    pde_operator->rhs(rhs, time);

    auto const n_iter = pde_operator->solve(displacement, rhs, time);
    iterations.first += 1;
    iterations.second += n_iter;

    if(print_solver_info)
    {
      this->pcout << std::endl << "Solve moving mesh problem (Poisson):";
      print_solver_info_linear(pcout, n_iter, timer.wall_time());
    }

    this->moving_mapping->initialize_mapping_q_cache(this->mapping_undeformed,
                                                     displacement,
                                                     pde_operator->get_dof_handler());
  }

  /**
   * Prints information on iteration counts.
   */
  void
  print_iterations() const override
  {
    std::vector<std::string> names;
    std::vector<double>      iterations_avg;

    names = {"Linear iterations"};
    iterations_avg.resize(1);
    iterations_avg[0] = (double)iterations.second / std::max(1., (double)iterations.first);

    print_list_of_iterations(pcout, names, iterations_avg);
  }

private:
  // matrix-free
  std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data;
  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free;

  // PDE operator
  std::shared_ptr<Poisson::Operator<dim, dim, Number>> pde_operator;

  // store solution of previous time step / iteration so that a good initial
  // guess is available in the next step, easing convergence or reducing computational
  // costs by allowing larger tolerances
  VectorType displacement;

  dealii::ConditionalOStream pcout;

  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */> iterations;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_GRID_GRID_MOTION_POISSON_H_ */
