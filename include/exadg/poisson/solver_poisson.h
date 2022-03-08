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

#ifndef INCLUDE_EXADG_POISSON_SOLVER_POISSON_H_
#define INCLUDE_EXADG_POISSON_SOLVER_POISSON_H_

// ExaDG
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/poisson/spatial_discretization/operator.h>
#include <exadg/poisson/user_interface/application_base.h>

namespace ExaDG
{
namespace Poisson
{
template<int dim, int n_components, typename Number>
class SolverPoisson
{
public:
  void
  setup(std::shared_ptr<ApplicationBase<dim, n_components, Number>> application,
        MPI_Comm const                                              mpi_comm,
        bool const                                                  is_throughput_study)
  {
    pde_operator =
      std::make_shared<Operator<dim, Number, n_components>>(application->get_grid(),
                                                            application->get_boundary_descriptor(),
                                                            application->get_field_functions(),
                                                            application->get_parameters(),
                                                            "Poisson",
                                                            mpi_comm);

    matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();
    matrix_free_data->append(pde_operator);

    matrix_free = std::make_shared<dealii::MatrixFree<dim, Number>>();
    if(application->get_parameters().enable_cell_based_face_loops)
      Categorization::do_cell_based_loops(*application->get_grid()->triangulation,
                                          matrix_free_data->data);
    matrix_free->reinit(*application->get_grid()->mapping,
                        matrix_free_data->get_dof_handler_vector(),
                        matrix_free_data->get_constraint_vector(),
                        matrix_free_data->get_quadrature_vector(),
                        matrix_free_data->data);

    pde_operator->setup(matrix_free, matrix_free_data);

    if(not(is_throughput_study))
    {
      pde_operator->setup_solver();

      postprocessor = application->create_postprocessor();
      postprocessor->setup(pde_operator->get_dof_handler(), *application->get_grid()->mapping);
    }
  }

  std::shared_ptr<Operator<dim, Number, n_components>> pde_operator;
  std::shared_ptr<PostProcessorBase<dim, Number>>      postprocessor;

private:
  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free;
  std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data;
};

} // namespace Poisson
} // namespace ExaDG



#endif /* INCLUDE_EXADG_POISSON_SOLVER_POISSON_H_ */
