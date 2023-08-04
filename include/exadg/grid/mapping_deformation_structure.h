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

#ifndef INCLUDE_EXADG_GRID_MAPPING_DEFORMATION_STRUCTURE_H_
#define INCLUDE_EXADG_GRID_MAPPING_DEFORMATION_STRUCTURE_H_

// deal.II
#include <deal.II/base/timer.h>

// ExaDG
#include <exadg/grid/mapping_deformation_base.h>
#include <exadg/structure/spatial_discretization/operator.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace Structure
{
/**
 * Class for moving grid problems based on a pseudo-solid grid motion technique.
 */
template<int dim, typename Number>
class DeformedMapping : public DeformedMappingBase<dim, Number>
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  /**
   * Constructor.
   */
  DeformedMapping(std::shared_ptr<Grid<dim> const>               grid,
                  std::shared_ptr<dealii::Mapping<dim> const>    mapping_undeformed,
                  std::shared_ptr<BoundaryDescriptor<dim> const> boundary_descriptor,
                  std::shared_ptr<FieldFunctions<dim> const>     field_functions,
                  std::shared_ptr<MaterialDescriptor const>      material_descriptor,
                  Parameters const &                             param,
                  std::string const &                            field,
                  MPI_Comm const &                               mpi_comm)
    : DeformedMappingBase<dim, Number>(mapping_undeformed, param.degree, *grid->triangulation),
      param(param),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
      iterations({0, {0, 0}})
  {
    // initialize PDE operator
    pde_operator = std::make_shared<Operator<dim, Number>>(grid,
                                                           mapping_undeformed,
                                                           boundary_descriptor,
                                                           field_functions,
                                                           material_descriptor,
                                                           param,
                                                           field,
                                                           mpi_comm);

    // ALE: initialize matrix_free_data
    matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();

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
    pde_operator->setup_solver(0.0 /* no acceleration term */, 0.0 /* no damping term */);

    // finally, initialize dof vector
    pde_operator->initialize_dof_vector(displacement);
  }

  std::shared_ptr<Operator<dim, Number> const>
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
   * Updates the mapping, i.e., moves the grid by solving a pseudo-solid problem.
   */
  void
  update(double const     time,
         bool const       print_solver_info,
         types::time_step time_step_number) override
  {
    dealii::Timer timer;
    timer.restart();

    if(param.large_deformation) // nonlinear problem
    {
      VectorType const_vector;

      bool const update_preconditioner =
        this->param.update_preconditioner &&
        time_step_number % this->param.update_preconditioner_every_time_steps == 0;

      auto const iter = pde_operator->solve_nonlinear(displacement,
                                                      const_vector,
                                                      0.0 /* no acceleration term */,
                                                      0.0 /* no damping term */,
                                                      time,
                                                      update_preconditioner);

      iterations.first += 1;
      std::get<0>(iterations.second) += std::get<0>(iter);
      std::get<1>(iterations.second) += std::get<1>(iter);

      if(print_solver_info)
      {
        this->pcout << std::endl << "Solve moving mesh problem (nonlinear elasticity):";
        print_solver_info_nonlinear(pcout, std::get<0>(iter), std::get<1>(iter), timer.wall_time());
      }
    }
    else // linear problem
    {
      // calculate right-hand side vector
      VectorType rhs;
      pde_operator->initialize_dof_vector(rhs);
      pde_operator->rhs(rhs, time);

      auto const iter = pde_operator->solve_linear(displacement,
                                                   rhs,
                                                   0.0 /* no acceleration term */,
                                                   0.0 /* no damping term */,
                                                   time,
                                                   false /* do not update preconditioner */);

      iterations.first += 1;
      std::get<1>(iterations.second) += iter;

      if(print_solver_info)
      {
        this->pcout << std::endl << "Solve moving mesh problem (linear elasticity):";
        print_solver_info_linear(pcout, iter, timer.wall_time());
      }
    }

    this->initialize_mapping_q_cache(this->mapping_undeformed,
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
  // matrix-free
  std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data;
  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free;

  // PDE operator
  std::shared_ptr<Operator<dim, Number>> pde_operator;

  Parameters const & param;

  // store solution of previous time step / iteration so that a good initial
  // guess is available in the next step, easing convergence or reducing computational
  // costs by allowing larger tolerances
  VectorType displacement;

  dealii::ConditionalOStream pcout;

  std::pair<
    unsigned int /* calls */,
    std::tuple<unsigned long long, unsigned long long> /* iteration counts {Newton, linear}*/>
    iterations;
};

} // namespace Structure
} // namespace ExaDG

#endif /* INCLUDE_EXADG_GRID_MAPPING_DEFORMATION_STRUCTURE_H_ */
