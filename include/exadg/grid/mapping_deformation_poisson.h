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

#ifndef INCLUDE_EXADG_GRID_MAPPING_DEFORMATION_POISSON_H_
#define INCLUDE_EXADG_GRID_MAPPING_DEFORMATION_POISSON_H_

// deal.II
#include <deal.II/base/timer.h>

// ExaDG
#include <exadg/grid/mapping_deformation_base.h>
#include <exadg/poisson/spatial_discretization/operator.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace Poisson
{
/**
 * Class for moving grid problems based on a Poisson-type grid motion technique.
 *
 * TODO: extend this class to simplicial elements.
 */
template<int dim, typename Number>
class DeformedMapping : public DeformedMappingBase<dim, Number>
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  /**
   * Constructor.
   */
  DeformedMapping(
    std::shared_ptr<Grid<dim> const>                           grid,
    std::shared_ptr<dealii::Mapping<dim> const>                mapping_undeformed,
    std::shared_ptr<MultigridMappings<dim, Number>> const      multigrid_mappings_undeformed,
    std::shared_ptr<Poisson::BoundaryDescriptor<1, dim> const> boundary_descriptor,
    std::shared_ptr<Poisson::FieldFunctions<dim> const>        field_functions,
    Poisson::Parameters const &                                param,
    std::string const &                                        field,
    MPI_Comm const &                                           mpi_comm)
    : DeformedMappingBase<dim, Number>(mapping_undeformed, param.degree, *grid->triangulation),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
      iterations({0, 0})
  {
    // create and setup PDE operator
    pde_operator =
      std::make_shared<Poisson::Operator<dim, dim, Number>>(grid,
                                                            mapping_undeformed,
                                                            multigrid_mappings_undeformed,
                                                            boundary_descriptor,
                                                            field_functions,
                                                            param,
                                                            field,
                                                            mpi_comm);

    pde_operator->setup();

    // finally, initialize dof vector
    pde_operator->initialize_dof_vector(displacement);
  }

  std::shared_ptr<Poisson::Operator<dim, dim, Number> const>
  get_pde_operator() const
  {
    return pde_operator;
  }

  dealii::MatrixFree<dim, Number> const &
  get_matrix_free() const
  {
    return *pde_operator->get_matrix_free();
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

    this->initialize_mapping_from_dof_vector(this->mapping_undeformed,
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
  // PDE operator
  std::shared_ptr<Poisson::Operator<dim, dim, Number>> pde_operator;

  // store solution of previous time step / iteration so that a good initial
  // guess is available in the next step, easing convergence or reducing computational
  // costs by allowing larger tolerances
  VectorType displacement;

  dealii::ConditionalOStream pcout;

  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */> iterations;
};

} // namespace Poisson
} // namespace ExaDG

#endif /* INCLUDE_EXADG_GRID_MAPPING_DEFORMATION_POISSON_H_ */
