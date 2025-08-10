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

#ifndef EXADG_SOLVERS_AND_PRECONDITIONERS_UTILITIES_CHECK_MULTIGRID_H_
#define EXADG_SOLVERS_AND_PRECONDITIONERS_UTILITIES_CHECK_MULTIGRID_H_

// C/C++
#include <fstream>

// deal.II
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/numerics/data_out.h>

// ExaDG
#include <exadg/postprocessor/output_data_base.h>
#include <exadg/postprocessor/write_output.h>

namespace ExaDG
{
template<int dim, typename Number, typename Operator, typename Preconditioner>
class CheckMultigrid
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::LinearAlgebra::distributed::Vector<typename Preconditioner::MultigridNumber>
    VectorTypeMG;

  CheckMultigrid(Operator const &                underlying_operator_in,
                 std::shared_ptr<Preconditioner> preconditioner_in,
                 MPI_Comm const &                mpi_comm_in)
    : underlying_operator(underlying_operator_in),
      preconditioner(preconditioner_in),
      mpi_comm(mpi_comm_in)
  {
  }

  /*
   *  Function that verifies the multigrid algorithm,
   *  especially the smoothing (on the finest level) via
   *  applying the whole multigrid cycle to a random solution vector
   *
   *
   *  Richardson iteration: x^{k+1} = x^{k} + M^{-1} * ( b - A*x^{k} )
   *
   *  A: system matrix
   *  M^{-1}: preconditioner (should approximate A^{-1})
   *
   *  rhs vector b: b = 0 (-> x_ex = 0)
   *  initial guess x^{0} = rand()
   *
   *  --> calculate x^{1} = x^{0} - M^{-1}*A*x^{0}
   *  --> If multigrid cycle works well, x^{1} should be small (entries < 0.1)
   */
  void
  check()
  {
    /*
     *  Whole MG Cycle
     */
    VectorType initial_solution;
    underlying_operator.initialize_dof_vector(initial_solution);
    VectorType solution_after_mg_cycle(initial_solution), tmp(initial_solution);

    for(unsigned int i = 0; i < initial_solution.locally_owned_size(); ++i)
      initial_solution.local_element(i) = (double)rand() / RAND_MAX;

    underlying_operator.vmult(tmp, initial_solution);
    tmp *= -1.0;
    preconditioner->vmult(solution_after_mg_cycle, tmp);
    solution_after_mg_cycle += initial_solution;

    /*
     *  Smoothing
     */

    VectorTypeMG initial_solution_float;
    initial_solution_float = initial_solution;
    VectorTypeMG solution_after_smoothing, tmp_float;
    solution_after_smoothing = initial_solution;
    tmp_float                = tmp;

    preconditioner->apply_smoother_on_fine_level(solution_after_smoothing, tmp_float);
    solution_after_smoothing += initial_solution_float;

    /*
     *  Output
     */
    write_output(initial_solution, solution_after_mg_cycle, solution_after_smoothing);

    /*
     *  Terminate simulation
     */
    //    std::abort();
  }

  void
  write_output(VectorType const &   initial_solution,
               VectorType const &   solution_after_mg_cycle,
               VectorTypeMG const & solution_after_smoothing) const
  {
    OutputDataBase output_data;
    output_data.filename         = "smoothing";
    unsigned int const dof_index = underlying_operator.get_dof_index();
    output_data.degree =
      underlying_operator.get_matrix_free().get_dof_handler(dof_index).get_fe().degree;

    VectorWriter<dim, Number> vector_writer(output_data, 0 /* output_counter */, mpi_comm);

    unsigned int const n_components =
      underlying_operator.get_matrix_free().get_dof_handler(dof_index).get_fe().n_components();
    dealii::DoFHandler<dim> const & dof_handler =
      underlying_operator.get_matrix_free().get_dof_handler(dof_index);
    if(n_components == 1)
    {
      vector_writer.add_data_vector(initial_solution, dof_handler, {"initial"});
      vector_writer.add_data_vector(solution_after_mg_cycle, dof_handler, {"after_mg_cycle"});
      vector_writer.add_data_vector(solution_after_smoothing, dof_handler, {"after_smoothing"});
    }
    else
    {
      std::vector<bool>        component_is_part_of_vector(dim, true);
      std::vector<std::string> component_names(dim, "initial");

      vector_writer.add_data_vector(initial_solution,
                                    dof_handler,
                                    component_names,
                                    component_is_part_of_vector);

      std::fill(component_names.begin(), component_names.end(), "after_mg_cycle");
      vector_writer.add_data_vector(solution_after_mg_cycle,
                                    dof_handler,
                                    component_names,
                                    component_is_part_of_vector);

      std::fill(component_names.begin(), component_names.end(), "after_smoothing");
      vector_writer.add_data_vector(solution_after_smoothing,
                                    dof_handler,
                                    component_names,
                                    component_is_part_of_vector);
    }

    vector_writer.write_pvtu();
  }

private:
  Operator const & underlying_operator;

  std::shared_ptr<Preconditioner> preconditioner;

  MPI_Comm mpi_comm;
};

} // namespace ExaDG

#endif /* EXADG_SOLVERS_AND_PRECONDITIONERS_UTILITIES_CHECK_MULTIGRID_H_ */
