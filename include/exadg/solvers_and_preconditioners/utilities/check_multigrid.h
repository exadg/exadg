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

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_CHECKMULTIGRID_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_CHECKMULTIGRID_H_

// C/C++
#include <fstream>

// deal.II
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/numerics/data_out.h>

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
   *  Function that verifies the multgrid algorithm,
   *  especially the smoothing (on the finest level) an
   *  the whole multigrid cycle applied to a random solution vector
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
    VectorType solution_after_mg_cylce(initial_solution), tmp(initial_solution);

    for(unsigned int i = 0; i < initial_solution.locally_owned_size(); ++i)
      initial_solution.local_element(i) = (double)rand() / RAND_MAX;

    underlying_operator.vmult(tmp, initial_solution);
    tmp *= -1.0;
    preconditioner->vmult(solution_after_mg_cylce, tmp);
    solution_after_mg_cylce += initial_solution;

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
    write_output(initial_solution, solution_after_mg_cylce, solution_after_smoothing);

    /*
     *  Terminate simulation
     */
    //    std::abort();
  }

  void
  write_output(VectorType const &   initial_solution,
               VectorType const &   solution_after_mg_cylce,
               VectorTypeMG const & solution_after_smoothing) const
  {
    dealii::DataOut<dim>          data_out;
    dealii::DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    unsigned int dof_index = underlying_operator.get_dof_index();

    // TODO: use scalar = false for velocity field
    bool scalar = true;

    if(scalar)
    {
      // pressure
      data_out.add_data_vector(underlying_operator.get_matrix_free().get_dof_handler(dof_index),
                               initial_solution,
                               "initial");
      data_out.add_data_vector(underlying_operator.get_matrix_free().get_dof_handler(dof_index),
                               solution_after_mg_cylce,
                               "mg_cycle");
      data_out.add_data_vector(underlying_operator.get_matrix_free().get_dof_handler(dof_index),
                               solution_after_smoothing,
                               "smoother");
    }
    else
    {
      // velocity
      std::vector<std::string> initial(dim, "initial");
      std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        initial_component_interpretation(
          dim, dealii::DataComponentInterpretation::component_is_part_of_vector);

      data_out.add_data_vector(underlying_operator.get_matrix_free().get_dof_handler(dof_index),
                               initial_solution,
                               initial,
                               initial_component_interpretation);

      std::vector<std::string> mg_cycle(dim, "mg_cycle");
      std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        mg_cylce_component_interpretation(
          dim, dealii::DataComponentInterpretation::component_is_part_of_vector);

      data_out.add_data_vector(underlying_operator.get_matrix_free().get_dof_handler(dof_index),
                               solution_after_mg_cylce,
                               mg_cycle,
                               mg_cylce_component_interpretation);

      std::vector<std::string> smoother(dim, "smoother");
      std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        smoother_component_interpretation(
          dim, dealii::DataComponentInterpretation::component_is_part_of_vector);

      data_out.add_data_vector(underlying_operator.get_matrix_free().get_dof_handler(dof_index),
                               solution_after_smoothing,
                               smoother,
                               smoother_component_interpretation);
    }

    data_out.build_patches(
      underlying_operator.get_matrix_free().get_dof_handler(dof_index).get_fe().degree);

    std::ostringstream filename;
    std::string        name = "smoothing";
    filename << name << "_Proc" << dealii::Utilities::MPI::this_mpi_process(mpi_comm) << ".vtu";

    std::ofstream output(filename.str().c_str());
    data_out.write_vtu(output);

    if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      std::vector<std::string> filenames;
      for(unsigned int i = 0; i < dealii::Utilities::MPI::n_mpi_processes(mpi_comm); ++i)
      {
        std::ostringstream filename;
        filename << name << "_Proc" << i << ".vtu";

        filenames.push_back(filename.str().c_str());
      }
      std::string   master_name = name + ".pvtu";
      std::ofstream master_output(master_name.c_str());
      data_out.write_pvtu_record(master_output, filenames);
    }
  }

private:
  Operator const & underlying_operator;

  std::shared_ptr<Preconditioner> preconditioner;

  MPI_Comm mpi_comm;
};

} // namespace ExaDG

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_CHECKMULTIGRID_H_ */
