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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#include <exadg/structure/postprocessor/postprocessor.h>
#include <exadg/utilities/evaluate_convergence_study.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
PostProcessor<dim, Number>::PostProcessor(PostProcessorData<dim> const & pp_data_in,
                                          MPI_Comm const &               mpi_comm_in)
  : pp_data(pp_data_in),
    mpi_comm(mpi_comm_in),
    output_generator(OutputGenerator<dim, Number>(mpi_comm_in)),
    error_calculator(ErrorCalculator<dim, Number>(mpi_comm_in))
{
}

template<int dim, typename Number>
PostProcessor<dim, Number>::~PostProcessor()
{
  // Evaluate the convergence study.
  std::vector<std::string> error_directories;
  if(pp_data.error_data.compute_convergence_table)
    error_directories.push_back(pp_data.error_data.directory);

  evaluate_convergence_study(mpi_comm, error_directories);
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::setup(Operator<dim, Number> const & pde_operator_in)
{
  pde_operator = &pde_operator_in;

  initialize_derived_fields();

  output_generator.setup(pde_operator->get_dof_handler(),
                         pde_operator->get_mapping(),
                         pp_data.output_data);

  error_calculator.setup(pde_operator->get_dof_handler(),
                         pde_operator->get_mapping(),
                         pp_data.error_data);
}

template<int dim, typename Number>
bool
PostProcessor<dim, Number>::requires_scalar_field() const
{
  return (pp_data.output_data.write_displacement_magnitude or
          pp_data.output_data.write_displacement_jacobian);
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::do_postprocessing(VectorType const &     solution,
                                              double const           time,
                                              types::time_step const time_step_number)
{
  invalidate_derived_fields();

  /*
   *  write output
   */
  if(output_generator.time_control.needs_evaluation(time, time_step_number))
  {
    std::vector<dealii::ObserverPointer<SolutionField<dim, Number>>> additional_fields_vtu;

    if(pp_data.output_data.write_displacement_magnitude)
    {
      displacement_magnitude.evaluate(solution);
      additional_fields_vtu.push_back(&displacement_magnitude);
    }

    if(pp_data.output_data.write_displacement_jacobian)
    {
      displacement_jacobian.evaluate(solution);
      additional_fields_vtu.push_back(&displacement_jacobian);
    }

    if(pp_data.output_data.write_max_principal_stress)
    {
      max_principal_stress.evaluate(solution);
      additional_fields_vtu.push_back(&max_principal_stress);
    }

    output_generator.evaluate(solution,
                              additional_fields_vtu,
                              time,
                              Utilities::is_unsteady_timestep(time_step_number));
  }

  /*
   *  calculate error
   */
  if(error_calculator.time_control.needs_evaluation(time, time_step_number))
    error_calculator.evaluate(solution, time, Utilities::is_unsteady_timestep(time_step_number));
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::initialize_derived_fields()
{
  // displacement magnitude
  if(pp_data.output_data.write_displacement_magnitude)
  {
    displacement_magnitude.type              = SolutionFieldType::scalar;
    displacement_magnitude.name              = "displacement_magnitude";
    displacement_magnitude.dof_handler       = &pde_operator->get_dof_handler_scalar();
    displacement_magnitude.initialize_vector = [&](VectorType & dst) {
      pde_operator->initialize_dof_vector_scalar(dst);
    };
    displacement_magnitude.recompute_solution_field = [&](VectorType &       dst_scalar_valued,
                                                          const VectorType & src_vector_valued) {
      pde_operator->compute_displacement_magnitude(dst_scalar_valued, src_vector_valued);
    };

    displacement_magnitude.reinit();
  }

  // Jacobian of the displacement field
  if(pp_data.output_data.write_displacement_jacobian)
  {
    displacement_jacobian.type              = SolutionFieldType::scalar;
    displacement_jacobian.name              = "displacement_jacobian";
    displacement_jacobian.dof_handler       = &pde_operator->get_dof_handler_scalar();
    displacement_jacobian.initialize_vector = [&](VectorType & dst) {
      pde_operator->initialize_dof_vector_scalar(dst);
    };
    displacement_jacobian.recompute_solution_field = [&](VectorType &       dst_scalar_valued,
                                                         const VectorType & src_vector_valued) {
      pde_operator->compute_displacement_jacobian(dst_scalar_valued, src_vector_valued);
    };

    displacement_jacobian.reinit();
  }

  // Maximum principal stress
  if(pp_data.output_data.write_max_principal_stress)
  {
    max_principal_stress.type              = SolutionFieldType::scalar;
    max_principal_stress.name              = "max_principal_stress";
    max_principal_stress.dof_handler       = &pde_operator->get_dof_handler_scalar();
    max_principal_stress.initialize_vector = [&](VectorType & dst) {
      pde_operator->initialize_dof_vector_scalar(dst);
    };
    max_principal_stress.recompute_solution_field = [&](VectorType &       dst_scalar_valued,
                                                        const VectorType & src_vector_valued) {
      pde_operator->compute_max_principal_stress(dst_scalar_valued, src_vector_valued);
    };

    max_principal_stress.reinit();
  }
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::invalidate_derived_fields()
{
  displacement_magnitude.invalidate();
  displacement_jacobian.invalidate();
  max_principal_stress.invalidate();
}

template class PostProcessor<2, float>;
template class PostProcessor<3, float>;

template class PostProcessor<2, double>;
template class PostProcessor<3, double>;

} // namespace Structure
} // namespace ExaDG
