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

#ifndef EXADG_STRUCTURE_POSTPROCESSOR_OUTPUT_GENERATOR_H_
#define EXADG_STRUCTURE_POSTPROCESSOR_OUTPUT_GENERATOR_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/postprocessor/output_data_base.h>
#include <exadg/postprocessor/solution_field.h>
#include <exadg/postprocessor/time_control.h>

namespace ExaDG
{
namespace Structure
{
struct OutputData : public OutputDataBase
{
  OutputData()
    : write_displacement_magnitude(false),
      write_displacement_jacobian(false),
      write_max_principal_stress(false)
  {
  }

  void
  print(dealii::ConditionalOStream & pcout, bool unsteady)
  {
    OutputDataBase::print(pcout, unsteady);

    print_parameter(pcout, "Write displacement magnitude", write_displacement_magnitude);
    print_parameter(pcout, "Write displacement Jacobian", write_displacement_jacobian);
    print_parameter(pcout, "Write maximum principal stress", write_max_principal_stress);
  }

  // write displacement magnitude
  bool write_displacement_magnitude;

  // write Jacobian of the displacement field
  bool write_displacement_jacobian;

  // write maximum principal stress
  bool write_max_principal_stress;
};

template<int dim, typename Number>
class OutputGenerator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  OutputGenerator(MPI_Comm const & comm);

  void
  setup(dealii::DoFHandler<dim> const & dof_handler,
        dealii::Mapping<dim> const &    mapping,
        OutputData const &              output_data);

  void
  evaluate(
    VectorType const &                                                       solution,
    std::vector<dealii::ObserverPointer<SolutionField<dim, Number>>> const & additional_fields,
    double const                                                             time,
    bool const                                                               unsteady);

  TimeControl time_control;

private:
  MPI_Comm const mpi_comm;

  OutputData output_data;

  dealii::ObserverPointer<dealii::DoFHandler<dim> const> dof_handler;
  dealii::ObserverPointer<dealii::Mapping<dim> const>    mapping;
};

} // namespace Structure
} // namespace ExaDG

#endif /* EXADG_STRUCTURE_POSTPROCESSOR_OUTPUT_GENERATOR_H_ */
