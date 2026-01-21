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

#ifndef INCLUDE_EXADG_RANS_EQUATIONS_POSTPROCESSOR_OUTPUT_GENERATOR_H_
#define INCLUDE_EXADG_RANS_EQUATIONS_POSTPROCESSOR_OUTPUT_GENERATOR_H_

// deal.II
#include <deal.II/base/exceptions.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/postprocessor/output_data_base.h>
#include <exadg/postprocessor/solution_field.h>
#include <exadg/postprocessor/time_control.h>

namespace ExaDG
{
namespace RANS
{
struct OutputData : public OutputDataBase
{
  OutputData()
    : write_eddy_viscosity(false),
      write_modal_coefficients(false),
      write_modal_coefficient_ratio(false)
  {
  }

  void
  print(dealii::ConditionalOStream & pcout, bool unsteady)
  {
    OutputDataBase::print(pcout, unsteady);

    print_parameter(pcout, "Write eddy viscosity", write_eddy_viscosity);
    print_parameter(pcout, "Write modal coefficients", write_modal_coefficients);
    AssertThrow(false, dealii::ExcMessage("print function in output_data executed"));
  }

  // write vorticity of velocity field
  bool write_eddy_viscosity;
  // write modal coefficients
  bool write_modal_coefficients;
  // write modal coefficient ratio
  bool write_modal_coefficient_ratio;
};

template<int dim, typename Number>
class OutputGenerator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  OutputGenerator(MPI_Comm const & comm);

  void
  setup(dealii::DoFHandler<dim> const & dof_handler_in,
        dealii::Mapping<dim> const &    mapping_in,
        OutputData const &              output_data_in);

  void
  evaluate(VectorType const &                                                    solution,
           std::vector<dealii::SmartPointer<SolutionField<dim, Number>>> const & additional_fields,
           double const                                                          time,
           bool const                                                            unsteady);

  TimeControl time_control;

private:
  MPI_Comm const mpi_comm;

  dealii::SmartPointer<dealii::DoFHandler<dim> const> dof_handler;
  dealii::SmartPointer<dealii::Mapping<dim> const>    mapping;
  OutputData                                          output_data;
};

} // namespace RANS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_POSTPROCESSOR_OUTPUT_GENERATOR_SCALAR_H_ */
