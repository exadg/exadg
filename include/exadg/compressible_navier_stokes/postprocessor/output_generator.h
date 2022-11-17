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

#ifndef INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_OUTPUT_GENERATOR_H_
#define INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_OUTPUT_GENERATOR_H_

// C/C++
#include <fstream>

// ExaDG
#include <exadg/postprocessor/output_data_base.h>
#include <exadg/postprocessor/solution_field.h>
#include <exadg/postprocessor/time_control.h>

namespace ExaDG
{
namespace CompNS
{
struct OutputData : public OutputDataBase
{
  OutputData()
    : write_velocity(false),
      write_pressure(false),
      write_temperature(false),
      write_vorticity(false),
      write_divergence(false),
      write_processor_id(false)
  {
  }

  void
  print(dealii::ConditionalOStream & pcout, bool unsteady)
  {
    OutputDataBase::print(pcout, unsteady);

    print_parameter(pcout, "Write velocity", write_velocity);
    print_parameter(pcout, "Write pressure", write_pressure);
    print_parameter(pcout, "Write temperature", write_temperature);
    print_parameter(pcout, "Write vorticity", write_vorticity);
    print_parameter(pcout, "Write divergence", write_divergence);
    print_parameter(pcout, "Write processor ID", write_processor_id);
  }

  // write velocity
  bool write_velocity;

  // write pressure
  bool write_pressure;

  // write temperature
  bool write_temperature;

  // write vorticity of velocity field
  bool write_vorticity;

  // write divergence of velocity field
  bool write_divergence;

  // write processor ID to scalar field in order to visualize the
  // distribution of cells to processors
  bool write_processor_id;
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
  evaluate(VectorType const &                                                    solution_conserved,
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

} // namespace CompNS
} // namespace ExaDG


#endif /* INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_OUTPUT_GENERATOR_H_ */
