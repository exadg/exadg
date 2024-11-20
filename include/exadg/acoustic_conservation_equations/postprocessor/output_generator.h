/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_POSTPROCESSOR_OUTPUT_GENERATOR_H_
#define EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_POSTPROCESSOR_OUTPUT_GENERATOR_H_

#include <exadg/postprocessor/output_data_base.h>
#include <exadg/postprocessor/solution_field.h>
#include <exadg/postprocessor/time_control.h>

namespace ExaDG
{
namespace Acoustics
{
struct OutputData : public OutputDataBase
{
  OutputData() : write_pressure(false), write_velocity(false)
  {
  }

  void
  print(dealii::ConditionalOStream & pcout, bool unsteady)
  {
    OutputDataBase::print(pcout, unsteady);

    print_parameter(pcout, "Write pressure", write_pressure);
    print_parameter(pcout, "Write velocity", write_velocity);
  }

  bool write_pressure;
  bool write_velocity;
};

template<int dim, typename Number>
class OutputGenerator
{
public:
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  OutputGenerator(MPI_Comm const & comm);

  void
  setup(dealii::DoFHandler<dim> const & dof_handler_pressure,
        dealii::DoFHandler<dim> const & dof_handler_velocity,
        dealii::Mapping<dim> const &    mapping,
        OutputData const &              output_data);

  void
  evaluate(VectorType const & pressure,
           VectorType const & velocity,
           double const       time,
           bool const         unsteady) const;

  TimeControl time_control;

private:
  MPI_Comm const mpi_comm;

  OutputData output_data;

  dealii::SmartPointer<dealii::DoFHandler<dim> const> dof_handler_pressure;
  dealii::SmartPointer<dealii::DoFHandler<dim> const> dof_handler_velocity;
  dealii::SmartPointer<dealii::Mapping<dim> const>    mapping;
};

} // namespace Acoustics
} // namespace ExaDG

#endif /* EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_POSTPROCESSOR_OUTPUT_GENERATOR_H_ */
