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

#ifndef EXADG_POSTPROCESSOR_OUTPUT_GENERATOR_SCALAR_H_
#define EXADG_POSTPROCESSOR_OUTPUT_GENERATOR_SCALAR_H_

// deal.II
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/postprocessor/output_data_base.h>
#include <exadg/postprocessor/time_control.h>

namespace ExaDG
{
template<int dim, typename Number>
class OutputGenerator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  OutputGenerator(MPI_Comm const & comm);

  void
  setup(dealii::DoFHandler<dim> const & dof_handler_in,
        dealii::Mapping<dim> const &    mapping_in,
        OutputDataBase const &          output_data_in);

  void
  evaluate(VectorType const & solution, double const time, bool const unsteady);

  TimeControl time_control;

private:
  MPI_Comm const mpi_comm;

  dealii::ObserverPointer<dealii::DoFHandler<dim> const> dof_handler;
  dealii::ObserverPointer<dealii::Mapping<dim> const>    mapping;
  OutputDataBase                                         output_data;
};

} // namespace ExaDG

#endif /* EXADG_POSTPROCESSOR_OUTPUT_GENERATOR_SCALAR_H_ */
