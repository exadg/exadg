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

#ifndef INCLUDE_EXADG_STRUCTURE_POSTPROCESSOR_POSTPROCESSOR_H_
#define INCLUDE_EXADG_STRUCTURE_POSTPROCESSOR_POSTPROCESSOR_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/postprocessor/error_calculation.h>
#include <exadg/postprocessor/output_data_base.h>
#include <exadg/structure/postprocessor/output_generator.h>
#include <exadg/structure/postprocessor/postprocessor_base.h>
#include <exadg/utilities/numbers.h>

namespace ExaDG
{
namespace Structure
{
template<int dim>
struct PostProcessorData
{
  OutputDataBase            output_data;
  ErrorCalculationData<dim> error_data;
};

template<int dim, typename Number>
class PostProcessor : public PostProcessorBase<Number>
{
private:
  typedef typename PostProcessorBase<Number>::VectorType VectorType;

public:
  PostProcessor(PostProcessorData<dim> const & pp_data, MPI_Comm const & mpi_comm);

  void
  setup(dealii::DoFHandler<dim> const & dof_handler, dealii::Mapping<dim> const & mapping);

  void
  do_postprocessing(VectorType const & solution,
                    double const       time             = 0.0,
                    int const          time_step_number = numbers::steady_timestep);

private:
  PostProcessorData<dim> pp_data;

  MPI_Comm const mpi_comm;

  OutputGenerator<dim, Number> output_generator;
  ErrorCalculator<dim, Number> error_calculator;
};

} // namespace Structure
} // namespace ExaDG

#endif
