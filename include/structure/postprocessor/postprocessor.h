/*
 * postprocessor.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_POSTPROCESSOR_POSTPROCESSOR_H_
#define INCLUDE_STRUCTURE_POSTPROCESSOR_POSTPROCESSOR_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

#include "../../postprocessor/error_calculation.h"
#include "../../postprocessor/output_data.h"

#include "output_generator.h"

namespace Structure
{
template<int dim>
struct PostProcessorData
{
  OutputDataBase            output_data;
  ErrorCalculationData<dim> error_data;
};

template<int dim, typename Number>
class PostProcessor
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  PostProcessor(PostProcessorData<dim> const & pp_data, MPI_Comm const & mpi_comm);

  void
  setup(DoFHandler<dim> const & dof_handler, Mapping<dim> const & mapping);

  void
  do_postprocessing(VectorType const & solution,
                    double const       time             = 0.0,
                    int const          time_step_number = -1);

private:
  PostProcessorData<dim> pp_data;

  MPI_Comm const & mpi_comm;

  OutputGenerator<dim, Number> output_generator;
  ErrorCalculator<dim, Number> error_calculator;
};

} // namespace Structure

#endif
