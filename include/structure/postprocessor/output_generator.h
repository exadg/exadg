/*
 * output_generator.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_POSTPROCESSOR_OUTPUT_GENERATOR_H_
#define INCLUDE_STRUCTURE_POSTPROCESSOR_OUTPUT_GENERATOR_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

#include "../../postprocessor/output_data.h"

namespace Structure
{
template<int dim, typename Number>
class OutputGenerator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  OutputGenerator(MPI_Comm const & comm);

  void
  setup(DoFHandler<dim> const & dof_handler,
        Mapping<dim> const &    mapping,
        OutputDataBase const &  output_data);

  void
  evaluate(VectorType const & solution, double const & time, int const & time_step_number);

private:
  MPI_Comm const & mpi_comm;

  unsigned int output_counter;
  bool         reset_counter;

  SmartPointer<DoFHandler<dim> const> dof_handler;
  SmartPointer<Mapping<dim> const>    mapping;
  OutputDataBase                      output_data;
};

} // namespace Structure

#endif
