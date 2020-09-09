/*
 * output_generator.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_STRUCTURE_POSTPROCESSOR_OUTPUT_GENERATOR_H_
#define INCLUDE_EXADG_STRUCTURE_POSTPROCESSOR_OUTPUT_GENERATOR_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/postprocessor/output_data_base.h>

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

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
} // namespace ExaDG

#endif
