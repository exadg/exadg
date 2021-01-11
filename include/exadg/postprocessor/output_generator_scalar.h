/*
 * output_generator.h
 *
 *  Created on: Mar 7, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_POSTPROCESSOR_OUTPUT_GENERATOR_SCALAR_H_
#define INCLUDE_EXADG_POSTPROCESSOR_OUTPUT_GENERATOR_SCALAR_H_

// deal.II
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/postprocessor/output_data_base.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, typename Number>
class OutputGenerator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  OutputGenerator(MPI_Comm const & comm);

  void
  setup(DoFHandler<dim> const & dof_handler_in,
        Mapping<dim> const &    mapping_in,
        OutputDataBase const &  output_data_in);

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

} // namespace ExaDG

#endif /* INCLUDE_EXADG_POSTPROCESSOR_OUTPUT_GENERATOR_SCALAR_H_ */
