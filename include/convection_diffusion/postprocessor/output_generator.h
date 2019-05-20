/*
 * output_generator.h
 *
 *  Created on: Mar 7, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_OUTPUT_GENERATOR_H_
#define INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_OUTPUT_GENERATOR_H_

// deal.II
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "../../postprocessor/output_data.h"

namespace ConvDiff
{
template<int dim, typename Number>
class OutputGenerator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  OutputGenerator();

  void
  setup(DoFHandler<dim> const & dof_handler_in,
        Mapping<dim> const &    mapping_in,
        OutputDataBase const &  output_data_in);

  void
  evaluate(VectorType const & solution, double const & time, int const & time_step_number);

private:
  unsigned int output_counter;
  bool         reset_counter;

  SmartPointer<DoFHandler<dim> const> dof_handler;
  SmartPointer<Mapping<dim> const>    mapping;
  OutputDataBase                      output_data;
};

} // namespace ConvDiff


#endif /* INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_OUTPUT_GENERATOR_H_ */
