/*
 * postprocessor.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
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

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

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
} // namespace ExaDG

#endif
