/*
 * postprocessor.h
 *
 *  Created on: Oct 12, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_POISSON_POSTPROCESSOR_H_
#define INCLUDE_POISSON_POSTPROCESSOR_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/convection_diffusion/user_interface/analytical_solution.h>
#include <exadg/poisson/postprocessor/postprocessor_base.h>
#include <exadg/postprocessor/error_calculation.h>
#include <exadg/postprocessor/output_data_base.h>
#include <exadg/postprocessor/output_generator_scalar.h>

namespace ExaDG
{
namespace Poisson
{
using namespace dealii;

template<int dim>
struct PostProcessorData
{
  PostProcessorData()
  {
  }

  OutputDataBase            output_data;
  ErrorCalculationData<dim> error_data;
};

template<int dim, typename Number>
class PostProcessor : public PostProcessorBase<dim, Number>
{
protected:
  typedef PostProcessorBase<dim, Number> Base;

  typedef typename Base::VectorType VectorType;

public:
  PostProcessor(PostProcessorData<dim> const & pp_data, MPI_Comm const & mpi_comm);

  void
  setup(DoFHandler<dim, dim> const & dof_handler, Mapping<dim> const & mapping) override;

  virtual void
  do_postprocessing(VectorType const & solution,
                    double const       time             = 0.0,
                    int const          time_step_number = -1);

protected:
  MPI_Comm const & mpi_comm;

private:
  PostProcessorData<dim> pp_data;

  OutputGenerator<dim, Number> output_generator;
  ErrorCalculator<dim, Number> error_calculator;
};

} // namespace Poisson
} // namespace ExaDG


#endif /* INCLUDE_POISSON_POSTPROCESSOR_H_ */
