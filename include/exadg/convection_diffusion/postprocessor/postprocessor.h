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

#ifndef INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_H_
#define INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/convection_diffusion/postprocessor/postprocessor_base.h>
#include <exadg/convection_diffusion/user_interface/analytical_solution.h>
#include <exadg/postprocessor/error_calculation.h>
#include <exadg/postprocessor/output_data_base.h>
#include <exadg/postprocessor/output_generator_scalar.h>

namespace ExaDG
{
namespace ConvDiff
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

  virtual ~PostProcessor()
  {
  }

  void
  setup(Operator<dim, Number> const & pde_operator, Mapping<dim> const & mapping) override;

  virtual void
  do_postprocessing(VectorType const & solution,
                    double const       time             = 0.0,
                    int const          time_step_number = -1);

protected:
  MPI_Comm const mpi_comm;

private:
  PostProcessorData<dim> pp_data;

  OutputGenerator<dim, Number> output_generator;
  ErrorCalculator<dim, Number> error_calculator;
};

} // namespace ConvDiff
} // namespace ExaDG


#endif /* INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_H_ */
