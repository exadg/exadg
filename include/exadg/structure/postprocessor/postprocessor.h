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

#ifndef EXADG_STRUCTURE_POSTPROCESSOR_POSTPROCESSOR_H_
#define EXADG_STRUCTURE_POSTPROCESSOR_POSTPROCESSOR_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/postprocessor/error_calculation.h>
#include <exadg/structure/postprocessor/output_generator.h>
#include <exadg/structure/postprocessor/postprocessor_base.h>
#include <exadg/structure/spatial_discretization/operator.h>

namespace ExaDG
{
namespace Structure
{
template<int dim>
struct PostProcessorData
{
  OutputData                output_data;
  ErrorCalculationData<dim> error_data;
};

template<int dim, typename Number>
class PostProcessor : public PostProcessorBase<dim, Number>
{
public:
  typedef PostProcessorBase<dim, Number> Base;

  typedef typename Base::VectorType VectorType;

  PostProcessor(PostProcessorData<dim> const & pp_data, MPI_Comm const & mpi_comm);

  void
  setup(Operator<dim, Number> const & pde_operator_in) override;

  bool
  requires_scalar_field() const;

  void
  do_postprocessing(VectorType const &     solution,
                    double const           time             = 0.0,
                    types::time_step const time_step_number = numbers::steady_timestep) override;

private:
  void
  initialize_derived_fields();

  void
  invalidate_derived_fields();

  PostProcessorData<dim> pp_data;

  MPI_Comm const mpi_comm;

  dealii::ObserverPointer<Operator<dim, Number> const> pde_operator;

  // Fields for derived quantities
  SolutionField<dim, Number> displacement_magnitude;

  // write output for visualization of results (e.g., using paraview)
  OutputGenerator<dim, Number> output_generator;

  // calculate errors for verification purposes for problems with known analytical solution
  ErrorCalculator<dim, Number> error_calculator;
};

} // namespace Structure
} // namespace ExaDG

#endif /* EXADG_STRUCTURE_POSTPROCESSOR_POSTPROCESSOR_H_ */
