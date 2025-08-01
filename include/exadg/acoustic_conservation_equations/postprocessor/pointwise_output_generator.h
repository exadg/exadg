/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_POSTPROCESSOR_POINTWISE_OUTPUT_GENERATOR_H_
#define EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_POSTPROCESSOR_POINTWISE_OUTPUT_GENERATOR_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/postprocessor/pointwise_output_generator_base.h>

namespace ExaDG
{
namespace Acoustics
{
template<int dim>
struct PointwiseOutputData : public PointwiseOutputDataBase<dim>
{
  PointwiseOutputData();

  void
  print(dealii::ConditionalOStream & pcout) const;

  bool write_pressure;
  bool write_velocity;
};

template<int dim, typename Number>
class PointwiseOutputGenerator : public PointwiseOutputGeneratorBase<dim, Number>
{
public:
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  PointwiseOutputGenerator(MPI_Comm const & comm);

  void
  setup(dealii::DoFHandler<dim> const &  dof_handler_pressure_in,
        dealii::DoFHandler<dim> const &  dof_handler_velocity_in,
        dealii::Mapping<dim> const &     mapping_in,
        PointwiseOutputData<dim> const & pointwise_output_data_in);

  void
  evaluate(VectorType const & pressure,
           VectorType const & velocity,
           double const       time,
           bool const         unsteady);

private:
  dealii::SmartPointer<dealii::DoFHandler<dim> const> dof_handler_pressure;
  dealii::SmartPointer<dealii::DoFHandler<dim> const> dof_handler_velocity;

  PointwiseOutputData<dim> pointwise_output_data;
};

} // namespace Acoustics
} // namespace ExaDG

#endif /* EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_POSTPROCESSOR_POINTWISE_OUTPUT_GENERATOR_H_ \
        */
