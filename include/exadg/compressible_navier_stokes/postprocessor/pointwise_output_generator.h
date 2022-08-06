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

#ifndef INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POINTWISE_OUTPUT_GENERATOR_H_
#define INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POINTWISE_OUTPUT_GENERATOR_H_

#include <exadg/postprocessor/pointwise_output_generator_base.h>

namespace ExaDG
{
namespace CompNS
{
template<int dim>
struct PointwiseOutputData : public PointwiseOutputDataBase<dim>
{
  PointwiseOutputData();

  void
  print(dealii::ConditionalOStream & pcout) const;

  bool write_rho;
  bool write_rho_u;
  bool write_rho_E;
};

template<int dim, typename Number>
class PointwiseOutputGenerator : public PointwiseOutputGeneratorBase<dim, Number>
{
public:
  using VectorType = typename PointwiseOutputGeneratorBase<dim, Number>::VectorType;

  PointwiseOutputGenerator(MPI_Comm const & comm);

  void
  setup(dealii::DoFHandler<dim> const &  dof_handler_in,
        dealii::Mapping<dim> const &     mapping_in,
        PointwiseOutputData<dim> const & pointwise_output_data_in);

private:
  void
  do_evaluate(VectorType const & solution) final;

  PointwiseOutputData<dim> pointwise_output_data;
};

} // namespace CompNS
} // namespace ExaDG


#endif /* INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POINTWISE_OUTPUT_GENERATOR_H_ \
        */
