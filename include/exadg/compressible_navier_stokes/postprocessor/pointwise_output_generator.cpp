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


// deal.II
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/compressible_navier_stokes/postprocessor/pointwise_output_generator.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace CompNS
{
template<int dim>
PointwiseOutputData<dim>::PointwiseOutputData()
  : write_rho(false), write_rho_u(false), write_rho_E(false)
{
}

template<int dim>
void
PointwiseOutputData<dim>::print(dealii::ConditionalOStream & pcout) const
{
  PointwiseOutputDataBase<dim>::print(pcout);

  if(this->time_control_data.is_active && this->evaluation_points.size() > 0)
  {
    print_parameter(pcout, "Write rho", write_rho);
    print_parameter(pcout, "Write rho_u", write_rho_u);
    print_parameter(pcout, "Write rho_E", write_rho_E);
  }
}

template struct PointwiseOutputData<2>;
template struct PointwiseOutputData<3>;

template<int dim, typename Number>
PointwiseOutputGenerator<dim, Number>::PointwiseOutputGenerator(MPI_Comm const & comm)
  : PointwiseOutputGeneratorBase<dim, Number>(comm)
{
}

template<int dim, typename Number>
void
PointwiseOutputGenerator<dim, Number>::setup(
  dealii::DoFHandler<dim> const &  dof_handler_in,
  dealii::Mapping<dim> const &     mapping_in,
  PointwiseOutputData<dim> const & pointwise_output_data_in)
{
  this->setup_base(dof_handler_in, mapping_in, pointwise_output_data_in);

  pointwise_output_data = pointwise_output_data_in;

  if(pointwise_output_data.write_rho)
    this->add_quantity("Rho", 1);
  if(pointwise_output_data.write_rho_u)
    this->add_quantity("Rho_U", dim);
  if(pointwise_output_data.write_rho_E)
    this->add_quantity("Rho_E", 1);
}

template<int dim, typename Number>
void
PointwiseOutputGenerator<dim, Number>::do_evaluate(VectorType const & solution)
{
  if(pointwise_output_data.write_rho || pointwise_output_data.write_rho_u ||
     pointwise_output_data.write_rho_E)
  {
    auto const values = this->template compute_point_values<dim + 2>(solution);
    if(pointwise_output_data.write_rho)
      this->write_quantity("Rho", values, 0);
    if(pointwise_output_data.write_rho_u)
      this->write_quantity("Rho_U", values, 1);
    if(pointwise_output_data.write_rho_E)
      this->write_quantity("Rho_E", values, dim + 2);
  }
}

template class PointwiseOutputGenerator<2, float>;
template class PointwiseOutputGenerator<2, double>;

template class PointwiseOutputGenerator<3, float>;
template class PointwiseOutputGenerator<3, double>;

} // namespace CompNS
} // namespace ExaDG
