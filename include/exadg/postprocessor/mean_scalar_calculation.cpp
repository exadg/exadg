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

#include <exadg/postprocessor/mean_scalar_calculation.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, typename Number>
MeanScalarCalculator<dim, Number>::MeanScalarCalculator(
  const MatrixFree<dim, Number> & matrix_free_in,
  const unsigned int              dof_index_in,
  const unsigned int              quad_index_in,
  MPI_Comm const &                mpi_comm_in)
  : matrix_free(matrix_free_in),
    dof_index(dof_index_in),
    quad_index(quad_index_in),
    mpi_comm(mpi_comm_in)
{
}

template<int dim, typename Number>
Number
MeanScalarCalculator<dim, Number>::calculate_mean_scalar(
  const VectorType &                     solution,
  std::map<types::boundary_id, Number> & mean_scalar)
{
  // zero mean scalars since we sum into these variables
  for(auto & iterator : mean_scalar)
  {
    iterator.second = 0.0;
  }

  FaceIntegratorScalar integrator(matrix_free, true, dof_index, quad_index);

  std::map<types::boundary_id, Number> area(mean_scalar);

  for(unsigned int face = matrix_free.n_inner_face_batches();
      face < (matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches());
      face++)
  {
    typename std::map<types::boundary_id, Number>::iterator it;
    types::boundary_id boundary_id = matrix_free.get_boundary_id(face);

    it = mean_scalar.find(boundary_id);
    if(it != mean_scalar.end())
    {
      integrator.reinit(face);
      integrator.read_dof_values(solution);
      integrator.evaluate(true, false);

      scalar mean_scalar_face = make_vectorized_array<Number>(0.0);
      scalar area_face        = make_vectorized_array<Number>(0.0);

      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        mean_scalar_face += integrator.JxW(q) * integrator.get_value(q);
        area_face += integrator.JxW(q);
      }

      // sum over all entries of VectorizedArray
      for(unsigned int n = 0; n < matrix_free.n_active_entries_per_face_batch(face); ++n)
      {
        mean_scalar.at(boundary_id) += mean_scalar_face[n];
        area.at(boundary_id) += area_face[n];
      }
    }
  }

  std::vector<double> mean_scalar_vector(mean_scalar.size());
  std::vector<double> area_vector(area.size());
  auto                iterator      = mean_scalar.begin();
  auto                iterator_area = area.begin();
  for(unsigned int counter = 0; counter < mean_scalar.size(); ++counter)
  {
    mean_scalar_vector[counter] = (iterator++)->second;
    area_vector[counter]        = (iterator_area++)->second;
  }

  Utilities::MPI::sum(ArrayView<const double>(&(*mean_scalar_vector.begin()),
                                              mean_scalar_vector.size()),
                      mpi_comm,
                      ArrayView<double>(&(*mean_scalar_vector.begin()), mean_scalar_vector.size()));

  Utilities::MPI::sum(ArrayView<const double>(&(*area_vector.begin()), area_vector.size()),
                      mpi_comm,
                      ArrayView<double>(&(*area_vector.begin()), area_vector.size()));

  iterator = mean_scalar.begin();
  for(unsigned int counter = 0; counter < mean_scalar.size(); ++counter)
  {
    (iterator++)->second = mean_scalar_vector[counter] / area_vector[counter];
  }

  return 0;
}

template class MeanScalarCalculator<2, float>;
template class MeanScalarCalculator<2, double>;

template class MeanScalarCalculator<3, float>;
template class MeanScalarCalculator<3, double>;

} // namespace ExaDG
