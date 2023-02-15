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

#include <exadg/postprocessor/normal_flux_calculation.h>

namespace ExaDG
{
template<int dim, typename Number>
NormalFluxCalculator<dim, Number>::NormalFluxCalculator(
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  unsigned int const                      dof_index_in,
  unsigned int const                      quad_index_in,
  MPI_Comm const &                        mpi_comm_in)
  : matrix_free(matrix_free_in),
    dof_index(dof_index_in),
    quad_index(quad_index_in),
    mpi_comm(mpi_comm_in)
{
}

template<int dim, typename Number>
void
NormalFluxCalculator<dim, Number>::evaluate(VectorType const &                             solution,
                                            std::map<dealii::types::boundary_id, Number> & flux)
{
  // zero values since we sum into these variables
  for(auto & iterator : flux)
  {
    iterator.second = 0.0;
  }

  FaceIntegratorScalar integrator(matrix_free, true, dof_index, quad_index);

  std::map<dealii::types::boundary_id, Number> area(flux);

  for(unsigned int face = matrix_free.n_inner_face_batches();
      face < (matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches());
      face++)
  {
    typename std::map<dealii::types::boundary_id, Number>::iterator it;
    dealii::types::boundary_id boundary_id = matrix_free.get_boundary_id(face);

    it = flux.find(boundary_id);
    if(it != flux.end())
    {
      integrator.reinit(face);
      integrator.read_dof_values(solution);
      integrator.evaluate(dealii::EvaluationFlags::gradients);

      scalar flux_face = dealii::make_vectorized_array<Number>(0.0);

      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        flux_face +=
          integrator.JxW(q) * integrator.get_gradient(q) * integrator.get_normal_vector(q);
      }

      // sum over all entries of dealii::VectorizedArray
      for(unsigned int n = 0; n < matrix_free.n_active_entries_per_face_batch(face); ++n)
      {
        flux.at(boundary_id) += flux_face[n];
      }
    }
  }

  // map -> vector
  std::vector<double> flux_vector(flux.size());
  auto                iterator = flux.begin();
  for(unsigned int counter = 0; counter < flux.size(); ++counter)
  {
    flux_vector[counter] = (iterator++)->second;
  }

  // sum over MPI processes
  dealii::Utilities::MPI::sum(
    dealii::ArrayView<double const>(&(*flux_vector.begin()), flux_vector.size()),
    mpi_comm,
    dealii::ArrayView<double>(&(*flux_vector.begin()), flux_vector.size()));

  // vector -> map
  iterator = flux.begin();
  for(unsigned int counter = 0; counter < flux.size(); ++counter)
  {
    (iterator++)->second = flux_vector[counter];
  }
}

template class NormalFluxCalculator<2, float>;
template class NormalFluxCalculator<2, double>;

template class NormalFluxCalculator<3, float>;
template class NormalFluxCalculator<3, double>;

} // namespace ExaDG
