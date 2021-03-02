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
#ifndef INCLUDE_EXADG_POSTPROCESSOR_MEAN_SCALAR_CALCULATION_H
#define INCLUDE_EXADG_POSTPROCESSOR_MEAN_SCALAR_CALCULATION_H

#include <exadg/matrix_free/integrators.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, typename Number>
class MeanScalarCalculator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef FaceIntegrator<dim, 1, Number> FaceIntegratorScalar;

  typedef VectorizedArray<Number> scalar;

  MeanScalarCalculator(MatrixFree<dim, Number> const & matrix_free_in,
                       unsigned int                    dof_index_in,
                       unsigned int                    quad_index_in,
                       MPI_Comm const &                mpi_comm_in);

  Number
  calculate_mean_scalar(VectorType const &                     solution,
                        std::map<types::boundary_id, Number> & mean_scalar);

private:
  MatrixFree<dim, Number> const & matrix_free;
  unsigned int                    dof_index, quad_index;

  MPI_Comm const mpi_comm;
};

} // namespace ExaDG



#endif // INCLUDE_EXADG_POSTPROCESSOR_MEAN_SCALAR_CALCULATION_H
