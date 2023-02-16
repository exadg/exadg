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
#ifndef INCLUDE_EXADG_POSTPROCESSOR_NORMAL_FLUX_CALCULATION_H
#define INCLUDE_EXADG_POSTPROCESSOR_NORMAL_FLUX_CALCULATION_H

// ExaDG
#include <exadg/matrix_free/integrators.h>

namespace ExaDG
{
struct NormalFluxCalculatorData
{
  NormalFluxCalculatorData() : evaluate(false)
  {
  }

  bool                                 evaluate;
  std::set<dealii::types::boundary_id> boundary_ids;

  // specify where to write output files
  std::string directory;
  std::string filename;
};

template<int dim, typename Number>
class NormalFluxCalculator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef FaceIntegrator<dim, 1, Number> FaceIntegratorScalar;

  typedef dealii::VectorizedArray<Number> scalar;

  NormalFluxCalculator(dealii::MatrixFree<dim, Number> const & matrix_free_in,
                       unsigned int                            dof_index_in,
                       unsigned int                            quad_index_in,
                       NormalFluxCalculatorData const &        data,
                       MPI_Comm const &                        mpi_comm_in);

  void
  evaluate(VectorType const & solution, double const time, bool const unsteady);

private:
  dealii::MatrixFree<dim, Number> const & matrix_free;
  unsigned int                            dof_index, quad_index;

  NormalFluxCalculatorData data;

  bool clear_files;

  std::map<dealii::types::boundary_id, double> flux;

  MPI_Comm const mpi_comm;
};

} // namespace ExaDG



#endif // INCLUDE_EXADG_POSTPROCESSOR_NORMAL_FLUX_CALCULATION_H
