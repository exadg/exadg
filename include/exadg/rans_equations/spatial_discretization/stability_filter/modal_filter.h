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

#ifndef INCLUDE_EXADG_RANS_EQUATIONS_SPATIAL_DISCRETIZATION_STABILITY_FILTER_MODAL_FILTERING_H
#define INCLUDE_EXADG_RANS_EQUATIONS_SPATIAL_DISCRETIZATION_STABILITY_FILTER_MODAL_FILTERING_H

#include <exadg/matrix_free/integrators.h>

namespace ExaDG
{
namespace RANS
{
template<int dim, typename Number>
class ModalFilterOperator
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> tensor;

  typedef CellIntegrator<dim, 1, Number> IntegratorCell;

  typedef ModalFilterOperator<dim, Number> This;

  typedef std::pair<unsigned int, unsigned int> Range;

public:
  ModalFilterOperator()
    : maximum_gradient(0.0),
      average_gradient(0.0),
      local_maximum_gradient(0.0),
      local_sum_gradient(0.0),
      local_count_gradient(0.0)
  {
  }

  void
  initialize(std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free_in,
             std::shared_ptr<dealii::Mapping<dim> const>            mapping_in,
             unsigned int                                           dof_index_in,
             unsigned int                                           quad_index_in);

  void
  apply_modal_filter(VectorType const & src, VectorType & dst);

private:
  void
  evaluate_critical_gradients(VectorType const & src);

  void
  apply_weighted_relaxation(VectorType const & src, VectorType & dst);

  void
  cell_loop_calculate_critical_gradients(dealii::MatrixFree<dim, Number> const & matrix_free_in,
                                         VectorType &                            dst,
                                         VectorType const &                      src,
                                         Range const &                           cell_range);

  void
  cell_loop_apply_weighted_relaxation(dealii::MatrixFree<dim, Number> const & matrix_free_in,
                                      VectorType &                            dst,
                                      VectorType const &                      src,
                                      Range const &                           cell_range);

  dealii::VectorizedArray<Number>
  weight_function(uint dof, uint dofs_per_cell);

  void
  convert_unit_cell_to_reference_cell(const std::vector<dealii::Point<dim>> & src,
                                      std::vector<dealii::Point<dim>> &       dst);

  void
  evaluate_vandermonde_matrix(IntegratorCell const &       integrator,
                              dealii::FullMatrix<Number> & V,
                              dealii::FullMatrix<Number> & V_inv,
                              uint                         cell_id);

  std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free;
  std::shared_ptr<dealii::Mapping<dim> const>            mapping;

  unsigned int dof_index;
  unsigned int quad_index;

public:
  VectorType global_gradient_mag_vector;

  Number maximum_gradient;
  Number average_gradient;
  Number local_maximum_gradient;
  Number local_sum_gradient;
  Number local_count_gradient;
};

} // namespace RANS
} // namespace ExaDG

#endif
