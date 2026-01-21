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

#include <exadg/rans_equations/spatial_discretization/stability_filter/modal_filter.h>

namespace ExaDG
{
namespace RANS
{
template<int dim, typename Number>
void
ModalFilterOperator<dim, Number>::initialize(
  std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free_in,
  std::shared_ptr<dealii::Mapping<dim> const>            mapping_in,
  unsigned int                                           dof_index_in,
  unsigned int                                           quad_index_in)
{
  this->matrix_free = matrix_free_in;
  this->mapping     = mapping_in;
  this->dof_index   = dof_index_in;
  this->quad_index  = quad_index_in;
}

template<int dim, typename Number>
void
ModalFilterOperator<dim, Number>::evaluate_critical_gradients(VectorType const & src)
{
  VectorType dummy;
  matrix_free->cell_loop(&This::cell_loop_calculate_critical_gradients, this, dummy, src);

  MPI_Comm communicator = matrix_free->get_dof_handler(dof_index).get_communicator();

  maximum_gradient    = dealii::Utilities::MPI::max(local_maximum_gradient, communicator);
  Number global_count = dealii::Utilities::MPI::sum(local_count_gradient, communicator);
  Number global_sum   = dealii::Utilities::MPI::sum(local_sum_gradient, communicator);
  average_gradient    = global_sum / global_count;
}

template<int dim, typename Number>
void
ModalFilterOperator<dim, Number>::cell_loop_calculate_critical_gradients(
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           cell_range)
{
  IntegratorCell integrator(matrix_free_in, dof_index, quad_index);

  scalar gradient_magnitude;
  tensor gradient;

  scalar local_sum(0.0);
  scalar local_max(0.0);
  scalar local_count(0.0);

  for(uint cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);
    integrator.read_dof_values(src);
    integrator.evaluate(dealii::EvaluationFlags::gradients);

    scalar cell_sum(0.0);
    scalar cell_max(0.0);
    scalar cell_quad_count(0.0);

    for(uint q = 0; q < integrator.n_q_points; ++q)
    {
      gradient           = integrator.get_gradient(q);
      gradient_magnitude = std::sqrt(gradient.norm_square());
      cell_sum += gradient_magnitude;
      cell_max = std::max(cell_max, gradient_magnitude);
      cell_quad_count += 1.0;
    }
    local_sum += cell_sum;
    local_count += cell_quad_count;
    local_max = std::max(local_max, cell_max);
  }
  // Thread-safe accumulation using deal.II's Mutex
  static dealii::Threads::Mutex           assembly_mutex;
  std::lock_guard<dealii::Threads::Mutex> lock(assembly_mutex);

  local_sum_gradient += local_sum.sum();
  local_count_gradient += local_count.sum();
  for(uint v = 0; v < scalar::size(); ++v)
  {
    local_maximum_gradient = std::max(local_maximum_gradient, local_max[v]);
  }
}

template<int dim, typename Number>
void
ModalFilterOperator<dim, Number>::apply_weighted_relaxation(VectorType const & src,
                                                            VectorType &       dst)
{
  matrix_free->cell_loop(&This::cell_loop_apply_weighted_relaxation, this, dst, src);
}

template<int dim, typename Number>
void
ModalFilterOperator<dim, Number>::cell_loop_apply_weighted_relaxation(
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           cell_range)
{
  IntegratorCell integrator(matrix_free_in, dof_index, quad_index);

  scalar gradient_magnitude;
  tensor gradient;

  scalar threshold(average_gradient);
  scalar flag        = 0.0;
  Number is_critical = 0.0;

  uint dofs_per_cell = integrator.dofs_per_cell;

  for(uint cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);
    integrator.read_dof_values(src);
    integrator.evaluate(dealii::EvaluationFlags::gradients);

    for(uint q = 0; q < integrator.n_q_points; ++q)
    {
      gradient           = integrator.get_gradient(q);
      gradient_magnitude = std::sqrt(gradient.norm_square());
      flag               = dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
        gradient_magnitude, threshold, 1.0, 0.0);
    }
    is_critical = flag.sum();

    if(is_critical > 0.0)
    {
      std::vector<scalar>        nodal(dofs_per_cell);
      std::vector<scalar>        modal(dofs_per_cell);
      dealii::FullMatrix<Number> V_mat;
      dealii::FullMatrix<Number> V_inverse;
      evaluate_vandermonde_matrix(integrator, V_mat, V_inverse, cell);

      /*
       * Modal Transformation
       */
      for(uint i = 0; i < dofs_per_cell; ++i)
      {
        modal[i] = 0.0;
        for(uint j = 0; j < dofs_per_cell; ++j)
        {
          modal[i] += V_inverse(i, j) * integrator.get_dof_value(j);
        }
      }

      /*
       * Weighted reduction of modal coefficients
       */
      for(uint dof = 0; dof < dofs_per_cell; ++dof)
      {
        modal[dof] *= weight_function(dof, dofs_per_cell);
      }

      /*
       * Weighted modal coefficients are transformed back to nodal coefficients
       */
      for(uint i = 0; i < dofs_per_cell; ++i)
      {
        nodal[i] = 0.0;
        for(uint j = 0; j < dofs_per_cell; ++j)
        {
          nodal[i] += V_mat(i, j) * modal[j];
        }
        integrator.submit_dof_value(nodal[i], i);
      }
    }
    else
    {
      // No filtering
      for(uint dof = 0; dof < dofs_per_cell; ++dof)
      {
        integrator.submit_dof_value(integrator.get_dof_value(dof), dof);
      }
    }
    integrator.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number>
dealii::VectorizedArray<Number>
ModalFilterOperator<dim, Number>::weight_function(uint dof, uint dofs_per_cell)
{
  scalar weight(0.0);
  if(dof == 0)
  {
    weight = 1.0;
  }
  return weight;
}

template<int dim, typename Number>
void
ModalFilterOperator<dim, Number>::apply_modal_filter(VectorType const & src, VectorType & dst)
{
  evaluate_critical_gradients(src);
  apply_weighted_relaxation(src, dst);
}

template<int dim, typename Number>
void
ModalFilterOperator<dim, Number>::convert_unit_cell_to_reference_cell(
  std::vector<dealii::Point<dim>> const & src,
  std::vector<dealii::Point<dim>> &       dst)
{
  dst.resize(src.size());
  for(uint i = 0; i < src.size(); ++i)
  {
    for(uint d = 0; d < dim; ++d)
    {
      dst[i][d] = 2.0 * src[i][d] - 1.0;
    }
  }
}

template<int dim, typename Number>
void
ModalFilterOperator<dim, Number>::evaluate_vandermonde_matrix(IntegratorCell const & integrator,
                                                              dealii::FullMatrix<Number> & V,
                                                              dealii::FullMatrix<Number> & V_inv,
                                                              uint                         cell_id)
{
  const dealii::DoFHandler<dim> &    dof_handler = matrix_free->get_dof_handler(dof_index);
  const dealii::FiniteElement<dim> & fe          = dof_handler.get_fe();

  const uint n_dofs    = fe.dofs_per_cell;
  const uint fe_degree = fe.degree;

  V.reinit(n_dofs, n_dofs);
  V_inv.reinit(n_dofs, n_dofs);

  const std::vector<dealii::Point<dim>> & unit_support_points = fe.get_unit_support_points();
  std::vector<dealii::Point<dim>>         reference_support_points(unit_support_points.size());

  convert_unit_cell_to_reference_cell(unit_support_points, reference_support_points);

  for(uint row = 0; row < n_dofs; ++row)
  {
    Number x = reference_support_points[row][0];
    if(dim == 1)
    {
      for(uint col = 0; col <= fe_degree; ++col)
      {
        V[row][col] = std::pow(x, col);
      }
    }
    else if(dim == 2)
    {
      Number y   = reference_support_points[row][1];
      uint   col = 0;
      for(uint py = 0; py <= fe_degree; ++py)
      {
        for(uint px = 0; px <= fe_degree; ++px)
        {
          V[row][col] = std::pow(x, px) * std::pow(y, py);
          ++col;
        }
      }
    }
    else if(dim == 3)
    {
      Number y   = reference_support_points[row][1];
      Number z   = reference_support_points[row][2];
      uint   col = 0;
      for(uint pz = 0; pz <= fe_degree; ++pz)
      {
        for(uint py = 0; py <= fe_degree; ++py)
        {
          for(uint px = 0; px <= fe_degree; ++px)
          {
            V[row][col] = std::pow(x, px) * std::pow(y, py) * std::pow(z, pz);
            ++col;
          }
        }
      }
    }
  }
  V_inv.invert(V);
}

template class ModalFilterOperator<2, float>;
template class ModalFilterOperator<2, double>;
template class ModalFilterOperator<3, float>;
template class ModalFilterOperator<3, double>;

} // namespace RANS
} // namespace ExaDG
