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
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/matrix_free/fe_evaluation.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/transfers/mg_transfer_p.h>

namespace ExaDG
{
namespace
{
template<typename Number>
void
convert_to_eo(dealii::AlignedVector<dealii::VectorizedArray<Number>> & shape_values,
              dealii::AlignedVector<dealii::VectorizedArray<Number>> & shape_values_eo,
              unsigned int                                             fe_degree,
              unsigned int                                             n_q_points_1d)
{
  unsigned int const stride = (n_q_points_1d + 1) / 2;
  shape_values_eo.resize((fe_degree + 1) * stride);
  for(unsigned int i = 0; i < (fe_degree + 1) / 2; ++i)
  {
    for(unsigned int q = 0; q < stride; ++q)
    {
      shape_values_eo[i * stride + q] =
        0.5 * (shape_values[i * n_q_points_1d + q] +
               shape_values[i * n_q_points_1d + n_q_points_1d - 1 - q]);
      shape_values_eo[(fe_degree - i) * stride + q] =
        0.5 * (shape_values[i * n_q_points_1d + q] -
               shape_values[i * n_q_points_1d + n_q_points_1d - 1 - q]);
    }
  }
  if(fe_degree % 2 == 0)
  {
    for(unsigned int q = 0; q < stride; ++q)
    {
      shape_values_eo[fe_degree / 2 * stride + q] =
        shape_values[(fe_degree / 2) * n_q_points_1d + q];
    }
  }
}

template<typename Number>
void
fill_shape_values(dealii::AlignedVector<dealii::VectorizedArray<Number>> & shape_values,
                  unsigned int                                             fe_degree_src,
                  unsigned int                                             fe_degree_dst,
                  bool                                                     do_transpose)
{
  dealii::FullMatrix<double> matrix(fe_degree_dst + 1, fe_degree_src + 1);

  if(do_transpose)
  {
    dealii::FullMatrix<double> matrix_temp(fe_degree_src + 1, fe_degree_dst + 1);
    dealii::FETools::get_projection_matrix(dealii::FE_DGQ<1>(fe_degree_dst),
                                           dealii::FE_DGQ<1>(fe_degree_src),
                                           matrix_temp);
    matrix.copy_transposed(matrix_temp);
  }
  else
  {
    dealii::FETools::get_projection_matrix(dealii::FE_DGQ<1>(fe_degree_src),
                                           dealii::FE_DGQ<1>(fe_degree_dst),
                                           matrix);
  }

  // ... and convert to linearized format
  dealii::AlignedVector<dealii::VectorizedArray<Number>> shape_values_temp;
  shape_values_temp.resize((fe_degree_src + 1) * (fe_degree_dst + 1));
  for(unsigned int i = 0; i < fe_degree_src + 1; ++i)
    for(unsigned int q = 0; q < fe_degree_dst + 1; ++q)
      shape_values_temp[i * (fe_degree_dst + 1) + q] = matrix(q, i);

  convert_to_eo(shape_values_temp, shape_values, fe_degree_src, fe_degree_dst + 1);
}

template<int dim, int points, int surface, typename Number, typename Number2>
void
loop_over_face_points(Number values, Number2 w)
{
  int const d = surface / 2; // direction
  int const s = surface % 2; // left or right surface

  // collapsed iteration
  for(unsigned int i = 0; i < dealii::Utilities::pow(points, dim - d - 1); i++)
  {
    for(unsigned int j = 0; j < dealii::Utilities::pow(points, d); j++)
    {
      unsigned int index = (i * dealii::Utilities::pow(points, d + 1) +
                            (s == 0 ? 0 : (points - 1)) * dealii::Utilities::pow(points, d) + j);
      values[index]      = values[index] * w;
    }
  }
}

template<int dim, int fe_degree_1, typename Number, typename MatrixFree, typename FEEval>
void
weight_residuum(MatrixFree &                                                   data_1,
                FEEval &                                                       fe_eval1,
                unsigned int                                                   cell,
                dealii::AlignedVector<dealii::VectorizedArray<Number>> const & weights)
{
#if false
  (void) weights;
  int const          POINTS         = fe_degree_1 + 1;
  unsigned int const n_filled_lanes = data_1.n_active_entries_per_cell_batch(cell);
  for(int surface = 0; surface < 2 * dim; surface++)
  {
    dealii::VectorizedArray<Number> weights;
    weights          = 1.0;
    bool do_not_loop = true;
    for(unsigned int v = 0; v < n_filled_lanes; v++)
    {
      auto cell_i = data_1.get_cell_iterator(cell, v);
      if(not cell_i->at_boundary(surface))
      {
        weights[v]  = 0.5;
        do_not_loop = false;
      }
    }

    if(do_not_loop)
      continue;

    switch(surface)
    {
      case 0:
        loop_over_face_points<dim, POINTS, 0>(fe_eval1.begin_dof_values(), weights);
        break;
      case 1:
        loop_over_face_points<dim, POINTS, 1>(fe_eval1.begin_dof_values(), weights);
        break;
      case 2:
        loop_over_face_points<dim, POINTS, 2>(fe_eval1.begin_dof_values(), weights);
        break;
      case 3:
        loop_over_face_points<dim, POINTS, 3>(fe_eval1.begin_dof_values(), weights);
        break;
      case 4:
        loop_over_face_points<dim, POINTS, 4>(fe_eval1.begin_dof_values(), weights);
        break;
      case 5:
        loop_over_face_points<dim, POINTS, 5>(fe_eval1.begin_dof_values(), weights);
        break;
    }
  }

#else
  (void)data_1;
  auto points = fe_eval1.dofs_per_cell;
  auto values = fe_eval1.begin_dof_values();

  for(unsigned int i = 0; i < points; i++)
    values[i] = values[i] * weights[i + points * cell];
#endif
}
} // namespace


template<int dim, typename Number, typename VectorType, int components>
template<int fe_degree_1, int fe_degree_2>
void
MGTransferP<dim, Number, VectorType, components>::do_interpolate(VectorType &       dst,
                                                                 VectorType const & src) const
{
  dealii::FEEvaluation<dim, fe_degree_1, fe_degree_1 + 1, components, Number> fe_eval1(
    *matrixfree_1, dof_handler_index, quad_index);
  dealii::FEEvaluation<dim, fe_degree_2, fe_degree_2 + 1, components, Number> fe_eval2(
    *matrixfree_2, dof_handler_index, quad_index);

  for(unsigned int cell = 0; cell < matrixfree_1->n_cell_batches(); ++cell)
  {
    fe_eval1.reinit(cell);
    fe_eval2.reinit(cell);

    fe_eval1.read_dof_values(src);

    dealii::internal::FEEvaluationImplBasisChange<
      dealii::internal::evaluate_evenodd,
      dealii::internal::EvaluatorQuantity::value,
      dim,
      fe_degree_2 + 1,
      fe_degree_1 + 1,
      dealii::VectorizedArray<Number>,
      dealii::VectorizedArray<Number>>::do_backward(components,
                                                    interpolation_matrix_1d,
                                                    false,
                                                    fe_eval1.begin_dof_values(),
                                                    fe_eval1.begin_dof_values());

    for(unsigned int q = 0; q < fe_eval2.dofs_per_cell; ++q)
      fe_eval2.begin_dof_values()[q] = fe_eval1.begin_dof_values()[q];

    fe_eval2.set_dof_values(dst);
  }
}

template<int dim, typename Number, typename VectorType, int components>
template<int fe_degree_1, int fe_degree_2>
void
MGTransferP<dim, Number, VectorType, components>::do_restrict_and_add(VectorType &       dst,
                                                                      VectorType const & src) const
{
  dealii::FEEvaluation<dim, fe_degree_1, fe_degree_1 + 1, components, Number> fe_eval1(
    *matrixfree_1, dof_handler_index, quad_index);
  dealii::FEEvaluation<dim, fe_degree_2, fe_degree_2 + 1, components, Number> fe_eval2(
    *matrixfree_2, dof_handler_index, quad_index);

  for(unsigned int cell = 0; cell < matrixfree_1->n_cell_batches(); ++cell)
  {
    fe_eval1.reinit(cell);
    fe_eval2.reinit(cell);

    fe_eval1.read_dof_values(src);

    if(not(is_dg))
      weight_residuum<dim, fe_degree_1, Number>(*matrixfree_1, fe_eval1, cell, this->weights);

    dealii::internal::FEEvaluationImplBasisChange<
      dealii::internal::evaluate_evenodd,
      dealii::internal::EvaluatorQuantity::value,
      dim,
      fe_degree_2 + 1,
      fe_degree_1 + 1,
      dealii::VectorizedArray<Number>,
      dealii::VectorizedArray<Number>>::do_backward(components,
                                                    prolongation_matrix_1d,
                                                    false,
                                                    fe_eval1.begin_dof_values(),
                                                    fe_eval1.begin_dof_values());

    for(unsigned int q = 0; q < fe_eval2.dofs_per_cell; ++q)
      fe_eval2.begin_dof_values()[q] = fe_eval1.begin_dof_values()[q];

    fe_eval2.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number, typename VectorType, int components>
template<int fe_degree_1, int fe_degree_2>
void
MGTransferP<dim, Number, VectorType, components>::do_prolongate(VectorType &       dst,
                                                                VectorType const & src) const
{
  dealii::FEEvaluation<dim, fe_degree_1, fe_degree_1 + 1, components, Number> fe_eval1(
    *matrixfree_1, dof_handler_index, quad_index);
  dealii::FEEvaluation<dim, fe_degree_2, fe_degree_2 + 1, components, Number> fe_eval2(
    *matrixfree_2, dof_handler_index, quad_index);

  for(unsigned int cell = 0; cell < matrixfree_1->n_cell_batches(); ++cell)
  {
    fe_eval1.reinit(cell);
    fe_eval2.reinit(cell);

    fe_eval2.read_dof_values(src);

    dealii::internal::FEEvaluationImplBasisChange<
      dealii::internal::evaluate_evenodd,
      dealii::internal::EvaluatorQuantity::value,
      dim,
      fe_degree_2 + 1,
      fe_degree_1 + 1,
      dealii::VectorizedArray<Number>,
      dealii::VectorizedArray<Number>>::do_forward(components,
                                                   prolongation_matrix_1d,
                                                   fe_eval2.begin_dof_values(),
                                                   fe_eval1.begin_dof_values());

    if(not(is_dg))
      weight_residuum<dim, fe_degree_1, Number>(*matrixfree_1, fe_eval1, cell, this->weights);

    fe_eval1.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number, typename VectorType, int components>
MGTransferP<dim, Number, VectorType, components>::MGTransferP()
  : matrixfree_1(nullptr),
    matrixfree_2(nullptr),
    degree_1(0),
    degree_2(0),
    dof_handler_index(0),
    quad_index(0),
    is_dg(false)
{
}

template<int dim, typename Number, typename VectorType, int components>
MGTransferP<dim, Number, VectorType, components>::MGTransferP(
  dealii::MatrixFree<dim, value_type> const * matrixfree_1,
  dealii::MatrixFree<dim, value_type> const * matrixfree_2,
  int                                         degree_1,
  int                                         degree_2,
  int                                         dof_handler_index)
{
  reinit(matrixfree_1, matrixfree_2, degree_1, degree_2, dof_handler_index);
}

template<int dim, typename Number, typename VectorType, int components>
void
MGTransferP<dim, Number, VectorType, components>::reinit(
  dealii::MatrixFree<dim, value_type> const * matrixfree_1,
  dealii::MatrixFree<dim, value_type> const * matrixfree_2,
  int                                         degree_1,
  int                                         degree_2,
  int                                         dof_handler_index)
{
  this->matrixfree_1 = matrixfree_1;
  this->matrixfree_2 = matrixfree_2;

  this->degree_1 = degree_1;
  this->degree_2 = degree_2;

  this->dof_handler_index = dof_handler_index;

  this->quad_index                 = dealii::numbers::invalid_unsigned_int;
  unsigned int const n_q_points_1d = degree_1 + 1;
  unsigned int const n_q_points    = dealii::Utilities::pow(n_q_points_1d, dim);

  for(unsigned int quad_index = 0; quad_index < matrixfree_1->get_mapping_info().cell_data.size();
      quad_index++)
  {
    if(matrixfree_1->get_mapping_info().cell_data[quad_index].descriptor[0].n_q_points ==
       n_q_points)
    {
      this->quad_index = quad_index;
      break;
    }
  }

  AssertThrow(this->quad_index != dealii::numbers::invalid_unsigned_int,
              dealii::ExcMessage("You need for p-transfer quadrature of type k+1"));

  this->is_dg = matrixfree_1->get_dof_handler().get_fe().dofs_per_vertex == 0;

  fill_shape_values(prolongation_matrix_1d, this->degree_2, this->degree_1, false);
  fill_shape_values(interpolation_matrix_1d, this->degree_2, this->degree_1, true);

  if(not(is_dg))
  {
    dealii::LinearAlgebra::distributed::Vector<Number> vec;
    dealii::IndexSet                                   relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_level_dofs(matrixfree_1->get_dof_handler(
                                                            dof_handler_index),
                                                          matrixfree_1->get_mg_level(),
                                                          relevant_dofs);
    vec.reinit(matrixfree_1->get_dof_handler(dof_handler_index)
                 .locally_owned_mg_dofs(matrixfree_1->get_mg_level()),
               relevant_dofs,
               matrixfree_1->get_vector_partitioner()->get_mpi_communicator());

    std::vector<dealii::types::global_dof_index> dof_indices(
      dealii::Utilities::pow(this->degree_1 + 1, dim) * components);
    for(unsigned int cell = 0; cell < matrixfree_1->n_cell_batches(); ++cell)
    {
      for(unsigned int v = 0; v < matrixfree_1->n_active_entries_per_cell_batch(cell); v++)
      {
        auto cell_v = matrixfree_1->get_cell_iterator(cell, v, dof_handler_index);
        cell_v->get_mg_dof_indices(dof_indices);
        for(auto i : dof_indices)
          vec[i]++;
      }
    }

    vec.compress(dealii::VectorOperation::add);
    vec.update_ghost_values();

    weights.resize(matrixfree_1->n_cell_batches() *
                   dealii::Utilities::pow(this->degree_1 + 1, dim) * components);

    for(unsigned int cell = 0; cell < matrixfree_1->n_cell_batches(); ++cell)
    {
      for(unsigned int v = 0; v < matrixfree_1->n_active_entries_per_cell_batch(cell); v++)
      {
        auto cell_v = matrixfree_1->get_cell_iterator(cell, v, dof_handler_index);
        cell_v->get_mg_dof_indices(dof_indices);

        for(unsigned int i = 0; i < dealii::Utilities::pow(this->degree_1 + 1, dim) * components;
            i++)
          weights[cell * dealii::Utilities::pow(this->degree_1 + 1, dim) * components + i][v] =
            1.0 / vec[dof_indices[matrixfree_1->get_shape_info(dof_handler_index)
                                    .lexicographic_numbering[i]]];
      }
    }
  }
}

template<int dim, typename Number, typename VectorType, int components>
MGTransferP<dim, Number, VectorType, components>::~MGTransferP()
{
}

template<int dim, typename Number, typename VectorType, int components>
void
MGTransferP<dim, Number, VectorType, components>::interpolate(unsigned int const level,
                                                              VectorType &       dst,
                                                              VectorType const & src) const
{
  (void)level;
  if(not this->is_dg) // only if CG
    src.update_ghost_values();

  // clang-format off
  switch(this->degree_1*100+this->degree_2)
  {
    // degree  2
    case  201: do_interpolate< 2, 1>(dst, src); break;
    // degree  3
    case  301: do_interpolate< 3, 1>(dst, src); break;
    case  302: do_interpolate< 3, 2>(dst, src); break;
    // degree  4
    case  401: do_interpolate< 4, 1>(dst, src); break;
    case  402: do_interpolate< 4, 2>(dst, src); break;
    case  403: do_interpolate< 4, 3>(dst, src); break;
    // degree  5
    case  501: do_interpolate< 5, 1>(dst, src); break;
    case  502: do_interpolate< 5, 2>(dst, src); break;
    case  504: do_interpolate< 5, 4>(dst, src); break;
    // degree  6
    case  601: do_interpolate< 6, 1>(dst, src); break;
    case  603: do_interpolate< 6, 3>(dst, src); break;
    case  605: do_interpolate< 6, 5>(dst, src); break;
    // degree  7
    case  701: do_interpolate< 7, 1>(dst, src); break;
    case  703: do_interpolate< 7, 3>(dst, src); break;
    case  706: do_interpolate< 7, 6>(dst, src); break;
    // degree  8
    case  801: do_interpolate< 8, 1>(dst, src); break;
    case  804: do_interpolate< 8, 4>(dst, src); break;
    case  807: do_interpolate< 8, 7>(dst, src); break;
    // degree  9
    case  901: do_interpolate< 9, 1>(dst, src); break;
    case  904: do_interpolate< 9, 4>(dst, src); break;
    case  908: do_interpolate< 9, 8>(dst, src); break;
    // degree 10
    case 1001: do_interpolate<10, 1>(dst, src); break;
    case 1005: do_interpolate<10, 5>(dst, src); break;
    case 1009: do_interpolate<10, 9>(dst, src); break;
    // degree 11
    case 1101: do_interpolate<11, 1>(dst, src); break;
    case 1105: do_interpolate<11, 5>(dst, src); break;
    case 1110: do_interpolate<11,10>(dst, src); break;
    // degree 12
    case 1201: do_interpolate<12, 1>(dst, src); break;
    case 1206: do_interpolate<12, 6>(dst, src); break;
    case 1211: do_interpolate<12,11>(dst, src); break;
    // degree 13
    case 1301: do_interpolate<13, 1>(dst, src); break;
    case 1306: do_interpolate<13, 6>(dst, src); break;
    case 1312: do_interpolate<13,12>(dst, src); break;
    // degree 14
    case 1401: do_interpolate<14, 1>(dst, src); break;
    case 1407: do_interpolate<14, 7>(dst, src); break;
    case 1413: do_interpolate<14,13>(dst, src); break;
    // degree 15
    case 1501: do_interpolate<15, 1>(dst, src); break;
    case 1507: do_interpolate<15, 7>(dst, src); break;
    case 1514: do_interpolate<15,14>(dst, src); break;
    // error:
    default:
      AssertThrow(false, dealii::ExcMessage("MGTransferP::restrict_and_add() not implemented for this degree combination!"));
  }
  // clang-format on
}

template<int dim, typename Number, typename VectorType, int components>
void
MGTransferP<dim, Number, VectorType, components>::restrict_and_add(unsigned int const /*level*/,
                                                                   VectorType &       dst,
                                                                   VectorType const & src) const
{
  if(not this->is_dg) // only if CG
    src.update_ghost_values();

  // clang-format off
  switch(this->degree_1*100+this->degree_2)
  {
    // degree  2
    case  201: do_restrict_and_add< 2, 1>(dst, src); break;
    // degree  3
    case  301: do_restrict_and_add< 3, 1>(dst, src); break;
    case  302: do_restrict_and_add< 3, 2>(dst, src); break;
    // degree  4
    case  401: do_restrict_and_add< 4, 1>(dst, src); break;
    case  402: do_restrict_and_add< 4, 2>(dst, src); break;
    case  403: do_restrict_and_add< 4, 3>(dst, src); break;
    // degree  5
    case  501: do_restrict_and_add< 5, 1>(dst, src); break;
    case  502: do_restrict_and_add< 5, 2>(dst, src); break;
    case  504: do_restrict_and_add< 5, 4>(dst, src); break;
    // degree  6
    case  601: do_restrict_and_add< 6, 1>(dst, src); break;
    case  603: do_restrict_and_add< 6, 3>(dst, src); break;
    case  605: do_restrict_and_add< 6, 5>(dst, src); break;
    // degree  7
    case  701: do_restrict_and_add< 7, 1>(dst, src); break;
    case  703: do_restrict_and_add< 7, 3>(dst, src); break;
    case  706: do_restrict_and_add< 7, 6>(dst, src); break;
    // degree  8
    case  801: do_restrict_and_add< 8, 1>(dst, src); break;
    case  804: do_restrict_and_add< 8, 4>(dst, src); break;
    case  807: do_restrict_and_add< 8, 7>(dst, src); break;
    // degree  9
    case  901: do_restrict_and_add< 9, 1>(dst, src); break;
    case  904: do_restrict_and_add< 9, 4>(dst, src); break;
    case  908: do_restrict_and_add< 9, 8>(dst, src); break;
    // degree 10
    case 1001: do_restrict_and_add<10, 1>(dst, src); break;
    case 1005: do_restrict_and_add<10, 5>(dst, src); break;
    case 1009: do_restrict_and_add<10, 9>(dst, src); break;
    // degree 11
    case 1101: do_restrict_and_add<11, 1>(dst, src); break;
    case 1105: do_restrict_and_add<11, 5>(dst, src); break;
    case 1110: do_restrict_and_add<11,10>(dst, src); break;
    // degree 12
    case 1201: do_restrict_and_add<12, 1>(dst, src); break;
    case 1206: do_restrict_and_add<12, 6>(dst, src); break;
    case 1211: do_restrict_and_add<12,11>(dst, src); break;
    // degree 13
    case 1301: do_restrict_and_add<13, 1>(dst, src); break;
    case 1306: do_restrict_and_add<13, 6>(dst, src); break;
    case 1312: do_restrict_and_add<13,12>(dst, src); break;
    // degree 14
    case 1401: do_restrict_and_add<14, 1>(dst, src); break;
    case 1407: do_restrict_and_add<14, 7>(dst, src); break;
    case 1413: do_restrict_and_add<14,13>(dst, src); break;
    // degree 15
    case 1501: do_restrict_and_add<15, 1>(dst, src); break;
    case 1507: do_restrict_and_add<15, 7>(dst, src); break;
    case 1514: do_restrict_and_add<15,14>(dst, src); break;
    // error:
    default:
      AssertThrow(false, dealii::ExcMessage("MGTransferP::restrict_and_add() not implemented for this degree combination!"));
  }
  // clang-format on

  if(not this->is_dg) // only if CG
  {
    dst.compress(dealii::VectorOperation::add);
    src.zero_out_ghost_values();
  }
}

template<int dim, typename Number, typename VectorType, int components>
void
MGTransferP<dim, Number, VectorType, components>::prolongate_and_add(unsigned int const /*level*/,
                                                                     VectorType &       dst,
                                                                     VectorType const & src) const
{
  if(not this->is_dg) // only if CG
  {
    src.update_ghost_values();
  }

  // clang-format off
  switch(this->degree_1*100+this->degree_2)
  {
    // degree  2
    case  201: do_prolongate< 2, 1>(dst, src); break;
    // degree  3
    case  301: do_prolongate< 3, 1>(dst, src); break;
    case  302: do_prolongate< 3, 2>(dst, src); break;
    // degree  4
    case  401: do_prolongate< 4, 1>(dst, src); break;
    case  402: do_prolongate< 4, 2>(dst, src); break;
    case  403: do_prolongate< 4, 3>(dst, src); break;
    // degree  5
    case  501: do_prolongate< 5, 1>(dst, src); break;
    case  502: do_prolongate< 5, 2>(dst, src); break;
    case  504: do_prolongate< 5, 4>(dst, src); break;
    // degree  6
    case  601: do_prolongate< 6, 1>(dst, src); break;
    case  603: do_prolongate< 6, 3>(dst, src); break;
    case  605: do_prolongate< 6, 5>(dst, src); break;
    // degree  7
    case  701: do_prolongate< 7, 1>(dst, src); break;
    case  703: do_prolongate< 7, 3>(dst, src); break;
    case  706: do_prolongate< 7, 6>(dst, src); break;
    // degree  8
    case  801: do_prolongate< 8, 1>(dst, src); break;
    case  804: do_prolongate< 8, 4>(dst, src); break;
    case  807: do_prolongate< 8, 7>(dst, src); break;
    // degree  9
    case  901: do_prolongate< 9, 1>(dst, src); break;
    case  904: do_prolongate< 9, 4>(dst, src); break;
    case  908: do_prolongate< 9, 8>(dst, src); break;
    // degree 10
    case 1001: do_prolongate<10, 1>(dst, src); break;
    case 1005: do_prolongate<10, 5>(dst, src); break;
    case 1009: do_prolongate<10, 9>(dst, src); break;
    // degree 11
    case 1101: do_prolongate<11, 1>(dst, src); break;
    case 1105: do_prolongate<11, 5>(dst, src); break;
    case 1110: do_prolongate<11,10>(dst, src); break;
    // degree 12
    case 1201: do_prolongate<12, 1>(dst, src); break;
    case 1206: do_prolongate<12, 6>(dst, src); break;
    case 1211: do_prolongate<12,11>(dst, src); break;
    // degree 13
    case 1301: do_prolongate<13, 1>(dst, src); break;
    case 1306: do_prolongate<13, 6>(dst, src); break;
    case 1312: do_prolongate<13,12>(dst, src); break;
    // degree 14
    case 1401: do_prolongate<14, 1>(dst, src); break;
    case 1407: do_prolongate<14, 7>(dst, src); break;
    case 1413: do_prolongate<14,13>(dst, src); break;
    // degree 15
    case 1501: do_prolongate<15, 1>(dst, src); break;
    case 1507: do_prolongate<15, 7>(dst, src); break;
    case 1514: do_prolongate<15,14>(dst, src); break;
    // error:
    default:
      AssertThrow(false, dealii::ExcMessage("MGTransferP::prolongate() not implemented for this degree combination!"));
  }
  // clang-format on

  if(not this->is_dg) // only if CG
  {
    dst.compress(dealii::VectorOperation::add);
    src.zero_out_ghost_values();
  }
}


typedef dealii::LinearAlgebra::distributed::Vector<float>  VectorTypeFloat;
typedef dealii::LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

template class MGTransferP<2, float, VectorTypeFloat, 1>;
template class MGTransferP<2, float, VectorTypeFloat, 2>;

template class MGTransferP<3, float, VectorTypeFloat, 1>;
template class MGTransferP<3, float, VectorTypeFloat, 3>;

template class MGTransferP<2, double, VectorTypeDouble, 1>;
template class MGTransferP<2, double, VectorTypeDouble, 2>;

template class MGTransferP<3, double, VectorTypeDouble, 1>;
template class MGTransferP<3, double, VectorTypeDouble, 3>;

} // namespace ExaDG
