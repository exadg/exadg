#include "mg_transfer_mf_p.h"

#include <deal.II/dofs/dof_tools.h>

namespace
{
template<typename Number>
void
convert_to_eo(AlignedVector<VectorizedArray<Number>> & shape_values,
              AlignedVector<VectorizedArray<Number>> & shape_values_eo,
              unsigned int                             fe_degree,
              unsigned int                             n_q_points_1d)
{
  const unsigned int stride = (n_q_points_1d + 1) / 2;
  shape_values_eo.resize((fe_degree + 1) * stride);
  for(unsigned int i = 0; i < (fe_degree + 1) / 2; ++i)
    for(unsigned int q = 0; q < stride; ++q)
    {
      shape_values_eo[i * stride + q] =
        0.5 * (shape_values[i * n_q_points_1d + q] +
               shape_values[i * n_q_points_1d + n_q_points_1d - 1 - q]);
      shape_values_eo[(fe_degree - i) * stride + q] =
        0.5 * (shape_values[i * n_q_points_1d + q] -
               shape_values[i * n_q_points_1d + n_q_points_1d - 1 - q]);
    }
  if(fe_degree % 2 == 0)
    for(unsigned int q = 0; q < stride; ++q)
    {
      shape_values_eo[fe_degree / 2 * stride + q] =
        shape_values[(fe_degree / 2) * n_q_points_1d + q];
    }
}

template<typename Number>
void
fill_shape_values(AlignedVector<VectorizedArray<Number>> & shape_values,
                  unsigned int                             fe_degree_src,
                  unsigned int                             fe_degree_dst,
                  bool                                     do_transpose)
{
  FullMatrix<double> matrix(fe_degree_dst + 1, fe_degree_src + 1);

  if(do_transpose)
  {
    FullMatrix<double> matrix_temp(fe_degree_src + 1, fe_degree_dst + 1);
    FETools::get_projection_matrix(FE_DGQ<1>(fe_degree_dst), FE_DGQ<1>(fe_degree_src), matrix_temp);
    matrix.copy_transposed(matrix_temp);
  }
  else
    FETools::get_projection_matrix(FE_DGQ<1>(fe_degree_src), FE_DGQ<1>(fe_degree_dst), matrix);

  // ... and convert to linearized format
  AlignedVector<VectorizedArray<Number>> shape_values_temp;
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
  const int d = surface / 2; // direction
  const int s = surface % 2; // left or right surface

  // collapsed iteration
  for(unsigned int i = 0; i < Utilities::pow(points, dim - d - 1); i++)
    for(unsigned int j = 0; j < Utilities::pow(points, d); j++)
    {
      unsigned int index = (i * Utilities::pow(points, d + 1) +
                            (s == 0 ? 0 : (points - 1)) * Utilities::pow(points, d) + j);
      values[index]      = values[index] * w;
    }
}

template<int dim, int fe_degree_1, typename Number, typename MatrixFree, typename FEEval>
void
weight_residuum(MatrixFree & data_1, FEEval & fe_eval1, unsigned int cell, 
        const AlignedVector<VectorizedArray<Number>> & weights)
{
#if false
  (void) weights;
  const int          POINTS         = fe_degree_1 + 1;
  const unsigned int n_filled_lanes = data_1.n_components_filled(cell);
  for(int surface = 0; surface < 2 * dim; surface++)
  {
    VectorizedArray<Number> weights;
    weights          = 1.0;
    bool do_not_loop = true;
    for(unsigned int v = 0; v < n_filled_lanes; v++)
    {
      auto cell_i = data_1.get_cell_iterator(cell, v);
      if(!cell_i->at_boundary(surface))
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
  (void) data_1;
  const int points = Utilities::pow(fe_degree_1+1,dim);
  auto values      = fe_eval1.begin_dof_values();
  
  for(unsigned int i = 0; i < points; i++)
      values[i] = values[i] * weights[i+points * cell];
#endif
}
} // namespace


template<int dim, typename Number, typename VectorType, int components>
template<int fe_degree_1, int fe_degree_2>
void
MGTransferMFP<dim, Number, VectorType, components>::do_interpolate(VectorType &       dst,
                                                                   const VectorType & src) const
{
  FEEvaluation<dim, fe_degree_1, fe_degree_1 + 1, components, Number> fe_eval1(*data_1_cm,
                                                                               dof_handler_index,
                                                                               quad_index);
  FEEvaluation<dim, fe_degree_2, fe_degree_2 + 1, components, Number> fe_eval2(*data_2_cm,
                                                                               dof_handler_index,
                                                                               quad_index);

  for(unsigned int cell = 0; cell < data_1_cm->n_macro_cells(); ++cell)
  {
    fe_eval1.reinit(cell);
    fe_eval2.reinit(cell);

    fe_eval1.read_dof_values(src);

    internal::FEEvaluationImplBasisChange<
      internal::evaluate_evenodd,
      dim,
      fe_degree_2 + 1,
      fe_degree_1 + 1,
      components,
      VectorizedArray<Number>,
      VectorizedArray<Number>>::do_backward(interpolation_matrix_1d,
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
MGTransferMFP<dim, Number, VectorType, components>::do_restrict_and_add(
  VectorType &       dst,
  const VectorType & src) const
{
  FEEvaluation<dim, fe_degree_1, fe_degree_1 + 1, components, Number> fe_eval1(*data_1_cm,
                                                                               dof_handler_index,
                                                                               quad_index);
  FEEvaluation<dim, fe_degree_2, fe_degree_2 + 1, components, Number> fe_eval2(*data_2_cm,
                                                                               dof_handler_index,
                                                                               quad_index);

  for(unsigned int cell = 0; cell < data_1_cm->n_macro_cells(); ++cell)
  {
    fe_eval1.reinit(cell);
    fe_eval2.reinit(cell);

    fe_eval1.read_dof_values(src);

    if(!is_dg)
      weight_residuum<dim, fe_degree_1, Number>(*data_1_cm, fe_eval1, cell, this->weights);

    internal::FEEvaluationImplBasisChange<
      internal::evaluate_evenodd,
      dim,
      fe_degree_2 + 1,
      fe_degree_1 + 1,
      components,
      VectorizedArray<Number>,
      VectorizedArray<Number>>::do_backward(prolongation_matrix_1d,
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
MGTransferMFP<dim, Number, VectorType, components>::do_prolongate(VectorType &       dst,
                                                                  const VectorType & src) const
{
  FEEvaluation<dim, fe_degree_1, fe_degree_1 + 1, components, Number> fe_eval1(*data_1_cm,
                                                                               dof_handler_index,
                                                                               quad_index);
  FEEvaluation<dim, fe_degree_2, fe_degree_2 + 1, components, Number> fe_eval2(*data_2_cm,
                                                                               dof_handler_index,
                                                                               quad_index);

  for(unsigned int cell = 0; cell < data_1_cm->n_macro_cells(); ++cell)
  {
    fe_eval1.reinit(cell);
    fe_eval2.reinit(cell);

    fe_eval2.read_dof_values(src);

    internal::FEEvaluationImplBasisChange<
      internal::evaluate_evenodd,
      dim,
      fe_degree_2 + 1,
      fe_degree_1 + 1,
      components,
      VectorizedArray<Number>,
      VectorizedArray<Number>>::do_forward(prolongation_matrix_1d,
                                           fe_eval2.begin_dof_values(),
                                           fe_eval1.begin_dof_values());

    if(is_dg)
    {
      fe_eval1.set_dof_values(dst);
    }
    else
    {
      weight_residuum<dim, fe_degree_1, Number>(*data_1_cm, fe_eval1, cell, this->weights);
      fe_eval1.distribute_local_to_global(dst);
    }
  }
}

template<int dim, typename Number, typename VectorType, int components>
MGTransferMFP<dim, Number, VectorType, components>::MGTransferMFP()
{
}

template<int dim, typename Number, typename VectorType, int components>
MGTransferMFP<dim, Number, VectorType, components>::MGTransferMFP(
  const MatrixFree<dim, value_type> * data_1_cm,
  const MatrixFree<dim, value_type> * data_2_cm,
  int                                 degree_1,
  int                                 degree_2,
  int                                 dof_handler_index)
{
  reinit(data_1_cm, data_2_cm, degree_1, degree_2, dof_handler_index);
}

template<int dim, typename Number, typename VectorType, int components>
void
MGTransferMFP<dim, Number, VectorType, components>::reinit(
  const MatrixFree<dim, value_type> * data_1_cm,
  const MatrixFree<dim, value_type> * data_2_cm,
  int                                 degree_1,
  int                                 degree_2,
  int                                 dof_handler_index)
{
  this->data_1_cm = data_1_cm;
  this->data_2_cm = data_2_cm;

  this->degree_1 = degree_1;
  this->degree_2 = degree_2;

  this->dof_handler_index = dof_handler_index;

  this->quad_index                 = numbers::invalid_unsigned_int;
  const unsigned int n_q_points_1d = degree_1 + 1;
  const unsigned int n_q_points    = std::pow(n_q_points_1d, dim);

  for(unsigned int quad_index = 0; quad_index < data_1_cm->get_mapping_info().cell_data.size();
      quad_index++)
  {
    if(data_1_cm->get_mapping_info().cell_data[quad_index].descriptor[0].n_q_points == n_q_points)
    {
      this->quad_index = quad_index;
      break;
    }
  }

  AssertThrow(this->quad_index != numbers::invalid_unsigned_int,
              ExcMessage("You need for p-transfer quadrature of type k+1"));

  this->is_dg = data_1_cm->get_dof_handler().get_fe().dofs_per_vertex == 0;

  fill_shape_values(prolongation_matrix_1d, this->degree_2, this->degree_1, false);
  fill_shape_values(interpolation_matrix_1d, this->degree_2, this->degree_1, true);
  
  if(!is_dg)
  {
    LinearAlgebra::distributed::Vector<Number> vec;
    IndexSet                                   relevant_dofs;
    DoFTools::extract_locally_relevant_level_dofs(data_1_cm->get_dof_handler(dof_handler_index), 
            data_1_cm->get_level_mg_handler(), relevant_dofs);
    vec.reinit(data_1_cm->get_dof_handler(dof_handler_index).locally_owned_mg_dofs(data_1_cm->get_level_mg_handler()),
               relevant_dofs,
               data_1_cm->get_vector_partitioner()->get_mpi_communicator () );
      
    std::vector<types::global_dof_index> dof_indices(Utilities::pow(this->degree_1+1,dim));
    for (unsigned int cell = 0; cell < data_1_cm->n_macro_cells(); ++cell)
        for (unsigned int v = 0; v < data_1_cm->n_components_filled(cell); v++)
        {
            auto cell_v = data_1_cm->get_cell_iterator(cell, v, dof_handler_index);
            cell_v->get_mg_dof_indices(dof_indices);
            for(auto i : dof_indices)
              vec[i]++;
        }
    
    vec.compress(VectorOperation::add);
    vec.update_ghost_values();
    
    
    weights.resize(data_1_cm->n_macro_cells()*Utilities::pow(this->degree_1+1,dim));
    
    for (unsigned int cell = 0; cell < data_1_cm->n_macro_cells(); ++cell)
      for (unsigned int v = 0; v < data_1_cm->n_components_filled(cell); v++)
      {
        auto cell_v = data_1_cm->get_cell_iterator(cell, v, dof_handler_index);
        cell_v->get_mg_dof_indices(dof_indices);
        for(unsigned int i = 0; i < dof_indices.size(); i++)
          weights[cell * Utilities::pow(this->degree_1+1,dim) + i][v] = 
            1.0 / vec[dof_indices[data_1_cm->get_shape_info(dof_handler_index).lexicographic_numbering[i]]];
      }
    
  }
}

template<int dim, typename Number, typename VectorType, int components>
MGTransferMFP<dim, Number, VectorType, components>::~MGTransferMFP()
{
}

template<int dim, typename Number, typename VectorType, int components>
void
MGTransferMFP<dim, Number, VectorType, components>::interpolate(const unsigned int level,
                                                                VectorType &       dst,
                                                                const VectorType & src) const
{
  (void)level;
  if(!this->is_dg) // only if CG
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
      AssertThrow(false, ExcMessage("MGTransferMFP::restrict_and_add not implemented for this degree combination!"));
  }
  // clang-format on
}

template<int dim, typename Number, typename VectorType, int components>
void
MGTransferMFP<dim, Number, VectorType, components>::restrict_and_add(const unsigned int /*level*/,
                                                                     VectorType &       dst,
                                                                     const VectorType & src) const
{
  if(!this->is_dg) // only if CG
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
      AssertThrow(false, ExcMessage("MGTransferMFP::restrict_and_add not implemented for this degree combination!"));
  }
  // clang-format on

  if(!this->is_dg) // only if CG
    dst.compress(VectorOperation::add);
}

template<int dim, typename Number, typename VectorType, int components>
void
MGTransferMFP<dim, Number, VectorType, components>::prolongate(const unsigned int /*level*/,
                                                               VectorType &       dst,
                                                               const VectorType & src) const
{
  if(!this->is_dg) // only if CG
  {
    dst = 0.0;
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
      AssertThrow(false, ExcMessage("MGTransferMFP::prolongate not implemented for this degree combination!"));
  }
  // clang-format on

  if(!this->is_dg) // only if CG
    dst.compress(VectorOperation::add);
}


typedef dealii::LinearAlgebra::distributed::Vector<float>  VectorTypeFloat;
typedef dealii::LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

template class MGTransferMFP<2, float, VectorTypeFloat, 1>;
template class MGTransferMFP<2, float, VectorTypeFloat, 2>;

template class MGTransferMFP<3, float, VectorTypeFloat, 1>;
template class MGTransferMFP<3, float, VectorTypeFloat, 3>;

template class MGTransferMFP<2, double, VectorTypeDouble, 1>;
template class MGTransferMFP<2, double, VectorTypeDouble, 2>;

template class MGTransferMFP<3, double, VectorTypeDouble, 1>;
template class MGTransferMFP<3, double, VectorTypeDouble, 3>;
