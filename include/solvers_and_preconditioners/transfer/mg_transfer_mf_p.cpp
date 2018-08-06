#include "mg_transfer_mf_p.h"

template<int dim, int fe_degree_1, int fe_degree_2, typename Number, typename VNumber>
MGTransferMatrixFreeP<dim, fe_degree_1, fe_degree_2, Number, VNumber>::MGTransferMatrixFreeP()
{
}

template<int dim, int fe_degree_1, int fe_degree_2, typename Number, typename VNumber>
MGTransferMatrixFreeP<dim, fe_degree_1, fe_degree_2, Number, VNumber>::MGTransferMatrixFreeP(
  const DoFHandler<dim> & dof_handler_1,
  const DoFHandler<dim> & dof_handler_2,
  const unsigned int      level)
{
  reinit(dof_handler_1, dof_handler_2, level);
}

template<int dim, int fe_degree_1, int fe_degree_2, typename Number, typename VNumber>
void
MGTransferMatrixFreeP<dim, fe_degree_1, fe_degree_2, Number, VNumber>::reinit(
  const DoFHandler<dim> & dof_handler_1,
  const DoFHandler<dim> & dof_handler_2,
  const unsigned int      level)
{
  {
    QGaussLobatto<1>                                     quadrature(fe_degree_2 + 1);
    typename MatrixFree<dim, value_type>::AdditionalData additional_data;
    additional_data.level_mg_handler = level;

    ConstraintMatrix dummy;
    dummy.close();
    data_1.reinit(dof_handler_1, dummy, quadrature, additional_data);
  }
  {
    QGaussLobatto<1>                                     quadrature(fe_degree_1 + 1);
    typename MatrixFree<dim, value_type>::AdditionalData additional_data;
    additional_data.level_mg_handler = level;

    ConstraintMatrix dummy;
    dummy.close();
    data_2.reinit(dof_handler_2, dummy, quadrature, additional_data);
  }

  fill_shape_values(shape_values_prol, fe_degree_2, fe_degree_1);
}

template<int dim, int fe_degree_1, int fe_degree_2, typename Number, typename VNumber>
void
MGTransferMatrixFreeP<dim, fe_degree_1, fe_degree_2, Number, VNumber>::initialize_dof_vector(VNumber & vec_1,
                                                                                             VNumber & vec_2)
{
  data_2.initialize_dof_vector(vec_1);
  data_1.initialize_dof_vector(vec_2);
}

template<int dim, int fe_degree_1, int fe_degree_2, typename Number, typename VNumber>
MGTransferMatrixFreeP<dim, fe_degree_1, fe_degree_2, Number, VNumber>::~MGTransferMatrixFreeP()
{
}

template<int dim, int fe_degree_1, int fe_degree_2, typename Number, typename VNumber>
void
MGTransferMatrixFreeP<dim, fe_degree_1, fe_degree_2, Number, VNumber>::restrict_and_add(
  const unsigned int /*level*/,
  VNumber &       dst,
  const VNumber & src) const
{
  data_1.cell_loop(
    &MGTransferMatrixFreeP<dim, fe_degree_1, fe_degree_2, Number, VNumber>::restrict_and_add_local,
    this,
    dst,
    src);
}

template<int dim, int fe_degree_1, int fe_degree_2, typename Number, typename VNumber>
void
MGTransferMatrixFreeP<dim, fe_degree_1, fe_degree_2, Number, VNumber>::prolongate(
  const unsigned int /*level*/,
  VNumber &       dst,
  const VNumber & src) const
{
  data_1.cell_loop(&MGTransferMatrixFreeP<dim, fe_degree_1, fe_degree_2, Number, VNumber>::prolongate_local,
                   this,
                   dst,
                   src);
}

template<int dim, int fe_degree_1, int fe_degree_2, typename Number, typename VNumber>
void
MGTransferMatrixFreeP<dim, fe_degree_1, fe_degree_2, Number, VNumber>::restrict_and_add_local(
  const MatrixFree<dim, value_type> & /*data*/,
  VNumber &                                     dst,
  const VNumber &                               src,
  const std::pair<unsigned int, unsigned int> & cell_range) const
{
  FEEvaluation<dim, fe_degree_1, fe_degree_2 + 1, 1, value_type> phi1(data_1);
  FEEvaluation<dim, fe_degree_2, fe_degree_2 + 1, 1, value_type> phi2(data_2);

  AlignedVector<VectorizedArray<Number>> temp1(std::max(phi1.n_q_points, phi2.n_q_points));
  AlignedVector<VectorizedArray<Number>> temp2(std::max(phi1.n_q_points, phi2.n_q_points));

  // iterate over all macro cells ...
  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    // ... set macro cell
    phi1.reinit(cell);
    phi2.reinit(cell);

    // ... gather dofs
    phi1.read_dof_values(src);
    phi2.read_dof_values(dst);

    // ... interpolate
    if(dim == 2)
    {
      // ... 2D
      dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                               2,
                                               fe_degree_2 + 1,
                                               fe_degree_1 + 1,
                                               VectorizedArray<Number>>
        eval_val(shape_values_prol, shape_values_prol, shape_values_prol);
      eval_val.template values<1, false, false>(phi1.begin_dof_values(), temp2.begin());
      eval_val.template values<0, false, false>(temp2.begin(), temp1.begin());
    }
    else
    {
      // ... 3D
      dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                               3,
                                               fe_degree_2 + 1,
                                               fe_degree_1 + 1,
                                               VectorizedArray<Number>>
        eval_val(shape_values_prol, shape_values_prol, shape_values_prol);
      eval_val.template values<2, false, false>(phi1.begin_dof_values(), temp1.begin());
      eval_val.template values<1, false, false>(temp1.begin(), temp2.begin());
      eval_val.template values<0, false, false>(temp2.begin(), temp1.begin());
    }

    // ... copy to phi2
    for(unsigned int q = 0; q < phi2.dofs_per_cell; ++q)
      phi2.begin_dof_values()[q] += temp1[q];

    // ... scatter dofs
    phi2.set_dof_values(dst);
  }
}

template<int dim, int fe_degree_1, int fe_degree_2, typename Number, typename VNumber>
void
MGTransferMatrixFreeP<dim, fe_degree_1, fe_degree_2, Number, VNumber>::prolongate_local(
  const MatrixFree<dim, value_type> & /*data*/,
  VNumber &                                     dst,
  const VNumber &                               src,
  const std::pair<unsigned int, unsigned int> & cell_range) const
{
  FEEvaluation<dim, fe_degree_1, fe_degree_1 + 1, 1, value_type> phi1(data_1);
  FEEvaluation<dim, fe_degree_2, fe_degree_1 + 1, 1, value_type> phi2(data_2);

  AlignedVector<VectorizedArray<Number>> temp1(std::max(phi1.n_q_points, phi2.n_q_points));
  AlignedVector<VectorizedArray<Number>> temp2(std::max(phi1.n_q_points, phi2.n_q_points));

  // iterate over all macro cells ...
  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    // ... set macro cell
    phi1.reinit(cell);
    phi2.reinit(cell);

    // ... gather dofs
    phi2.read_dof_values(src);

    // ... interpolate
    if(dim == 2)
    {
      // ... 2D
      dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                               2,
                                               fe_degree_2 + 1,
                                               fe_degree_1 + 1,
                                               VectorizedArray<Number>>
        eval_val(shape_values_prol, shape_values_prol, shape_values_prol);
      eval_val.template values<0, true, false>(phi2.begin_dof_values(), temp2.begin());
      eval_val.template values<1, true, false>(temp2.begin(), temp1.begin());
    }
    else
    {
      // ... 3D
      dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                               3,
                                               fe_degree_2 + 1,
                                               fe_degree_1 + 1,
                                               VectorizedArray<Number>>
        eval_val(shape_values_prol, shape_values_prol, shape_values_prol);
      eval_val.template values<0, true, false>(phi2.begin_dof_values(), temp1.begin());
      eval_val.template values<1, true, false>(temp1.begin(), temp2.begin());
      eval_val.template values<2, true, false>(temp2.begin(), temp1.begin());
    }

    // ... copy to phi2
    for(unsigned int q = 0; q < phi2.n_q_points; ++q)
      phi1.begin_dof_values()[q] = temp1[q];

    // ... scatter dofs
    phi1.set_dof_values(dst);
  }
}

template<int dim, int fe_degree_1, int fe_degree_2, typename Number, typename VNumber>
void
MGTransferMatrixFreeP<dim, fe_degree_1, fe_degree_2, Number, VNumber>::convert_to_eo(
  AlignedVector<VectorizedArray<Number>> & shape_values,
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
        0.5 * (shape_values[i * n_q_points_1d + q] + shape_values[i * n_q_points_1d + n_q_points_1d - 1 - q]);
      shape_values_eo[(fe_degree - i) * stride + q] =
        0.5 * (shape_values[i * n_q_points_1d + q] - shape_values[i * n_q_points_1d + n_q_points_1d - 1 - q]);
    }
  if(fe_degree % 2 == 0)
    for(unsigned int q = 0; q < stride; ++q)
    {
      shape_values_eo[fe_degree / 2 * stride + q] = shape_values[(fe_degree / 2) * n_q_points_1d + q];
    }
}

template<int dim, int fe_degree_1, int fe_degree_2, typename Number, typename VNumber>
void
MGTransferMatrixFreeP<dim, fe_degree_1, fe_degree_2, Number, VNumber>::fill_shape_values(
  AlignedVector<VectorizedArray<Number>> & shape_values,
  unsigned int                             fe_degree_src,
  unsigned int                             fe_degree_dst)
{
  FullMatrix<double> matrix(fe_degree_dst + 1, fe_degree_src + 1);
  FETools::get_projection_matrix(FE_DGQ<1>(fe_degree_src), FE_DGQ<1>(fe_degree_dst), matrix);

  // ... and convert to linearized format
  AlignedVector<VectorizedArray<Number>> shape_values_temp;
  shape_values_temp.resize((fe_degree_src + 1) * (fe_degree_dst + 1));
  for(unsigned int i = 0; i < fe_degree_src + 1; ++i)
    for(unsigned int q = 0; q < fe_degree_dst + 1; ++q)
      shape_values_temp[i * (fe_degree_dst + 1) + q] = matrix(q, i);

  convert_to_eo(shape_values_temp, shape_values, fe_degree_src, fe_degree_dst + 1);
}

#include "mg_transfer_mf_p.hpp"