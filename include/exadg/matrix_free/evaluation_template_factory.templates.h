#ifndef INCLUDE_EXADG_MATRIX_FREE_EVALUATION_TEMPLATE_FACTORY_TEMPLATES_H_
#define INCLUDE_EXADG_MATRIX_FREE_EVALUATION_TEMPLATE_FACTORY_TEMPLATES_H_

// deal.II
#include <deal.II/base/config.h>
#include <deal.II/matrix_free/evaluation_kernels.h>
#include <deal.II/matrix_free/evaluation_selector.h>

// ExaDG
#include <exadg/matrix_free/evaluation_template_factory.h>

// set the maximum polynomial degree
#define FE_EVAL_FACTORY_DEGREE_MAX 15

DEAL_II_NAMESPACE_OPEN

namespace internal
{
template<int fe_degree, int dim, typename Number, typename VectorizedArrayType>
struct FEEvaluationFactoryImplBasic
{
  static DEAL_II_ALWAYS_INLINE void
  evaluate(const unsigned int                                          n_components,
           const EvaluationFlags::EvaluationFlags                      evaluation_flag,
           const MatrixFreeFunctions::ShapeInfo<VectorizedArrayType> & shape_info,
           VectorizedArrayType *                                       values_dofs_actual,
           VectorizedArrayType *                                       values_quad,
           VectorizedArrayType *                                       gradients_quad,
           VectorizedArrayType *                                       hessians_quad,
           VectorizedArrayType *                                       scratch_data)
  {
    // n_q_points = fe_degree+1
    if(shape_info.data[0].fe_degree == fe_degree)
    {
      if(shape_info.data[0].n_q_points_1d == fe_degree + 1)
        SelectEvaluator<dim, fe_degree, fe_degree + 1, VectorizedArrayType>::evaluate(
          n_components,
          evaluation_flag,
          shape_info,
          values_dofs_actual,
          values_quad,
          gradients_quad,
          hessians_quad,
          scratch_data);
      else if(shape_info.data[0].n_q_points_1d == fe_degree + 2)
        SelectEvaluator<dim, fe_degree, fe_degree + 2, VectorizedArrayType>::evaluate(
          n_components,
          evaluation_flag,
          shape_info,
          values_dofs_actual,
          values_quad,
          gradients_quad,
          hessians_quad,
          scratch_data);
      else if(shape_info.data[0].n_q_points_1d == fe_degree)
        SelectEvaluator<dim, fe_degree, fe_degree, VectorizedArrayType>::evaluate(
          n_components,
          evaluation_flag,
          shape_info,
          values_dofs_actual,
          values_quad,
          gradients_quad,
          hessians_quad,
          scratch_data);
      else if(shape_info.data[0].n_q_points_1d == 3 * fe_degree / 2 + 1)
        SelectEvaluator<dim, fe_degree, 3 * fe_degree / 2 + 1, VectorizedArrayType>::evaluate(
          n_components,
          evaluation_flag,
          shape_info,
          values_dofs_actual,
          values_quad,
          gradients_quad,
          hessians_quad,
          scratch_data);
      else if(shape_info.data[0].n_q_points_1d == 3 * (fe_degree + 1) / 2 + 1)
        SelectEvaluator<dim, fe_degree, 3 * (fe_degree + 1) / 2 + 1, VectorizedArrayType>::evaluate(
          n_components,
          evaluation_flag,
          shape_info,
          values_dofs_actual,
          values_quad,
          gradients_quad,
          hessians_quad,
          scratch_data);
      else if(shape_info.data[0].n_q_points_1d == 2 * fe_degree + 1)
        SelectEvaluator<dim, fe_degree, 2 * fe_degree + 1, VectorizedArrayType>::evaluate(
          n_components,
          evaluation_flag,
          shape_info,
          values_dofs_actual,
          values_quad,
          gradients_quad,
          hessians_quad,
          scratch_data);
      else
      {
        AssertThrow(false,
                    ExcNotImplemented("Specific quadrature choice " +
                                      std::to_string(shape_info.data[0].n_q_points_1d) +
                                      " for degree " + std::to_string(fe_degree) +
                                      " not available"));
      }
    }
    else if(fe_degree < FE_EVAL_FACTORY_DEGREE_MAX)
    {
      FEEvaluationFactoryImplBasic<(fe_degree < FE_EVAL_FACTORY_DEGREE_MAX ? fe_degree + 1 :
                                                                             fe_degree),
                                   dim,
                                   Number,
                                   VectorizedArrayType>::evaluate(n_components,
                                                                  evaluation_flag,
                                                                  shape_info,
                                                                  values_dofs_actual,
                                                                  values_quad,
                                                                  gradients_quad,
                                                                  hessians_quad,
                                                                  scratch_data);
    }
    else
    {
      AssertThrow(false,
                  ExcNotImplemented("Degree " + std::to_string(shape_info.data[0].fe_degree) +
                                    " not available. Adjust " +
                                    std::to_string(FE_EVAL_FACTORY_DEGREE_MAX)));
    }
  }



  static DEAL_II_ALWAYS_INLINE void
  integrate(const unsigned int                                          n_components,
            const EvaluationFlags::EvaluationFlags                      integration_flag,
            const MatrixFreeFunctions::ShapeInfo<VectorizedArrayType> & shape_info,
            VectorizedArrayType *                                       values_dofs_actual,
            VectorizedArrayType *                                       values_quad,
            VectorizedArrayType *                                       gradients_quad,
            VectorizedArrayType *                                       scratch_data,
            const bool                                                  sum_into_values_array)
  {
    // n_q_points = fe_degree+1
    if(shape_info.data[0].fe_degree == fe_degree)
    {
      if(shape_info.data[0].n_q_points_1d == fe_degree + 1)
        SelectEvaluator<dim, fe_degree, fe_degree + 1, VectorizedArrayType>::integrate(
          n_components,
          integration_flag,
          shape_info,
          values_dofs_actual,
          values_quad,
          gradients_quad,
          scratch_data,
          sum_into_values_array);
      else if(shape_info.data[0].n_q_points_1d == fe_degree + 2)
        SelectEvaluator<dim, fe_degree, fe_degree + 2, VectorizedArrayType>::integrate(
          n_components,
          integration_flag,
          shape_info,
          values_dofs_actual,
          values_quad,
          gradients_quad,
          scratch_data,
          sum_into_values_array);
      else if(shape_info.data[0].n_q_points_1d == fe_degree)
        SelectEvaluator<dim, fe_degree, fe_degree, VectorizedArrayType>::integrate(
          n_components,
          integration_flag,
          shape_info,
          values_dofs_actual,
          values_quad,
          gradients_quad,
          scratch_data,
          sum_into_values_array);
      else if(shape_info.data[0].n_q_points_1d == 3 * fe_degree / 2 + 1)
        SelectEvaluator<dim, fe_degree, 3 * fe_degree / 2 + 1, VectorizedArrayType>::integrate(
          n_components,
          integration_flag,
          shape_info,
          values_dofs_actual,
          values_quad,
          gradients_quad,
          scratch_data,
          sum_into_values_array);
      else if(shape_info.data[0].n_q_points_1d == 3 * (fe_degree + 1) / 2 + 1)
        SelectEvaluator<dim, fe_degree, 3 * (fe_degree + 1) / 2 + 1, VectorizedArrayType>::
          integrate(n_components,
                    integration_flag,
                    shape_info,
                    values_dofs_actual,
                    values_quad,
                    gradients_quad,
                    scratch_data,
                    sum_into_values_array);
      else if(shape_info.data[0].n_q_points_1d == 2 * fe_degree + 1)
        SelectEvaluator<dim, fe_degree, 2 * fe_degree + 1, VectorizedArrayType>::integrate(
          n_components,
          integration_flag,
          shape_info,
          values_dofs_actual,
          values_quad,
          gradients_quad,
          scratch_data,
          sum_into_values_array);
      else
      {
        AssertThrow(false,
                    ExcNotImplemented("Specific quadrature choice " +
                                      std::to_string(shape_info.data[0].n_q_points_1d) +
                                      " for degree " + std::to_string(fe_degree) +
                                      " not available"));
      }
    }
    else if(fe_degree < FE_EVAL_FACTORY_DEGREE_MAX)
    {
      FEEvaluationFactoryImplBasic<(fe_degree < FE_EVAL_FACTORY_DEGREE_MAX ? fe_degree + 1 :
                                                                             fe_degree),
                                   dim,
                                   Number,
                                   VectorizedArrayType>::integrate(n_components,
                                                                   integration_flag,
                                                                   shape_info,
                                                                   values_dofs_actual,
                                                                   values_quad,
                                                                   gradients_quad,
                                                                   scratch_data,
                                                                   sum_into_values_array);
    }
    else
    {
      AssertThrow(false,
                  ExcNotImplemented("Degree " + std::to_string(shape_info.data[0].fe_degree) +
                                    " not available. Adjust " +
                                    std::to_string(FE_EVAL_FACTORY_DEGREE_MAX)));
    }
  }
};



template<int fe_degree, int dim, int n_components, typename Number, typename VectorizedArrayType>
struct FEEvaluationFactoryImpl
{
  static DEAL_II_ALWAYS_INLINE void
  evaluate_face(const MatrixFreeFunctions::ShapeInfo<VectorizedArrayType> & data,
                const VectorizedArrayType *                                 values_array,
                VectorizedArrayType *                                       values_quad,
                VectorizedArrayType *                                       gradients_quad,
                VectorizedArrayType *                                       scratch_data,
                const bool                                                  evaluate_values,
                const bool                                                  evaluate_gradients,
                const unsigned int                                          face_no,
                const unsigned int                                          subface_index,
                const unsigned int                                          face_orientation,
                const Table<2, unsigned int> &                              orientation_map)
  {
    if(data.data[0].fe_degree == fe_degree)
    {
      if(data.data[0].n_q_points_1d == fe_degree + 1)
        FEFaceEvaluationSelector<dim,
                                 fe_degree,
                                 fe_degree + 1,
                                 n_components,
                                 Number,
                                 VectorizedArrayType>::evaluate(data,
                                                                values_array,
                                                                values_quad,
                                                                gradients_quad,
                                                                scratch_data,
                                                                evaluate_values,
                                                                evaluate_gradients,
                                                                face_no,
                                                                subface_index,
                                                                face_orientation,
                                                                orientation_map);
      else if(data.data[0].n_q_points_1d == fe_degree + 2)
        FEFaceEvaluationSelector<dim,
                                 fe_degree,
                                 fe_degree + 2,
                                 n_components,
                                 Number,
                                 VectorizedArrayType>::evaluate(data,
                                                                values_array,
                                                                values_quad,
                                                                gradients_quad,
                                                                scratch_data,
                                                                evaluate_values,
                                                                evaluate_gradients,
                                                                face_no,
                                                                subface_index,
                                                                face_orientation,
                                                                orientation_map);
      else if(data.data[0].n_q_points_1d == fe_degree)
        FEFaceEvaluationSelector<dim,
                                 fe_degree,
                                 fe_degree,
                                 n_components,
                                 Number,
                                 VectorizedArrayType>::evaluate(data,
                                                                values_array,
                                                                values_quad,
                                                                gradients_quad,
                                                                scratch_data,
                                                                evaluate_values,
                                                                evaluate_gradients,
                                                                face_no,
                                                                subface_index,
                                                                face_orientation,
                                                                orientation_map);
      else if(data.data[0].n_q_points_1d == 3 * fe_degree / 2 + 1)
        FEFaceEvaluationSelector<dim,
                                 fe_degree,
                                 3 * fe_degree / 2 + 1,
                                 n_components,
                                 Number,
                                 VectorizedArrayType>::evaluate(data,
                                                                values_array,
                                                                values_quad,
                                                                gradients_quad,
                                                                scratch_data,
                                                                evaluate_values,
                                                                evaluate_gradients,
                                                                face_no,
                                                                subface_index,
                                                                face_orientation,
                                                                orientation_map);
      else if(data.data[0].n_q_points_1d == 3 * (fe_degree + 1) / 2 + 1)
        FEFaceEvaluationSelector<dim,
                                 fe_degree,
                                 3 * (fe_degree + 1) / 2 + 1,
                                 n_components,
                                 Number,
                                 VectorizedArrayType>::evaluate(data,
                                                                values_array,
                                                                values_quad,
                                                                gradients_quad,
                                                                scratch_data,
                                                                evaluate_values,
                                                                evaluate_gradients,
                                                                face_no,
                                                                subface_index,
                                                                face_orientation,
                                                                orientation_map);
      else if(data.data[0].n_q_points_1d == 2 * fe_degree + 1)
        FEFaceEvaluationSelector<dim,
                                 fe_degree,
                                 2 * fe_degree + 1,
                                 n_components,
                                 Number,
                                 VectorizedArrayType>::evaluate(data,
                                                                values_array,
                                                                values_quad,
                                                                gradients_quad,
                                                                scratch_data,
                                                                evaluate_values,
                                                                evaluate_gradients,
                                                                face_no,
                                                                subface_index,
                                                                face_orientation,
                                                                orientation_map);
      else
      {
        AssertThrow(false,
                    ExcNotImplemented("Specific quadrature choice " +
                                      std::to_string(data.data[0].n_q_points_1d) + " for degree " +
                                      std::to_string(fe_degree) + " not available"));
      }
    }
    else if(fe_degree < FE_EVAL_FACTORY_DEGREE_MAX)
    {
      FEEvaluationFactoryImpl<(fe_degree < FE_EVAL_FACTORY_DEGREE_MAX ? fe_degree + 1 : fe_degree),
                              dim,
                              n_components,
                              Number,
                              VectorizedArrayType>::evaluate_face(data,
                                                                  values_array,
                                                                  values_quad,
                                                                  gradients_quad,
                                                                  scratch_data,
                                                                  evaluate_values,
                                                                  evaluate_gradients,
                                                                  face_no,
                                                                  subface_index,
                                                                  face_orientation,
                                                                  orientation_map);
    }
    else
    {
      AssertThrow(false,
                  ExcNotImplemented("Degree " + std::to_string(data.data[0].fe_degree) +
                                    " not available. Adjust " +
                                    std::to_string(FE_EVAL_FACTORY_DEGREE_MAX)));
    }
  }



  static DEAL_II_ALWAYS_INLINE void
  integrate_face(const MatrixFreeFunctions::ShapeInfo<VectorizedArrayType> & data,
                 VectorizedArrayType *                                       values_array,
                 VectorizedArrayType *                                       values_quad,
                 VectorizedArrayType *                                       gradients_quad,
                 VectorizedArrayType *                                       scratch_data,
                 const bool                                                  integrate_values,
                 const bool                                                  integrate_gradients,
                 const unsigned int                                          face_no,
                 const unsigned int                                          subface_index,
                 const unsigned int                                          face_orientation,
                 const Table<2, unsigned int> &                              orientation_map)
  {
    if(data.data[0].fe_degree == fe_degree)
    {
      if(data.data[0].n_q_points_1d == fe_degree + 1)
        FEFaceEvaluationSelector<dim,
                                 fe_degree,
                                 fe_degree + 1,
                                 n_components,
                                 Number,
                                 VectorizedArrayType>::integrate(data,
                                                                 values_array,
                                                                 values_quad,
                                                                 gradients_quad,
                                                                 scratch_data,
                                                                 integrate_values,
                                                                 integrate_gradients,
                                                                 face_no,
                                                                 subface_index,
                                                                 face_orientation,
                                                                 orientation_map);
      else if(data.data[0].n_q_points_1d == fe_degree + 2)
        FEFaceEvaluationSelector<dim,
                                 fe_degree,
                                 fe_degree + 2,
                                 n_components,
                                 Number,
                                 VectorizedArrayType>::integrate(data,
                                                                 values_array,
                                                                 values_quad,
                                                                 gradients_quad,
                                                                 scratch_data,
                                                                 integrate_values,
                                                                 integrate_gradients,
                                                                 face_no,
                                                                 subface_index,
                                                                 face_orientation,
                                                                 orientation_map);
      else if(data.data[0].n_q_points_1d == fe_degree)
        FEFaceEvaluationSelector<dim,
                                 fe_degree,
                                 fe_degree,
                                 n_components,
                                 Number,
                                 VectorizedArrayType>::integrate(data,
                                                                 values_array,
                                                                 values_quad,
                                                                 gradients_quad,
                                                                 scratch_data,
                                                                 integrate_values,
                                                                 integrate_gradients,
                                                                 face_no,
                                                                 subface_index,
                                                                 face_orientation,
                                                                 orientation_map);
      else if(data.data[0].n_q_points_1d == 3 * fe_degree / 2 + 1)
        FEFaceEvaluationSelector<dim,
                                 fe_degree,
                                 3 * fe_degree / 2 + 1,
                                 n_components,
                                 Number,
                                 VectorizedArrayType>::integrate(data,
                                                                 values_array,
                                                                 values_quad,
                                                                 gradients_quad,
                                                                 scratch_data,
                                                                 integrate_values,
                                                                 integrate_gradients,
                                                                 face_no,
                                                                 subface_index,
                                                                 face_orientation,
                                                                 orientation_map);
      else if(data.data[0].n_q_points_1d == 3 * (fe_degree + 1) / 2 + 1)
        FEFaceEvaluationSelector<dim,
                                 fe_degree,
                                 3 * (fe_degree + 1) / 2 + 1,
                                 n_components,
                                 Number,
                                 VectorizedArrayType>::integrate(data,
                                                                 values_array,
                                                                 values_quad,
                                                                 gradients_quad,
                                                                 scratch_data,
                                                                 integrate_values,
                                                                 integrate_gradients,
                                                                 face_no,
                                                                 subface_index,
                                                                 face_orientation,
                                                                 orientation_map);
      else if(data.data[0].n_q_points_1d == 2 * fe_degree + 1)
        FEFaceEvaluationSelector<dim,
                                 fe_degree,
                                 2 * fe_degree + 1,
                                 n_components,
                                 Number,
                                 VectorizedArrayType>::integrate(data,
                                                                 values_array,
                                                                 values_quad,
                                                                 gradients_quad,
                                                                 scratch_data,
                                                                 integrate_values,
                                                                 integrate_gradients,
                                                                 face_no,
                                                                 subface_index,
                                                                 face_orientation,
                                                                 orientation_map);
      else
      {
        AssertThrow(false,
                    ExcNotImplemented("Specific quadrature choice " +
                                      std::to_string(data.data[0].n_q_points_1d) + " for degree " +
                                      std::to_string(fe_degree) + " not available"));
      }
    }
    else if(fe_degree < FE_EVAL_FACTORY_DEGREE_MAX)
    {
      FEEvaluationFactoryImpl<(fe_degree < FE_EVAL_FACTORY_DEGREE_MAX ? fe_degree + 1 : fe_degree),
                              dim,
                              n_components,
                              Number,
                              VectorizedArrayType>::integrate_face(data,
                                                                   values_array,
                                                                   values_quad,
                                                                   gradients_quad,
                                                                   scratch_data,
                                                                   integrate_values,
                                                                   integrate_gradients,
                                                                   face_no,
                                                                   subface_index,
                                                                   face_orientation,
                                                                   orientation_map);
    }
    else
    {
      AssertThrow(false,
                  ExcNotImplemented("Degree " + std::to_string(data.data[0].fe_degree) +
                                    " not available. Adjust " +
                                    std::to_string(FE_EVAL_FACTORY_DEGREE_MAX)));
    }
  }



  static DEAL_II_ALWAYS_INLINE bool
  gather_evaluate(const Number *                                              src_ptr,
                  const MatrixFreeFunctions::ShapeInfo<VectorizedArrayType> & data,
                  const MatrixFreeFunctions::DoFInfo &                        dof_info,
                  VectorizedArrayType *                                       values_quad,
                  VectorizedArrayType *                                       gradients_quad,
                  VectorizedArrayType *                                       scratch_data,
                  const bool                                                  evaluate_values,
                  const bool                                                  evaluate_gradients,
                  const unsigned int                                          active_fe_index,
                  const unsigned int                                 first_selected_component,
                  const unsigned int                                 cell,
                  const unsigned int                                 face_no,
                  const unsigned int                                 subface_index,
                  const MatrixFreeFunctions::DoFInfo::DoFAccessIndex dof_access_index,
                  const unsigned int                                 face_orientation,
                  const Table<2, unsigned int> &                     orientation_map)
  {
    if(data.data[0].fe_degree == fe_degree)
    {
      if(data.data[0].n_q_points_1d == fe_degree + 1)
        return FEFaceEvaluationSelector<
          dim,
          fe_degree,
          fe_degree + 1,
          n_components,
          Number,
          VectorizedArrayType>::gather_evaluate(src_ptr,
                                                data,
                                                dof_info,
                                                values_quad,
                                                gradients_quad,
                                                scratch_data,
                                                evaluate_values,
                                                evaluate_gradients,
                                                active_fe_index,
                                                first_selected_component,
                                                std::array<unsigned int, 1>{{cell}},
                                                std::array<unsigned int, 1>{{face_no}},
                                                subface_index,
                                                dof_access_index,
                                                std::array<unsigned int, 1>{{face_orientation}},
                                                orientation_map);
      else if(data.data[0].n_q_points_1d == fe_degree + 2)
        return FEFaceEvaluationSelector<
          dim,
          fe_degree,
          fe_degree + 2,
          n_components,
          Number,
          VectorizedArrayType>::gather_evaluate(src_ptr,
                                                data,
                                                dof_info,
                                                values_quad,
                                                gradients_quad,
                                                scratch_data,
                                                evaluate_values,
                                                evaluate_gradients,
                                                active_fe_index,
                                                first_selected_component,
                                                std::array<unsigned int, 1>{{cell}},
                                                std::array<unsigned int, 1>{{face_no}},
                                                subface_index,
                                                dof_access_index,
                                                std::array<unsigned int, 1>{{face_orientation}},
                                                orientation_map);
      else if(data.data[0].n_q_points_1d == fe_degree)
        return FEFaceEvaluationSelector<
          dim,
          fe_degree,
          fe_degree,
          n_components,
          Number,
          VectorizedArrayType>::gather_evaluate(src_ptr,
                                                data,
                                                dof_info,
                                                values_quad,
                                                gradients_quad,
                                                scratch_data,
                                                evaluate_values,
                                                evaluate_gradients,
                                                active_fe_index,
                                                first_selected_component,
                                                std::array<unsigned int, 1>{{cell}},
                                                std::array<unsigned int, 1>{{face_no}},
                                                subface_index,
                                                dof_access_index,
                                                std::array<unsigned int, 1>{{face_orientation}},
                                                orientation_map);
      else if(data.data[0].n_q_points_1d == 3 * fe_degree / 2 + 1)
        return FEFaceEvaluationSelector<
          dim,
          fe_degree,
          3 * fe_degree / 2 + 1,
          n_components,
          Number,
          VectorizedArrayType>::gather_evaluate(src_ptr,
                                                data,
                                                dof_info,
                                                values_quad,
                                                gradients_quad,
                                                scratch_data,
                                                evaluate_values,
                                                evaluate_gradients,
                                                active_fe_index,
                                                first_selected_component,
                                                std::array<unsigned int, 1>{{cell}},
                                                std::array<unsigned int, 1>{{face_no}},
                                                subface_index,
                                                dof_access_index,
                                                std::array<unsigned int, 1>{{face_orientation}},
                                                orientation_map);
      else if(data.data[0].n_q_points_1d == 3 * (fe_degree + 1) / 2 + 1)
        return FEFaceEvaluationSelector<
          dim,
          fe_degree,
          3 * (fe_degree + 1) / 2 + 1,
          n_components,
          Number,
          VectorizedArrayType>::gather_evaluate(src_ptr,
                                                data,
                                                dof_info,
                                                values_quad,
                                                gradients_quad,
                                                scratch_data,
                                                evaluate_values,
                                                evaluate_gradients,
                                                active_fe_index,
                                                first_selected_component,
                                                std::array<unsigned int, 1>{{cell}},
                                                std::array<unsigned int, 1>{{face_no}},
                                                subface_index,
                                                dof_access_index,
                                                std::array<unsigned int, 1>{{face_orientation}},
                                                orientation_map);
      else if(data.data[0].n_q_points_1d == 2 * fe_degree + 1)
        return FEFaceEvaluationSelector<
          dim,
          fe_degree,
          2 * fe_degree + 1,
          n_components,
          Number,
          VectorizedArrayType>::gather_evaluate(src_ptr,
                                                data,
                                                dof_info,
                                                values_quad,
                                                gradients_quad,
                                                scratch_data,
                                                evaluate_values,
                                                evaluate_gradients,
                                                active_fe_index,
                                                first_selected_component,
                                                std::array<unsigned int, 1>{{cell}},
                                                std::array<unsigned int, 1>{{face_no}},
                                                subface_index,
                                                dof_access_index,
                                                std::array<unsigned int, 1>{{face_orientation}},
                                                orientation_map);
      else
      {
        AssertThrow(false,
                    ExcNotImplemented("Specific quadrature choice " +
                                      std::to_string(data.data[0].n_q_points_1d) + " for degree " +
                                      std::to_string(fe_degree) + " not available"));
      }
    }
    else if(fe_degree < FE_EVAL_FACTORY_DEGREE_MAX)
    {
      return FEEvaluationFactoryImpl<(fe_degree < FE_EVAL_FACTORY_DEGREE_MAX ? fe_degree + 1 :
                                                                               fe_degree),
                                     dim,
                                     n_components,
                                     Number,
                                     VectorizedArrayType>::gather_evaluate(src_ptr,
                                                                           data,
                                                                           dof_info,
                                                                           values_quad,
                                                                           gradients_quad,
                                                                           scratch_data,
                                                                           evaluate_values,
                                                                           evaluate_gradients,
                                                                           active_fe_index,
                                                                           first_selected_component,
                                                                           cell,
                                                                           face_no,
                                                                           subface_index,
                                                                           dof_access_index,
                                                                           face_orientation,
                                                                           orientation_map);
    }
    else
    {
      AssertThrow(false,
                  ExcNotImplemented("Degree " + std::to_string(data.data[0].fe_degree) +
                                    " not available. Adjust " +
                                    std::to_string(FE_EVAL_FACTORY_DEGREE_MAX)));
    }
    return false;
  }



  static DEAL_II_ALWAYS_INLINE bool
  integrate_scatter(Number *                                                    dst_ptr,
                    const MatrixFreeFunctions::ShapeInfo<VectorizedArrayType> & data,
                    const MatrixFreeFunctions::DoFInfo &                        dof_info,
                    VectorizedArrayType *                                       values_array,
                    VectorizedArrayType *                                       values_quad,
                    VectorizedArrayType *                                       gradients_quad,
                    VectorizedArrayType *                                       scratch_data,
                    const bool                                                  integrate_values,
                    const bool                                                  integrate_gradients,
                    const unsigned int                                          active_fe_index,
                    const unsigned int                                 first_selected_component,
                    const unsigned int                                 cell,
                    const unsigned int                                 face_no,
                    const unsigned int                                 subface_index,
                    const MatrixFreeFunctions::DoFInfo::DoFAccessIndex dof_access_index,
                    const unsigned int                                 face_orientation,
                    const Table<2, unsigned int> &                     orientation_map)
  {
    if(data.data[0].fe_degree == fe_degree)
    {
      if(data.data[0].n_q_points_1d == fe_degree + 1)
        return FEFaceEvaluationSelector<
          dim,
          fe_degree,
          fe_degree + 1,
          n_components,
          Number,
          VectorizedArrayType>::integrate_scatter(dst_ptr,
                                                  data,
                                                  dof_info,
                                                  values_array,
                                                  values_quad,
                                                  gradients_quad,
                                                  scratch_data,
                                                  integrate_values,
                                                  integrate_gradients,
                                                  active_fe_index,
                                                  first_selected_component,
                                                  std::array<unsigned int, 1>{{cell}},
                                                  std::array<unsigned int, 1>{{face_no}},
                                                  subface_index,
                                                  dof_access_index,
                                                  std::array<unsigned int, 1>{{face_orientation}},
                                                  orientation_map);
      else if(data.data[0].n_q_points_1d == fe_degree + 2)
        return FEFaceEvaluationSelector<
          dim,
          fe_degree,
          fe_degree + 2,
          n_components,
          Number,
          VectorizedArrayType>::integrate_scatter(dst_ptr,
                                                  data,
                                                  dof_info,
                                                  values_array,
                                                  values_quad,
                                                  gradients_quad,
                                                  scratch_data,
                                                  integrate_values,
                                                  integrate_gradients,
                                                  active_fe_index,
                                                  first_selected_component,
                                                  std::array<unsigned int, 1>{{cell}},
                                                  std::array<unsigned int, 1>{{face_no}},
                                                  subface_index,
                                                  dof_access_index,
                                                  std::array<unsigned int, 1>{{face_orientation}},
                                                  orientation_map);
      else if(data.data[0].n_q_points_1d == fe_degree)
        return FEFaceEvaluationSelector<
          dim,
          fe_degree,
          fe_degree,
          n_components,
          Number,
          VectorizedArrayType>::integrate_scatter(dst_ptr,
                                                  data,
                                                  dof_info,
                                                  values_array,
                                                  values_quad,
                                                  gradients_quad,
                                                  scratch_data,
                                                  integrate_values,
                                                  integrate_gradients,
                                                  active_fe_index,
                                                  first_selected_component,
                                                  std::array<unsigned int, 1>{{cell}},
                                                  std::array<unsigned int, 1>{{face_no}},
                                                  subface_index,
                                                  dof_access_index,
                                                  std::array<unsigned int, 1>{{face_orientation}},
                                                  orientation_map);
      else if(data.data[0].n_q_points_1d == 3 * fe_degree / 2 + 1)
        return FEFaceEvaluationSelector<
          dim,
          fe_degree,
          3 * fe_degree / 2 + 1,
          n_components,
          Number,
          VectorizedArrayType>::integrate_scatter(dst_ptr,
                                                  data,
                                                  dof_info,
                                                  values_array,
                                                  values_quad,
                                                  gradients_quad,
                                                  scratch_data,
                                                  integrate_values,
                                                  integrate_gradients,
                                                  active_fe_index,
                                                  first_selected_component,
                                                  std::array<unsigned int, 1>{{cell}},
                                                  std::array<unsigned int, 1>{{face_no}},
                                                  subface_index,
                                                  dof_access_index,
                                                  std::array<unsigned int, 1>{{face_orientation}},
                                                  orientation_map);
      else if(data.data[0].n_q_points_1d == 3 * (fe_degree + 1) / 2 + 1)
        return FEFaceEvaluationSelector<
          dim,
          fe_degree,
          3 * (fe_degree + 1) / 2 + 1,
          n_components,
          Number,
          VectorizedArrayType>::integrate_scatter(dst_ptr,
                                                  data,
                                                  dof_info,
                                                  values_array,
                                                  values_quad,
                                                  gradients_quad,
                                                  scratch_data,
                                                  integrate_values,
                                                  integrate_gradients,
                                                  active_fe_index,
                                                  first_selected_component,
                                                  std::array<unsigned int, 1>{{cell}},
                                                  std::array<unsigned int, 1>{{face_no}},
                                                  subface_index,
                                                  dof_access_index,
                                                  std::array<unsigned int, 1>{{face_orientation}},
                                                  orientation_map);
      else if(data.data[0].n_q_points_1d == 2 * fe_degree + 1)
        return FEFaceEvaluationSelector<
          dim,
          fe_degree,
          2 * fe_degree + 1,
          n_components,
          Number,
          VectorizedArrayType>::integrate_scatter(dst_ptr,
                                                  data,
                                                  dof_info,
                                                  values_array,
                                                  values_quad,
                                                  gradients_quad,
                                                  scratch_data,
                                                  integrate_values,
                                                  integrate_gradients,
                                                  active_fe_index,
                                                  first_selected_component,
                                                  std::array<unsigned int, 1>{{cell}},
                                                  std::array<unsigned int, 1>{{face_no}},
                                                  subface_index,
                                                  dof_access_index,
                                                  std::array<unsigned int, 1>{{face_orientation}},
                                                  orientation_map);
      else
      {
        AssertThrow(false,
                    ExcNotImplemented("Specific quadrature choice " +
                                      std::to_string(data.data[0].n_q_points_1d) + " for degree " +
                                      std::to_string(fe_degree) + " not available"));
      }
    }
    else if(fe_degree < FE_EVAL_FACTORY_DEGREE_MAX)
    {
      return FEEvaluationFactoryImpl<
        (fe_degree < FE_EVAL_FACTORY_DEGREE_MAX ? fe_degree + 1 : fe_degree),
        dim,
        n_components,
        Number,
        VectorizedArrayType>::integrate_scatter(dst_ptr,
                                                data,
                                                dof_info,
                                                values_array,
                                                values_quad,
                                                gradients_quad,
                                                scratch_data,
                                                integrate_values,
                                                integrate_gradients,
                                                active_fe_index,
                                                first_selected_component,
                                                cell,
                                                face_no,
                                                subface_index,
                                                dof_access_index,
                                                face_orientation,
                                                orientation_map);
    }
    else
    {
      AssertThrow(false,
                  ExcNotImplemented("Degree " + std::to_string(data.data[0].fe_degree) +
                                    " not available. Adjust " +
                                    std::to_string(FE_EVAL_FACTORY_DEGREE_MAX)));
    }
    return false;
  }

  static DEAL_II_ALWAYS_INLINE void
  apply_inverse_mass_matrix(const unsigned int                         n_desired_components,
                            const unsigned int                         given_degree,
                            const AlignedVector<VectorizedArrayType> & inverse_shape,
                            const AlignedVector<VectorizedArrayType> & inverse_coefficients,
                            const VectorizedArrayType *                in_array,
                            VectorizedArrayType *                      out_array)
  {
    if(fe_degree == given_degree)
      internal::CellwiseInverseMassMatrixImpl<dim, fe_degree, VectorizedArrayType>::apply(
        n_desired_components, inverse_shape, inverse_coefficients, in_array, out_array);
    else if(fe_degree < FE_EVAL_FACTORY_DEGREE_MAX)
      FEEvaluationFactoryImpl<(fe_degree < FE_EVAL_FACTORY_DEGREE_MAX ? fe_degree + 1 : fe_degree),
                              dim,
                              n_components,
                              Number,
                              VectorizedArrayType>::apply_inverse_mass_matrix(n_desired_components,
                                                                              given_degree,
                                                                              inverse_shape,
                                                                              inverse_coefficients,
                                                                              in_array,
                                                                              out_array);
    else
    {
      AssertThrow(false,
                  ExcNotImplemented("Degree " + std::to_string(given_degree) +
                                    " not available. Adjust " +
                                    std::to_string(FE_EVAL_FACTORY_DEGREE_MAX)));
    }
  }

  static DEAL_II_ALWAYS_INLINE void
  apply_inverse_mass_matrix(
    const unsigned int                                                              given_degree,
    const FEEvaluationBase<dim, n_components, Number, false, VectorizedArrayType> & fe_eval,
    const VectorizedArrayType *                                                     in_array,
    VectorizedArrayType *                                                           out_array)
  {
    if(fe_degree == given_degree)
      internal::CellwiseInverseMassMatrixImpl<dim, fe_degree, VectorizedArrayType>::apply(
        n_components, fe_eval, in_array, out_array);
    else if(fe_degree < FE_EVAL_FACTORY_DEGREE_MAX)
      FEEvaluationFactoryImpl<(fe_degree < FE_EVAL_FACTORY_DEGREE_MAX ? fe_degree + 1 : fe_degree),
                              dim,
                              n_components,
                              Number,
                              VectorizedArrayType>::apply_inverse_mass_matrix(given_degree,
                                                                              fe_eval,
                                                                              in_array,
                                                                              out_array);
    else
    {
      AssertThrow(false,
                  ExcNotImplemented("Degree " + std::to_string(given_degree) +
                                    " not available. Adjust " +
                                    std::to_string(FE_EVAL_FACTORY_DEGREE_MAX)));
    }
  }

  static DEAL_II_ALWAYS_INLINE void
  transform_from_q_points_to_basis(const unsigned int                         n_desired_components,
                                   const unsigned int                         given_degree,
                                   const AlignedVector<VectorizedArrayType> & inverse_shape,
                                   const VectorizedArrayType *                in_array,
                                   VectorizedArrayType *                      out_array)
  {
    if(fe_degree == given_degree)
      internal::CellwiseInverseMassMatrixImpl<dim, fe_degree, VectorizedArrayType>::
        transform_from_q_points_to_basis(n_desired_components, inverse_shape, in_array, out_array);
    else if(fe_degree < FE_EVAL_FACTORY_DEGREE_MAX)
      FEEvaluationFactoryImpl<
        (fe_degree < FE_EVAL_FACTORY_DEGREE_MAX ? fe_degree + 1 : fe_degree),
        dim,
        n_components,
        Number,
        VectorizedArrayType>::transform_from_q_points_to_basis(n_desired_components,
                                                               given_degree,
                                                               inverse_shape,
                                                               in_array,
                                                               out_array);
    else
    {
      AssertThrow(false,
                  ExcNotImplemented("Degree " + std::to_string(given_degree) +
                                    " not available. Adjust " +
                                    std::to_string(FE_EVAL_FACTORY_DEGREE_MAX)));
    }
  }
};



template<int dim, typename Number, typename VectorizedArrayType>
void
FEEvaluationFactory<dim, Number, VectorizedArrayType>::evaluate(
  const unsigned int                                          n_components,
  const EvaluationFlags::EvaluationFlags                      evaluation_flag,
  const MatrixFreeFunctions::ShapeInfo<VectorizedArrayType> & shape_info,
  VectorizedArrayType *                                       values_dofs_actual,
  VectorizedArrayType *                                       values_quad,
  VectorizedArrayType *                                       gradients_quad,
  VectorizedArrayType *                                       hessians_quad,
  VectorizedArrayType *                                       scratch_data)
{
  FEEvaluationFactoryImplBasic<1, dim, Number, VectorizedArrayType>::evaluate(n_components,
                                                                              evaluation_flag,
                                                                              shape_info,
                                                                              values_dofs_actual,
                                                                              values_quad,
                                                                              gradients_quad,
                                                                              hessians_quad,
                                                                              scratch_data);
}

template<int dim, typename Number, typename VectorizedArrayType>
void
FEEvaluationFactory<dim, Number, VectorizedArrayType>::integrate(
  const unsigned int                                          n_components,
  const EvaluationFlags::EvaluationFlags                      integration_flag,
  const MatrixFreeFunctions::ShapeInfo<VectorizedArrayType> & shape_info,
  VectorizedArrayType *                                       values_dofs_actual,
  VectorizedArrayType *                                       values_quad,
  VectorizedArrayType *                                       gradients_quad,
  VectorizedArrayType *                                       scratch_data,
  const bool                                                  sum_into_values_array)
{
  FEEvaluationFactoryImplBasic<1, dim, Number, VectorizedArrayType>::integrate(
    n_components,
    integration_flag,
    shape_info,
    values_dofs_actual,
    values_quad,
    gradients_quad,
    scratch_data,
    sum_into_values_array);
}



template<int dim, int n_components, typename Number, typename VectorizedArrayType>
void
FEFaceEvaluationFactory<dim, n_components, Number, VectorizedArrayType>::evaluate(
  const MatrixFreeFunctions::ShapeInfo<VectorizedArrayType> & data,
  const VectorizedArrayType *                                 values_array,
  VectorizedArrayType *                                       values_quad,
  VectorizedArrayType *                                       gradients_quad,
  VectorizedArrayType *                                       scratch_data,
  const bool                                                  evaluate_values,
  const bool                                                  evaluate_gradients,
  const unsigned int                                          face_no,
  const unsigned int                                          subface_index,
  const unsigned int                                          face_orientation,
  const Table<2, unsigned int> &                              orientation_map)
{
  FEEvaluationFactoryImpl<1, dim, n_components, Number, VectorizedArrayType>::evaluate_face(
    data,
    values_array,
    values_quad,
    gradients_quad,
    scratch_data,
    evaluate_values,
    evaluate_gradients,
    face_no,
    subface_index,
    face_orientation,
    orientation_map);
}



template<int dim, int n_components, typename Number, typename VectorizedArrayType>
void
FEFaceEvaluationFactory<dim, n_components, Number, VectorizedArrayType>::integrate(
  const MatrixFreeFunctions::ShapeInfo<VectorizedArrayType> & data,
  VectorizedArrayType *                                       values_array,
  VectorizedArrayType *                                       values_quad,
  VectorizedArrayType *                                       gradients_quad,
  VectorizedArrayType *                                       scratch_data,
  const bool                                                  integrate_values,
  const bool                                                  integrate_gradients,
  const unsigned int                                          face_no,
  const unsigned int                                          subface_index,
  const unsigned int                                          face_orientation,
  const Table<2, unsigned int> &                              orientation_map)
{
  FEEvaluationFactoryImpl<1, dim, n_components, Number, VectorizedArrayType>::integrate_face(
    data,
    values_array,
    values_quad,
    gradients_quad,
    scratch_data,
    integrate_values,
    integrate_gradients,
    face_no,
    subface_index,
    face_orientation,
    orientation_map);
}



template<int dim, int n_components, typename Number, typename VectorizedArrayType>
bool
FEFaceEvaluationFactory<dim, n_components, Number, VectorizedArrayType>::gather_evaluate(
  const Number *                                              src_ptr,
  const MatrixFreeFunctions::ShapeInfo<VectorizedArrayType> & data,
  const MatrixFreeFunctions::DoFInfo &                        dof_info,
  VectorizedArrayType *                                       values_quad,
  VectorizedArrayType *                                       gradients_quad,
  VectorizedArrayType *                                       scratch_data,
  const bool                                                  evaluate_values,
  const bool                                                  evaluate_gradients,
  const unsigned int                                          active_fe_index,
  const unsigned int                                          first_selected_component,
  const unsigned int                                          cell,
  const unsigned int                                          face_no,
  const unsigned int                                          subface_index,
  const MatrixFreeFunctions::DoFInfo::DoFAccessIndex          dof_access_index,
  const unsigned int                                          face_orientation,
  const Table<2, unsigned int> &                              orientation_map)
{
  return FEEvaluationFactoryImpl<1, dim, n_components, Number, VectorizedArrayType>::
    gather_evaluate(src_ptr,
                    data,
                    dof_info,
                    values_quad,
                    gradients_quad,
                    scratch_data,
                    evaluate_values,
                    evaluate_gradients,
                    active_fe_index,
                    first_selected_component,
                    cell,
                    face_no,
                    subface_index,
                    dof_access_index,
                    face_orientation,
                    orientation_map);
}

template<int dim, int n_components, typename Number, typename VectorizedArrayType>
bool
FEFaceEvaluationFactory<dim, n_components, Number, VectorizedArrayType>::integrate_scatter(
  Number *                                                    dst_ptr,
  const MatrixFreeFunctions::ShapeInfo<VectorizedArrayType> & data,
  const MatrixFreeFunctions::DoFInfo &                        dof_info,
  VectorizedArrayType *                                       values_array,
  VectorizedArrayType *                                       values_quad,
  VectorizedArrayType *                                       gradients_quad,
  VectorizedArrayType *                                       scratch_data,
  const bool                                                  integrate_values,
  const bool                                                  integrate_gradients,
  const unsigned int                                          active_fe_index,
  const unsigned int                                          first_selected_component,
  const unsigned int                                          cell,
  const unsigned int                                          face_no,
  const unsigned int                                          subface_index,
  const MatrixFreeFunctions::DoFInfo::DoFAccessIndex          dof_access_index,
  const unsigned int                                          face_orientation,
  const Table<2, unsigned int> &                              orientation_map)
{
  return FEEvaluationFactoryImpl<1, dim, n_components, Number, VectorizedArrayType>::
    integrate_scatter(dst_ptr,
                      data,
                      dof_info,
                      values_array,
                      values_quad,
                      gradients_quad,
                      scratch_data,
                      integrate_values,
                      integrate_gradients,
                      active_fe_index,
                      first_selected_component,
                      cell,
                      face_no,
                      subface_index,
                      dof_access_index,
                      face_orientation,
                      orientation_map);
}


template<int dim, int n_components, typename Number, typename VectorizedArrayType>
void
CellwiseInverseMassFactory<dim, n_components, Number, VectorizedArrayType>::apply(
  const unsigned int                                                              fe_degree,
  const FEEvaluationBase<dim, n_components, Number, false, VectorizedArrayType> & fe_eval,
  const VectorizedArrayType *                                                     in_array,
  VectorizedArrayType *                                                           out_array)
{
  FEEvaluationFactoryImpl<1, dim, n_components, Number, VectorizedArrayType>::
    apply_inverse_mass_matrix(fe_degree, fe_eval, in_array, out_array);
}


template<int dim, int n_components, typename Number, typename VectorizedArrayType>
void
CellwiseInverseMassFactory<dim, n_components, Number, VectorizedArrayType>::apply(
  const unsigned int                         n_desired_components,
  const unsigned int                         fe_degree,
  const AlignedVector<VectorizedArrayType> & inverse_shape,
  const AlignedVector<VectorizedArrayType> & inverse_coefficients,
  const VectorizedArrayType *                in_array,
  VectorizedArrayType *                      out_array)
{
  FEEvaluationFactoryImpl<1, dim, n_components, Number, VectorizedArrayType>::
    apply_inverse_mass_matrix(
      n_desired_components, fe_degree, inverse_shape, inverse_coefficients, in_array, out_array);
}

template<int dim, int n_components, typename Number, typename VectorizedArrayType>
void
CellwiseInverseMassFactory<dim, n_components, Number, VectorizedArrayType>::
  transform_from_q_points_to_basis(const unsigned int                         n_desired_components,
                                   const unsigned int                         fe_degree,
                                   const AlignedVector<VectorizedArrayType> & inverse_shape,
                                   const VectorizedArrayType *                in_array,
                                   VectorizedArrayType *                      out_array)
{
  FEEvaluationFactoryImpl<1, dim, n_components, Number, VectorizedArrayType>::
    transform_from_q_points_to_basis(
      n_desired_components, fe_degree, inverse_shape, in_array, out_array);
}

} // namespace internal


DEAL_II_NAMESPACE_CLOSE

#endif // EVALUATION_TEMPLATE_FACTORY_H
