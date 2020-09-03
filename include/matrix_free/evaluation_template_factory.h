#ifndef INCLUDE_MATRIX_FREE_EVALUATION_TEMPLATE_FACTORY_H_
#define INCLUDE_MATRIX_FREE_EVALUATION_TEMPLATE_FACTORY_H_

#include <deal.II/base/config.h>

#include <deal.II/matrix_free/dof_info.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/shape_info.h>

DEAL_II_NAMESPACE_OPEN

namespace internal
{
template<int dim, typename Number, typename VectorizedArrayType = VectorizedArray<Number>>
struct FEEvaluationFactory
{
  static void
  evaluate(const unsigned int                                          n_components,
           const EvaluationFlags::EvaluationFlags                      evaluation_flag,
           const MatrixFreeFunctions::ShapeInfo<VectorizedArrayType> & shape_info,
           VectorizedArrayType *                                       values_dofs_actual,
           VectorizedArrayType *                                       values_quad,
           VectorizedArrayType *                                       gradients_quad,
           VectorizedArrayType *                                       hessians_quad,
           VectorizedArrayType *                                       scratch_data);

  static void
  integrate(const unsigned int                                          n_components,
            const EvaluationFlags::EvaluationFlags                      integration_flag,
            const MatrixFreeFunctions::ShapeInfo<VectorizedArrayType> & shape_info,
            VectorizedArrayType *                                       values_dofs_actual,
            VectorizedArrayType *                                       values_quad,
            VectorizedArrayType *                                       gradients_quad,
            VectorizedArrayType *                                       scratch_data,
            const bool                                                  sum_into_values_array);
};



template<int dim,
         int n_components,
         typename Number,
         typename VectorizedArrayType = VectorizedArray<Number>>
struct FEFaceEvaluationFactory
{
  static void
  evaluate(const MatrixFreeFunctions::ShapeInfo<VectorizedArrayType> & data,
           const VectorizedArrayType *                                 values_array,
           VectorizedArrayType *                                       values_quad,
           VectorizedArrayType *                                       gradients_quad,
           VectorizedArrayType *                                       scratch_data,
           const bool                                                  evaluate_values,
           const bool                                                  evaluate_gradients,
           const unsigned int                                          face_no,
           const unsigned int                                          subface_index,
           const unsigned int                                          face_orientation,
           const Table<2, unsigned int> &                              orientation_map);

  static void
  integrate(const MatrixFreeFunctions::ShapeInfo<VectorizedArrayType> & data,
            VectorizedArrayType *                                       values_array,
            VectorizedArrayType *                                       values_quad,
            VectorizedArrayType *                                       gradients_quad,
            VectorizedArrayType *                                       scratch_data,
            const bool                                                  integrate_values,
            const bool                                                  integrate_gradients,
            const unsigned int                                          face_no,
            const unsigned int                                          subface_index,
            const unsigned int                                          face_orientation,
            const Table<2, unsigned int> &                              orientation_map);

  static bool
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
                  const Table<2, unsigned int> &                     orientation_map);

  static bool
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
                    const Table<2, unsigned int> &                     orientation_map);
};



template<int dim,
         int n_components,
         typename Number,
         typename VectorizedArrayType = VectorizedArray<Number>>
struct CellwiseInverseMassFactory
{
  static void
  apply(const unsigned int                                                              fe_degree,
        const FEEvaluationBase<dim, n_components, Number, false, VectorizedArrayType> & fe_eval,
        const VectorizedArrayType *                                                     in_array,
        VectorizedArrayType *                                                           out_array);

  static void
  apply(const unsigned int                         n_desired_components,
        const unsigned int                         fe_degree,
        const AlignedVector<VectorizedArrayType> & inverse_shape,
        const AlignedVector<VectorizedArrayType> & inverse_coefficients,
        const VectorizedArrayType *                in_array,
        VectorizedArrayType *                      out_array);

  static void
  transform_from_q_points_to_basis(const unsigned int                         n_desired_components,
                                   const unsigned int                         fe_degree,
                                   const AlignedVector<VectorizedArrayType> & inverse_shape,
                                   const VectorizedArrayType *                in_array,
                                   VectorizedArrayType *                      out_array);
};

} // namespace internal

DEAL_II_NAMESPACE_CLOSE

#endif // EVALUATION_TEMPLATE_FACTORY_H
