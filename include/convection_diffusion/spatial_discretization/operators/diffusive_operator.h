#ifndef CONV_DIFF_DIFFUSIVE_OPERATOR
#define CONV_DIFF_DIFFUSIVE_OPERATOR

#include "../../../operators/operator_base.h"
#include "../../user_interface/boundary_descriptor.h"
#include "../../user_interface/input_parameters.h"

namespace ConvDiff
{
template<int dim>
struct DiffusiveOperatorData : public OperatorBaseData
{
  DiffusiveOperatorData()
    // clang-format off
    : OperatorBaseData(
          0, // dof_index
          0, // quad_index
          false, true, false, // cell evaluate
          false, true, false, // cell integrate
          true,  true,        // face evaluate
          true,  true         // face integrate
      ),
      // clang-format on
      IP_factor(1.0),
      degree(1),
      degree_mapping(1),
      diffusivity(1.0)
  {
    this->mapping_update_flags = update_gradients | update_JxW_values | update_quadrature_points;
    this->mapping_update_flags_inner_faces =
      this->mapping_update_flags | update_values | update_normal_vectors;
    this->mapping_update_flags_boundary_faces = this->mapping_update_flags_inner_faces;
  }

  double       IP_factor;
  unsigned int degree;
  int          degree_mapping;
  double       diffusivity;

  std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> bc;
};

template<int dim, typename Number>
class DiffusiveOperator : public OperatorBase<dim, Number, DiffusiveOperatorData<dim>>
{
private:
  typedef OperatorBase<dim, Number, DiffusiveOperatorData<dim>> Base;

  typedef typename Base::VectorType VectorType;

  typedef typename Base::FEEvalCell FEEvalCell;
  typedef typename Base::FEEvalFace FEEvalFace;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

public:
  DiffusiveOperator();

  void
  reinit(MatrixFree<dim, Number> const &    matrix_free,
         AffineConstraints<double> const &  constraint_matrix,
         DiffusiveOperatorData<dim> const & operator_data) const;

private:
  /*
   *  Calculation of "value_flux".
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_value_flux(scalar const & value_m, scalar const & value_p) const;

  /*
   *  Calculation of "gradient_flux".
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_gradient_flux(scalar const & normal_gradient_m,
                            scalar const & normal_gradient_p,
                            scalar const & value_m,
                            scalar const & value_p,
                            scalar const & penalty_parameter) const;

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux(FEEvalCell & fe_eval, unsigned int const q) const;

  void
  reinit_face(unsigned int const face) const;

  void
  reinit_boundary_face(unsigned int const face) const;

  void
  reinit_face_cell_based(unsigned int const       cell,
                         unsigned int const       face,
                         types::boundary_id const boundary_id) const;

  void
  do_cell_integral(FEEvalCell & fe_eval) const;

  void
  do_face_integral(FEEvalFace & fe_eval_m, FEEvalFace & fe_eval_p) const;

  void
  do_face_int_integral(FEEvalFace & fe_eval_m, FEEvalFace & fe_eval_p) const;

  void
  do_face_ext_integral(FEEvalFace & fe_eval_m, FEEvalFace & fe_eval_p) const;

  void
  do_boundary_integral(FEEvalFace &               fe_eval_m,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id) const;

  void
  do_verify_boundary_conditions(types::boundary_id const             boundary_id,
                                DiffusiveOperatorData<dim> const &   operator_data,
                                std::set<types::boundary_id> const & periodic_boundary_ids) const;

  mutable AlignedVector<scalar> array_penalty_parameter;
  mutable double                diffusivity;
  mutable scalar                tau;
};
} // namespace ConvDiff

#endif
