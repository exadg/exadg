#ifndef CONV_DIFF_DIFFUSIVE_OPERATOR
#define CONV_DIFF_DIFFUSIVE_OPERATOR

#include "../../../operators/operator_base.h"
#include "../../user_interface/boundary_descriptor.h"
#include "../../user_interface/input_parameters.h"

namespace ConvDiff
{
template<int dim>
struct DiffusiveOperatorData : public OperatorBaseData<dim>
{
  DiffusiveOperatorData()
    // clang-format off
    : OperatorBaseData<dim>(0, 0,
          false, true, false, false, true, false, // cell
          true,  true,        true,  true         // face
      ),
      // clang-format on
      IP_factor(1.0),
      diffusivity(1.0)
  {
    this->mapping_update_flags = update_gradients | update_JxW_values | update_quadrature_points;
    this->mapping_update_flags_inner_faces =
      this->mapping_update_flags | update_values | update_normal_vectors;
    this->mapping_update_flags_boundary_faces = this->mapping_update_flags_inner_faces;
  }

  double IP_factor;
  double diffusivity;

  std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> bc;
};

template<int dim, int degree, typename Number>
class DiffusiveOperator : public OperatorBase<dim, degree, Number, DiffusiveOperatorData<dim>>
{
public:
  typedef DiffusiveOperator<dim, degree, Number> This;

  typedef OperatorBase<dim, degree, Number, DiffusiveOperatorData<dim>> Parent;

  typedef typename Parent::FEEvalCell FEEvalCell;
  typedef typename Parent::FEEvalFace FEEvalFace;
  typedef typename Parent::VectorType VectorType;

  typedef VectorizedArray<Number> scalar;

  DiffusiveOperator() : diffusivity(-1.0)
  {
  }

  void
  initialize(Mapping<dim> const &               mapping,
             MatrixFree<dim, Number> const &    mf_data,
             DiffusiveOperatorData<dim> const & operator_data_in,
             unsigned int                       level_mg_handler = numbers::invalid_unsigned_int);

  void
  initialize(Mapping<dim> const &               mapping,
             MatrixFree<dim, Number> const &    mf_data,
             AffineConstraints<double> const &  constraint_matrx,
             DiffusiveOperatorData<dim> const & operator_data_in,
             unsigned int                       level_mg_handler = numbers::invalid_unsigned_int);

  void
  apply_add(VectorType & dst, VectorType const & src, Number const time) const;

  void
  apply_add(VectorType & dst, VectorType const & src) const;

private:
  /*
   *  Calculation of "value_flux".
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_value_flux(scalar const & jump_value) const;

  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_interior_value(unsigned int const   q,
                             FEEvalFace const &   fe_eval,
                             OperatorType const & operator_type) const;

  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_exterior_value(scalar const &           value_m,
                             unsigned int const       q,
                             FEEvalFace const &       fe_eval,
                             OperatorType const &     operator_type,
                             BoundaryType const &     boundary_type,
                             types::boundary_id const boundary_id) const;

  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_gradient_flux(scalar const & normal_gradient_m,
                            scalar const & normal_gradient_p,
                            scalar const & jump_value,
                            scalar const & penalty_parameter) const;

  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_interior_normal_gradient(unsigned int const   q,
                                       FEEvalFace const &   fe_eval,
                                       OperatorType const & operator_type) const;

  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_exterior_normal_gradient(scalar const &           normal_gradient_m,
                                       unsigned int const       q,
                                       FEEvalFace const &       fe_eval,
                                       OperatorType const &     operator_type,
                                       BoundaryType const &     boundary_type,
                                       types::boundary_id const boundary_id) const;

  void
  do_cell_integral(FEEvalCell & fe_eval) const;

  void
  do_face_integral(FEEvalFace & fe_eval, FEEvalFace & fe_eval_neighbor) const;

  void
  do_face_int_integral(FEEvalFace & fe_eval, FEEvalFace & fe_eval_neighbor) const;

  void
  do_face_ext_integral(FEEvalFace & fe_eval, FEEvalFace & fe_eval_neighbor) const;

  void
  do_boundary_integral(FEEvalFace &               fe_eval,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id) const;

  void
  do_verify_boundary_conditions(types::boundary_id const             boundary_id,
                                DiffusiveOperatorData<dim> const &   operator_data,
                                std::set<types::boundary_id> const & periodic_boundary_ids) const;

  AlignedVector<scalar> array_penalty_parameter;
  double                diffusivity;
};
} // namespace ConvDiff

#endif
