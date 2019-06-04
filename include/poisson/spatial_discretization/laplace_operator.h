#ifndef LAPLACE_OPERATOR_H
#define LAPLACE_OPERATOR_H

#include "../../operators/interior_penalty_parameter.h"
#include "../../operators/operator_base.h"
#include "../../operators/operator_type.h"

#include "../user_interface/boundary_descriptor.h"

namespace Poisson
{
template<int dim>
struct LaplaceOperatorData : public OperatorBaseData
{
public:
  LaplaceOperatorData()
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
      degree_mapping(1)
  {
    this->mapping_update_flags = update_gradients | update_JxW_values;
    this->mapping_update_flags_inner_faces =
      this->mapping_update_flags | update_values | update_normal_vectors;
    this->mapping_update_flags_boundary_faces =
      this->mapping_update_flags_inner_faces | update_quadrature_points;
  }

  double       IP_factor;
  unsigned int degree;
  int          degree_mapping;

  std::shared_ptr<Poisson::BoundaryDescriptor<dim>> bc;
};

template<int dim, typename Number>
class LaplaceOperator : public OperatorBase<dim, Number, LaplaceOperatorData<dim>>
{
private:
  typedef OperatorBase<dim, Number, LaplaceOperatorData<dim>> Base;

  typedef typename Base::FEEvalCell FEEvalCell;
  typedef typename Base::FEEvalFace FEEvalFace;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

public:
  typedef Number                    value_type;
  typedef typename Base::VectorType VectorType;

  LaplaceOperator();

  void
  reinit(MatrixFree<dim, Number> const &   mf_data,
         AffineConstraints<double> const & constraint_matrix,
         LaplaceOperatorData<dim> const &  operator_data) const;

private:
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_value_flux(scalar const & value_m, scalar const & value_p) const;

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
                                LaplaceOperatorData<dim> const &     operator_data,
                                std::set<types::boundary_id> const & periodic_boundary_ids) const;

  // stores the penalty parameter of the interior penalty method for each cell
  mutable AlignedVector<scalar> array_penalty_parameter;

  mutable scalar tau;
};

} // namespace Poisson

#endif
