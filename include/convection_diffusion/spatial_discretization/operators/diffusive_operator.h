#ifndef CONV_DIFF_DIFFUSIVE_OPERATOR
#define CONV_DIFF_DIFFUSIVE_OPERATOR

#include "../types.h"
#include "../../user_interface/input_parameters.h"
#include "../../user_interface/boundary_descriptor.h"
#include "../../../operators/operation_base.h"

namespace ConvDiff {
template <int dim>
struct DiffusiveOperatorData
    : public OperatorBaseData<dim, BoundaryType, OperatorType,
                              ConvDiff::BoundaryDescriptor<dim>> {
  DiffusiveOperatorData()
      : OperatorBaseData<dim, BoundaryType, OperatorType,
                         ConvDiff::BoundaryDescriptor<dim>>(
            0, 0, false, true, false, false, true, false, // cell
            true, true, true, true,                       // face
            true, true, true, true                        // boundary
            ),
        IP_factor(1.0), diffusivity(1.0) {}

  double IP_factor;
  double diffusivity;
};

template <int dim, int fe_degree, typename value_type>
class DiffusiveOperator : public OperatorBase<dim, fe_degree, value_type,
                                              DiffusiveOperatorData<dim>> {
public:
  typedef DiffusiveOperator<dim, fe_degree, value_type> This;
  typedef OperatorBase<dim, fe_degree, value_type, DiffusiveOperatorData<dim>>
      Parent;
  typedef typename Parent::FEEvalCell FEEvalCell;
  typedef typename Parent::FEEvalFace FEEvalFace;
  typedef typename Parent::VNumber VNumber;

  DiffusiveOperator() : diffusivity(-1.0) {}

  void initialize(Mapping<dim> const &mapping,
                  MatrixFree<dim, value_type> const &mf_data,
                  DiffusiveOperatorData<dim> const &operator_data_in);

  void apply_add(VNumber &dst, VNumber const &src) const;

  /*
   *  Calculation of "value_flux".
   */
  inline DEAL_II_ALWAYS_INLINE VectorizedArray<value_type>
  calculate_value_flux(VectorizedArray<value_type> const &jump_value) const;

  /*
   *  The following two functions calculate the interior_value/exterior_value
   *  depending on the operator type, the type of the boundary face
   *  and the given boundary conditions.
   *
   *                            +----------------------+--------------------+
   *                            | Dirichlet boundaries | Neumann boundaries |
   *  +-------------------------+----------------------+--------------------+
   *  | full operator           | phi⁺ = -phi⁻ + 2g    | phi⁺ = phi⁻        |
   *  +-------------------------+----------------------+--------------------+
   *  | homogeneous operator    | phi⁺ = -phi⁻         | phi⁺ = phi⁻        |
   *  +-------------------------+----------------------+--------------------+
   *  | inhomogeneous operator  | phi⁻ = 0, phi⁺ = 2g  | phi⁻ = 0, phi⁺ = 0 |
   *  +-------------------------+----------------------+--------------------+
   */
  inline DEAL_II_ALWAYS_INLINE VectorizedArray<value_type>
  calculate_interior_value(unsigned int const q, FEEvalFace const &fe_eval,
                           OperatorType const &operator_type) const;

  inline DEAL_II_ALWAYS_INLINE VectorizedArray<value_type>
  calculate_exterior_value(VectorizedArray<value_type> const &value_m,
                           unsigned int const q, FEEvalFace const &fe_eval,
                           OperatorType const &operator_type,
                           BoundaryType const &boundary_type,
                           types::boundary_id const boundary_id) const;

  /*
   *  Calculation of gradient flux. Strictly speaking, this value is not a
   * numerical flux since
   *  the flux is multiplied by the normal vector, i.e., "gradient_flux" =
   * numerical_flux * normal,
   *  where normal denotes the normal vector of element e⁻.
   */
  inline DEAL_II_ALWAYS_INLINE VectorizedArray<value_type>
  calculate_gradient_flux(
      VectorizedArray<value_type> const &normal_gradient_m,
      VectorizedArray<value_type> const &normal_gradient_p,
      VectorizedArray<value_type> const &jump_value,
      VectorizedArray<value_type> const &penalty_parameter) const;

  // clang-format off
  /*
   *  The following two functions calculate the interior/exterior velocity gradient
   *  in normal direction depending on the operator type, the type of the boundary face
   *  and the given boundary conditions.
   *
   *                            +-------------------------------------+---------------------------------------+
   *                            | Dirichlet boundaries                | Neumann boundaries                    |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   *  | full operator           | grad(phi⁺)*n = grad(phi⁻)*n         | grad(phi⁺)*n = -grad(phi⁻)*n + 2h     |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   *  | homogeneous operator    | grad(phi⁺)*n = grad(phi⁻)*n         | grad(phi⁺)*n = -grad(phi⁻)*n          |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   *  | inhomogeneous operator  | grad(phi⁻)*n  = 0, grad(phi⁺)*n = 0 | grad(phi⁻)*n  = 0, grad(phi⁺)*n  = 2h |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   *
   *                            +-------------------------------------+---------------------------------------+
   *                            | Dirichlet boundaries                | Neumann boundaries                    |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   *  | full operator           | {{grad(phi)}}*n = grad(phi⁻)*n      | {{grad(phi)}}*n = h                   |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   *  | homogeneous operator    | {{grad(phi)}}*n = grad(phi⁻)*n      | {{grad(phi)}}*n = 0                   |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   *  | inhomogeneous operator  | {{grad(phi)}}*n = 0                 | {{grad(phi)}}*n = h                   |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   */
  // clang-format on
  inline DEAL_II_ALWAYS_INLINE VectorizedArray<value_type>
  calculate_interior_normal_gradient(unsigned int const q,
                                     FEEvalFace const &fe_eval,
                                     OperatorType const &operator_type) const;

  inline DEAL_II_ALWAYS_INLINE VectorizedArray<value_type>
  calculate_exterior_normal_gradient(
      VectorizedArray<value_type> const &normal_gradient_m,
      unsigned int const q, FEEvalFace const &fe_eval,
      OperatorType const &operator_type, BoundaryType const &boundary_type,
      types::boundary_id const boundary_id) const;

  void do_cell_integral(FEEvalCell &fe_eval) const;

  void do_face_integral(FEEvalFace &fe_eval,
                        FEEvalFace &fe_eval_neighbor) const;

  void do_face_int_integral(FEEvalFace &fe_eval,
                            FEEvalFace &fe_eval_neighbor) const;

  void do_face_ext_integral(FEEvalFace &fe_eval,
                            FEEvalFace &fe_eval_neighbor) const;

  void do_boundary_integral(FEEvalFace &fe_eval,
                            OperatorType const &operator_type,
                            types::boundary_id const &boundary_id) const;

  AlignedVector<VectorizedArray<value_type>> array_penalty_parameter;
  double diffusivity;
};
}

#endif