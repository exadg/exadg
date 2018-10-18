#ifndef CONV_DIFF_DIFFUSIVE_OPERATOR
#define CONV_DIFF_DIFFUSIVE_OPERATOR

#include "../../../operators/operation_base.h"
#include "../../user_interface/boundary_descriptor.h"
#include "../../user_interface/input_parameters.h"
#include "../types.h"

namespace ConvDiff
{
template<int dim>
struct DiffusiveOperatorData : public OperatorBaseData<dim, ConvDiff::BoundaryDescriptor<dim>>
{
  DiffusiveOperatorData()
    // clang-format off
    : OperatorBaseData<dim, ConvDiff::BoundaryDescriptor<dim>>(0, 0,
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

  inline DEAL_II_ALWAYS_INLINE //
    BoundaryType
    get_boundary_type(types::boundary_id const & boundary_id) const
  {
    if(this->bc->dirichlet_bc.find(boundary_id) != this->bc->dirichlet_bc.end())
      return BoundaryType::dirichlet;
    else if(this->bc->neumann_bc.find(boundary_id) != this->bc->neumann_bc.end())
      return BoundaryType::neumann;

    AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));

    return BoundaryType::undefined;
  }

  double IP_factor;
  double diffusivity;
};

template<int dim, int fe_degree, typename value_type>
class DiffusiveOperator
  : public OperatorBase<dim, fe_degree, value_type, DiffusiveOperatorData<dim>>
{
public:
  typedef DiffusiveOperator<dim, fe_degree, value_type> This;

  typedef OperatorBase<dim, fe_degree, value_type, DiffusiveOperatorData<dim>> Parent;

  typedef typename Parent::FEEvalCell FEEvalCell;
  typedef typename Parent::FEEvalFace FEEvalFace;
  typedef typename Parent::VectorType VectorType;

  DiffusiveOperator() : diffusivity(-1.0)
  {
  }

  void
  initialize(Mapping<dim> const &                mapping,
             MatrixFree<dim, value_type> const & mf_data,
             DiffusiveOperatorData<dim> const &  operator_data_in,
             unsigned int                        level_mg_handler = numbers::invalid_unsigned_int);

  void
  initialize(Mapping<dim> const &                mapping,
             MatrixFree<dim, value_type> const & mf_data,
             ConstraintMatrix const &            constraint_matrx,
             DiffusiveOperatorData<dim> const &  operator_data_in,
             unsigned int                        level_mg_handler = numbers::invalid_unsigned_int);

  virtual void
  apply_add(VectorType & dst, VectorType const & src, value_type const time) const;

  virtual void
  apply_add(VectorType & dst, VectorType const & src) const;

  /*
   *  Calculation of "value_flux".
   */
  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<value_type>
    calculate_value_flux(VectorizedArray<value_type> const & jump_value) const;

  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<value_type>
    calculate_interior_value(unsigned int const   q,
                             FEEvalFace const &   fe_eval,
                             OperatorType const & operator_type) const;

  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<value_type>
    calculate_exterior_value(VectorizedArray<value_type> const & value_m,
                             unsigned int const                  q,
                             FEEvalFace const &                  fe_eval,
                             OperatorType const &                operator_type,
                             BoundaryType const &                boundary_type,
                             types::boundary_id const            boundary_id) const;

  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<value_type>
    calculate_gradient_flux(VectorizedArray<value_type> const & normal_gradient_m,
                            VectorizedArray<value_type> const & normal_gradient_p,
                            VectorizedArray<value_type> const & jump_value,
                            VectorizedArray<value_type> const & penalty_parameter) const;

  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<value_type>
    calculate_interior_normal_gradient(unsigned int const   q,
                                       FEEvalFace const &   fe_eval,
                                       OperatorType const & operator_type) const;

  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<value_type>
    calculate_exterior_normal_gradient(VectorizedArray<value_type> const & normal_gradient_m,
                                       unsigned int const                  q,
                                       FEEvalFace const &                  fe_eval,
                                       OperatorType const &                operator_type,
                                       BoundaryType const &                boundary_type,
                                       types::boundary_id const            boundary_id) const;

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

  virtual void
  do_verify_boundary_conditions(types::boundary_id const             boundary_id,
                                DiffusiveOperatorData<dim> const &   operator_data,
                                std::set<types::boundary_id> const & periodic_boundary_ids) const;

  AlignedVector<VectorizedArray<value_type>> array_penalty_parameter;
  double                                     diffusivity;
};
} // namespace ConvDiff

#endif
