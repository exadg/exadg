#ifndef LAPLACE_OPERATOR_H
#define LAPLACE_OPERATOR_H

#include "../../../include/operators/operation_base.h"
#include "../../operators/interior_penalty_parameter.h"
#include "../user_interface/boundary_descriptor.h"

#include "../../operators/operator_type.h"

namespace Poisson
{
enum class BoundaryType
{
  undefined,
  dirichlet,
  neumann
};

template<int dim>
struct LaplaceOperatorData : public OperatorBaseData<dim, BoundaryDescriptor<dim>>
{
public:
  LaplaceOperatorData()
    // clang-format off
    : OperatorBaseData<dim, BoundaryDescriptor<dim>>(0, 0,
          false, true, false, false, true, false, // cell
          true,  true,        true,  true         // face
      ),
      // clang-format on
      IP_factor(1.0)
  {
    this->mapping_update_flags = update_gradients | update_JxW_values;
    this->mapping_update_flags_inner_faces =
      this->mapping_update_flags | update_values | update_normal_vectors;
    this->mapping_update_flags_boundary_faces =
      this->mapping_update_flags_inner_faces | update_quadrature_points;
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
};

template<int dim, int degree, typename Number>
class LaplaceOperator : public OperatorBase<dim, degree, Number, LaplaceOperatorData<dim>>
{
public:
  typedef LaplaceOperator<dim, degree, Number>                        This;
  typedef OperatorBase<dim, degree, Number, LaplaceOperatorData<dim>> Parent;
  typedef typename Parent::FEEvalCell                                 FEEvalCell;
  typedef typename Parent::FEEvalFace                                 FEEvalFace;
  typedef typename Parent::VectorType                                 VectorType;

  // static constants
  static const int DIM = Parent::DIM;

  LaplaceOperator();

  void
  initialize(Mapping<dim> const &             mapping,
             MatrixFree<dim, Number> const &  mf_data,
             LaplaceOperatorData<dim> const & operator_data_in)
  {
    ConstraintMatrix constraint_matrix;
    Parent::reinit(mf_data, constraint_matrix, operator_data_in);

    // calculate penalty parameters
    IP::calculate_penalty_parameter<dim, degree, Number>(array_penalty_parameter,
                                                         *this->data,
                                                         mapping,
                                                         this->operator_settings.dof_index);
  }

  void
  initialize(Mapping<dim> const &             mapping,
             MatrixFree<dim, Number> &        mf_data,
             ConstraintMatrix &               constraint_matrix,
             LaplaceOperatorData<dim> const & operator_settings)
  {
    Parent::reinit(mf_data, constraint_matrix, operator_settings);

    // calculate penalty parameters
    IP::calculate_penalty_parameter<dim, degree, Number>(array_penalty_parameter,
                                                         *this->data,
                                                         mapping,
                                                         this->operator_settings.dof_index);
  }

  void
  reinit(const DoFHandler<dim> &   dof_handler,
         const Mapping<dim> &      mapping,
         void *                    operator_data_in,
         const MGConstrainedDoFs & mg_constrained_dofs,
         const unsigned int        level)
  {
    Parent::reinit(dof_handler, mapping, operator_data_in, mg_constrained_dofs, level);

    // calculate penalty parameters
    IP::calculate_penalty_parameter<dim, degree, Number>(array_penalty_parameter,
                                                         *this->data,
                                                         mapping,
                                                         this->operator_settings.dof_index);
  }

  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<Number>
    calculate_value_flux(VectorizedArray<Number> const & jump_value) const;

  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<Number>
    calculate_interior_value(unsigned int const   q,
                             FEEvalFace const &   fe_eval,
                             OperatorType const & operator_type) const;

  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<Number>
    calculate_exterior_value(VectorizedArray<Number> const & value_m,
                             unsigned int const              q,
                             FEEvalFace const &              fe_eval,
                             OperatorType const &            operator_type,
                             BoundaryType const &            boundary_type,
                             types::boundary_id const        boundary_id) const;

  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<Number>
    calculate_gradient_flux(VectorizedArray<Number> const & normal_gradient_m,
                            VectorizedArray<Number> const & normal_gradient_p,
                            VectorizedArray<Number> const & jump_value,
                            VectorizedArray<Number> const & penalty_parameter) const;

  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<Number>
    calculate_interior_normal_gradient(unsigned int const   q,
                                       FEEvalFace const &   fe_eval,
                                       OperatorType const & operator_type) const;

  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<Number>
    calculate_exterior_normal_gradient(VectorizedArray<Number> const & normal_gradient_m,
                                       unsigned int const              q,
                                       FEEvalFace const &              fe_eval,
                                       OperatorType const &            operator_type,
                                       BoundaryType const &            boundary_type,
                                       types::boundary_id const        boundary_id) const;

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

  MultigridOperatorBase<dim, Number> *
  get_new(unsigned int deg) const;

private:
  AlignedVector<VectorizedArray<Number>> array_penalty_parameter;
};

} // namespace Poisson

#endif