#ifndef CONV_DIFF_CONVECTION_OPERATOR
#define CONV_DIFF_CONVECTION_OPERATOR

#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "../../../operators/operator_base.h"
#include "../../user_interface/boundary_descriptor.h"
#include "../../user_interface/input_parameters.h"

namespace ConvDiff
{
template<int dim>
struct ConvectiveOperatorData : public OperatorBaseData
{
  ConvectiveOperatorData()
    // clang-format off
  : OperatorBaseData(
              0, // dof_index
              0, // quad_index
              true, false, false, // cell evaluate
              false, true, false, // cell integrate
              true, false,        // face evaluate
              true, false         // face integrate
          ),
      type_velocity_field(TypeVelocityField::Analytical),
      dof_index_velocity(1),
      numerical_flux_formulation(NumericalFluxConvectiveOperator::Undefined)
  // clang-format on
  {
    this->mapping_update_flags = update_gradients | update_JxW_values | update_quadrature_points;
    this->mapping_update_flags_inner_faces =
      this->mapping_update_flags | update_values | update_normal_vectors;
    this->mapping_update_flags_boundary_faces = this->mapping_update_flags_inner_faces;
  }

  TypeVelocityField type_velocity_field;

  // TypeVelocityField::Numerical
  unsigned int dof_index_velocity;

  // TypeVelocityField::Analytical
  std::shared_ptr<Function<dim>> velocity;

  NumericalFluxConvectiveOperator numerical_flux_formulation;

  std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> bc;
};

template<int dim, typename Number>
class ConvectiveOperator : public OperatorBase<dim, Number, ConvectiveOperatorData<dim>>
{
private:
  typedef OperatorBase<dim, Number, ConvectiveOperatorData<dim>> Base;

  typedef typename Base::FEEvalCell FEEvalCell;
  typedef typename Base::FEEvalFace FEEvalFace;

  typedef typename Base::VectorType VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef CellIntegrator<dim, dim, Number> FEEvalCellVelocity;
  typedef FaceIntegrator<dim, dim, Number> FEEvalFaceVelocity;

public:
  void
  reinit(MatrixFree<dim, Number> const &     data,
         AffineConstraints<double> const &   constraint_matrix,
         ConvectiveOperatorData<dim> const & operator_data) const;

  LinearAlgebra::distributed::Vector<Number> &
  get_velocity() const;

  void
  set_velocity(VectorType const & velocity) const;

private:
  /*
   * This function calculates the numerical flux using the central flux.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_central_flux(scalar const & value_m,
                           scalar const & value_p,
                           scalar const & normal_velocity) const;

  /*
   * The same as above, but with discontinuous velocity field.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_central_flux(scalar const & value_m,
                           scalar const & value_p,
                           scalar const & normal_velocity_m,
                           scalar const & normal_velocity_p) const;

  /*
   * This function calculates the numerical flux using the Lax-Friedrichs flux.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_lax_friedrichs_flux(scalar const & value_m,
                                  scalar const & value_p,
                                  scalar const & normal_velocity) const;

  /*
   * The same as above, but with discontinuous velocity field.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_lax_friedrichs_flux(scalar const & value_m,
                                  scalar const & value_p,
                                  scalar const & normal_velocity_m,
                                  scalar const & normal_velocity_p) const;

  /*
   * This function calculates the numerical flux where the type of the numerical flux depends on the
   * specified input parameter. This function handles both analytical and numerical velocity fields.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_flux(unsigned int const q,
                   FEEvalFace &       fe_eval_m,
                   scalar const &     value_m,
                   scalar const &     value_p,
                   bool const         exterior_velocity_available) const;

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux(FEEvalCell & fe_eval, unsigned int const q) const;

  /*
   * Calculation of interior and exterior values on domain boundaries.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_interior_value(unsigned int const   q,
                             FEEvalFace const &   fe_eval_m,
                             OperatorType const & operator_type) const;

  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_exterior_value(scalar const &           value_m,
                             unsigned int const       q,
                             FEEvalFace const &       fe_eval_m,
                             OperatorType const &     operator_type,
                             BoundaryType const &     boundary_type,
                             types::boundary_id const boundary_id = types::boundary_id()) const;

  void
  reinit_cell(unsigned int const cell) const;

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

  // TODO can be removed later once matrix-free evaluation allows accessing neighboring data for
  // cell-based face loops
  void
  do_face_int_integral_cell_based(FEEvalFace & fe_eval_m, FEEvalFace & fe_eval_p) const;

  void
  do_verify_boundary_conditions(types::boundary_id const             boundary_id,
                                ConvectiveOperatorData<dim> const &  operator_data,
                                std::set<types::boundary_id> const & periodic_boundary_ids) const;

  mutable VectorType velocity;

  mutable std::shared_ptr<FEEvalCellVelocity> fe_eval_velocity;
  mutable std::shared_ptr<FEEvalFaceVelocity> fe_eval_velocity_m;
  mutable std::shared_ptr<FEEvalFaceVelocity> fe_eval_velocity_p;
};
} // namespace ConvDiff

#endif
