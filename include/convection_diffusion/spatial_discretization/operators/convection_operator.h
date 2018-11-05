#ifndef CONV_DIFF_CONVECTION_OPERATOR
#define CONV_DIFF_CONVECTION_OPERATOR

#include "../../../operators/operator_base.h"
#include "../../user_interface/boundary_descriptor.h"
#include "../../user_interface/input_parameters.h"

namespace ConvDiff
{
template<int dim>
struct ConvectiveOperatorData : public OperatorBaseData<dim>
{
  ConvectiveOperatorData()
    // clang-format off
    : OperatorBaseData<dim>(0, 0,
          true, false, false, false, true, false, // cell
          true, false,        true,  false        // face
      ),
      // clang-format on
      numerical_flux_formulation(NumericalFluxConvectiveOperator::Undefined)
  {
    this->mapping_update_flags = update_gradients | update_JxW_values | update_quadrature_points;
    this->mapping_update_flags_inner_faces =
      this->mapping_update_flags | update_values | update_normal_vectors;
    this->mapping_update_flags_boundary_faces = this->mapping_update_flags_inner_faces;
  }

  NumericalFluxConvectiveOperator numerical_flux_formulation;
  std::shared_ptr<Function<dim>>  velocity;

  std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> bc;
};

template<int dim, int degree, typename Number>
class ConvectiveOperator : public OperatorBase<dim, degree, Number, ConvectiveOperatorData<dim>>
{
public:
  typedef ConvectiveOperator<dim, degree, Number> This;

  typedef OperatorBase<dim, degree, Number, ConvectiveOperatorData<dim>> Parent;

  typedef typename Parent::FEEvalCell FEEvalCell;
  typedef typename Parent::FEEvalFace FEEvalFace;
  typedef typename Parent::VectorType VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  void
  initialize(MatrixFree<dim, Number> const &     mf_data,
             ConvectiveOperatorData<dim> const & operator_data_in,
             unsigned int                        level_mg_handler = numbers::invalid_unsigned_int);

  void
  initialize(MatrixFree<dim, Number> const &     mf_data,
             ConstraintMatrix const &            constraint_matrix,
             ConvectiveOperatorData<dim> const & operator_data_in,
             unsigned int                        level_mg_handler = numbers::invalid_unsigned_int);

private:
  /*
   * This function calculates the numerical flux using the central flux.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_central_flux(scalar & value_m, scalar & value_p, scalar & normal_velocity) const;

  /*
   * This function calculates the numerical flux using the Lax-Friedrichs flux.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_lax_friedrichs_flux(scalar & value_m,
                                  scalar & value_p,
                                  scalar & normal_velocity) const;

  /*
   * This function calculates the numerical flux where the type of the numerical flux depends on the
   * specified input parameter.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_flux(unsigned int const q,
                   FEEvalFace &       fe_eval_m,
                   scalar &           value_m,
                   scalar &           value_p) const;

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
  do_cell_integral(FEEvalCell & fe_eval) const;

  void
  do_face_integral(FEEvalFace & fe_eval_m, FEEvalFace & fe_eval_p) const;

  void
  do_face_int_integral(FEEvalFace & fe_eval_m, FEEvalFace & /*fe_eval_p*/) const;

  void
  do_face_ext_integral(FEEvalFace & /*fe_eval_m*/, FEEvalFace & fe_eval_p) const;

  void
  do_boundary_integral(FEEvalFace &               fe_eval_m,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id) const;

  void
  do_verify_boundary_conditions(types::boundary_id const             boundary_id,
                                ConvectiveOperatorData<dim> const &  operator_data,
                                std::set<types::boundary_id> const & periodic_boundary_ids) const;
};
} // namespace ConvDiff

#endif
