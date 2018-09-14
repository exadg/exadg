#ifndef CONV_DIFF_CONVECTION_OPERATOR
#define CONV_DIFF_CONVECTION_OPERATOR

#include "../../../operators/operation_base.h"
#include "../../user_interface/boundary_descriptor.h"
#include "../../user_interface/input_parameters.h"
#include "../types.h"

namespace ConvDiff
{
template<int dim>
struct ConvectiveOperatorData : public OperatorBaseData<dim, ConvDiff::BoundaryDescriptor<dim>>
{
  ConvectiveOperatorData()
    // clang-format off
    : OperatorBaseData<dim, ConvDiff::BoundaryDescriptor<dim>>(0, 0,
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

  NumericalFluxConvectiveOperator numerical_flux_formulation;
  std::shared_ptr<Function<dim>>  velocity;
};

template<int dim, int fe_degree, typename value_type>
class ConvectiveOperator
  : public OperatorBase<dim, fe_degree, value_type, ConvectiveOperatorData<dim>>
{
public:
  typedef ConvectiveOperator<dim, fe_degree, value_type> This;

  typedef OperatorBase<dim, fe_degree, value_type, ConvectiveOperatorData<dim>> Parent;

  typedef typename Parent::FEEvalCell FEEvalCell;
  typedef typename Parent::FEEvalFace FEEvalFace;
  typedef typename Parent::VectorType VectorType;

  void
  initialize(MatrixFree<dim, value_type> const & mf_data,
             ConvectiveOperatorData<dim> const & operator_data_in,
             unsigned int                        level_mg_handler = numbers::invalid_unsigned_int);

  void
  initialize(MatrixFree<dim, value_type> const & mf_data,
             ConstraintMatrix const &            constraint_matrx,
             ConvectiveOperatorData<dim> const & operator_data_in,
             unsigned int                        level_mg_handler = numbers::invalid_unsigned_int);

  /*
   * This function calculates the numerical flux using the central flux.
   */
  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<value_type>
    calculate_central_flux(VectorizedArray<value_type> & value_m,
                           VectorizedArray<value_type> & value_p,
                           VectorizedArray<value_type> & normal_velocity) const;

  /*
   * This function calculates the numerical flux using the Lax-Friedrichs flux.
   */
  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<value_type>
    calculate_lax_friedrichs_flux(VectorizedArray<value_type> & value_m,
                                  VectorizedArray<value_type> & value_p,
                                  VectorizedArray<value_type> & normal_velocity) const;

  /*
   * This function calculates the numerical flux where the type of the numerical flux depends on the
   * specified input parameter.
   */
  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<value_type>
    calculate_flux(unsigned int const            q,
                   FEEvalFace &                  fe_eval_m,
                   VectorizedArray<value_type> & value_m,
                   VectorizedArray<value_type> & value_p) const;

  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<value_type>
    calculate_interior_value(unsigned int const   q,
                             FEEvalFace const &   fe_eval_m,
                             OperatorType const & operator_type) const;

  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArray<value_type>
    calculate_exterior_value(VectorizedArray<value_type> const & value_m,
                             unsigned int const                  q,
                             FEEvalFace const &                  fe_eval_m,
                             OperatorType const &                operator_type,
                             BoundaryType const &                boundary_type,
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
};
} // namespace ConvDiff

#endif
