#ifndef CONV_DIFF_CONVECTION_OPERATOR
#define CONV_DIFF_CONVECTION_OPERATOR

#include "../types.h"
#include "../../user_interface/input_parameters.h"
#include "../../user_interface/boundary_descriptor.h"
#include "../../../operators/operation_base.h"

namespace ConvDiff {

template <int dim>
struct ConvectiveOperatorData
    : public OperatorBaseData<dim, BoundaryType, OperatorType,
                              ConvDiff::BoundaryDescriptor<dim>> {

  ConvectiveOperatorData()
      : OperatorBaseData<dim, BoundaryType, OperatorType,
                         ConvDiff::BoundaryDescriptor<dim>>(
            0, 0, true, false, false, false, true, false, // cell
            true, false, true, false,                     // face
            true, false, true, false                      // boundary
            ),
        numerical_flux_formulation(NumericalFluxConvectiveOperator::Undefined) {
  }

  NumericalFluxConvectiveOperator numerical_flux_formulation;
  std::shared_ptr<Function<dim>> velocity;
};

template <int dim, int fe_degree, typename value_type>
class ConvectiveOperator : public OperatorBase<dim, fe_degree, value_type,
                                               ConvectiveOperatorData<dim>> {
public:
  typedef ConvectiveOperator<dim, fe_degree, value_type> This;
  typedef OperatorBase<dim, fe_degree, value_type, ConvectiveOperatorData<dim>>
      Parent;
  typedef typename Parent::FEEvalCell FEEvalCell;
  typedef typename Parent::FEEvalFace FEEvalFace;
  typedef typename Parent::VNumber VNumber;

  void initialize(MatrixFree<dim, value_type> const &mf_data,
                  ConvectiveOperatorData<dim> const &operator_data_in);

  /*
   *  This function calculates the numerical flux for interior faces
   *  using the central flux.
   */
  inline DEAL_II_ALWAYS_INLINE VectorizedArray<value_type>
  calculate_central_flux(VectorizedArray<value_type> &value_m,
                         VectorizedArray<value_type> &value_p,
                         VectorizedArray<value_type> &normal_velocity) const;

  /*
   *  This function calculates the numerical flux for interior faces
   *  using the Lax-Friedrichs flux.
   */
  inline DEAL_II_ALWAYS_INLINE VectorizedArray<value_type>
  calculate_lax_friedrichs_flux(
      VectorizedArray<value_type> &value_m,
      VectorizedArray<value_type> &value_p,
      VectorizedArray<value_type> &normal_velocity) const;

  /*
   *  This function calculates the numerical flux for interior faces where
   *  the type of the numerical flux depends on the specified input parameter.
   */
  inline DEAL_II_ALWAYS_INLINE VectorizedArray<value_type>
  calculate_flux(unsigned int const q, FEEvalFace &phi_n,
                 VectorizedArray<value_type> &value_m,
                 VectorizedArray<value_type> &value_p) const;

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
  calculate_interior_value(unsigned int const q, FEEvalFace const &phi_n,
                           OperatorType const &op_type) const;

  inline DEAL_II_ALWAYS_INLINE VectorizedArray<value_type>
  calculate_exterior_value(
      VectorizedArray<value_type> const &value_m, unsigned int const q,
      FEEvalFace const &phi_n, OperatorType const &operator_type,
      BoundaryType const &boundary_type,
      types::boundary_id const boundary_id = types::boundary_id()) const;

  void do_cell_integral(FEEvalCell &phi) const;

  void do_face_integral(FEEvalFace &phi_n, FEEvalFace &phi_p) const;

  void do_face_int_integral(FEEvalFace &phi_n, FEEvalFace & /*phi_p*/) const;

  void do_face_ext_integral(FEEvalFace & /*phi_n*/, FEEvalFace &phi_p) const;

  void do_boundary_integral(FEEvalFace &phi_n,
                            OperatorType const &operator_type,
                            types::boundary_id const &boundary_id) const;
};
}

#endif