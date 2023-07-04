/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_EXADG_OPERATORS_GENERALIZED_LAPLACE_OPERATOR_H_
#define INCLUDE_EXADG_OPERATORS_GENERALIZED_LAPLACE_OPERATOR_H_

#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/grid/grid_utilities.h>
#include <exadg/operators/interior_penalty_parameter.h>
#include <exadg/operators/operator_base.h>
#include <exadg/operators/variable_coefficients.h>
#include <exadg/poisson/user_interface/boundary_descriptor.h>

#include <variant>

namespace ExaDG
{
namespace GeneralizedLaplace
{
/**
 * Free function for a generic expression of the multiplication of a @p rhs_type
 * from the left with a @p coefficient_type object.
 *
 * @p coefficient_type must provide a public constexpr attribute `rank`.
 */
template<typename coefficient_type, typename rhs_type>
static inline DEAL_II_ALWAYS_INLINE //
  rhs_type
  coeff_mult(coefficient_type const & coefficient, rhs_type const & x)
{
  if constexpr(coefficient_type::rank == 4)
    return dealii::double_contract<2, 0, 3, 1>(coefficient, x); // double contraction as ijkl_kl
  else
    return coefficient * x;
}

/**
 * Collection of data for the generalized Laplace kernel.
 */
template<int dim>
struct KernelData
{
  //! Interior penalty factor for the calculation of `tau`
  double IP_factor{1.0};
};

/**
 * Generalized Laplace kernel. This class is responsible for providing the core
 * pieces of information required to evaluate the Laplace operator, i.e. the
 * fluxes, numerical parameters (like the interior penalty parameter), and the
 * coefficients.
 *
 * @tparam n_components Number of components of the solution (vector) field.
 * Solution is scalar if this is equal to 1.
 * @tparam coefficient_is_scalar Boolean switch to differentiate if the
 * coefficient is scalar or a higher rank tensor.
 */
template<int dim, typename Number, int n_components = 1, bool coefficient_is_scalar = true>
class Kernel
{
private:
  using IntegratorCell = CellIntegrator<dim, n_components, Number>;
  using IntegratorFace = FaceIntegrator<dim, n_components, Number>;

public:
  /*
   * If there is more than one component to the solution, i.e. vector-valued, then its rank is 1,
   * if there is only one component to the solution, i.e. scalar-valued, then the rank is 0.
   */
  static constexpr unsigned int value_rank = (n_components > 1) ? 1 : 0;

  /*
   * If the solution is scalar-valued, i.e. value_rank is 0, then a non-scalar coefficient means
   * that the rank of the coefficient tensor is 2.
   * If the solution is vector-valued, i.e. value_rank is 1, then a non-scalar coefficient means
   * that the rank of the coefficient tensor is 4.
   * A scalar coefficient in any case means that the coefficient rank is 0.
   */
  static constexpr unsigned int coefficient_rank =
    (coefficient_is_scalar) ? 0 : ((n_components > 1) ? 4 : 2);

  using Scalar = dealii::VectorizedArray<Number>;
  using Vector = dealii::Tensor<1, dim, Scalar>;

  using ValueType    = dealii::Tensor<value_rank, dim, Scalar>;
  using GradientType = dealii::Tensor<value_rank + 1, dim, Scalar>;

  using CoefficientType = dealii::Tensor<coefficient_rank, dim, Scalar>;
  using Coefficients    = std::variant<VariableCoefficients<CoefficientType>, CoefficientType>;

  /**
   * @brief Reinitializes the kernel by copying the kernel data and extracting
   * relevant information from the matrix-free object.
   *
   * - Calculates the penalty parameters.
   * - Initializes storage for the variable coefficients if necessary.
   *
   * @param matrix_free Underlying matrix-free object
   * @param data_in Kernel data
   * @param dof_index Desired DoF index of matrix-free the kernel will work on
   * storage)
   */
  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free,
         KernelData<dim> const &                 data_in,
         unsigned int const                      dof_index)
  {
    data   = data_in;
    degree = matrix_free.get_dof_handler(dof_index).get_fe().degree;

    calculate_penalty_parameter(matrix_free, dof_index);
  }

  /**
   * Returns integrator flags required for the kernel evaluation.
   */
  static IntegratorFlags
  get_integrator_flags()
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = dealii::EvaluationFlags::gradients;
    flags.cell_integrate = dealii::EvaluationFlags::gradients;

    flags.face_evaluate  = dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;
    flags.face_integrate = dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;

    return flags;
  }

  /**
   * Returns mapping flags required for the kernel evaluation.
   */
  static MappingFlags
  get_mapping_flags(bool const compute_interior_face_integrals,
                    bool const compute_boundary_face_integrals)
  {
    MappingFlags flags;

    flags.cells =
      dealii::update_JxW_values | dealii::update_gradients | dealii::update_quadrature_points;
    if(compute_interior_face_integrals)
      flags.inner_faces = dealii::update_JxW_values | dealii::update_gradients |
                          dealii::update_normal_vectors | dealii::update_quadrature_points;
    if(compute_boundary_face_integrals)
      flags.boundary_faces = dealii::update_JxW_values | dealii::update_gradients |
                             dealii::update_normal_vectors | dealii::update_quadrature_points;

    return flags;
  }

  /**
   * Calculates and returns the `flux` inside the cell integral. This flux
   * should be weighted with the gradient of the shape functions.
   */
  static inline DEAL_II_ALWAYS_INLINE //
    GradientType
    get_volume_flux(GradientType const & gradient, CoefficientType const & coefficient)
  {
    return coeff_mult(coefficient, gradient);
  }

  /**
   * Calculates and returns the `flux` on the faces which should be weighted
   * with the gradient of the shape functions.
   */
  static inline DEAL_II_ALWAYS_INLINE //
    GradientType
    calculate_gradient_flux(ValueType const &       value_m,
                            ValueType const &       value_p,
                            Vector const &          normal,
                            CoefficientType const & coefficient)
  {
    ValueType const    jump_value  = value_m - value_p;
    GradientType const jump_tensor = outer_product(jump_value, normal);

    return -0.5 * coeff_mult(coefficient, jump_tensor);
  }

  /**
   * Calculates and returns the `flux` on the faces which should be weighted
   * with the normal derivative of the shape functions.
   *
   * Weighting with the normal derivative (gradient times face normal) is
   * mathematically only possible if the coefficient is non-coupling type. So,
   * this is asserted inside the function body.
   */
  static inline DEAL_II_ALWAYS_INLINE //
    ValueType
    calculate_normal_derivative_flux(ValueType const &       value_m,
                                     ValueType const &       value_p,
                                     CoefficientType const & coefficient)
  {
    {Assert(coefficient_is_scalar,
            dealii::ExcMessage("Normal derivative flux only makes"
                               "sense with scalar coefficients."))}

    ValueType const jump_value = value_m - value_p;

    return -0.5 * coefficient * jump_value;
  }

  /**
   * Calculates and returns the `flux` on the faces which should be weighted
   * with the shape function values.
   */
  inline DEAL_II_ALWAYS_INLINE //
    ValueType
    calculate_value_flux(GradientType const &    gradient_m,
                         GradientType const &    gradient_p,
                         ValueType const &       value_m,
                         ValueType const &       value_p,
                         Vector const &          normal,
                         CoefficientType const & coefficient)
  {
    ValueType const    jump_value  = value_m - value_p;
    GradientType const jump_tensor = outer_product(jump_value, normal);

    GradientType const average_gradient = 0.5 * (gradient_m + gradient_p);

    return coeff_mult(coefficient, (average_gradient - tau * jump_tensor)) * normal;
  }

  /**
   * Same as above, but takes in as input the product of the coefficients, the
   * gradient, and the the face normal vector. This version is for example
   * useful for boundary integrals, where boundary conditions may prescribe the
   * value of the this product.
   */
  inline DEAL_II_ALWAYS_INLINE //
    ValueType
    calculate_value_flux(ValueType const &       coeff_times_gradient_times_normal_m,
                         ValueType const &       coeff_times_gradient_times_normal_p,
                         ValueType const &       value_m,
                         ValueType const &       value_p,
                         Vector const &          normal,
                         CoefficientType const & coefficient)
  {
    ValueType const    jump_value  = value_m - value_p;
    GradientType const jump_tensor = outer_product(jump_value, normal);

    ValueType const average_coeff_times_gradient_times_normal =
      0.5 * (coeff_times_gradient_times_normal_m + coeff_times_gradient_times_normal_p);

    return average_coeff_times_gradient_times_normal -
           ValueType{coeff_mult(coefficient, (tau * jump_tensor)) * normal};
  }

  /**
   * Calculates the interior penalty parameter `tau` for the discretization.
   */
  void
  calculate_penalty_parameter(dealii::MatrixFree<dim, Number> const & matrix_free,
                              unsigned int const                      dof_index)
  {
    IP::calculate_penalty_parameter<dim, Number>(penalty_parameters, matrix_free, dof_index);
  }

  Coefficients &
  get_coefficients()
  {
    return coefficients;
  }

  /**
   * Returns the coefficient for the requested (@p cell index, @p q point index)
   * pair.
   *
   * If the coefficient is not variable, returns the constant coefficient.
   */
  CoefficientType
  get_coefficient_cell(unsigned int const cell, unsigned int const q) const
  {
    return std::visit(VarCoeffUtils::GetCoefficientCell<CoefficientType>{cell, q}, coefficients);
  }

  /**
   * Returns the coefficient for the requested (@p face index, @p q point index)
   * pair.
   *
   * If the coefficient is not variable, returns the constant coefficient.
   */
  CoefficientType
  get_coefficient_face(unsigned int const face, unsigned int const q) const
  {
    return std::visit(VarCoeffUtils::GetCoefficientFace<CoefficientType>{face, q}, coefficients);
  }

  /**
   * Returns the coefficient for the requested (@p face index (cell-based), @p q
   * point index) pair.
   *
   * If the coefficient is not variable, returns the constant coefficient.
   */
  CoefficientType
  get_coefficient_face_cell_based(unsigned int const cell_based_face, unsigned int const q) const
  {
    return std::visit(VarCoeffUtils::GetCoefficientFaceCellBased<CoefficientType>{cell_based_face,
                                                                                  q},
                      coefficients);
  }

  /**
   * Recomputes `tau` before a face evaluation.
   */
  void
  reinit_face(IntegratorFace &   integrator_m,
              IntegratorFace &   integrator_p,
              unsigned int const dof_index) const
  {
    tau = std::max(integrator_m.read_cell_data(penalty_parameters),
                   integrator_p.read_cell_data(penalty_parameters)) *
          IP::get_penalty_factor<dim, Number>(
            degree,
            GridUtilities::get_element_type(
              integrator_m.get_matrix_free().get_dof_handler(dof_index).get_triangulation()),
            data.IP_factor);
  }

  /**
   * Recomputes `tau` before a boundary face evaluation.
   */
  void
  reinit_boundary_face(IntegratorFace & integrator_m, unsigned int const dof_index) const
  {
    tau = integrator_m.read_cell_data(penalty_parameters) *
          IP::get_penalty_factor<dim, Number>(
            degree,
            GridUtilities::get_element_type(
              integrator_m.get_matrix_free().get_dof_handler(dof_index).get_triangulation()),
            data.IP_factor);
  }

  /**
   * Recomputes `tau` before a cell-based face evaluation.
   */
  void
  reinit_face_cell_based(dealii::types::boundary_id const boundary_id,
                         IntegratorFace &                 integrator_m,
                         IntegratorFace &                 integrator_p,
                         unsigned int const               dof_index) const
  {
    if(boundary_id == dealii::numbers::internal_face_boundary_id) // internal face
    {
      tau = std::max(integrator_m.read_cell_data(penalty_parameters),
                     integrator_p.read_cell_data(penalty_parameters)) *
            IP::get_penalty_factor<dim, Number>(
              degree,
              GridUtilities::get_element_type(
                integrator_m.get_matrix_free().get_dof_handler(dof_index).get_triangulation()),
              data.IP_factor);
    }
    else // boundary face
    {
      tau = integrator_m.read_cell_data(penalty_parameters) *
            IP::get_penalty_factor<dim, Number>(
              degree,
              GridUtilities::get_element_type(
                integrator_m.get_matrix_free().get_dof_handler(dof_index).get_triangulation()),
              data.IP_factor);
    }
  }

private:
  //! Generalized Laplace kernel data
  KernelData<dim> data{};

  //! Element polynomial degree
  unsigned int degree{1};

  //! Interior penalty parameter of the symmetric interior penalty (SIP) method
  mutable Scalar tau{0.0};

  //! Geometry-dependent penalty parameters required for the calculation of
  //! `tau`
  dealii::AlignedVector<Scalar> penalty_parameters{};

  //! Coefficient
  Coefficients coefficients{};
};

/**
 * Enumeration of boundary types supported in the Generalized Laplace operator
 */
enum class BoundaryType
{
  Undefined,
  Dirichlet,
  Neumann
};

/**
 * @brief Boundary descriptor for the generalized Laplace operator
 *
 * This class defines the valid boundary types and stores boundary information
 * for the generalized Laplace operator.
 */
template<int dim>
class BoundaryDescriptor
{
private:
  using bc_map = std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>;

public:
  /**
   * @brief Public constructor.
   *
   * The generalized Laplace boundary descriptor is constructed by passing in
   * boundary condition maps that can be extracted from the boundary descriptor
   * of the respective module who uses the generalized Laplace operator. The
   * module must provide maps for Dirichlet and Neumann conditions.
   *
   * @param dirichlet_bc (Boundary Id -> Function) map for Dirichlet boundary
   * conditions
   * @param neumann_bc (Boundary Id -> Function) map for Neumann boundary
   * conditions
   */
  BoundaryDescriptor(bc_map const & dirichlet_bc, bc_map const & neumann_bc)
    : dirichlet_bc(dirichlet_bc), neumann_bc(neumann_bc){};

  /**
   * Return the `BoundaryType` for the given @p boundary_id by searching for it
   * in the boundary condition maps.
   */
  inline DEAL_II_ALWAYS_INLINE //
    BoundaryType
    get_boundary_type(dealii::types::boundary_id const & boundary_id) const
  {
    if(this->dirichlet_bc.find(boundary_id) != this->dirichlet_bc.end())
      return BoundaryType::Dirichlet;
    else if(this->neumann_bc.find(boundary_id) != this->neumann_bc.end())
      return BoundaryType::Neumann;
    else
    {
      AssertThrow(false, dealii::ExcMessage("Boundary type of face is invalid or not implemented."))
    }
    return BoundaryType::Undefined;
  }

  //! Dirichlet boundary conditions map
  bc_map const dirichlet_bc;

  //! Neumann boundary conditions map
  bc_map const neumann_bc;
};

/**
 * Creates a generalized Laplace boundary descriptor from another boundary
 * descriptor.
 */
template<int dim, typename BCDescriptorType>
std::shared_ptr<BoundaryDescriptor<dim>>
create_boundary_descriptor(BCDescriptorType module_bc_descriptor)
{
  return std::make_shared<BoundaryDescriptor<dim>>(module_bc_descriptor->dirichlet_bc,
                                                   module_bc_descriptor->neumann_bc);
}

/**
 * Collection of functions to evaluate weak boundary conditions for the
 * generalized Laplace operator.
 */
template<int dim, typename Number, int n_components, bool coefficient_is_scalar>
struct WeakBoundaryConditions
{
  static constexpr unsigned int value_rank = (n_components > 1) ? 1 : 0;
  static constexpr unsigned int coefficient_rank =
    (coefficient_is_scalar) ? 0 : ((n_components > 1) ? 4 : 2);

  using Scalar          = dealii::VectorizedArray<Number>;
  using ValueType       = dealii::Tensor<value_rank, dim, Scalar>;
  using CoefficientType = dealii::Tensor<coefficient_rank, dim, Scalar>;

  /**
   * Returns the appropriate interior value for the evaluation of boundary
   * integrals depending on the operator type.
   */
  static inline DEAL_II_ALWAYS_INLINE //
    ValueType
    calculate_interior_value(unsigned int const                                q,
                             FaceIntegrator<dim, n_components, Number> const & integrator,
                             OperatorType const &                              operator_type)
  {
    if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
      return integrator.get_value(q);
    else if(operator_type == OperatorType::inhomogeneous)
      return ValueType{};
    else
    {
      AssertThrow(false, dealii::ExcMessage("Specified OperatorType is not implemented!"))
    }

    return ValueType{};
  }

  /**
   * Returns the appropriate exterior value for the evaluation of boundary
   * integrals depending on the operator and boundary type.
   */
  static inline DEAL_II_ALWAYS_INLINE //
    ValueType
    calculate_exterior_value(ValueType const &                                 value_m,
                             unsigned int const                                q,
                             FaceIntegrator<dim, n_components, Number> const & integrator,
                             OperatorType const &                              operator_type,
                             BoundaryType const &                              boundary_type,
                             dealii::types::boundary_id const                  boundary_id,
                             std::shared_ptr<BoundaryDescriptor<dim> const>    boundary_descriptor,
                             double const                                      time)
  {
    if(boundary_type == BoundaryType::Neumann)
      return value_m;

    if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
    {
      ValueType g{};

      if(boundary_type == BoundaryType::Dirichlet)
      {
        auto &       bc       = *boundary_descriptor->dirichlet_bc.find(boundary_id)->second;
        auto const & q_points = integrator.quadrature_point(q);

        g = FunctionEvaluator<ValueType::rank, dim, Number>::value(bc, q_points, time);
      }
      else
      {
        AssertThrow(false,
                    dealii::ExcMessage("Boundary type of face is invalid or not implemented."))
      }
      return -value_m + ValueType{2.0 * g};
    }
    else if(operator_type == OperatorType::homogeneous)
      return -value_m;
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented())
    }
    return ValueType{};
  }

  /**
   * Returns the appropriate interior boundary flux, i.e.
   *
   * ( coefficient * gradient ) * normal
   *
   * for the evaluation of boundary integrals depending on the operator type.
   */
  static inline DEAL_II_ALWAYS_INLINE //
    ValueType
    calculate_interior_coeff_times_gradient_times_normal(
      unsigned int const                                q,
      FaceIntegrator<dim, n_components, Number> const & integrator,
      OperatorType const &                              operator_type,
      CoefficientType const &                           coefficient)
  {
    if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
      return coeff_mult(coefficient, integrator.get_gradient(q)) * integrator.get_normal_vector(q);
    else if(operator_type == OperatorType::inhomogeneous)
      return ValueType{};
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented())
    }
    return ValueType{};
  }

  /**
   * Returns the appropriate exterior boundary flux, i.e.
   *
   * ( coefficient * gradient ) * normal
   *
   * for the evaluation of boundary integrals depending on the operator and
   * boundary type.
   */
  static inline DEAL_II_ALWAYS_INLINE //
    ValueType
    calculate_exterior_coeff_times_gradient_times_normal(
      ValueType const &                                 coeff_times_gradient_times_normal_m,
      unsigned int const                                q,
      FaceIntegrator<dim, n_components, Number> const & integrator,
      OperatorType const &                              operator_type,
      BoundaryType const &                              boundary_type,
      dealii::types::boundary_id const                  boundary_id,
      std::shared_ptr<BoundaryDescriptor<dim> const>    boundary_descriptor,
      double const                                      time)
  {
    if(boundary_type == BoundaryType::Dirichlet)
      return coeff_times_gradient_times_normal_m;

    if(boundary_type == BoundaryType::Neumann)
    {
      if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
      {
        auto &       bc       = *boundary_descriptor->neumann_bc.find(boundary_id)->second;
        auto const & q_points = integrator.quadrature_point(q);

        auto const h = FunctionEvaluator<ValueType::rank, dim, Number>::value(bc, q_points, time);

        return coeff_times_gradient_times_normal_m + ValueType{2.0 * h};
      }
      else if(operator_type == OperatorType::homogeneous)
        return -coeff_times_gradient_times_normal_m;
      else
        AssertThrow(false, dealii::ExcNotImplemented())
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Boundary type of face is invalid or not implemented."))
    }
    return ValueType{};
  }
};

/**
 * Collection of data for the generalized Laplace operator
 */
template<int dim>
struct OperatorData : public OperatorBaseData
{
  KernelData<dim> kernel_data{};

  std::shared_ptr<BoundaryDescriptor<dim>> bc{};
};

/**
 * Generalized Laplace operator
 *
 * This class provides the loop structures for the integrals that are evaluated
 * for a generalized Laplace term.
 *
 * For the template parameter descriptions, see @Kernel
 *
 */
template<int dim, typename Number, int n_components = 1, bool coefficient_is_scalar = true>
class Operator : public OperatorBase<dim, Number, n_components>
{
private:
  using MyKernel = Kernel<dim, Number, n_components, coefficient_is_scalar>;

  using Vector          = typename MyKernel::Vector;
  using ValueType       = typename MyKernel::ValueType;
  using GradientType    = typename MyKernel::GradientType;
  using CoefficientType = typename MyKernel::CoefficientType;

  using Base = OperatorBase<dim, Number, n_components>;
  using This = Operator<dim, Number, n_components, coefficient_is_scalar>;

  using Range          = typename Base::Range;
  using VectorType     = typename Base::VectorType;
  using IntegratorCell = typename Base::IntegratorCell;
  using IntegratorFace = typename Base::IntegratorFace;

  using BC = WeakBoundaryConditions<dim, Number, n_components, coefficient_is_scalar>;

public:
  /**
   * Initializes the operator's data structures and the kernel, sets flags, and
   * calculates the coefficients.
   */
  void
  initialize(dealii::MatrixFree<dim, Number> const &   matrix_free,
             dealii::AffineConstraints<Number> const & affine_constraints,
             OperatorData<dim> const &                 data);

  /**
   * Same as above, except receives an (already initialized) kernel from the
   * outside, but still calculates the coefficients.
   */
  void
  initialize(dealii::MatrixFree<dim, Number> const &   matrix_free,
             dealii::AffineConstraints<Number> const & affine_constraints,
             OperatorData<dim> const &                 data_in,
             std::shared_ptr<MyKernel>                 kernel_in);

private:
  void
  reinit_face(IntegratorFace & integrator_m,
              IntegratorFace & integrator_p,
              unsigned int     face) const override;

  void
  reinit_boundary_face(IntegratorFace & integrator_m, unsigned int face) const override;

  void
  reinit_face_cell_based(IntegratorFace &           integrator_m,
                         IntegratorFace &           integrator_p,
                         unsigned int               cell,
                         unsigned int               face,
                         dealii::types::boundary_id boundary_id) const override;

  void
  do_cell_integral(IntegratorCell & integrator) const override;

  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const override;

  void
  do_face_int_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const override;

  void
  do_face_int_integral_cell_based(IntegratorFace & integrator_m,
                                  IntegratorFace & integrator_p) const override;

  void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const override;

  void
  do_boundary_integral(IntegratorFace &                   integrator,
                       OperatorType const &               operator_type,
                       dealii::types::boundary_id const & boundary_id) const override;

  void
  do_boundary_integral_cell_based(IntegratorFace &                   integrator,
                                  OperatorType const &               operator_type,
                                  dealii::types::boundary_id const & boundary_id) const override;

  //! Operator data
  OperatorData<dim> operator_data;

  //! Generalized Laplace Kernel
  std::shared_ptr<MyKernel> kernel;
};
} // namespace GeneralizedLaplace
} // namespace ExaDG

#endif /* INCLUDE_EXADG_OPERATORS_GENERALIZED_LAPLACE_OPERATOR_H_ */
