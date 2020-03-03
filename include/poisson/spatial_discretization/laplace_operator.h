#ifndef LAPLACE_OPERATOR_H
#define LAPLACE_OPERATOR_H

#include "../../operators/interior_penalty_parameter.h"
#include "../../operators/operator_base.h"
#include "../../operators/operator_type.h"

#include "../../convection_diffusion/user_interface/boundary_descriptor.h"

namespace Poisson
{
namespace Operators
{
struct LaplaceKernelData
{
  LaplaceKernelData() : IP_factor(1.0)
  {
  }

  double IP_factor;
};

template<int dim, typename Number, int n_components = 1>
class LaplaceKernel
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number> scalar;

  typedef FaceIntegrator<dim, n_components, Number> IntegratorFace;

public:
  LaplaceKernel() : degree(1), tau(make_vectorized_array<Number>(0.0))
  {
  }

  void
  reinit(MatrixFree<dim, Number> const & matrix_free,
         LaplaceKernelData const &       data_in,
         unsigned int const              dof_index)
  {
    data = data_in;

    FiniteElement<dim> const & fe = matrix_free.get_dof_handler(dof_index).get_fe();
    degree                        = fe.degree;

    calculate_penalty_parameter(matrix_free, dof_index);
  }

  void
  calculate_penalty_parameter(MatrixFree<dim, Number> const & matrix_free,
                              unsigned int const              dof_index)
  {
    IP::calculate_penalty_parameter<dim, Number>(array_penalty_parameter, matrix_free, dof_index);
  }

  IntegratorFlags
  get_integrator_flags(bool const is_dg) const
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = CellFlags(false, true, false);
    flags.cell_integrate = CellFlags(false, true, false);

    if(is_dg)
    {
      flags.face_evaluate  = FaceFlags(true, true);
      flags.face_integrate = FaceFlags(true, true);
    }
    else
    {
      // evaluation of Neumann BCs for continuous elements
      flags.face_evaluate  = FaceFlags(false, false);
      flags.face_integrate = FaceFlags(true, false);
    }

    return flags;
  }

  static MappingFlags
  get_mapping_flags(bool const compute_interior_face_integrals,
                    bool const compute_boundary_face_integrals)
  {
    MappingFlags flags;

    flags.cells = update_gradients | update_JxW_values;

    if(compute_interior_face_integrals)
    {
      flags.inner_faces = update_gradients | update_JxW_values | update_normal_vectors;
    }

    if(compute_boundary_face_integrals)
    {
      flags.boundary_faces =
        update_gradients | update_JxW_values | update_normal_vectors | update_quadrature_points;
    }

    return flags;
  }

  void
  reinit_face(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    tau = std::max(integrator_m.read_cell_data(array_penalty_parameter),
                   integrator_p.read_cell_data(array_penalty_parameter)) *
          IP::get_penalty_factor<Number>(degree, data.IP_factor);
  }

  void
  reinit_boundary_face(IntegratorFace & integrator_m) const
  {
    tau = integrator_m.read_cell_data(array_penalty_parameter) *
          IP::get_penalty_factor<Number>(degree, data.IP_factor);
  }

  void
  reinit_face_cell_based(types::boundary_id const boundary_id,
                         IntegratorFace &         integrator_m,
                         IntegratorFace &         integrator_p) const
  {
    if(boundary_id == numbers::internal_face_boundary_id) // internal face
    {
      tau = std::max(integrator_m.read_cell_data(array_penalty_parameter),
                     integrator_p.read_cell_data(array_penalty_parameter)) *
            IP::get_penalty_factor<Number>(degree, data.IP_factor);
    }
    else // boundary face
    {
      tau = integrator_m.read_cell_data(array_penalty_parameter) *
            IP::get_penalty_factor<Number>(degree, data.IP_factor);
    }
  }

  template<typename T>
  inline DEAL_II_ALWAYS_INLINE //
    T
    calculate_gradient_flux(T const & value_m, T const & value_p) const
  {
    return -0.5 * (value_m - value_p);
  }

  template<typename T>
  inline DEAL_II_ALWAYS_INLINE //
    T
    calculate_value_flux(T const & normal_gradient_m,
                         T const & normal_gradient_p,
                         T const & value_m,
                         T const & value_p) const
  {
    return 0.5 * (normal_gradient_m + normal_gradient_p) - tau * (value_m - value_p);
  }

private:
  LaplaceKernelData data;

  unsigned int degree;

  AlignedVector<scalar> array_penalty_parameter;

  mutable scalar tau;
};

} // namespace Operators

template<int dim>
struct LaplaceOperatorData : public OperatorBaseData
{
  LaplaceOperatorData() : OperatorBaseData(0 /* dof_index */, 0 /* quad_index */)
  {
  }

  Operators::LaplaceKernelData kernel_data;

  std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> bc;
};

template<int dim, typename Number, int n_components = 1>
class LaplaceOperator : public OperatorBase<dim, Number, LaplaceOperatorData<dim>, n_components>
{
private:
  typedef OperatorBase<dim, Number, LaplaceOperatorData<dim>, n_components> Base;
  typedef LaplaceOperator<dim, Number, n_components>                        This;

  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  typedef typename Base::Range Range;

  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : numbers::invalid_unsigned_int);

  typedef Tensor<rank, dim, VectorizedArray<Number>> value;

public:
  typedef Number                    value_type;
  typedef typename Base::VectorType VectorType;

  void
  reinit(MatrixFree<dim, Number> const &   matrix_free,
         AffineConstraints<double> const & constraint_matrix,
         LaplaceOperatorData<dim> const &  data);

  void
  calculate_penalty_parameter(MatrixFree<dim, Number> const & matrix_free,
                              unsigned int const              dof_index);

  void
  update_after_mesh_movement();

  // Some more functionality on top of what is provided by the base class.
  // This function evaluates the inhomogeneous boundary face integrals where the
  // Dirichlet boundary condition is extracted from a dof vector instead of a Function<dim>.
  void
  rhs_add_dirichlet_bc_from_dof_vector(VectorType & dst, VectorType const & src) const;

private:
  void
  reinit_face(unsigned int const face) const;

  void
  reinit_boundary_face(unsigned int const face) const;

  void
  reinit_face_cell_based(unsigned int const       cell,
                         unsigned int const       face,
                         types::boundary_id const boundary_id) const;

  void
  do_cell_integral(IntegratorCell & integrator) const;

  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  void
  do_face_int_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  void
  do_boundary_integral(IntegratorFace &           integrator_m,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id) const;

  void
  do_boundary_integral_continuous(IntegratorFace &           integrator_m,
                                  OperatorType const &       operator_type,
                                  types::boundary_id const & boundary_id) const;

  // Some more functionality on top of what is provided by the base class.
  void
  cell_loop_empty(MatrixFree<dim, Number> const & matrix_free,
                  VectorType &                    dst,
                  VectorType const &              src,
                  Range const &                   range) const;

  void
  face_loop_empty(MatrixFree<dim, Number> const & matrix_free,
                  VectorType &                    dst,
                  VectorType const &              src,
                  Range const &                   range) const;

  void
  boundary_face_loop_inhom_operator_dirichlet_bc_from_dof_vector(
    MatrixFree<dim, Number> const & matrix_free,
    VectorType &                    dst,
    VectorType const &              src,
    Range const &                   range) const;

  void
  do_boundary_integral_dirichlet_bc_from_dof_vector(IntegratorFace &           integrator_m,
                                                    OperatorType const &       operator_type,
                                                    types::boundary_id const & boundary_id) const;

  void
  do_verify_boundary_conditions(types::boundary_id const             boundary_id,
                                LaplaceOperatorData<dim> const &     data,
                                std::set<types::boundary_id> const & periodic_boundary_ids) const;

  Operators::LaplaceKernel<dim, Number, n_components> kernel;
};

} // namespace Poisson

#endif
