#ifndef LAPLACE_OPERATOR_H
#define LAPLACE_OPERATOR_H

#include "../../operators/interior_penalty_parameter.h"
#include "../../operators/operator_base.h"
#include "../../operators/operator_type.h"

#include "../user_interface/boundary_descriptor.h"

namespace Poisson
{
namespace Operators
{
struct LaplaceKernelData
{
  LaplaceKernelData() : IP_factor(1.0), degree(1), degree_mapping(1)
  {
  }

  double       IP_factor;
  unsigned int degree;
  unsigned int degree_mapping;
};

template<int dim, typename Number>
class LaplaceKernel
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef CellIntegrator<dim, 1, Number> IntegratorCell;
  typedef FaceIntegrator<dim, 1, Number> IntegratorFace;

public:
  LaplaceKernel() : tau(make_vectorized_array<Number>(0.0))
  {
  }

  void
  reinit(MatrixFree<dim, Number> const & matrix_free,
         LaplaceKernelData const &       data_in,
         unsigned int const              dof_index) const
  {
    data = data_in;

    MappingQGeneric<dim> mapping(data_in.degree_mapping);
    IP::calculate_penalty_parameter<dim, Number>(
      array_penalty_parameter, matrix_free, mapping, data_in.degree, dof_index);
  }

  IntegratorFlags
  get_integrator_flags() const
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = CellFlags(false, true, false);
    flags.cell_integrate = CellFlags(false, true, false);

    flags.face_evaluate  = FaceFlags(true, true);
    flags.face_integrate = FaceFlags(true, true);

    return flags;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells       = update_gradients | update_JxW_values;
    flags.inner_faces = update_gradients | update_JxW_values | update_normal_vectors;
    flags.boundary_faces =
      update_gradients | update_JxW_values | update_normal_vectors | update_quadrature_points;

    return flags;
  }

  void
  reinit_face(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    tau = std::max(integrator_m.read_cell_data(array_penalty_parameter),
                   integrator_p.read_cell_data(array_penalty_parameter)) *
          IP::get_penalty_factor<Number>(data.degree, data.IP_factor);
  }

  void
  reinit_boundary_face(IntegratorFace & integrator_m) const
  {
    tau = integrator_m.read_cell_data(array_penalty_parameter) *
          IP::get_penalty_factor<Number>(data.degree, data.IP_factor);
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
            IP::get_penalty_factor<Number>(data.degree, data.IP_factor);
    }
    else // boundary face
    {
      tau = integrator_m.read_cell_data(array_penalty_parameter) *
            IP::get_penalty_factor<Number>(data.degree, data.IP_factor);
    }
  }

  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_gradient_flux(scalar const & value_m, scalar const & value_p) const
  {
    return -0.5 * (value_m - value_p);
  }

  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_value_flux(scalar const & normal_gradient_m,
                         scalar const & normal_gradient_p,
                         scalar const & value_m,
                         scalar const & value_p) const
  {
    return 0.5 * (normal_gradient_m + normal_gradient_p) - tau * (value_m - value_p);
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux(IntegratorCell & integrator, unsigned int const q) const
  {
    return integrator.get_gradient(q);
  }

private:
  mutable LaplaceKernelData data;

  mutable AlignedVector<scalar> array_penalty_parameter;
  mutable scalar                tau;
};

} // namespace Operators

template<int dim>
struct LaplaceOperatorData : public OperatorBaseData
{
  LaplaceOperatorData() : OperatorBaseData(0 /* dof_index */, 0 /* quad_index */)
  {
  }

  Operators::LaplaceKernelData kernel_data;

  std::shared_ptr<Poisson::BoundaryDescriptor<dim>> bc;
};

template<int dim, typename Number>
class LaplaceOperator : public OperatorBase<dim, Number, LaplaceOperatorData<dim>>
{
private:
  typedef OperatorBase<dim, Number, LaplaceOperatorData<dim>> Base;

  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

public:
  typedef Number                    value_type;
  typedef typename Base::VectorType VectorType;

  void
  reinit(MatrixFree<dim, Number> const &   matrix_free,
         AffineConstraints<double> const & constraint_matrix,
         LaplaceOperatorData<dim> const &  data) const;

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
  do_verify_boundary_conditions(types::boundary_id const             boundary_id,
                                LaplaceOperatorData<dim> const &     data,
                                std::set<types::boundary_id> const & periodic_boundary_ids) const;

  Operators::LaplaceKernel<dim, Number> kernel;
};

} // namespace Poisson

#endif
