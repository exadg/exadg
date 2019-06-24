#ifndef CONV_DIFF_DIFFUSIVE_OPERATOR
#define CONV_DIFF_DIFFUSIVE_OPERATOR

#include "../../../operators/operator_base.h"
#include "../../user_interface/boundary_descriptor.h"
#include "../../user_interface/input_parameters.h"

#include "../../../operators/interior_penalty_parameter.h"

namespace ConvDiff
{
namespace Operators
{
struct DiffusiveKernelData
{
  DiffusiveKernelData() : IP_factor(1.0), degree(1), degree_mapping(1), diffusivity(1.0)
  {
  }

  double       IP_factor;
  unsigned int degree;
  unsigned int degree_mapping;
  double       diffusivity;
};

template<int dim, typename Number>
class DiffusiveKernel
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef CellIntegrator<dim, 1, Number> IntegratorCell;
  typedef FaceIntegrator<dim, 1, Number> IntegratorFace;

public:
  DiffusiveKernel() : tau(make_vectorized_array<Number>(0.0))
  {
  }

  void
  reinit(MatrixFree<dim, Number> const & matrix_free,
         DiffusiveKernelData const &     data_in,
         unsigned int const              dof_index) const
  {
    data = data_in;

    MappingQGeneric<dim> mapping(data_in.degree_mapping);
    IP::calculate_penalty_parameter<dim, Number>(
      array_penalty_parameter, matrix_free, mapping, data_in.degree, dof_index);

    AssertThrow(data.diffusivity > 0.0, ExcMessage("Diffusivity is not set!"));
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
    return -0.5 * data.diffusivity * (value_m - value_p);
  }

  /*
   * Calculation of gradient flux. Strictly speaking, this value is not a numerical flux since the
   * flux is multiplied by the normal vector, i.e., "gradient_flux" = numerical_flux * normal, where
   * normal denotes the normal vector of element e⁻.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_value_flux(scalar const & normal_gradient_m,
                         scalar const & normal_gradient_p,
                         scalar const & value_m,
                         scalar const & value_p) const
  {
    return data.diffusivity *
           (0.5 * (normal_gradient_m + normal_gradient_p) - tau * (value_m - value_p));
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux(IntegratorCell & integrator, unsigned int const q) const
  {
    return integrator.get_gradient(q) * data.diffusivity;
  }

private:
  mutable DiffusiveKernelData data;

  mutable AlignedVector<scalar> array_penalty_parameter;
  mutable scalar                tau;
};

} // namespace Operators


template<int dim>
struct DiffusiveOperatorData : public OperatorBaseData
{
  DiffusiveOperatorData() : OperatorBaseData(0 /* dof_index */, 0 /* quad_index */)
  {
  }

  Operators::DiffusiveKernelData kernel_data;

  std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> bc;
};


template<int dim, typename Number>
class DiffusiveOperator : public OperatorBase<dim, Number, DiffusiveOperatorData<dim>>
{
private:
  typedef OperatorBase<dim, Number, DiffusiveOperatorData<dim>> Base;

  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

public:
  void
  reinit(MatrixFree<dim, Number> const &    matrix_free,
         AffineConstraints<double> const &  constraint_matrix,
         DiffusiveOperatorData<dim> const & data) const;

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
                                DiffusiveOperatorData<dim> const &   operator_data,
                                std::set<types::boundary_id> const & periodic_boundary_ids) const;

  Operators::DiffusiveKernel<dim, Number> kernel;
};
} // namespace ConvDiff

#endif
