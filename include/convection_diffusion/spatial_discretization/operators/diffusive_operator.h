#ifndef CONV_DIFF_DIFFUSIVE_OPERATOR
#define CONV_DIFF_DIFFUSIVE_OPERATOR

#include "../../../operators/operator_base.h"
#include "../../user_interface/boundary_descriptor.h"
#include "../../user_interface/input_parameters.h"

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
  int          degree_mapping;
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
  DiffusiveKernel();

  void
  reinit(MatrixFree<dim, Number> const & matrix_free,
         DiffusiveKernelData const &     data_in,
         unsigned int const              dof_index) const;

  IntegratorFlags
  get_integrator_flags() const;

  static MappingFlags
  get_mapping_flags();

  void
  reinit_face(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  void
  reinit_boundary_face(IntegratorFace & integrator_m) const;

  void
  reinit_face_cell_based(types::boundary_id const boundary_id,
                         IntegratorFace &         integrator_m,
                         IntegratorFace &         integrator_p) const;

  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_value_flux(scalar const & value_m, scalar const & value_p) const;

  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_gradient_flux(scalar const & normal_gradient_m,
                            scalar const & normal_gradient_p,
                            scalar const & value_m,
                            scalar const & value_p) const;

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux(IntegratorCell & integrator, unsigned int const q) const;

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
         DiffusiveOperatorData<dim> const & operator_data) const;

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
