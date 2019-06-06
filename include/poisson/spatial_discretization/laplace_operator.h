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
  int          degree_mapping;
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
  LaplaceKernel();

  void
  reinit(MatrixFree<dim, Number> const & matrix_free,
         LaplaceKernelData const &       data_in,
         unsigned int const              dof_index) const;

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
    this->cell_evaluate  = Cell(false, true, false);
    this->cell_integrate = Cell(false, true, false);
    this->face_evaluate  = Face(true, true);
    this->face_integrate = Face(true, true);

    this->mapping_update_flags = update_gradients | update_JxW_values;
    this->mapping_update_flags_inner_faces =
      this->mapping_update_flags | update_values | update_normal_vectors;
    this->mapping_update_flags_boundary_faces =
      this->mapping_update_flags_inner_faces | update_quadrature_points;
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
  reinit(MatrixFree<dim, Number> const &   mf_data,
         AffineConstraints<double> const & constraint_matrix,
         LaplaceOperatorData<dim> const &  operator_data) const;

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
                                LaplaceOperatorData<dim> const &     operator_data,
                                std::set<types::boundary_id> const & periodic_boundary_ids) const;

  Operators::LaplaceKernel<dim, Number> kernel;
};

} // namespace Poisson

#endif
