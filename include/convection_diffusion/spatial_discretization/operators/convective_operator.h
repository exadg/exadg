#ifndef CONV_DIFF_CONVECTION_OPERATOR
#define CONV_DIFF_CONVECTION_OPERATOR

#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "../../../operators/operator_base.h"
#include "../../user_interface/boundary_descriptor.h"
#include "../../user_interface/input_parameters.h"

namespace ConvDiff
{
namespace Operators
{
template<int dim>
struct ConvectiveKernelData
{
  ConvectiveKernelData()
    : type_velocity_field(TypeVelocityField::Analytical),
      dof_index_velocity(1),
      numerical_flux_formulation(NumericalFluxConvectiveOperator::Undefined)
  {
  }

  // analytical vs. numerical velocity field
  TypeVelocityField type_velocity_field;

  // TypeVelocityField::Numerical
  unsigned int dof_index_velocity;

  // TypeVelocityField::Analytical
  std::shared_ptr<Function<dim>> velocity;

  // numerical flux (e.g., central flux vs. Lax-Friedrichs flux)
  NumericalFluxConvectiveOperator numerical_flux_formulation;
};

template<int dim, typename Number>
class ConvectiveKernel
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorVelocity;
  typedef FaceIntegrator<dim, dim, Number> FaceIntegratorVelocity;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef CellIntegrator<dim, 1, Number> IntegratorCell;
  typedef FaceIntegrator<dim, 1, Number> IntegratorFace;

public:
  void
  reinit(MatrixFree<dim, Number> const &   matrix_free,
         ConvectiveKernelData<dim> const & data_in,
         unsigned int const                quad_index) const;

  LinearAlgebra::distributed::Vector<Number> &
  get_velocity() const;

  void
  set_velocity(VectorType const & velocity) const;

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
                   IntegratorFace &   integrator_m,
                   scalar const &     value_m,
                   scalar const &     value_p,
                   Number const &     time,
                   bool const         exterior_velocity_available) const;

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux(IntegratorCell & integrator, unsigned int const q, Number const & time) const;

private:
  mutable ConvectiveKernelData<dim> data;

  mutable VectorType velocity;

  mutable std::shared_ptr<CellIntegratorVelocity> integrator_velocity;
  mutable std::shared_ptr<FaceIntegratorVelocity> integrator_velocity_m;
  mutable std::shared_ptr<FaceIntegratorVelocity> integrator_velocity_p;
};

} // namespace Operators


template<int dim>
struct ConvectiveOperatorData : public OperatorBaseData
{
  ConvectiveOperatorData() : OperatorBaseData(0 /* dof_index */, 0 /* quad_index */)
  {
    this->cell_evaluate  = Cell(true, false, false);
    this->cell_integrate = Cell(false, true, false);
    this->face_evaluate  = Face(true, false);
    this->face_integrate = Face(true, false);

    this->mapping_update_flags = update_gradients | update_JxW_values | update_quadrature_points;
    this->mapping_update_flags_inner_faces =
      this->mapping_update_flags | update_values | update_normal_vectors;
    this->mapping_update_flags_boundary_faces = this->mapping_update_flags_inner_faces;
  }

  Operators::ConvectiveKernelData<dim> kernel_data;

  std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> bc;
};

template<int dim, typename Number>
class ConvectiveOperator : public OperatorBase<dim, Number, ConvectiveOperatorData<dim>>
{
private:
  typedef OperatorBase<dim, Number, ConvectiveOperatorData<dim>> Base;

  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  typedef typename Base::VectorType VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

public:
  void
  reinit(MatrixFree<dim, Number> const &     matrix_free,
         AffineConstraints<double> const &   constraint_matrix,
         ConvectiveOperatorData<dim> const & operator_data) const;

  LinearAlgebra::distributed::Vector<Number> &
  get_velocity() const;

  void
  set_velocity(VectorType const & velocity) const;

private:
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

  // TODO can be removed later once matrix-free evaluation allows accessing neighboring data for
  // cell-based face loops
  void
  do_face_int_integral_cell_based(IntegratorFace & integrator_m,
                                  IntegratorFace & integrator_p) const;

  void
  do_verify_boundary_conditions(types::boundary_id const             boundary_id,
                                ConvectiveOperatorData<dim> const &  operator_data,
                                std::set<types::boundary_id> const & periodic_boundary_ids) const;

  Operators::ConvectiveKernel<dim, Number> kernel;
};
} // namespace ConvDiff

#endif
