/*
 * projection_operator.h
 *
 *  Created on: Jun 17, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_OPERATOR_H_

#include "../user_interface/input_parameters.h"

#include "../../operators/operator_base.h"

using namespace dealii;

namespace IncNS
{
/*
 *  Combined divergence and continuity penalty operator: applies the operation
 *
 *   mass matrix operator + dt * divergence penalty operator + dt * continuity penalty operator .
 *
 *  The divergence and continuity penalty operators can also be applied separately. In detail
 *
 *  Mass matrix operator: ( v_h , u_h )_Omega^e where
 *   v_h : test function
 *   u_h : solution
 *
 *
 *  Divergence penalty operator: ( div(v_h) , tau_div * div(u_h) )_Omega^e where
 *   v_h : test function
 *   u_h : solution
 *   tau_div: divergence penalty factor
 *
 *            use convective term:  tau_div_conv = K * ||U||_mean * h_eff
 *
 *                                  where h_eff = h / (k_u+1) and
 *                                  h = V_e^{1/3} with the element volume V_e
 *
 *            use viscous term:     tau_div_viscous = K * nu
 *
 *            use both terms:       tau_div = tau_div_conv + tau_div_viscous
 *
 *
 *  Continuity penalty operator: ( v_h , tau_conti * jump(u_h) )_dOmega^e where
 *   v_h : test function
 *   u_h : solution
 *
 *   jump(u_h) = u_h^{-} - u_h^{+} or ( (u_h^{-} - u_h^{+})*normal ) * normal
 *
 *     where "-" denotes interior information and "+" exterior information
 *
 *   tau_conti: continuity penalty factor
 *
 *            use convective term:  tau_conti_conv = K * ||U||_mean
 *
 *            use viscous term:     tau_conti_viscous = K * nu / h
 *
 *                                  where h_eff = h / (k_u+1) and
 *                                  h = V_e^{1/3} with the element volume V_e
 *
 *            use both terms:       tau_conti = tau_conti_conv + tau_conti_viscous
 */

namespace Operators
{
struct DivergencePenaltyKernelData
{
  DivergencePenaltyKernelData()
    : type_penalty_parameter(TypePenaltyParameter::ConvectiveTerm),
      viscosity(0.0),
      degree(1),
      penalty_factor(1.0),
      dof_index(0),
      quad_index(0)
  {
  }

  // type of penalty parameter (viscous and/or convective terms)
  TypePenaltyParameter type_penalty_parameter;

  // viscosity, needed for computation of penalty factor
  double viscosity;

  // degree of finite element shape functions
  unsigned int degree;

  // the penalty term can be scaled by 'penalty_factor'
  double penalty_factor;

  unsigned int dof_index;
  unsigned int quad_index;
};

template<int dim, typename Number>
class DivergencePenaltyKernel
{
private:
  typedef CellIntegrator<dim, dim, Number> IntegratorCell;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number> scalar;

public:
  DivergencePenaltyKernel() : matrix_free(nullptr), array_penalty_parameter(0)
  {
  }

  void
  reinit(MatrixFree<dim, Number> const & matrix_free, DivergencePenaltyKernelData const & data)
  {
    this->matrix_free = &matrix_free;

    this->data = data;

    unsigned int n_cells = matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
    array_penalty_parameter.resize(n_cells);
  }

  IntegratorFlags
  get_integrator_flags() const
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = CellFlags(false, true, false);
    flags.cell_integrate = CellFlags(false, true, false);

    // no face integrals

    return flags;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells = update_JxW_values | update_gradients;

    // no face integrals

    return flags;
  }

  // TODO remove this later if possible
  AlignedVector<VectorizedArray<Number>> const &
  get_array_penalty_parameter() const
  {
    return array_penalty_parameter;
  }

  void
  calculate_penalty_parameter(VectorType const & velocity)
  {
    velocity.update_ghost_values();

    IntegratorCell integrator(*matrix_free, data.dof_index, data.quad_index);

    AlignedVector<scalar> JxW_values(integrator.n_q_points);

    unsigned int n_cells = matrix_free->n_cell_batches() + matrix_free->n_ghost_cell_batches();
    for(unsigned int cell = 0; cell < n_cells; ++cell)
    {
      scalar tau_convective = make_vectorized_array<Number>(0.0);
      scalar tau_viscous    = make_vectorized_array<Number>(data.viscosity);

      if(data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm ||
         data.type_penalty_parameter == TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        integrator.reinit(cell);
        integrator.read_dof_values(velocity);
        integrator.evaluate(true, false);

        scalar volume      = make_vectorized_array<Number>(0.0);
        scalar norm_U_mean = make_vectorized_array<Number>(0.0);
        for(unsigned int q = 0; q < integrator.n_q_points; ++q)
        {
          volume += integrator.JxW(q);
          norm_U_mean += integrator.JxW(q) * integrator.get_value(q).norm();
        }
        norm_U_mean /= volume;

        tau_convective =
          norm_U_mean * std::exp(std::log(volume) / (double)dim) / (double)(data.degree + 1);
      }

      if(data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm)
      {
        array_penalty_parameter[cell] = data.penalty_factor * tau_convective;
      }
      else if(data.type_penalty_parameter == TypePenaltyParameter::ViscousTerm)
      {
        array_penalty_parameter[cell] = data.penalty_factor * tau_viscous;
      }
      else if(data.type_penalty_parameter == TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        array_penalty_parameter[cell] = data.penalty_factor * (tau_convective + tau_viscous);
      }
    }
  }

  void
  reinit_cell(IntegratorCell & integrator) const
  {
    tau = integrator.read_cell_data(array_penalty_parameter);
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    get_volume_flux(IntegratorCell const & integrator, unsigned int const q) const
  {
    return tau * integrator.get_divergence(q);
  }

private:
  MatrixFree<dim, Number> const * matrix_free;

  DivergencePenaltyKernelData data;

  AlignedVector<scalar> array_penalty_parameter;

  mutable scalar tau;
};

struct ContinuityPenaltyKernelData
{
  ContinuityPenaltyKernelData()
    : type_penalty_parameter(TypePenaltyParameter::ConvectiveTerm),
      which_components(ContinuityPenaltyComponents::Normal),
      viscosity(0.0),
      degree(1),
      penalty_factor(1.0),
      dof_index(0),
      quad_index(0)
  {
  }

  // type of penalty parameter (viscous and/or convective terms)
  TypePenaltyParameter type_penalty_parameter;

  // the continuity penalty term can be applied to all velocity components or to the normal
  // component only
  ContinuityPenaltyComponents which_components;

  // viscosity, needed for computation of penalty factor
  double viscosity;

  // degree of finite element shape functions
  unsigned int degree;

  // the penalty term can be scaled by 'penalty_factor'
  double penalty_factor;

  unsigned int dof_index;
  unsigned int quad_index;
};

template<int dim, typename Number>
class ContinuityPenaltyKernel
{
private:
  typedef CellIntegrator<dim, dim, Number> IntegratorCell;
  typedef FaceIntegrator<dim, dim, Number> IntegratorFace;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

public:
  ContinuityPenaltyKernel() : matrix_free(nullptr), array_penalty_parameter(0)
  {
  }

  void
  reinit(MatrixFree<dim, Number> const & matrix_free, ContinuityPenaltyKernelData const & data)
  {
    this->matrix_free = &matrix_free;

    this->data = data;

    unsigned int n_cells = matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
    array_penalty_parameter.resize(n_cells);
  }

  IntegratorFlags
  get_integrator_flags() const
  {
    IntegratorFlags flags;

    // no cell integrals

    flags.face_evaluate  = FaceFlags(true, false);
    flags.face_integrate = FaceFlags(true, false);

    return flags;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    // no cell integrals

    flags.inner_faces = update_JxW_values | update_normal_vectors;

    // no boundary face integrals

    return flags;
  }

  void
  calculate_penalty_parameter(VectorType const & velocity)
  {
    velocity.update_ghost_values();

    IntegratorCell integrator(*matrix_free, data.dof_index, data.quad_index);

    AlignedVector<scalar> JxW_values(integrator.n_q_points);

    unsigned int n_cells = matrix_free->n_cell_batches() + matrix_free->n_ghost_cell_batches();
    for(unsigned int cell = 0; cell < n_cells; ++cell)
    {
      integrator.reinit(cell);
      integrator.read_dof_values(velocity);
      integrator.evaluate(true, false);
      scalar volume      = make_vectorized_array<Number>(0.0);
      scalar norm_U_mean = make_vectorized_array<Number>(0.0);

      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        volume += integrator.JxW(q);
        norm_U_mean += integrator.JxW(q) * integrator.get_value(q).norm();
      }

      norm_U_mean /= volume;

      scalar tau_convective = norm_U_mean;
      scalar h              = std::exp(std::log(volume) / (double)dim) / (double)(data.degree + 1);
      scalar tau_viscous    = make_vectorized_array<Number>(data.viscosity) / h;

      if(data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm)
      {
        array_penalty_parameter[cell] = data.penalty_factor * tau_convective;
      }
      else if(data.type_penalty_parameter == TypePenaltyParameter::ViscousTerm)
      {
        array_penalty_parameter[cell] = data.penalty_factor * tau_viscous;
      }
      else if(data.type_penalty_parameter == TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        array_penalty_parameter[cell] = data.penalty_factor * (tau_convective + tau_viscous);
      }
    }
  }

  void
  reinit_face(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    tau = 0.5 * (integrator_m.read_cell_data(array_penalty_parameter) +
                 integrator_p.read_cell_data(array_penalty_parameter));
  }

  void
  reinit_face_cell_based(types::boundary_id const boundary_id,
                         IntegratorFace &         integrator_m,
                         IntegratorFace &         integrator_p) const
  {
    if(boundary_id == numbers::internal_face_boundary_id) // internal face
    {
      tau = 0.5 * (integrator_m.read_cell_data(array_penalty_parameter) +
                   integrator_p.read_cell_data(array_penalty_parameter));
    }
    else // boundary face
    {
      // will not be used since the continuity penalty operator is zero on boundary faces
      tau = integrator_m.read_cell_data(array_penalty_parameter);
    }
  }

  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_flux(vector const & u_m, vector const & u_p, vector const & normal_m) const
  {
    vector jump_value = u_m - u_p;

    vector flux;

    if(data.which_components == ContinuityPenaltyComponents::All)
    {
      // penalize all velocity components
      flux = tau * jump_value;
    }
    else if(data.which_components == ContinuityPenaltyComponents::Normal)
    {
      flux = tau * (jump_value * normal_m) * normal_m;
    }
    else
    {
      AssertThrow(false, ExcMessage("not implemented."));
    }

    return flux;
  }


private:
  MatrixFree<dim, Number> const * matrix_free;

  ContinuityPenaltyKernelData data;

  AlignedVector<scalar> array_penalty_parameter;

  mutable scalar tau;
};

} // namespace Operators

struct DivergencePenaltyData
{
  DivergencePenaltyData() : dof_index(0), quad_index(0)
  {
  }

  unsigned int dof_index;

  unsigned int quad_index;
};

template<int dim, typename Number>
class DivergencePenaltyOperator
{
private:
  typedef DivergencePenaltyOperator<dim, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> IntegratorCell;

public:
  DivergencePenaltyOperator() : matrix_free(nullptr)
  {
  }

  void
  reinit(MatrixFree<dim, Number> const &                                        matrix_free,
         DivergencePenaltyData const &                                          data,
         std::shared_ptr<Operators::DivergencePenaltyKernel<dim, Number>> const kernel)
  {
    this->matrix_free = &matrix_free;
    this->data        = data;
    this->kernel      = kernel;
  }

  void
  update(VectorType const & velocity)
  {
    kernel->calculate_penalty_parameter(velocity);
  }

  void
  apply(VectorType & dst, VectorType const & src) const
  {
    matrix_free->cell_loop(&This::cell_loop, this, dst, src, true);
  }

  void
  apply_add(VectorType & dst, VectorType const & src) const
  {
    matrix_free->cell_loop(&This::cell_loop, this, dst, src, false);
  }

private:
  void
  cell_loop(MatrixFree<dim, Number> const & matrix_free,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   range) const
  {
    IntegratorCell integrator(matrix_free, data.dof_index, data.quad_index);

    for(unsigned int cell = range.first; cell < range.second; ++cell)
    {
      integrator.reinit(cell);
      integrator.gather_evaluate(src, false, true);

      kernel->reinit_cell(integrator);

      do_cell_integral(integrator);

      integrator.integrate_scatter(false, true, dst);
    }
  }

  void
  do_cell_integral(IntegratorCell & integrator) const
  {
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      integrator.submit_divergence(kernel->get_volume_flux(integrator, q), q);
    }
  }

  MatrixFree<dim, Number> const * matrix_free;

  DivergencePenaltyData data;

  std::shared_ptr<Operators::DivergencePenaltyKernel<dim, Number>> kernel;
};

struct ContinuityPenaltyData
{
  ContinuityPenaltyData() : dof_index(0), quad_index(0)
  {
  }

  unsigned int dof_index;

  unsigned int quad_index;
};

template<int dim, typename Number>
class ContinuityPenaltyOperator
{
private:
  typedef ContinuityPenaltyOperator<dim, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef FaceIntegrator<dim, dim, Number> IntegratorFace;

public:
  ContinuityPenaltyOperator() : matrix_free(nullptr)
  {
  }

  void
  reinit(MatrixFree<dim, Number> const &                                        matrix_free,
         ContinuityPenaltyData const &                                          data,
         std::shared_ptr<Operators::ContinuityPenaltyKernel<dim, Number>> const kernel)
  {
    this->matrix_free = &matrix_free;
    this->data        = data;
    this->kernel      = kernel;
  }

  void
  update(VectorType const & velocity)
  {
    kernel->calculate_penalty_parameter(velocity);
  }

  void
  apply(VectorType & dst, VectorType const & src) const
  {
    matrix_free->loop(&This::cell_loop_empty,
                      &This::face_loop,
                      &This::boundary_face_loop_empty,
                      this,
                      dst,
                      src,
                      true,
                      MatrixFree<dim, Number>::DataAccessOnFaces::values,
                      MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  void
  apply_add(VectorType & dst, VectorType const & src) const
  {
    matrix_free->loop(&This::cell_loop_empty,
                      &This::face_loop,
                      &This::boundary_face_loop_empty,
                      this,
                      dst,
                      src,
                      false,
                      MatrixFree<dim, Number>::DataAccessOnFaces::values,
                      MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

private:
  void
  cell_loop_empty(MatrixFree<dim, Number> const & matrix_free,
                  VectorType &                    dst,
                  VectorType const &              src,
                  Range const &                   range) const
  {
    (void)matrix_free;
    (void)dst;
    (void)src;
    (void)range;
  }

  void
  face_loop(MatrixFree<dim, Number> const & matrix_free,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   face_range) const
  {
    IntegratorFace integrator_m(matrix_free, true, data.dof_index, data.quad_index);
    IntegratorFace integrator_p(matrix_free, false, data.dof_index, data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_m.reinit(face);
      integrator_p.reinit(face);

      integrator_m.gather_evaluate(src, true, false);
      integrator_p.gather_evaluate(src, true, false);

      kernel->reinit_face(integrator_m, integrator_p);

      do_face_integral(integrator_m, integrator_p);

      integrator_m.integrate_scatter(true, false, dst);
      integrator_p.integrate_scatter(true, false, dst);
    }
  }

  void
  boundary_face_loop_empty(MatrixFree<dim, Number> const & matrix_free,
                           VectorType &                    dst,
                           VectorType const &              src,
                           Range const &                   range) const
  {
    (void)matrix_free;
    (void)dst;
    (void)src;
    (void)range;
  }

  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      vector u_m      = integrator_m.get_value(q);
      vector u_p      = integrator_p.get_value(q);
      vector normal_m = integrator_m.get_normal_vector(q);

      vector flux = kernel->calculate_flux(u_m, u_p, normal_m);

      integrator_m.submit_value(flux, q);
      integrator_p.submit_value(-flux, q);
    }
  }

  MatrixFree<dim, Number> const * matrix_free;

  ContinuityPenaltyData data;

  std::shared_ptr<Operators::ContinuityPenaltyKernel<dim, Number>> kernel;
};

/*
 *  Operator data.
 */
struct ProjectionOperatorData : public OperatorBaseData
{
  ProjectionOperatorData()
    : OperatorBaseData(0 /* dof_index */, 0 /* quad_index */),
      use_divergence_penalty(true),
      use_continuity_penalty(true)
  {
  }

  // specify which penalty terms are used
  bool use_divergence_penalty, use_continuity_penalty;
};

template<int dim, typename Number>
class ProjectionOperator : public OperatorBase<dim, Number, ProjectionOperatorData, dim>
{
private:
  typedef OperatorBase<dim, Number, ProjectionOperatorData, dim> Base;

  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef typename Base::VectorType     VectorType;
  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

public:
  typedef Number value_type;

  void
  reinit(MatrixFree<dim, Number> const &   matrix_free,
         AffineConstraints<double> const & constraint_matrix,
         ProjectionOperatorData const &    data) const;

  void
  reinit(MatrixFree<dim, Number> const &                                  matrix_free,
         AffineConstraints<double> const &                                constraint_matrix,
         ProjectionOperatorData const &                                   data,
         std::shared_ptr<Operators::DivergencePenaltyKernel<dim, Number>> div_penalty_kernel,
         std::shared_ptr<Operators::ContinuityPenaltyKernel<dim, Number>> conti_penalty_kernel);

  // TODO remove if possible
  AlignedVector<VectorizedArray<Number>> const &
  get_array_div_penalty_parameter() const
  {
    return div_kernel->get_array_penalty_parameter();
  }

  // TODO remove if possible
  /*
   *  Get the time step size.
   */
  double
  get_time_step_size() const
  {
    return time_step_size;
  }

  void
  update(VectorType const & velocity, double const & dt);

private:
  void
  reinit_cell(unsigned int const cell) const;

  void
  reinit_face(unsigned int const face) const;

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

  double time_step_size;

  std::shared_ptr<Operators::DivergencePenaltyKernel<dim, Number>> div_kernel;
  std::shared_ptr<Operators::ContinuityPenaltyKernel<dim, Number>> conti_kernel;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_OPERATOR_H_ \
        */
