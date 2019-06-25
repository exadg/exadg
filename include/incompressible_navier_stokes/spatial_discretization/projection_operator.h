/*
 * projection_operator.h
 *
 *  Created on: Jun 17, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_OPERATOR_H_

// TODO already included in OperatorBase
#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "../user_interface/input_parameters.h"

// TODO already included in OperatorBase
#include "../../operators/mapping_flags.h"
#include "../../solvers_and_preconditioners/util/block_jacobi_matrices.h"
#include "../../solvers_and_preconditioners/util/invert_diagonal.h"
#include "../../solvers_and_preconditioners/util/verify_calculation_of_diagonal.h"

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
struct ProjectionOperatorData
{
  ProjectionOperatorData()
    : use_divergence_penalty(true),
      use_continuity_penalty(true),
      implement_block_diagonal_preconditioner_matrix_free(false),
      use_cell_based_loops(false),
      preconditioner_block_jacobi(Elementwise::Preconditioner::InverseMassMatrix),
      block_jacobi_solver_data(SolverData(1000, 1.e-12, 1.e-2 /*rel_tol TODO*/, 1000))
  {
  }

  // specify which penalty terms to be used
  bool use_divergence_penalty, use_continuity_penalty;

  // TODO already in base class OperatorBase

  // block diagonal preconditioner
  bool implement_block_diagonal_preconditioner_matrix_free;

  // use cell based loops
  bool use_cell_based_loops;

  // elementwise iterative solution of block Jacobi problems
  Elementwise::Preconditioner preconditioner_block_jacobi;
  SolverData                  block_jacobi_solver_data;
};

template<int dim, typename Number>
class ProjectionOperator : public dealii::Subscriptor
{
private:
  // TODO can be removed later
  typedef ProjectionOperator<dim, Number> This;

  // TODO use definitions of base class
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> IntegratorCell;
  typedef FaceIntegrator<dim, dim, Number> IntegratorFace;

public:
  typedef Number value_type;

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells          = update_JxW_values | update_gradients;
    flags.inner_faces    = update_JxW_values | update_normal_vectors;
    flags.boundary_faces = update_JxW_values | update_normal_vectors;

    return flags;
  }

  ProjectionOperator(
    MatrixFree<dim, Number> const &                                  matrix_free_in,
    unsigned int const                                               dof_index_in,
    unsigned int const                                               quad_index_in,
    ProjectionOperatorData const                                     operator_data_in,
    std::shared_ptr<Operators::DivergencePenaltyKernel<dim, Number>> div_penalty_kernel,
    std::shared_ptr<Operators::ContinuityPenaltyKernel<dim, Number>> conti_penalty_kernel)
    : dealii::Subscriptor(),
      matrix_free(matrix_free_in),
      dof_index(dof_index_in),
      quad_index(quad_index_in),
      time_step_size(1.0),
      operator_data(operator_data_in),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
      block_diagonal_preconditioner_is_initialized(false)
  {
    if(operator_data.use_divergence_penalty)
      integrator.reset(
        new IntegratorCell(this->get_matrix_free(), this->get_dof_index(), this->get_quad_index()));

    if(operator_data.use_continuity_penalty)
    {
      integrator_m.reset(new IntegratorFace(
        this->get_matrix_free(), true, this->get_dof_index(), this->get_quad_index()));
      integrator_p.reset(new IntegratorFace(
        this->get_matrix_free(), false, this->get_dof_index(), this->get_quad_index()));
    }

    this->div_kernel   = div_penalty_kernel;
    this->conti_kernel = conti_penalty_kernel;
  }

  MatrixFree<dim, Number> const &
  get_matrix_free() const
  {
    return matrix_free;
  }

  AlignedVector<VectorizedArray<Number>> const &
  get_array_div_penalty_parameter() const
  {
    return div_kernel->get_array_penalty_parameter();
  }

  unsigned int
  get_dof_index() const
  {
    return dof_index;
  }

  unsigned int
  get_quad_index() const
  {
    return quad_index;
  }

  /*
   *  Set the time step size.
   */
  void
  set_time_step_size(double const & delta_t)
  {
    time_step_size = delta_t;
  }

  /*
   *  Get the time step size.
   */
  double
  get_time_step_size() const
  {
    return time_step_size;
  }

  void
  calculate_penalty_parameter(VectorType const & velocity)
  {
    if(operator_data.use_divergence_penalty)
      div_kernel->calculate_penalty_parameter(velocity);
    if(operator_data.use_continuity_penalty)
      conti_kernel->calculate_penalty_parameter(velocity);
  }

  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    apply(dst, src);
  }

  void
  apply(VectorType & dst, VectorType const & src) const
  {
    if(operator_data.use_divergence_penalty && operator_data.use_continuity_penalty)
      do_apply(dst, src, true);
    else if(operator_data.use_divergence_penalty && !operator_data.use_continuity_penalty)
      do_apply_mass_div_penalty(dst, src, true);
    else
      AssertThrow(false, ExcMessage("Not implemented."));
  }

  void
  apply_add(VectorType & dst, VectorType const & src) const
  {
    if(operator_data.use_divergence_penalty && operator_data.use_continuity_penalty)
      do_apply(dst, src, false);
    else if(operator_data.use_divergence_penalty && !operator_data.use_continuity_penalty)
      do_apply_mass_div_penalty(dst, src, false);
    else
      AssertThrow(false, ExcMessage("Not implemented."));
  }

  /*
   *  Calculate inverse diagonal which is needed for the Jacobi preconditioner.
   */
  void
  calculate_inverse_diagonal(VectorType & diagonal) const
  {
    calculate_diagonal(diagonal);

    // verify_calculation_of_diagonal(*this,diagonal);

    invert_diagonal(diagonal);
  }

  /*
   *  Initialize dof vector (required when using the Jacobi preconditioner).
   */
  void
  initialize_dof_vector(VectorType & vector) const
  {
    matrix_free.initialize_dof_vector(vector, dof_index);
  }

  /*
   * Block diagonal preconditioner.
   */

  // apply the inverse block diagonal operator (for matrix-based and matrix-free variants)
  void
  apply_inverse_block_diagonal(VectorType & dst, VectorType const & src) const
  {
    // matrix-free
    if(this->operator_data.implement_block_diagonal_preconditioner_matrix_free)
    {
      // Solve block Jacobi problems iteratively using an elementwise solver vectorized
      // over several elements.
      bool const variable_not_needed = false;
      elementwise_solver->solve(dst, src, variable_not_needed);
    }
    else // matrix based
    {
      // Simply apply inverse of block matrices (using the LU factorization that has been computed
      // before).
      matrix_free.cell_loop(&This::cell_loop_apply_inverse_block_diagonal, this, dst, src);
    }
  }

  /*
   * Update block diagonal preconditioner: initialize everything related to block diagonal
   * preconditioner when this function is called the first time. Recompute block matrices in case of
   * matrix-based implementation.
   */
  void
  update_block_diagonal_preconditioner() const
  {
    // initialization

    if(!block_diagonal_preconditioner_is_initialized)
    {
      if(operator_data.implement_block_diagonal_preconditioner_matrix_free)
      {
        initialize_block_diagonal_preconditioner_matrix_free();
      }
      else // matrix-based variant
      {
        // Note that the velocity has dim components.
        unsigned int dofs_per_cell = matrix_free.get_shape_info().dofs_per_component_on_cell * dim;

        matrices.resize(matrix_free.n_macro_cells() * VectorizedArray<Number>::n_array_elements,
                        LAPACKFullMatrix<Number>(dofs_per_cell, dofs_per_cell));
      }

      block_diagonal_preconditioner_is_initialized = true;
    }

    // update

    // For the matrix-free variant there is nothing to do.
    // For the matrix-based variant we have to recompute the block matrices.
    if(!operator_data.implement_block_diagonal_preconditioner_matrix_free)
    {
      // clear matrices
      initialize_block_jacobi_matrices_with_zero(matrices);

      // compute block matrices and add
      this->add_block_diagonal_matrices(matrices);

      calculate_lu_factorization_block_jacobi(matrices);
    }
  }

  void
  apply_add_block_diagonal_elementwise(unsigned int const   cell,
                                       scalar * const       dst,
                                       scalar const * const src,
                                       unsigned int const   problem_size = 1.0) const
  {
    (void)problem_size;

    if(operator_data.use_divergence_penalty)
    {
      integrator->reinit(cell);

      reinit_cell(*integrator);

      unsigned int dofs_per_cell = integrator->dofs_per_cell;

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        integrator->begin_dof_values()[i] = src[i];

      integrator->evaluate(true, true, false);

      do_cell_integral(*integrator);

      integrator->integrate(true, true);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        dst[i] += integrator->begin_dof_values()[i];
    }

    if(operator_data.use_continuity_penalty)
    {
      // face integrals
      unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
      for(unsigned int face = 0; face < n_faces; ++face)
      {
        integrator_m->reinit(cell, face);
        integrator_p->reinit(cell, face);

        auto bids = matrix_free.get_faces_by_cells_boundary_id(cell, face);
        auto bid  = bids[0];

        reinit_face_cell_based(bid, *integrator_m, *integrator_p);

        unsigned int dofs_per_cell = integrator_m->dofs_per_cell;

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator_m->begin_dof_values()[i] = src[i];

        // do not need to read dof values for integrator_p (already initialized with 0)

        integrator_m->evaluate(true, false);

        if(bid == numbers::internal_face_boundary_id) // internal face
        {
          do_face_int_integral(*integrator_m, *integrator_p);
        }
        else // boundary face
        {
          do_boundary_integral(*integrator_m, OperatorType::homogeneous, bid);
        }

        integrator_m->integrate(true, false);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          dst[i] += integrator_m->begin_dof_values()[i];
      }
    }
  }

private:
  void
  do_apply_mass_div_penalty(VectorType &       dst,
                            VectorType const & src,
                            bool const         zero_dst_vector) const
  {
    matrix_free.cell_loop(&This::cell_loop, this, dst, src, zero_dst_vector);
  }

  void
  do_apply(VectorType & dst, VectorType const & src, bool const zero_dst_vector) const
  {
    matrix_free.loop(&This::cell_loop,
                     &This::face_loop,
                     &This::boundary_face_loop_empty,
                     this,
                     dst,
                     src,
                     zero_dst_vector,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  void
  reinit_cell(IntegratorCell & integrator) const
  {
    div_kernel->reinit_cell(integrator);
  }

  void
  reinit_face(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    conti_kernel->reinit_face(integrator_m, integrator_p);
  }

  void
  reinit_face_cell_based(types::boundary_id const boundary_id,
                         IntegratorFace &         integrator_m,
                         IntegratorFace &         integrator_p) const
  {
    conti_kernel->reinit_face_cell_based(boundary_id, integrator_m, integrator_p);
  }

  void
  do_cell_integral(IntegratorCell & integrator) const
  {
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      integrator.submit_value(integrator.get_value(q), q);
      integrator.submit_divergence(time_step_size * div_kernel->get_volume_flux(integrator, q), q);
    }
  }

  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      vector u_m      = integrator_m.get_value(q);
      vector u_p      = integrator_p.get_value(q);
      vector normal_m = integrator_m.get_normal_vector(q);

      vector flux = time_step_size * conti_kernel->calculate_flux(u_m, u_p, normal_m);

      integrator_m.submit_value(flux, q);
      integrator_p.submit_value(-flux, q);
    }
  }

  void
  do_face_int_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    (void)integrator_p;

    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      vector u_m = integrator_m.get_value(q);
      vector u_p; // set u_p to zero
      vector normal_m = integrator_m.get_normal_vector(q);

      vector flux = time_step_size * conti_kernel->calculate_flux(u_m, u_p, normal_m);

      integrator_m.submit_value(flux, q);
    }
  }

  void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    (void)integrator_m;

    for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
    {
      vector u_m; // set u_m to zero
      vector u_p      = integrator_p.get_value(q);
      vector normal_p = -integrator_p.get_normal_vector(q);

      vector flux = time_step_size * conti_kernel->calculate_flux(u_p, u_m, normal_p);

      integrator_p.submit_value(flux, q);
    }
  }

  void
  do_boundary_integral(IntegratorFace &           integrator_m,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id) const
  {
    (void)operator_type;
    (void)boundary_id;

    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      vector flux; // continuity penalty term is zero on boundary faces

      integrator_m.submit_value(flux, q);
    }
  }


  void
  cell_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const
  {
    IntegratorCell integrator(data, dof_index, quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);
      integrator.gather_evaluate(src, true, true);

      reinit_cell(integrator);

      do_cell_integral(integrator);

      integrator.integrate_scatter(true, true, dst);
    }
  }

  void
  cell_loop_empty(MatrixFree<dim, Number> const & /*data*/,
                  VectorType & /*dst*/,
                  VectorType const & /*src*/,
                  Range const & /*cell_range*/) const
  {
    // do nothing
  }

  void
  face_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   face_range) const
  {
    IntegratorFace integrator_m(data, true, dof_index, quad_index);
    IntegratorFace integrator_p(data, false, dof_index, quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_m.reinit(face);
      integrator_p.reinit(face);

      integrator_m.gather_evaluate(src, true, false);
      integrator_p.gather_evaluate(src, true, false);

      reinit_face(integrator_m, integrator_p);

      do_face_integral(integrator_m, integrator_p);

      integrator_m.integrate_scatter(true, false, dst);
      integrator_p.integrate_scatter(true, false, dst);
    }
  }

  void
  boundary_face_loop_empty(MatrixFree<dim, Number> const &,
                           VectorType &,
                           VectorType const &,
                           Range const &) const
  {
    // do nothing
  }

  /*
   *  This function calculates the diagonal of the projection operator including the mass matrix,
   * divergence penalty and continuity penalty operators. A prerequisite to call this function is
   * that the time step size is set correctly.
   */
  void
  calculate_diagonal(VectorType & diagonal) const
  {
    VectorType src_dummy(diagonal);
    matrix_free.loop(&This::cell_loop_diagonal,
                     &This::face_loop_diagonal,
                     &This::boundary_face_loop_diagonal,
                     this,
                     diagonal,
                     src_dummy,
                     true /*zero dst vector = true*/,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  /*
   * Calculation of diagonal (cell loop).
   */
  void
  cell_loop_diagonal(MatrixFree<dim, Number> const & data,
                     VectorType &                    dst,
                     VectorType const & /*src*/,
                     Range const & cell_range) const
  {
    IntegratorCell integrator(data, dof_index, quad_index);

    unsigned int const    dofs_per_cell = integrator.dofs_per_cell;
    AlignedVector<scalar> local_diagonal_vector(dofs_per_cell);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);

      reinit_cell(integrator);

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator.evaluate(true, true);

        do_cell_integral(integrator);

        integrator.integrate(true, true);

        local_diagonal_vector[j] = integrator.begin_dof_values()[j];
      }
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        integrator.begin_dof_values()[j] = local_diagonal_vector[j];

      integrator.distribute_local_to_global(dst);
    }
  }

  /*
   * Calculation of diagonal (face loop).
   */
  void
  face_loop_diagonal(MatrixFree<dim, Number> const & data,
                     VectorType &                    dst,
                     VectorType const & /*src*/,
                     Range const & face_range) const
  {
    IntegratorFace integrator_m(data, true, dof_index, quad_index);
    IntegratorFace integrator_p(data, false, dof_index, quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_m.reinit(face);
      integrator_p.reinit(face);

      reinit_face(integrator_m, integrator_p);

      // element-
      unsigned int const    dofs_per_cell = integrator_m.dofs_per_cell;
      AlignedVector<scalar> local_diagonal_vector(dofs_per_cell);
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator_m.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator_m.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator_m.evaluate(true, false);

        do_face_int_integral(integrator_m, integrator_p);

        integrator_m.integrate(true, false);

        local_diagonal_vector[j] = integrator_m.begin_dof_values()[j];
      }
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        integrator_m.begin_dof_values()[j] = local_diagonal_vector[j];

      integrator_m.distribute_local_to_global(dst);

      // neighbor (element+)
      unsigned int const    dofs_per_cell_neighbor = integrator_p.dofs_per_cell;
      AlignedVector<scalar> local_diagonal_vector_neighbor(dofs_per_cell_neighbor);
      for(unsigned int j = 0; j < dofs_per_cell_neighbor; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for(unsigned int i = 0; i < dofs_per_cell_neighbor; ++i)
          integrator_p.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator_p.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator_p.evaluate(true, false);

        do_face_ext_integral(integrator_m, integrator_p);

        integrator_p.integrate(true, false);

        local_diagonal_vector_neighbor[j] = integrator_p.begin_dof_values()[j];
      }
      for(unsigned int j = 0; j < dofs_per_cell_neighbor; ++j)
        integrator_p.begin_dof_values()[j] = local_diagonal_vector_neighbor[j];

      integrator_p.distribute_local_to_global(dst);
    }
  }

  /*
   * Calculation of diagonal (boundary face loop).
   */
  void
  boundary_face_loop_diagonal(MatrixFree<dim, Number> const &,
                              VectorType &,
                              VectorType const &,
                              Range const &) const
  {
    // do nothing
  }

  void
  initialize_block_diagonal_preconditioner_matrix_free() const
  {
    elementwise_operator.reset(new ELEMENTWISE_OPERATOR(*this));

    if(this->operator_data.preconditioner_block_jacobi == Elementwise::Preconditioner::None)
    {
      typedef Elementwise::PreconditionerIdentity<VectorizedArray<Number>> IDENTITY;
      elementwise_preconditioner.reset(new IDENTITY(elementwise_operator->get_problem_size()));
    }
    else if(this->operator_data.preconditioner_block_jacobi ==
            Elementwise::Preconditioner::InverseMassMatrix)
    {
      typedef Elementwise::InverseMassMatrixPreconditioner<dim, dim, Number> INVERSE_MASS;

      elementwise_preconditioner.reset(
        new INVERSE_MASS(this->get_matrix_free(), this->get_dof_index(), this->get_quad_index()));
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    Elementwise::IterativeSolverData iterative_solver_data;
    iterative_solver_data.solver_type = Elementwise::Solver::CG;
    iterative_solver_data.solver_data = this->operator_data.block_jacobi_solver_data;

    elementwise_solver.reset(new ELEMENTWISE_SOLVER(
      *std::dynamic_pointer_cast<ELEMENTWISE_OPERATOR>(elementwise_operator),
      *std::dynamic_pointer_cast<PRECONDITIONER_BASE>(elementwise_preconditioner),
      iterative_solver_data));
  }

  void
  add_block_diagonal_matrices(std::vector<LAPACKFullMatrix<Number>> & matrices) const
  {
    VectorType src;

    if(operator_data.use_cell_based_loops)
    {
      matrix_free.cell_loop(&This::cell_based_loop_calculate_block_diagonal, this, matrices, src);
    }
    else
    {
      AssertThrow(
        n_mpi_processes == 1,
        ExcMessage(
          "Block diagonal calculation with separate loops over cells and faces only works in serial. "
          "Use cell based loops for parallel computations."));

      matrix_free.loop(&This::cell_loop_calculate_block_diagonal,
                       &This::face_loop_calculate_block_diagonal,
                       &This::boundary_face_loop_calculate_block_diagonal,
                       this,
                       matrices,
                       src);
    }
  }


  void
  cell_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         data,
                                     std::vector<LAPACKFullMatrix<Number>> & matrices,
                                     VectorType const &,
                                     Range const & cell_range) const
  {
    IntegratorCell integrator(data, dof_index, quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);

      reinit_cell(integrator);

      unsigned int dofs_per_cell = integrator.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator.evaluate(true, true);

        do_cell_integral(integrator);

        integrator.integrate(true, true);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
            matrices[cell * VectorizedArray<Number>::n_array_elements + v](i, j) +=
              integrator.begin_dof_values()[i][v];
      }
    }
  }

  void
  face_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         data,
                                     std::vector<LAPACKFullMatrix<Number>> & matrices,
                                     VectorType const &,
                                     Range const & face_range) const
  {
    IntegratorFace integrator_m(data, true, dof_index, quad_index);
    IntegratorFace integrator_p(data, false, dof_index, quad_index);

    // Perform face integrals for element e⁻.
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_m.reinit(face);
      integrator_p.reinit(face);

      reinit_face(integrator_m, integrator_p);

      unsigned int dofs_per_cell = integrator_m.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator_m.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator_m.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator_m.evaluate(true, false);

        do_face_int_integral(integrator_m, integrator_p);

        integrator_m.integrate(true, false);

        for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_interior[v];
          if(cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
              matrices[cell_number](i, j) += integrator_m.begin_dof_values()[i][v];
        }
      }
    }

    // Perform face integrals for element e⁺.
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_m.reinit(face);
      integrator_p.reinit(face);

      reinit_face(integrator_m, integrator_p);

      // Note that the velocity has dim components.
      unsigned int dofs_per_cell = integrator_p.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator_p.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator_p.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator_p.evaluate(true, false);

        do_face_ext_integral(integrator_m, integrator_p);

        integrator_p.integrate(true, false);

        for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_exterior[v];
          if(cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
              matrices[cell_number](i, j) += integrator_p.begin_dof_values()[i][v];
        }
      }
    }
  }

  void
  boundary_face_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &,
                                              std::vector<LAPACKFullMatrix<Number>> &,
                                              VectorType const &,
                                              Range const &) const
  {
    // do nothing
  }

  void
  cell_based_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         data,
                                           std::vector<LAPACKFullMatrix<Number>> & matrices,
                                           VectorType const &,
                                           Range const & cell_range) const
  {
    IntegratorCell integrator(data, dof_index, quad_index);
    IntegratorFace integrator_m(data, true, dof_index, quad_index);
    IntegratorFace integrator_p(data, false, dof_index, quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // cell integral
      unsigned int const n_filled_lanes = data.n_active_entries_per_cell_batch(cell);

      integrator.reinit(cell);

      reinit_cell(integrator);

      unsigned int dofs_per_cell = integrator.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator.evaluate(true, true);

        do_cell_integral(integrator);

        integrator.integrate(true, true);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int v = 0; v < n_filled_lanes; ++v)
            matrices[cell * VectorizedArray<Number>::n_array_elements + v](i, j) +=
              integrator.begin_dof_values()[i][v];
      }

      // loop over all faces
      unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
      for(unsigned int face = 0; face < n_faces; ++face)
      {
        integrator_m.reinit(cell, face);
        integrator_p.reinit(cell, face);
        auto bids = data.get_faces_by_cells_boundary_id(cell, face);
        auto bid  = bids[0];

        reinit_face_cell_based(bid, integrator_m, integrator_p);

        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            integrator_m.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
          integrator_m.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

          integrator_m.evaluate(true, false);

          if(bid == numbers::internal_face_boundary_id) // internal face
          {
            do_face_int_integral(integrator_m, integrator_p);
          }
          else // boundary face
          {
            do_boundary_integral(integrator_m, OperatorType::homogeneous, bid);
          }

          integrator_m.integrate(true, false);

          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            for(unsigned int v = 0; v < n_filled_lanes; ++v)
              matrices[cell * VectorizedArray<Number>::n_array_elements + v](i, j) +=
                integrator_m.begin_dof_values()[i][v];
        }
      }
    }
  }

  /*
   * Apply inverse block diagonal:
   *
   * instead of applying the block matrix B we compute dst = B^{-1} * src (LU factorization
   * should have already been performed with the method update_inverse_block_diagonal())
   */
  void
  cell_loop_apply_inverse_block_diagonal(MatrixFree<dim, Number> const & data,
                                         VectorType &                    dst,
                                         VectorType const &              src,
                                         Range const &                   cell_range) const
  {
    IntegratorCell integrator(data, dof_index, quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);
      integrator.read_dof_values(src);

      unsigned int dofs_per_cell = integrator.dofs_per_cell;

      for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
      {
        // fill source vector
        Vector<Number> src_vector(dofs_per_cell);
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
          src_vector(j) = integrator.begin_dof_values()[j][v];

        // apply inverse matrix
        matrices[cell * VectorizedArray<Number>::n_array_elements + v].solve(src_vector, false);

        // write solution to dst-vector
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
          integrator.begin_dof_values()[j][v] = src_vector(j);
      }

      integrator.set_dof_values(dst);
    }
  }

  MatrixFree<dim, Number> const & matrix_free;

  unsigned int const dof_index;
  unsigned int const quad_index;

  double time_step_size;

  ProjectionOperatorData operator_data;

  unsigned int n_mpi_processes;

  /*
   * Vector of matrices for block-diagonal preconditioners.
   */
  mutable std::vector<LAPACKFullMatrix<Number>> matrices;

  /*
   * We want to initialize the block diagonal preconditioner (block diagonal matrices or elementwise
   * iterative solvers in case of matrix-free implementation) only once, so we store the status of
   * initialization in a variable.
   */
  mutable bool block_diagonal_preconditioner_is_initialized;


  /*
   * Block Jacobi preconditioner/smoother: matrix-free version with elementwise iterative solver
   */
  typedef Elementwise::OperatorBase<dim, Number, This>             ELEMENTWISE_OPERATOR;
  typedef Elementwise::PreconditionerBase<VectorizedArray<Number>> PRECONDITIONER_BASE;
  typedef Elementwise::IterativeSolver<dim, dim, Number, ELEMENTWISE_OPERATOR, PRECONDITIONER_BASE>
    ELEMENTWISE_SOLVER;

  mutable std::shared_ptr<ELEMENTWISE_OPERATOR> elementwise_operator;
  mutable std::shared_ptr<PRECONDITIONER_BASE>  elementwise_preconditioner;
  mutable std::shared_ptr<ELEMENTWISE_SOLVER>   elementwise_solver;

  /*
   * FEEvaluation objects required for elementwise block Jacobi operations
   */
  std::shared_ptr<IntegratorCell> integrator;
  std::shared_ptr<IntegratorFace> integrator_m;
  std::shared_ptr<IntegratorFace> integrator_p;

  std::shared_ptr<Operators::DivergencePenaltyKernel<dim, Number>> div_kernel;
  std::shared_ptr<Operators::ContinuityPenaltyKernel<dim, Number>> conti_kernel;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_OPERATOR_H_ \
        */
