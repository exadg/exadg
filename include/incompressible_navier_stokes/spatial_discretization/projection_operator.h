/*
 * projection_operator.h
 *
 *  Created on: Jun 17, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_OPERATOR_H_

#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "../user_interface/input_parameters.h"

#include "../../operators/linear_operator_base.h"
#include "../../solvers_and_preconditioners/util/block_jacobi_matrices.h"
#include "../../solvers_and_preconditioners/util/invert_diagonal.h"
#include "../../solvers_and_preconditioners/util/verify_calculation_of_diagonal.h"

#include "operators/elementwise_operator.h"
#include "solvers_and_preconditioners/preconditioner/elementwise_preconditioners.h"
#include "solvers_and_preconditioners/solvers/wrapper_elementwise_solvers.h"

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

/*
 *  Operator data.
 */
struct ProjectionOperatorData
{
  ProjectionOperatorData()
    : type_penalty_parameter(TypePenaltyParameter::ConvectiveTerm),
      viscosity(0.0),
      use_divergence_penalty(true),
      use_continuity_penalty(true),
      degree(1),
      penalty_factor_div(1.0),
      penalty_factor_conti(1.0),
      which_components(ContinuityPenaltyComponents::Normal),
      implement_block_diagonal_preconditioner_matrix_free(false),
      use_cell_based_loops(false),
      preconditioner_block_jacobi(PreconditionerBlockDiagonal::InverseMassMatrix),
      block_jacobi_solver_data(SolverData(1000, 1.e-12, 1.e-1 /*rel_tol TODO*/, 1000))
  {
  }

  // type of penalty parameter (viscous and/or convective terms)
  TypePenaltyParameter type_penalty_parameter;

  // kinematic viscosity
  double viscosity;

  // specify which penalty terms to be used
  bool use_divergence_penalty, use_continuity_penalty;

  // degree of finite element shape functions
  unsigned int degree;

  // scaling factor
  double penalty_factor_div, penalty_factor_conti;

  // the continuity penalty term can be applied to all velocity components or to the normal
  // component only
  ContinuityPenaltyComponents which_components;

  // block diagonal preconditioner
  bool implement_block_diagonal_preconditioner_matrix_free;

  // use cell based loops
  bool use_cell_based_loops;

  // elementwise iterative solution of block Jacobi problems
  PreconditionerBlockDiagonal preconditioner_block_jacobi;
  SolverData                  block_jacobi_solver_data;
};

template<int dim, typename Number>
class ProjectionOperator : public LinearOperatorBase
{
private:
  typedef ProjectionOperator<dim, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorU;
  typedef FaceIntegrator<dim, dim, Number> FaceIntegratorU;

public:
  typedef Number value_type;

  ProjectionOperator(MatrixFree<dim, Number> const & matrix_free_in,
                     unsigned int const              dof_index_in,
                     unsigned int const              quad_index_in,
                     ProjectionOperatorData const    operator_data_in)
    : matrix_free(matrix_free_in),
      dof_index(dof_index_in),
      quad_index(quad_index_in),
      array_conti_penalty_parameter(0),
      array_div_penalty_parameter(0),
      time_step_size(1.0),
      operator_data(operator_data_in),
      scaling_factor_div(1.0),
      scaling_factor_conti(1.0),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
      block_diagonal_preconditioner_is_initialized(false)
  {
    unsigned int n_cells = matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();

    if(operator_data.use_divergence_penalty)
      array_div_penalty_parameter.resize(n_cells);

    if(operator_data.use_continuity_penalty)
      array_conti_penalty_parameter.resize(n_cells);

    if(operator_data.use_divergence_penalty)
      integrator.reset(new CellIntegratorU(this->get_matrix_free(),
                                           this->get_dof_index(),
                                           this->get_quad_index()));

    if(operator_data.use_continuity_penalty)
    {
      integrator_m.reset(new FaceIntegratorU(
        this->get_matrix_free(), true, this->get_dof_index(), this->get_quad_index()));
      integrator_p.reset(new FaceIntegratorU(
        this->get_matrix_free(), false, this->get_dof_index(), this->get_quad_index()));
    }
  }

  MatrixFree<dim, Number> const &
  get_matrix_free() const
  {
    return matrix_free;
  }

  AlignedVector<VectorizedArray<Number>> const &
  get_array_div_penalty_parameter() const
  {
    return array_div_penalty_parameter;
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
      calculate_div_penalty_parameter(velocity);
    if(operator_data.use_continuity_penalty)
      calculate_conti_penalty_parameter(velocity);
  }

  void
  calculate_div_penalty_parameter(VectorType const & velocity)
  {
    velocity.update_ghost_values();

    CellIntegratorU integrator(matrix_free, dof_index, quad_index);

    AlignedVector<scalar> JxW_values(integrator.n_q_points);

    unsigned int n_cells = matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
    for(unsigned int cell = 0; cell < n_cells; ++cell)
    {
      scalar tau_convective = make_vectorized_array<Number>(0.0);
      scalar tau_viscous    = make_vectorized_array<Number>(operator_data.viscosity);

      if(operator_data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm ||
         operator_data.type_penalty_parameter == TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        integrator.reinit(cell);
        integrator.read_dof_values(velocity);
        integrator.evaluate(true, false);

        scalar volume      = make_vectorized_array<Number>(0.0);
        scalar norm_U_mean = make_vectorized_array<Number>(0.0);
        JxW_values.resize(integrator.n_q_points);
        integrator.fill_JxW_values(JxW_values);
        for(unsigned int q = 0; q < integrator.n_q_points; ++q)
        {
          volume += JxW_values[q];
          norm_U_mean += JxW_values[q] * integrator.get_value(q).norm();
        }
        norm_U_mean /= volume;

        tau_convective = norm_U_mean * std::exp(std::log(volume) / (double)dim) /
                         (double)(operator_data.degree + 1);
      }

      if(operator_data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm)
      {
        array_div_penalty_parameter[cell] = operator_data.penalty_factor_div * tau_convective;
      }
      else if(operator_data.type_penalty_parameter == TypePenaltyParameter::ViscousTerm)
      {
        array_div_penalty_parameter[cell] = operator_data.penalty_factor_div * tau_viscous;
      }
      else if(operator_data.type_penalty_parameter ==
              TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        array_div_penalty_parameter[cell] =
          operator_data.penalty_factor_div * (tau_convective + tau_viscous);
      }
    }
  }

  void
  calculate_conti_penalty_parameter(VectorType const & velocity)
  {
    velocity.update_ghost_values();

    CellIntegratorU integrator(matrix_free, dof_index, quad_index);

    AlignedVector<scalar> JxW_values(integrator.n_q_points);

    unsigned int n_cells = matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
    for(unsigned int cell = 0; cell < n_cells; ++cell)
    {
      integrator.reinit(cell);
      integrator.read_dof_values(velocity);
      integrator.evaluate(true, false);
      scalar volume      = make_vectorized_array<Number>(0.0);
      scalar norm_U_mean = make_vectorized_array<Number>(0.0);
      JxW_values.resize(integrator.n_q_points);
      integrator.fill_JxW_values(JxW_values);
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        volume += JxW_values[q];
        norm_U_mean += JxW_values[q] * integrator.get_value(q).norm();
      }
      norm_U_mean /= volume;

      scalar tau_convective = norm_U_mean;
      scalar h = std::exp(std::log(volume) / (double)dim) / (double)(operator_data.degree + 1);
      scalar tau_viscous = make_vectorized_array<Number>(operator_data.viscosity) / h;

      if(operator_data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm)
      {
        array_conti_penalty_parameter[cell] = operator_data.penalty_factor_conti * tau_convective;
      }
      else if(operator_data.type_penalty_parameter == TypePenaltyParameter::ViscousTerm)
      {
        array_conti_penalty_parameter[cell] = operator_data.penalty_factor_conti * tau_viscous;
      }
      else if(operator_data.type_penalty_parameter ==
              TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        array_conti_penalty_parameter[cell] =
          operator_data.penalty_factor_conti * (tau_convective + tau_viscous);
      }
    }
  }

  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    apply(dst, src);
  }

  void
  apply(VectorType & dst, VectorType const & src) const
  {
    scaling_factor_div   = time_step_size;
    scaling_factor_conti = time_step_size;

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
    scaling_factor_div   = time_step_size;
    scaling_factor_conti = time_step_size;

    if(operator_data.use_divergence_penalty && operator_data.use_continuity_penalty)
      do_apply(dst, src, false);
    else if(operator_data.use_divergence_penalty && !operator_data.use_continuity_penalty)
      do_apply_mass_div_penalty(dst, src, false);
    else
      AssertThrow(false, ExcMessage("Not implemented."));
  }

  void
  apply_div_penalty(VectorType & dst, VectorType const & src) const
  {
    scaling_factor_div = 1.0;

    do_apply_div_penalty(dst, src, true);
  }

  void
  apply_add_div_penalty(VectorType & dst, VectorType const & src) const
  {
    scaling_factor_div = 1.0;

    do_apply_div_penalty(dst, src, false);
  }

  void
  apply_conti_penalty(VectorType & dst, VectorType const & src) const
  {
    scaling_factor_conti = 1.0;

    do_apply_conti_penalty(dst, src, true);
  }

  void
  apply_add_conti_penalty(VectorType & dst, VectorType const & src) const
  {
    scaling_factor_conti = 1.0;

    do_apply_conti_penalty(dst, src, false);
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

    scaling_factor_div   = time_step_size;
    scaling_factor_conti = time_step_size;

    if(operator_data.use_divergence_penalty)
    {
      integrator->reinit(cell);

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

        unsigned int dofs_per_cell = integrator_m->dofs_per_cell;

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator_m->begin_dof_values()[i] = src[i];

        // do not need to read dof values for integrator_p (already initialized with 0)

        integrator_m->evaluate(true, false);

        auto bids = matrix_free.get_faces_by_cells_boundary_id(cell, face);
        auto bid  = bids[0];

        if(bid == numbers::internal_face_boundary_id) // internal face
        {
          do_face_int_integral(*integrator_m, *integrator_p);
        }
        else // boundary face
        {
          // use same integrator so that the result becomes zero (only jumps involved)
          do_face_int_integral(*integrator_m, *integrator_m);
        }

        integrator_m->integrate(true, false);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          dst[i] += integrator_m->begin_dof_values()[i];
      }
    }
  }

private:
  void
  do_apply_div_penalty(VectorType & dst, VectorType const & src, bool const zero_dst_vector) const
  {
    matrix_free.cell_loop(&This::cell_loop_div_penalty, this, dst, src, zero_dst_vector);
  }

  void
  do_apply_mass_div_penalty(VectorType &       dst,
                            VectorType const & src,
                            bool const         zero_dst_vector) const
  {
    matrix_free.cell_loop(&This::cell_loop, this, dst, src, zero_dst_vector);
  }

  void
  do_apply_conti_penalty(VectorType & dst, VectorType const & src, bool const zero_dst_vector) const
  {
    matrix_free.loop(&This::cell_loop_empty,
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

  template<typename FEEval>
  void
  do_cell_integral(FEEval & integrator) const
  {
    scalar tau = integrator.read_cell_data(array_div_penalty_parameter) * scaling_factor_div;

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      integrator.submit_value(integrator.get_value(q), q);
      integrator.submit_divergence(tau * integrator.get_divergence(q), q);
    }
  }

  template<typename FEEval>
  void
  do_cell_integral_div_penalty(FEEval & integrator) const
  {
    scalar tau = integrator.read_cell_data(array_div_penalty_parameter) * scaling_factor_div;

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      integrator.submit_divergence(tau * integrator.get_divergence(q), q);
    }
  }

  template<typename FEFaceEval>
  void
  do_face_integral(FEFaceEval & integrator_m, FEFaceEval & integrator_p) const
  {
    scalar tau = 0.5 *
                 (integrator_m.read_cell_data(array_conti_penalty_parameter) +
                  integrator_p.read_cell_data(array_conti_penalty_parameter)) *
                 scaling_factor_conti;

    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      vector uM         = integrator_m.get_value(q);
      vector uP         = integrator_p.get_value(q);
      vector jump_value = uM - uP;

      if(operator_data.which_components == ContinuityPenaltyComponents::All)
      {
        // penalize all velocity components
        integrator_m.submit_value(tau * jump_value, q);
        integrator_p.submit_value(-tau * jump_value, q);
      }
      else if(operator_data.which_components == ContinuityPenaltyComponents::Normal)
      {
        // penalize normal components only
        vector normal = integrator_m.get_normal_vector(q);

        integrator_m.submit_value(tau * (jump_value * normal) * normal, q);
        integrator_p.submit_value(-tau * (jump_value * normal) * normal, q);
      }
      else
      {
        AssertThrow(operator_data.which_components == ContinuityPenaltyComponents::All ||
                      operator_data.which_components == ContinuityPenaltyComponents::Normal,
                    ExcMessage("not implemented."));
      }
    }
  }

  template<typename FEFaceEval>
  void
  do_face_int_integral(FEFaceEval & integrator_m, FEFaceEval & integrator_p) const
  {
    scalar tau = 0.5 *
                 (integrator_m.read_cell_data(array_conti_penalty_parameter) +
                  integrator_p.read_cell_data(array_conti_penalty_parameter)) *
                 scaling_factor_conti;

    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      vector uM = integrator_m.get_value(q);
      vector uP; // set uP to zero
      vector jump_value = uM - uP;

      if(operator_data.which_components == ContinuityPenaltyComponents::All)
      {
        // penalize all velocity components
        integrator_m.submit_value(tau * jump_value, q);
      }
      else if(operator_data.which_components == ContinuityPenaltyComponents::Normal)
      {
        // penalize normal components only
        vector normal = integrator_m.get_normal_vector(q);
        integrator_m.submit_value(tau * (jump_value * normal) * normal, q);
      }
      else
      {
        AssertThrow(operator_data.which_components == ContinuityPenaltyComponents::All ||
                      operator_data.which_components == ContinuityPenaltyComponents::Normal,
                    ExcMessage("not implemented."));
      }
    }
  }

  template<typename FEFaceEval>
  void
  do_face_ext_integral(FEFaceEval & integrator_m, FEFaceEval & integrator_p) const
  {
    scalar tau = 0.5 *
                 (integrator_m.read_cell_data(array_conti_penalty_parameter) +
                  integrator_p.read_cell_data(array_conti_penalty_parameter)) *
                 scaling_factor_conti;

    for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
    {
      vector uM; // set uM to zero
      vector uP = integrator_p.get_value(q);

      vector jump_value = uP - uM; // interior - exterior = uP - uM (neighbor!)

      if(operator_data.which_components == ContinuityPenaltyComponents::All)
      {
        // penalize all velocity components
        integrator_p.submit_value(tau * jump_value, q);
      }
      else if(operator_data.which_components == ContinuityPenaltyComponents::Normal)
      {
        // penalize normal components only
        vector normal = integrator_p.get_normal_vector(q);
        integrator_p.submit_value(tau * (jump_value * normal) * normal, q);
      }
      else
      {
        AssertThrow(operator_data.which_components == ContinuityPenaltyComponents::All ||
                      operator_data.which_components == ContinuityPenaltyComponents::Normal,
                    ExcMessage("not implemented."));
      }
    }
  }


  void
  cell_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const
  {
    CellIntegratorU integrator(data, dof_index, quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);
      integrator.gather_evaluate(src, true, true);

      do_cell_integral(integrator);

      integrator.integrate_scatter(true, true, dst);
    }
  }

  void
  cell_loop_div_penalty(MatrixFree<dim, Number> const & data,
                        VectorType &                    dst,
                        VectorType const &              src,
                        Range const &                   cell_range) const
  {
    CellIntegratorU integrator(data, dof_index, quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);
      integrator.gather_evaluate(src, false, true);

      do_cell_integral_div_penalty(integrator);

      integrator.integrate_scatter(false, true, dst);
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
    FaceIntegratorU integrator_m(data, true, dof_index, quad_index);
    FaceIntegratorU integrator_p(data, false, dof_index, quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_m.reinit(face);
      integrator_p.reinit(face);

      integrator_m.gather_evaluate(src, true, false);
      integrator_p.gather_evaluate(src, true, false);

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
    scaling_factor_div   = time_step_size;
    scaling_factor_conti = time_step_size;

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
    CellIntegratorU integrator(data, dof_index, quad_index);

    unsigned int const    dofs_per_cell = integrator.dofs_per_cell;
    AlignedVector<scalar> local_diagonal_vector(dofs_per_cell);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);

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
    FaceIntegratorU integrator_m(data, true, dof_index, quad_index);
    FaceIntegratorU integrator_p(data, false, dof_index, quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_m.reinit(face);
      integrator_p.reinit(face);

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

    if(this->operator_data.preconditioner_block_jacobi == PreconditionerBlockDiagonal::None)
    {
      typedef Elementwise::PreconditionerIdentity<VectorizedArray<Number>> IDENTITY;
      elementwise_preconditioner.reset(new IDENTITY(elementwise_operator->get_problem_size()));
    }
    else if(this->operator_data.preconditioner_block_jacobi ==
            PreconditionerBlockDiagonal::InverseMassMatrix)
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
    iterative_solver_data.solver_type = Elementwise::SolverType::CG;
    iterative_solver_data.solver_data = this->operator_data.block_jacobi_solver_data;

    elementwise_solver.reset(new ELEMENTWISE_SOLVER(
      *std::dynamic_pointer_cast<ELEMENTWISE_OPERATOR>(elementwise_operator),
      *std::dynamic_pointer_cast<PRECONDITIONER_BASE>(elementwise_preconditioner),
      iterative_solver_data));
  }

  void
  add_block_diagonal_matrices(std::vector<LAPACKFullMatrix<Number>> & matrices) const
  {
    scaling_factor_div   = time_step_size;
    scaling_factor_conti = time_step_size;

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
    CellIntegratorU integrator(data, dof_index, quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);

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
    FaceIntegratorU integrator_m(data, true, dof_index, quad_index);
    FaceIntegratorU integrator_p(data, false, dof_index, quad_index);

    // Perform face integrals for element e⁻.
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_m.reinit(face);
      integrator_p.reinit(face);

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
    CellIntegratorU integrator(data, dof_index, quad_index);
    FaceIntegratorU integrator_m(data, true, dof_index, quad_index);
    FaceIntegratorU integrator_p(data, false, dof_index, quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // cell integral
      unsigned int const n_filled_lanes = data.n_active_entries_per_cell_batch(cell);

      integrator.reinit(cell);

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
            // use same integrator so that the result becomes zero (only jumps involved)
            do_face_int_integral(integrator_m, integrator_m);
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
    CellIntegratorU integrator(data, dof_index, quad_index);

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

  AlignedVector<scalar> array_conti_penalty_parameter;
  AlignedVector<scalar> array_div_penalty_parameter;

  double time_step_size;

  ProjectionOperatorData operator_data;

  // Scaling factors for divergence and continuity penalty term:
  // Normally, the scaling factor equals the time step size when applying the combined operator
  // consisting of mass, divergence penalty and continuity penalty operators. In case that the
  // divergence and continuity penalty terms are applied separately (e.g. coupled solution approach,
  // penalty terms added to monolithic system), these scaling factors have to be set to a value
  // of 1.
  mutable double scaling_factor_div, scaling_factor_conti;

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
  std::shared_ptr<CellIntegratorU> integrator;
  std::shared_ptr<FaceIntegratorU> integrator_m;
  std::shared_ptr<FaceIntegratorU> integrator_p;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_OPERATOR_H_ \
        */
