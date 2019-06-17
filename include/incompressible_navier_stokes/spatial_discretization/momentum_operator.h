/*
 * velocity_conv_diff_operator.h
 *
 *  Created on: Aug 8, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_MOMENTUM_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_MOMENTUM_OPERATOR_H_

#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>

#include "operators/convective_operator.h"
#include "operators/mass_matrix_operator.h"
#include "operators/viscous_operator.h"

#include "../../operators/elementwise_operator.h"
#include "../../operators/linear_operator_base.h"
#include "../../operators/operator_base.h"
#include "../../solvers_and_preconditioners/util/invert_diagonal.h"
#include "../../solvers_and_preconditioners/util/verify_calculation_of_diagonal.h"

#include "solvers_and_preconditioners/preconditioner/elementwise_preconditioners.h"
#include "solvers_and_preconditioners/solvers/wrapper_elementwise_solvers.h"

namespace IncNS
{
template<int dim>
struct MomentumOperatorData
{
  MomentumOperatorData()
    : unsteady_problem(true),
      convective_problem(true),
      dof_index(0),
      quad_index_std(0),
      quad_index_over(1),
      scaling_factor_time_derivative_term(-1.0),
      implement_block_diagonal_preconditioner_matrix_free(false),
      use_cell_based_loops(false),
      preconditioner_block_jacobi(Elementwise::Preconditioner::InverseMassMatrix),
      block_jacobi_solver_data(SolverData(1000, 1.e-12, 1.e-2 /*rel_tol TODO*/, 1000)),
      mg_operator_type(MultigridOperatorType::Undefined)
  {
  }

  bool unsteady_problem;
  bool convective_problem;

  unsigned int dof_index;

  unsigned int quad_index_std;
  unsigned int quad_index_over;

  double scaling_factor_time_derivative_term;

  MassMatrixOperatorData      mass_matrix_operator_data;
  ViscousOperatorData<dim>    viscous_operator_data;
  ConvectiveOperatorData<dim> convective_operator_data;

  // block diagonal preconditioner
  bool implement_block_diagonal_preconditioner_matrix_free;

  // use cell based loops
  bool use_cell_based_loops;

  // elementwise iterative solution of block Jacobi problems
  Elementwise::Preconditioner preconditioner_block_jacobi;
  SolverData                  block_jacobi_solver_data;

  // Multigrid
  MultigridOperatorType mg_operator_type;
};

template<int dim, typename Number = double>
class MomentumOperator : public LinearOperatorBase
{
public:
  typedef MomentumOperator<dim, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  static const int DIM = dim;
  typedef Number   value_type;

  typedef CellIntegrator<dim, dim, Number> Integrator;

  MomentumOperator();

  virtual ~MomentumOperator()
  {
  }

  void
  reinit_multigrid(MatrixFree<dim, Number> const &   data,
                   AffineConstraints<double> const & constraint_matrix,
                   MomentumOperatorData<dim> const & operator_data);

  void
  reinit(MatrixFree<dim, Number> const &         data,
         MomentumOperatorData<dim> const &       operator_data,
         MassMatrixOperator<dim, Number> const & mass_matrix_operator,
         ViscousOperator<dim, Number> const &    viscous_operator,
         ConvectiveOperator<dim, Number> const & convective_operator);

  void
  vmult_interface_down(VectorType & dst, VectorType const & src) const
  {
    vmult(dst, src);
  }

  void
  vmult_add_interface_up(VectorType & dst, VectorType const & src) const
  {
    vmult_add(dst, src);
  }

  types::global_dof_index
  m() const
  {
    return n();
  }

  types::global_dof_index
  n() const
  {
    MatrixFree<dim, Number> const & data      = get_matrix_free();
    unsigned int                    dof_index = get_dof_index();

    return data.get_vector_partitioner(dof_index)->size();
  }

  Number
  el(const unsigned int, const unsigned int) const
  {
    AssertThrow(false, ExcMessage("Matrix-free does not allow for entry access"));
    return Number();
  }

  bool
  is_empty_locally() const
  {
    MatrixFree<dim, Number> const & data = get_matrix_free();
    return (data.n_macro_cells() == 0);
  }

  void
  initialize_dof_vector(VectorType & vector) const
  {
    MatrixFree<dim, Number> const & data      = get_matrix_free();
    unsigned int                    dof_index = get_dof_index();

    data.initialize_dof_vector(vector, dof_index);
  }

  virtual AffineConstraints<double> const &
  get_constraint_matrix() const
  {
    AssertThrow(false,
                ExcMessage("MomentumOperator::get_constraint_matrix should be overwritten!"));
    return *(new AffineConstraints<double>());
  }

  virtual bool
  is_singular() const
  {
    // per default the operator is not singular
    // if an operator can be singular, this method has to be overwritten
    return false;
  }

#ifdef DEAL_II_WITH_TRILINOS
  virtual void
  init_system_matrix(TrilinosWrappers::SparseMatrix & /*system_matrix*/) const
  {
    AssertThrow(false, ExcMessage("MomentumOperator::init_system_matrix should be overwritten!"));
  }

  virtual void
  calculate_system_matrix(TrilinosWrappers::SparseMatrix & /*system_matrix*/) const
  {
    AssertThrow(false,
                ExcMessage("MomentumOperator::calculate_system_matrix should be overwritten!"));
  }
#endif

  /*
   * Setters and getters.
   */

  MatrixFree<dim, Number> const &
  get_matrix_free() const;

  unsigned int
  get_dof_index() const;

  unsigned int
  get_quad_index() const;

  /*
   *  Scaling factor of time derivative term (mass matrix term)
   */
  void
  set_scaling_factor_time_derivative_term(double const & factor);

  double
  get_scaling_factor_time_derivative_term() const;

  /*
   *  Linearized velocity field for convective operator
   */
  void
  set_solution_linearization(VectorType const & solution_linearization);

  VectorType const &
  get_solution_linearization() const;

  /*
   *  Evaluation time that is needed for evaluation of linearized convective operator.
   */
  void
  set_evaluation_time(double const time);

  double
  get_evaluation_time() const;

  /*
   *  Operator data
   */
  MomentumOperatorData<dim> const &
  get_operator_data() const;

  /*
   *  Operator data of basic operators: mass matrix, convective operator, viscous operator
   */
  MassMatrixOperatorData const &
  get_mass_matrix_operator_data() const;

  ConvectiveOperatorData<dim> const &
  get_convective_operator_data() const;

  ViscousOperatorData<dim> const &
  get_viscous_operator_data() const;

  /*
   *  This function applies the matrix vector multiplication.
   */
  void
  vmult(VectorType & dst, VectorType const & src) const;

  /*
   *  This function applies matrix vector product and adds the result
   *  to the dst-vector.
   */
  void
  vmult_add(VectorType & dst, VectorType const & src) const;


  // TODO no longer needed -> block Jacobi implementation is shifted to base class OperatorBase
  //  /*
  //   *  This function applies the matrix-vector multiplication for the block Jacobi operation.
  //   */
  //  void
  //  vmult_block_jacobi(VectorType & dst, VectorType const & src) const;

  /*
   *  Calculation of inverse diagonal (needed for smoothers and preconditioners)
   */
  void
  calculate_inverse_diagonal(VectorType & diagonal) const;

  /*
   *  Apply block Jacobi preconditioner.
   */
  void
  apply_inverse_block_diagonal(VectorType & dst, VectorType const & src) const;


  /*
   *  This function updates the block Jacobi preconditioner.
   *  Since this function also initializes the block Jacobi preconditioner,
   *  make sure that the block Jacobi matrices are allocated before calculating
   *  the matrices and the LU factorization.
   */
  void
  update_block_diagonal_preconditioner() const;

  void
  apply_add_block_diagonal_elementwise(unsigned int const                    cell,
                                       VectorizedArray<Number> * const       dst,
                                       VectorizedArray<Number> const * const src,
                                       unsigned int const problem_size = 1) const;

private:
  /*
   *  This function calculates the diagonal of the discrete operator representing the
   *  velocity convection-diffusion operator.
   */
  void
  calculate_diagonal(VectorType & diagonal) const;

  /*
   * This function calculates the block Jacobi matrices.
   * This is done sequentially for the different operators.
   */
  void
  add_block_diagonal_matrices(std::vector<LAPACKFullMatrix<value_type>> & matrices) const;

  /*
   * Apply inverse block diagonal:
   *
   * instead of applying the block matrix B we compute dst = B^{-1} * src (LU factorization
   * should have already been performed with the method update_inverse_block_diagonal())
   */
  void
  cell_loop_apply_inverse_block_diagonal(
    MatrixFree<dim, Number> const &               data,
    VectorType &                                  dst,
    VectorType const &                            src,
    std::pair<unsigned int, unsigned int> const & cell_range) const;

  void
  initialize_block_diagonal_preconditioner_matrix_free() const;

  // TODO no longer needed -> block Jacobi implementation is shifted to base class OperatorBase
  //  /*
  //   * Verify computation of block Jacobi matrices.
  //   */
  //  void
  //  check_block_jacobi_matrices() const;

  /*
   *  This function is only needed for testing.
   */
  void
  cell_loop_apply_block_diagonal(MatrixFree<dim, Number> const &               data,
                                 VectorType &                                  dst,
                                 VectorType const &                            src,
                                 std::pair<unsigned int, unsigned int> const & cell_range) const;

  MomentumOperatorData<dim> operator_data;

  MatrixFree<dim, Number> const * matrix_free;

  MassMatrixOperator<dim, Number> const * mass_matrix_operator;

  ViscousOperator<dim, Number> const * viscous_operator;

  ConvectiveOperator<dim, Number> const * convective_operator;

  /*
   * The following variables are necessary when applying the multigrid
   * preconditioner to the velocity convection-diffusion operator.
   * In that case, the VelocityConvDiffOperator has to be generated
   * for each level of the multigrid algorithm.
   * Accordingly, in a first step one has to setup own objects of
   * MassMatrixOperator, ViscousOperator,
   *   e.g., own_mass_matrix_operator_storage.reinit(...);
   * and later initialize the VelocityConvDiffOperator with these
   * ojects by setting the above pointers to the own_objects_storage,
   *   e.g., data = &own_mass_matrix_operator_storage;
   */
  MassMatrixOperator<dim, Number> own_mass_matrix_operator_storage;

  ViscousOperator<dim, Number> own_viscous_operator_storage;

  ConvectiveOperator<dim, Number> own_convective_operator_storage;

  VectorType mutable temp_vector;
  VectorType mutable velocity_linearization;

  double evaluation_time;
  double scaling_factor_time_derivative_term;

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
};

struct MomentumOperatorMergedData : public OperatorBaseData
{
  MomentumOperatorMergedData()
    : OperatorBaseData(0 /* dof_index */, 0 /* quad_index */),
      unsteady_problem(false),
      convective_problem(false),
      viscous_problem(false)
  {
  }

  bool unsteady_problem;
  bool convective_problem;
  bool viscous_problem;

  Operators::ConvectiveKernelData convective_kernel_data;
};

template<int dim, typename Number>
class MomentumOperatorMerged : public OperatorBase<dim, Number, MomentumOperatorMergedData, dim>
{
private:
  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;
  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  typedef OperatorBase<dim, Number, MomentumOperatorMergedData, dim> Base;

  typedef typename Base::VectorType     VectorType;
  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

public:
  void
  rhs(VectorType & dst) const
  {
    (void)dst;

    AssertThrow(false,
                ExcMessage("The function rhs() does not make sense for the momentum operator."));
  }

  void
  rhs_add(VectorType & dst) const
  {
    (void)dst;

    AssertThrow(
      false, ExcMessage("The function rhs_add() does not make sense for the momentum operator."));
  }

  void
  evaluate(VectorType & dst, VectorType const & src) const
  {
    (void)dst;
    (void)src;

    AssertThrow(
      false, ExcMessage("The function evaluate() does not make sense for the momentum operator."));
  }

  void
  evaluate_add(VectorType & dst, VectorType const & src) const
  {
    (void)dst;
    (void)src;

    AssertThrow(false,
                ExcMessage(
                  "The function evaluate_add() does not make sense for the momentum operator."));
  }

  void
  reinit_cell(unsigned int const cell) const
  {
    Base::reinit_cell(cell);

    if(this->operator_data.convective_problem)
      convective_kernel.reinit_cell(cell);
  }

  void
  reinit_face(unsigned int const face) const
  {
    Base::reinit_face(face);

    if(this->operator_data.convective_problem)
      convective_kernel.reinit_face(face);

    if(this->operator_data.viscous_problem)
      viscous_kernel.reinit_face(*this->integrator_m, *this->integrator_p);
  }

  void
  reinit_boundary_face(unsigned int const face) const
  {
    Base::reinit_boundary_face(face);

    if(this->operator_data.convective_problem)
      convective_kernel.reinit_boundary_face(face);

    if(this->operator_data.viscous_problem)
      viscous_kernel.reinit_boundary_face(*this->integrator_m);
  }

  void
  reinit_face_cell_based(unsigned int const       cell,
                         unsigned int const       face,
                         types::boundary_id const boundary_id) const
  {
    Base::reinit_face_cell_based(cell, face, boundary_id);

    if(this->operator_data.convective_problem)
      convective_kernel.reinit_face_cell_based(cell, face, boundary_id);

    if(this->operator_data.viscous_problem)
      viscous_kernel.reinit_face_cell_based(boundary_id, *this->integrator_m, *this->integrator_p);
  }

  // linearized operator
  void
  do_cell_integral(IntegratorCell & integrator) const
  {
    bool const get_value =
      this->operator_data.unsteady_problem || this->operator_data.convective_problem;

    bool const get_gradient = this->operator_data.viscous_problem ||
                              (this->operator_data.convective_problem &&
                               this->operator_data.convective_kernel_data.formulation ==
                                 FormulationConvectiveTerm::ConvectiveFormulation);

    bool const submit_value = this->operator_data.unsteady_problem ||
                              (this->operator_data.convective_problem &&
                               this->operator_data.convective_kernel_data.formulation ==
                                 FormulationConvectiveTerm::ConvectiveFormulation);

    bool const submit_gradient = this->operator_data.viscous_problem ||
                                 (this->operator_data.convective_problem &&
                                  this->operator_data.convective_kernel_data.formulation ==
                                    FormulationConvectiveTerm::DivergenceFormulation);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      vector value_flux;
      tensor gradient_flux;

      vector value;
      if(get_value)
        value = integrator.get_value(q);

      tensor gradient;
      if(get_gradient)
        gradient = integrator.get_gradient(q);

      if(this->operator_data.unsteady_problem)
        value_flux += mass_kernel.get_volume_flux(value);

      if(this->operator_data.convective_problem)
      {
        vector u = convective_kernel.get_velocity_cell(q);

        if(this->operator_data.convective_kernel_data.formulation ==
           FormulationConvectiveTerm::DivergenceFormulation)
        {
          gradient_flux +=
            convective_kernel.get_volume_flux_linearized_divergence_formulation(u, value);
        }
        else if(this->operator_data.convective_kernel_data.formulation ==
                FormulationConvectiveTerm::ConvectiveFormulation)
        {
          tensor grad_u = convective_kernel.get_velocity_gradient_cell(q);

          value_flux += convective_kernel.get_volume_flux_linearized_convective_formulation(
            u, value, grad_u, gradient);
        }
        else
        {
          AssertThrow(false, ExcMessage("Not implemented."));
        }
      }

      if(this->operator_data.viscous_problem)
      {
        scalar viscosity = viscous_kernel.get_viscosity_cell(integrator.get_cell_index(), q);
        gradient_flux += viscous_kernel.get_volume_flux(gradient, viscosity);
      }

      if(submit_value)
        integrator.submit_value(value_flux, q);

      if(submit_gradient)
        integrator.submit_gradient(gradient_flux, q);
    }
  }

  // linearized operator
  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      vector value_m  = integrator_m.get_value(q);
      vector value_p  = integrator_p.get_value(q);
      vector normal_m = integrator_m.get_normal_vector(q);

      vector value_flux_m, value_flux_p;
      tensor gradient_flux;

      if(this->operator_data.convective_problem)
      {
        vector u_m = convective_kernel.get_velocity_m(q);
        vector u_p = convective_kernel.get_velocity_p(q);

        std::tuple<vector, vector> flux =
          convective_kernel.calculate_flux_linearized_interior_and_neighbor(
            u_m, u_p, value_m, value_p, normal_m);

        value_flux_m += std::get<0>(flux);
        value_flux_p += std::get<1>(flux);
      }

      if(this->operator_data.viscous_problem)
      {
        scalar average_viscosity =
          viscous_kernel.get_viscosity_interior_face(integrator_m.get_face_index(), q);

        gradient_flux =
          viscous_kernel.calculate_gradient_flux(value_m, value_p, normal_m, average_viscosity);

        vector normal_gradient_m = viscous_kernel.calculate_normal_gradient(q, integrator_m);
        vector normal_gradient_p = viscous_kernel.calculate_normal_gradient(q, integrator_p);

        vector value_flux = viscous_kernel.calculate_value_flux(
          normal_gradient_m, normal_gradient_p, value_m, value_p, normal_m, average_viscosity);

        value_flux_m += -value_flux;
        value_flux_p += value_flux; // + sign since n⁺ = -n⁻
      }

      integrator_m.submit_value(value_flux_m, q);
      integrator_p.submit_value(value_flux_p, q);

      if(this->operator_data.viscous_problem)
      {
        integrator_m.submit_gradient(gradient_flux, q);
        integrator_p.submit_gradient(gradient_flux, q);
      }
    }
  }

  // linearized operator
  void
  do_face_int_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    (void)integrator_p;

    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      vector value_m = integrator_m.get_value(q);
      vector value_p; // set exterior value to zero
      vector normal_m = integrator_m.get_normal_vector(q);

      vector value_flux_m;
      tensor gradient_flux;

      if(this->operator_data.convective_problem)
      {
        vector u_m = convective_kernel.get_velocity_m(q);
        vector u_p = convective_kernel.get_velocity_p(q);

        value_flux_m += convective_kernel.calculate_flux_linearized_interior(
          u_m, u_p, value_m, value_p, normal_m);
      }

      if(this->operator_data.viscous_problem)
      {
        scalar average_viscosity =
          viscous_kernel.get_viscosity_interior_face(integrator_m.get_face_index(), q);

        gradient_flux +=
          viscous_kernel.calculate_gradient_flux(value_m, value_p, normal_m, average_viscosity);

        vector normal_gradient_m = viscous_kernel.calculate_normal_gradient(q, integrator_m);
        vector normal_gradient_p; // set exterior gradient to zero

        vector value_flux = viscous_kernel.calculate_value_flux(
          normal_gradient_m, normal_gradient_p, value_m, value_p, normal_m, average_viscosity);

        value_flux_m += -value_flux;
      }

      integrator_m.submit_value(value_flux_m, q);

      if(this->operator_data.viscous_problem)
      {
        integrator_m.submit_gradient(gradient_flux, q);
      }
    }
  }

  // linearized operator

  // TODO
  // This function is currently only needed due to limitations of deal.II which do
  // currently not allow to access neighboring data in case of cell-based face loops.
  // Once this functionality is available, this function should be removed again.
  void
  do_face_int_integral_cell_based(IntegratorFace & integrator_m,
                                  IntegratorFace & integrator_p) const
  {
    (void)integrator_p;

    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      vector value_m = integrator_m.get_value(q);
      vector value_p; // set exterior value to zero
      vector normal_m = integrator_m.get_normal_vector(q);

      vector value_flux_m;
      tensor gradient_flux;

      if(this->operator_data.convective_problem)
      {
        vector u_m = convective_kernel.get_velocity_m(q);
        // TODO
        // Accessing exterior data is currently not available in deal.II/matrixfree.
        // Hence, we simply use the interior value, but note that the diagonal and block-diagonal
        // are not calculated exactly.
        vector u_p = u_m;

        value_flux_m += convective_kernel.calculate_flux_linearized_interior(
          u_m, u_p, value_m, value_p, normal_m);
      }

      if(this->operator_data.viscous_problem)
      {
        scalar average_viscosity =
          viscous_kernel.get_viscosity_interior_face(integrator_m.get_face_index(), q);

        gradient_flux +=
          viscous_kernel.calculate_gradient_flux(value_m, value_p, normal_m, average_viscosity);

        vector normal_gradient_m = viscous_kernel.calculate_normal_gradient(q, integrator_m);
        vector normal_gradient_p; // set exterior gradient to zero

        vector value_flux = viscous_kernel.calculate_value_flux(
          normal_gradient_m, normal_gradient_p, value_m, value_p, normal_m, average_viscosity);

        value_flux_m += -value_flux;
      }

      integrator_m.submit_value(value_flux_m, q);

      if(this->operator_data.viscous_problem)
      {
        integrator_m.submit_gradient(gradient_flux, q);
      }
    }
  }

  // linearized operator
  void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    (void)integrator_m;

    for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
    {
      vector value_m; // set exterior values to zero
      vector value_p = integrator_p.get_value(q);
      // multiply by -1.0 to get the correct normal vector !!!
      vector normal_p = -integrator_p.get_normal_vector(q);

      vector value_flux_p;
      tensor gradient_flux;

      if(this->operator_data.convective_problem)
      {
        vector u_m = convective_kernel.get_velocity_m(q);
        vector u_p = convective_kernel.get_velocity_p(q);

        value_flux_p += convective_kernel.calculate_flux_linearized_interior(
          u_p, u_m, value_p, value_m, normal_p);
      }

      if(this->operator_data.viscous_problem)
      {
        scalar average_viscosity =
          viscous_kernel.get_viscosity_interior_face(integrator_p.get_face_index(), q);

        gradient_flux +=
          viscous_kernel.calculate_gradient_flux(value_p, value_m, normal_p, average_viscosity);

        // set exterior gradient to zero
        vector normal_gradient_m;
        // multiply by -1.0 since normal vector n⁺ = -n⁻ !!!
        vector normal_gradient_p = -viscous_kernel.calculate_normal_gradient(q, integrator_p);

        vector value_flux = viscous_kernel.calculate_value_flux(
          normal_gradient_p, normal_gradient_m, value_p, value_m, normal_p, average_viscosity);

        value_flux_p += -value_flux;
      }

      integrator_p.submit_value(value_flux_p, q);

      if(this->operator_data.viscous_problem)
      {
        integrator_p.submit_gradient(gradient_flux, q);
      }
    }
  }

  // linearized operator
  void
  do_boundary_integral(IntegratorFace &           integrator,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id) const
  {
    // make sure that this function is only accessed for OperatorType::homogeneous
    AssertThrow(
      operator_type == OperatorType::homogeneous,
      ExcMessage(
        "For the linearized momentum operator, only OperatorType::homogeneous makes sense."));

    BoundaryTypeU boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      vector value_m = integrator.get_value(q);
      vector value_p;
      vector normal_m = integrator.get_normal_vector(q);

      vector value_flux_m;
      tensor gradient_flux;

      if(this->operator_data.convective_problem)
      {
        // value_p is calculated differently for the convective term and the viscous term
        value_p = convective_kernel.calculate_exterior_value_linearized(value_m,
                                                                        q,
                                                                        integrator,
                                                                        boundary_type);

        vector u_m = convective_kernel.get_velocity_m(q);
        vector u_p = convective_kernel.calculate_exterior_value_nonlinear(
          u_m, q, integrator, boundary_type, boundary_id, this->operator_data.bc, this->eval_time);

        value_flux_m += convective_kernel.calculate_flux_linearized_boundary(
          u_m, u_p, value_m, value_p, normal_m, boundary_type);
      }

      if(this->operator_data.viscous_problem)
      {
        // value_p is calculated differently for the convective term and the viscous term
        value_p = calculate_exterior_value(value_m,
                                           q,
                                           integrator,
                                           operator_type,
                                           boundary_type,
                                           boundary_id,
                                           this->operator_data.bc,
                                           this->eval_time);

        scalar viscosity =
          viscous_kernel.get_viscosity_boundary_face(integrator.get_face_index(), q);
        gradient_flux +=
          viscous_kernel.calculate_gradient_flux(value_m, value_p, normal_m, viscosity);

        vector normal_gradient_m =
          viscous_kernel.calculate_interior_normal_gradient(q, integrator, operator_type);
        vector normal_gradient_p = calculate_exterior_normal_gradient(normal_gradient_m,
                                                                      q,
                                                                      integrator,
                                                                      operator_type,
                                                                      boundary_type,
                                                                      boundary_id,
                                                                      this->operator_data.bc,
                                                                      this->eval_time);

        vector value_flux = viscous_kernel.calculate_value_flux(
          normal_gradient_m, normal_gradient_p, value_m, value_p, normal_m, viscosity);

        value_flux_m += -value_flux;
      }

      integrator.submit_value(value_flux_m, q);

      if(this->operator_data.viscous_problem)
      {
        integrator.submit_gradient(gradient_flux, q);
      }
    }
  }


private:
  Operators::MassMatrixKernel<dim, Number> mass_kernel;
  Operators::ConvectiveKernel<dim, Number> convective_kernel;
  Operators::ViscousKernel<dim, Number>    viscous_kernel;
};



} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_MOMENTUM_OPERATOR_H_ \
        */
