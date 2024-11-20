/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// deal.II
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix_tools.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/matrix_free/tools.h>
#include <deal.II/multigrid/mg_tools.h>

// ExaDG
#include <exadg/operators/operator_base.h>
#include <exadg/solvers_and_preconditioners/utilities/block_jacobi_matrices.h>
#include <exadg/solvers_and_preconditioners/utilities/invert_diagonal.h>
#include <exadg/solvers_and_preconditioners/utilities/linear_algebra_utilities.h>
#include <exadg/solvers_and_preconditioners/utilities/verify_calculation_of_diagonal.h>

namespace ExaDG
{
template<int dim, typename Number, int n_components>
OperatorBase<dim, Number, n_components>::OperatorBase()
  : dealii::Subscriptor(),
    matrix_free(),
    time(0.0),
    is_mg(false),
    is_dg(true),
    data(OperatorBaseData()),
    level(dealii::numbers::invalid_unsigned_int),
    n_mpi_processes(0),
    system_matrix_based_been_initialized(false)
{
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::reinit(
  dealii::MatrixFree<dim, Number> const &   matrix_free,
  dealii::AffineConstraints<Number> const & constraints,
  OperatorBaseData const &                  data)
{
  // reinit data structures
  this->matrix_free.reset(matrix_free);
  this->constraint.reset(constraints);
  this->constraint_double.copy_from(*constraint);
  this->data = data;

  // check if DG or CG
  // An approximation can have degrees of freedom on vertices, edges, quads and
  // hexes. A vertex degree of freedom means that the degree of freedom is
  // the same on all cells that are adjacent to this vertex. A face degree of
  // freedom means that the two cells adjacent to that face see the same degree
  // of freedom. A DG element does not share any degrees of freedom over a
  // vertex but has all of them in the last item, i.e., quads in 2D and hexes
  // in 3D, and thus necessarily has dofs_per_vertex=0
  is_dg = (this->matrix_free->get_dof_handler(this->data.dof_index).get_fe().dofs_per_vertex == 0);

  // set multigrid level
  this->level = this->matrix_free->get_mg_level();

  // The default value is is_mg = false and this variable is set to true in case
  // the operator is applied in multigrid algorithm. By convention, the default
  // argument dealii::numbers::invalid_unsigned_int corresponds to the default
  // value is_mg = false
  this->is_mg = (this->level != dealii::numbers::invalid_unsigned_int);

  // initialize n_mpi_proceses
  dealii::DoFHandler<dim> const & dof_handler =
    this->matrix_free->get_dof_handler(this->data.dof_index);

  n_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(dof_handler.get_communicator());
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::set_time(double const t) const
{
  this->time = t;
}

template<int dim, typename Number, int n_components>
double
OperatorBase<dim, Number, n_components>::get_time() const
{
  return time;
}

template<int dim, typename Number, int n_components>
unsigned int
OperatorBase<dim, Number, n_components>::get_level() const
{
  return level;
}

template<int dim, typename Number, int n_components>
dealii::AffineConstraints<Number> const &
OperatorBase<dim, Number, n_components>::get_affine_constraints() const
{
  return *constraint;
}

template<int dim, typename Number, int n_components>
dealii::MatrixFree<dim, Number> const &
OperatorBase<dim, Number, n_components>::get_matrix_free() const
{
  return *this->matrix_free;
}

template<int dim, typename Number, int n_components>
unsigned int
OperatorBase<dim, Number, n_components>::get_dof_index() const
{
  return this->data.dof_index;
}

template<int dim, typename Number, int n_components>
unsigned int
OperatorBase<dim, Number, n_components>::get_quad_index() const
{
  return this->data.quad_index;
}

template<int dim, typename Number, int n_components>
bool
OperatorBase<dim, Number, n_components>::operator_is_singular() const
{
  return this->data.operator_is_singular;
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::vmult(VectorType & dst, VectorType const & src) const
{
  if(this->data.use_matrix_based_vmult)
    this->apply_matrix_based(dst, src);
  else
    this->apply(dst, src);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::vmult_add(VectorType & dst, VectorType const & src) const
{
  if(this->data.use_matrix_based_vmult)
    this->apply_matrix_based_add(dst, src);
  else
    this->apply_add(dst, src);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::vmult_interface_down(VectorType &       dst,
                                                              VectorType const & src) const
{
  vmult(dst, src);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::vmult_add_interface_up(VectorType &       dst,
                                                                VectorType const & src) const
{
  vmult_add(dst, src);
}

template<int dim, typename Number, int n_components>
dealii::types::global_dof_index
OperatorBase<dim, Number, n_components>::m() const
{
  return n();
}

template<int dim, typename Number, int n_components>
dealii::types::global_dof_index
OperatorBase<dim, Number, n_components>::n() const
{
  unsigned int dof_index = get_dof_index();

  return this->matrix_free->get_vector_partitioner(dof_index)->size();
}

template<int dim, typename Number, int n_components>
Number
OperatorBase<dim, Number, n_components>::el(unsigned int const, unsigned int const) const
{
  AssertThrow(false, dealii::ExcMessage("Matrix-free does not allow for entry access"));
  return Number();
}

template<int dim, typename Number, int n_components>
bool
OperatorBase<dim, Number, n_components>::is_empty_locally() const
{
  return (this->matrix_free->n_cell_batches() == 0);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::initialize_dof_vector(VectorType & vector) const
{
  this->matrix_free->initialize_dof_vector(vector, this->data.dof_index);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::set_constrained_dofs_to_zero(VectorType & vector) const
{
  for(unsigned int const constrained_index :
      matrix_free->get_constrained_dofs(this->data.dof_index))
  {
    vector.local_element(constrained_index) = 0.0;
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::calculate_inverse_diagonal(VectorType & diagonal) const
{
  this->calculate_diagonal(diagonal);

  if(false)
  {
    verify_calculation_of_diagonal(
      *this, diagonal, matrix_free->get_dof_handler(this->data.dof_index).get_communicator());
  }

  invert_diagonal(diagonal);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::apply(VectorType & dst, VectorType const & src) const
{
  if(is_dg)
  {
    if(evaluate_face_integrals())
    {
      matrix_free->loop(&This::cell_loop,
                        &This::face_loop,
                        &This::boundary_face_loop_hom_operator,
                        this,
                        dst,
                        src,
                        true);
    }
    else
    {
      matrix_free->cell_loop(&This::cell_loop, this, dst, src, true);
    }
  }
  else
  {
    // Compute matrix-vector product. Constrained degrees of freedom in the src-vector will not be
    // used. The function read_dof_values() (or gather_evaluate()) uses the homogeneous boundary
    // data passed to MatrixFree via AffineConstraints with the standard "dof_index".
    matrix_free->cell_loop(&This::cell_loop, this, dst, src, true);

    // Constrained degree of freedom are not removed from the system of equations.
    // Instead, we set the diagonal entries of the matrix to 1 for these constrained
    // degrees of freedom. This means that we simply copy the constrained values to the
    // dst vector.
    for(unsigned int const constrained_index :
        matrix_free->get_constrained_dofs(this->data.dof_index))
    {
      dst.local_element(constrained_index) = src.local_element(constrained_index);
    }
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::apply_add(VectorType & dst, VectorType const & src) const
{
  if(is_dg)
  {
    if(evaluate_face_integrals())
    {
      matrix_free->loop(
        &This::cell_loop, &This::face_loop, &This::boundary_face_loop_hom_operator, this, dst, src);
    }
    else
    {
      matrix_free->cell_loop(&This::cell_loop, this, dst, src);
    }
  }
  else
  {
    // See function apply() for additional comments.
    // Note that MatrixFree will not touch constrained degrees of freedom in the dst-vector.
    matrix_free->cell_loop(&This::cell_loop, this, dst, src);

    for(unsigned int const constrained_index :
        matrix_free->get_constrained_dofs(this->data.dof_index))
    {
      dst.local_element(constrained_index) += src.local_element(constrained_index);
    }
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::assemble_matrix_if_necessary() const
{
  if(this->data.use_matrix_based_vmult)
  {
    // initialize matrix
    if(not(system_matrix_based_been_initialized))
    {
      dealii::DoFHandler<dim> const & dof_handler =
        this->matrix_free->get_dof_handler(this->data.dof_index);

      if(this->data.sparse_matrix_type == SparseMatrixType::Trilinos)
      {
#ifdef DEAL_II_WITH_TRILINOS
        init_system_matrix(system_matrix_trilinos, dof_handler.get_communicator());
#else
        AssertThrow(
          false,
          dealii::ExcMessage(
            "Make sure that DEAL_II_WITH_TRILINOS is activated if you want to use SparseMatrixType::Trilinos."));
#endif
      }
      else if(this->data.sparse_matrix_type == SparseMatrixType::PETSc)
      {
#ifdef DEAL_II_WITH_PETSC
        init_system_matrix(system_matrix_petsc, dof_handler.get_communicator());
#else
        AssertThrow(
          false,
          dealii::ExcMessage(
            "Make sure that DEAL_II_WITH_PETSC is activated if you want to use SparseMatrixType::PETSc."));
#endif
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("not implemented."));
      }

      system_matrix_based_been_initialized = true;
    }

    // calculate matrix
    if(this->data.sparse_matrix_type == SparseMatrixType::Trilinos)
    {
#ifdef DEAL_II_WITH_TRILINOS
      system_matrix_trilinos *= 0.0;
      calculate_system_matrix(system_matrix_trilinos);
#else
      AssertThrow(
        false,
        dealii::ExcMessage(
          "Make sure that DEAL_II_WITH_TRILINOS is activated if you want to use SparseMatrixType::Trilinos."));
#endif
    }
    else if(this->data.sparse_matrix_type == SparseMatrixType::PETSc)
    {
#ifdef DEAL_II_WITH_PETSC
      system_matrix_petsc *= 0.0;
      calculate_system_matrix(system_matrix_petsc);

      if(system_matrix_petsc.m() > 0)
      {
        // get vector partitioner
        dealii::LinearAlgebra::distributed::Vector<Number> vector;
        initialize_dof_vector(vector);
        VecCreateMPI(system_matrix_petsc.get_mpi_communicator(),
                     vector.get_partitioner()->locally_owned_size(),
                     PETSC_DETERMINE,
                     &petsc_vector_dst);
        VecCreateMPI(system_matrix_petsc.get_mpi_communicator(),
                     vector.get_partitioner()->locally_owned_size(),
                     PETSC_DETERMINE,
                     &petsc_vector_src);
      }
#else
      AssertThrow(
        false,
        dealii::ExcMessage(
          "Make sure that DEAL_II_WITH_PETSC is activated if you want to use SparseMatrixType::PETSc."));
#endif
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("not implemented."));
    }
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::apply_matrix_based(VectorType &       dst,
                                                            VectorType const & src) const
{
  if(this->data.sparse_matrix_type == SparseMatrixType::Trilinos)
  {
#ifdef DEAL_II_WITH_TRILINOS
    apply_function_in_double_precision(
      dst,
      src,
      [&](dealii::LinearAlgebra::distributed::Vector<double> &       dst_double,
          dealii::LinearAlgebra::distributed::Vector<double> const & src_double) {
        system_matrix_trilinos.vmult(dst_double, src_double);
      });
#else
    AssertThrow(
      false,
      dealii::ExcMessage(
        "Make sure that DEAL_II_WITH_TRILINOS is activated if you want to use SparseMatrixType::Trilinos."));
#endif
  }
  else if(this->data.sparse_matrix_type == SparseMatrixType::PETSc)
  {
#ifdef DEAL_II_WITH_PETSC
    if(system_matrix_petsc.m() > 0)
    {
      apply_petsc_operation(dst,
                            src,
                            petsc_vector_dst,
                            petsc_vector_src,
                            [&](dealii::PETScWrappers::VectorBase &       petsc_dst,
                                dealii::PETScWrappers::VectorBase const & petsc_src) {
                              system_matrix_petsc.vmult(petsc_dst, petsc_src);
                            });
    }
#else
    AssertThrow(
      false,
      dealii::ExcMessage(
        "Make sure that DEAL_II_WITH_PETSC is activated if you want to use SparseMatrixType::PETSc."));
#endif
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("not implemented."));
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::apply_matrix_based_add(VectorType &       dst,
                                                                VectorType const & src) const
{
  VectorType tmp;
  tmp.reinit(dst, false);

  apply_matrix_based(tmp, src);

  dst += tmp;
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::rhs(VectorType & rhs) const
{
  rhs = 0;
  this->rhs_add(rhs);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::rhs_add(VectorType & rhs) const
{
  if(is_dg)
  {
    if(evaluate_face_integrals())
    {
      VectorType tmp;
      tmp.reinit(rhs, false);

      matrix_free->loop(&This::cell_loop_empty,
                        &This::face_loop_empty,
                        &This::boundary_face_loop_inhom_operator,
                        this,
                        tmp,
                        tmp);

      // multiply by -1.0 since the boundary face integrals have to be shifted to the right hand
      // side
      rhs.add(-1.0, tmp);
    }
  }
  else
  {
    VectorType src_tmp, dst_tmp;
    src_tmp.reinit(rhs, false);
    dst_tmp.reinit(rhs, false);

    // Set constrained degrees of freedom according to inhomogeneous Dirichlet boundary conditions.
    //  The rest of the vector remains unchanged.
    set_inhomogeneous_boundary_values(src_tmp);

    // Since src_tmp = 0 apart from inhomogeneous boundary data, the function evaluate_add() only
    // computes the inhomogeneous part of the operator.
    this->evaluate_add(dst_tmp, src_tmp);
    // Minus sign since we compute the contribution to the right-hand side.
    rhs -= dst_tmp;
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::evaluate(VectorType & dst, VectorType const & src) const
{
  dst = 0;
  evaluate_add(dst, src);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::evaluate_add(VectorType &       dst,
                                                      VectorType const & src) const
{
  if(is_dg)
  {
    if(evaluate_face_integrals())
    {
      matrix_free->loop(&This::cell_loop,
                        &This::face_loop,
                        &This::boundary_face_loop_full_operator,
                        this,
                        dst,
                        src);
    }
    else
    {
      matrix_free->cell_loop(&This::cell_loop, this, dst, src);
    }
  }
  else
  {
    AssertThrow(data.dof_index_inhomogeneous != dealii::numbers::invalid_unsigned_int,
                dealii::ExcMessage("dof_index_inhomogeneous is uninitialized."));

    // Perform matrix-vector product using a src-vector which contains inhomogeneous Dirichlet
    // values to obtain the hom+inhom action of the operator.
    if(evaluate_face_integrals())
    {
      matrix_free->loop(&This::cell_loop_full_operator,
                        &This::face_loop_empty,
                        &This::boundary_face_loop_inhom_operator,
                        this,
                        dst,
                        src);
    }
    else
    {
      matrix_free->cell_loop(&This::cell_loop_full_operator, this, dst, src);
    }
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::calculate_diagonal(VectorType & diagonal) const
{
  if(diagonal.size() == 0)
    this->initialize_dof_vector(diagonal);
  diagonal = 0;
  add_diagonal(diagonal);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::add_diagonal(VectorType & diagonal) const
{
  // compute diagonal
  if(is_dg and evaluate_face_integrals())
  {
    if(data.use_cell_based_loops)
    {
      matrix_free->cell_loop(&This::cell_based_loop_diagonal, this, diagonal, diagonal);
    }
    else
    {
      matrix_free->loop(&This::cell_loop_diagonal,
                        &This::face_loop_diagonal,
                        &This::boundary_face_loop_diagonal,
                        this,
                        diagonal,
                        diagonal);
    }
  }
  else
  {
    dealii::MatrixFreeTools::
      compute_diagonal<dim, -1, 0, n_components, Number, dealii::VectorizedArray<Number>>(
        *matrix_free,
        diagonal,
        [&](auto & integrator) -> void {
          // TODO: this is currently done for every column, but would only be necessary
          // once per cell
          this->reinit_cell_derived(integrator, integrator.get_current_cell_index());

          integrator.evaluate(integrator_flags.cell_evaluate);

          this->do_cell_integral(integrator);

          integrator.integrate(integrator_flags.cell_integrate);
        },
        data.dof_index,
        data.quad_index);
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::add_block_diagonal_matrices(
  std::vector<LAPACKMatrix> & matrices) const
{
  AssertThrow(is_dg, dealii::ExcMessage("Block Jacobi only implemented for DG!"));

  if(evaluate_face_integrals())
  {
    if(data.use_cell_based_loops)
    {
      matrix_free->cell_loop(&This::cell_based_loop_block_diagonal, this, matrices, matrices);
    }
    else
    {
      AssertThrow(
        n_mpi_processes == 1,
        dealii::ExcMessage(
          "Block diagonal calculation with separate loops over cells and faces only works in serial. "
          "Use cell based loops for parallel computations."));

      matrix_free->loop(&This::cell_loop_block_diagonal,
                        &This::face_loop_block_diagonal,
                        &This::boundary_face_loop_block_diagonal,
                        this,
                        matrices,
                        matrices);
    }
  }
  else
  {
    matrix_free->cell_loop(&This::cell_loop_block_diagonal, this, matrices, matrices);
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::apply_inverse_block_diagonal(VectorType &       dst,
                                                                      VectorType const & src) const
{
  // matrix-free
  if(this->data.implement_block_diagonal_preconditioner_matrix_free)
  {
    if(evaluate_face_integrals())
    {
      AssertThrow(data.use_cell_based_loops,
                  dealii::ExcMessage("Cell based face loops have to be activated for matrix-free "
                                     "implementation of block diagonal preconditioner, if face "
                                     "integrals need to be evaluated."));
      AssertThrow(matrix_free->get_dof_handler(data.dof_index)
                    .get_triangulation()
                    .all_reference_cells_are_hyper_cube(),
                  dealii::ExcMessage("Can't do cell based loop over faces for simplices."));
    }

    // Solve elementwise block Jacobi problems iteratively using an elementwise solver vectorized
    // over several elements.
    elementwise_solver->solve(dst, src);
  }
  else // matrix-based
  {
    // Simply apply inverse of block matrices (using the LU factorization that has been computed
    // before).
    apply_inverse_block_diagonal_matrix_based(dst, src);
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::apply_inverse_block_diagonal_matrix_based(
  VectorType &       dst,
  VectorType const & src) const
{
  AssertThrow(is_dg, dealii::ExcMessage("Block Jacobi only implemented for DG!"));

  matrix_free->cell_loop(&This::cell_loop_apply_inverse_block_diagonal_matrix_based,
                         this,
                         dst,
                         src);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::initialize_block_diagonal_preconditioner_matrix_free(
  bool const initialize) const
{
  elementwise_operator = std::make_shared<ELEMENTWISE_OPERATOR>(*this);

  if(data.preconditioner_block_diagonal == Elementwise::Preconditioner::None)
  {
    typedef Elementwise::PreconditionerIdentity<dealii::VectorizedArray<Number>> IDENTITY;

    IntegratorCell integrator =
      IntegratorCell(*this->matrix_free, this->data.dof_index, this->data.quad_index);

    elementwise_preconditioner = std::make_shared<IDENTITY>(integrator.dofs_per_cell);
  }
  else if(data.preconditioner_block_diagonal == Elementwise::Preconditioner::PointJacobi)
  {
    typedef Elementwise::JacobiPreconditioner<dim, n_components, Number, This> POINT_JACOBI;

    elementwise_preconditioner = std::make_shared<POINT_JACOBI>(
      get_matrix_free(), get_dof_index(), get_quad_index(), *this, initialize);
  }
  else if(data.preconditioner_block_diagonal == Elementwise::Preconditioner::InverseMassMatrix)
  {
    typedef Elementwise::InverseMassPreconditioner<dim, n_components, Number> INVERSE_MASS;

    elementwise_preconditioner =
      std::make_shared<INVERSE_MASS>(get_matrix_free(), get_dof_index(), get_quad_index());
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }

  Elementwise::IterativeSolverData iterative_solver_data;
  iterative_solver_data.solver_type = data.solver_block_diagonal;
  iterative_solver_data.solver_data = data.solver_data_block_diagonal;

  elementwise_solver = std::make_shared<ELEMENTWISE_SOLVER>(
    *std::dynamic_pointer_cast<ELEMENTWISE_OPERATOR>(elementwise_operator),
    *std::dynamic_pointer_cast<ELEMENTWISE_PRECONDITIONER>(elementwise_preconditioner),
    iterative_solver_data);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::update_block_diagonal_preconditioner_matrix_free() const
{
  elementwise_preconditioner->update();
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::initialize_block_diagonal_preconditioner_matrix_based(
  bool const initialize) const
{
  // allocate memory
  auto dofs =
    matrix_free->get_shape_info(this->data.dof_index).dofs_per_component_on_cell * n_components;
  matrices.resize(matrix_free->n_cell_batches() * vectorization_length, LAPACKMatrix(dofs, dofs));

  // compute and factorize matrices
  if(initialize)
    update_block_diagonal_preconditioner_matrix_based();
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::update_block_diagonal_preconditioner_matrix_based() const
{
  // clear matrices
  initialize_block_jacobi_matrices_with_zero(matrices);

  // compute block matrices and add
  add_block_diagonal_matrices(matrices);

  calculate_lu_factorization_block_jacobi(matrices);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::apply_add_block_diagonal_elementwise(
  unsigned int const                            cell,
  dealii::VectorizedArray<Number> * const       dst,
  dealii::VectorizedArray<Number> const * const src,
  unsigned int const                            problem_size) const
{
  (void)problem_size;

  AssertThrow(is_dg, dealii::ExcMessage("Block Jacobi only implemented for DG!"));

  IntegratorCell integrator =
    IntegratorCell(*this->matrix_free, this->data.dof_index, this->data.quad_index);

  this->reinit_cell(integrator, cell);

  for(unsigned int i = 0; i < integrator.dofs_per_cell; ++i)
    integrator.begin_dof_values()[i] = src[i];

  integrator.evaluate(integrator_flags.cell_evaluate);

  this->do_cell_integral(integrator);

  integrator.integrate(integrator_flags.cell_integrate);

  for(unsigned int i = 0; i < integrator.dofs_per_cell; ++i)
    dst[i] += integrator.begin_dof_values()[i];

  if(is_dg and evaluate_face_integrals())
  {
    IntegratorFace integrator_m =
      IntegratorFace(*this->matrix_free, true, this->data.dof_index, this->data.quad_index);
    IntegratorFace integrator_p =
      IntegratorFace(*this->matrix_free, false, this->data.dof_index, this->data.quad_index);

    // face integrals
    unsigned int const n_faces = dealii::ReferenceCells::template get_hypercube<dim>().n_faces();
    for(unsigned int face = 0; face < n_faces; ++face)
    {
      auto bids = (*matrix_free).get_faces_by_cells_boundary_id(cell, face);
      auto bid  = bids[0];

      this->reinit_face_cell_based(integrator_m, integrator_p, cell, face, bid);

      for(unsigned int i = 0; i < integrator_m.dofs_per_cell; ++i)
        integrator_m.begin_dof_values()[i] = src[i];

      // no need to read dof values for integrator_p (already initialized with 0)

      integrator_m.evaluate(integrator_flags.face_evaluate);

      if(bid == dealii::numbers::internal_face_boundary_id) // internal face
      {
        this->do_face_int_integral_cell_based(integrator_m, integrator_p);
      }
      else // boundary face
      {
        this->do_boundary_integral(integrator_m, OperatorType::homogeneous, bid);
      }

      integrator_m.integrate(integrator_flags.face_integrate);

      for(unsigned int i = 0; i < integrator_m.dofs_per_cell; ++i)
        dst[i] += integrator_m.begin_dof_values()[i];
    }
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::initialize_block_diagonal_preconditioner(
  bool const initialize) const
{
  AssertThrow(is_dg, dealii::ExcMessage("Block Jacobi only implemented for DG!"));

  if(data.implement_block_diagonal_preconditioner_matrix_free)
  {
    initialize_block_diagonal_preconditioner_matrix_free(initialize);
  }
  else // matrix-based variant
  {
    initialize_block_diagonal_preconditioner_matrix_based(initialize);
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::update_block_diagonal_preconditioner() const
{
  AssertThrow(is_dg, dealii::ExcMessage("Block Jacobi only implemented for DG!"));

  if(data.implement_block_diagonal_preconditioner_matrix_free)
  {
    // For the matrix-free variant we have to update the elementwise preconditioner.
    update_block_diagonal_preconditioner_matrix_free();
  }
  else // matrix-based variant
  {
    // For the matrix-based variant we have to recompute the block matrices.
    update_block_diagonal_preconditioner_matrix_based();
  }
}

#ifdef DEAL_II_WITH_TRILINOS
template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::init_system_matrix(
  dealii::TrilinosWrappers::SparseMatrix & system_matrix,
  MPI_Comm const &                         mpi_comm) const
{
  dealii::DynamicSparsityPattern dsp;
  internal_init_system_matrix(system_matrix, dsp, mpi_comm);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::calculate_system_matrix(
  dealii::TrilinosWrappers::SparseMatrix & system_matrix) const
{
  internal_calculate_system_matrix(system_matrix);
}
#endif

#ifdef DEAL_II_WITH_PETSC
template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::init_system_matrix(
  dealii::PETScWrappers::MPI::SparseMatrix & system_matrix,
  MPI_Comm const &                           mpi_comm) const
{
  dealii::DynamicSparsityPattern dsp;
  internal_init_system_matrix(system_matrix, dsp, mpi_comm);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::calculate_system_matrix(
  dealii::PETScWrappers::MPI::SparseMatrix & system_matrix) const
{
  internal_calculate_system_matrix(system_matrix);
}
#endif

template<int dim, typename Number, int n_components>
template<typename SparseMatrix>
void
OperatorBase<dim, Number, n_components>::internal_init_system_matrix(
  SparseMatrix &                   system_matrix,
  dealii::DynamicSparsityPattern & dsp,
  MPI_Comm const &                 mpi_comm) const
{
  dealii::DoFHandler<dim> const & dof_handler =
    this->matrix_free->get_dof_handler(this->data.dof_index);

  dealii::IndexSet const & owned_dofs =
    is_mg ? dof_handler.locally_owned_mg_dofs(this->level) : dof_handler.locally_owned_dofs();

  // check for a valid subcommunicator by asserting that the sum of the dofs
  // owned by all participating processes is equal to the sum of global dofs -
  // the second check is needed on the MPI processes not participating in the
  // actual communication, i.e., which are left out from sub-communication
  dealii::types::global_dof_index const sum_of_locally_owned_dofs =
    dealii::Utilities::MPI::sum(owned_dofs.n_elements(), mpi_comm);
  bool const my_rank_is_part_of_subcommunicator = sum_of_locally_owned_dofs == owned_dofs.size();
  AssertThrow(my_rank_is_part_of_subcommunicator or owned_dofs.n_elements() == 0,
              dealii::ExcMessage(
                "The given communicator mpi_comm in init_system_matrix does not span "
                "all MPI processes needed to cover the index space of all dofs: " +
                std::to_string(sum_of_locally_owned_dofs) + " vs " +
                std::to_string(owned_dofs.size())));

  dealii::IndexSet relevant_dofs;
  if(is_mg)
    dealii::DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, relevant_dofs);
  else
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
  dsp.reinit(relevant_dofs.size(), relevant_dofs.size(), relevant_dofs);

  if(is_dg and is_mg)
    dealii::MGTools::make_flux_sparsity_pattern(dof_handler, dsp, this->level);
  else if(is_dg and not is_mg)
    dealii::DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
  else if(/*not(is_dg) and*/ is_mg)
    dealii::MGTools::make_sparsity_pattern(dof_handler, dsp, this->level, *this->constraint);
  else /* if (not(is_dg) and not(is_mg))*/
    dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, *this->constraint);

  if(my_rank_is_part_of_subcommunicator)
  {
    dealii::SparsityTools::distribute_sparsity_pattern(dsp, owned_dofs, mpi_comm, relevant_dofs);
    system_matrix.reinit(owned_dofs, owned_dofs, dsp, mpi_comm);
  }
}

template<int dim, typename Number, int n_components>
template<typename SparseMatrix>
void
OperatorBase<dim, Number, n_components>::internal_calculate_system_matrix(
  SparseMatrix & system_matrix) const
{
  // assemble matrix locally on each process
  if(evaluate_face_integrals() and is_dg)
  {
    matrix_free->loop(&This::cell_loop_calculate_system_matrix,
                      &This::face_loop_calculate_system_matrix,
                      &This::boundary_face_loop_calculate_system_matrix,
                      this,
                      system_matrix,
                      system_matrix);
  }
  else
  {
    matrix_free->cell_loop(&This::cell_loop_calculate_system_matrix,
                           this,
                           system_matrix,
                           system_matrix);
  }

  // communicate overlapping matrix parts
  system_matrix.compress(dealii::VectorOperation::add);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::reinit_cell(IntegratorCell &   integrator,
                                                     unsigned int const cell) const
{
  integrator.reinit(cell);

  reinit_cell_derived(integrator, cell);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::reinit_cell_derived(IntegratorCell &   integrator,
                                                             unsigned int const cell) const
{
  (void)integrator;
  (void)cell;

  // override this function in derived classes if additional initialization is necessary
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::reinit_face(IntegratorFace &   integrator_m,
                                                     IntegratorFace &   integrator_p,
                                                     unsigned int const face) const
{
  integrator_m.reinit(face);
  integrator_p.reinit(face);

  reinit_face_derived(integrator_m, integrator_p, face);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::reinit_face_derived(IntegratorFace &   integrator_m,
                                                             IntegratorFace &   integrator_p,
                                                             unsigned int const face) const
{
  (void)integrator_m;
  (void)integrator_p;
  (void)face;
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::reinit_boundary_face(IntegratorFace &   integrator_m,
                                                              unsigned int const face) const
{
  integrator_m.reinit(face);

  reinit_boundary_face_derived(integrator_m, face);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::reinit_boundary_face_derived(IntegratorFace & integrator_m,
                                                                      unsigned int const face) const
{
  (void)integrator_m;
  (void)face;
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::do_cell_integral(IntegratorCell & integrator) const
{
  (void)integrator;

  AssertThrow(false,
              dealii::ExcMessage("OperatorBase::do_cell_integral() has not been implemented!"));
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::do_face_integral(IntegratorFace & integrator_m,
                                                          IntegratorFace & integrator_p) const
{
  (void)integrator_m;
  (void)integrator_p;

  AssertThrow(false,
              dealii::ExcMessage("OperatorBase::do_face_integral() has not been implemented!"));
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::do_boundary_integral(
  IntegratorFace &                   integrator,
  OperatorType const &               operator_type,
  dealii::types::boundary_id const & boundary_id) const
{
  (void)integrator;
  (void)operator_type;
  (void)boundary_id;

  AssertThrow(false,
              dealii::ExcMessage("OperatorBase::do_boundary_integral() has not been implemented!"));
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::do_boundary_integral_continuous(
  IntegratorFace &                   integrator,
  dealii::types::boundary_id const & boundary_id) const
{
  (void)integrator;
  (void)boundary_id;

  AssertThrow(
    false,
    dealii::ExcMessage(
      "OperatorBase::do_boundary_integral_continuous() has to be overridden by derived class!"));
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::set_inhomogeneous_boundary_values(
  VectorType & solution) const
{
  (void)solution;

  AssertThrow(
    false,
    dealii::ExcMessage(
      "OperatorBase::set_inhomogeneous_boundary_values() has to be overridden by derived class!"));
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::do_face_int_integral(IntegratorFace & integrator_m,
                                                              IntegratorFace & integrator_p) const
{
  (void)integrator_m;
  (void)integrator_p;

  AssertThrow(false,
              dealii::ExcMessage("OperatorBase::do_face_int_integral() has not been implemented!"));
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::do_face_ext_integral(IntegratorFace & integrator_m,
                                                              IntegratorFace & integrator_p) const
{
  (void)integrator_m;
  (void)integrator_p;

  AssertThrow(false,
              dealii::ExcMessage("OperatorBase::do_face_ext_integral() has not been implemented!"));
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::reinit_face_cell_based(
  IntegratorFace &                 integrator_m,
  IntegratorFace &                 integrator_p,
  unsigned int const               cell,
  unsigned int const               face,
  dealii::types::boundary_id const boundary_id) const
{
  integrator_m.reinit(cell, face);

  if(boundary_id == dealii::numbers::internal_face_boundary_id) // internal face
  {
    integrator_p.reinit(cell, face);
  }

  reinit_face_cell_based_derived(integrator_m, integrator_p, cell, face, boundary_id);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::reinit_face_cell_based_derived(
  IntegratorFace &                 integrator_m,
  IntegratorFace &                 integrator_p,
  unsigned int const               cell,
  unsigned int const               face,
  dealii::types::boundary_id const boundary_id) const
{
  (void)integrator_m;
  (void)integrator_p;
  (void)cell;
  (void)face;
  (void)boundary_id;
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::do_face_int_integral_cell_based(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  this->do_face_int_integral(integrator_m, integrator_p);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::create_standard_basis(unsigned int     j,
                                                               IntegratorCell & integrator) const
{
  // create a standard basis in the dof values of FEEvalution
  for(unsigned int i = 0; i < integrator.dofs_per_cell; ++i)
    integrator.begin_dof_values()[i] = dealii::make_vectorized_array<Number>(0.);
  integrator.begin_dof_values()[j] = dealii::make_vectorized_array<Number>(1.);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::create_standard_basis(unsigned int     j,
                                                               IntegratorFace & integrator) const
{
  // create a standard basis in the dof values of FEEvalution
  for(unsigned int i = 0; i < integrator.dofs_per_cell; ++i)
    integrator.begin_dof_values()[i] = dealii::make_vectorized_array<Number>(0.);
  integrator.begin_dof_values()[j] = dealii::make_vectorized_array<Number>(1.);
}


template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::create_standard_basis(unsigned int     j,
                                                               IntegratorFace & integrator_1,
                                                               IntegratorFace & integrator_2) const
{
  // create a standard basis in the dof values of the first FEFaceEvalution
  for(unsigned int i = 0; i < integrator_1.dofs_per_cell; ++i)
    integrator_1.begin_dof_values()[i] = dealii::make_vectorized_array<Number>(0.);
  integrator_1.begin_dof_values()[j] = dealii::make_vectorized_array<Number>(1.);

  // clear dof values of the second FEFaceEvalution
  for(unsigned int i = 0; i < integrator_2.dofs_per_cell; ++i)
    integrator_2.begin_dof_values()[i] = dealii::make_vectorized_array<Number>(0.);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::cell_loop_full_operator(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  AssertThrow(not is_dg,
              dealii::ExcMessage("This function should not be called for is_dg = true."));

  IntegratorCell integrator_inhom =
    IntegratorCell(matrix_free, this->data.dof_index_inhomogeneous, this->data.quad_index);

  IntegratorCell integrator =
    IntegratorCell(matrix_free, this->data.dof_index, this->data.quad_index);

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    this->reinit_cell(integrator_inhom, cell);
    integrator.reinit(cell);

    integrator_inhom.gather_evaluate(src, integrator_flags.cell_evaluate);

    this->do_cell_integral(integrator_inhom);

    // make sure that we do not write into Dirichlet degrees of freedom
    integrator_inhom.integrate(integrator_flags.cell_integrate, integrator.begin_dof_values());
    integrator.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::cell_loop(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  IntegratorCell integrator =
    IntegratorCell(matrix_free, this->data.dof_index, this->data.quad_index);

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    this->reinit_cell(integrator, cell);

    integrator.gather_evaluate(src, integrator_flags.cell_evaluate);

    this->do_cell_integral(integrator);

    integrator.integrate_scatter(integrator_flags.cell_integrate, dst);
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::face_loop(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  IntegratorFace integrator_m =
    IntegratorFace(matrix_free, true, this->data.dof_index, this->data.quad_index);
  IntegratorFace integrator_p =
    IntegratorFace(matrix_free, false, this->data.dof_index, this->data.quad_index);

  for(auto face = range.first; face < range.second; ++face)
  {
    this->reinit_face(integrator_m, integrator_p, face);

    integrator_m.gather_evaluate(src, integrator_flags.face_evaluate);
    integrator_p.gather_evaluate(src, integrator_flags.face_evaluate);

    this->do_face_integral(integrator_m, integrator_p);

    integrator_m.integrate_scatter(integrator_flags.face_integrate, dst);
    integrator_p.integrate_scatter(integrator_flags.face_integrate, dst);
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::boundary_face_loop_hom_operator(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  IntegratorFace integrator_m =
    IntegratorFace(matrix_free, true, this->data.dof_index, this->data.quad_index);

  for(unsigned int face = range.first; face < range.second; face++)
  {
    this->reinit_boundary_face(integrator_m, face);

    integrator_m.gather_evaluate(src, integrator_flags.face_evaluate);

    do_boundary_integral(integrator_m,
                         OperatorType::homogeneous,
                         matrix_free.get_boundary_id(face));

    integrator_m.integrate_scatter(integrator_flags.face_integrate, dst);
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::boundary_face_loop_inhom_operator(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  (void)src;

  if(is_dg)
  {
    IntegratorFace integrator_m =
      IntegratorFace(matrix_free, true, this->data.dof_index, this->data.quad_index);

    for(unsigned int face = range.first; face < range.second; face++)
    {
      this->reinit_boundary_face(integrator_m, face);

      // note: no gathering/evaluation is necessary when calculating the
      //       inhomogeneous part of boundary face integrals

      do_boundary_integral(integrator_m,
                           OperatorType::inhomogeneous,
                           matrix_free.get_boundary_id(face));

      integrator_m.integrate_scatter(integrator_flags.face_integrate, dst);
    }
  }
  else // continuous FE discretization (e.g., apply Neumann BCs)
  {
    IntegratorFace integrator_m_inhom =
      IntegratorFace(matrix_free, true, this->data.dof_index_inhomogeneous, this->data.quad_index);

    IntegratorFace integrator_m =
      IntegratorFace(matrix_free, true, this->data.dof_index, this->data.quad_index);

    for(unsigned int face = range.first; face < range.second; face++)
    {
      this->reinit_boundary_face(integrator_m_inhom, face);
      integrator_m.reinit(face);

      // note: no gathering/evaluation is necessary when calculating the
      //       inhomogeneous part of boundary face integrals

      do_boundary_integral_continuous(integrator_m_inhom, matrix_free.get_boundary_id(face));

      // make sure that we do not write into Dirichlet degrees of freedom
      integrator_m_inhom.integrate(integrator_flags.face_integrate,
                                   integrator_m.begin_dof_values());
      integrator_m.distribute_local_to_global(dst);
    }
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::boundary_face_loop_full_operator(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  AssertThrow(is_dg,
              dealii::ExcMessage("OperatorBase::boundary_face_loop_full_operator() "
                                 "should only be called in case of is_dg == true."));

  IntegratorFace integrator_m =
    IntegratorFace(matrix_free, true, this->data.dof_index, this->data.quad_index);

  for(unsigned int face = range.first; face < range.second; face++)
  {
    this->reinit_boundary_face(integrator_m, face);

    integrator_m.gather_evaluate(src, integrator_flags.face_evaluate);

    do_boundary_integral(integrator_m, OperatorType::full, matrix_free.get_boundary_id(face));

    integrator_m.integrate_scatter(integrator_flags.face_integrate, dst);
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::cell_loop_empty(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  (void)matrix_free;
  (void)dst;
  (void)src;
  (void)range;

  // do nothing
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::face_loop_empty(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  (void)matrix_free;
  (void)dst;
  (void)src;
  (void)range;

  // do nothing
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::cell_loop_diagonal(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  (void)src;

  IntegratorCell integrator =
    IntegratorCell(matrix_free, this->data.dof_index, this->data.quad_index);

  // create temporal array for local diagonal
  unsigned int const                                     dofs_per_cell = integrator.dofs_per_cell;
  dealii::AlignedVector<dealii::VectorizedArray<Number>> local_diag(dofs_per_cell);

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    this->reinit_cell(integrator, cell);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of dealii::FEEvaluation
      this->create_standard_basis(j, integrator);

      integrator.evaluate(integrator_flags.cell_evaluate);

      this->do_cell_integral(integrator);

      integrator.integrate(integrator_flags.cell_integrate);

      // extract single value from result vector and temporally store it
      local_diag[j] = integrator.begin_dof_values()[j];
    }
    // copy local diagonal entries into dof values of dealii::FEEvaluation ...
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      integrator.begin_dof_values()[j] = local_diag[j];

    // ... and write it back to global vector
    integrator.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::face_loop_diagonal(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  (void)src;

  IntegratorFace integrator_m =
    IntegratorFace(matrix_free, true, this->data.dof_index, this->data.quad_index);
  IntegratorFace integrator_p =
    IntegratorFace(matrix_free, false, this->data.dof_index, this->data.quad_index);

  // create temporal array for local diagonal
  unsigned int const                                     dofs_per_cell = integrator_m.dofs_per_cell;
  dealii::AlignedVector<dealii::VectorizedArray<Number>> local_diag(dofs_per_cell);

  for(auto face = range.first; face < range.second; ++face)
  {
    this->reinit_face(integrator_m, integrator_p, face);

    // interior face
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, integrator_m);

      integrator_m.evaluate(integrator_flags.face_evaluate);

      this->do_face_int_integral(integrator_m, integrator_p);

      integrator_m.integrate(integrator_flags.face_integrate);

      local_diag[j] = integrator_m.begin_dof_values()[j];
    }

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      integrator_m.begin_dof_values()[j] = local_diag[j];

    integrator_m.distribute_local_to_global(dst);

    // exterior face
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, integrator_p);

      integrator_p.evaluate(integrator_flags.face_evaluate);

      this->do_face_ext_integral(integrator_m, integrator_p);

      integrator_p.integrate(integrator_flags.face_integrate);

      local_diag[j] = integrator_p.begin_dof_values()[j];
    }

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      integrator_p.begin_dof_values()[j] = local_diag[j];

    integrator_p.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::boundary_face_loop_diagonal(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  (void)src;

  IntegratorFace integrator_m =
    IntegratorFace(matrix_free, true, this->data.dof_index, this->data.quad_index);

  // create temporal array for local diagonal
  unsigned int const                                     dofs_per_cell = integrator_m.dofs_per_cell;
  dealii::AlignedVector<dealii::VectorizedArray<Number>> local_diag(dofs_per_cell);

  for(unsigned int face = range.first; face < range.second; face++)
  {
    auto bid = matrix_free.get_boundary_id(face);

    this->reinit_boundary_face(integrator_m, face);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, integrator_m);

      integrator_m.evaluate(integrator_flags.face_evaluate);

      this->do_boundary_integral(integrator_m, OperatorType::homogeneous, bid);

      integrator_m.integrate(integrator_flags.face_integrate);

      local_diag[j] = integrator_m.begin_dof_values()[j];
    }

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      integrator_m.begin_dof_values()[j] = local_diag[j];

    integrator_m.distribute_local_to_global(dst);
  }
}


template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::cell_based_loop_diagonal(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  (void)src;

  IntegratorCell integrator =
    IntegratorCell(matrix_free, this->data.dof_index, this->data.quad_index);
  IntegratorFace integrator_m =
    IntegratorFace(matrix_free, true, this->data.dof_index, this->data.quad_index);
  IntegratorFace integrator_p =
    IntegratorFace(matrix_free, false, this->data.dof_index, this->data.quad_index);

  // create temporal array for local diagonal
  unsigned int const                                     dofs_per_cell = integrator.dofs_per_cell;
  dealii::AlignedVector<dealii::VectorizedArray<Number>> local_diag(dofs_per_cell);

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    this->reinit_cell(integrator, cell);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, integrator);

      integrator.evaluate(integrator_flags.cell_evaluate);

      this->do_cell_integral(integrator);

      integrator.integrate(integrator_flags.cell_integrate);

      local_diag[j] = integrator.begin_dof_values()[j];
    }

    // loop over all faces and gather results into local diagonal local_diag
    if(evaluate_face_integrals())
    {
      unsigned int const n_faces = dealii::ReferenceCells::template get_hypercube<dim>().n_faces();
      for(unsigned int face = 0; face < n_faces; ++face)
      {
        auto bids = matrix_free.get_faces_by_cells_boundary_id(cell, face);
        auto bid  = bids[0];

        this->reinit_face_cell_based(integrator_m, integrator_p, cell, face, bid);

#ifdef DEBUG
        unsigned int const n_filled_lanes = matrix_free.n_active_entries_per_cell_batch(cell);
        for(unsigned int v = 0; v < n_filled_lanes; v++)
          Assert(bid == bids[v],
                 dealii::ExcMessage(
                   "Cell-based face loop encountered face batch with different bids."));
#endif

        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          this->create_standard_basis(j, integrator_m);

          integrator_m.evaluate(integrator_flags.face_evaluate);

          if(bid == dealii::numbers::internal_face_boundary_id) // internal face
          {
            this->do_face_int_integral_cell_based(integrator_m, integrator_p);
          }
          else // boundary face
          {
            this->do_boundary_integral(integrator_m, OperatorType::homogeneous, bid);
          }

          integrator_m.integrate(integrator_flags.face_integrate);

          // note: += for accumulation of all contributions of this (macro) cell
          //          including: cell-, face-, boundary-stiffness matrix
          local_diag[j] += integrator_m.begin_dof_values()[j];
        }
      }
    }

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      integrator.begin_dof_values()[j] = local_diag[j];

    integrator.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::cell_loop_apply_inverse_block_diagonal_matrix_based(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           cell_range) const
{
  IntegratorCell integrator =
    IntegratorCell(matrix_free, this->data.dof_index, this->data.quad_index);

  unsigned int const dofs_per_cell = integrator.dofs_per_cell;

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    this->reinit_cell(integrator, cell);

    integrator.read_dof_values(src);

    for(unsigned int v = 0; v < vectorization_length; ++v)
    {
      dealii::Vector<Number> src_vector(dofs_per_cell);
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        src_vector(j) = integrator.begin_dof_values()[j][v];

      // apply inverse matrix
      matrices[cell * vectorization_length + v].solve(src_vector, false);

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        integrator.begin_dof_values()[j][v] = src_vector(j);
    }

    integrator.set_dof_values(dst);
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::cell_loop_block_diagonal(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  std::vector<LAPACKMatrix> &             matrices,
  std::vector<LAPACKMatrix> const &,
  Range const & range) const
{
  IntegratorCell integrator =
    IntegratorCell(matrix_free, this->data.dof_index, this->data.quad_index);

  unsigned int const dofs_per_cell = integrator.dofs_per_cell;

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    unsigned int const n_filled_lanes = matrix_free.n_active_entries_per_cell_batch(cell);

    this->reinit_cell(integrator, cell);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, integrator);

      integrator.evaluate(integrator_flags.cell_evaluate);

      this->do_cell_integral(integrator);

      integrator.integrate(integrator_flags.cell_integrate);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices[cell * vectorization_length + v](i, j) += integrator.begin_dof_values()[i][v];
    }
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::face_loop_block_diagonal(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  std::vector<LAPACKMatrix> &             matrices,
  std::vector<LAPACKMatrix> const &,
  Range const & range) const
{
  IntegratorFace integrator_m =
    IntegratorFace(matrix_free, true, this->data.dof_index, this->data.quad_index);
  IntegratorFace integrator_p =
    IntegratorFace(matrix_free, false, this->data.dof_index, this->data.quad_index);

  unsigned int const dofs_per_cell = integrator_m.dofs_per_cell;

  for(auto face = range.first; face < range.second; ++face)
  {
    unsigned int const n_filled_lanes = matrix_free.n_active_entries_per_face_batch(face);

    this->reinit_face(integrator_m, integrator_p, face);

    // interior face
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, integrator_m);

      integrator_m.evaluate(integrator_flags.face_evaluate);

      this->do_face_int_integral(integrator_m, integrator_p);

      integrator_m.integrate(integrator_flags.face_integrate);

      for(unsigned int v = 0; v < n_filled_lanes; ++v)
      {
        unsigned int const cell = matrix_free.get_face_info(face).cells_interior[v];
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          matrices[cell](i, j) += integrator_m.begin_dof_values()[i][v];
      }
    }

    // exterior face
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, integrator_p);

      integrator_p.evaluate(integrator_flags.face_evaluate);

      this->do_face_ext_integral(integrator_m, integrator_p);

      integrator_p.integrate(integrator_flags.face_integrate);

      for(unsigned int v = 0; v < n_filled_lanes; ++v)
      {
        unsigned int const cell = matrix_free.get_face_info(face).cells_exterior[v];
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          matrices[cell](i, j) += integrator_p.begin_dof_values()[i][v];
      }
    }
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::boundary_face_loop_block_diagonal(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  std::vector<LAPACKMatrix> &             matrices,
  std::vector<LAPACKMatrix> const &,
  Range const & range) const
{
  IntegratorFace integrator_m =
    IntegratorFace(matrix_free, true, this->data.dof_index, this->data.quad_index);

  unsigned int const dofs_per_cell = integrator_m.dofs_per_cell;

  for(auto face = range.first; face < range.second; ++face)
  {
    unsigned int const n_filled_lanes = matrix_free.n_active_entries_per_face_batch(face);

    this->reinit_boundary_face(integrator_m, face);

    auto bid = matrix_free.get_boundary_id(face);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, integrator_m);

      integrator_m.evaluate(integrator_flags.face_evaluate);

      this->do_boundary_integral(integrator_m, OperatorType::homogeneous, bid);

      integrator_m.integrate(integrator_flags.face_integrate);

      for(unsigned int v = 0; v < n_filled_lanes; ++v)
      {
        unsigned int const cell = matrix_free.get_face_info(face).cells_interior[v];
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          matrices[cell](i, j) += integrator_m.begin_dof_values()[i][v];
      }
    }
  }
}


template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::cell_based_loop_block_diagonal(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  std::vector<LAPACKMatrix> &             matrices,
  std::vector<LAPACKMatrix> const &,
  Range const & range) const
{
  IntegratorCell integrator =
    IntegratorCell(matrix_free, this->data.dof_index, this->data.quad_index);
  IntegratorFace integrator_m =
    IntegratorFace(matrix_free, true, this->data.dof_index, this->data.quad_index);
  IntegratorFace integrator_p =
    IntegratorFace(matrix_free, false, this->data.dof_index, this->data.quad_index);

  unsigned int const dofs_per_cell = integrator.dofs_per_cell;

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    unsigned int const n_filled_lanes = matrix_free.n_active_entries_per_cell_batch(cell);

    this->reinit_cell(integrator, cell);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, integrator);

      integrator.evaluate(integrator_flags.cell_evaluate);

      this->do_cell_integral(integrator);

      integrator.integrate(integrator_flags.cell_integrate);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices[cell * vectorization_length + v](i, j) += integrator.begin_dof_values()[i][v];
    }

    if(evaluate_face_integrals())
    {
      // loop over all faces
      unsigned int const n_faces = dealii::ReferenceCells::template get_hypercube<dim>().n_faces();
      for(unsigned int face = 0; face < n_faces; ++face)
      {
        auto bids = matrix_free.get_faces_by_cells_boundary_id(cell, face);
        auto bid  = bids[0];

        this->reinit_face_cell_based(integrator_m, integrator_p, cell, face, bid);

#ifdef DEBUG
        for(unsigned int v = 0; v < n_filled_lanes; v++)
          Assert(bid == bids[v],
                 dealii::ExcMessage(
                   "Cell-based face loop encountered face batch with different bids."));
#endif

        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          this->create_standard_basis(j, integrator_m);

          integrator_m.evaluate(integrator_flags.face_evaluate);

          if(bid == dealii::numbers::internal_face_boundary_id) // internal face
          {
            this->do_face_int_integral_cell_based(integrator_m, integrator_p);
          }
          else // boundary face
          {
            this->do_boundary_integral(integrator_m, OperatorType::homogeneous, bid);
          }

          integrator_m.integrate(integrator_flags.face_integrate);

          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            for(unsigned int v = 0; v < n_filled_lanes; ++v)
              matrices[cell * vectorization_length + v](i, j) +=
                integrator_m.begin_dof_values()[i][v];
        }
      }
    }
  }
}

template<int dim, typename Number, int n_components>
template<typename SparseMatrix>
void
OperatorBase<dim, Number, n_components>::cell_loop_calculate_system_matrix(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  SparseMatrix &                          dst,
  SparseMatrix const &                    src,
  Range const &                           range) const
{
  (void)src;

  IntegratorCell integrator =
    IntegratorCell(matrix_free, this->data.dof_index, this->data.quad_index);

  unsigned int const dofs_per_cell = integrator.dofs_per_cell;

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    unsigned int const n_filled_lanes = matrix_free.n_active_entries_per_cell_batch(cell);

    // create a temporal full matrix for the local element matrix of each ...
    // cell of each macro cell and ...
    FullMatrix_ matrices[vectorization_length];
    // set their size
    std::fill_n(matrices, vectorization_length, FullMatrix_(dofs_per_cell, dofs_per_cell));

    this->reinit_cell(integrator, cell);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, integrator);

      integrator.evaluate(integrator_flags.cell_evaluate);

      this->do_cell_integral(integrator);

      integrator.integrate(integrator_flags.cell_integrate);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices[v](i, j) = integrator.begin_dof_values()[i][v];
    }

    // finally assemble local matrices into global matrix
    for(unsigned int v = 0; v < n_filled_lanes; v++)
    {
      auto cell_v = matrix_free.get_cell_iterator(cell, v);

      std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);
      if(is_mg)
        cell_v->get_mg_dof_indices(dof_indices);
      else
        cell_v->get_dof_indices(dof_indices);

      if(not is_dg)
      {
        // in the case of CG: shape functions are not ordered lexicographically
        // see (https://www.dealii.org/8.5.1/doxygen/deal.II/classFE__Q.html)
        // so we have to fix the order
        auto temp = dof_indices;
        for(unsigned int j = 0; j < dof_indices.size(); j++)
          dof_indices[j] =
            temp[matrix_free.get_shape_info(this->data.dof_index).lexicographic_numbering[j]];
      }

      // choose the version of distribute_local_to_global with a single
      // `dof_indices` argument to indicate that we write to a diagonal block
      // of the matrix (vs 2 for off-diagonal ones); this implies a non-zero
      // entry is added to the diagonal of constrained matrix rows, ensuring
      // positive definiteness
      constraint_double.distribute_local_to_global(matrices[v], dof_indices, dst);
    }
  }
}

template<int dim, typename Number, int n_components>
template<typename SparseMatrix>
void
OperatorBase<dim, Number, n_components>::face_loop_calculate_system_matrix(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  SparseMatrix &                          dst,
  SparseMatrix const &                    src,
  Range const &                           range) const
{
  (void)src;

  IntegratorFace integrator_m =
    IntegratorFace(matrix_free, true, this->data.dof_index, this->data.quad_index);
  IntegratorFace integrator_p =
    IntegratorFace(matrix_free, false, this->data.dof_index, this->data.quad_index);

  unsigned int const dofs_per_cell = integrator_m.dofs_per_cell;

  // There are four matrices: M_mm, M_mp, M_pm, M_pp with M_mm, M_pp denoting
  // the block diagonal matrices for elements m,p and M_mp, M_pm the matrices
  // related to the coupling of neighboring elements. In the following, both
  // M_mm and M_mp are called matrices_m and both M_pm and M_pp are called
  // matrices_p so that we only have to store two matrices (matrices_m,
  // matrices_p) instead of four. This is possible since we compute M_mm, M_pm
  // in a first step (by varying solution functions on element m), and M_mp,
  // M_pp in a second step (by varying solution functions on element p).

  // create two local matrix: first one tested by test functions on element m and ...
  FullMatrix_ matrices_m[vectorization_length];
  std::fill_n(matrices_m, vectorization_length, FullMatrix_(dofs_per_cell, dofs_per_cell));
  // ... the other tested by test functions on element p
  FullMatrix_ matrices_p[vectorization_length];
  std::fill_n(matrices_p, vectorization_length, FullMatrix_(dofs_per_cell, dofs_per_cell));

  for(auto face = range.first; face < range.second; ++face)
  {
    // determine number of filled vector lanes
    unsigned int const n_filled_lanes = matrix_free.n_active_entries_per_face_batch(face);

    this->reinit_face(integrator_m, integrator_p, face);

    // process minus trial function
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of first dealii::FEFaceEvaluation and
      // clear dof values of second dealii::FEFaceEvaluation
      this->create_standard_basis(j, integrator_m, integrator_p);

      integrator_m.evaluate(integrator_flags.face_evaluate);
      integrator_p.evaluate(integrator_flags.face_evaluate);

      this->do_face_integral(integrator_m, integrator_p);

      integrator_m.integrate(integrator_flags.face_integrate);
      integrator_p.integrate(integrator_flags.face_integrate);

      // insert result vector into local matrix u1_v1
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices_m[v](i, j) = integrator_m.begin_dof_values()[i][v];

      // insert result vector into local matrix  u1_v2
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices_p[v](i, j) = integrator_p.begin_dof_values()[i][v];
    }

    // save local matrices into global matrix
    for(unsigned int v = 0; v < n_filled_lanes; v++)
    {
      auto const cell_number_m = matrix_free.get_face_info(face).cells_interior[v];
      auto const cell_number_p = matrix_free.get_face_info(face).cells_exterior[v];

      auto cell_m = matrix_free.get_cell_iterator(cell_number_m / vectorization_length,
                                                  cell_number_m % vectorization_length);
      auto cell_p = matrix_free.get_cell_iterator(cell_number_p / vectorization_length,
                                                  cell_number_p % vectorization_length);

      // get position in global matrix
      std::vector<dealii::types::global_dof_index> dof_indices_m(dofs_per_cell);
      std::vector<dealii::types::global_dof_index> dof_indices_p(dofs_per_cell);
      if(is_mg)
      {
        cell_m->get_mg_dof_indices(dof_indices_m);
        cell_p->get_mg_dof_indices(dof_indices_p);
      }
      else
      {
        cell_m->get_dof_indices(dof_indices_m);
        cell_p->get_dof_indices(dof_indices_p);
      }

      // save M_mm
      constraint_double.distribute_local_to_global(matrices_m[v], dof_indices_m, dst);
      // save M_pm
      constraint_double.distribute_local_to_global(matrices_p[v],
                                                   dof_indices_p,
                                                   dof_indices_m,
                                                   dst);
    }

    // process positive trial function
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of first dealii::FEFaceEvaluation and
      // clear dof values of second dealii::FEFaceEvaluation
      this->create_standard_basis(j, integrator_p, integrator_m);

      integrator_m.evaluate(integrator_flags.face_evaluate);
      integrator_p.evaluate(integrator_flags.face_evaluate);

      this->do_face_integral(integrator_m, integrator_p);

      integrator_m.integrate(integrator_flags.face_integrate);
      integrator_p.integrate(integrator_flags.face_integrate);

      // insert result vector into local matrix M_mp
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices_m[v](i, j) = integrator_m.begin_dof_values()[i][v];

      // insert result vector into local matrix  M_pp
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices_p[v](i, j) = integrator_p.begin_dof_values()[i][v];
    }

    // save local matrices into global matrix
    for(unsigned int v = 0; v < n_filled_lanes; v++)
    {
      auto const cell_number_m = matrix_free.get_face_info(face).cells_interior[v];
      auto const cell_number_p = matrix_free.get_face_info(face).cells_exterior[v];

      auto cell_m = matrix_free.get_cell_iterator(cell_number_m / vectorization_length,
                                                  cell_number_m % vectorization_length);
      auto cell_p = matrix_free.get_cell_iterator(cell_number_p / vectorization_length,
                                                  cell_number_p % vectorization_length);

      // get position in global matrix
      std::vector<dealii::types::global_dof_index> dof_indices_m(dofs_per_cell);
      std::vector<dealii::types::global_dof_index> dof_indices_p(dofs_per_cell);
      if(is_mg)
      {
        cell_m->get_mg_dof_indices(dof_indices_m);
        cell_p->get_mg_dof_indices(dof_indices_p);
      }
      else
      {
        cell_m->get_dof_indices(dof_indices_m);
        cell_p->get_dof_indices(dof_indices_p);
      }

      // save M_mp
      constraint_double.distribute_local_to_global(matrices_m[v],
                                                   dof_indices_m,
                                                   dof_indices_p,
                                                   dst);
      // save M_pp
      constraint_double.distribute_local_to_global(matrices_p[v], dof_indices_p, dst);
    }
  }
}

template<int dim, typename Number, int n_components>
template<typename SparseMatrix>
void
OperatorBase<dim, Number, n_components>::boundary_face_loop_calculate_system_matrix(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  SparseMatrix &                          dst,
  SparseMatrix const &                    src,
  Range const &                           range) const
{
  (void)src;

  IntegratorFace integrator_m =
    IntegratorFace(matrix_free, true, this->data.dof_index, this->data.quad_index);

  unsigned int const dofs_per_cell = integrator_m.dofs_per_cell;

  for(auto face = range.first; face < range.second; ++face)
  {
    unsigned int const n_filled_lanes = matrix_free.n_active_entries_per_face_batch(face);

    // create temporary matrices for local blocks
    FullMatrix_ matrices[vectorization_length];
    std::fill_n(matrices, vectorization_length, FullMatrix_(dofs_per_cell, dofs_per_cell));

    this->reinit_boundary_face(integrator_m, face);

    auto bid = matrix_free.get_boundary_id(face);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, integrator_m);

      integrator_m.evaluate(integrator_flags.face_evaluate);

      this->do_boundary_integral(integrator_m, OperatorType::homogeneous, bid);

      integrator_m.integrate(integrator_flags.face_integrate);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices[v](i, j) = integrator_m.begin_dof_values()[i][v];
    }

    // save local matrices into global matrix
    for(unsigned int v = 0; v < n_filled_lanes; v++)
    {
      unsigned int const cell_number = matrix_free.get_face_info(face).cells_interior[v];

      auto cell_v = matrix_free.get_cell_iterator(cell_number / vectorization_length,
                                                  cell_number % vectorization_length);

      std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);
      if(is_mg)
        cell_v->get_mg_dof_indices(dof_indices);
      else
        cell_v->get_dof_indices(dof_indices);

      constraint_double.distribute_local_to_global(matrices[v], dof_indices, dst);
    }
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::set_constrained_dofs_to_one(VectorType & vector) const
{
  // set (diagonal) entries to 1.0 for constrained dofs
  for(auto i : matrix_free->get_constrained_dofs(this->data.dof_index))
    vector.local_element(i) = 1.0;
}

template<int dim, typename Number, int n_components>
bool
OperatorBase<dim, Number, n_components>::evaluate_face_integrals() const
{
  return (integrator_flags.face_evaluate != dealii::EvaluationFlags::nothing) or
         (integrator_flags.face_integrate != dealii::EvaluationFlags::nothing);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::compute_factorized_additive_schwarz_matrices() const
{
#ifdef DEAL_II_WITH_TRILINOS
  internal_compute_factorized_additive_schwarz_matrices<dealii::TrilinosWrappers::SparseMatrix>();
#elif defined(DEAL_II_WITH_PETSC)
  internal_compute_factorized_additive_schwarz_matrices<dealii::PETScWrappers::MPI::SparseMatrix>();
#else
  AssertThrow(
    n_mpi_processes == 1,
    dealii::ExcMessage(
      "If you want to use this function in parallel you have to compile deal.II with either "
      "Trilinos or Petsc support for distributed sparse matrices."));
  internal_compute_factorized_additive_schwarz_matrices<dealii::SparseMatrix>();
#endif
}

template<int dim, typename Number, int n_components>
template<typename SparseMatrix>
void
OperatorBase<dim, Number, n_components>::internal_compute_factorized_additive_schwarz_matrices()
  const
{
  if(is_dg)
  {
    update_block_diagonal_preconditioner();
  }
  else
  {
    dealii::DoFHandler<dim> const & dof_handler =
      this->matrix_free->get_dof_handler(this->data.dof_index);

    unsigned int const dofs_per_cell = matrix_free->get_dofs_per_cell(this->data.dof_index);

    unsigned int const n_cells = matrix_free->n_cell_batches() * vectorization_length;

    // assemble a temporary sparse matrix to cut out the blocks
    SparseMatrix                   tmp_matrix;
    dealii::DynamicSparsityPattern dsp;
    internal_init_system_matrix(tmp_matrix, dsp, dof_handler.get_communicator());
    internal_calculate_system_matrix(tmp_matrix);

    // collect the DoF indices of all cells
    std::vector<std::vector<dealii::types::global_dof_index>> dof_indices_all_cells(
      n_cells, std::vector<dealii::types::global_dof_index>(dofs_per_cell));
    // and compute weights by counting the contributions to a DoF
    initialize_dof_vector(weights);
    for(unsigned int cell = 0; cell < matrix_free->n_cell_batches(); ++cell)
    {
      unsigned int const n_filled_lanes = matrix_free->n_active_entries_per_cell_batch(cell);

      for(unsigned int v = 0; v < n_filled_lanes; ++v)
      {
        auto const & cell_v = matrix_free->get_cell_iterator(cell, v);

        auto & dof_indices = dof_indices_all_cells[cell * vectorization_length + v];
        if(is_mg)
          cell_v->get_mg_dof_indices(dof_indices);
        else
          cell_v->get_dof_indices(dof_indices);

        for(auto const & i : dof_indices)
          weights[i] += 1.;
      }
    }
    weights.compress(dealii::VectorOperation::add);

    // prepare the weights vector for symmetric weighting
    for(unsigned int i = 0; i < weights.size(); ++i)
    {
      if(weights.in_local_range(i))
        weights[i] = 1. / std::sqrt(weights[i]);
    }
    weights.update_ghost_values();

    // cut out overlapped block matrices
    std::vector<dealii::FullMatrix<Number>> overlapped_cell_matrices(
      n_cells, dealii::FullMatrix<Number>(dofs_per_cell));

    dealii::SparseMatrixTools::restrict_to_full_matrices<SparseMatrix,
                                                         dealii::DynamicSparsityPattern,
                                                         Number>(tmp_matrix,
                                                                 dsp,
                                                                 dof_indices_all_cells,
                                                                 overlapped_cell_matrices);

    // factorize and store cell matrices
    matrices.resize(n_cells, LAPACKMatrix(dofs_per_cell));
    for(unsigned int cell = 0; cell < matrix_free->n_cell_batches(); ++cell)
    {
      unsigned int const n_filled_lanes = matrix_free->n_active_entries_per_cell_batch(cell);

      for(unsigned int v = 0; v < n_filled_lanes; ++v)
      {
        // get overlapped cell matrix
        auto const & overlapped_cell_matrix =
          overlapped_cell_matrices[cell * vectorization_length + v];

        // store the cell matrix and renumber lexicographic
        auto & lapack_matrix = matrices[cell * vectorization_length + v];
        lapack_matrix.reinit(dofs_per_cell);

        auto const & lex_to_hier =
          matrix_free->get_shape_info(this->data.dof_index).lexicographic_numbering;
        for(unsigned int i = 0; i < dofs_per_cell; i++)
          for(unsigned int j = 0; j < dofs_per_cell; j++)
            lapack_matrix.set(i, j, overlapped_cell_matrix[lex_to_hier[i]][lex_to_hier[j]]);

        // factorize the cell matrix
        lapack_matrix.compute_lu_factorization();
      }
    }
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::apply_inverse_additive_schwarz_matrices(
  VectorType &       dst,
  VectorType const & src) const
{
  if(is_dg)
    apply_inverse_block_diagonal(dst, src);
  else
  {
    src.update_ghost_values();

    matrix_free->template cell_loop<VectorType, VectorType>(
      [&](auto const & matrix_free, auto & dst, auto const & src, auto const & cell_range) {
        auto const dofs_per_cell = matrix_free.get_dofs_per_cell(this->data.dof_index);

        IntegratorCell integrator =
          IntegratorCell(matrix_free, this->data.dof_index, this->data.quad_index);

        dealii::Vector<Number>                                 local_vector(dofs_per_cell);
        dealii::AlignedVector<dealii::VectorizedArray<Number>> local_weights_vector(dofs_per_cell);
        for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          this->reinit_cell(integrator, cell);

          integrator.read_dof_values(weights);
          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            local_weights_vector[i] = integrator.begin_dof_values()[i];

          integrator.read_dof_values(src);

          unsigned int const n_filled_lanes = matrix_free.n_active_entries_per_cell_batch(cell);

          for(unsigned int v = 0; v < n_filled_lanes; ++v)
          {
            // apply symmetric weighting, first before applying the inverse
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
              local_vector[i] = integrator.begin_dof_values()[i][v] * local_weights_vector[i][v];

            matrices[cell * vectorization_length + v].solve(local_vector);

            // and after applying the inverse
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
              integrator.begin_dof_values()[i][v] = local_vector[i] * local_weights_vector[i][v];
          }

          integrator.distribute_local_to_global(dst);
        }
      },
      dst,
      src,
      true);

    src.zero_out_ghost_values();
  }
}


template class OperatorBase<2, float, 1>;
template class OperatorBase<2, float, 2>;
template class OperatorBase<2, float, 3>;
template class OperatorBase<2, float, 4>;

template class OperatorBase<2, double, 1>;
template class OperatorBase<2, double, 2>;
template class OperatorBase<2, double, 3>;
template class OperatorBase<2, double, 4>;

template class OperatorBase<3, float, 1>;
template class OperatorBase<3, float, 3>;
template class OperatorBase<3, float, 4>;
template class OperatorBase<3, float, 5>;

template class OperatorBase<3, double, 1>;
template class OperatorBase<3, double, 3>;
template class OperatorBase<3, double, 4>;
template class OperatorBase<3, double, 5>;

} // namespace ExaDG
