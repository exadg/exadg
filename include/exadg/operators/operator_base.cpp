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
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/matrix_free/tools.h>

// ExaDG
#include <exadg/operators/operator_base.h>
#include <exadg/solvers_and_preconditioners/utilities/block_jacobi_matrices.h>
#include <exadg/solvers_and_preconditioners/utilities/invert_diagonal.h>
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
    block_diagonal_preconditioner_is_initialized(false),
    n_mpi_processes(0)
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

  integrator   = std::make_shared<IntegratorCell>(*this->matrix_free,
                                                this->data.dof_index,
                                                this->data.quad_index);
  integrator_m = std::make_shared<IntegratorFace>(*this->matrix_free,
                                                  true,
                                                  this->data.dof_index,
                                                  this->data.quad_index);
  integrator_p = std::make_shared<IntegratorFace>(*this->matrix_free,
                                                  false,
                                                  this->data.dof_index,
                                                  this->data.quad_index);

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
  this->apply(dst, src);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::vmult_add(VectorType & dst, VectorType const & src) const
{
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
  unsigned int dof_index = get_dof_index();

  this->matrix_free->initialize_dof_vector(vector, dof_index);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::set_constrained_values_to_zero(VectorType & vector) const
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
    verify_calculation_of_diagonal(
      *this, diagonal, matrix_free->get_dof_handler(this->data.dof_index).get_communicator());

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
    // data passed to MatrixFree via AffineConstraints.
    matrix_free->cell_loop(&This::cell_loop, this, dst, src, true);

    // Constrained degree of freedom are not removed from the system of equations.
    // Instead, we set the diagonal entries of the matrix to 1 for these constrained
    // degrees of freedom. This means that we simply copy the constrained values to the
    // dst vector.
    for(unsigned int const constrained_index :
        matrix_free->get_constrained_dofs(this->data.dof_index))
      dst.local_element(constrained_index) = src.local_element(constrained_index);
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
      dst.local_element(constrained_index) += src.local_element(constrained_index);
  }
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
  VectorType tmp;
  tmp.reinit(rhs, false);

  matrix_free->loop(&This::cell_loop_empty,
                    &This::face_loop_empty,
                    &This::boundary_face_loop_inhom_operator,
                    this,
                    tmp,
                    tmp);

  // multiply by -1.0 since the boundary face integrals have to be shifted to the right hand side
  rhs.add(-1.0, tmp);

  if(not(is_dg))
  {
    // Set constrained degrees of freedom according to Dirichlet boundary conditions. The rest of
    // the vector contains zeros.
    VectorType temp1;
    matrix_free->initialize_dof_vector(temp1, data.dof_index);
    set_constrained_values(temp1, time);

    // Perform matrix-vector product using read_dof_values_plain() with temp1 as src-vector to
    // obtain the inhomogeneous action of the operator. Subsequently, we shift the resulting vector
    // to the right-hand side. Note that constrained degrees of freedom in the dst-vector will not
    // be touched and remain zero. Hence, the constrained degrees of freedom in the rhs-vector will
    // not be changed.
    VectorType temp2;
    matrix_free->initialize_dof_vector(temp2, data.dof_index);
    matrix_free->cell_loop(&This::cell_loop_dbc, this, temp2, temp1);
    rhs -= temp2;
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
  matrix_free->loop(
    &This::cell_loop, &This::face_loop, &This::boundary_face_loop_full_operator, this, dst, src);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::calculate_diagonal(VectorType & diagonal) const
{
  if(diagonal.size() == 0)
    matrix_free->initialize_dof_vector(diagonal);
  diagonal = 0;
  add_diagonal(diagonal);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::add_diagonal(VectorType & diagonal) const
{
  // compute diagonal
  if(is_dg && evaluate_face_integrals())
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
          // TODO this line is currently needed as bugfix, but should be
          // removed because reinit is now done twice
          this->reinit_cell(integrator.get_current_cell_index());

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
OperatorBase<dim, Number, n_components>::calculate_block_diagonal_matrices() const
{
  AssertThrow(is_dg, dealii::ExcMessage("Block Jacobi only implemented for DG!"));

  // allocate memory only the first time
  if(!block_diagonal_preconditioner_is_initialized ||
     matrix_free->n_cell_batches() * vectorization_length != matrices.size())
  {
    auto dofs =
      matrix_free->get_shape_info(this->data.dof_index).dofs_per_component_on_cell * n_components;

    matrices.resize(matrix_free->n_cell_batches() * vectorization_length,
                    dealii::LAPACKFullMatrix<Number>(dofs, dofs));

    block_diagonal_preconditioner_is_initialized = true;
  }
  // else: reuse old memory

  // clear matrices
  initialize_block_jacobi_matrices_with_zero(matrices);

  // compute block matrices
  add_block_diagonal_matrices(matrices);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::add_block_diagonal_matrices(BlockMatrix & matrices) const
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
OperatorBase<dim, Number, n_components>::apply_block_diagonal_matrix_based(
  VectorType &       dst,
  VectorType const & src) const
{
  AssertThrow(is_dg, dealii::ExcMessage("Block Jacobi only implemented for DG!"));
  AssertThrow(block_diagonal_preconditioner_is_initialized,
              dealii::ExcMessage("Block Jacobi matrices have not been initialized!"));

  matrix_free->cell_loop(&This::cell_loop_apply_block_diagonal_matrix_based, this, dst, src);
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
  AssertThrow(block_diagonal_preconditioner_is_initialized,
              dealii::ExcMessage("Block Jacobi matrices have not been initialized!"));

  matrix_free->cell_loop(&This::cell_loop_apply_inverse_block_diagonal_matrix_based,
                         this,
                         dst,
                         src);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::initialize_block_diagonal_preconditioner_matrix_free()
  const
{
  elementwise_operator = std::make_shared<ELEMENTWISE_OPERATOR>(*this);

  if(data.preconditioner_block_diagonal == Elementwise::Preconditioner::None)
  {
    typedef Elementwise::PreconditionerIdentity<dealii::VectorizedArray<Number>> IDENTITY;

    elementwise_preconditioner =
      std::make_shared<IDENTITY>(elementwise_operator->get_problem_size());
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
OperatorBase<dim, Number, n_components>::apply_add_block_diagonal_elementwise(
  unsigned int const                            cell,
  dealii::VectorizedArray<Number> * const       dst,
  dealii::VectorizedArray<Number> const * const src,
  unsigned int const                            problem_size) const
{
  (void)problem_size;

  AssertThrow(is_dg, dealii::ExcMessage("Block Jacobi only implemented for DG!"));

  this->reinit_cell(cell);

  for(unsigned int i = 0; i < integrator->dofs_per_cell; ++i)
    integrator->begin_dof_values()[i] = src[i];

  integrator->evaluate(integrator_flags.cell_evaluate);

  this->do_cell_integral(*integrator);

  integrator->integrate(integrator_flags.cell_integrate);

  for(unsigned int i = 0; i < integrator->dofs_per_cell; ++i)
    dst[i] += integrator->begin_dof_values()[i];

  if(is_dg && evaluate_face_integrals())
  {
    // face integrals
    unsigned int const n_faces = dealii::ReferenceCells::template get_hypercube<dim>().n_faces();
    for(unsigned int face = 0; face < n_faces; ++face)
    {
      auto bids = (*matrix_free).get_faces_by_cells_boundary_id(cell, face);
      auto bid  = bids[0];

      this->reinit_face_cell_based(cell, face, bid);

      for(unsigned int i = 0; i < integrator_m->dofs_per_cell; ++i)
        integrator_m->begin_dof_values()[i] = src[i];

      // no need to read dof values for integrator_p (already initialized with 0)

      integrator_m->evaluate(integrator_flags.face_evaluate);

      if(bid == dealii::numbers::internal_face_boundary_id) // internal face
      {
        this->do_face_int_integral_cell_based(*integrator_m, *integrator_p);
      }
      else // boundary face
      {
        this->do_boundary_integral(*integrator_m, OperatorType::homogeneous, bid);
      }

      integrator_m->integrate(integrator_flags.face_integrate);

      for(unsigned int i = 0; i < integrator_m->dofs_per_cell; ++i)
        dst[i] += integrator_m->begin_dof_values()[i];
    }
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::update_block_diagonal_preconditioner() const
{
  AssertThrow(is_dg, dealii::ExcMessage("Block Jacobi only implemented for DG!"));

  // initialization

  if(!block_diagonal_preconditioner_is_initialized)
  {
    if(data.implement_block_diagonal_preconditioner_matrix_free)
    {
      initialize_block_diagonal_preconditioner_matrix_free();
    }
    else // matrix-based variant
    {
      // allocate memory only the first time
      auto dofs =
        matrix_free->get_shape_info(this->data.dof_index).dofs_per_component_on_cell * n_components;
      matrices.resize(matrix_free->n_cell_batches() * vectorization_length,
                      dealii::LAPACKFullMatrix<Number>(dofs, dofs));
    }

    block_diagonal_preconditioner_is_initialized = true;
  }

  // update

  // For the matrix-free variant there is nothing to do.
  // For the matrix-based variant we have to recompute the block matrices.
  if(!data.implement_block_diagonal_preconditioner_matrix_free)
  {
    // clear matrices
    initialize_block_jacobi_matrices_with_zero(matrices);

    // compute block matrices and add
    add_block_diagonal_matrices(matrices);

    calculate_lu_factorization_block_jacobi(matrices);
  }
}

#ifdef DEAL_II_WITH_TRILINOS
template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::init_system_matrix(
  dealii::TrilinosWrappers::SparseMatrix & system_matrix,
  MPI_Comm const &                         mpi_comm) const
{
  internal_init_system_matrix(system_matrix, mpi_comm);
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
  internal_init_system_matrix(system_matrix, mpi_comm);
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
  SparseMatrix &   system_matrix,
  MPI_Comm const & mpi_comm) const
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
  AssertThrow(my_rank_is_part_of_subcommunicator || owned_dofs.n_elements() == 0,
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
  dealii::DynamicSparsityPattern dsp(relevant_dofs);

  if(is_dg && is_mg)
    dealii::MGTools::make_flux_sparsity_pattern(dof_handler, dsp, this->level);
  else if(is_dg && !is_mg)
    dealii::DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
  else if(/*!is_dg &&*/ is_mg)
    dealii::MGTools::make_sparsity_pattern(dof_handler, dsp, this->level, *this->constraint);
  else /* if (!is_dg && !is_mg)*/
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
  if(evaluate_face_integrals() && is_dg)
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
OperatorBase<dim, Number, n_components>::reinit_cell(unsigned int const cell) const
{
  integrator->reinit(cell);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::reinit_face(unsigned int const face) const
{
  integrator_m->reinit(face);
  integrator_p->reinit(face);
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::reinit_boundary_face(unsigned int const face) const
{
  integrator_m->reinit(face);
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
      "OperatorBase::do_boundary_integral_continuous() has to be overwritten by derived class!"));
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::set_constrained_values(VectorType & solution,
                                                                double const time) const
{
  (void)solution;
  (void)time;

  AssertThrow(false,
              dealii::ExcMessage(
                "OperatorBase::set_constrained_values() has to be overwritten by derived class!"));
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
  unsigned int const               cell,
  unsigned int const               face,
  dealii::types::boundary_id const boundary_id) const
{
  integrator_m->reinit(cell, face);

  if(boundary_id == dealii::numbers::internal_face_boundary_id) // internal face
  {
    integrator_p->reinit(cell, face);
  }
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
OperatorBase<dim, Number, n_components>::cell_loop_dbc(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  (void)matrix_free;

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    this->reinit_cell(cell);

    integrator->read_dof_values_plain(src);

    integrator->evaluate(integrator_flags.cell_evaluate);

    this->do_cell_integral(*integrator);

    integrator->integrate_scatter(integrator_flags.cell_integrate, dst);
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
  (void)matrix_free;

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    this->reinit_cell(cell);

    integrator->gather_evaluate(src, integrator_flags.cell_evaluate);

    this->do_cell_integral(*integrator);

    integrator->integrate_scatter(integrator_flags.cell_integrate, dst);
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
  (void)matrix_free;

  for(auto face = range.first; face < range.second; ++face)
  {
    this->reinit_face(face);

    integrator_m->gather_evaluate(src, integrator_flags.face_evaluate);
    integrator_p->gather_evaluate(src, integrator_flags.face_evaluate);

    this->do_face_integral(*integrator_m, *integrator_p);

    integrator_m->integrate_scatter(integrator_flags.face_integrate, dst);
    integrator_p->integrate_scatter(integrator_flags.face_integrate, dst);
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
  for(unsigned int face = range.first; face < range.second; face++)
  {
    this->reinit_boundary_face(face);

    integrator_m->gather_evaluate(src, integrator_flags.face_evaluate);

    do_boundary_integral(*integrator_m,
                         OperatorType::homogeneous,
                         matrix_free.get_boundary_id(face));

    integrator_m->integrate_scatter(integrator_flags.face_integrate, dst);
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
    for(unsigned int face = range.first; face < range.second; face++)
    {
      this->reinit_boundary_face(face);

      // note: no gathering/evaluation is necessary when calculating the
      //       inhomogeneous part of boundary face integrals

      do_boundary_integral(*integrator_m,
                           OperatorType::inhomogeneous,
                           matrix_free.get_boundary_id(face));

      integrator_m->integrate_scatter(integrator_flags.face_integrate, dst);
    }
  }
  else // continuous FE discretization (e.g., apply Neumann BCs)
  {
    for(unsigned int face = range.first; face < range.second; face++)
    {
      this->reinit_boundary_face(face);

      // note: no gathering/evaluation is necessary when calculating the
      //       inhomogeneous part of boundary face integrals

      do_boundary_integral_continuous(*integrator_m, matrix_free.get_boundary_id(face));

      integrator_m->integrate_scatter(integrator_flags.face_integrate, dst);
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
  for(unsigned int face = range.first; face < range.second; face++)
  {
    this->reinit_boundary_face(face);

    integrator_m->gather_evaluate(src, integrator_flags.face_evaluate);

    do_boundary_integral(*integrator_m, OperatorType::full, matrix_free.get_boundary_id(face));

    integrator_m->integrate_scatter(integrator_flags.face_integrate, dst);
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
  (void)matrix_free;
  (void)src;

  // create temporal array for local diagonal
  unsigned int const                                     dofs_per_cell = integrator->dofs_per_cell;
  dealii::AlignedVector<dealii::VectorizedArray<Number>> local_diag(dofs_per_cell);

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    this->reinit_cell(cell);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of dealii::FEEvaluation
      this->create_standard_basis(j, *integrator);

      integrator->evaluate(integrator_flags.cell_evaluate);

      this->do_cell_integral(*integrator);

      integrator->integrate(integrator_flags.cell_integrate);

      // extract single value from result vector and temporally store it
      local_diag[j] = integrator->begin_dof_values()[j];
    }
    // copy local diagonal entries into dof values of dealii::FEEvaluation ...
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      integrator->begin_dof_values()[j] = local_diag[j];

    // ... and write it back to global vector
    integrator->distribute_local_to_global(dst);
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
  (void)matrix_free;
  (void)src;

  // create temporal array for local diagonal
  unsigned int const dofs_per_cell = integrator_m->dofs_per_cell;
  dealii::AlignedVector<dealii::VectorizedArray<Number>> local_diag(dofs_per_cell);

  for(auto face = range.first; face < range.second; ++face)
  {
    this->reinit_face(face);

    // interior face
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, *integrator_m);

      integrator_m->evaluate(integrator_flags.face_evaluate);

      this->do_face_int_integral(*integrator_m, *integrator_p);

      integrator_m->integrate(integrator_flags.face_integrate);

      local_diag[j] = integrator_m->begin_dof_values()[j];
    }

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      integrator_m->begin_dof_values()[j] = local_diag[j];

    integrator_m->distribute_local_to_global(dst);

    // exterior face
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, *integrator_p);

      integrator_p->evaluate(integrator_flags.face_evaluate);

      this->do_face_ext_integral(*integrator_m, *integrator_p);

      integrator_p->integrate(integrator_flags.face_integrate);

      local_diag[j] = integrator_p->begin_dof_values()[j];
    }

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      integrator_p->begin_dof_values()[j] = local_diag[j];

    integrator_p->distribute_local_to_global(dst);
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

  // create temporal array for local diagonal
  unsigned int const dofs_per_cell = integrator_m->dofs_per_cell;
  dealii::AlignedVector<dealii::VectorizedArray<Number>> local_diag(dofs_per_cell);

  for(unsigned int face = range.first; face < range.second; face++)
  {
    auto bid = matrix_free.get_boundary_id(face);

    this->reinit_boundary_face(face);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, *integrator_m);

      integrator_m->evaluate(integrator_flags.face_evaluate);

      this->do_boundary_integral(*integrator_m, OperatorType::homogeneous, bid);

      integrator_m->integrate(integrator_flags.face_integrate);

      local_diag[j] = integrator_m->begin_dof_values()[j];
    }

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      integrator_m->begin_dof_values()[j] = local_diag[j];

    integrator_m->distribute_local_to_global(dst);
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

  // create temporal array for local diagonal
  unsigned int const                                     dofs_per_cell = integrator->dofs_per_cell;
  dealii::AlignedVector<dealii::VectorizedArray<Number>> local_diag(dofs_per_cell);

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    this->reinit_cell(cell);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, *integrator);

      integrator->evaluate(integrator_flags.cell_evaluate);

      this->do_cell_integral(*integrator);

      integrator->integrate(integrator_flags.cell_integrate);

      local_diag[j] = integrator->begin_dof_values()[j];
    }

    // loop over all faces and gather results into local diagonal local_diag
    unsigned int const n_faces = dealii::ReferenceCells::template get_hypercube<dim>().n_faces();
    for(unsigned int face = 0; face < n_faces; ++face)
    {
      auto bids = matrix_free.get_faces_by_cells_boundary_id(cell, face);
      auto bid  = bids[0];

      this->reinit_face_cell_based(cell, face, bid);

#ifdef DEBUG
      unsigned int const n_filled_lanes = matrix_free.n_active_entries_per_cell_batch(cell);
      for(unsigned int v = 0; v < n_filled_lanes; v++)
        Assert(bid == bids[v],
               dealii::ExcMessage(
                 "Cell-based face loop encountered face batch with different bids."));
#endif

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        this->create_standard_basis(j, *integrator_m);

        integrator_m->evaluate(integrator_flags.face_evaluate);

        if(bid == dealii::numbers::internal_face_boundary_id) // internal face
        {
          this->do_face_int_integral_cell_based(*integrator_m, *integrator_p);
        }
        else // boundary face
        {
          this->do_boundary_integral(*integrator_m, OperatorType::homogeneous, bid);
        }

        integrator_m->integrate(integrator_flags.face_integrate);

        // note: += for accumulation of all contributions of this (macro) cell
        //          including: cell-, face-, boundary-stiffness matrix
        local_diag[j] += integrator_m->begin_dof_values()[j];
      }
    }

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      integrator->begin_dof_values()[j] = local_diag[j];

    integrator->distribute_local_to_global(dst);
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
  (void)matrix_free;

  unsigned int const dofs_per_cell = integrator->dofs_per_cell;

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    this->reinit_cell(cell);

    integrator->read_dof_values(src);

    for(unsigned int v = 0; v < vectorization_length; ++v)
    {
      dealii::Vector<Number> src_vector(dofs_per_cell);
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        src_vector(j) = integrator->begin_dof_values()[j][v];

      // apply inverse matrix
      matrices[cell * vectorization_length + v].solve(src_vector, false);

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        integrator->begin_dof_values()[j][v] = src_vector(j);
    }

    integrator->set_dof_values(dst);
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::cell_loop_apply_block_diagonal_matrix_based(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  (void)matrix_free;

  unsigned int const dofs_per_cell = integrator->dofs_per_cell;

  for(unsigned int cell = range.first; cell < range.second; ++cell)
  {
    this->reinit_cell(cell);

    integrator->read_dof_values(src);

    for(unsigned int v = 0; v < vectorization_length; ++v)
    {
      dealii::Vector<Number> src_vector(dofs_per_cell);
      dealii::Vector<Number> dst_vector(dofs_per_cell);
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        src_vector(j) = integrator->begin_dof_values()[j][v];

      // apply matrix
      matrices[cell * vectorization_length + v].vmult(dst_vector, src_vector, false);

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        integrator->begin_dof_values()[j][v] = dst_vector(j);
    }

    integrator->set_dof_values(dst);
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::cell_loop_block_diagonal(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  BlockMatrix &                           matrices,
  BlockMatrix const &,
  Range const & range) const
{
  unsigned int const dofs_per_cell = integrator->dofs_per_cell;

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    unsigned int const n_filled_lanes = matrix_free.n_active_entries_per_cell_batch(cell);

    this->reinit_cell(cell);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, *integrator);

      integrator->evaluate(integrator_flags.cell_evaluate);

      this->do_cell_integral(*integrator);

      integrator->integrate(integrator_flags.cell_integrate);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices[cell * vectorization_length + v](i, j) += integrator->begin_dof_values()[i][v];
    }
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::face_loop_block_diagonal(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  BlockMatrix &                           matrices,
  BlockMatrix const &,
  Range const & range) const
{
  unsigned int const dofs_per_cell = integrator_m->dofs_per_cell;

  for(auto face = range.first; face < range.second; ++face)
  {
    unsigned int const n_filled_lanes = matrix_free.n_active_entries_per_face_batch(face);

    this->reinit_face(face);

    // interior face
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, *integrator_m);

      integrator_m->evaluate(integrator_flags.face_evaluate);

      this->do_face_int_integral(*integrator_m, *integrator_p);

      integrator_m->integrate(integrator_flags.face_integrate);

      for(unsigned int v = 0; v < n_filled_lanes; ++v)
      {
        unsigned int const cell = matrix_free.get_face_info(face).cells_interior[v];
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          matrices[cell](i, j) += integrator_m->begin_dof_values()[i][v];
      }
    }

    // exterior face
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, *integrator_p);

      integrator_p->evaluate(integrator_flags.face_evaluate);

      this->do_face_ext_integral(*integrator_m, *integrator_p);

      integrator_p->integrate(integrator_flags.face_integrate);

      for(unsigned int v = 0; v < n_filled_lanes; ++v)
      {
        unsigned int const cell = matrix_free.get_face_info(face).cells_exterior[v];
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          matrices[cell](i, j) += integrator_p->begin_dof_values()[i][v];
      }
    }
  }
}

template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::boundary_face_loop_block_diagonal(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  BlockMatrix &                           matrices,
  BlockMatrix const &,
  Range const & range) const
{
  unsigned int const dofs_per_cell = integrator_m->dofs_per_cell;

  for(auto face = range.first; face < range.second; ++face)
  {
    unsigned int const n_filled_lanes = matrix_free.n_active_entries_per_face_batch(face);

    this->reinit_boundary_face(face);

    auto bid = matrix_free.get_boundary_id(face);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, *integrator_m);

      integrator_m->evaluate(integrator_flags.face_evaluate);

      this->do_boundary_integral(*integrator_m, OperatorType::homogeneous, bid);

      integrator_m->integrate(integrator_flags.face_integrate);

      for(unsigned int v = 0; v < n_filled_lanes; ++v)
      {
        unsigned int const cell = matrix_free.get_face_info(face).cells_interior[v];
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          matrices[cell](i, j) += integrator_m->begin_dof_values()[i][v];
      }
    }
  }
}


template<int dim, typename Number, int n_components>
void
OperatorBase<dim, Number, n_components>::cell_based_loop_block_diagonal(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  BlockMatrix &                           matrices,
  BlockMatrix const &,
  Range const & range) const
{
  unsigned int const dofs_per_cell = integrator->dofs_per_cell;

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    unsigned int const n_filled_lanes = matrix_free.n_active_entries_per_cell_batch(cell);

    this->reinit_cell(cell);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, *integrator);

      integrator->evaluate(integrator_flags.cell_evaluate);

      this->do_cell_integral(*integrator);

      integrator->integrate(integrator_flags.cell_integrate);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices[cell * vectorization_length + v](i, j) += integrator->begin_dof_values()[i][v];
    }

    // loop over all faces
    unsigned int const n_faces = dealii::ReferenceCells::template get_hypercube<dim>().n_faces();
    for(unsigned int face = 0; face < n_faces; ++face)
    {
      auto bids = matrix_free.get_faces_by_cells_boundary_id(cell, face);
      auto bid  = bids[0];

      this->reinit_face_cell_based(cell, face, bid);

#ifdef DEBUG
      for(unsigned int v = 0; v < n_filled_lanes; v++)
        Assert(bid == bids[v],
               dealii::ExcMessage(
                 "Cell-based face loop encountered face batch with different bids."));
#endif

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        this->create_standard_basis(j, *integrator_m);

        integrator_m->evaluate(integrator_flags.face_evaluate);

        if(bid == dealii::numbers::internal_face_boundary_id) // internal face
        {
          this->do_face_int_integral_cell_based(*integrator_m, *integrator_p);
        }
        else // boundary face
        {
          this->do_boundary_integral(*integrator_m, OperatorType::homogeneous, bid);
        }

        integrator_m->integrate(integrator_flags.face_integrate);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int v = 0; v < n_filled_lanes; ++v)
            matrices[cell * vectorization_length + v](i, j) +=
              integrator_m->begin_dof_values()[i][v];
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

  unsigned int const dofs_per_cell = integrator->dofs_per_cell;

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    unsigned int const n_filled_lanes = matrix_free.n_active_entries_per_cell_batch(cell);

    // create a temporal full matrix for the local element matrix of each ...
    // cell of each macro cell and ...
    FullMatrix_ matrices[vectorization_length];
    // set their size
    std::fill_n(matrices, vectorization_length, FullMatrix_(dofs_per_cell, dofs_per_cell));

    this->reinit_cell(cell);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, *integrator);

      integrator->evaluate(integrator_flags.cell_evaluate);

      this->do_cell_integral(*integrator);

      integrator->integrate(integrator_flags.cell_integrate);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices[v](i, j) = integrator->begin_dof_values()[i][v];
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

      if(!is_dg)
      {
        // in the case of CG: shape functions are not ordered lexicographically
        // see (https://www.dealii.org/8.5.1/doxygen/deal.II/classFE__Q.html)
        // so we have to fix the order
        auto temp = dof_indices;
        for(unsigned int j = 0; j < dof_indices.size(); j++)
          dof_indices[j] = temp[matrix_free.get_shape_info().lexicographic_numbering[j]];
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

  unsigned int const dofs_per_cell = integrator_m->dofs_per_cell;

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

    this->reinit_face(face);

    // process minus trial function
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of first dealii::FEFaceEvaluation and
      // clear dof values of second dealii::FEFaceEvaluation
      this->create_standard_basis(j, *integrator_m, *integrator_p);

      integrator_m->evaluate(integrator_flags.face_evaluate);
      integrator_p->evaluate(integrator_flags.face_evaluate);

      this->do_face_integral(*integrator_m, *integrator_p);

      integrator_m->integrate(integrator_flags.face_integrate);
      integrator_p->integrate(integrator_flags.face_integrate);

      // insert result vector into local matrix u1_v1
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices_m[v](i, j) = integrator_m->begin_dof_values()[i][v];

      // insert result vector into local matrix  u1_v2
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices_p[v](i, j) = integrator_p->begin_dof_values()[i][v];
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
      this->create_standard_basis(j, *integrator_p, *integrator_m);

      integrator_m->evaluate(integrator_flags.face_evaluate);
      integrator_p->evaluate(integrator_flags.face_evaluate);

      this->do_face_integral(*integrator_m, *integrator_p);

      integrator_m->integrate(integrator_flags.face_integrate);
      integrator_p->integrate(integrator_flags.face_integrate);

      // insert result vector into local matrix M_mp
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices_m[v](i, j) = integrator_m->begin_dof_values()[i][v];

      // insert result vector into local matrix  M_pp
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices_p[v](i, j) = integrator_p->begin_dof_values()[i][v];
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

  unsigned int const dofs_per_cell = integrator_m->dofs_per_cell;

  for(auto face = range.first; face < range.second; ++face)
  {
    unsigned int const n_filled_lanes = matrix_free.n_active_entries_per_face_batch(face);

    // create temporary matrices for local blocks
    FullMatrix_ matrices[vectorization_length];
    std::fill_n(matrices, vectorization_length, FullMatrix_(dofs_per_cell, dofs_per_cell));

    this->reinit_boundary_face(face);

    auto bid = matrix_free.get_boundary_id(face);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, *integrator_m);

      integrator_m->evaluate(integrator_flags.face_evaluate);

      this->do_boundary_integral(*integrator_m, OperatorType::homogeneous, bid);

      integrator_m->integrate(integrator_flags.face_integrate);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices[v](i, j) = integrator_m->begin_dof_values()[i][v];
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
OperatorBase<dim, Number, n_components>::set_constraint_diagonal(VectorType & diagonal) const
{
  // set (diagonal) entries to 1.0 for constrained dofs
  for(auto i : matrix_free->get_constrained_dofs(this->data.dof_index))
    diagonal.local_element(i) = 1.0;
}

template<int dim, typename Number, int n_components>
bool
OperatorBase<dim, Number, n_components>::evaluate_face_integrals() const
{
  return (integrator_flags.face_evaluate != dealii::EvaluationFlags::nothing) ||
         (integrator_flags.face_integrate != dealii::EvaluationFlags::nothing);
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
