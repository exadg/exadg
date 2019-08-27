#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/multigrid/mg_tools.h>

#include "../solvers_and_preconditioners/util/block_jacobi_matrices.h"
#include "../solvers_and_preconditioners/util/invert_diagonal.h"
#include "../solvers_and_preconditioners/util/verify_calculation_of_diagonal.h"

#include "../functionalities/categorization.h"
#include "../functionalities/constraints.h"
#include "../functionalities/set_zero_mean_value.h"
#include "operator_base.h"

template<int dim, typename Number, typename AdditionalData, int n_components>
OperatorBase<dim, Number, AdditionalData, n_components>::OperatorBase()
  : dealii::Subscriptor(),
    data(AdditionalData()),
    matrix_free(),
    time(0.0),
    is_mg(false),
    is_dg(true),
    level(numbers::invalid_unsigned_int),
    block_diagonal_preconditioner_is_initialized(false),
    n_mpi_processes(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
{
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::reinit(
  MatrixFree<dim, Number> const &   matrix_free,
  AffineConstraints<double> const & constraint_matrix,
  AdditionalData const &            operator_data)
{
  // reinit data structures
  this->matrix_free.reset(matrix_free);
  this->constraint.reset(constraint_matrix);
  this->data = operator_data;

  // TODO: do it somewhere else where we have access to periodic_face_pairs
  //  // verify boundary conditions
  //  if(this->data.evaluate_face_integrals)
  //  {
  //    this->verify_boundary_conditions(this->matrix_free->get_dof_handler(this->data.dof_index),
  //                                     this->data);
  //  }

  // check if DG or CG
  // An approximation can have degrees of freedom on vertices, edges, quads and
  // hexes. A vertex degree of freedom means that the degree of freedom is
  // the same on all cells that are adjacent to this vertex. A face degree of
  // freedom means that the two cells adjacent to that face see the same degree
  // of freedom. A DG element does not share any degrees of freedom over a
  // vertex but has all of them in the last item, i.e., quads in 2D and hexes
  // in 3D, and thus necessarily has dofs_per_vertex=0
  is_dg = (this->matrix_free->get_dof_handler(this->data.dof_index).get_fe().dofs_per_vertex == 0);

  integrator.reset(
    new IntegratorCell(*this->matrix_free, this->data.dof_index, this->data.quad_index));
  integrator_m.reset(
    new IntegratorFace(*this->matrix_free, true, this->data.dof_index, this->data.quad_index));
  integrator_p.reset(
    new IntegratorFace(*this->matrix_free, false, this->data.dof_index, this->data.quad_index));

  if(!is_dg)
  {
    constrained_indices.clear();
    for(auto i : this->matrix_free->get_constrained_dofs())
      constrained_indices.push_back(i);
    constrained_values.resize(constrained_indices.size());
  }

  // set multigrid level
  this->level = this->matrix_free->get_level_mg_handler();

  // The default value is is_mg = false and this variable is set to true in case
  // the operator is applied in multigrid algorithm. By convention, the default
  // argument numbers::invalid_unsigned_int corresponds to the default
  // value is_mg = false
  this->is_mg = (this->level != numbers::invalid_unsigned_int);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
AdditionalData const &
OperatorBase<dim, Number, AdditionalData, n_components>::get_data() const
{
  return data;
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::set_time(double const t) const
{
  this->time = t;
}

template<int dim, typename Number, typename AdditionalData, int n_components>
double
OperatorBase<dim, Number, AdditionalData, n_components>::get_time() const
{
  return time;
}

template<int dim, typename Number, typename AdditionalData, int n_components>
unsigned int
OperatorBase<dim, Number, AdditionalData, n_components>::get_level() const
{
  return level;
}

template<int dim, typename Number, typename AdditionalData, int n_components>
AffineConstraints<double> const &
OperatorBase<dim, Number, AdditionalData, n_components>::get_constraint_matrix() const
{
  return *constraint;
}

template<int dim, typename Number, typename AdditionalData, int n_components>
MatrixFree<dim, Number> const &
OperatorBase<dim, Number, AdditionalData, n_components>::get_matrix_free() const
{
  return *this->matrix_free;
}

template<int dim, typename Number, typename AdditionalData, int n_components>
unsigned int
OperatorBase<dim, Number, AdditionalData, n_components>::get_dof_index() const
{
  return this->data.dof_index;
}

template<int dim, typename Number, typename AdditionalData, int n_components>
unsigned int
OperatorBase<dim, Number, AdditionalData, n_components>::get_quad_index() const
{
  return this->data.quad_index;
}

template<int dim, typename Number, typename AdditionalData, int n_components>
bool
OperatorBase<dim, Number, AdditionalData, n_components>::operator_is_singular() const
{
  return this->data.operator_is_singular;
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::vmult(VectorType &       dst,
                                                               VectorType const & src) const
{
  this->apply(dst, src);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::vmult_add(VectorType &       dst,
                                                                   VectorType const & src) const
{
  this->apply_add(dst, src);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::vmult_interface_down(
  VectorType &       dst,
  VectorType const & src) const
{
  vmult(dst, src);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::vmult_add_interface_up(
  VectorType &       dst,
  VectorType const & src) const
{
  vmult_add(dst, src);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
types::global_dof_index
OperatorBase<dim, Number, AdditionalData, n_components>::m() const
{
  return n();
}

template<int dim, typename Number, typename AdditionalData, int n_components>
types::global_dof_index
OperatorBase<dim, Number, AdditionalData, n_components>::n() const
{
  unsigned int dof_index = get_dof_index();

  return this->matrix_free->get_vector_partitioner(dof_index)->size();
}

template<int dim, typename Number, typename AdditionalData, int n_components>
Number
OperatorBase<dim, Number, AdditionalData, n_components>::el(const unsigned int,
                                                            const unsigned int) const
{
  AssertThrow(false, ExcMessage("Matrix-free does not allow for entry access"));
  return Number();
}

template<int dim, typename Number, typename AdditionalData, int n_components>
bool
OperatorBase<dim, Number, AdditionalData, n_components>::is_empty_locally() const
{
  return (this->matrix_free->n_macro_cells() == 0);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::initialize_dof_vector(
  VectorType & vector) const
{
  unsigned int dof_index = get_dof_index();

  this->matrix_free->initialize_dof_vector(vector, dof_index);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::calculate_inverse_diagonal(
  VectorType & diagonal) const
{
  this->calculate_diagonal(diagonal);

  //  verify_calculation_of_diagonal(*this,diagonal);

  invert_diagonal(diagonal);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::apply(VectorType &       dst,
                                                               VectorType const & src) const
{
  dst = 0;
  apply_add(dst, src);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::apply_add(VectorType &       dst,
                                                                   VectorType const & src) const
{
  if(is_dg && evaluate_face_integrals())
  {
    matrix_free->loop(
      &This::cell_loop, &This::face_loop, &This::boundary_face_loop_hom_operator, this, dst, src);
  }
  else
  {
    for(unsigned int i = 0; i < constrained_indices.size(); ++i)
    {
      constrained_values[i] = std::pair<Number, Number>(src.local_element(constrained_indices[i]),
                                                        dst.local_element(constrained_indices[i]));

      const_cast<VectorType &>(src).local_element(constrained_indices[i]) = 0.;
    }

    matrix_free->cell_loop(&This::cell_loop, this, dst, src);

    for(unsigned int i = 0; i < constrained_indices.size(); ++i)
    {
      const_cast<VectorType &>(src).local_element(constrained_indices[i]) =
        constrained_values[i].first;
      dst.local_element(constrained_indices[i]) =
        constrained_values[i].second + constrained_values[i].first;
    }
  }
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::rhs(VectorType & dst) const
{
  dst = 0;
  this->rhs_add(dst);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::rhs_add(VectorType & dst) const
{
  AssertThrow(is_dg == true,
              ExcMessage("Inhomogeneous BCs are not implemented for continuous elements."));

  VectorType tmp;
  tmp.reinit(dst, false);

  matrix_free->loop(&This::cell_loop_empty,
                    &This::face_loop_empty,
                    &This::boundary_face_loop_inhom_operator,
                    this,
                    tmp,
                    tmp);

  // multiply by -1.0 since the boundary face integrals have to be shifted to the right hand side
  dst.add(-1.0, tmp);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::evaluate(VectorType &       dst,
                                                                  VectorType const & src) const
{
  dst = 0;
  evaluate_add(dst, src);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::evaluate_add(VectorType &       dst,
                                                                      VectorType const & src) const
{
  matrix_free->loop(
    &This::cell_loop, &This::face_loop, &This::boundary_face_loop_full_operator, this, dst, src);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::calculate_diagonal(
  VectorType & diagonal) const
{
  if(diagonal.size() == 0)
    matrix_free->initialize_dof_vector(diagonal);
  diagonal = 0;
  add_diagonal(diagonal);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::add_diagonal(VectorType & diagonal) const
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
    matrix_free->cell_loop(&This::cell_loop_diagonal, this, diagonal, diagonal);
  }

  // multiple processes might have contributions to the same diagonal entry in
  // the cg case, so we have to sum them up
  if(!is_dg)
    diagonal.compress(VectorOperation::add);

  // apply constraints in the case of cg
  if(!is_dg)
    set_constraint_diagonal(diagonal);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::calculate_block_diagonal_matrices() const
{
  AssertThrow(is_dg, ExcMessage("Block Jacobi only implemented for DG!"));

  // allocate memory only the first time
  if(!block_diagonal_preconditioner_is_initialized ||
     matrix_free->n_macro_cells() * vectorization_length != matrices.size())
  {
    auto dofs =
      matrix_free->get_shape_info(this->data.dof_index).dofs_per_component_on_cell * n_components;

    matrices.resize(matrix_free->n_macro_cells() * vectorization_length,
                    LAPACKFullMatrix<Number>(dofs, dofs));

    block_diagonal_preconditioner_is_initialized = true;
  }
  // else: reuse old memory

  // clear matrices
  initialize_block_jacobi_matrices_with_zero(matrices);

  // compute block matrices
  add_block_diagonal_matrices(matrices);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::add_block_diagonal_matrices(
  BlockMatrix & matrices) const
{
  AssertThrow(is_dg, ExcMessage("Block Jacobi only implemented for DG!"));

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
        ExcMessage(
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

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::apply_block_diagonal_matrix_based(
  VectorType &       dst,
  VectorType const & src) const
{
  AssertThrow(is_dg, ExcMessage("Block Jacobi only implemented for DG!"));
  AssertThrow(block_diagonal_preconditioner_is_initialized,
              ExcMessage("Block Jacobi matrices have not been initialized!"));

  matrix_free->cell_loop(&This::cell_loop_apply_block_diagonal_matrix_based, this, dst, src);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::apply_inverse_block_diagonal(
  VectorType &       dst,
  VectorType const & src) const
{
  // matrix-free
  if(this->data.implement_block_diagonal_preconditioner_matrix_free)
  {
    // Solve elementwise block Jacobi problems iteratively using an elementwise solver vectorized
    // over several elements.
    bool update_preconditioner = false;
    elementwise_solver->solve(dst, src, update_preconditioner);
  }
  else // matrix-based
  {
    // Simply apply inverse of block matrices (using the LU factorization that has been computed
    // before).
    apply_inverse_block_diagonal_matrix_based(dst, src);
  }
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::apply_inverse_block_diagonal_matrix_based(
  VectorType &       dst,
  VectorType const & src) const
{
  AssertThrow(is_dg, ExcMessage("Block Jacobi only implemented for DG!"));
  AssertThrow(block_diagonal_preconditioner_is_initialized,
              ExcMessage("Block Jacobi matrices have not been initialized!"));

  matrix_free->cell_loop(&This::cell_loop_apply_inverse_block_diagonal_matrix_based,
                         this,
                         dst,
                         src);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::
  initialize_block_diagonal_preconditioner_matrix_free() const
{
  elementwise_operator.reset(new ELEMENTWISE_OPERATOR(*this));

  if(data.preconditioner_block_diagonal == Elementwise::Preconditioner::None)
  {
    typedef Elementwise::PreconditionerIdentity<VectorizedArray<Number>> IDENTITY;

    elementwise_preconditioner.reset(new IDENTITY(elementwise_operator->get_problem_size()));
  }
  else if(data.preconditioner_block_diagonal == Elementwise::Preconditioner::InverseMassMatrix)
  {
    typedef Elementwise::InverseMassMatrixPreconditioner<dim, n_components, Number> INVERSE_MASS;

    elementwise_preconditioner.reset(
      new INVERSE_MASS(get_matrix_free(), get_dof_index(), get_quad_index()));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  Elementwise::IterativeSolverData iterative_solver_data;
  iterative_solver_data.solver_type = data.solver_block_diagonal;
  iterative_solver_data.solver_data = data.solver_data_block_diagonal;

  elementwise_solver.reset(new ELEMENTWISE_SOLVER(
    *std::dynamic_pointer_cast<ELEMENTWISE_OPERATOR>(elementwise_operator),
    *std::dynamic_pointer_cast<ELEMENTWISE_PRECONDITIONER>(elementwise_preconditioner),
    iterative_solver_data));
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::apply_add_block_diagonal_elementwise(
  unsigned int const                    cell,
  VectorizedArray<Number> * const       dst,
  VectorizedArray<Number> const * const src,
  unsigned int const                    problem_size) const
{
  (void)problem_size;

  AssertThrow(is_dg, ExcMessage("Block Jacobi only implemented for DG!"));

  this->reinit_cell(cell);

  for(unsigned int i = 0; i < integrator->dofs_per_cell; ++i)
    integrator->begin_dof_values()[i] = src[i];

  integrator->evaluate(integrator_flags.cell_evaluate.value,
                       integrator_flags.cell_evaluate.gradient,
                       integrator_flags.cell_evaluate.hessian);

  this->do_cell_integral(*integrator);

  integrator->integrate(integrator_flags.cell_integrate.value,
                        integrator_flags.cell_integrate.gradient);

  for(unsigned int i = 0; i < integrator->dofs_per_cell; ++i)
    dst[i] += integrator->begin_dof_values()[i];

  if(is_dg && evaluate_face_integrals())
  {
    // face integrals
    unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
    for(unsigned int face = 0; face < n_faces; ++face)
    {
      auto bids = (*matrix_free).get_faces_by_cells_boundary_id(cell, face);
      auto bid  = bids[0];

      this->reinit_face_cell_based(cell, face, bid);

      for(unsigned int i = 0; i < integrator_m->dofs_per_cell; ++i)
        integrator_m->begin_dof_values()[i] = src[i];

      // no need to read dof values for integrator_p (already initialized with 0)

      integrator_m->evaluate(integrator_flags.face_evaluate.value,
                             integrator_flags.face_evaluate.gradient);

      if(bid == numbers::internal_face_boundary_id) // internal face
      {
        this->do_face_int_integral_cell_based(*integrator_m, *integrator_p);
      }
      else // boundary face
      {
        this->do_boundary_integral(*integrator_m, OperatorType::homogeneous, bid);
      }

      integrator_m->integrate(integrator_flags.face_integrate.value,
                              integrator_flags.face_integrate.gradient);

      for(unsigned int i = 0; i < integrator_m->dofs_per_cell; ++i)
        dst[i] += integrator_m->begin_dof_values()[i];
    }
  }
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::update_block_diagonal_preconditioner()
  const
{
  AssertThrow(is_dg, ExcMessage("Block Jacobi only implemented for DG!"));

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
      matrices.resize(matrix_free->n_macro_cells() * vectorization_length,
                      LAPACKFullMatrix<Number>(dofs, dofs));
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
template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::init_system_matrix(
  SparseMatrix & system_matrix) const
{
  DoFHandler<dim> const & dof_handler = this->matrix_free->get_dof_handler();

  MPI_Comm comm;

  // extract communicator
  {
    auto tria =
      dynamic_cast<parallel::TriangulationBase<dim> const *>(&dof_handler.get_triangulation());

    if(tria != NULL)
      comm = tria->get_communicator();
    else // not distributed triangulation
      comm = MPI_COMM_SELF;
  }

  TrilinosWrappers::SparsityPattern dsp(is_mg ? dof_handler.locally_owned_mg_dofs(this->level) :
                                                dof_handler.locally_owned_dofs(),
                                        comm);

  if(is_dg && is_mg)
    MGTools::make_flux_sparsity_pattern(dof_handler, dsp, this->level);
  else if(is_dg && !is_mg)
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
  else if(/*!is_dg &&*/ is_mg)
    MGTools::make_sparsity_pattern(dof_handler, dsp, this->level, *this->constraint);
  else /* if (!is_dg && !is_mg)*/
    DoFTools::make_sparsity_pattern(dof_handler, dsp, *this->constraint);

  dsp.compress();
  system_matrix.reinit(dsp);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::calculate_system_matrix(
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
  system_matrix.compress(VectorOperation::add);

  if(!is_dg)
  {
    // make zero entries on diagonal (due to constrained dofs) to one:
    auto p = system_matrix.local_range();
    for(auto i = p.first; i < p.second; i++)
      if(system_matrix(i, i) == 0.0 && constraint->is_constrained(i))
        system_matrix.add(i, i, 1);
  } // nothing to do for dg
}
#endif

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::reinit_cell(unsigned int const cell) const
{
  integrator->reinit(cell);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::reinit_face(unsigned int const face) const
{
  integrator_m->reinit(face);
  integrator_p->reinit(face);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::reinit_boundary_face(
  unsigned int const face) const
{
  integrator_m->reinit(face);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::do_cell_integral(
  IntegratorCell & integrator) const
{
  (void)integrator;

  AssertThrow(false, ExcMessage("OperatorBase::do_cell_integral() has not been implemented!"));
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::do_face_integral(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  (void)integrator_m;
  (void)integrator_p;

  AssertThrow(false, ExcMessage("OperatorBase::do_face_integral() has not been implemented!"));
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::do_boundary_integral(
  IntegratorFace &           integrator,
  OperatorType const &       operator_type,
  types::boundary_id const & boundary_id) const
{
  (void)integrator;
  (void)operator_type;
  (void)boundary_id;

  AssertThrow(false, ExcMessage("OperatorBase::do_boundary_integral() has not been implemented!"));
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::do_face_int_integral(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  (void)integrator_m;
  (void)integrator_p;

  AssertThrow(false, ExcMessage("OperatorBase::do_face_int_integral() has not been implemented!"));
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::do_face_ext_integral(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  (void)integrator_m;
  (void)integrator_p;

  AssertThrow(false, ExcMessage("OperatorBase::do_face_ext_integral() has not been implemented!"));
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::reinit_face_cell_based(
  unsigned int const       cell,
  unsigned int const       face,
  types::boundary_id const boundary_id) const
{
  integrator_m->reinit(cell, face);

  if(boundary_id == numbers::internal_face_boundary_id) // internal face
  {
    integrator_p->reinit(cell, face);
  }
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::do_face_int_integral_cell_based(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  this->do_face_int_integral(integrator_m, integrator_p);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::create_standard_basis(
  unsigned int     j,
  IntegratorCell & integrator) const
{
  // create a standard basis in the dof values of FEEvalution
  for(unsigned int i = 0; i < integrator.dofs_per_cell; ++i)
    integrator.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
  integrator.begin_dof_values()[j] = make_vectorized_array<Number>(1.);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::create_standard_basis(
  unsigned int     j,
  IntegratorFace & integrator) const
{
  // create a standard basis in the dof values of FEEvalution
  for(unsigned int i = 0; i < integrator.dofs_per_cell; ++i)
    integrator.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
  integrator.begin_dof_values()[j] = make_vectorized_array<Number>(1.);
}


template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::create_standard_basis(
  unsigned int     j,
  IntegratorFace & integrator_1,
  IntegratorFace & integrator_2) const
{
  // create a standard basis in the dof values of the first FEFaceEvalution
  for(unsigned int i = 0; i < integrator_1.dofs_per_cell; ++i)
    integrator_1.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
  integrator_1.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

  // clear dof values of the second FEFaceEvalution
  for(unsigned int i = 0; i < integrator_2.dofs_per_cell; ++i)
    integrator_2.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::cell_loop(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   range) const
{
  (void)matrix_free;

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    this->reinit_cell(cell);

    integrator->gather_evaluate(src,
                                integrator_flags.cell_evaluate.value,
                                integrator_flags.cell_evaluate.gradient,
                                integrator_flags.cell_evaluate.hessian);

    this->do_cell_integral(*integrator);

    integrator->integrate_scatter(integrator_flags.cell_integrate.value,
                                  integrator_flags.cell_integrate.gradient,
                                  dst);
  }
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::face_loop(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   range) const
{
  (void)matrix_free;

  for(auto face = range.first; face < range.second; ++face)
  {
    this->reinit_face(face);

    integrator_m->gather_evaluate(src,
                                  integrator_flags.face_evaluate.value,
                                  integrator_flags.face_evaluate.gradient);
    integrator_p->gather_evaluate(src,
                                  integrator_flags.face_evaluate.value,
                                  integrator_flags.face_evaluate.gradient);

    this->do_face_integral(*integrator_m, *integrator_p);

    integrator_m->integrate_scatter(integrator_flags.face_integrate.value,
                                    integrator_flags.face_integrate.gradient,
                                    dst);
    integrator_p->integrate_scatter(integrator_flags.face_integrate.value,
                                    integrator_flags.face_integrate.gradient,
                                    dst);
  }
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::boundary_face_loop_hom_operator(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   range) const
{
  for(unsigned int face = range.first; face < range.second; face++)
  {
    this->reinit_boundary_face(face);

    integrator_m->gather_evaluate(src,
                                  integrator_flags.face_evaluate.value,
                                  integrator_flags.face_evaluate.gradient);

    do_boundary_integral(*integrator_m,
                         OperatorType::homogeneous,
                         matrix_free.get_boundary_id(face));

    integrator_m->integrate_scatter(integrator_flags.face_integrate.value,
                                    integrator_flags.face_integrate.gradient,
                                    dst);
  }
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::boundary_face_loop_inhom_operator(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   range) const
{
  (void)src;

  for(unsigned int face = range.first; face < range.second; face++)
  {
    this->reinit_boundary_face(face);

    // note: no gathering/evaluation is necessary when calculating the
    //       inhomogeneous part of boundary face integrals

    do_boundary_integral(*integrator_m,
                         OperatorType::inhomogeneous,
                         matrix_free.get_boundary_id(face));

    integrator_m->integrate_scatter(integrator_flags.face_integrate.value,
                                    integrator_flags.face_integrate.gradient,
                                    dst);
  }
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::boundary_face_loop_full_operator(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   range) const
{
  for(unsigned int face = range.first; face < range.second; face++)
  {
    this->reinit_boundary_face(face);

    integrator_m->gather_evaluate(src,
                                  integrator_flags.face_evaluate.value,
                                  integrator_flags.face_evaluate.gradient);

    do_boundary_integral(*integrator_m, OperatorType::full, matrix_free.get_boundary_id(face));

    integrator_m->integrate_scatter(integrator_flags.face_integrate.value,
                                    integrator_flags.face_integrate.gradient,
                                    dst);
  }
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::cell_loop_empty(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   range) const
{
  (void)matrix_free;
  (void)dst;
  (void)src;
  (void)range;

  // do nothing
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::face_loop_empty(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   range) const
{
  (void)matrix_free;
  (void)dst;
  (void)src;
  (void)range;

  // do nothing
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::cell_loop_diagonal(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   range) const
{
  (void)matrix_free;
  (void)src;

  // create temporal array for local diagonal
  unsigned int const                     dofs_per_cell = integrator->dofs_per_cell;
  AlignedVector<VectorizedArray<Number>> local_diag(dofs_per_cell);

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    this->reinit_cell(cell);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of FEEvaluation
      this->create_standard_basis(j, *integrator);

      integrator->evaluate(integrator_flags.cell_evaluate.value,
                           integrator_flags.cell_evaluate.gradient,
                           integrator_flags.cell_evaluate.hessian);

      this->do_cell_integral(*integrator);

      integrator->integrate(integrator_flags.cell_integrate.value,
                            integrator_flags.cell_integrate.gradient);

      // extract single value from result vector and temporally store it
      local_diag[j] = integrator->begin_dof_values()[j];
    }
    // copy local diagonal entries into dof values of FEEvaluation ...
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      integrator->begin_dof_values()[j] = local_diag[j];

    // ... and write it back to global vector
    integrator->distribute_local_to_global(dst);
  }
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::face_loop_diagonal(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   range) const
{
  (void)matrix_free;
  (void)src;

  // create temporal array for local diagonal
  unsigned int const                     dofs_per_cell = integrator_m->dofs_per_cell;
  AlignedVector<VectorizedArray<Number>> local_diag(dofs_per_cell);

  for(auto face = range.first; face < range.second; ++face)
  {
    this->reinit_face(face);

    // interior face
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, *integrator_m);

      integrator_m->evaluate(integrator_flags.face_evaluate.value,
                             integrator_flags.face_evaluate.gradient);

      this->do_face_int_integral(*integrator_m, *integrator_p);

      integrator_m->integrate(integrator_flags.face_integrate.value,
                              integrator_flags.face_integrate.gradient);

      local_diag[j] = integrator_m->begin_dof_values()[j];
    }

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      integrator_m->begin_dof_values()[j] = local_diag[j];

    integrator_m->distribute_local_to_global(dst);

    // exterior face
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, *integrator_p);

      integrator_p->evaluate(integrator_flags.face_evaluate.value,
                             integrator_flags.face_evaluate.gradient);

      this->do_face_ext_integral(*integrator_m, *integrator_p);

      integrator_p->integrate(integrator_flags.face_integrate.value,
                              integrator_flags.face_integrate.gradient);

      local_diag[j] = integrator_p->begin_dof_values()[j];
    }

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      integrator_p->begin_dof_values()[j] = local_diag[j];

    integrator_p->distribute_local_to_global(dst);
  }
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::boundary_face_loop_diagonal(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   range) const
{
  (void)src;

  // create temporal array for local diagonal
  unsigned int const                     dofs_per_cell = integrator_m->dofs_per_cell;
  AlignedVector<VectorizedArray<Number>> local_diag(dofs_per_cell);

  for(unsigned int face = range.first; face < range.second; face++)
  {
    auto bid = matrix_free.get_boundary_id(face);

    this->reinit_boundary_face(face);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, *integrator_m);

      integrator_m->evaluate(integrator_flags.face_evaluate.value,
                             integrator_flags.face_evaluate.gradient);

      this->do_boundary_integral(*integrator_m, OperatorType::homogeneous, bid);

      integrator_m->integrate(integrator_flags.face_integrate.value,
                              integrator_flags.face_integrate.gradient);

      local_diag[j] = integrator_m->begin_dof_values()[j];
    }

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      integrator_m->begin_dof_values()[j] = local_diag[j];

    integrator_m->distribute_local_to_global(dst);
  }
}


template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::cell_based_loop_diagonal(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   range) const
{
  (void)src;

  // create temporal array for local diagonal
  unsigned int const                     dofs_per_cell = integrator->dofs_per_cell;
  AlignedVector<VectorizedArray<Number>> local_diag(dofs_per_cell);

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    this->reinit_cell(cell);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, *integrator);

      integrator->evaluate(integrator_flags.cell_evaluate.value,
                           integrator_flags.cell_evaluate.gradient,
                           integrator_flags.cell_evaluate.hessian);

      this->do_cell_integral(*integrator);

      integrator->integrate(integrator_flags.cell_integrate.value,
                            integrator_flags.cell_integrate.gradient);

      local_diag[j] = integrator->begin_dof_values()[j];
    }

    // loop over all faces and gather results into local diagonal local_diag
    unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
    for(unsigned int face = 0; face < n_faces; ++face)
    {
      auto bids = matrix_free.get_faces_by_cells_boundary_id(cell, face);
      auto bid  = bids[0];

      this->reinit_face_cell_based(cell, face, bid);

#ifdef DEBUG
      unsigned int const n_filled_lanes = matrix_free.n_active_entries_per_cell_batch(cell);
      for(unsigned int v = 0; v < n_filled_lanes; v++)
        Assert(bid == bids[v],
               ExcMessage("Cell-based face loop encountered face batch with different bids."));
#endif

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        this->create_standard_basis(j, *integrator_m);

        integrator_m->evaluate(integrator_flags.face_evaluate.value,
                               integrator_flags.face_evaluate.gradient);

        if(bid == numbers::internal_face_boundary_id) // internal face
        {
          this->do_face_int_integral_cell_based(*integrator_m, *integrator_p);
        }
        else // boundary face
        {
          this->do_boundary_integral(*integrator_m, OperatorType::homogeneous, bid);
        }

        integrator_m->integrate(integrator_flags.face_integrate.value,
                                integrator_flags.face_integrate.gradient);

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

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::
  cell_loop_apply_inverse_block_diagonal_matrix_based(MatrixFree<dim, Number> const & matrix_free,
                                                      VectorType &                    dst,
                                                      VectorType const &              src,
                                                      Range const & cell_range) const
{
  (void)matrix_free;

  unsigned int const dofs_per_cell = integrator->dofs_per_cell;

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    this->reinit_cell(cell);

    integrator->read_dof_values(src);

    for(unsigned int v = 0; v < vectorization_length; ++v)
    {
      Vector<Number> src_vector(dofs_per_cell);
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

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::
  cell_loop_apply_block_diagonal_matrix_based(MatrixFree<dim, Number> const & matrix_free,
                                              VectorType &                    dst,
                                              VectorType const &              src,
                                              Range const &                   range) const
{
  (void)matrix_free;

  unsigned int const dofs_per_cell = integrator->dofs_per_cell;

  for(unsigned int cell = range.first; cell < range.second; ++cell)
  {
    this->reinit(cell);

    integrator->read_dof_values(src);

    for(unsigned int v = 0; v < vectorization_length; ++v)
    {
      Vector<Number> src_vector(dofs_per_cell);
      Vector<Number> dst_vector(dofs_per_cell);
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

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::cell_loop_block_diagonal(
  MatrixFree<dim, Number> const & matrix_free,
  BlockMatrix &                   matrices,
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

      integrator->evaluate(integrator_flags.cell_evaluate.value,
                           integrator_flags.cell_evaluate.gradient,
                           integrator_flags.cell_evaluate.hessian);

      this->do_cell_integral(*integrator);

      integrator->integrate(integrator_flags.cell_integrate.value,
                            integrator_flags.cell_integrate.gradient);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices[cell * vectorization_length + v](i, j) += integrator->begin_dof_values()[i][v];
    }
  }
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::face_loop_block_diagonal(
  MatrixFree<dim, Number> const & matrix_free,
  BlockMatrix &                   matrices,
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

      integrator_m->evaluate(integrator_flags.face_evaluate.value,
                             integrator_flags.face_evaluate.gradient);

      this->do_face_int_integral(*integrator_m, *integrator_p);

      integrator_m->integrate(integrator_flags.face_integrate.value,
                              integrator_flags.face_integrate.gradient);

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

      integrator_p->evaluate(integrator_flags.face_evaluate.value,
                             integrator_flags.face_evaluate.gradient);

      this->do_face_ext_integral(*integrator_m, *integrator_p);

      integrator_p->integrate(integrator_flags.face_integrate.value,
                              integrator_flags.face_integrate.gradient);

      for(unsigned int v = 0; v < n_filled_lanes; ++v)
      {
        unsigned int const cell = matrix_free.get_face_info(face).cells_exterior[v];
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          matrices[cell](i, j) += integrator_p->begin_dof_values()[i][v];
      }
    }
  }
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::boundary_face_loop_block_diagonal(
  MatrixFree<dim, Number> const & matrix_free,
  BlockMatrix &                   matrices,
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

      integrator_m->evaluate(integrator_flags.face_evaluate.value,
                             integrator_flags.face_evaluate.gradient);

      this->do_boundary_integral(*integrator_m, OperatorType::homogeneous, bid);

      integrator_m->integrate(integrator_flags.face_integrate.value,
                              integrator_flags.face_integrate.gradient);

      for(unsigned int v = 0; v < n_filled_lanes; ++v)
      {
        unsigned int const cell = matrix_free.get_face_info(face).cells_interior[v];
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          matrices[cell](i, j) += integrator_m->begin_dof_values()[i][v];
      }
    }
  }
}


template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::cell_based_loop_block_diagonal(
  MatrixFree<dim, Number> const & matrix_free,
  BlockMatrix &                   matrices,
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

      integrator->evaluate(integrator_flags.cell_evaluate.value,
                           integrator_flags.cell_evaluate.gradient,
                           integrator_flags.cell_evaluate.hessian);

      this->do_cell_integral(*integrator);

      integrator->integrate(integrator_flags.cell_integrate.value,
                            integrator_flags.cell_integrate.gradient);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices[cell * vectorization_length + v](i, j) += integrator->begin_dof_values()[i][v];
    }

    // loop over all faces
    unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
    for(unsigned int face = 0; face < n_faces; ++face)
    {
      auto bids = matrix_free.get_faces_by_cells_boundary_id(cell, face);
      auto bid  = bids[0];

      this->reinit_face_cell_based(cell, face, bid);

#ifdef DEBUG
      for(unsigned int v = 0; v < n_filled_lanes; v++)
        Assert(bid == bids[v],
               ExcMessage("Cell-based face loop encountered face batch with different bids."));
#endif

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        this->create_standard_basis(j, *integrator_m);

        integrator_m->evaluate(integrator_flags.face_evaluate.value,
                               integrator_flags.face_evaluate.gradient);

        if(bid == numbers::internal_face_boundary_id) // internal face
        {
          this->do_face_int_integral_cell_based(*integrator_m, *integrator_p);
        }
        else // boundary face
        {
          this->do_boundary_integral(*integrator_m, OperatorType::homogeneous, bid);
        }

        integrator_m->integrate(integrator_flags.face_integrate.value,
                                integrator_flags.face_integrate.gradient);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int v = 0; v < n_filled_lanes; ++v)
            matrices[cell * vectorization_length + v](i, j) +=
              integrator_m->begin_dof_values()[i][v];
      }
    }
  }
}

#ifdef DEAL_II_WITH_TRILINOS
template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::cell_loop_calculate_system_matrix(
  MatrixFree<dim, Number> const & matrix_free,
  SparseMatrix &                  dst,
  SparseMatrix const &            src,
  Range const &                   range) const
{
  (void)src;

  unsigned int const dofs_per_cell = integrator->dofs_per_cell;

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    unsigned int const n_filled_lanes = matrix_free.n_components_filled(cell);

    // create a temporal full matrix for the local element matrix of each ...
    // cell of each macro cell and ...
    FullMatrix_ matrices[vectorization_length];
    // set their size
    std::fill_n(matrices, vectorization_length, FullMatrix_(dofs_per_cell, dofs_per_cell));

    this->reinit_cell(cell);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, *integrator);

      integrator->evaluate(integrator_flags.cell_evaluate.value,
                           integrator_flags.cell_evaluate.gradient,
                           integrator_flags.cell_evaluate.hessian);

      this->do_cell_integral(*integrator);

      integrator->integrate(integrator_flags.cell_integrate.value,
                            integrator_flags.cell_integrate.gradient);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices[v](i, j) = integrator->begin_dof_values()[i][v];
    }

    // finally assemble local matrices into global matrix
    for(unsigned int v = 0; v < n_filled_lanes; v++)
    {
      auto cell_v = matrix_free.get_cell_iterator(cell, v);

      std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
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

      constraint->distribute_local_to_global(matrices[v], dof_indices, dof_indices, dst);
    }
  }
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::face_loop_calculate_system_matrix(
  MatrixFree<dim, Number> const & matrix_free,
  SparseMatrix &                  dst,
  SparseMatrix const &            src,
  Range const &                   range) const
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
      // write standard basis into dof values of first FEFaceEvaluation and
      // clear dof values of second FEFaceEvaluation
      this->create_standard_basis(j, *integrator_m, *integrator_p);

      integrator_m->evaluate(integrator_flags.face_evaluate.value,
                             integrator_flags.face_evaluate.gradient);
      integrator_p->evaluate(integrator_flags.face_evaluate.value,
                             integrator_flags.face_evaluate.gradient);

      this->do_face_integral(*integrator_m, *integrator_p);

      integrator_m->integrate(integrator_flags.face_integrate.value,
                              integrator_flags.face_integrate.gradient);
      integrator_p->integrate(integrator_flags.face_integrate.value,
                              integrator_flags.face_integrate.gradient);

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
      std::vector<types::global_dof_index> dof_indices_m(dofs_per_cell);
      std::vector<types::global_dof_index> dof_indices_p(dofs_per_cell);
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
      constraint->distribute_local_to_global(matrices_m[v], dof_indices_m, dof_indices_m, dst);
      // save M_pm
      constraint->distribute_local_to_global(matrices_p[v], dof_indices_p, dof_indices_m, dst);
    }

    // process positive trial function
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of first FEFaceEvaluation and
      // clear dof values of second FEFaceEvaluation
      this->create_standard_basis(j, *integrator_p, *integrator_m);

      integrator_m->evaluate(integrator_flags.face_evaluate.value,
                             integrator_flags.face_evaluate.gradient);
      integrator_p->evaluate(integrator_flags.face_evaluate.value,
                             integrator_flags.face_evaluate.gradient);

      this->do_face_integral(*integrator_m, *integrator_p);

      integrator_m->integrate(integrator_flags.face_integrate.value,
                              integrator_flags.face_integrate.gradient);
      integrator_p->integrate(integrator_flags.face_integrate.value,
                              integrator_flags.face_integrate.gradient);

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
      std::vector<types::global_dof_index> dof_indices_m(dofs_per_cell);
      std::vector<types::global_dof_index> dof_indices_p(dofs_per_cell);
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
      constraint->distribute_local_to_global(matrices_m[v], dof_indices_m, dof_indices_p, dst);
      // save M_pp
      constraint->distribute_local_to_global(matrices_p[v], dof_indices_p, dof_indices_p, dst);
    }
  }
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::boundary_face_loop_calculate_system_matrix(
  MatrixFree<dim, Number> const & matrix_free,
  SparseMatrix &                  dst,
  SparseMatrix const &            src,
  Range const &                   range) const
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

      integrator_m->evaluate(integrator_flags.face_evaluate.value,
                             integrator_flags.face_evaluate.gradient);

      this->do_boundary_integral(*integrator_m, OperatorType::homogeneous, bid);

      integrator_m->integrate(integrator_flags.face_integrate.value,
                              integrator_flags.face_integrate.gradient);

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

      std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
      if(is_mg)
        cell_v->get_mg_dof_indices(dof_indices);
      else
        cell_v->get_dof_indices(dof_indices);

      constraint->distribute_local_to_global(matrices[v], dof_indices, dof_indices, dst);
    }
  }
}
#endif

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::set_constraint_diagonal(
  VectorType & diagonal) const
{
  // set (diagonal) entries to 1.0 for constrained dofs
  for(auto i : matrix_free->get_constrained_dofs())
    diagonal.local_element(i) = 1.0;
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::verify_boundary_conditions(
  DoFHandler<dim> const &                 dof_handler,
  std::vector<PeriodicFacePairIterator> & periodic_face_pairs_level0) const
{
  // fill set with periodic boundary ids
  std::set<types::boundary_id> periodic_boundary_ids;
  for(unsigned int i = 0; i < data.periodic_face_pairs_level0.size(); ++i)
  {
    AssertThrow(data.periodic_face_pairs_level0[i].cell[0]->h_level() == 0,
                ExcMessage("Received periodic cell pairs on non-zero level"));
    periodic_boundary_ids.insert(data.periodic_face_pairs_level0[i]
                                   .cell[0]
                                   ->face(data.periodic_face_pairs_level0[i].face_idx[0])
                                   ->boundary_id());
    periodic_boundary_ids.insert(data.periodic_face_pairs_level0[i]
                                   .cell[1]
                                   ->face(data.periodic_face_pairs_level0[i].face_idx[1])
                                   ->boundary_id());
  }

  // Make sure that each boundary face has exactly one boundary type
  Triangulation<dim> const & tria = dof_handler.get_triangulation();
  for(typename Triangulation<dim>::cell_iterator cell = tria.begin(); cell != tria.end(); ++cell)
  {
    for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
    {
      if(cell->at_boundary(f))
      {
        types::boundary_id bid = cell->face(f)->boundary_id();
        do_verify_boundary_conditions(bid, data, periodic_boundary_ids);
      }
    }
  }
}

template<int dim, typename Number, typename AdditionalData, int n_components>
void
OperatorBase<dim, Number, AdditionalData, n_components>::do_verify_boundary_conditions(
  types::boundary_id const             boundary_id,
  AdditionalData const &               data,
  std::set<types::boundary_id> const & periodic_boundary_ids) const
{
  (void)boundary_id;
  (void)data;
  (void)periodic_boundary_ids;

  AssertThrow(
    false,
    ExcMessage(
      "OperatorBase::do_verify_boundary_conditions() has to be implemented by derived classes."));
}

template<int dim, typename Number, typename AdditionalData, int n_components>
bool
OperatorBase<dim, Number, AdditionalData, n_components>::evaluate_face_integrals() const
{
  return integrator_flags.face_evaluate.do_eval() || integrator_flags.face_integrate.do_eval();
}
