#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_tools.h>

#include "../solvers_and_preconditioners/util/block_jacobi_matrices.h"
#include "../solvers_and_preconditioners/util/invert_diagonal.h"

#include "../functionalities/categorization.h"
#include "../functionalities/set_zero_mean_value.h"
#include "operator_base.h"

template<int dim, int degree, typename Number, typename AdditionalData>
OperatorBase<dim, degree, Number, AdditionalData>::OperatorBase()
  : operator_data(AdditionalData()),
    data(),
    eval_time(0.0),
    do_eval_faces(operator_data.face_evaluate.do_eval() || operator_data.face_integrate.do_eval()),
    is_dg(true),
    is_mg(false),
    level_mg_handler(numbers::invalid_unsigned_int),
    block_diagonal_preconditioner_is_initialized(false),
    n_mpi_processes(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
{
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::reinit(
  MatrixFree<dim, Number> const & matrix_free,
  AdditionalData const &          operator_data) const
{
  AffineConstraints<double> constraint_matrix;
  reinit(matrix_free, constraint_matrix, operator_data);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::reinit(
  MatrixFree<dim, Number> const &   matrix_free,
  AffineConstraints<double> const & constraint_matrix,
  AdditionalData const &            operator_data) const
{
  // reinit data structures
  this->data.reinit(matrix_free);
  this->constraint.reinit(constraint_matrix);
  this->operator_data = operator_data;

  // verify boundary conditions
  if(this->operator_data.evaluate_face_integrals)
  {
    this->verify_boundary_conditions(data->get_dof_handler(this->operator_data.dof_index),
                                     this->operator_data);
  }

  // check if DG or CG
  // An approximation can have degrees of freedom on vertices, edges, quads and
  // hexes. A vertex degree of freedom means that the degree of freedom is
  // the same on all cells that are adjacent to this vertex. A face degree of
  // freedom means that the two cells adjacent to that face see the same degree
  // of freedom. A DG element does not share any degrees of freedom over a
  // vertex but has all of them in the last item, i.e., quads in 2D and hexes
  // in 3D, and thus necessarily has dofs_per_vertex=0
  is_dg = (data->get_dof_handler(this->operator_data.dof_index).get_fe().dofs_per_vertex == 0);

  // initialize FEEvaluation objects required for elementwise block Jacobi operations
  if(this->operator_data.implement_block_diagonal_preconditioner_matrix_free)
  {
    fe_eval.reset(
      new FEEvalCell(*this->data, this->operator_data.dof_index, this->operator_data.quad_index));
    fe_eval_m.reset(new FEEvalFace(
      *this->data, true, this->operator_data.dof_index, this->operator_data.quad_index));
    fe_eval_p.reset(new FEEvalFace(
      *this->data, false, this->operator_data.dof_index, this->operator_data.quad_index));
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::reinit_multigrid(
  MatrixFree<dim, Number> const &   matrix_free,
  AffineConstraints<double> const & constraint_matrix,
  AdditionalData const &            operator_data,
  unsigned int                      level) const
{
  // set multigrid level
  this->level_mg_handler = level;

  // The default value is is_mg = false and this variable is set to true in case
  // the operator is applied in multigrid algorithm. By convention, the default
  // argument numbers::invalid_unsigned_int corresponds to the default
  // value is_mg = false
  this->is_mg = (level != numbers::invalid_unsigned_int);

  reinit(matrix_free, constraint_matrix, operator_data);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::do_reinit_multigrid(
  DoFHandler<dim> const &   dof_handler,
  Mapping<dim> const &      mapping,
  void *                    operator_data_in,
  MGConstrainedDoFs const & mg_constrained_dofs,
  unsigned int const        level)
{
  // create copy of data and ...
  auto operator_data = *static_cast<AdditionalData *>(operator_data_in);
  // set dof_index and quad_index to 0 since we only consider a subset
  operator_data.dof_index  = 0;
  operator_data.quad_index = 0;

  // check if DG or CG (for explanation: see above)
  bool is_dg = dof_handler.get_fe().dofs_per_vertex == 0;

  // setup MatrixFree::AdditionalData
  typename MatrixFree<dim, Number>::AdditionalData additional_data;

  // ... level of this mg level
  additional_data.level_mg_handler = level;

  // ... update flags
  additional_data.mapping_update_flags = operator_data.mapping_update_flags;

  if(is_dg)
  {
    additional_data.mapping_update_flags_inner_faces =
      operator_data.mapping_update_flags_inner_faces;
    additional_data.mapping_update_flags_boundary_faces =
      operator_data.mapping_update_flags_boundary_faces;
  }

  if(operator_data.use_cell_based_loops && is_dg)
  {
    auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> const *>(
      &dof_handler.get_triangulation());
    Categorization::do_cell_based_loops(*tria, additional_data, level);
  }

  // ... on each level
  auto & constraint_own = constraint.own();
  auto & data_own       = data.own();

  // setup constraint matrix for CG
  if(!is_dg)
  {
    this->add_constraints(dof_handler, constraint_own, mg_constrained_dofs, operator_data, level);
  }

  // ...finalize constraint matrix
  constraint_own.close();

  QGauss<1> const quad(dof_handler.get_fe().degree + 1);
  data_own.reinit(mapping, dof_handler, constraint_own, quad, additional_data);


  reinit_multigrid(data_own, constraint_own, operator_data, level);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::apply(VectorType &       dst,
                                                         VectorType const & src) const
{
  dst = 0;
  apply_add(dst, src);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::apply_add(VectorType &       dst,
                                                             VectorType const & src) const
{
  VectorType const * actual_src = &src;
  VectorType         tmp_projection_vector;
  if(operator_is_singular() && !is_mg && is_dg)
  {
    tmp_projection_vector = src;
    set_zero_mean_value(tmp_projection_vector);
    actual_src = &tmp_projection_vector;
  }

  if(is_dg && do_eval_faces)
  {
    data->loop(&This::cell_loop,
               &This::face_loop,
               &This::boundary_face_loop_hom_operator,
               this,
               dst,
               *actual_src);
  }
  else
  {
    data->cell_loop(&This::cell_loop, this, dst, *actual_src);
  }

  if(operator_is_singular() && !is_mg && is_dg)
    set_zero_mean_value(dst);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::apply_add(VectorType &       dst,
                                                             VectorType const & src,
                                                             Number const       time) const
{
  this->set_evaluation_time(time);
  this->apply_add(dst, src);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::rhs(VectorType & dst) const
{
  dst = 0;
  this->rhs_add(dst);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::rhs(VectorType & dst, Number const time) const
{
  this->set_evaluation_time(time);
  this->rhs(dst);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::rhs_add(VectorType & dst) const
{
  VectorType tmp;
  tmp.reinit(dst, false);

  data->loop(&This::cell_loop_empty,
             &This::face_loop_empty,
             &This::boundary_face_loop_inhom_operator,
             this,
             tmp,
             tmp);

  // multiply by -1.0 since the boundary face integrals have to be shifted to the right hand side
  dst.add(-1.0, tmp);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::rhs_add(VectorType & dst,
                                                           Number const time) const
{
  this->set_evaluation_time(time);
  this->rhs_add(dst);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::evaluate(VectorType &       dst,
                                                            VectorType const & src,
                                                            Number const       time) const
{
  dst = 0;
  evaluate_add(dst, src, time);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::evaluate_add(VectorType &       dst,
                                                                VectorType const & src,
                                                                Number const       time) const
{
  this->eval_time = time;

  data->loop(
    &This::cell_loop, &This::face_loop, &This::boundary_face_loop_full_operator, this, dst, src);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::calculate_diagonal(VectorType & diagonal) const
{
  if(diagonal.size() == 0)
    data->initialize_dof_vector(diagonal);
  diagonal = 0;
  add_diagonal(diagonal);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::add_diagonal(VectorType & diagonal) const
{
  // compute diagonal
  if(is_dg && do_eval_faces)
  {
    if(operator_data.use_cell_based_loops)
    {
      data->cell_loop(&This::cell_based_loop_diagonal, this, diagonal, diagonal);
    }
    else
    {
      data->loop(&This::cell_loop_diagonal,
                 &This::face_loop_diagonal,
                 &This::boundary_face_loop_diagonal,
                 this,
                 diagonal,
                 diagonal);
    }
  }
  else
  {
    data->cell_loop(&This::cell_loop_diagonal, this, diagonal, diagonal);
  }

  // multiple processes might have contributions to the same diagonal entry in
  // the cg case, so we have to sum them up
  if(!is_dg)
    diagonal.compress(VectorOperation::add);

  // in case that the operator is singular, the diagonal has to be adjusted
  if(operator_is_singular() && !is_mg)
    adjust_diagonal_for_singular_operator(diagonal);

  // apply constraints in the case of cg
  if(!is_dg)
    set_constraint_diagonal(diagonal);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::add_diagonal(VectorType & diagonal,
                                                                Number const time) const
{
  this->set_evaluation_time(time);
  this->add_diagonal(diagonal);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::apply_inverse_block_diagonal_matrix_based(
  VectorType &       dst,
  VectorType const & src) const
{
  AssertThrow(is_dg, ExcMessage("Block Jacobi only implemented for DG!"));
  AssertThrow(block_diagonal_preconditioner_is_initialized,
              ExcMessage("Block Jacobi matrices have not been initialized!"));

  data->cell_loop(&This::cell_loop_apply_inverse_block_diagonal_matrix_based, this, dst, src);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::apply_add_block_diagonal_elementwise(
  unsigned int const                    cell,
  VectorizedArray<Number> * const       dst,
  VectorizedArray<Number> const * const src,
  Number const                          evaluation_time) const
{
  this->set_evaluation_time(evaluation_time);

  apply_add_block_diagonal_elementwise(cell, dst, src);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::apply_add_block_diagonal_elementwise(
  unsigned int const                    cell,
  VectorizedArray<Number> * const       dst,
  VectorizedArray<Number> const * const src) const
{
  AssertThrow(is_dg, ExcMessage("Block Jacobi only implemented for DG!"));

  fe_eval->reinit(cell);

  for(unsigned int i = 0; i < dofs_per_cell; ++i)
    fe_eval->begin_dof_values()[i] = src[i];

  fe_eval->evaluate(this->operator_data.cell_evaluate.value,
                    this->operator_data.cell_evaluate.gradient,
                    this->operator_data.cell_evaluate.hessians);

  this->do_cell_integral(*fe_eval);

  fe_eval->integrate(this->operator_data.cell_integrate.value,
                     this->operator_data.cell_integrate.gradient);

  for(unsigned int i = 0; i < dofs_per_cell; ++i)
    dst[i] += fe_eval->begin_dof_values()[i];

  if(is_dg && do_eval_faces)
  {
    // face integrals
    unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
    for(unsigned int face = 0; face < n_faces; ++face)
    {
      fe_eval_m->reinit(cell, face);
      fe_eval_p->reinit(cell, face);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        fe_eval_m->begin_dof_values()[i] = src[i];

      // do not need to read dof values for fe_eval_p (already initialized with 0)

      fe_eval_m->evaluate(this->operator_data.face_evaluate.value,
                          this->operator_data.face_evaluate.gradient);

      auto bids = (*data).get_faces_by_cells_boundary_id(cell, face);
      auto bid  = bids[0];
      if(bid == numbers::internal_face_boundary_id) // internal face
        this->do_face_int_integral(*fe_eval_m, *fe_eval_p);
      else // boundary face
        this->do_boundary_integral(*fe_eval_m, OperatorType::homogeneous, bid);

      fe_eval_m->integrate(this->operator_data.face_integrate.value,
                           this->operator_data.face_integrate.gradient);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        dst[i] += fe_eval_m->begin_dof_values()[i];
    }
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::apply_block_diagonal_matrix_based(
  VectorType &       dst,
  VectorType const & src) const
{
  AssertThrow(is_dg, ExcMessage("Block Jacobi only implemented for DG!"));
  AssertThrow(block_diagonal_preconditioner_is_initialized,
              ExcMessage("Block Jacobi matrices have not been initialized!"));

  data->cell_loop(&This::cell_loop_apply_block_diagonal_matrix_based, this, dst, src);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::do_update_block_diagonal_preconditioner() const
{
  AssertThrow(is_dg, ExcMessage("Block Jacobi only implemented for DG!"));

  // initialization

  if(!block_diagonal_preconditioner_is_initialized)
  {
    if(operator_data.implement_block_diagonal_preconditioner_matrix_free)
    {
      initialize_block_diagonal_preconditioner_matrix_free();
    }
    else // matrix-based variant
    {
      // allocate memory only the first time
      auto dofs = data->get_shape_info().dofs_per_component_on_cell;
      matrices.resize(data->n_macro_cells() * vectorization_length,
                      LAPACKFullMatrix<Number>(dofs, dofs));
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
    add_block_diagonal_matrices(matrices);

    calculate_lu_factorization_block_jacobi(matrices);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::add_block_diagonal_matrices(
  BlockMatrix & matrices) const
{
  AssertThrow(is_dg, ExcMessage("Block Jacobi only implemented for DG!"));

  if(do_eval_faces)
  {
    if(operator_data.use_cell_based_loops)
    {
      data->cell_loop(&This::cell_based_loop_block_diagonal, this, matrices, matrices);
    }
    else
    {
      AssertThrow(
        n_mpi_processes == 1,
        ExcMessage(
          "Block diagonal calculation with separate loops over cells and faces only works in serial. "
          "Use cell based loops for parallel computations."));

      data->loop(&This::cell_loop_block_diagonal,
                 &This::face_loop_block_diagonal,
                 &This::boundary_face_loop_block_diagonal,
                 this,
                 matrices,
                 matrices);
    }
  }
  else
  {
    data->cell_loop(&This::cell_loop_block_diagonal, this, matrices, matrices);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::add_block_diagonal_matrices(
  BlockMatrix & matrices,
  Number const  time) const
{
  this->set_evaluation_time(time);
  this->add_block_diagonal_matrices(matrices);
}

#ifdef DEAL_II_WITH_TRILINOS
template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::do_init_system_matrix(
  SparseMatrix & system_matrix) const
{
  DoFHandler<dim> const & dof_handler = this->data->get_dof_handler();

  MPI_Comm comm;

  // extract communicator
  {
    auto tria =
      dynamic_cast<parallel::Triangulation<dim> const *>(&dof_handler.get_triangulation());

    if(tria != NULL)
      comm = tria->get_communicator();
    else // not distributed triangulation
      comm = MPI_COMM_SELF;
  }

  TrilinosWrappers::SparsityPattern dsp(is_mg ? dof_handler.locally_owned_mg_dofs(
                                                  this->level_mg_handler) :
                                                dof_handler.locally_owned_dofs(),
                                        comm);

  if(is_dg && is_mg)
    MGTools::make_flux_sparsity_pattern(dof_handler, dsp, this->level_mg_handler);
  else if(is_dg && !is_mg)
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
  else if(/*!is_dg &&*/ is_mg)
    MGTools::make_sparsity_pattern(dof_handler, dsp, this->level_mg_handler);
  else /* if (!is_dg && !is_mg)*/
    DoFTools::make_sparsity_pattern(dof_handler, dsp);

  dsp.compress();
  system_matrix.reinit(dsp);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::do_calculate_system_matrix(
  SparseMatrix & system_matrix) const
{
  // assemble matrix locally on each process
  if(do_eval_faces && is_dg)
  {
    data->loop(&This::cell_loop_calculate_system_matrix,
               &This::face_loop_calculate_system_matrix,
               &This::boundary_face_loop_calculate_system_matrix,
               this,
               system_matrix,
               system_matrix);
  }
  else
  {
    data->cell_loop(&This::cell_loop_calculate_system_matrix, this, system_matrix, system_matrix);
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

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::do_calculate_system_matrix(
  SparseMatrix & system_matrix,
  Number const   time) const
{
  this->eval_time = time;
  do_calculate_system_matrix(system_matrix);
}
#endif

template<int dim, int degree, typename Number, typename AdditionalData>
AdditionalData const &
OperatorBase<dim, degree, Number, AdditionalData>::get_operator_data() const
{
  return operator_data;
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::set_evaluation_time(double const time) const
{
  eval_time = time;
}

template<int dim, int degree, typename Number, typename AdditionalData>
double
OperatorBase<dim, degree, Number, AdditionalData>::get_evaluation_time() const
{
  return eval_time;
}

template<int dim, int degree, typename Number, typename AdditionalData>
unsigned int
OperatorBase<dim, degree, Number, AdditionalData>::get_level() const
{
  return level_mg_handler;
}

template<int dim, int degree, typename Number, typename AdditionalData>
AffineConstraints<double> const &
OperatorBase<dim, degree, Number, AdditionalData>::get_constraint_matrix() const
{
  return *constraint;
}

template<int dim, int degree, typename Number, typename AdditionalData>
bool
OperatorBase<dim, degree, Number, AdditionalData>::operator_is_singular() const
{
  return this->operator_data.operator_is_singular;
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::create_standard_basis(unsigned int j,
                                                                         FEEvalCell & fe_eval) const
{
  // create a standard basis in the dof values of FEEvalution
  for(unsigned int i = 0; i < dofs_per_cell; ++i)
    fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
  fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::create_standard_basis(unsigned int j,
                                                                         FEEvalFace & fe_eval) const
{
  // create a standard basis in the dof values of FEEvalution
  for(unsigned int i = 0; i < dofs_per_cell; ++i)
    fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
  fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);
}


template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::create_standard_basis(
  unsigned int j,
  FEEvalFace & fe_eval_1,
  FEEvalFace & fe_eval_2) const
{
  // create a standard basis in the dof values of the first FEFaceEvalution
  for(unsigned int i = 0; i < dofs_per_cell; ++i)
    fe_eval_1.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
  fe_eval_1.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

  // clear dof values of the second FEFaceEvalution
  for(unsigned int i = 0; i < dofs_per_cell; ++i)
    fe_eval_2.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::cell_loop(MatrixFree<dim, Number> const & data,
                                                             VectorType &                    dst,
                                                             VectorType const &              src,
                                                             Range const & range) const
{
  FEEvalCell fe_eval(data, operator_data.dof_index, operator_data.quad_index);

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    fe_eval.reinit(cell);

    fe_eval.gather_evaluate(src,
                            this->operator_data.cell_evaluate.value,
                            this->operator_data.cell_evaluate.gradient,
                            this->operator_data.cell_evaluate.hessians);

    this->do_cell_integral(fe_eval);

    fe_eval.integrate_scatter(this->operator_data.cell_integrate.value,
                              this->operator_data.cell_integrate.gradient,
                              dst);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::face_loop(MatrixFree<dim, Number> const & data,
                                                             VectorType &                    dst,
                                                             VectorType const &              src,
                                                             Range const & range) const
{
  FEEvalFace fe_eval_m(data, true, operator_data.dof_index, operator_data.quad_index);
  FEEvalFace fe_eval_p(data, false, operator_data.dof_index, operator_data.quad_index);

  for(auto face = range.first; face < range.second; ++face)
  {
    fe_eval_m.reinit(face);
    fe_eval_p.reinit(face);

    fe_eval_m.gather_evaluate(src,
                              this->operator_data.face_evaluate.value,
                              this->operator_data.face_evaluate.gradient);
    fe_eval_p.gather_evaluate(src,
                              this->operator_data.face_evaluate.value,
                              this->operator_data.face_evaluate.gradient);

    this->do_face_integral(fe_eval_m, fe_eval_p);

    fe_eval_m.integrate_scatter(this->operator_data.face_integrate.value,
                                this->operator_data.face_integrate.gradient,
                                dst);
    fe_eval_p.integrate_scatter(this->operator_data.face_integrate.value,
                                this->operator_data.face_integrate.gradient,
                                dst);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::boundary_face_loop_hom_operator(
  MatrixFree<dim, Number> const & data,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   range) const
{
  FEEvalFace fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

  for(unsigned int face = range.first; face < range.second; face++)
  {
    fe_eval.reinit(face);

    fe_eval.gather_evaluate(src,
                            this->operator_data.face_evaluate.value,
                            this->operator_data.face_evaluate.gradient);

    do_boundary_integral(fe_eval, OperatorType::homogeneous, data.get_boundary_id(face));

    fe_eval.integrate_scatter(this->operator_data.face_integrate.value,
                              this->operator_data.face_integrate.gradient,
                              dst);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::boundary_face_loop_inhom_operator(
  MatrixFree<dim, Number> const & data,
  VectorType &                    dst,
  VectorType const & /*src*/,
  Range const & range) const
{
  FEEvalFace fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

  for(unsigned int face = range.first; face < range.second; face++)
  {
    fe_eval.reinit(face);

    // note: no gathering/evaluation is necessary when calculating the
    //       inhomogeneous part of boundary face integrals

    do_boundary_integral(fe_eval, OperatorType::inhomogeneous, data.get_boundary_id(face));

    fe_eval.integrate_scatter(this->operator_data.face_integrate.value,
                              this->operator_data.face_integrate.gradient,
                              dst);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::boundary_face_loop_full_operator(
  MatrixFree<dim, Number> const & data,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   range) const
{
  FEEvalFace fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

  for(unsigned int face = range.first; face < range.second; face++)
  {
    fe_eval.reinit(face);

    fe_eval.gather_evaluate(src,
                            this->operator_data.face_evaluate.value,
                            this->operator_data.face_evaluate.gradient);

    do_boundary_integral(fe_eval, OperatorType::full, data.get_boundary_id(face));

    fe_eval.integrate_scatter(this->operator_data.face_integrate.value,
                              this->operator_data.face_integrate.gradient,
                              dst);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::cell_loop_diagonal(
  MatrixFree<dim, Number> const & data,
  VectorType &                    dst,
  VectorType const & /*src*/,
  Range const & range) const
{
  FEEvalCell fe_eval(data, operator_data.dof_index, operator_data.quad_index);

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    // create temporal array for local diagonal
    VectorizedArray<Number> local_diag[dofs_per_cell];

    fe_eval.reinit(cell);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of FEEvaluation
      this->create_standard_basis(j, fe_eval);

      fe_eval.evaluate(this->operator_data.cell_evaluate.value,
                       this->operator_data.cell_evaluate.gradient,
                       this->operator_data.cell_evaluate.hessians);

      this->do_cell_integral(fe_eval);

      fe_eval.integrate(this->operator_data.cell_integrate.value,
                        this->operator_data.cell_integrate.gradient);

      // extract single value from result vector and temporally store it
      local_diag[j] = fe_eval.begin_dof_values()[j];
    }
    // copy local diagonal entries into dof values of FEEvaluation ...
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      fe_eval.begin_dof_values()[j] = local_diag[j];

    // ... and write it back to global vector
    fe_eval.distribute_local_to_global(dst);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::face_loop_diagonal(
  MatrixFree<dim, Number> const & data,
  VectorType &                    dst,
  VectorType const & /*src*/,
  Range const & range) const
{
  FEEvalFace fe_eval_m(data, true, operator_data.dof_index, operator_data.quad_index);
  FEEvalFace fe_eval_p(data, false, operator_data.dof_index, operator_data.quad_index);

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    VectorizedArray<Number> local_diag[dofs_per_cell];

    fe_eval_m.reinit(cell);
    fe_eval_p.reinit(cell);

    // interior face
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, fe_eval_m);

      fe_eval_m.evaluate(this->operator_data.face_evaluate.value,
                         this->operator_data.face_evaluate.gradient);

      this->do_face_int_integral(fe_eval_m, fe_eval_p);

      fe_eval_m.integrate(this->operator_data.face_integrate.value,
                          this->operator_data.face_integrate.gradient);

      local_diag[j] = fe_eval_m.begin_dof_values()[j];
    }

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      fe_eval_m.begin_dof_values()[j] = local_diag[j];

    fe_eval_m.distribute_local_to_global(dst);

    // exterior face
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, fe_eval_p);

      fe_eval_p.evaluate(this->operator_data.face_evaluate.value,
                         this->operator_data.face_evaluate.gradient);

      this->do_face_ext_integral(fe_eval_m, fe_eval_p);

      fe_eval_p.integrate(this->operator_data.face_integrate.value,
                          this->operator_data.face_integrate.gradient);

      local_diag[j] = fe_eval_p.begin_dof_values()[j];
    }

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      fe_eval_p.begin_dof_values()[j] = local_diag[j];

    fe_eval_p.distribute_local_to_global(dst);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::boundary_face_loop_diagonal(
  MatrixFree<dim, Number> const & data,
  VectorType &                    dst,
  VectorType const & /*src*/,
  Range const & range) const
{
  FEEvalFace fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

  for(unsigned int face = range.first; face < range.second; face++)
  {
    VectorizedArray<Number> local_diag[dofs_per_cell];

    auto bid = data.get_boundary_id(face);

    fe_eval.reinit(face);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, fe_eval);

      fe_eval.evaluate(this->operator_data.face_evaluate.value,
                       this->operator_data.face_evaluate.gradient);

      this->do_boundary_integral(fe_eval, OperatorType::homogeneous, bid);

      fe_eval.integrate(this->operator_data.face_integrate.value,
                        this->operator_data.face_integrate.gradient);

      local_diag[j] = fe_eval.begin_dof_values()[j];
    }

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      fe_eval.begin_dof_values()[j] = local_diag[j];

    fe_eval.distribute_local_to_global(dst);
  }
}


template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::cell_based_loop_diagonal(
  MatrixFree<dim, Number> const & data,
  VectorType &                    dst,
  VectorType const & /*src*/,
  Range const & range) const
{
  FEEvalCell fe_eval(data, operator_data.dof_index, operator_data.quad_index);
  FEEvalFace fe_eval_m(data, true, operator_data.dof_index, operator_data.quad_index);
  FEEvalFace fe_eval_p(data, false, operator_data.dof_index, operator_data.quad_index);

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    VectorizedArray<Number> local_diag[dofs_per_cell];
    fe_eval.reinit(cell);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, fe_eval);

      fe_eval.evaluate(this->operator_data.cell_evaluate.value,
                       this->operator_data.cell_evaluate.gradient,
                       this->operator_data.cell_evaluate.hessians);

      this->do_cell_integral(fe_eval);

      fe_eval.integrate(this->operator_data.cell_integrate.value,
                        this->operator_data.cell_integrate.gradient);

      local_diag[j] = fe_eval.begin_dof_values()[j];
    }

    // loop over all faces and gather results into local diagonal local_diag
    unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
    for(unsigned int face = 0; face < n_faces; ++face)
    {
      fe_eval_m.reinit(cell, face);
      fe_eval_p.reinit(cell, face);
      auto bids = data.get_faces_by_cells_boundary_id(cell, face);
      auto bid  = bids[0];
#ifdef DEBUG
      unsigned int const n_filled_lanes = data.n_active_entries_per_cell_batch(cell);
      for(unsigned int v = 0; v < n_filled_lanes; v++)
        Assert(bid == bids[v],
               ExcMessage("Cell-based face loop encountered face batch with different bids."));
#endif

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        this->create_standard_basis(j, fe_eval_m);

        fe_eval_m.evaluate(this->operator_data.face_evaluate.value,
                           this->operator_data.face_evaluate.gradient);

        if(bid == numbers::internal_face_boundary_id) // internal face
          this->do_face_int_integral(fe_eval_m, fe_eval_p);
        else // boundary face
          this->do_boundary_integral(fe_eval_m, OperatorType::homogeneous, bid);

        fe_eval_m.integrate(this->operator_data.face_integrate.value,
                            this->operator_data.face_integrate.gradient);

        // note: += for accumulation of all contributions of this (macro) cell
        //          including: cell-, face-, boundary-stiffness matrix
        local_diag[j] += fe_eval_m.begin_dof_values()[j];
      }
    }

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      fe_eval.begin_dof_values()[j] = local_diag[j];

    fe_eval.distribute_local_to_global(dst);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::
  cell_loop_apply_inverse_block_diagonal_matrix_based(MatrixFree<dim, Number> const & data,
                                                      VectorType &                    dst,
                                                      VectorType const &              src,
                                                      Range const & cell_range) const
{
  FEEvalCell fe_eval(data, operator_data.dof_index, operator_data.quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    fe_eval.reinit(cell);
    fe_eval.read_dof_values(src);

    for(unsigned int v = 0; v < vectorization_length; ++v)
    {
      Vector<Number> src_vector(fe_eval.dofs_per_cell);
      for(unsigned int j = 0; j < fe_eval.dofs_per_cell; ++j)
        src_vector(j) = fe_eval.begin_dof_values()[j][v];

      // apply inverse matrix
      matrices[cell * vectorization_length + v].solve(src_vector, false);

      for(unsigned int j = 0; j < fe_eval.dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j][v] = src_vector(j);
    }

    fe_eval.set_dof_values(dst);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::cell_loop_apply_block_diagonal_matrix_based(
  MatrixFree<dim, Number> const & data,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   range) const
{
  FEEvalCell fe_eval(data, operator_data.dof_index, operator_data.quad_index);

  for(unsigned int cell = range.first; cell < range.second; ++cell)
  {
    fe_eval.reinit(cell);
    fe_eval.read_dof_values(src);

    for(unsigned int v = 0; v < vectorization_length; ++v)
    {
      Vector<Number> src_vector(fe_eval.dofs_per_cell);
      Vector<Number> dst_vector(fe_eval.dofs_per_cell);
      for(unsigned int j = 0; j < fe_eval.dofs_per_cell; ++j)
        src_vector(j) = fe_eval.begin_dof_values()[j][v];

      // apply matrix
      matrices[cell * vectorization_length + v].vmult(dst_vector, src_vector, false);

      for(unsigned int j = 0; j < fe_eval.dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j][v] = dst_vector(j);
    }

    fe_eval.set_dof_values(dst);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::cell_loop_block_diagonal(
  MatrixFree<dim, Number> const & data,
  BlockMatrix &                   matrices,
  BlockMatrix const &,
  Range const & range) const
{
  FEEvalCell fe_eval(data, operator_data.dof_index, operator_data.quad_index);

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    unsigned int const n_filled_lanes = data.n_active_entries_per_cell_batch(cell);
    fe_eval.reinit(cell);
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, fe_eval);

      fe_eval.evaluate(this->operator_data.cell_evaluate.value,
                       this->operator_data.cell_evaluate.gradient,
                       this->operator_data.cell_evaluate.hessians);

      this->do_cell_integral(fe_eval);

      fe_eval.integrate(this->operator_data.cell_integrate.value,
                        this->operator_data.cell_integrate.gradient);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices[cell * vectorization_length + v](i, j) += fe_eval.begin_dof_values()[i][v];
    }
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::face_loop_block_diagonal(
  MatrixFree<dim, Number> const & data,
  BlockMatrix &                   matrices,
  BlockMatrix const &,
  Range const & range) const
{
  FEEvalFace fe_eval_m(data, true, operator_data.dof_index, operator_data.quad_index);
  FEEvalFace fe_eval_p(data, false, operator_data.dof_index, operator_data.quad_index);

  for(auto face = range.first; face < range.second; ++face)
  {
    unsigned int const n_filled_lanes = data.n_active_entries_per_face_batch(face);

    fe_eval_m.reinit(face);
    fe_eval_p.reinit(face);

    // interior face
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, fe_eval_m);

      fe_eval_m.evaluate(this->operator_data.face_evaluate.value,
                         this->operator_data.face_evaluate.gradient);

      this->do_face_int_integral(fe_eval_m, fe_eval_p);

      fe_eval_m.integrate(this->operator_data.face_integrate.value,
                          this->operator_data.face_integrate.gradient);

      for(unsigned int v = 0; v < n_filled_lanes; ++v)
      {
        unsigned int const cell = data.get_face_info(face).cells_interior[v];
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          matrices[cell](i, j) += fe_eval_m.begin_dof_values()[i][v];
      }
    }

    // exterior face
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, fe_eval_p);

      fe_eval_p.evaluate(this->operator_data.face_evaluate.value,
                         this->operator_data.face_evaluate.gradient);

      this->do_face_ext_integral(fe_eval_m, fe_eval_p);

      fe_eval_p.integrate(this->operator_data.face_integrate.value,
                          this->operator_data.face_integrate.gradient);

      for(unsigned int v = 0; v < n_filled_lanes; ++v)
      {
        unsigned int const cell = data.get_face_info(face).cells_exterior[v];
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          matrices[cell](i, j) += fe_eval_p.begin_dof_values()[i][v];
      }
    }
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::boundary_face_loop_block_diagonal(
  MatrixFree<dim, Number> const & data,
  BlockMatrix &                   matrices,
  BlockMatrix const &,
  Range const & range) const
{
  FEEvalFace fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

  for(auto face = range.first; face < range.second; ++face)
  {
    unsigned int const n_filled_lanes = data.n_active_entries_per_face_batch(face);
    fe_eval.reinit(face);
    auto bid = data.get_boundary_id(face);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, fe_eval);

      fe_eval.evaluate(this->operator_data.face_evaluate.value,
                       this->operator_data.face_evaluate.gradient);

      this->do_boundary_integral(fe_eval, OperatorType::homogeneous, bid);

      fe_eval.integrate(this->operator_data.face_integrate.value,
                        this->operator_data.face_integrate.gradient);

      for(unsigned int v = 0; v < n_filled_lanes; ++v)
      {
        unsigned int const cell = data.get_face_info(face).cells_interior[v];
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          matrices[cell](i, j) += fe_eval.begin_dof_values()[i][v];
      }
    }
  }
}


template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::cell_based_loop_block_diagonal(
  MatrixFree<dim, Number> const & data,
  BlockMatrix &                   matrices,
  BlockMatrix const &,
  Range const & range) const
{
  FEEvalCell fe_eval(data, operator_data.dof_index, operator_data.quad_index);
  FEEvalFace fe_eval_m(data, true, operator_data.dof_index, operator_data.quad_index);
  FEEvalFace fe_eval_p(data, false, operator_data.dof_index, operator_data.quad_index);

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    unsigned int const n_filled_lanes = data.n_active_entries_per_cell_batch(cell);

    fe_eval.reinit(cell);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, fe_eval);

      fe_eval.evaluate(this->operator_data.cell_evaluate.value,
                       this->operator_data.cell_evaluate.gradient,
                       this->operator_data.cell_evaluate.hessians);

      this->do_cell_integral(fe_eval);

      fe_eval.integrate(this->operator_data.cell_integrate.value,
                        this->operator_data.cell_integrate.gradient);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices[cell * vectorization_length + v](i, j) += fe_eval.begin_dof_values()[i][v];
    }

    // loop over all faces
    unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
    for(unsigned int face = 0; face < n_faces; ++face)
    {
      fe_eval_m.reinit(cell, face);
      fe_eval_p.reinit(cell, face);
      auto bids = data.get_faces_by_cells_boundary_id(cell, face);
      auto bid  = bids[0];

#ifdef DEBUG
      for(unsigned int v = 0; v < n_filled_lanes; v++)
        Assert(bid == bids[v],
               ExcMessage("Cell-based face loop encountered face batch with different bids."));
#endif

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        this->create_standard_basis(j, fe_eval_m);

        fe_eval_m.evaluate(this->operator_data.face_evaluate.value,
                           this->operator_data.face_evaluate.gradient);

        if(bid == numbers::internal_face_boundary_id) // internal face
          this->do_face_int_integral(fe_eval_m, fe_eval_p);
        else // boundary face
          this->do_boundary_integral(fe_eval_m, OperatorType::homogeneous, bid);

        fe_eval_m.integrate(this->operator_data.face_integrate.value,
                            this->operator_data.face_integrate.gradient);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int v = 0; v < n_filled_lanes; ++v)
            matrices[cell * vectorization_length + v](i, j) += fe_eval_m.begin_dof_values()[i][v];
      }
    }
  }
}

#ifdef DEAL_II_WITH_TRILINOS
template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::cell_loop_calculate_system_matrix(
  MatrixFree<dim, Number> const & data,
  SparseMatrix &                  dst,
  SparseMatrix const & /*src*/,
  Range const & range) const
{
  FEEvalCell fe_eval(data, operator_data.dof_index, operator_data.quad_index);

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    unsigned int const n_filled_lanes = data.n_components_filled(cell);

    // create a temporal full matrix for the local element matrix of each ...
    // cell of each macro cell and ...
    FullMatrix_ matrices[vectorization_length];
    // set their size
    std::fill_n(matrices, vectorization_length, FullMatrix_(dofs_per_cell, dofs_per_cell));

    fe_eval.reinit(cell);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, fe_eval);

      fe_eval.evaluate(this->operator_data.cell_evaluate.value,
                       this->operator_data.cell_evaluate.gradient,
                       this->operator_data.cell_evaluate.hessians);

      this->do_cell_integral(fe_eval);

      fe_eval.integrate(this->operator_data.cell_integrate.value,
                        this->operator_data.cell_integrate.gradient);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices[v](i, j) = fe_eval.begin_dof_values()[i][v];
    }

    // finally assemble local matrices into global matrix
    for(unsigned int v = 0; v < n_filled_lanes; v++)
    {
      auto cell_v = data.get_cell_iterator(cell, v);

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
          dof_indices[j] = temp[data.get_shape_info().lexicographic_numbering[j]];
      }

      constraint->distribute_local_to_global(matrices[v], dof_indices, dof_indices, dst);
    }
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::face_loop_calculate_system_matrix(
  MatrixFree<dim, Number> const & data,
  SparseMatrix &                  dst,
  SparseMatrix const & /*src*/,
  Range const & range) const
{
  FEEvalFace fe_eval_m(data, true, operator_data.dof_index, operator_data.quad_index);
  FEEvalFace fe_eval_p(data, false, operator_data.dof_index, operator_data.quad_index);

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
    unsigned int const n_filled_lanes = data.n_active_entries_per_face_batch(face);

    fe_eval_m.reinit(face);
    fe_eval_p.reinit(face);

    // process minus trial function
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of first FEFaceEvaluation and
      // clear dof values of second FEFaceEvaluation
      this->create_standard_basis(j, fe_eval_m, fe_eval_p);

      fe_eval_m.evaluate(this->operator_data.face_evaluate.value,
                         this->operator_data.face_evaluate.gradient);
      fe_eval_p.evaluate(this->operator_data.face_evaluate.value,
                         this->operator_data.face_evaluate.gradient);

      this->do_face_integral(fe_eval_m, fe_eval_p);

      fe_eval_m.integrate(this->operator_data.face_integrate.value,
                          this->operator_data.face_integrate.gradient);
      fe_eval_p.integrate(this->operator_data.face_integrate.value,
                          this->operator_data.face_integrate.gradient);

      // insert result vector into local matrix u1_v1
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices_m[v](i, j) = fe_eval_m.begin_dof_values()[i][v];

      // insert result vector into local matrix  u1_v2
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices_p[v](i, j) = fe_eval_p.begin_dof_values()[i][v];
    }

    // save local matrices into global matrix
    for(unsigned int v = 0; v < n_filled_lanes; v++)
    {
      auto const cell_number_m = data.get_face_info(face).cells_interior[v];
      auto const cell_number_p = data.get_face_info(face).cells_exterior[v];

      auto cell_m = data.get_cell_iterator(cell_number_m / vectorization_length,
                                           cell_number_m % vectorization_length);
      auto cell_p = data.get_cell_iterator(cell_number_p / vectorization_length,
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
      this->create_standard_basis(j, fe_eval_p, fe_eval_m);

      fe_eval_m.evaluate(this->operator_data.face_evaluate.value,
                         this->operator_data.face_evaluate.gradient);
      fe_eval_p.evaluate(this->operator_data.face_evaluate.value,
                         this->operator_data.face_evaluate.gradient);

      this->do_face_integral(fe_eval_m, fe_eval_p);

      fe_eval_m.integrate(this->operator_data.face_integrate.value,
                          this->operator_data.face_integrate.gradient);
      fe_eval_p.integrate(this->operator_data.face_integrate.value,
                          this->operator_data.face_integrate.gradient);

      // insert result vector into local matrix M_mp
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices_m[v](i, j) = fe_eval_m.begin_dof_values()[i][v];

      // insert result vector into local matrix  M_pp
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices_p[v](i, j) = fe_eval_p.begin_dof_values()[i][v];
    }

    // save local matrices into global matrix
    for(unsigned int v = 0; v < n_filled_lanes; v++)
    {
      auto const cell_number_m = data.get_face_info(face).cells_interior[v];
      auto const cell_number_p = data.get_face_info(face).cells_exterior[v];

      auto cell_m = data.get_cell_iterator(cell_number_m / vectorization_length,
                                           cell_number_m % vectorization_length);
      auto cell_p = data.get_cell_iterator(cell_number_p / vectorization_length,
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

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::boundary_face_loop_calculate_system_matrix(
  MatrixFree<dim, Number> const & data,
  SparseMatrix &                  dst,
  SparseMatrix const & /*src*/,
  Range const & range) const
{
  FEEvalFace fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

  for(auto face = range.first; face < range.second; ++face)
  {
    unsigned int const n_filled_lanes = data.n_active_entries_per_face_batch(face);

    // create temporary matrices for local blocks
    FullMatrix_ matrices[vectorization_length];
    std::fill_n(matrices, vectorization_length, FullMatrix_(dofs_per_cell, dofs_per_cell));

    fe_eval.reinit(face);
    auto bid = data.get_boundary_id(face);

    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      this->create_standard_basis(j, fe_eval);

      fe_eval.evaluate(this->operator_data.face_evaluate.value,
                       this->operator_data.face_evaluate.gradient);

      this->do_boundary_integral(fe_eval, OperatorType::homogeneous, bid);

      fe_eval.integrate(this->operator_data.face_integrate.value,
                        this->operator_data.face_integrate.gradient);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices[v](i, j) = fe_eval.begin_dof_values()[i][v];
    }

    // save local matrices into global matrix
    for(unsigned int v = 0; v < n_filled_lanes; v++)
    {
      unsigned int const cell_number = data.get_face_info(face).cells_interior[v];

      auto cell_v = data.get_cell_iterator(cell_number / vectorization_length,
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

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::adjust_diagonal_for_singular_operator(
  VectorType & diagonal) const
{
  VectorType vec1, d;
  vec1.reinit(diagonal, true);
  d.reinit(diagonal, true);
  for(unsigned int i = 0; i < vec1.local_size(); ++i)
    vec1.local_element(i) = 1.;
  apply(d, vec1);
  double length = vec1 * vec1;
  double factor = vec1 * d;
  diagonal.add(-2. / length, d, factor / pow(length, 2.), vec1);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::set_constraint_diagonal(
  VectorType & diagonal) const
{
  // set (diagonal) entries to 1.0 for constrained dofs
  for(auto i : data->get_constrained_dofs())
    diagonal.local_element(i) = 1.0;
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::add_constraints(
  DoFHandler<dim> const &     dof_handler,
  AffineConstraints<double> & constraint_own,
  MGConstrainedDoFs const &   mg_constrained_dofs,
  AdditionalData &            operator_data,
  unsigned int const          level)
{
  // 0) clear old content (to be on the safe side)
  constraint_own.clear();

  // 1) add periodic BCs
  this->add_periodicity_constraints(dof_handler,
                                    level,
                                    operator_data.periodic_face_pairs_level0,
                                    constraint_own);

  // 2) add Dirichlet BCs
  constraint_own.add_lines(mg_constrained_dofs.get_boundary_indices(level));

  // constrain zeroth DoF in continuous case (the mean value constraint will
  // be applied in the DG case). In case we have interface matrices, there are
  // Dirichlet constraints on parts of the boundary and no such transformation
  // is required.
  if(operator_is_singular() && constraint_own.can_store_line(0))
  {
    // if dof 0 is constrained, it must be a periodic dof, so we take the
    // value on the other side
    types::global_dof_index line_index = 0;
    while(true)
    {
      std::vector<std::pair<types::global_dof_index, double>> const * lines =
        constraint_own.get_constraint_entries(line_index);

      if(lines == 0)
      {
        constraint_own.add_line(line_index);
        // add the constraint back to the MGConstrainedDoFs field. This
        // is potentially dangerous but we know what we are doing... ;-)
        if(level != numbers::invalid_unsigned_int)
        {
          const_cast<IndexSet &>(mg_constrained_dofs.get_boundary_indices(level))
            .add_index(line_index);
        }

        break;
      }
      else
      {
        Assert(lines->size() == 1 && std::abs((*lines)[0].second - 1.) < 1e-15,
               ExcMessage("Periodic index expected, bailing out"));

        line_index = (*lines)[0].first;
      }
    }
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::add_periodicity_constraints(
  DoFHandler<dim> const &                 dof_handler,
  unsigned int const                      level,
  std::vector<PeriodicFacePairIterator> & periodic_face_pairs_level0,
  AffineConstraints<double> &             constraint_own)
{
  // loop over all periodic face pairs of level 0
  for(auto & it : periodic_face_pairs_level0)
  {
    // get reference to the cells on level 0 sharing the periodic face
    typename DoFHandler<dim>::cell_iterator cell1(&dof_handler.get_triangulation(),
                                                  0,
                                                  it.cell[1]->index(),
                                                  &dof_handler);
    typename DoFHandler<dim>::cell_iterator cell0(&dof_handler.get_triangulation(),
                                                  0,
                                                  it.cell[0]->index(),
                                                  &dof_handler);

    // get reference to periodic faces on level and add recursively their
    // subfaces on the given level
    add_periodicity_constraints(
      level, level, cell1->face(it.face_idx[1]), cell0->face(it.face_idx[0]), constraint_own);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::add_periodicity_constraints(
  unsigned int const                            level,
  unsigned int const                            target_level,
  typename DoFHandler<dim>::face_iterator const face1,
  typename DoFHandler<dim>::face_iterator const face2,
  AffineConstraints<double> &                   constraints)
{
  if(level == 0)
  {
    // level of interest has been reached
    unsigned int const dofs_per_face = face1->get_fe(0).dofs_per_face;

    std::vector<types::global_dof_index> dofs_1(dofs_per_face);
    std::vector<types::global_dof_index> dofs_2(dofs_per_face);

    face1->get_mg_dof_indices(target_level, dofs_1, 0);
    face2->get_mg_dof_indices(target_level, dofs_2, 0);

    for(unsigned int i = 0; i < dofs_per_face; ++i)
    {
      if(constraints.can_store_line(dofs_2[i]) && constraints.can_store_line(dofs_1[i]) &&
         !constraints.is_constrained(dofs_2[i]))
      {
        // constraint dof and ...
        constraints.add_line(dofs_2[i]);
        // specify type of constraint: equality (dof_2[i]=dof_1[j]*1.0)
        constraints.add_entry(dofs_2[i], dofs_1[i], 1.);
      }
    }
  }
  else if(face1->has_children() && face2->has_children())
  {
    // recursively visit all subfaces
    for(unsigned int c = 0; c < face1->n_children(); ++c)
    {
      add_periodicity_constraints(
        level - 1, target_level, face1->child(c), face2->child(c), constraints);
    }
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::verify_boundary_conditions(
  DoFHandler<dim> const & dof_handler,
  AdditionalData const &  operator_data) const
{
  // fill set with periodic boundary ids
  std::set<types::boundary_id> periodic_boundary_ids;
  for(unsigned int i = 0; i < operator_data.periodic_face_pairs_level0.size(); ++i)
  {
    AssertThrow(operator_data.periodic_face_pairs_level0[i].cell[0]->level() == 0,
                ExcMessage("Received periodic cell pairs on non-zero level"));
    periodic_boundary_ids.insert(operator_data.periodic_face_pairs_level0[i]
                                   .cell[0]
                                   ->face(operator_data.periodic_face_pairs_level0[i].face_idx[0])
                                   ->boundary_id());
    periodic_boundary_ids.insert(operator_data.periodic_face_pairs_level0[i]
                                   .cell[1]
                                   ->face(operator_data.periodic_face_pairs_level0[i].face_idx[1])
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
        do_verify_boundary_conditions(bid, operator_data, periodic_boundary_ids);
      }
    }
  }
}
