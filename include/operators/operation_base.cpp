#include "operation_base.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/distributed/tria.h>

#include "../solvers_and_preconditioners/util/block_jacobi_matrices.h"
#include "../solvers_and_preconditioners/util/invert_diagonal.h"

#include "../functionalities/set_zero_mean_value.h"

template<int dim, int degree, typename Number, typename AdditionalData>
OperatorBase<dim, degree, Number, AdditionalData>::OperatorBase()
  : operator_settings(AdditionalData()),
    data(),
    eval_time(0.0),
    do_eval_faces(
      operator_settings.internal_evaluate.do_eval() || operator_settings.internal_integrate.do_eval() ||
      operator_settings.boundary_evaluate.do_eval() || operator_settings.boundary_integrate.do_eval()),
    level_mg_handler(numbers::invalid_unsigned_int),
    block_jacobi_matrices_have_been_initialized(false)
{
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::reinit(MatrixFree_ const &             matrix_free,
                                                          ConstraintMatrix &              constraint_matrix,
                                                          AdditionalData const & operator_settings,
                                                          unsigned int           level_mg_handler) const
{
  // reinit data structures
  this->data.reinit(matrix_free);
  this->constraint.reinit(constraint_matrix);
  this->operator_settings = operator_settings;

  // check if dg or cg
  is_dg = data->get_dof_handler(operator_settings.dof_index).get_fe().dofs_per_vertex == 0;

  // set mg level
  this->level_mg_handler = level_mg_handler;

  // The default value is is_mg = false and this variable is set to true in case 
  // the operator is applied in multigrid algorithm. By convention, the default 
  // argument numbers::invalid_unsigned_int corresponds to the default 
  // value is_mg = false
  this->is_mg = level_mg_handler != numbers::invalid_unsigned_int;

}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::reinit(const DoFHandler<dim> &   dof_handler,
                                                          const Mapping<dim> &      mapping,
                                                          void *                    operator_data_in,
                                                          const MGConstrainedDoFs & mg_constrained_dofs,
                                                          const unsigned int        level_mg_handler)
{
  // create copy of data and ...
  auto operator_settings = *static_cast<AdditionalData *>(operator_data_in);
  // set dof_index and quad_index to 0 since we only consider a subset
  operator_settings.dof_index  = 0;
  operator_settings.quad_index = 0;

  // check it dg or cg
  is_dg = dof_handler.get_fe().dofs_per_vertex == 0;

  // setup MatrixFree::AdditionalData
  typename MatrixFree<dim, Number>::AdditionalData additional_data;

  // ... shall the faces be evaluated?
  if(is_dg)
    additional_data.build_face_info = true;

  // ... level of this mg level
  additional_data.level_mg_handler = level_mg_handler;

  // TODO: move to additional data
  additional_data.mapping_update_flags =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors | update_values);

  if(is_dg)
  {
    additional_data.mapping_update_flags_inner_faces =
      (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);

    additional_data.mapping_update_flags_boundary_faces =
      (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);
  }

  // ... on each level
  auto & constraint_own = constraint.own();
  auto & data_own       = data.own();

  // setup constraint matrix for CG
  if(!is_dg)
    this->add_constraints(dof_handler, constraint_own, mg_constrained_dofs, operator_settings, level_mg_handler);

  // ...finalize constraint matrix
  constraint_own.close();

  const QGauss<1> quad(dof_handler.get_fe().degree + 1);
  data_own.reinit(mapping, dof_handler, constraint_own, quad, additional_data);
  reinit(data_own, constraint_own, operator_settings, level_mg_handler);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::vmult(VectorType & dst, VectorType const & src) const
{
  dst = 0;
  vmult_add(dst, src);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::apply(VectorType & dst, VectorType const & src) const
{
  vmult(dst, src);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::vmult_add(VectorType & dst, VectorType const & src) const
{
  const VectorType * actual_src = &src;
  VectorType         tmp_projection_vector;
  if(this->is_singular() && !is_mg)
  {
    tmp_projection_vector = src;
    set_zero_mean_value(tmp_projection_vector);
    actual_src = &tmp_projection_vector;
  }

  if(is_dg && do_eval_faces)
    data->loop(
      &This::local_cell_hom, &This::local_face_hom, &This::local_boundary_hom, this, dst, *actual_src);
  else
    data->cell_loop(&This::local_cell_hom, this, dst, *actual_src);

  if(this->is_singular() && !is_mg)
    set_zero_mean_value(dst);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::apply_add(VectorType & dst, VectorType const & src) const
{
  vmult_add(dst, src);
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
OperatorBase<dim, degree, Number, AdditionalData>::vmult_interface_down(VectorType &       dst,
                                                                        VectorType const & src) const
{
  vmult(dst, src);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::vmult_add_interface_up(VectorType &       dst,
                                                                          VectorType const & src) const
{
  vmult_add(dst, src);
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

  data->loop(&This::local_cell_inhom, &This::local_face_inhom, &This::local_boundary_inhom, this, tmp, tmp);

  // multiply by -1.0 since the boundary face integrals have to be shifted
  // to the right hand side
  dst.add(-1.0, tmp);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::rhs_add(VectorType & dst, Number const time) const
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

  data->loop(&This::local_cell_hom, &This::local_face_hom, &This::local_boundary_full, this, dst, src);
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
  // compute diagonal (not regarding: mean value constraint and constraints)
  if(is_dg && do_eval_faces)
    if(operator_settings.use_cell_based_loops)
      data->cell_loop(&This::local_add_diagonal_cell_based, this, diagonal, diagonal);
    else
      data->loop(&This::local_add_diagonal_cell,
                 &This::local_add_diagonal_face,
                 &This::local_add_diagonal_boundary,
                 this,
                 diagonal,
                 diagonal);
  else
    data->cell_loop(&This::local_add_diagonal_cell, this, diagonal, diagonal);

  // multiple processes might have contributions to the same diagonal entry in
  // the cg case, so we have to sum them up
  if(!is_dg)
    diagonal.compress(VectorOperation::add);

  // in case that the operator is singular, the diagonal has to be adjusted
  if(this->is_singular() && !is_mg)
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
OperatorBase<dim, degree, Number, AdditionalData>::calculate_inverse_diagonal(VectorType & diagonal) const
{
  calculate_diagonal(diagonal);
  invert_diagonal(diagonal);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::apply_block_jacobi(VectorType &       dst,
                                                                      VectorType const & src) const
{
  AssertThrow(is_dg, ExcMessage("Block Jacobi only implemented for DG!"));
  AssertThrow(block_jacobi_matrices_have_been_initialized,
              ExcMessage("Block Jacobi matrices have not been initialized!"));

  data->cell_loop(&This::local_apply_block_jacobi, this, dst, src);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::apply_block_diagonal(VectorType &       dst,
                                                                        VectorType const & src) const
{
  AssertThrow(is_dg, ExcMessage("Block Jacobi only implemented for DG!"));
  AssertThrow(block_jacobi_matrices_have_been_initialized,
              ExcMessage("Block Jacobi matrices have not been initialized!"));

  data->cell_loop(&This::local_apply_block_diagonal, this, dst, src);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::update_block_jacobi() const
{
  this->calculate_block_diagonal_matrices();
  // perform lu factorization for block matrices
  calculate_lu_factorization_block_jacobi(matrices);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::calculate_block_diagonal_matrices() const
{
  AssertThrow(is_dg, ExcMessage("Block Jacobi only implemented for DG!"));

  // allocate memory only the first time
  if(block_jacobi_matrices_have_been_initialized == false)
  {
    auto dofs = data->get_shape_info().dofs_per_component_on_cell;
    matrices.resize(data->n_macro_cells() * vectorization_length, LAPACKFullMatrix<Number>(dofs, dofs));
    block_jacobi_matrices_have_been_initialized = true;
  } // else: reuse old memory

  // clear matrices
  initialize_block_jacobi_matrices_with_zero(matrices);

  // compute block matrices
  add_block_jacobi_matrices(matrices);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::add_block_jacobi_matrices(BlockMatrix & matrices) const
{
  AssertThrow(is_dg, ExcMessage("Block Jacobi only implemented for DG!"));
    
  if(do_eval_faces)
    if(operator_settings.use_cell_based_loops)
      data->cell_loop(&This::local_add_block_diagonal_cell_based, this, matrices, matrices);
    else
      data->loop(&This::local_add_block_diagonal_cell,
                 &This::local_add_block_diagonal_face,
                 &This::local_add_block_diagonal_boundary,
                 this,
                 matrices,
                 matrices);
  else
    data->cell_loop(&This::local_add_block_diagonal_cell, this, matrices, matrices);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::add_block_jacobi_matrices(BlockMatrix &    matrices,
                                                                             Number const time) const
{
  this->set_evaluation_time(time);
  this->add_block_jacobi_matrices(matrices);
}

#ifdef DEAL_II_WITH_TRILINOS
template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::init_system_matrix(SparseMatrix & system_matrix) const
{
  const DoFHandler<dim> & dof_handler = this->data->get_dof_handler();
  MPI_Comm                comm;

  // extract communicator
  {
    auto tria = dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation());

    if(tria != NULL)
      comm = tria->get_communicator();
    else // not distributed triangulation
      comm = MPI_COMM_SELF;
  }

  TrilinosWrappers::SparsityPattern dsp( is_mg ? dof_handler.locally_owned_mg_dofs(this->level_mg_handler) : dof_handler.locally_owned_dofs(), comm);
  
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
OperatorBase<dim, degree, Number, AdditionalData>::calculate_system_matrix(SparseMatrix & system_matrix) const
{
  // assemble matrix locally on each process
  if(do_eval_faces && is_dg)
    data->loop(&This::local_calculate_system_matrix_cell,
               &This::local_calculate_system_matrix_face,
               &This::local_calculate_system_matrix_boundary,
               this,
               system_matrix,
               system_matrix);
  else
    data->cell_loop(&This::local_calculate_system_matrix_cell, this, system_matrix, system_matrix);

  // communicate overlapping matrix parts
  system_matrix.compress(VectorOperation::add);

  if(!is_dg)
  {
    // make zero entries on diagonal (due to constrained dofs) to one:
    for(auto & entry : system_matrix)
      if(entry.row() == entry.column() && entry.value() == 0.0)
        entry.value() = 1.0;
  } // nothing to do for dg

}
#endif

template<int dim, int degree, typename Number, typename AdditionalData>
types::global_dof_index
OperatorBase<dim, degree, Number, AdditionalData>::m() const
{
  return data->get_vector_partitioner(operator_settings.dof_index)->size();
}

template<int dim, int degree, typename Number, typename AdditionalData>
types::global_dof_index
OperatorBase<dim, degree, Number, AdditionalData>::n() const
{
  return data->get_vector_partitioner(operator_settings.dof_index)->size();
}

template<int dim, int degree, typename Number, typename AdditionalData>
Number
OperatorBase<dim, degree, Number, AdditionalData>::el(const unsigned int, const unsigned int) const
{
  AssertThrow(false, ExcMessage("Matrix-free does not allow for entry access"));
  return Number();
}

template<int dim, int degree, typename Number, typename AdditionalData>
bool
OperatorBase<dim, degree, Number, AdditionalData>::is_empty_locally() const
{
  return data->n_macro_cells() == 0;
}

template<int dim, int degree, typename Number, typename AdditionalData>
const MatrixFree<dim, Number> &
OperatorBase<dim, degree, Number, AdditionalData>::get_data() const
{
  return *data;
}

template<int dim, int degree, typename Number, typename AdditionalData>
unsigned int
OperatorBase<dim, degree, Number, AdditionalData>::get_dof_index() const
{
  return operator_settings.dof_index;
}

template<int dim, int degree, typename Number, typename AdditionalData>
unsigned int
OperatorBase<dim, degree, Number, AdditionalData>::get_quad_index() const
{
  return operator_settings.quad_index;
}

template<int dim, int degree, typename Number, typename AdditionalData>
AdditionalData const &
OperatorBase<dim, degree, Number, AdditionalData>::get_operator_data() const
{
  return operator_settings;
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::initialize_dof_vector(VectorType & vector) const
{
  data->initialize_dof_vector(vector, operator_settings.dof_index);
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
bool
OperatorBase<dim, degree, Number, AdditionalData>::is_singular() const
{
  return this->operator_settings.operator_is_singular;
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::create_standard_basis(unsigned int j,
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
OperatorBase<dim, degree, Number, AdditionalData>::local_cell_hom(const MatrixFree_ &         data,
                                                                  VectorType &       dst,
                                                                  const VectorType & src,
                                                                  const Range &      range) const
{
  FEEvalCell fe_eval(data, operator_settings.dof_index, operator_settings.quad_index);
  // loop over the range of macro cells
  for(auto cell = range.first; cell < range.second; ++cell)
  {
    // reinit cell
    fe_eval.reinit(cell);
    // reoperator_settings dof values from global vector src and evaluate
    fe_eval.gather_evaluate(src,
                            this->operator_settings.cell_evaluate.value,
                            this->operator_settings.cell_evaluate.gradient,
                            this->operator_settings.cell_evaluate.hessians);
    // perform local vmult
    this->do_cell_integral(fe_eval);
    // integrate and write result back to the global vector dst
    fe_eval.integrate_scatter(this->operator_settings.cell_integrate.value,
                              this->operator_settings.cell_integrate.gradient,
                              dst);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::local_face_hom(const MatrixFree_ &         data,
                                                                  VectorType &       dst,
                                                                  const VectorType & src,
                                                                  const Range &      range) const
{
  FEEvalFace fe_eval_m(data, true, operator_settings.dof_index, operator_settings.quad_index);
  FEEvalFace fe_eval_p(data, false, operator_settings.dof_index, operator_settings.quad_index);
  // loop over the range of macro cells
  for(auto face = range.first; face < range.second; ++face)
  {
    // reinit cell
    fe_eval_m.reinit(face);
    fe_eval_p.reinit(face);
    // read dof values from global vector src
    fe_eval_m.gather_evaluate(src,
                              this->operator_settings.internal_evaluate.value,
                              this->operator_settings.internal_evaluate.gradient);
    fe_eval_p.gather_evaluate(src,
                              this->operator_settings.internal_evaluate.value,
                              this->operator_settings.internal_evaluate.gradient);
    // perform local vmult
    this->do_face_integral(fe_eval_m, fe_eval_p);
    // write result back to the global vector dst
    fe_eval_m.integrate_scatter(this->operator_settings.internal_integrate.value,
                                this->operator_settings.internal_integrate.gradient,
                                dst);
    fe_eval_p.integrate_scatter(this->operator_settings.internal_integrate.value,
                                this->operator_settings.internal_integrate.gradient,
                                dst);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::local_boundary_hom(const MatrixFree_ &         data,
                                                                      VectorType &       dst,
                                                                      const VectorType & src,
                                                                      const Range &      range) const
{
  FEEvalFace fe_eval(data, true, operator_settings.dof_index, operator_settings.quad_index);

  for(unsigned int face = range.first; face < range.second; face++)
  {
    auto bid = data.get_boundary_id(face);

    fe_eval.reinit(face);
    fe_eval.gather_evaluate(src,
                            this->operator_settings.boundary_evaluate.value,
                            this->operator_settings.boundary_evaluate.gradient);
    do_boundary_integral(fe_eval, OperatorType::homogeneous, bid);
    fe_eval.integrate_scatter(this->operator_settings.boundary_integrate.value,
                              this->operator_settings.boundary_integrate.gradient,
                              dst);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::local_boundary_inhom(const MatrixFree_ &   data,
                                                                        VectorType & dst,
                                                                        const VectorType & /*src*/,
                                                                        const Range & range) const
{
  FEEvalFace fe_eval(data, true, operator_settings.dof_index, operator_settings.quad_index);

  for(unsigned int face = range.first; face < range.second; face++)
  {
    auto bid = data.get_boundary_id(face);
    fe_eval.reinit(face);
    // note: no gathering/evaluation is necessary in the case of
    //       inhomogeneous boundary
    do_boundary_integral(fe_eval, OperatorType::inhomogeneous, bid);
    fe_eval.integrate_scatter(this->operator_settings.boundary_integrate.value,
                              this->operator_settings.boundary_integrate.gradient,
                              dst);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::local_boundary_full(const MatrixFree_ &         data,
                                                                       VectorType &       dst,
                                                                       const VectorType & src,
                                                                       const Range &      range) const
{
  FEEvalFace fe_eval(data, true, operator_settings.dof_index, operator_settings.quad_index);

  for(unsigned int face = range.first; face < range.second; face++)
  {
    auto bid = data.get_boundary_id(face);
    fe_eval.reinit(face);
    fe_eval.gather_evaluate(src,
                            this->operator_settings.boundary_evaluate.value,
                            this->operator_settings.boundary_evaluate.gradient);
    do_boundary_integral(fe_eval, OperatorType::full, bid);
    fe_eval.integrate_scatter(this->operator_settings.boundary_integrate.value,
                              this->operator_settings.boundary_integrate.gradient,
                              dst);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::local_add_diagonal_cell(const MatrixFree_ &   data,
                                                                           VectorType & dst,
                                                                           const VectorType & /*src*/,
                                                                           const Range & range) const
{
  FEEvalCell fe_eval(data, operator_settings.dof_index, operator_settings.quad_index);
  // loop over the range of macro cells
  for(auto cell = range.first; cell < range.second; ++cell)
  {
    // create temporal array for local diagonal
    VectorizedArray<Number> local_diag[dofs_per_cell];
    // reinit cell
    fe_eval.reinit(cell);
    // loop over all standard basis
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of FEEvaluation
      this->create_standard_basis(j, fe_eval);
      fe_eval.evaluate(this->operator_settings.cell_evaluate.value,
                       this->operator_settings.cell_evaluate.gradient,
                       this->operator_settings.cell_evaluate.hessians);
      // perform local vmult
      this->do_cell_integral(fe_eval);

      fe_eval.integrate(this->operator_settings.cell_integrate.value,
                        this->operator_settings.cell_integrate.gradient);
      // extract single value from result vector and temporally store it
      local_diag[j] = fe_eval.begin_dof_values()[j];
    }
    // copy local diagonal entries into dof values of FEEvaluation and ...
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      fe_eval.begin_dof_values()[j] = local_diag[j];
    // ... write it back to global vector
    fe_eval.distribute_local_to_global(dst);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::local_add_diagonal_face(const MatrixFree_ &   data,
                                                                           VectorType & dst,
                                                                           const VectorType & /*src*/,
                                                                           const Range & range) const
{
  FEEvalFace fe_eval_m(data, true, operator_settings.dof_index, operator_settings.quad_index);
  FEEvalFace fe_eval_p(data, false, operator_settings.dof_index, operator_settings.quad_index);

  // loop over the range of macro cells
  for(auto cell = range.first; cell < range.second; ++cell)
  {
    // create temporal array for local diagonal
    VectorizedArray<Number> local_diag[dofs_per_cell];
    // reinit cell
    fe_eval_m.reinit(cell);
    fe_eval_p.reinit(cell);

    // interior: loop over all standard basis
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of FEEvaluation
      this->create_standard_basis(j, fe_eval_m);
      // perform local vmult
      fe_eval_m.evaluate(this->operator_settings.internal_evaluate.value,
                         this->operator_settings.internal_evaluate.gradient);
      this->do_face_int_integral(fe_eval_m, fe_eval_p);
      fe_eval_m.integrate(this->operator_settings.internal_integrate.value,
                          this->operator_settings.internal_integrate.gradient);

      // extract single value from result vector and temporally store it
      local_diag[j] = fe_eval_m.begin_dof_values()[j];
    }
    // copy local diagonal entries into dof values of FEEvaluation and ...
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      fe_eval_m.begin_dof_values()[j] = local_diag[j];
    // ... write it back to global vector
    fe_eval_m.distribute_local_to_global(dst);

    // exterior: loop over all standard basis
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of FEEvaluation
      this->create_standard_basis(j, fe_eval_p);
      // perform local vmult
      fe_eval_p.evaluate(this->operator_settings.internal_evaluate.value,
                         this->operator_settings.internal_evaluate.gradient);
      this->do_face_ext_integral(fe_eval_m, fe_eval_p);
      fe_eval_p.integrate(this->operator_settings.internal_integrate.value,
                          this->operator_settings.internal_integrate.gradient);
      // extract single value from result vector and temporally store it
      local_diag[j] = fe_eval_p.begin_dof_values()[j];
    }
    // copy local diagonal entries into dof values of FEEvaluation and ...
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      fe_eval_p.begin_dof_values()[j] = local_diag[j];
    // ... write it back to global vector
    fe_eval_p.distribute_local_to_global(dst);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::local_add_diagonal_boundary(const MatrixFree_ &   data,
                                                                               VectorType & dst,
                                                                               const VectorType & /*src*/,
                                                                               const Range & range) const
{
  FEEvalFace fe_eval(data, true, operator_settings.dof_index, operator_settings.quad_index);

  for(unsigned int face = range.first; face < range.second; face++)
  {
    // create temporal array for local diagonal
    VectorizedArray<Number> local_diag[dofs_per_cell];
    auto                    bid = data.get_boundary_id(face);

    // reinit cell
    fe_eval.reinit(face);
    // loop over all standard basis
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of FEEvaluation
      this->create_standard_basis(j, fe_eval);
      fe_eval.evaluate(this->operator_settings.boundary_evaluate.value,
                       this->operator_settings.boundary_evaluate.gradient);
      // perform local vmult
      this->do_boundary_integral(fe_eval, OperatorType::homogeneous, bid);

      fe_eval.integrate(this->operator_settings.boundary_integrate.value,
                        this->operator_settings.boundary_integrate.gradient);
      // extract single value from result vector and temporally store it
      local_diag[j] = fe_eval.begin_dof_values()[j];
    }
    // copy local diagonal entries into dof values of FEEvaluation and ...
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      fe_eval.begin_dof_values()[j] = local_diag[j];
    // ... write it back to global vector
    fe_eval.distribute_local_to_global(dst);
  }
}


template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::local_add_diagonal_cell_based(const MatrixFree_ &   data,
                                                                                 VectorType & dst,
                                                                                 const VectorType & /*src*/,
                                                                                 const Range & range) const
{
  FEEvalCell fe_eval(data, operator_settings.dof_index, operator_settings.quad_index);
  FEEvalFace fe_eval_m(data, true, operator_settings.dof_index, operator_settings.quad_index);
  FEEvalFace fe_eval_p(data, false, operator_settings.dof_index, operator_settings.quad_index);
  // loop over the range of macro cells
  for(auto cell = range.first; cell < range.second; ++cell)
  {
    // create temporal array for local diagonal
    VectorizedArray<Number> local_diag[dofs_per_cell];
    // reinit cell
    fe_eval.reinit(cell);
    // loop over all standard basis
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of FEEvaluation
      this->create_standard_basis(j, fe_eval);
      fe_eval.evaluate(this->operator_settings.cell_evaluate.value,
                       this->operator_settings.cell_evaluate.gradient,
                       this->operator_settings.cell_evaluate.hessians);
      // perform local vmult
      this->do_cell_integral(fe_eval);

      fe_eval.integrate(this->operator_settings.cell_integrate.value,
                        this->operator_settings.cell_integrate.gradient);
      // extract single value from result vector and temporally store it
      local_diag[j] = fe_eval.begin_dof_values()[j];
    }

    // loop over all faces
    const unsigned int nr_faces = GeometryInfo<dim>::faces_per_cell;
    for(unsigned int face = 0; face < nr_faces; ++face)
    {
      fe_eval_m.reinit(cell, face);
      fe_eval_p.reinit(cell, face);
      auto bids = data.get_faces_by_cells_boundary_id(cell, face);
      // TODO: check if all same
      auto bid = bids[0];

      // check if internal or boundary face
      if(bid == numbers::internal_face_boundary_id)
      {
        // internal face
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          // write standard basis into dof values of FEEvaluation
          this->create_standard_basis(j, fe_eval_m);
          // perform local vmult
          fe_eval_m.evaluate(this->operator_settings.internal_evaluate.value,
                             this->operator_settings.internal_evaluate.gradient);
          this->do_face_int_integral(fe_eval_m, fe_eval_p);
          fe_eval_m.integrate(this->operator_settings.internal_integrate.value,
                              this->operator_settings.internal_integrate.gradient);

          // extract single value from result vector and temporally store it
          local_diag[j] += fe_eval_m.begin_dof_values()[j];
        }
      }
      else
      {
        // boundary face
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          // write standard basis into dof values of FEEvaluation
          this->create_standard_basis(j, fe_eval_m);
          fe_eval_m.evaluate(this->operator_settings.boundary_evaluate.value,
                             this->operator_settings.boundary_evaluate.gradient);
          // perform local vmult
          this->do_boundary_integral(fe_eval_m, OperatorType::homogeneous, bid);

          fe_eval_m.integrate(this->operator_settings.boundary_integrate.value,
                              this->operator_settings.boundary_integrate.gradient);
          // extract single value from result vector and temporally store it
          local_diag[j] += fe_eval_m.begin_dof_values()[j];
        }
      }
    }
    // copy local diagonal entries into dof values of FEEvaluation and ...
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
      fe_eval.begin_dof_values()[j] = local_diag[j];
    // ... write it back to global vector
    fe_eval.distribute_local_to_global(dst);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::local_apply_block_jacobi(
  const MatrixFree_ &         data,
  VectorType &       dst,
  const VectorType & src,
  const Range &      cell_range) const
{
  // apply inverse block matrices
  FEEvalCell fe_eval(data, operator_settings.dof_index, operator_settings.quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    fe_eval.reinit(cell);
    fe_eval.read_dof_values(src);

    for(unsigned int v = 0; v < vectorization_length; ++v)
    {
      // fill source vector
      Vector<Number> src_vector(fe_eval.dofs_per_cell);
      for(unsigned int j = 0; j < fe_eval.dofs_per_cell; ++j)
        src_vector(j) = fe_eval.begin_dof_values()[j][v];

      // apply inverse matrix
      matrices[cell * vectorization_length + v].solve(src_vector, false);

      // write solution to dst-vector
      for(unsigned int j = 0; j < fe_eval.dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j][v] = src_vector(j);
    }

    fe_eval.set_dof_values(dst);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::local_apply_block_diagonal(const MatrixFree_ &         data,
                                                                              VectorType &       dst,
                                                                              const VectorType & src,
                                                                              const Range &      range) const
{
  FEEvalCell fe_eval(data, operator_settings.dof_index, operator_settings.quad_index);
  for(unsigned int cell = range.first; cell < range.second; ++cell)
  {
    fe_eval.reinit(cell);
    fe_eval.read_dof_values(src);

    for(unsigned int v = 0; v < vectorization_length; ++v)
    {
      // fill source vector
      Vector<Number> src_vector(fe_eval.dofs_per_cell);
      Vector<Number> dst_vector(fe_eval.dofs_per_cell);
      for(unsigned int j = 0; j < fe_eval.dofs_per_cell; ++j)
        src_vector(j) = fe_eval.begin_dof_values()[j][v];

      // apply matrix
      matrices[cell * vectorization_length + v].vmult(dst_vector, src_vector, false);

      // write solution to dst-vector
      for(unsigned int j = 0; j < fe_eval.dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j][v] = dst_vector(j);
    }

    fe_eval.set_dof_values(dst);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::local_add_block_diagonal_cell(const MatrixFree_ & data,
                                                                                 BlockMatrix &  dst,
                                                                                 const BlockMatrix & /*src*/,
                                                                                 const Range & range) const
{
  FEEvalCell fe_eval(data, operator_settings.dof_index, operator_settings.quad_index);
  // loop over the range of macro cells
  for(auto cell = range.first; cell < range.second; ++cell)
  {
    const unsigned int n_filled_lanes = data.n_active_entries_per_cell_batch(cell);
    // reinit cell
    fe_eval.reinit(cell);
    // loop over all standard basis
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of FEEvaluation
      this->create_standard_basis(j, fe_eval);
      fe_eval.evaluate(this->operator_settings.cell_evaluate.value,
                       this->operator_settings.cell_evaluate.gradient,
                       this->operator_settings.cell_evaluate.hessians);
      // perform local vmult
      this->do_cell_integral(fe_eval);
      fe_eval.integrate(this->operator_settings.cell_integrate.value,
                        this->operator_settings.cell_integrate.gradient);
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          dst[cell * vectorization_length + v](i, j) += fe_eval.begin_dof_values()[i][v];
    }
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::local_add_block_diagonal_face(const MatrixFree_ & data,
                                                                                 BlockMatrix &  dst,
                                                                                 const BlockMatrix & /*src*/,
                                                                                 const Range & range) const
{
  FEEvalFace fe_eval_m(data, true, operator_settings.dof_index, operator_settings.quad_index);
  FEEvalFace fe_eval_p(data, false, operator_settings.dof_index, operator_settings.quad_index);

  // loop over the range of macro cells
  for(auto face = range.first; face < range.second; ++face)
  {
    const unsigned int n_filled_lanes = data.n_active_entries_per_face_batch(face);
    // reinit cell
    fe_eval_m.reinit(face);
    fe_eval_p.reinit(face);

    // interior: loop over all standard basis
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of FEEvaluation
      this->create_standard_basis(j, fe_eval_m);
      // perform local vmult
      fe_eval_m.evaluate(this->operator_settings.internal_evaluate.value,
                         this->operator_settings.internal_evaluate.gradient);
      this->do_face_int_integral(fe_eval_m, fe_eval_p);
      fe_eval_m.integrate(this->operator_settings.internal_integrate.value,
                          this->operator_settings.internal_integrate.gradient);
      for(unsigned int v = 0; v < n_filled_lanes; ++v)
      {
        const unsigned int cell = data.get_face_info(face).cells_interior[v];
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          dst[cell](i, j) += fe_eval_m.begin_dof_values()[i][v];
      }
    }

    // exterior: loop over all standard basis
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of FEEvaluation
      this->create_standard_basis(j, fe_eval_p);
      // perform local vmult
      fe_eval_p.evaluate(this->operator_settings.internal_evaluate.value,
                         this->operator_settings.internal_evaluate.gradient);
      this->do_face_ext_integral(fe_eval_m, fe_eval_p);
      fe_eval_p.integrate(this->operator_settings.internal_integrate.value,
                          this->operator_settings.internal_integrate.gradient);
      for(unsigned int v = 0; v < n_filled_lanes; ++v)
      {
        const unsigned int cell = data.get_face_info(face).cells_exterior[v];
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          dst[cell](i, j) += fe_eval_p.begin_dof_values()[i][v];
      }
    }
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::local_add_block_diagonal_boundary(
  const MatrixFree_ & data,
  BlockMatrix &  dst,
  const BlockMatrix & /*src*/,
  const Range & range) const
{
  FEEvalFace fe_eval(data, true, operator_settings.dof_index, operator_settings.quad_index);

  // loop over the range of macro cells
  for(auto face = range.first; face < range.second; ++face)
  {
    const unsigned int n_filled_lanes = data.n_active_entries_per_face_batch(face);
    // reinit cell
    fe_eval.reinit(face);
    auto bid = data.get_boundary_id(face);
    // interior: loop over all standard basis
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of FEEvaluation
      this->create_standard_basis(j, fe_eval);
      // perform local vmult
      fe_eval.evaluate(this->operator_settings.boundary_evaluate.value,
                       this->operator_settings.boundary_evaluate.gradient);
      this->do_boundary_integral(fe_eval, OperatorType::homogeneous, bid);
      fe_eval.integrate(this->operator_settings.boundary_integrate.value,
                        this->operator_settings.boundary_integrate.gradient);
      for(unsigned int v = 0; v < n_filled_lanes; ++v)
      {
        const unsigned int cell = data.get_face_info(face).cells_interior[v];
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          dst[cell](i, j) += fe_eval.begin_dof_values()[i][v];
      }
    }
  }
}


template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::local_add_block_diagonal_cell_based(
  const MatrixFree_ & data,
  BlockMatrix &  dst,
  const BlockMatrix & /*src*/,
  const Range & range) const
{
  FEEvalCell fe_eval(data, operator_settings.dof_index, operator_settings.quad_index);
  FEEvalFace fe_eval_m(data, true, operator_settings.dof_index, operator_settings.quad_index);
  FEEvalFace fe_eval_p(data, false, operator_settings.dof_index, operator_settings.quad_index);
  // loop over the range of macro cells
  for(auto cell = range.first; cell < range.second; ++cell)
  {
    const unsigned int n_filled_lanes = data.n_active_entries_per_cell_batch(cell);
    // reinit cell
    fe_eval.reinit(cell);
    // loop over all standard basis
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of FEEvaluation
      this->create_standard_basis(j, fe_eval);
      fe_eval.evaluate(this->operator_settings.cell_evaluate.value,
                       this->operator_settings.cell_evaluate.gradient,
                       this->operator_settings.cell_evaluate.hessians);
      // perform local vmult
      this->do_cell_integral(fe_eval);

      fe_eval.integrate(this->operator_settings.cell_integrate.value,
                        this->operator_settings.cell_integrate.gradient);
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          dst[cell * vectorization_length + v](i, j) = fe_eval.begin_dof_values()[i][v];
    }

    // loop over all faces
    const unsigned int nr_faces = GeometryInfo<dim>::faces_per_cell;
    for(unsigned int face = 0; face < nr_faces; ++face)
    {
      fe_eval_m.reinit(cell, face);
      fe_eval_p.reinit(cell, face);
      auto bids = data.get_faces_by_cells_boundary_id(cell, face);
      // TODO: check if all same
      auto bid = bids[0];

      // check if internal or boundary face
      if(bid == numbers::internal_face_boundary_id) // internal face
      {
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          // write standard basis into dof values of FEEvaluation
          this->create_standard_basis(j, fe_eval_m);
          // perform local vmult
          fe_eval_m.evaluate(this->operator_settings.internal_evaluate.value,
                             this->operator_settings.internal_evaluate.gradient);
          this->do_face_int_integral(fe_eval_m, fe_eval_p);
          fe_eval_m.integrate(this->operator_settings.internal_integrate.value,
                              this->operator_settings.internal_integrate.gradient);
          for(unsigned int v = 0; v < n_filled_lanes; ++v)
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
              dst[cell * vectorization_length + v](i, j) += fe_eval_m.begin_dof_values()[i][v];
        }
      }
      else // boundary face
      {
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          // write standard basis into dof values of FEEvaluation
          this->create_standard_basis(j, fe_eval_m);
          fe_eval_m.evaluate(this->operator_settings.boundary_evaluate.value,
                             this->operator_settings.boundary_evaluate.gradient);
          // perform local vmult
          this->do_boundary_integral(fe_eval_m, OperatorType::homogeneous, bid);

          fe_eval_m.integrate(this->operator_settings.boundary_integrate.value,
                              this->operator_settings.boundary_integrate.gradient);
          for(unsigned int v = 0; v < n_filled_lanes; ++v)
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
              dst[cell * vectorization_length + v](i, j) += fe_eval_m.begin_dof_values()[i][v];
        }
      }
    }
  }
}

#ifdef DEAL_II_WITH_TRILINOS
template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::local_calculate_system_matrix_cell(
  const MatrixFree_ & data,
  SparseMatrix &  dst,
  const SparseMatrix & /*src*/,
  const Range & range) const
{
  FEEvalCell fe_eval(data, operator_settings.dof_index, operator_settings.quad_index);
  // loop over the range of macro cells
  for(auto cell = range.first; cell < range.second; ++cell)
  {
    // determine number of filled vector lanes
    const unsigned int n_filled_lanes = data.n_components_filled(cell);

    // create a temporal full matrix for the local element matrix of each ...
    // cell of each macro cell and ...
    FullMatrix_ matrices[vectorization_length];
    // set their size
    std::fill_n(matrices, vectorization_length, FullMatrix_(dofs_per_cell, dofs_per_cell));

    // reinit cell
    fe_eval.reinit(cell);

    // loop over all standard basis
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of FEEvaluation
      this->create_standard_basis(j, fe_eval);
      fe_eval.evaluate(this->operator_settings.cell_evaluate.value,
                       this->operator_settings.cell_evaluate.gradient,
                       this->operator_settings.cell_evaluate.hessians);
      // perform local vmult
      this->do_cell_integral(fe_eval);

      fe_eval.integrate(this->operator_settings.cell_integrate.value,
                        this->operator_settings.cell_integrate.gradient);

      // insert result vector into local matrix
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices[v](i, j) = fe_eval.begin_dof_values()[i][v];
    }

    // finally assemble local matrix into global matrix
    for(unsigned int i = 0; i < n_filled_lanes; i++)
    {
      auto cell_i = data.get_cell_iterator(cell, i);
      //      cell_i->distribute_local_to_global(matrices[i], dst);
      std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
      if(is_mg)
        cell_i->get_mg_dof_indices(dof_indices);
      else
        cell_i->get_dof_indices(dof_indices);

      if(!is_dg)
      {
        // in the case of CG: shape functions are not ordered lexicographically
        // see (https://www.dealii.org/8.5.1/doxygen/deal.II/classFE__Q.html)
        // so we have to fix the order
        auto temp = dof_indices;
        for(unsigned int j = 0; j < dof_indices.size(); j++)
          dof_indices[j] = temp[data.get_shape_info().lexicographic_numbering[j]];
      }

      constraint->distribute_local_to_global(matrices[i], dof_indices, dof_indices, dst);
    }
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::local_calculate_system_matrix_face(
  const MatrixFree_ & data,
  SparseMatrix &  dst,
  const SparseMatrix & /*src*/,
  const Range & range) const
{
  FEEvalFace fe_eval_m(data, true, operator_settings.dof_index, operator_settings.quad_index);
  FEEvalFace fe_eval_p(data, false, operator_settings.dof_index, operator_settings.quad_index);

  // loop over the range of macro faces
  for(auto face = range.first; face < range.second; ++face)
  {
    // determine number of filled vector lanes
    const unsigned int n_filled_lanes = data.n_active_entries_per_face_batch(face);
    // create two local matrix: first one tested by v1 and ...
    FullMatrix_ matrices_1[vectorization_length];
    std::fill_n(matrices_1, vectorization_length, FullMatrix_(dofs_per_cell, dofs_per_cell));
    // ... the other tested by v2
    FullMatrix_ matrices_2[vectorization_length];
    std::fill_n(matrices_2, vectorization_length, FullMatrix_(dofs_per_cell, dofs_per_cell));

    // reinit face
    fe_eval_m.reinit(face);
    fe_eval_p.reinit(face);

    // process trial function u1: loop over all standard basis
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of first FEFaceEvaluation and
      // clear dof values of second FEFaceEvaluation
      this->create_standard_basis(j, fe_eval_m, fe_eval_p);
      // do loacal vmult

      fe_eval_m.evaluate(this->operator_settings.internal_evaluate.value,
                         this->operator_settings.internal_evaluate.gradient);
      fe_eval_p.evaluate(this->operator_settings.internal_evaluate.value,
                         this->operator_settings.internal_evaluate.gradient);
      this->do_face_integral(fe_eval_m, fe_eval_p);
      fe_eval_m.integrate(this->operator_settings.internal_integrate.value,
                          this->operator_settings.internal_integrate.gradient);
      fe_eval_p.integrate(this->operator_settings.internal_integrate.value,
                          this->operator_settings.internal_integrate.gradient);

      // insert result vector into local matrix u1_v1
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices_1[v](i, j) = fe_eval_m.begin_dof_values()[i][v];

      // insert result vector into local matrix  u1_v2
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices_2[v](i, j) = fe_eval_p.begin_dof_values()[i][v];
    }

    // save local matrices into global matrix
    for(unsigned int i = 0; i < n_filled_lanes; i++)
    {
      // cell number of minus
      const unsigned int cell_number_1 = data.get_face_info(face).cells_interior[i];
      // cell number of plus
      const unsigned int cell_number_2 = data.get_face_info(face).cells_exterior[i];

      // cell reference to cell minus
      auto cell_m = data.get_cell_iterator(cell_number_1 / vectorization_length, cell_number_1 % vectorization_length);
      // cell reference to cell plus
      auto cell_p = data.get_cell_iterator(cell_number_2 / vectorization_length, cell_number_2 % vectorization_length);

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

      // save u1_v1
      constraint->distribute_local_to_global(matrices_1[i], dof_indices_m, dof_indices_m, dst);
      // save u1_v2
      constraint->distribute_local_to_global(matrices_2[i], dof_indices_p, dof_indices_m, dst);
    }

    // process trial function u1: loop over all standard basis
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of first FEFaceEvaluation and
      // clear dof values of second FEFaceEvaluation
      this->create_standard_basis(j, fe_eval_p, fe_eval_m);
      // do loacal vmult

      fe_eval_m.evaluate(this->operator_settings.internal_evaluate.value,
                         this->operator_settings.internal_evaluate.gradient);
      fe_eval_p.evaluate(this->operator_settings.internal_evaluate.value,
                         this->operator_settings.internal_evaluate.gradient);
      this->do_face_integral(fe_eval_m, fe_eval_p);
      fe_eval_m.integrate(this->operator_settings.internal_integrate.value,
                          this->operator_settings.internal_integrate.gradient);
      fe_eval_p.integrate(this->operator_settings.internal_integrate.value,
                          this->operator_settings.internal_integrate.gradient);

      // insert result vector into local matrix u1_v1
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices_1[v](i, j) = fe_eval_m.begin_dof_values()[i][v];

      // insert result vector into local matrix  u1_v2
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices_2[v](i, j) = fe_eval_p.begin_dof_values()[i][v];
    }

    // save local matrices into global matrix
    for(unsigned int i = 0; i < n_filled_lanes; i++)
    {
      // cell number of minus
      const unsigned int cell_number_1 = data.get_face_info(face).cells_interior[i];
      // cell number of plus
      const unsigned int cell_number_2 = data.get_face_info(face).cells_exterior[i];

      // cell reference to cell minus
      auto cell_m = data.get_cell_iterator(cell_number_1 / vectorization_length, cell_number_1 % vectorization_length);
      // cell reference to cell plus
      auto cell_p = data.get_cell_iterator(cell_number_2 / vectorization_length, cell_number_2 % vectorization_length);

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

      // save u2_v1
      constraint->distribute_local_to_global(matrices_1[i], dof_indices_m, dof_indices_p, dst);
      // save u2_v2
      constraint->distribute_local_to_global(matrices_2[i], dof_indices_p, dof_indices_p, dst);
    }
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::local_calculate_system_matrix_boundary(
  const MatrixFree_ & data,
  SparseMatrix &  dst,
  const SparseMatrix & /*src*/,
  const Range & range) const
{
  FEEvalFace fe_eval(data, true, operator_settings.dof_index, operator_settings.quad_index);

  // loop over the range of macro faces
  for(auto face = range.first; face < range.second; ++face)
  {
    // determine number of filled vector lanes
    const unsigned int n_filled_lanes = data.n_active_entries_per_face_batch(face);

    FullMatrix_ matrices[vectorization_length];
    std::fill_n(matrices, vectorization_length, FullMatrix_(dofs_per_cell, dofs_per_cell));

    // reinit face
    fe_eval.reinit(face);
    auto bid = data.get_boundary_id(face);

    // process trial function u1: loop over all standard basis
    for(unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      // write standard basis into dof values of first FEFaceEvaluation and
      // clear dof values of second FEFaceEvaluation
      this->create_standard_basis(j, fe_eval);

      fe_eval.evaluate(this->operator_settings.boundary_evaluate.value,
                       this->operator_settings.boundary_evaluate.gradient);
      // do loacal vmult
      this->do_boundary_integral(fe_eval, OperatorType::homogeneous, bid);
      fe_eval.integrate(this->operator_settings.boundary_integrate.value,
                        this->operator_settings.boundary_integrate.gradient);

      // insert result vector into local matrix u1_v1
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices[v](i, j) = fe_eval.begin_dof_values()[i][v];
    }

    // save local matrices into global matrix
    for(unsigned int i = 0; i < n_filled_lanes; i++)
    {
      // cell number of minus
      const unsigned int cell_num = data.get_face_info(face).cells_interior[i];

      auto cell_i = data.get_cell_iterator(cell_num / vectorization_length, cell_num % vectorization_length);
      std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
      if(is_mg)
        cell_i->get_mg_dof_indices(dof_indices);
      else
        cell_i->get_dof_indices(dof_indices);

      constraint->distribute_local_to_global(matrices[i], dof_indices, dof_indices, dst);
    }
  }
}
#endif

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::adjust_diagonal_for_singular_operator(VectorType & diagonal) const
{
  VectorType vec1, d;
  vec1.reinit(diagonal, true);
  d.reinit(diagonal, true);
  for(unsigned int i = 0; i < vec1.local_size(); ++i)
    vec1.local_element(i) = 1.;
  vmult(d, vec1);
  double length = vec1 * vec1;
  double factor = vec1 * d;
  diagonal.add(-2. / length, d, factor / pow(length, 2.), vec1);
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::set_constraint_diagonal(VectorType & diagonal) const
{
  // set (diagonal) entries to 1.0 for constrained dofs
  for(auto i : data->get_constrained_dofs())
    diagonal.local_element(i) = 1.0;
}

template<int dim, int degree, typename Number, typename AdditionalData>
void
OperatorBase<dim, degree, Number, AdditionalData>::add_constraints(const DoFHandler<dim> & dof_handler,
                  ConstraintMatrix& constraint_own, 
                  const MGConstrainedDoFs & mg_constrained_dofs,
                  AdditionalData & operator_settings,
                  const unsigned int        level)
{
  // 0) clear old content (to be on the safe side)
  constraint_own.clear();
  
  // 1) add periodic bcs
  this->add_periodicity_constraints(dof_handler, level, operator_settings.periodic_face_pairs_level0, constraint_own);

  // 2) add dirichlet bcs
  constraint_own.add_lines(mg_constrained_dofs.get_boundary_indices(level));

  // constraint zeroth DoF in continuous case (the mean value constraint will
  // be applied in the DG case). In case we have interface matrices, there are
  // Dirichlet constraints on parts of the boundary and no such transformation
  // is required.
  if(verify_boundary_conditions_and_check_if_pure_neumann_problem(dof_handler, operator_settings) && constraint_own.can_store_line(0))
  {
    // if dof 0 is constrained, it must be a periodic dof, so we take the
    // value on the other side
    types::global_dof_index line_index = 0;
    while(true)
    {
      const std::vector<std::pair<types::global_dof_index, double>> * lines =
        constraint_own.get_constraint_entries(line_index);
      if(lines == 0)
      {
        constraint_own.add_line(line_index);
        // add the constraint back to the MGConstrainedDoFs field. This
        // is potentially dangerous but we know what we are doing... ;-)
        if(level != numbers::invalid_unsigned_int)
          const_cast<IndexSet &>(mg_constrained_dofs.get_boundary_indices(level)).add_index(line_index);
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
OperatorBase<dim, degree, Number, AdditionalData>::add_periodicity_constraints(const DoFHandler<dim> & dof_handler, 
                              const unsigned int level,
                              std::vector<PeriodicFacePairIterator>& periodic_face_pairs_level0,
                              ConstraintMatrix& constraint_own)
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
  const unsigned int                            level,
  const unsigned int                            target_level,
  const typename DoFHandler<dim>::face_iterator face1,
  const typename DoFHandler<dim>::face_iterator face2,
  ConstraintMatrix &                            constraints)
{
  if(level == 0)
  {
    // level of interest has been reached
    const unsigned int dofs_per_face = face1->get_fe(0).dofs_per_face;
    
    std::vector<types::global_dof_index> dofs_1(dofs_per_face);
    std::vector<types::global_dof_index> dofs_2(dofs_per_face);

    face1->get_mg_dof_indices(target_level, dofs_1, 0);
    face2->get_mg_dof_indices(target_level, dofs_2, 0);

    for(unsigned int i = 0; i < dofs_per_face; ++i)
      if(constraints.can_store_line(dofs_2[i]) && constraints.can_store_line(dofs_1[i]) &&
         !constraints.is_constrained(dofs_2[i]))
      {
        // constraint dof and ...
        constraints.add_line(dofs_2[i]);
        // specify type of constraint: equality (dof_2[i]=dof_1[j]*1.0)
        constraints.add_entry(dofs_2[i], dofs_1[i], 1.);
      }
  }
  else if(face1->has_children() && face2->has_children())
  {
    // recursively visit all subfaces
    for(unsigned int c = 0; c < face1->n_children(); ++c)
      add_periodicity_constraints(level - 1, target_level, face1->child(c), face2->child(c), constraints);
  }
}

template<int dim, int degree, typename Number, typename AdditionalData>
bool
OperatorBase<dim, degree, Number, AdditionalData>::verify_boundary_conditions_and_check_if_pure_neumann_problem(
  DoFHandler<dim> const & dof_handler,
  AdditionalData const &  operator_data)
{
  // Check that the Dirichlet and Neumann boundary conditions do not overlap
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

  bool                       pure_neumann_problem = true;
  const Triangulation<dim> & tria                 = dof_handler.get_triangulation();
  for(typename Triangulation<dim>::cell_iterator cell = tria.begin(); cell != tria.end(); ++cell)
    for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      if(cell->at_boundary(f))
      {
        types::boundary_id bid = cell->face(f)->boundary_id();
        if(operator_data.bc->dirichlet_bc.find(bid) != operator_data.bc->dirichlet_bc.end())
        {
          AssertThrow(operator_data.bc->neumann_bc.find(bid) == operator_data.bc->neumann_bc.end(),
                      ExcMessage("Boundary id " + Utilities::to_string((int)bid) +
                                 " wants to set both Dirichlet and Neumann " +
                                 "boundary conditions, which is impossible!"));
          AssertThrow(periodic_boundary_ids.find(bid) == periodic_boundary_ids.end(),
                      ExcMessage("Boundary id " + Utilities::to_string((int)bid) +
                                 " wants to set both Dirichlet and periodic " +
                                 "boundary conditions, which is impossible!"));
          pure_neumann_problem = false;
          continue;
        }
        if(operator_data.bc->neumann_bc.find(bid) != operator_data.bc->neumann_bc.end())
        {
          AssertThrow(periodic_boundary_ids.find(bid) == periodic_boundary_ids.end(),
                      ExcMessage("Boundary id " + Utilities::to_string((int)bid) +
                                 " wants to set both Neumann and periodic " +
                                 "boundary conditions, which is impossible!"));
          continue;
        }
        AssertThrow(periodic_boundary_ids.find(bid) != periodic_boundary_ids.end(),
                    ExcMessage("Boundary id " + Utilities::to_string((int)bid) +
                               " does neither set Dirichlet, Neumann, nor periodic " +
                               "boundary conditions! Bailing out."));
      }

  // Check for consistency of 'pure_neumann_problem' over all participating
  // processors
  int      my_neumann = pure_neumann_problem;
  MPI_Comm mpi_communicator =
    dynamic_cast<const parallel::Triangulation<dim> *>(&tria) ?
      (dynamic_cast<const parallel::Triangulation<dim> *>(&tria))->get_communicator() :
      MPI_COMM_SELF;
  const int max_pure_neumann = Utilities::MPI::max(my_neumann, mpi_communicator);
  int       min_pure_neumann = Utilities::MPI::min(my_neumann, mpi_communicator);
  AssertThrow(max_pure_neumann == min_pure_neumann,
              ExcMessage("Neumann/Dirichlet assignment over processors does not match."));

  return pure_neumann_problem;
}