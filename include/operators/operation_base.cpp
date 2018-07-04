#include "operation_base.h"

#include "../solvers_and_preconditioners/block_jacobi_matrices.h"
#include "../solvers_and_preconditioners/invert_diagonal.h"

template <int dim, int degree, typename Number, typename AdditionalData>
OperatorBase<dim, degree, Number, AdditionalData>::OperatorBase()
    : ad(AdditionalData()), do_eval_faces(ad.internal_evaluate.do_eval() ||
                                          ad.internal_integrate.do_eval() ||
                                          ad.boundary_evaluate.do_eval() ||
                                          ad.boundary_integrate.do_eval()),
      block_jacobi_matrices_have_been_initialized(false) {}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::reinit(MF &mf,
                                                               CM &cm) const {
  this->data = &mf;
  this->constraint = &cm;

  // check if dg or cg
  is_dg = data->get_dof_handler(ad.dof_index).get_fe().dofs_per_vertex == 0;
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::vmult(
    VNumber &dst, VNumber const &src) const {
  dst = 0;
  vmult_add(dst, src);
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::apply(
    VNumber &dst, VNumber const &src) const {
  vmult(dst, src);
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::vmult_add(
    VNumber &dst, VNumber const &src) const {
  if (do_eval_faces && is_dg)
    data->loop(&This::local_apply_cell, &This::local_apply_face,
               &This::local_apply_boundary, this, dst, src);
  else
    data->cell_loop(&This::local_apply_cell, this, dst, src);
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::apply_add(
    VNumber &dst, VNumber const &src) const {
  vmult_add(dst, src);
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::vmult_interface_down(
    VNumber &dst, VNumber const &src) const {
  vmult(dst, src);
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::vmult_add_interface_up(
    VNumber &dst, VNumber const &src) const {
  vmult_add(dst, src);
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::rhs(
    VNumber &dst, Number const time) const {
  dst = 0;
  rhs_add(dst, time);
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::rhs_add(
    VNumber &dst, Number const time) const {

  this->eval_time = time;

  VNumber tmp(dst);

  data->loop(&This::local_apply_inhom_cell, &This::local_apply_inhom_face,
             &This::local_apply_inhom_boundary, this, tmp, tmp);

  // multiply by -1.0 since the boundary face integrals have to be shifted
  // to the right hand side
  dst.add(-1.0, tmp);
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::evaluate(
    VNumber &dst, VNumber const &src, Number const time) const {
  dst = 0;
  evaluate_add(dst, src, time);
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::evaluate_add(
    VNumber &dst, VNumber const &src, Number const time) const {
  this->eval_time = time;

  data->loop(&This::local_apply_cell, &This::local_apply_face,
             &This::local_apply_full_boundary, this, dst, src);
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::calculate_diagonal(
    VNumber &diagonal) const {
  if (diagonal.size() == 0)
    data->initialize_dof_vector(diagonal);
  diagonal = 0;
  add_diagonal(diagonal);
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::add_diagonal(
    VNumber &diagonal) const {

  if (do_eval_faces && is_dg)
    data->loop(&This::local_apply_cell_diagonal,
               &This::local_apply_face_diagonal,
               &This::local_apply_boundary_diagonal, this, diagonal, diagonal);
  else
    data->cell_loop(&This::local_apply_cell_diagonal, this, diagonal, diagonal);
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::calculate_inverse_diagonal(
    VNumber &diagonal) const {
    calculate_diagonal(diagonal);
    invert_diagonal(diagonal);
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::apply_block_jacobi(
    VNumber &dst, VNumber const &src) const {

  dst = 0;
  apply_block_jacobi_add(dst, src);
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::apply_block_jacobi_add(
    VNumber &dst, VNumber const &src) const {

  AssertThrow(is_dg, ExcMessage("Block Jacobi only implemented for DG!"));
  AssertThrow(block_jacobi_matrices_have_been_initialized,
              ExcMessage("Block Jacobi matrices have not been initialized!"));

  data->cell_loop(&This::cell_loop_apply_inverse_block_jacobi_matrices, this,
                  dst, src);
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::apply_block_diagonal(
    VNumber &dst, VNumber const &src) const {

  AssertThrow(is_dg, ExcMessage("Block Jacobi only implemented for DG!"));
  AssertThrow(block_jacobi_matrices_have_been_initialized,
              ExcMessage("Block Jacobi matrices have not been initialized!"));

  dst = 0;

  data->cell_loop(&This::local_apply_block_diagonal, this, dst, src);
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::update_block_jacobi()
    const {

  AssertThrow(is_dg, ExcMessage("Block Jacobi only implemented for DG!"));

  // allocate memory only the first time
  if (block_jacobi_matrices_have_been_initialized == false) {
    auto dofs = data->get_shape_info().dofs_per_component_on_cell;
    matrices.resize(data->n_macro_cells() * v_len,
                    LAPACKFullMatrix<Number>(dofs, dofs));
    block_jacobi_matrices_have_been_initialized = true;
  } // else: reuse old memory

  // clear matrices
  initialize_block_jacobi_matrices_with_zero(matrices);

  // compute block matrices
  add_block_jacobi_matrices(matrices);

  // perform lu factorization for block matrices
  // calculate_lu_factorization_block_jacobi(matrices); // TODO
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number,
                  AdditionalData>::add_block_jacobi_matrices(BMatrix &matrices)
    const {

  AssertThrow(is_dg, ExcMessage("Block Jacobi only implemented for DG!"));
  AssertThrow(matrices.size() != 0,
              ExcMessage("Block Jacobi matrices have not been initialized!"));

  if (do_eval_faces)
    data->loop(&This::local_apply_cell_block_diagonal,
               &This::local_apply_face_block_diagonal,
               &This::local_apply_boundary_block_diagonal, this, matrices,
               matrices);
  else
    data->cell_loop(&This::local_apply_cell_block_diagonal, this, matrices,
                    matrices);
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::init_system_matrix(
    SMatrix &system_matrix, MPI_Comm comm) const {

  const DoFHandler<dim> &dof_handler = this->data->get_dof_handler();
  TrilinosWrappers::SparsityPattern dsp(dof_handler.locally_owned_dofs(), comm);
  DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
  dsp.compress();
  system_matrix.reinit(dsp);
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::calculate_system_matrix(
    SMatrix &system_matrix) const {

  // assemble matrix locally on each process
  if (do_eval_faces && is_dg)
    data->loop(&This::local_apply_cell_system_matrix,
               &This::local_apply_face_system_matrix,
               &This::local_apply_boundary_system_matrix, this, system_matrix,
               system_matrix);
  else
    data->cell_loop(&This::local_apply_cell_system_matrix, this, system_matrix,
                    system_matrix);

  // communicate overlapping matrix parts
  system_matrix.compress(VectorOperation::add);
}

template <int dim, int degree, typename Number, typename AdditionalData>
types::global_dof_index
OperatorBase<dim, degree, Number, AdditionalData>::m() const {
  return data->get_vector_partitioner(ad.dof_index)->size();
}

template <int dim, int degree, typename Number, typename AdditionalData>
types::global_dof_index
OperatorBase<dim, degree, Number, AdditionalData>::n() const {
  return data->get_vector_partitioner(ad.dof_index)->size();
}

template <int dim, int degree, typename Number, typename AdditionalData>
Number OperatorBase<dim, degree, Number, AdditionalData>::el(
    const unsigned int, const unsigned int) const {
  AssertThrow(false, ExcMessage("Matrix-free does not allow for entry access"));
  return Number();
}

template <int dim, int degree, typename Number, typename AdditionalData>
bool OperatorBase<dim, degree, Number, AdditionalData>::is_empty_locally()
    const {
  return data->n_macro_cells() == 0;
}

template <int dim, int degree, typename Number, typename AdditionalData>
const MatrixFree<dim, Number> &
OperatorBase<dim, degree, Number, AdditionalData>::get_data() const {
  return *data;
}

template <int dim, int degree, typename Number, typename AdditionalData>
unsigned int
OperatorBase<dim, degree, Number, AdditionalData>::get_dof_index() const {
  return ad.dof_index;
}

template <int dim, int degree, typename Number, typename AdditionalData>
unsigned int
OperatorBase<dim, degree, Number, AdditionalData>::get_quad_index() const {
  return ad.quad_index;
}

template <int dim, int degree, typename Number, typename AdditionalData>
AdditionalData const &
OperatorBase<dim, degree, Number, AdditionalData>::get_operator_data() const {
  return ad;
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::initialize_dof_vector(
    VNumber &vector) const {
  data->initialize_dof_vector(vector, ad.dof_index);
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::set_evaluation_time(
    double const evaluation_time_in) {
  eval_time = evaluation_time_in;
}

template <int dim, int degree, typename Number, typename AdditionalData>
double
OperatorBase<dim, degree, Number, AdditionalData>::get_evaluation_time() const {
  return eval_time;
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::create_standard_basis(
    unsigned int j, FEEvalCell &phi) const {
  // create a standard basis in the dof values of FEEvalution
  for (unsigned int i = 0; i < dofs_per_cell; ++i)
    phi.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
  phi.begin_dof_values()[j] = make_vectorized_array<Number>(1.);
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::create_standard_basis(
    unsigned int j, FEEvalFace &phi) const {
  // create a standard basis in the dof values of FEEvalution
  for (unsigned int i = 0; i < dofs_per_cell; ++i)
    phi.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
  phi.begin_dof_values()[j] = make_vectorized_array<Number>(1.);
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::create_standard_basis(
    unsigned int j, FEEvalFace &phi_1, FEEvalFace &phi_2) const {
  // create a standard basis in the dof values of the first FEFaceEvalution
  for (unsigned int i = 0; i < dofs_per_cell; ++i)
    phi_1.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
  phi_1.begin_dof_values()[j] = make_vectorized_array<Number>(1.);
  for (unsigned int i = 0; i < dim * this->dofs_per_cell; i++)
    phi_1.begin_gradients()[i] = 0.0;

  // clear dof values of the second FEFaceEvalution
  for (unsigned int i = 0; i < dofs_per_cell; ++i)
    phi_2.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
  for (unsigned int i = 0; i < dim * this->dofs_per_cell; i++)
    phi_2.begin_gradients()[i] = 0.0;
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::local_apply_cell(
    const MF &data, VNumber &dst, const VNumber &src,
    const Range &range) const {
  FEEvalCell phi(data);
  // loop over the range of macro cells
  for (auto cell = range.first; cell < range.second; ++cell) {
    // reinit cell
    phi.reinit(cell);
    // read dof values from global vector src and evaluate
    phi.gather_evaluate(src, this->ad.cell_evaluate.value,
                        this->ad.cell_evaluate.gradient,
                        this->ad.cell_evaluate.hessians);
    // perform local vmult
    this->do_cell_integral(phi);
    // integrate and write result back to the global vector dst
    phi.integrate_scatter(this->ad.cell_integrate.value,
                          this->ad.cell_integrate.gradient, dst);
  }
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::local_apply_face(
    const MF &data, VNumber &dst, const VNumber &src,
    const Range &range) const {

  FEEvalFace phi_n(data, true, ad.dof_index, ad.quad_index);
  FEEvalFace phi_p(data, false, ad.dof_index, ad.quad_index);
  // loop over the range of macro cells
  for (auto face = range.first; face < range.second; ++face) {
    // reinit cell
    phi_n.reinit(face);
    phi_p.reinit(face);
    // read dof values from global vector src
    phi_n.gather_evaluate(src, this->ad.internal_evaluate.value,
                          this->ad.internal_evaluate.gradient);
    phi_p.gather_evaluate(src, this->ad.internal_evaluate.value,
                          this->ad.internal_evaluate.gradient);
    // perform local vmult
    this->do_face_integral(phi_n, phi_p);
    // write result back to the global vector dst
    phi_n.integrate_scatter(this->ad.internal_integrate.value,
                            this->ad.internal_integrate.gradient, dst);
    phi_p.integrate_scatter(this->ad.internal_integrate.value,
                            this->ad.internal_integrate.gradient, dst);
  }
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::local_apply_boundary(
    const MF &data, VNumber &dst, const VNumber &src,
    const Range &range) const {
    
  FEEvalFace phi(data, true, ad.dof_index, ad.quad_index);

  for (unsigned int face = range.first; face < range.second; face++) {
    auto bid = data.get_boundary_id(face);

    phi.reinit(face);
    phi.gather_evaluate(src, this->ad.boundary_evaluate.value,
                        this->ad.boundary_evaluate.gradient);
    do_boundary_integral(phi, OperatorType::homogeneous, bid);
    phi.integrate_scatter(this->ad.boundary_integrate.value,
                          this->ad.boundary_integrate.gradient, dst);
  }
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::local_apply_inhom_cell(
    const MF &, VNumber &, const VNumber &, const Range &) const {
  /*nothing to do*/
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::local_apply_inhom_face(
    const MF &, VNumber &, const VNumber &, const Range &) const {
  /*nothing to do*/
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::
    local_apply_inhom_boundary(const MF &data, VNumber &dst,
                               const VNumber & src,
                               const Range &range) const {
    
  FEEvalFace phi(data, true, ad.dof_index, ad.quad_index);

  for (unsigned int face = range.first; face < range.second; face++) {
    auto bid = data.get_boundary_id(face);
    phi.reinit(face);
    phi.gather_evaluate(src, this->ad.boundary_evaluate.value,
                        this->ad.boundary_evaluate.gradient);
    do_boundary_integral(phi, OperatorType::inhomogeneous, bid);
    phi.integrate_scatter(this->ad.boundary_integrate.value,
                          this->ad.boundary_integrate.gradient, dst);
  }
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number,
                  AdditionalData>::local_apply_full_boundary(const MF &data,
                                                             VNumber &dst,
                                                             const VNumber &src,
                                                             const Range &range)
    const {
    
  FEEvalFace phi(data, true, ad.dof_index, ad.quad_index);

  for (unsigned int face = range.first; face < range.second; face++) {
    auto bid = data.get_boundary_id(face);
    phi.reinit(face);
    phi.gather_evaluate(src, this->ad.boundary_evaluate.value,
                        this->ad.boundary_evaluate.gradient);
    do_boundary_integral(phi, OperatorType::full, bid);
    phi.integrate_scatter(this->ad.boundary_integrate.value,
                          this->ad.boundary_integrate.gradient, dst);
  }
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::
    local_apply_cell_diagonal(const MF &data, VNumber &dst,
                              const VNumber & /*src*/,
                              const Range &range) const {
  FEEvalCell phi(data, ad.dof_index, ad.quad_index);
  // loop over the range of macro cells
  for (auto cell = range.first; cell < range.second; ++cell) {
    // create temporal array for local diagonal
    VectorizedArray<Number> local_diag[dofs_per_cell];
    // reinit cell
    phi.reinit(cell);
    // loop over all standard basis
    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
      // write standard basis into dof values of FEEvaluation
      this->create_standard_basis(j, phi);
      phi.evaluate(this->ad.cell_evaluate.value,
                   this->ad.cell_evaluate.gradient,
                   this->ad.cell_evaluate.hessians);
      // perform local vmult
      this->do_cell_integral(phi);

      phi.integrate(this->ad.cell_integrate.value,
                    this->ad.cell_integrate.gradient);
      // extract single value from result vector and temporally store it
      local_diag[j] = phi.begin_dof_values()[j];
    }
    // copy local diagonal entries into dof values of FEEvaluation and ...
    for (unsigned int j = 0; j < dofs_per_cell; ++j)
      phi.begin_dof_values()[j] = local_diag[j];
    // ... write it back to global vector
    phi.distribute_local_to_global(dst);
  }
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::
    local_apply_face_diagonal(const MF &data, VNumber &dst,
                              const VNumber & /*src*/,
                              const Range &range) const {

  FEEvalFace phi_n(data, true, ad.dof_index, ad.quad_index);
  FEEvalFace phi_p(data, false, ad.dof_index, ad.quad_index);

  // loop over the range of macro cells
  for (auto cell = range.first; cell < range.second; ++cell) {
    // create temporal array for local diagonal
    VectorizedArray<Number> local_diag[dofs_per_cell];
    // reinit cell
    phi_n.reinit(cell);
    phi_p.reinit(cell);

    // interior: loop over all standard basis
    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
      // write standard basis into dof values of FEEvaluation
      this->create_standard_basis(j, phi_n);
      // perform local vmult
      phi_n.evaluate(this->ad.internal_evaluate.value,
                     this->ad.internal_evaluate.gradient);
      this->do_face_int_integral(phi_n);
      phi_n.integrate(this->ad.internal_integrate.value,
                      this->ad.internal_integrate.gradient);

      // extract single value from result vector and temporally store it
      local_diag[j] = phi_n.begin_dof_values()[j];
    }
    // copy local diagonal entries into dof values of FEEvaluation and ...
    for (unsigned int j = 0; j < dofs_per_cell; ++j)
      phi_n.begin_dof_values()[j] = local_diag[j];
    // ... write it back to global vector
    phi_n.distribute_local_to_global(dst);

    // exterior: loop over all standard basis
    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
      // write standard basis into dof values of FEEvaluation
      this->create_standard_basis(j, phi_p);
      // perform local vmult
      phi_p.evaluate(this->ad.internal_evaluate.value,
                     this->ad.internal_evaluate.gradient);
      this->do_face_ext_integral(phi_p);
      phi_p.integrate(this->ad.internal_integrate.value,
                      this->ad.internal_integrate.gradient);
      // extract single value from result vector and temporally store it
      local_diag[j] = phi_p.begin_dof_values()[j];
    }
    // copy local diagonal entries into dof values of FEEvaluation and ...
    for (unsigned int j = 0; j < dofs_per_cell; ++j)
      phi_p.begin_dof_values()[j] = local_diag[j];
    // ... write it back to global vector
    phi_p.distribute_local_to_global(dst);
  }
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::
    local_apply_boundary_diagonal(const MF & data, VNumber & dst,
                                  const VNumber & /*src*/,
                                  const Range & range) const {

  FEEvalFace phi(data, true, ad.dof_index, ad.quad_index);

  for (unsigned int face = range.first; face < range.second; face++) {
    // create temporal array for local diagonal
    VectorizedArray<Number> local_diag[dofs_per_cell];
    auto bid = data.get_boundary_id(face);

    // reinit cell
    phi.reinit(face);
    // loop over all standard basis
    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
      // write standard basis into dof values of FEEvaluation
      this->create_standard_basis(j, phi);
      phi.evaluate(this->ad.boundary_evaluate.value,
                   this->ad.boundary_evaluate.gradient);
      // perform local vmult
      this->do_boundary_integral(phi, OperatorType::homogeneous, bid);

      phi.integrate(this->ad.boundary_integrate.value,
                    this->ad.boundary_integrate.gradient);
      // extract single value from result vector and temporally store it
      local_diag[j] = phi.begin_dof_values()[j];
    }
    // copy local diagonal entries into dof values of FEEvaluation and ...
    for (unsigned int j = 0; j < dofs_per_cell; ++j)
      phi.begin_dof_values()[j] = local_diag[j];
    // ... write it back to global vector
    phi.distribute_local_to_global(dst);
  }
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::
    cell_loop_apply_inverse_block_jacobi_matrices(
        const MF &data, VNumber &dst, const VNumber &src,
        const Range &cell_range) const {
  // apply inverse block matrices
  FEEvalCell fe_eval(data, ad.dof_index, ad.quad_index);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
    fe_eval.reinit(cell);
    fe_eval.read_dof_values(src);

    for (unsigned int v = 0; v < v_len; ++v) {
      // fill source vector
      Vector<Number> src_vector(fe_eval.dofs_per_cell);
      for (unsigned int j = 0; j < fe_eval.dofs_per_cell; ++j)
        src_vector(j) = fe_eval.begin_dof_values()[j][v];

      // apply inverse matrix
      matrices[cell * v_len + v].solve(src_vector, false);

      // write solution to dst-vector
      for (unsigned int j = 0; j < fe_eval.dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j][v] = src_vector(j);
    }

    fe_eval.set_dof_values(dst);
  }
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::
    local_apply_block_diagonal(const MF &data, VNumber &dst, const VNumber &src,
                               const Range &range) const {
  FEEvalCell phi(data, ad.dof_index, ad.quad_index);
  for (unsigned int cell = range.first; cell < range.second; ++cell) {
    phi.reinit(cell);
    phi.read_dof_values(src);

    for (unsigned int v = 0; v < v_len; ++v) {
      // fill source vector
      Vector<Number> src_vector(phi.dofs_per_cell);
      Vector<Number> dst_vector(phi.dofs_per_cell);
      for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
        src_vector(j) = phi.begin_dof_values()[j][v];

      // apply inverse matrix
      matrices[cell * v_len + v].vmult(dst_vector, src_vector, false);

      // write solution to dst-vector
      for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
        phi.begin_dof_values()[j][v] = dst_vector(j);
    }

    phi.set_dof_values(dst);
  }
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::
    local_apply_cell_block_diagonal(const MF &data, BMatrix &dst,
                                    const BMatrix & /*src*/,
                                    const Range &range) const {
  FEEvalCell phi(data, ad.dof_index, ad.quad_index);
  // loop over the range of macro cells
  for (auto cell = range.first; cell < range.second; ++cell) {
    const unsigned int n_filled_lanes =
        data.n_active_entries_per_cell_batch(cell);
    // reinit cell
    phi.reinit(cell);
    // loop over all standard basis
    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
      // write standard basis into dof values of FEEvaluation
      this->create_standard_basis(j, phi);
      phi.evaluate(this->ad.cell_evaluate.value,
                   this->ad.cell_evaluate.gradient,
                   this->ad.cell_evaluate.hessians);
      // perform local vmult
      this->do_cell_integral(phi);
      phi.integrate(this->ad.cell_integrate.value,
                    this->ad.cell_integrate.gradient);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int v = 0; v < n_filled_lanes; ++v)
          dst[cell * v_len + v](i, j) += phi.begin_dof_values()[i][v];
    }
  }
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::
    local_apply_face_block_diagonal(const MF &data, BMatrix &dst,
                                    const BMatrix & /*src*/,
                                    const Range &range) const {

  FEEvalFace phi_n(data, true, ad.dof_index, ad.quad_index);
  FEEvalFace phi_p(data, false, ad.dof_index, ad.quad_index);

  // loop over the range of macro cells
  for (auto face = range.first; face < range.second; ++face) {
    const unsigned int n_filled_lanes =
        data.n_active_entries_per_face_batch(face);
    // reinit cell
    phi_n.reinit(face);
    phi_p.reinit(face);

    // interior: loop over all standard basis
    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
      // write standard basis into dof values of FEEvaluation
      this->create_standard_basis(j, phi_n);
      // perform local vmult
      phi_n.evaluate(this->ad.internal_evaluate.value,
                     this->ad.internal_evaluate.gradient);
      this->do_face_int_integral(phi_n);
      phi_n.integrate(this->ad.internal_integrate.value,
                      this->ad.internal_integrate.gradient);
      for (unsigned int v = 0; v < n_filled_lanes; ++v) {
        const unsigned int cell = data.get_face_info(face).cells_interior[v];
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          dst[cell](i, j) += phi_n.begin_dof_values()[i][v];
      }
    }

    // exterior: loop over all standard basis
    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
      // write standard basis into dof values of FEEvaluation
      this->create_standard_basis(j, phi_p);
      // perform local vmult
      phi_p.evaluate(this->ad.internal_evaluate.value,
                     this->ad.internal_evaluate.gradient);
      this->do_face_ext_integral(phi_p);
      phi_p.integrate(this->ad.internal_integrate.value,
                      this->ad.internal_integrate.gradient);
      for (unsigned int v = 0; v < n_filled_lanes; ++v) {
        const unsigned int cell = data.get_face_info(face).cells_exterior[v];
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          dst[cell](i, j) += phi_p.begin_dof_values()[i][v];
      }
    }
  }
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::
    local_apply_boundary_block_diagonal(const MF & data, BMatrix & dst,
                                        const BMatrix & /*src*/,
                                        const Range & range) const {
  FEEvalFace phi(data, true, ad.dof_index, ad.quad_index);

  // loop over the range of macro cells
  for (auto face = range.first; face < range.second; ++face) {
    const unsigned int n_filled_lanes =
        data.n_active_entries_per_face_batch(face);
    // reinit cell
    phi.reinit(face);
    auto bid = data.get_boundary_id(face);
    // interior: loop over all standard basis
    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
      // write standard basis into dof values of FEEvaluation
      this->create_standard_basis(j, phi);
      // perform local vmult
      phi.evaluate(this->ad.boundary_evaluate.value,
                     this->ad.boundary_evaluate.gradient);
      this->do_boundary_integral(phi, OperatorType::homogeneous, bid);
      phi.integrate(this->ad.boundary_integrate.value,
                      this->ad.boundary_integrate.gradient);
      for (unsigned int v = 0; v < n_filled_lanes; ++v) {
        const unsigned int cell = data.get_face_info(face).cells_interior[v];
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          dst[cell](i, j) += phi.begin_dof_values()[i][v];
      }
    }
  }
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::
    local_apply_cell_system_matrix(const MF &data, SMatrix &dst,
                                   const SMatrix & /*src*/,
                                   const Range &range) const {
  FEEvalCell phi(data, ad.dof_index, ad.quad_index);
  // loop over the range of macro cells
  for (auto cell = range.first; cell < range.second; ++cell) {
    // determine number of filled vector lanes
    const unsigned int n_filled_lanes = data.n_components_filled(cell);

    // create a temporal full matrix for the local element matrix of each ...
    // cell of each macro cell and ...
    FMatrix matrices[v_len];
    // set their size
    std::fill_n(matrices, v_len, FMatrix(dofs_per_cell, dofs_per_cell));

    // reinit cell
    phi.reinit(cell);

    // loop over all standard basis
    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
      // write standard basis into dof values of FEEvaluation
      this->create_standard_basis(j, phi);
      phi.evaluate(this->ad.cell_evaluate.value,
                   this->ad.cell_evaluate.gradient,
                   this->ad.cell_evaluate.hessians);
      // perform local vmult
      this->do_cell_integral(phi);

      phi.integrate(this->ad.cell_integrate.value,
                    this->ad.cell_integrate.gradient);

      // insert result vector into local matrix
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices[v](i, j) = phi.begin_dof_values()[i][v];
    }

    // finally assemble local matrix into global matrix
    for (unsigned int i = 0; i < n_filled_lanes; i++) {
      auto cell_i = data.get_cell_iterator(cell, i);
      cell_i->distribute_local_to_global(matrices[i], dst);
    }
  }
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::
    local_apply_face_system_matrix(const MF &data, SMatrix &dst,
                                   const SMatrix & /*src*/,
                                   const Range &range) const {

  FEEvalFace phi_n(data, true, ad.dof_index, ad.quad_index);
  FEEvalFace phi_p(data, false, ad.dof_index, ad.quad_index);

  // loop over the range of macro faces
  for (auto face = range.first; face < range.second; ++face) {
    // determine number of filled vector lanes
    const unsigned int n_filled_lanes =
        data.n_active_entries_per_face_batch(face);
    // create two local matrix: first one tested by v1 and ...
    FMatrix matrices_1[v_len];
    std::fill_n(matrices_1, v_len, FMatrix(dofs_per_cell, dofs_per_cell));
    // ... the other tested by v2
    FMatrix matrices_2[v_len];
    std::fill_n(matrices_2, v_len, FMatrix(dofs_per_cell, dofs_per_cell));

    // reinit face
    phi_n.reinit(face);
    phi_p.reinit(face);

    // process trial function u1: loop over all standard basis
    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
      // write standard basis into dof values of first FEFaceEvaluation and
      // clear dof values of second FEFaceEvaluation
      this->create_standard_basis(j, phi_n, phi_p);
      // do loacal vmult

      phi_n.evaluate(this->ad.internal_evaluate.value,
                     this->ad.internal_evaluate.gradient);
      phi_p.evaluate(this->ad.internal_evaluate.value,
                     this->ad.internal_evaluate.gradient);
      this->do_face_integral(phi_n, phi_p);
      phi_n.integrate(this->ad.internal_integrate.value,
                      this->ad.internal_integrate.gradient);
      phi_p.integrate(this->ad.internal_integrate.value,
                      this->ad.internal_integrate.gradient);

      // insert result vector into local matrix u1_v1
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices_1[v](i, j) = phi_n.begin_dof_values()[i][v];

      // insert result vector into local matrix  u1_v2
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices_2[v](i, j) = phi_p.begin_dof_values()[i][v];
    }

    // save local matrices into global matrix
    for (unsigned int i = 0; i < n_filled_lanes; i++) {
      // cell number of minus
      const unsigned int cell_number_1 =
          data.get_face_info(face).cells_interior[i];
      // cell number of plus
      const unsigned int cell_number_2 =
          data.get_face_info(face).cells_exterior[i];

      // cell reference to cell minus
      auto cell_m =
          data.get_cell_iterator(cell_number_1 / v_len, cell_number_1 % v_len);
      // cell reference to cell plus
      auto cell_p =
          data.get_cell_iterator(cell_number_2 / v_len, cell_number_2 % v_len);

      // get position in global matrix
      std::vector<types::global_dof_index> dof_indices_m(dofs_per_cell);
      cell_m->get_dof_indices(dof_indices_m);
      std::vector<types::global_dof_index> dof_indices_p(dofs_per_cell);
      cell_p->get_dof_indices(dof_indices_p);

      // save u1_v1
      constraint->distribute_local_to_global(matrices_1[i], dof_indices_m,
                                             dof_indices_m, dst);
      // save u1_v2
      constraint->distribute_local_to_global(matrices_2[i], dof_indices_p,
                                             dof_indices_m, dst);
    }

    // process trial function u1: loop over all standard basis
    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
      // write standard basis into dof values of first FEFaceEvaluation and
      // clear dof values of second FEFaceEvaluation
      this->create_standard_basis(j, phi_p, phi_n);
      // do loacal vmult

      phi_n.evaluate(this->ad.internal_evaluate.value,
                     this->ad.internal_evaluate.gradient);
      phi_p.evaluate(this->ad.internal_evaluate.value,
                     this->ad.internal_evaluate.gradient);
      this->do_face_integral(phi_n, phi_p);
      phi_n.integrate(this->ad.internal_integrate.value,
                      this->ad.internal_integrate.gradient);
      phi_p.integrate(this->ad.internal_integrate.value,
                      this->ad.internal_integrate.gradient);

      // insert result vector into local matrix u1_v1
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices_1[v](i, j) = phi_n.begin_dof_values()[i][v];

      // insert result vector into local matrix  u1_v2
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices_2[v](i, j) = phi_p.begin_dof_values()[i][v];
    }

    // save local matrices into global matrix
    for (unsigned int i = 0; i < n_filled_lanes; i++) {
      // cell number of minus
      const unsigned int cell_number_1 =
          data.get_face_info(face).cells_interior[i];
      // cell number of plus
      const unsigned int cell_number_2 =
          data.get_face_info(face).cells_exterior[i];

      // cell reference to cell minus
      auto cell_m =
          data.get_cell_iterator(cell_number_1 / v_len, cell_number_1 % v_len);
      // cell reference to cell plus
      auto cell_p =
          data.get_cell_iterator(cell_number_2 / v_len, cell_number_2 % v_len);

      // get position in global matrix
      std::vector<types::global_dof_index> dof_indices_m(dofs_per_cell);
      cell_m->get_dof_indices(dof_indices_m);
      std::vector<types::global_dof_index> dof_indices_p(dofs_per_cell);
      cell_p->get_dof_indices(dof_indices_p);

      // save u2_v1
      constraint->distribute_local_to_global(matrices_1[i], dof_indices_m,
                                             dof_indices_p, dst);
      // save u2_v2
      constraint->distribute_local_to_global(matrices_2[i], dof_indices_p,
                                             dof_indices_p, dst);
    }
  }
}

template <int dim, int degree, typename Number, typename AdditionalData>
void OperatorBase<dim, degree, Number, AdditionalData>::
    local_apply_boundary_system_matrix(const MF & data, SMatrix & dst,
                                       const SMatrix & /*src*/,
                                       const Range & range) const {

  FEEvalFace phi(data, true, ad.dof_index, ad.quad_index);

  // loop over the range of macro faces
  for (auto face = range.first; face < range.second; ++face) {
    // determine number of filled vector lanes
    const unsigned int n_filled_lanes =
        data.n_active_entries_per_face_batch(face);
    
    FMatrix matrices[v_len];
    std::fill_n(matrices, v_len, FMatrix(dofs_per_cell, dofs_per_cell));

    // reinit face
    phi.reinit(face);
    auto bid = data.get_boundary_id(face);

    // process trial function u1: loop over all standard basis
    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
      // write standard basis into dof values of first FEFaceEvaluation and
      // clear dof values of second FEFaceEvaluation
      this->create_standard_basis(j, phi);

      phi.evaluate(this->ad.boundary_evaluate.value,
                     this->ad.boundary_evaluate.gradient);
      // do loacal vmult
      this->do_boundary_integral(phi, OperatorType::homogeneous, bid);
      phi.integrate(this->ad.boundary_integrate.value,
                      this->ad.boundary_integrate.gradient);

      // insert result vector into local matrix u1_v1
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int v = 0; v < n_filled_lanes; ++v)
          matrices[v](i, j) = phi.begin_dof_values()[i][v];
    }

    // save local matrices into global matrix
    for (unsigned int i = 0; i < n_filled_lanes; i++) {
      // cell number of minus
      const unsigned int cell_num =
          data.get_face_info(face).cells_interior[i];

      // cell reference to cell minus
      data.get_cell_iterator(cell_num / v_len, cell_num % v_len)
              ->distribute_local_to_global(matrices[i], dst);
    }
  }
}
