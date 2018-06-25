#include "mg_coarse_ml_dg.h"

#include "mesh_worker_wrapper.h"

using namespace dealii;

template <int DIM, typename Number>
MGCoarseMLDG<DIM, Number>::MGCoarseMLDG(
    const int level, const MatrixOperatorBaseNew<DIM, Number> &coarse_matrix,
    const MatrixOperatorBase & /*coarse_matrix*/,
    TrilinosWrappers::SparseMatrix &system_matrix)
    : MGCoarseMLWrapper<DIM, Number>(level, coarse_matrix, system_matrix) {}

template <int DIM, typename Number>
void MGCoarseMLDG<DIM, Number>::init_system() {

  // extract relevant data structures
  const DoFHandler<DIM> &dof_handler =
      this->coarse_matrix.get_data().get_dof_handler();

  // create temporal classes
  FE_DGQ<DIM> fe(dof_handler.get_fe().degree);
  MappingQGeneric<DIM> mapping(dof_handler.get_fe().degree);

  // initialize the system matrix ...
  // ... create a sparsity pattern
  TrilinosWrappers::SparsityPattern dsp(
      dof_handler.locally_owned_mg_dofs(this->level), MPI_COMM_WORLD);
  MGTools::make_flux_sparsity_pattern(dof_handler, dsp, this->level);
  dsp.compress();
  this->system_matrix.reinit(dsp);

  // ... assemble system matrix with the help of MeshWorker and Assembler
  MeshWorker::IntegrationInfoBox<DIM> info_box;
  const unsigned int n_gauss_points = dof_handler.get_fe().degree + 1;
  info_box.initialize_gauss_quadrature(n_gauss_points, n_gauss_points,
                                       n_gauss_points);
  info_box.initialize_update_flags();
  UpdateFlags update_flags =
      update_quadrature_points | update_values | update_gradients;
  info_box.add_update_flags(update_flags, true, true, true, true);
  info_box.initialize(fe, mapping);
  MeshWorker::DoFInfo<DIM, DIM, double> dof_info(dof_handler);

  MeshWorker::Assembler::MatrixSimple<TrilinosWrappers::SparseMatrix> assembler;
  assembler.initialize(this->system_matrix);

  MeshWorker::LoopControl lc;
  lc.faces_to_ghost = MeshWorker::LoopControl::both;

  MeshWorker::integration_loop<DIM, DIM>(
      dof_handler.begin_mg(this->level), dof_handler.end_mg(this->level),
      dof_info, info_box, MeshWorkerWrapper<MatrixOperatorBaseNew<DIM, Number>>(
                              this->coarse_matrix),
      assembler, lc);
}

template <int DIM, typename Number>
void MGCoarseMLDG<DIM, Number>::vmult_pre(
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src) {
  dst = src;
}

template <int DIM, typename Number>
void MGCoarseMLDG<DIM, Number>::vmult_post(
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src) {
  dst = src;
}

#include "mg_coarse_ml_dg.hpp"