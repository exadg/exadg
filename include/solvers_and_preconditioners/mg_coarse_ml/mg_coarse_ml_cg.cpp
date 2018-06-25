#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/constraint_matrix.templates.h>

#include "mg_coarse_ml_cg.h"

#include "mesh_worker_wrapper.h"

using namespace dealii;

template <int DIM, typename Number>
MGCoarseMLCG<DIM, Number>::MGCoarseMLCG(
    const int level, const MatrixOperatorBaseNew<DIM, Number> &coarse_matrix_dg,
    const MatrixOperatorBaseNew<DIM, Number> &coarse_matrix_q,
    TrilinosWrappers::SparseMatrix &system_matrix)
    : MGCoarseMLWrapper<DIM, Number>(level, coarse_matrix_q, system_matrix),
      coarse_matrix_dg(coarse_matrix_dg),
      transfer(this->coarse_matrix_dg.get_data(),
               this->coarse_matrix.get_data(), level, this->degree) {}

template <int DIM, typename Number>
void MGCoarseMLCG<DIM, Number>::init_system() {

  const ConstraintMatrix &constraints_fe =
      this->coarse_matrix.get_constraint_matrix();

  // extract relevant data structures
  const DoFHandler<DIM> &dof_handler =
      this->coarse_matrix.get_data().get_dof_handler();

  FE_Q<DIM> fe(dof_handler.get_fe().degree);
  MappingQGeneric<DIM> mapping(dof_handler.get_fe().degree);

  TrilinosWrappers::SparsityPattern dsp(
      dof_handler.locally_owned_mg_dofs(this->level), MPI_COMM_WORLD);
  MGTools::make_sparsity_pattern(dof_handler, dsp, this->level);
  dsp.compress();

  this->system_matrix.reinit(dsp);

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

  // create dummy right hand side vector (it is needed by SystemSimple)
  parallel::distributed::Vector<double> rhs;
  // this->coarse_matrix.get_data().initialize_dof_vector(rhs);

  // To assemble only a matrix deal.II provides the class MatrixSimple. We do
  // not use this class since it keeps rows and columns of the matrix related
  // to DBC empty. Singular matrices are not treated well by solvers. Instead,
  // we use SystemSimple: it assembles both the matrix and the right hand side.
  // This class fills all entries of the main diagonal matrix.
  MeshWorker::Assembler::SystemSimple<
      TrilinosWrappers::SparseMatrix,
      LinearAlgebra::distributed::Vector<double>>
      assembler;
  assembler.initialize(this->system_matrix, rhs);
  assembler.initialize(constraints_fe);

  MeshWorker::integration_loop<DIM, DIM>(
      dof_handler.begin_mg(this->level), dof_handler.end_mg(this->level),
      dof_info, info_box, MeshWorkerWrapper<MatrixOperatorBaseNew<DIM, Number>>(
                              this->coarse_matrix, true, false, false),
      assembler, MeshWorker::LoopControl());

  this->system_matrix.compress(VectorOperation::add);
}

template <int DIM, typename Number>
void MGCoarseMLCG<DIM, Number>::vmult_pre(
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src) {

  this->transfer.toCG(dst, src);
}

template <int DIM, typename Number>
void MGCoarseMLCG<DIM, Number>::vmult_post(
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src) {

  this->transfer.toDG(dst, src);
}

#include "mg_coarse_ml_cg.hpp"