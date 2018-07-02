#include "mg_coarse_ml.h"

template <typename Operator, typename Number>
MGCoarseML<Operator, Number>::MGCoarseML(Operator const &matrix,
                                         Operator const &matrix_q, bool setup,
                                         int level)
    : coarse_matrix(matrix), coarse_matrix_q(matrix_q) {
  if (setup)
    this->reinit(level);
}

template <typename Operator, typename Number>
MGCoarseML<Operator, Number>::~MGCoarseML() {}

template <typename Operator, typename Number>
void MGCoarseML<Operator, Number>::reinit(int level) {

  bool type = false;

  // create wrapper
  if (type)
    // ... DG:
    wrapper.reset(new MGCoarseMLDG<DIM, MultigridNumber>(
        level, coarse_matrix, coarse_matrix_q, system_matrix));
  else
    // ... CG:
    wrapper.reset(new MGCoarseMLCG<DIM, MultigridNumber>(
        level, coarse_matrix, coarse_matrix_q, system_matrix));

  // initialize system matrix
  wrapper->init_system();

  // configure Trilinos' AMG
  auto data = TrilinosWrappers::PreconditionAMG::AdditionalData();
  data.smoother_sweeps = 1;
  data.n_cycles = 1;
  data.smoother_type = "ILU";

  // intialize Trilinos' AMG
  pamg.initialize(system_matrix, data);
}

template <typename Operator, typename Number>
void MGCoarseML<Operator, Number>::update(
    MatrixOperatorBase const * /*matrix_operator*/) {}

template <typename Operator, typename Number>
void MGCoarseML<Operator, Number>::
operator()(const unsigned int /*level*/,
           parallel::distributed::Vector<MultigridNumber> &dst,
           const parallel::distributed::Vector<MultigridNumber> &src) const {

  // TODO: remove unnecessary moves...
  parallel::distributed::Vector<MultigridNumber> src_0, dst_0;
  src_0.reinit(src, false);
  src_0.copy_locally_owned_data_from(src);
  dst_0.reinit(dst, false);

  parallel::distributed::Vector<MultigridNumber> src__, dst__;
  wrapper->init_vectors(src__, dst__);

  // [float] DG -> CG
  wrapper->vmult_pre(src__, src_0);

  // create temporal vectors of type TrilinosScalar
  parallel::distributed::Vector<TrilinosWrappers::SparseMatrix::value_type>
      dst_;
  dst_.reinit(dst__, false);
  parallel::distributed::Vector<TrilinosWrappers::SparseMatrix::value_type>
      src_;
  src_.reinit(src__, false);

  // [float -> double] convert Operator::value_type to TrilinosScalar
  src_.copy_locally_owned_data_from(src__);

  if (true) {
    // use PCG with Trilinos to perform AMG
    ReductionControl solver_control(10000, 1e-20, 1e-2);
    solver_control.set_failure_criterion(100.0);
    SolverCG<parallel::distributed::Vector<
        TrilinosWrappers::SparseMatrix::value_type>>
        solver(solver_control);
    solver.solve(system_matrix, dst_, src_, pamg);
    //std::cout << "   " << solver_control.last_step() << "   "
    //          << solver_control.last_value() << "   " << std::endl;
  } else {
    // use Trilinos to perform AMG
    pamg.vmult(dst_, src_);
  }

  // [double -> float]convert TrilinosScalar to Operator::value_type
  dst__.copy_locally_owned_data_from(dst_);
  dst__.update_ghost_values();
  // [float] CG -> DG
  wrapper->vmult_post(dst_0, dst__);
  dst.copy_locally_owned_data_from(dst_0);
}

template <typename Operator, typename Number>
void MGCoarseML<Operator, Number>::vmult(
    parallel::distributed::Vector<Number> &dst,
    const parallel::distributed::Vector<Number> &src) const {

  parallel::distributed::Vector<MultigridNumber> src_0, dst_0;
  src_0.reinit(src, true);
  src_0.copy_locally_owned_data_from(src);
  dst_0.reinit(dst, true);

  parallel::distributed::Vector<MultigridNumber> src__, dst__;
  wrapper->init_vectors(src__, dst__);

  // [float] DG -> CG
  wrapper->vmult_pre(src__, src_0);

  // create temporal vectors of type TrilinosScalar
  parallel::distributed::Vector<TrilinosWrappers::SparseMatrix::value_type>
      dst_;
  dst_.reinit(dst__, true);
  parallel::distributed::Vector<TrilinosWrappers::SparseMatrix::value_type>
      src_;
  src_.reinit(src__, true);

  // [float -> double] convert Operator::value_type to TrilinosScalar
  src_.copy_locally_owned_data_from(src__);

  // [double] use Trilinos to perform AMG
  pamg.vmult(dst_, src_);

  // [double -> float]convert TrilinosScalar to Operator::value_type
  dst__.copy_locally_owned_data_from(dst_);

  // [float] CG -> DG
  wrapper->vmult_post(dst_0, dst__);
  dst.copy_locally_owned_data_from(dst_0);
}

#include "mg_coarse_ml.hpp"
