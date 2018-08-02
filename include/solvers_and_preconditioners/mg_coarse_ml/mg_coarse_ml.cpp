#include "mg_coarse_ml.h"

#ifdef DEAL_II_WITH_TRILINOS

template <typename Operator, typename Number>
MGCoarseML<Operator, Number>::MGCoarseML(Operator const &matrix,
                                         Operator const &matrix_q, 
                                         bool setup,
                                         int level,
                                         MGCoarseMLData data)
    : coarse_matrix(matrix), coarse_matrix_q(matrix_q) {
  if (setup)
    this->reinit(level,data);
}

template <typename Operator, typename Number>
MGCoarseML<Operator, Number>::~MGCoarseML() {}

template <typename Operator, typename Number>
void MGCoarseML<Operator, Number>::reinit(int level, MGCoarseMLData data_in) {
    
  AssertThrow(level >= 0 , ExcMessage("Invalid level specified!"));

  // save additional_data locally: we need it later
  this->additional_data = data_in;

  // create wrapper
  if (this->additional_data.use_cg)
    // ... CG:
    wrapper.reset(new MGCoarseMLCG<DIM, MultigridNumber>(
        level, coarse_matrix, coarse_matrix_q, system_matrix));
  else
    // ... DG:
    wrapper.reset(new MGCoarseMLDG<DIM, MultigridNumber>(
        level, coarse_matrix, coarse_matrix_q, system_matrix));

  // initialize system matrix
  wrapper->init_system();

  // intialize Trilinos' AMG
  pamg.initialize(system_matrix, this->additional_data.amg_data);
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

  if (additional_data.use_pcg) {
    // use PCG with Trilinos to perform AMG
    ReductionControl solver_control(additional_data.pcg_max_iterations, 
                                    additional_data.pcg_abs_residuum, 
                                    additional_data.pcg_rel_residuum);
    solver_control.set_failure_criterion(additional_data.pcg_failure_criterion);
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

#else


template <typename Operator, typename Number>
MGCoarseML<Operator, Number>::MGCoarseML(Operator const &,
                                         Operator const &, 
                                         bool ,
                                         int ,
                                         MGCoarseMLData ){
}

template <typename Operator, typename Number>
MGCoarseML<Operator, Number>::~MGCoarseML() {
}

template <typename Operator, typename Number>
void MGCoarseML<Operator, Number>::reinit(int , MGCoarseMLData ) {
    AssertThrow(false, ExcMessage("deal.II is not compiled with Trilinos!"));
}

template <typename Operator, typename Number>
void MGCoarseML<Operator, Number>::update(
    MatrixOperatorBase const * ) {
    AssertThrow(false, ExcMessage("deal.II is not compiled with Trilinos!"));
}

template <typename Operator, typename Number>
void MGCoarseML<Operator, Number>::
operator()(const unsigned int ,
           parallel::distributed::Vector<MultigridNumber> &,
           const parallel::distributed::Vector<MultigridNumber> &) const {
    AssertThrow(false, ExcMessage("deal.II is not compiled with Trilinos!"));
}

template <typename Operator, typename Number>
void MGCoarseML<Operator, Number>::vmult(
    parallel::distributed::Vector<Number> &,
    const parallel::distributed::Vector<Number> &) const {
    AssertThrow(false, ExcMessage("deal.II is not compiled with Trilinos!"));
}

#endif

#include "mg_coarse_ml.hpp"
