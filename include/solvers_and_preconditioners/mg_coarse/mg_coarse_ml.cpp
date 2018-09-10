#include "mg_coarse_ml.h"

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#ifdef DEAL_II_WITH_TRILINOS

template<typename Operator, typename Number>
MGCoarseML<Operator, Number>::MGCoarseML(Operator const & matrix,
                                         Operator const & matrix_q,
                                         bool             setup,
                                         int              level,
                                         MGCoarseMLData   data)
  : operator_dg(matrix), operator_cg(matrix_q)
{
  if(setup)
    this->reinit(level, data);
}

template<typename Operator, typename Number>
MGCoarseML<Operator, Number>::~MGCoarseML()
{
}

template<typename Operator, typename Number>
void
MGCoarseML<Operator, Number>::reinit(int level, MGCoarseMLData data_in)
{
  AssertThrow(level >= 0, ExcMessage("Invalid level specified!"));

  // save additional_data locally: we need it later
  this->additional_data = data_in;

  // create wrapper
  if(this->additional_data.transfer_to_continuous_galerkin)
  {
    const unsigned int degree = operator_dg.get_data().get_dof_handler().get_fe().degree;
    this->transfer.reset(new CGToDGTransfer<Operator::DIM, MultigridNumber>(
      operator_dg.get_data(), operator_cg.get_data(), level, degree));
  }

  // initialize system matrix
  auto & matrix_temp = this->additional_data.transfer_to_continuous_galerkin ? operator_cg : operator_dg;
  matrix_temp.init_system_matrix(system_matrix);
  matrix_temp.calculate_system_matrix(system_matrix);

  // intialize Trilinos' AMG
  pamg.initialize(system_matrix, this->additional_data.amg_data);
}

template<typename Operator, typename Number>
void
MGCoarseML<Operator, Number>::update(MatrixOperatorBase const * /*matrix_operator*/)
{
}

template<typename Operator, typename Number>
void
MGCoarseML<Operator, Number>::operator()(const unsigned int /*level*/,
                                         parallel::distributed::Vector<MultigridNumber> &       dst,
                                         const parallel::distributed::Vector<MultigridNumber> & src) const
{
  // TODO: remove unnecessary moves...
  parallel::distributed::Vector<MultigridNumber> src_0, dst_0;
  src_0.reinit(src, false);
  src_0.copy_locally_owned_data_from(src);
  dst_0.reinit(dst, false);

  parallel::distributed::Vector<MultigridNumber>  cg_src__, cg_dst__;
  parallel::distributed::Vector<MultigridNumber> *src__, *dst__;
  if(this->additional_data.transfer_to_continuous_galerkin)
  {
    this->operator_cg.initialize_dof_vector(cg_src__);
    this->operator_cg.initialize_dof_vector(cg_dst__);
    src__ = &cg_src__;
    dst__ = &cg_dst__;
    transfer->toCG(*src__, src_0);
  }
  else
  {
    src__ = &src_0;
    dst__ = &dst_0;
  }

  // create temporal vectors of type TrilinosScalar
  parallel::distributed::Vector<TrilinosWrappers::SparseMatrix::value_type> dst_;
  dst_.reinit(*dst__, false);
  parallel::distributed::Vector<TrilinosWrappers::SparseMatrix::value_type> src_;
  src_.reinit(*src__, false);

  // [float -> double] convert Operator::value_type to TrilinosScalar
  src_.copy_locally_owned_data_from(*src__);

  if(additional_data.use_conjugate_gradient_solver)
  {
    // use PCG with Trilinos to perform AMG
    ReductionControl solver_control(additional_data.max_iter,
                                    additional_data.solver_tolerance_abs,
                                    additional_data.solver_tolerance_rel);
    solver_control.set_failure_criterion(additional_data.pcg_failure_criterion);
    SolverCG<parallel::distributed::Vector<TrilinosWrappers::SparseMatrix::value_type>> solver(
      solver_control);
    solver.solve(system_matrix, dst_, src_, pamg);
    // std::cout << "   " << solver_control.last_step() << "   "
    //          << solver_control.last_value() << "   " << std::endl;
  }
  else
  {
    // use Trilinos to perform AMG
    pamg.vmult(dst_, src_);
  }

  // [double -> float]convert TrilinosScalar to Operator::value_type
  dst__->copy_locally_owned_data_from(dst_);
  dst__->update_ghost_values();
  // [float] CG -> DG
  if(this->additional_data.transfer_to_continuous_galerkin)
    transfer->toDG(dst_0, *dst__);
  dst.copy_locally_owned_data_from(dst_0);
}

template<typename Operator, typename Number>
void
MGCoarseML<Operator, Number>::vmult(parallel::distributed::Vector<Number> &,
                                    const parallel::distributed::Vector<Number> &) const
{
  AssertThrow(false, ExcMessage("MGCoarseML::vmult not implemented yet!"));
}

#else


template<typename Operator, typename Number>
MGCoarseML<Operator, Number>::MGCoarseML(Operator const &, Operator const &, bool, int, MGCoarseMLData)
{
}

template<typename Operator, typename Number>
MGCoarseML<Operator, Number>::~MGCoarseML()
{
}

template<typename Operator, typename Number>
void
MGCoarseML<Operator, Number>::reinit(int, MGCoarseMLData)
{
  AssertThrow(false, ExcMessage("deal.II is not compiled with Trilinos!"));
}

template<typename Operator, typename Number>
void
MGCoarseML<Operator, Number>::update(MatrixOperatorBase const *)
{
  AssertThrow(false, ExcMessage("deal.II is not compiled with Trilinos!"));
}

template<typename Operator, typename Number>
void
MGCoarseML<Operator, Number>::operator()(const unsigned int,
                                         parallel::distributed::Vector<MultigridNumber> &,
                                         const parallel::distributed::Vector<MultigridNumber> &) const
{
  AssertThrow(false, ExcMessage("deal.II is not compiled with Trilinos!"));
}

template<typename Operator, typename Number>
void
MGCoarseML<Operator, Number>::vmult(parallel::distributed::Vector<Number> &,
                                    const parallel::distributed::Vector<Number> &) const
{
  AssertThrow(false, ExcMessage("deal.II is not compiled with Trilinos!"));
}

#endif

#include "mg_coarse_ml.hpp"
