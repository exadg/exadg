/*
 * MultigridPreconditionerWrapperBase.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_ADAPTER_BASE_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_ADAPTER_BASE_H_

#include "../transfer/mg_transfer_mf_p.h"

#include <deal.II/fe/fe_tools.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include "../mg_coarse_ml/mg_coarse_ml.h"

// smoothers
#include "../smoother/cg_smoother.h"
#include "../smoother/chebyshev_smoother.h"
#include "../smoother/gmres_smoother.h"
#include "../smoother/jacobi_smoother.h"
#include "../smoother/smoother_base.h"

#include "../mg_coarse/mg_coarse_grid_solvers.h"
#include "multigrid_input_parameters.h"
#include "multigrid_preconditioner.h"

namespace {
// manually compute eigenvalues for the coarsest level for proper setup of the
// Chebyshev iteration
template <typename Operator>
std::pair<double, double> compute_eigenvalues(
    const Operator &op,
    const parallel::distributed::Vector<typename Operator::value_type>
        &inverse_diagonal,
    const unsigned int eig_n_iter = 10000) {
  typedef typename Operator::value_type value_type;
  parallel::distributed::Vector<value_type> left, right;
  left.reinit(inverse_diagonal);
  right.reinit(inverse_diagonal, true);
  // NB: initialize rand in order to obtain "reproducible" results !!!
  srand(1);
  for (unsigned int i = 0; i < right.local_size(); ++i)
    right.local_element(i) = (double)rand() / RAND_MAX;
  op.apply_nullspace_projection(right);

  SolverControl control(eig_n_iter, right.l2_norm() * 1e-5);
  internal::PreconditionChebyshevImplementation::EigenvalueTracker
      eigenvalue_tracker;
  SolverCG<parallel::distributed::Vector<value_type>> solver(control);
  solver.connect_eigenvalues_slot(std::bind(
      &internal::PreconditionChebyshevImplementation::EigenvalueTracker::slot,
      &eigenvalue_tracker, std::placeholders::_1));

  JacobiPreconditioner<Operator> preconditioner(op);

  try {
    solver.solve(op, left, right, preconditioner);
  } catch (SolverControl::NoConvergence &) {
  }

  std::pair<double, double> eigenvalues;
  if (eigenvalue_tracker.values.empty()) {
    eigenvalues.first = eigenvalues.second = 1.;
  } else {
    eigenvalues.first = eigenvalue_tracker.values.front();
    eigenvalues.second = eigenvalue_tracker.values.back();
  }
  return eigenvalues;
}

template <typename Number> struct EigenvalueTracker {
public:
  void slot(const std::vector<Number> &eigenvalues) { values = eigenvalues; }

  std::vector<Number> values;
};

// manually compute eigenvalues for the coarsest level for proper setup of the
// Chebyshev iteration
template <typename Operator>
std::pair<std::complex<double>, std::complex<double>> compute_eigenvalues_gmres(
    const Operator &op,
    const parallel::distributed::Vector<typename Operator::value_type>
        &inverse_diagonal,
    const unsigned int eig_n_iter = 10000) {
  typedef typename Operator::value_type value_type;
  parallel::distributed::Vector<value_type> left, right;
  left.reinit(inverse_diagonal);
  right.reinit(inverse_diagonal, true);
  // NB: initialize rand in order to obtain "reproducible" results !!!
  srand(1);
  for (unsigned int i = 0; i < right.local_size(); ++i)
    right.local_element(i) = (double)rand() / RAND_MAX;
  op.apply_nullspace_projection(right);

  ReductionControl control(eig_n_iter, right.l2_norm() * 1.0e-5, 1.0e-5);

  EigenvalueTracker<std::complex<double>> eigenvalue_tracker;
  SolverGMRES<parallel::distributed::Vector<value_type>> solver(control);
  solver.connect_eigenvalues_slot(
      std::bind(&EigenvalueTracker<std::complex<double>>::slot,
                &eigenvalue_tracker, std::placeholders::_1));

  JacobiPreconditioner<Operator> preconditioner(op);

  try {
    solver.solve(op, left, right, preconditioner);
  } catch (SolverControl::NoConvergence &) {
  }

  std::pair<std::complex<double>, std::complex<double>> eigenvalues;
  if (eigenvalue_tracker.values.empty()) {
    eigenvalues.first = eigenvalues.second = 1.;
  } else {
    eigenvalues.first = eigenvalue_tracker.values.front();
    eigenvalues.second = eigenvalue_tracker.values.back();
  }
  return eigenvalues;
}
}

template <int dim, typename value_type, typename Operator>
class MyMultigridPreconditionerBase
    : public PreconditionerBase<value_type>,
      public MGCoarseGridBase<
          parallel::distributed::Vector<typename Operator::value_type>> {

  typedef typename Operator::value_type value_type_operator;

  std::shared_ptr<Operator> underlying_operator;

public:
  MyMultigridPreconditionerBase(std::shared_ptr<Operator> underlying_operator);

  static unsigned int get_next(unsigned int degree) {
    if (degree == 1)
      return 1;
    return (degree + 2) / 2 - 1;
  }

  virtual ~MyMultigridPreconditionerBase();

  MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dofhandler;
  MGLevelObject<std::shared_ptr<MGConstrainedDoFs>> mg_constrained_dofs;

  void initialize(const MultigridData &mg_data_in,
                  const DoFHandler<dim> &dof_handler,
                  const Mapping<dim> &mapping,
          std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const &dirichlet_bc, 
                  void* operator_data_in);

  virtual void initialize_mg_constrained_dofs(const DoFHandler<dim> &,
                                              MGConstrainedDoFs &,
          std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const &dirichlet_bc);

  virtual void update(MatrixOperatorBase const * /*matrix_operator*/);

  void vmult(parallel::distributed::Vector<value_type> &dst,
             const parallel::distributed::Vector<value_type> &src) const;

  virtual void operator()(
      const unsigned int /*level*/,
      parallel::distributed::Vector<value_type_operator> &dst,
      const parallel::distributed::Vector<value_type_operator> &src) const;

  virtual void apply_smoother_on_fine_level(
      parallel::distributed::Vector<value_type_operator> &dst,
      const parallel::distributed::Vector<value_type_operator> &src) const;
  
  virtual void update_smoother(unsigned int /*level*/){
      
  }
  
  virtual void update_coarse_solver(){
      
  }

protected:
  void initialize_smoother(Operator &matrix, unsigned int level);

  void initialize_coarse_solver(Operator &matrix, Operator &matrix_q,
                                const unsigned int coarse_level);

  virtual void
  initialize_multigrid_preconditioner(DoFHandler<dim> const & /*dof_handler*/);

  MultigridData mg_data;
  unsigned int n_global_levels; // TODO

  MGLevelObject<std::shared_ptr<Operator>> mg_matrices;
  typedef parallel::distributed::Vector<value_type_operator> VECTOR_TYPE;
  typedef MGTransferBase<VECTOR_TYPE> MG_TRANSFER;
  MGLevelObject<std::shared_ptr<MG_TRANSFER>> mg_transfer;

  typedef SmootherBase<VECTOR_TYPE> SMOOTHER;
  MGLevelObject<std::shared_ptr<SMOOTHER>> mg_smoother;

  std::shared_ptr<MGCoarseGridBase<VECTOR_TYPE>> mg_coarse;

  std::shared_ptr<
      MultigridPreconditioner<VECTOR_TYPE, Operator, MG_TRANSFER, SMOOTHER>>
      multigrid_preconditioner;

  // for CG
public:
  std::shared_ptr<const DoFHandler<dim>> cg_dofhandler;
  std::shared_ptr<MGConstrainedDoFs> cg_constrained_dofs;
  std::shared_ptr<Operator> cg_matrices;

private:
  void initialize_chebyshev_smoother(Operator &matrix, unsigned int level);

  void initialize_chebyshev_smoother_coarse_grid(Operator &matrix);

  void initialize_chebyshev_smoother_nonsymmetric_operator(Operator &matrix,
                                                           unsigned int level);

  void initialize_chebyshev_smoother_nonsymmetric_operator_coarse_grid(
      Operator &matrix);
};

#include "multigrid_preconditioner_adapter_base.cpp"

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_ADAPTER_BASE_H_ \
          */
