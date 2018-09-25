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

// smoothers
#include "../../functionalities/set_zero_mean_value.h"
#include "../smoother/cg_smoother.h"
#include "../smoother/chebyshev_smoother.h"
#include "../smoother/gmres_smoother.h"
#include "../smoother/jacobi_smoother.h"
#include "../smoother/smoother_base.h"

#include "../mg_coarse/mg_coarse_grid_solvers.h"
#include "multigrid_input_parameters.h"
#include "multigrid_preconditioner.h"

namespace
{
// manually compute eigenvalues for the coarsest level for proper setup of the
// Chebyshev iteration
template<typename Operator, typename VectorType>
std::pair<double, double>
compute_eigenvalues(Operator const &   op,
                    VectorType const & inverse_diagonal,
                    unsigned int const eig_n_iter = 10000)
{
  VectorType left, right;
  left.reinit(inverse_diagonal);
  right.reinit(inverse_diagonal, true);
  // NB: initialize rand in order to obtain "reproducible" results !!!
  srand(1);
  for(unsigned int i = 0; i < right.local_size(); ++i)
    right.local_element(i) = (double)rand() / RAND_MAX;
  if(op.is_singular())
    set_zero_mean_value(right);

  SolverControl control(eig_n_iter, right.l2_norm() * 1e-5);
  internal::PreconditionChebyshevImplementation::EigenvalueTracker eigenvalue_tracker;

  SolverCG<VectorType> solver(control);

  solver.connect_eigenvalues_slot(
    std::bind(&internal::PreconditionChebyshevImplementation::EigenvalueTracker::slot,
              &eigenvalue_tracker,
              std::placeholders::_1));

  JacobiPreconditioner<Operator> preconditioner(op);

  try
  {
    solver.solve(op, left, right, preconditioner);
  }
  catch(SolverControl::NoConvergence &)
  {
  }

  std::pair<double, double> eigenvalues;
  if(eigenvalue_tracker.values.empty())
  {
    eigenvalues.first = eigenvalues.second = 1.;
  }
  else
  {
    eigenvalues.first  = eigenvalue_tracker.values.front();
    eigenvalues.second = eigenvalue_tracker.values.back();
  }

  return eigenvalues;
}

template<typename Number>
struct EigenvalueTracker
{
public:
  void
  slot(const std::vector<Number> & eigenvalues)
  {
    values = eigenvalues;
  }

  std::vector<Number> values;
};

// manually compute eigenvalues for the coarsest level for proper setup of the
// Chebyshev iteration
template<typename Operator, typename VectorType>
std::pair<std::complex<double>, std::complex<double>>
compute_eigenvalues_gmres(Operator const &   op,
                          VectorType const & inverse_diagonal,
                          unsigned int const eig_n_iter = 10000)
{
  VectorType left, right;
  left.reinit(inverse_diagonal);
  right.reinit(inverse_diagonal, true);
  // NB: initialize rand in order to obtain "reproducible" results !!!
  srand(1);
  for(unsigned int i = 0; i < right.local_size(); ++i)
    right.local_element(i) = (double)rand() / RAND_MAX;
  if(op.is_singular())
    set_zero_mean_value(right);

  ReductionControl control(eig_n_iter, right.l2_norm() * 1.0e-5, 1.0e-5);

  EigenvalueTracker<std::complex<double>> eigenvalue_tracker;

  SolverGMRES<VectorType> solver(control);

  solver.connect_eigenvalues_slot(std::bind(&EigenvalueTracker<std::complex<double>>::slot,
                                            &eigenvalue_tracker,
                                            std::placeholders::_1));

  JacobiPreconditioner<Operator> preconditioner(op);

  try
  {
    solver.solve(op, left, right, preconditioner);
  }
  catch(SolverControl::NoConvergence &)
  {
  }

  std::pair<std::complex<double>, std::complex<double>> eigenvalues;
  if(eigenvalue_tracker.values.empty())
  {
    eigenvalues.first = eigenvalues.second = 1.;
  }
  else
  {
    eigenvalues.first  = eigenvalue_tracker.values.front();
    eigenvalues.second = eigenvalue_tracker.values.back();
  }
  return eigenvalues;
}
} // namespace

template<int dim, typename value_type, typename Operator>
class MyMultigridPreconditionerBase : public PreconditionerBase<value_type>
{
public:
  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

  typedef LinearAlgebra::distributed::Vector<typename Operator::value_type> VectorTypeMG;

  MyMultigridPreconditionerBase(std::shared_ptr<Operator> underlying_operator);

private:
  static unsigned int
  get_next_coarser_degree(unsigned int const degree)
  {
    // examples:
    // 9 -> 4; 8 -> 4; 7 -> 3; 6 -> 3; 5 -> 2; 4 -> 2; 3 -> 1; 2 -> 1
    if(degree == 1)
      return 1;

    return degree / 2;
  }

public:
  virtual ~MyMultigridPreconditionerBase();

  // initialization function for purely discontinuous Galerkin usage
  // (in this case no Dirchlet BC is needed for the constraint matrix)
  void
  initialize(MultigridData const &   mg_data_in,
             DoFHandler<dim> const & dof_handler,
             Mapping<dim> const &    mapping,
             void *                  operator_data_in);

  // initialization function for discontinuous and continuous Galerkin methods (WIP)
  // (also: if continuous Galerkin methods should be used as auxiliary space)
  void
  initialize(MultigridData const &                                                mg_data_in,
             DoFHandler<dim> const &                                              dof_handler,
             Mapping<dim> const &                                                 mapping,
             std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const & dirichlet_bc,
             void *                                                               operator_data_in);

private:
  void
  initialize_mg_sequence(parallel::Triangulation<dim> const *                 tria,
                         std::vector<std::pair<unsigned int, unsigned int>> & global_levels,
                         std::vector<unsigned int> &                          h_levels,
                         std::vector<unsigned int> &                          p_levels,
                         unsigned int const                                   degree,
                         MultigridType const                                  mg_type);

  void
  check_mg_sequence(std::vector<std::pair<unsigned int, unsigned int>> const & global_levels);

  void
  initialize_auxiliary_space(
    parallel::Triangulation<dim> const *                                 tria,
    std::vector<std::pair<unsigned int, unsigned int>> &                 global_levels,
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const & dirichlet_bc,
    Mapping<dim> const &                                                 mapping,
    void *                                                               operator_data_in);

  void
  initialize_mg_dof_handler_and_constraints(
    DoFHandler<dim> const &                                              dof_handler,
    parallel::Triangulation<dim> const *                                 tria,
    std::vector<std::pair<unsigned int, unsigned int>> &                 global_levels,
    std::vector<unsigned int> &                                          p_levels,
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const & dirichlet_bc,
    unsigned int                                                         degree);


  void
  initialize_mg_matrices(std::vector<std::pair<unsigned int, unsigned int>> & global_levels,
                         Mapping<dim> const &                                 mapping,
                         void *                                               operator_data_in);

  void
  initialize_smoothers();

  void
  initialize_mg_transfer(parallel::Triangulation<dim> const *                 tria,
                         std::vector<std::pair<unsigned int, unsigned int>> & global_levels,
                         std::vector<unsigned int> & /*h_levels*/,
                         std::vector<unsigned int> & p_levels);

  virtual void
  initialize_mg_constrained_dofs(
    DoFHandler<dim> const &,
    MGConstrainedDoFs &,
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const & dirichlet_bc);

public:
  virtual void
  update(MatrixOperatorBase const * /*matrix_operator*/);

  void
  vmult(VectorType & dst, VectorType const & src) const;

  virtual void
  apply_smoother_on_fine_level(VectorTypeMG & dst, VectorTypeMG const & src) const;

protected:
  virtual void
  update_smoother(unsigned int level);

  virtual void
  update_coarse_solver();

private:
  void
  initialize_smoother(Operator & matrix, unsigned int level);

  void
  initialize_coarse_solver(unsigned int const coarse_level);

  virtual void
  initialize_multigrid_preconditioner();

  MultigridData mg_data;

public:
  unsigned int n_global_levels;

private:
  MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dofhandler;
  MGLevelObject<std::shared_ptr<MGConstrainedDoFs>>     mg_constrained_dofs;

public:
  MGLevelObject<std::shared_ptr<Operator>> mg_matrices;

private:
  typedef MGTransferBase<VectorTypeMG>        MG_TRANSFER;
  MGLevelObject<std::shared_ptr<MG_TRANSFER>> mg_transfer;

  typedef SmootherBase<VectorTypeMG>       SMOOTHER;
  MGLevelObject<std::shared_ptr<SMOOTHER>> mg_smoother;

  std::shared_ptr<MGCoarseGridBase<VectorTypeMG>> mg_coarse;

  std::shared_ptr<MultigridPreconditioner<VectorTypeMG, Operator, MG_TRANSFER, SMOOTHER>>
    multigrid_preconditioner;

  std::shared_ptr<Operator> underlying_operator;

  // for CG
  std::shared_ptr<DoFHandler<dim> const> cg_dofhandler;
  std::shared_ptr<MGConstrainedDoFs>     cg_constrained_dofs;
  std::shared_ptr<Operator>              cg_matrices;

  void
  initialize_chebyshev_smoother(Operator & matrix, unsigned int level);

  void
  initialize_chebyshev_smoother_coarse_grid(Operator & matrix);

  void
  initialize_chebyshev_smoother_nonsymmetric_operator(Operator & matrix, unsigned int level);

  void
  initialize_chebyshev_smoother_nonsymmetric_operator_coarse_grid(Operator & matrix);
};

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_ADAPTER_BASE_H_ \
        */
