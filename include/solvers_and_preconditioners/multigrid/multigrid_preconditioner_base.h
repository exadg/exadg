/*
 * multigrid_preconditioner_base.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_ADAPTER_BASE_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_ADAPTER_BASE_H_

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/matrix_free/fe_evaluation.h>

// multigrid algorithm
#include "multigrid_algorithm.h"

// smoother
#include "../smoother/cg_smoother.h"
#include "../smoother/chebyshev_smoother.h"
#include "../smoother/gmres_smoother.h"
#include "../smoother/jacobi_smoother.h"
#include "../smoother/smoother_base.h"

// transfer
#include "../transfer/mg_transfer_mf_c.h"
#include "../transfer/mg_transfer_mf_h.h"
#include "../transfer/mg_transfer_mf_mg_level_object.h"
#include "../transfer/mg_transfer_mf_p.h"

// coarse grid solvers
#include "../mg_coarse/mg_coarse_grid_solvers.h"

// parameters
#include "multigrid_input_parameters.h"

template<int dim, typename Number, typename MultigridNumber>
class MultigridPreconditionerBase : public PreconditionerBase<Number>
{
private:
  typedef MultigridOperatorBase<dim, MultigridNumber> Operator;

  typedef std::vector<std::pair<unsigned int, unsigned int>> Levels;

protected:
  typedef std::map<types::boundary_id, std::shared_ptr<Function<dim>>> Map;

public:
  typedef LinearAlgebra::distributed::Vector<Number>          VectorType;
  typedef LinearAlgebra::distributed::Vector<MultigridNumber> VectorTypeMG;

  virtual ~MultigridPreconditionerBase()
  {
  }

  void
  initialize(MultigridData const &                mg_data,
             parallel::Triangulation<dim> const * tria,
             FiniteElement<dim> const &           fe,
             Mapping<dim> const &                 mapping,
             bool const                           operator_is_singular = false,
             Map const *                          dirichlet_bc         = nullptr,
             std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> *
               periodic_face_pairs = nullptr);

  /*
   * Update of multigrid preconditioner including mg_matrices, smoothers, etc. (e.g. for problems
   * with time-dependent coefficients).
   */
  virtual void
  update(LinearOperatorBase const * /*linear_operator*/);

  /*
   * This function applies the multigrid preconditioner dst = P^{-1} src.
   */
  void
  vmult(VectorType & dst, VectorType const & src) const;

  /*
   * Use multigrid as a solver.
   */
  unsigned int
  solve(VectorType & dst, VectorType const & src) const;

  virtual void
  apply_smoother_on_fine_level(VectorTypeMG & dst, VectorTypeMG const & src) const;

protected:
  virtual void
  update_smoother(unsigned int level);

  virtual void
  update_coarse_solver(bool const operator_is_singular);

  unsigned int n_global_levels;
  unsigned int min_level;
  unsigned int max_level;

private:
  /*
   * Multigrid sequence (i.e. coarsening strategy).
   */
  void
  initialize_mg_sequence(parallel::Triangulation<dim> const * tria,
                         unsigned int const                   degree,
                         bool const                           is_dg);

  void
  check_mg_sequence(std::vector<MGLevelInfo> const & global_levels);

protected:
  /*
   * Dof-handlers and constraints.
   */
  virtual void
  initialize_dof_handler_and_constraints(
    bool is_singular,
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                                                                         periodic_face_pairs,
    FiniteElement<dim> const &                                           fe,
    parallel::Triangulation<dim> const *                                 tria,
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const & dirichlet_bc);

  void
  initialize_mg_dof_handler_and_constraints(
    bool is_singular,
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                                                                         periodic_face_pairs,
    FiniteElement<dim> const &                                           fe,
    parallel::Triangulation<dim> const *                                 tria,
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const & dirichlet_bc,
    std::vector<MGLevelInfo> &                                           global_levels,
    std::vector<MGDofHandlerIdentifier> &                                p_levels,
    MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> &              mg_dofhandler,
    MGLevelObject<std::shared_ptr<MGConstrainedDoFs>> &                  mg_constrained_dofs,
    MGLevelObject<std::shared_ptr<AffineConstraints<double>>> &          mg_constraints);

  /*
   * Transfer operators.
   */
  virtual void
  initialize_mg_transfer();

private:
  virtual void
  initialize_mg_constrained_dofs(DoFHandler<dim> const &,
                                 MGConstrainedDoFs &,
                                 Map const & dirichlet_bc);

  void
  initialize_matrixfree(Mapping<dim> const & mapping);

  virtual std::shared_ptr<MatrixFree<dim, MultigridNumber>>
  initialize_matrix_free(unsigned int const level, Mapping<dim> const & mapping);

  /*
   * Initializes the multigrid operators for all multigrid levels.
   */
  void
  initialize_mg_matrices();

  /*
   * This function initializes an operator for a specified level. It needs to be implemented by
   * derived classes.
   */
  virtual std::shared_ptr<Operator>
  initialize_operator(unsigned int const level);

  /*
   * Smoother.
   */
  void
  initialize_smoothers();

  void
  initialize_smoother(Operator & matrix, unsigned int level);

  void
  initialize_chebyshev_smoother(Operator & matrix, unsigned int level);

  void
  initialize_chebyshev_smoother_nonsymmetric_operator(Operator & matrix, unsigned int level);

  /*
   * Coarse grid solver.
   */
  void
  initialize_coarse_solver(bool const operator_is_singular);

  void
  initialize_chebyshev_smoother_coarse_grid(Operator &         matrix,
                                            SolverData const & solver_data,
                                            bool const         operator_is_singular);

  void
  initialize_chebyshev_smoother_nonsymmetric_operator_coarse_grid(Operator &         matrix,
                                                                  SolverData const & solver_data,
                                                                  bool const operator_is_singular);

  virtual void
  initialize_multigrid_preconditioner();

  MultigridData mg_data;

protected:
  MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>            mg_dofhandler;
  MGLevelObject<std::shared_ptr<MGConstrainedDoFs>>                mg_constrained_dofs;
  MGLevelObject<std::shared_ptr<AffineConstraints<double>>>        mg_constraints;
  MGLevelObject<std::shared_ptr<MatrixFree<dim, MultigridNumber>>> mg_matrixfree;
  MGLevelObject<std::shared_ptr<Operator>>                         mg_matrices;

  std::vector<unsigned int>           h_levels;
  std::vector<MGDofHandlerIdentifier> p_levels;
  std::vector<MGLevelInfo>            global_levels;

  MGTransferMF_MGLevelObject<dim, VectorTypeMG> mg_transfer;

private:
  typedef SmootherBase<VectorTypeMG>       SMOOTHER;
  MGLevelObject<std::shared_ptr<SMOOTHER>> mg_smoother;

  std::shared_ptr<MGCoarseGridBase<VectorTypeMG>> mg_coarse;

  std::shared_ptr<MultigridPreconditioner<VectorTypeMG, Operator, SMOOTHER>>
    multigrid_preconditioner;
};

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_ADAPTER_BASE_H_ \
        */
