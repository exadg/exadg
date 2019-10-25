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

// level definition
#include "levels_hybrid_multigrid.h"

// smoother
#include "smoothers/cg_smoother.h"
#include "smoothers/chebyshev_smoother.h"
#include "smoothers/gmres_smoother.h"
#include "smoothers/jacobi_smoother.h"
#include "smoothers/smoother_base.h"

// transfer
#include "transfer/mg_transfer_mf_c.h"
#include "transfer/mg_transfer_mf_h.h"
#include "transfer/mg_transfer_mf_mg_level_object.h"
#include "transfer/mg_transfer_mf_p.h"

// coarse grid solvers
#include "coarse_grid_solvers.h"

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
  typedef std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    PeriodicFacePairs;

  typedef LinearAlgebra::distributed::Vector<Number>          VectorType;
  typedef LinearAlgebra::distributed::Vector<MultigridNumber> VectorTypeMG;

public:
  virtual ~MultigridPreconditionerBase()
  {
  }

  void
  initialize(MultigridData const &                data,
             parallel::TriangulationBase<dim> const * tria,
             FiniteElement<dim> const &           fe,
             Mapping<dim> const &                 mapping,
             bool const                           operator_is_singular = false,
             Map const *                          dirichlet_bc         = nullptr,
             PeriodicFacePairs *                  periodic_face_pairs  = nullptr);

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

  /*
   * Update of multigrid preconditioner including operators, smoothers, etc. (e.g. for problems
   * with time-dependent coefficients).
   */
  virtual void
  update();

protected:
  /*
   * Update functions that have to be called/implemented by derived classes.
   */
  virtual void
  update_smoother(unsigned int level);

  virtual void
  update_coarse_solver(bool const operator_is_singular);

  /*
   * Dof-handlers and constraints.
   */
  virtual void
  initialize_dof_handler_and_constraints(bool                                 is_singular,
                                         PeriodicFacePairs *                  periodic_face_pairs,
                                         FiniteElement<dim> const &           fe,
                                         parallel::TriangulationBase<dim> const * tria,
                                         Map const *                          dirichlet_bc);

  void
  do_initialize_dof_handler_and_constraints(
    bool                                                        is_singular,
    PeriodicFacePairs &                                         periodic_face_pairs,
    FiniteElement<dim> const &                                  fe,
    parallel::TriangulationBase<dim> const *                        tria,
    Map const &                                                 dirichlet_bc,
    std::vector<MGLevelInfo> &                                  level_info,
    std::vector<MGDoFHandlerIdentifier> &                       p_levels,
    MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> &     dofhandlers,
    MGLevelObject<std::shared_ptr<MGConstrainedDoFs>> &         constrained_dofs,
    MGLevelObject<std::shared_ptr<AffineConstraints<double>>> & constraints);

  /*
   * Transfer operators.
   */
  virtual void
  initialize_transfer_operators();

  MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>            dof_handlers;
  MGLevelObject<std::shared_ptr<MGConstrainedDoFs>>                constrained_dofs;
  MGLevelObject<std::shared_ptr<AffineConstraints<double>>>        constraints;
  MGLevelObject<std::shared_ptr<MatrixFree<dim, MultigridNumber>>> matrix_free_objects;
  MGLevelObject<std::shared_ptr<Operator>>                         operators;
  MGTransferMF_MGLevelObject<dim, VectorTypeMG>                    transfers;

  std::vector<MGDoFHandlerIdentifier> p_levels;
  std::vector<MGLevelInfo>            level_info;
  unsigned int                        n_levels;
  unsigned int                        coarse_level;
  unsigned int                        fine_level;

private:
  bool
  mg_transfer_to_continuous_elements() const;

  /*
   * Multigrid levels (i.e. coarsening strategy, h-/p-/hp-/ph-MG).
   */
  void
  initialize_levels(parallel::TriangulationBase<dim> const * tria,
                    unsigned int const                   degree,
                    bool const                           is_dg);

  void
  check_levels(std::vector<MGLevelInfo> const & level_info);


  /*
   * Constrained dofs.
   */
  virtual void
  initialize_constrained_dofs(DoFHandler<dim> const &,
                              MGConstrainedDoFs &,
                              Map const & dirichlet_bc);

  /*
   * Data structures needed for matrix-free operator evaluation.
   */
  void
  initialize_matrix_free(Mapping<dim> const & mapping);

  virtual std::shared_ptr<MatrixFree<dim, MultigridNumber>>
  initialize_matrix_free(unsigned int const level, Mapping<dim> const & mapping);

  /*
   * Initializes the multigrid operators for all multigrid levels.
   */
  void
  initialize_operators();

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

  /*
   * Coarse grid solver.
   */
  void
  initialize_coarse_solver(bool const operator_is_singular);

  void
  initialize_chebyshev_smoother_coarse_grid(Operator &         matrix,
                                            SolverData const & solver_data,
                                            bool const         operator_is_singular);

  /*
   * Initialization of actual multigrid algorithm.
   */
  virtual void
  initialize_multigrid_preconditioner();

  MultigridData data;

  typedef SmootherBase<VectorTypeMG>       SMOOTHER;
  MGLevelObject<std::shared_ptr<SMOOTHER>> smoothers;

  std::shared_ptr<MGCoarseGridBase<VectorTypeMG>> coarse_grid_solver;

  std::shared_ptr<MultigridPreconditioner<VectorTypeMG, Operator, SMOOTHER>>
    multigrid_preconditioner;
};

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_ADAPTER_BASE_H_ \
        */
