/*
 * multigrid_preconditioner_base.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_ADAPTER_BASE_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_ADAPTER_BASE_H_

#include <deal.II/fe/fe_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/matrix_free/fe_evaluation.h>

// multigrid algorithm
#include "../../functionalities/set_zero_mean_value.h"
#include "../smoother/cg_smoother.h"
#include "../smoother/chebyshev_smoother.h"
#include "../smoother/gmres_smoother.h"
#include "../smoother/jacobi_smoother.h"
#include "../smoother/smoother_base.h"

// transfer
#include "../transfer/mg_transfer_mf_h.h"
#include "../transfer/mg_transfer_mf_p.h"

// coarse grid solvers
#include "../mg_coarse/mg_coarse_grid_solvers.h"
#include "multigrid_algorithm.h"

// parameters
#include "multigrid_input_parameters.h"

#include "../transfer/mg_transfer_mf_mg_level_object.h"

template<int dim, typename Number, typename MultigridNumber>
class MultigridPreconditionerBase : public PreconditionerBase<Number>
{
private:
  typedef MultigridOperatorBase<dim, MultigridNumber> Operator;

  typedef std::vector<std::pair<unsigned int, unsigned int>>           Levels;
  typedef std::map<types::boundary_id, std::shared_ptr<Function<dim>>> Map;

public:
  typedef LinearAlgebra::distributed::Vector<Number>          VectorType;
  typedef LinearAlgebra::distributed::Vector<MultigridNumber> VectorTypeMG;

  MultigridPreconditionerBase(std::shared_ptr<Operator> underlying_operator);

  virtual ~MultigridPreconditionerBase();

  /*
   * Initialization function for both discontinuous and continuous Galerkin methods (and for DG with
   * continuous Galerkin discretizations used as auxiliary space).
   */
  void
  initialize(MultigridData const &   mg_data,
             DoFHandler<dim> const & dof_handler,
             Mapping<dim> const &    mapping,
             void *                  operator_data,
             Map const *             dirichlet_bc = nullptr,
             std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> *
                                     periodic_face_pairs = nullptr,
             DoFHandler<dim> const * add_dof_handler     = nullptr);

  /*
   * Update of multigrid preconditioner including mg_matrices, smoothers, etc. (e.g. for problems
   * time-dependent coefficients).
   */
  virtual void
  update(LinearOperatorBase const * /*linear_operator*/);

  /*
   * This function applies the multigrid preconditioner dst = P^{-1} src.
   */
  void
  vmult(VectorType & dst, VectorType const & src) const;

  virtual void
  apply_smoother_on_fine_level(VectorTypeMG & dst, VectorTypeMG const & src) const;

protected:
  virtual void
  update_smoother(unsigned int level);

  virtual void
  update_coarse_solver();

  unsigned int n_global_levels;

  MGLevelObject<std::shared_ptr<Operator>> mg_matrices;

private:
  /*
   * Multigrid sequence (i.e. coarsening strategy).
   */
  void
  initialize_mg_sequence(parallel::Triangulation<dim> const *  tria,
                         std::vector<MGLevelIdentifier> &      global_levels,
                         std::vector<unsigned int> &           h_levels,
                         std::vector<MGDofHandlerIdentifier> & p_levels,
                         unsigned int const                    degree,
                         MultigridType const                   mg_type,
                         const bool                            is_dg);

  void
  check_mg_sequence(std::vector<MGLevelIdentifier> const & global_levels);

  static unsigned int
  get_next_coarser_degree(unsigned int const degree)
  {
    // examples:
    // 9 -> 4; 8 -> 4; 7 -> 3; 6 -> 3; 5 -> 2; 4 -> 2; 3 -> 1; 2 -> 1
    if(degree == 1)
      return 1;

    return degree / 2;
  }

  /*
   * Dof-handlers and constraints.
   */
  void
  initialize_mg_dof_handler_and_constraints(
    DoFHandler<dim> const &                                              dof_handler,
    parallel::Triangulation<dim> const *                                 tria,
    std::vector<MGLevelIdentifier> &                                     global_levels,
    std::vector<MGDofHandlerIdentifier> &                                p_levels,
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const & dirichlet_bc);

  virtual void
  initialize_mg_constrained_dofs(DoFHandler<dim> const &,
                                 MGConstrainedDoFs &,
                                 Map const & dirichlet_bc);

  /*
   * Multigrid operators on each multigrid level.
   */
  void
  initialize_mg_matrices(
    std::vector<MGLevelIdentifier> & global_levels,
    Mapping<dim> const &             mapping,
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                            periodic_face_pairs,
    void *                  operator_data,
    DoFHandler<dim> const * add_dof_handler = nullptr);

  /*
   * Multigrid transfer operators.
   */
  static void
  initialize_mg_transfer(
    const int                                                      n_components,
    const int                                                      rank,
    std::vector<MGLevelIdentifier> &                               global_levels,
    std::vector<MGDofHandlerIdentifier> &                          p_levels,
    MGLevelObject<std::shared_ptr<Operator>> &                     mg_matrices,
    MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> &        mg_dofhandler,
    MGLevelObject<std::shared_ptr<MGConstrainedDoFs>> &            mg_constrained_dofs,
    MGLevelObject<std::shared_ptr<MGTransferBase<VectorTypeMG>>> & mg_transfer);

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
  initialize_coarse_solver(unsigned int const coarse_level);

  void
  initialize_chebyshev_smoother_coarse_grid(Operator & matrix);

  void
  initialize_chebyshev_smoother_nonsymmetric_operator_coarse_grid(Operator & matrix);

  virtual void
  initialize_multigrid_preconditioner();

  MultigridData mg_data;

  MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dofhandler;
  MGLevelObject<std::shared_ptr<MGConstrainedDoFs>>     mg_constrained_dofs;

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
};

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_ADAPTER_BASE_H_ \
        */
