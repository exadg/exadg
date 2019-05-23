#include "multigrid_preconditioner_base.h"

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include "../preconditioner/preconditioner_amg.h"

#include "../../functionalities/categorization.h"
#include "../../functionalities/constraints.h"
#include "../../operators/operator_base.h"
#include "../util/compute_eigenvalues.h"

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize(
  MultigridData const &                mg_data,
  parallel::Triangulation<dim> const * tria,
  FiniteElement<dim> const &           fe,
  Mapping<dim> const &                 mapping,
  bool const                           operator_is_singular,
  Map const *                          dirichlet_bc_in,
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> *
    periodic_face_pairs_in)
{
  this->mg_data = mg_data;

  bool const is_dg = fe.dofs_per_vertex == 0;
  if((!is_dg || (is_dg && (mg_data.dg_to_cg_transfer != DG_To_CG_Transfer::None)) ||
      mg_data.coarse_problem.solver == MultigridCoarseGridSolver::AMG) &&
     ((dirichlet_bc_in == nullptr) || (periodic_face_pairs_in == nullptr)))
  {
    AssertThrow(
      mg_data.coarse_problem.solver != MultigridCoarseGridSolver::AMG,
      ExcMessage(
        "You have to provide Dirichlet BCs and periodic face pairs if you want to use CG or AMG!"));
  }

  // In the case of nullptr, these data structures simply remain empty.
  Map dirichlet_bc;
  if(dirichlet_bc_in != nullptr)
    dirichlet_bc = *dirichlet_bc_in;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_face_pairs;
  if(dirichlet_bc_in != nullptr)
    periodic_face_pairs = *periodic_face_pairs_in;

  this->initialize_mg_sequence(tria, fe.degree, is_dg);

  this->initialize_dof_handler_and_constraints(
    operator_is_singular, periodic_face_pairs, fe, tria, dirichlet_bc);

  this->initialize_matrixfree(mapping);

  this->initialize_mg_matrices();

  this->initialize_smoothers();

  this->initialize_coarse_solver(operator_is_singular);

  this->initialize_mg_transfer();

  this->initialize_multigrid_preconditioner();
}

/*
 *
 * example: h_levels = [0 1 2], p_levels = [1 3 7]
 *
 * p-MG:
 * global_levels  h_levels  p_levels
 * 2              2         7
 * 1              2         3
 * 0              2         1
 *
 * ph-MG:
 * global_levels  h_levels  p_levels
 * 4              2         7
 * 3              2         3
 * 2              2         1
 * 1              1         1
 * 0              0         1
 *
 * h-MG:
 * global_levels  h_levels  p_levels
 * 2              2         7
 * 1              1         7
 * 0              0         7
 *
 * hp-MG:
 * global_levels  h_levels  p_levels
 * 4              2         7
 * 3              1         7
 * 2              0         7
 * 1              0         3
 * 0              0         1
 *
 */

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_mg_sequence(
  parallel::Triangulation<dim> const * tria,
  unsigned int const                   degree,
  bool const                           is_dg)
{
  MultigridType const mg_type = this->mg_data.type;

  // setup h-levels
  if(mg_type == MultigridType::pMG) // p-MG is only working on the finest h-level
  {
    h_levels.push_back(tria->n_global_levels() - 1);
  }
  else // h-MG, hp-MG, and ph-MG are working on all h-levels
  {
    for(unsigned int i = 0; i < tria->n_global_levels(); i++)
      h_levels.push_back(i);
  }

  // setup p-levels
  if(mg_type == MultigridType::hMG) // h-MG is only working on high-order
  {
    p_levels.push_back({degree, is_dg});
  }
  else // p-MG, hp-MG, and ph-MG are working on high- and low- order elements
  {
    unsigned int temp = degree;
    do
    {
      p_levels.push_back({temp, is_dg});
      switch(this->mg_data.p_sequence)
      {
          // clang-format off
        case PSequenceType::GoToOne:       temp = 1;                                                break;
        case PSequenceType::DecreaseByOne: temp = std::max(temp-1, 1u);                             break;
        case PSequenceType::Bisect:        temp = std::max(temp/2, 1u);                             break;
        case PSequenceType::Manual:        temp = (degree==3&&temp==3) ? 2 : std::max(degree/2, 1u);break;
        default:
          AssertThrow(false, ExcMessage("No valid p-sequence selected!"));
          // clang-format on
      }
    } while(temp != p_levels.back().degree);
    std::reverse(std::begin(p_levels), std::end(p_levels));
  }

  if(mg_data.dg_to_cg_transfer == DG_To_CG_Transfer::Coarse && is_dg)
    p_levels.insert(p_levels.begin(), {p_levels.front().degree, false});

  if(mg_data.dg_to_cg_transfer == DG_To_CG_Transfer::Fine && is_dg)
  {
    for(auto & i : p_levels)
      i.is_dg = false;
    p_levels.push_back({p_levels.back().degree, true});
  }

  // setup global-levels
  if(mg_type == MultigridType::pMG || mg_type == MultigridType::phMG)
  {
    // top level: p-MG
    if(mg_type == MultigridType::phMG) // low level: h-MG
      for(unsigned int i = 0; i < h_levels.size() - 1; i++)
        global_levels.push_back({h_levels[i], p_levels.front()});
    for(auto deg : p_levels)
      global_levels.push_back({h_levels.back(), deg});
  }
  else if(mg_type == MultigridType::hMG || mg_type == MultigridType::hpMG)
  {
    // top level: h-MG
    if(mg_type == MultigridType::hpMG) // low level: p-MG
      for(unsigned int i = 0; i < p_levels.size() - 1; i++)
        global_levels.push_back({h_levels.front(), p_levels[i]});
    for(auto geo : h_levels)
      global_levels.push_back({geo, p_levels.back()});
  }
  else
    AssertThrow(false, ExcMessage("This multigrid type does not exist!"));

  this->n_global_levels = global_levels.size(); // number of actual multigrid levels
  this->min_level       = 0;
  this->max_level       = this->n_global_levels - 1;

  this->check_mg_sequence(global_levels);
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::check_mg_sequence(
  std::vector<MGLevelInfo> const & global_levels)
{
  AssertThrow(this->n_global_levels == global_levels.size(),
              ExcMessage("Variable n_global_levels is not initialized correctly."));
  AssertThrow(this->min_level == 0, ExcMessage("Variable min_level is not initialized correctly."));
  AssertThrow(this->max_level == this->n_global_levels - 1,
              ExcMessage("Variable max_level is not initialized correctly."));

  for(unsigned int i = 1; i < global_levels.size(); i++)
  {
    auto fine_level   = global_levels[i];
    auto coarse_level = global_levels[i - 1];

    AssertThrow((fine_level.level != coarse_level.level) ^
                  (fine_level.degree != coarse_level.degree) ^
                  (fine_level.is_dg != coarse_level.is_dg),
                ExcMessage(
                  "Between levels there is only ONE change allowed: either in h- or p-level!"));
  }
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_dof_handler_and_constraints(
  bool const operator_is_singular,
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                                                                       periodic_face_pairs,
  FiniteElement<dim> const &                                           fe,
  parallel::Triangulation<dim> const *                                 tria,
  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const & dirichlet_bc)
{
  this->initialize_mg_dof_handler_and_constraints(operator_is_singular,
                                                  periodic_face_pairs,
                                                  fe,
                                                  tria,
                                                  dirichlet_bc,
                                                  this->global_levels,
                                                  this->p_levels,
                                                  this->mg_dofhandler,
                                                  this->mg_constrained_dofs,
                                                  this->mg_constraints);
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::
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
    MGLevelObject<std::shared_ptr<AffineConstraints<double>>> &          mg_constraints)
{
  mg_constrained_dofs.resize(0, this->n_global_levels - 1);
  mg_dofhandler.resize(0, this->n_global_levels - 1);
  mg_constraints.resize(0, this->n_global_levels - 1);

  const unsigned int n_components = fe.n_components();

  // temporal storage for new dofhandlers and constraints on each p-level
  std::map<MGDofHandlerIdentifier, std::shared_ptr<const DoFHandler<dim>>> map_dofhandlers;
  std::map<MGDofHandlerIdentifier, std::shared_ptr<MGConstrainedDoFs>>     map_constraints;

  // setup dof-handler and constrained dofs for each p-level
  for(auto degree : p_levels)
  {
    // setup dof_handler: create dof_handler...
    auto dof_handler = new DoFHandler<dim>(*tria);
    // ... create FE and distribute it
    if(degree.is_dg)
      dof_handler->distribute_dofs(FESystem<dim>(FE_DGQ<dim>(degree.degree), n_components));
    else
      dof_handler->distribute_dofs(FESystem<dim>(FE_Q<dim>(degree.degree), n_components));
    dof_handler->distribute_mg_dofs();
    // setup constrained dofs:
    auto constrained_dofs = new MGConstrainedDoFs();
    constrained_dofs->clear();
    this->initialize_mg_constrained_dofs(*dof_handler, *constrained_dofs, dirichlet_bc);

    // put in temporal storage
    map_dofhandlers[degree] = std::shared_ptr<DoFHandler<dim> const>(dof_handler);
    map_constraints[degree] = std::shared_ptr<MGConstrainedDoFs>(constrained_dofs);
  }

  // populate dof-handler and constrained dofs to all hp-levels with the same degree
  for(unsigned int i = 0; i < global_levels.size(); i++)
  {
    auto degree            = global_levels[i].id;
    mg_dofhandler[i]       = map_dofhandlers[degree];
    mg_constrained_dofs[i] = map_constraints[degree];
  }

  for(unsigned int i = 0; i < global_levels.size(); i++)
  {
    auto constraint_own = new AffineConstraints<double>;

    ConstraintUtil::add_constraints<dim>(global_levels[i].is_dg,
                                         is_singular,
                                         *mg_dofhandler[i],
                                         *constraint_own,
                                         *mg_constrained_dofs[i],
                                         periodic_face_pairs,
                                         global_levels[i].level);

    mg_constraints[i].reset(constraint_own);
  }
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_matrixfree(
  Mapping<dim> const & mapping)
{
  this->mg_matrixfree.resize(0, this->n_global_levels - 1);

  for(unsigned int i = 0; i < this->n_global_levels; i++)
    this->mg_matrixfree[i] = this->initialize_matrix_free(i, mapping);
}


template<int dim, typename Number, typename MultigridNumber>
std::shared_ptr<MatrixFree<dim, MultigridNumber>>
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_matrix_free(
  unsigned int const   level,
  Mapping<dim> const & mapping)
{
  (void)level;
  (void)mapping;

  AssertThrow(false, ExcMessage("This function needs to be implemented by derived classes."));

  std::shared_ptr<MatrixFree<dim, MultigridNumber>> matrix_free;

  return matrix_free;
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_mg_matrices()
{
  this->mg_matrices.resize(0, this->n_global_levels - 1);

  // create and setup operator on each level
  for(unsigned int i = 0; i < this->n_global_levels; i++)
    mg_matrices[i] = this->initialize_operator(i);
}

template<int dim, typename Number, typename MultigridNumber>
std::shared_ptr<MultigridOperatorBase<dim, MultigridNumber>>
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_operator(
  unsigned int const level)
{
  (void)level;

  AssertThrow(false, ExcMessage("This function needs to be implemented by derived classes."));

  std::shared_ptr<Operator> op;

  return op;
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_smoothers()
{
  this->mg_smoother.resize(0, this->n_global_levels - 1);

  for(unsigned int i = 1; i < this->n_global_levels; i++)
    this->initialize_smoother(*this->mg_matrices[i], i);
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_mg_constrained_dofs(
  DoFHandler<dim> const & dof_handler,
  MGConstrainedDoFs &     constrained_dofs,
  Map const &             dirichlet_bc)
{
  std::set<types::boundary_id> dirichlet_boundary;
  for(auto & it : dirichlet_bc)
    dirichlet_boundary.insert(it.first);
  constrained_dofs.initialize(dof_handler);
  constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary);
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::update(
  LinearOperatorBase const * /*linear_operator*/)
{
  // do nothing in base class (has to be implemented by derived classes if necessary)
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::vmult(VectorType &       dst,
                                                                 VectorType const & src) const
{
  multigrid_preconditioner->vmult(dst, src);
}

template<int dim, typename Number, typename MultigridNumber>
unsigned int
MultigridPreconditionerBase<dim, Number, MultigridNumber>::solve(VectorType &       dst,
                                                                 VectorType const & src) const
{
  return multigrid_preconditioner->solve(dst, src);
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::apply_smoother_on_fine_level(
  VectorTypeMG &       dst,
  VectorTypeMG const & src) const
{
  this->mg_smoother[this->mg_smoother.max_level()]->vmult(dst, src);
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_smoother(Operator &   matrix,
                                                                               unsigned int level)
{
  AssertThrow(level > 0,
              ExcMessage("Multigrid level is invalid when initializing multigrid smoother!"));

  switch(mg_data.smoother_data.smoother)
  {
    case MultigridSmoother::Chebyshev:
    {
      mg_smoother[level].reset(new ChebyshevSmoother<Operator, VectorTypeMG>());
      initialize_chebyshev_smoother(matrix, level);
      break;
    }
    case MultigridSmoother::ChebyshevNonsymmetricOperator:
    {
      mg_smoother[level].reset(new ChebyshevSmoother<Operator, VectorTypeMG>());
      initialize_chebyshev_smoother_nonsymmetric_operator(matrix, level);
      break;
    }
    case MultigridSmoother::GMRES:
    {
      typedef GMRESSmoother<Operator, VectorTypeMG> GMRES_SMOOTHER;
      mg_smoother[level].reset(new GMRES_SMOOTHER());

      typename GMRES_SMOOTHER::AdditionalData smoother_data;
      smoother_data.preconditioner       = mg_data.smoother_data.preconditioner;
      smoother_data.number_of_iterations = mg_data.smoother_data.iterations;

      std::shared_ptr<GMRES_SMOOTHER> smoother =
        std::dynamic_pointer_cast<GMRES_SMOOTHER>(mg_smoother[level]);
      smoother->initialize(matrix, smoother_data);
      break;
    }
    case MultigridSmoother::CG:
    {
      typedef CGSmoother<Operator, VectorTypeMG> CG_SMOOTHER;
      mg_smoother[level].reset(new CG_SMOOTHER());

      typename CG_SMOOTHER::AdditionalData smoother_data;
      smoother_data.preconditioner       = mg_data.smoother_data.preconditioner;
      smoother_data.number_of_iterations = mg_data.smoother_data.iterations;

      std::shared_ptr<CG_SMOOTHER> smoother =
        std::dynamic_pointer_cast<CG_SMOOTHER>(mg_smoother[level]);
      smoother->initialize(matrix, smoother_data);
      break;
    }
    case MultigridSmoother::Jacobi:
    {
      typedef JacobiSmoother<Operator, VectorTypeMG> JACOBI_SMOOTHER;
      mg_smoother[level].reset(new JACOBI_SMOOTHER());

      typename JACOBI_SMOOTHER::AdditionalData smoother_data;
      smoother_data.preconditioner            = mg_data.smoother_data.preconditioner;
      smoother_data.number_of_smoothing_steps = mg_data.smoother_data.iterations;
      smoother_data.damping_factor            = mg_data.smoother_data.relaxation_factor;

      std::shared_ptr<JACOBI_SMOOTHER> smoother =
        std::dynamic_pointer_cast<JACOBI_SMOOTHER>(mg_smoother[level]);
      smoother->initialize(matrix, smoother_data);
      break;
    }
    default:
    {
      AssertThrow(false, ExcMessage("Specified MultigridSmoother not implemented!"));
    }
  }
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::update_smoother(unsigned int level)
{
  AssertThrow(level > 0,
              ExcMessage("Multigrid level is invalid when initializing multigrid smoother!"));

  switch(mg_data.smoother_data.smoother)
  {
    case MultigridSmoother::Chebyshev:
    {
      initialize_chebyshev_smoother(*mg_matrices[level], level);
      break;
    }
    case MultigridSmoother::ChebyshevNonsymmetricOperator:
    {
      initialize_chebyshev_smoother_nonsymmetric_operator(*mg_matrices[level], level);
      break;
    }
    case MultigridSmoother::GMRES:
    {
      typedef GMRESSmoother<Operator, VectorTypeMG> GMRES_SMOOTHER;

      std::shared_ptr<GMRES_SMOOTHER> smoother =
        std::dynamic_pointer_cast<GMRES_SMOOTHER>(mg_smoother[level]);
      smoother->update();
      break;
    }
    case MultigridSmoother::CG:
    {
      typedef CGSmoother<Operator, VectorTypeMG> CG_SMOOTHER;

      std::shared_ptr<CG_SMOOTHER> smoother =
        std::dynamic_pointer_cast<CG_SMOOTHER>(mg_smoother[level]);
      smoother->update();
      break;
    }
    case MultigridSmoother::Jacobi:
    {
      typedef JacobiSmoother<Operator, VectorTypeMG> JACOBI_SMOOTHER;

      std::shared_ptr<JACOBI_SMOOTHER> smoother =
        std::dynamic_pointer_cast<JACOBI_SMOOTHER>(mg_smoother[level]);
      smoother->update();
      break;
    }
    default:
    {
      AssertThrow(false, ExcMessage("Specified MultigridSmoother not implemented!"));
    }
  }
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::update_coarse_solver(
  bool const operator_is_singular)
{
  switch(mg_data.coarse_problem.solver)
  {
    case MultigridCoarseGridSolver::Chebyshev:
    {
      AssertThrow(
        mg_data.coarse_problem.preconditioner == MultigridCoarseGridPreconditioner::PointJacobi,
        ExcMessage(
          "Only PointJacobi preconditioner implemented for Chebyshev coarse grid solver."));

      initialize_chebyshev_smoother_coarse_grid(*mg_matrices[0],
                                                mg_data.coarse_problem.solver_data,
                                                operator_is_singular);
      break;
    }
    case MultigridCoarseGridSolver::ChebyshevNonsymmetricOperator:
    {
      AssertThrow(
        mg_data.coarse_problem.preconditioner == MultigridCoarseGridPreconditioner::PointJacobi,
        ExcMessage(
          "Only PointJacobi preconditioner implemented for Chebyshev coarse grid solver."));

      initialize_chebyshev_smoother_nonsymmetric_operator_coarse_grid(
        *mg_matrices[0], mg_data.coarse_problem.solver_data, operator_is_singular);
      break;
    }
    case MultigridCoarseGridSolver::CG:
    case MultigridCoarseGridSolver::GMRES:
    {
      if(mg_data.coarse_problem.preconditioner != MultigridCoarseGridPreconditioner::None)
      {
        std::shared_ptr<MGCoarseKrylov<Operator>> coarse_solver =
          std::dynamic_pointer_cast<MGCoarseKrylov<Operator>>(mg_coarse);
        coarse_solver->update(*this->mg_matrices[0]);
      }

      break;
    }
    case MultigridCoarseGridSolver::AMG:
    {
      std::shared_ptr<MGCoarseAMG<Operator>> coarse_solver =
        std::dynamic_pointer_cast<MGCoarseAMG<Operator>>(mg_coarse);
      coarse_solver->update(&(*this->mg_matrices[0]));

      break;
    }
    default:
    {
      AssertThrow(false, ExcMessage("Unknown coarse-grid solver given"));
    }
  }
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_coarse_solver(
  bool const operator_is_singular)
{
  Operator & matrix = *mg_matrices[0];

  switch(mg_data.coarse_problem.solver)
  {
    case MultigridCoarseGridSolver::Chebyshev:
    {
      AssertThrow(
        mg_data.coarse_problem.preconditioner == MultigridCoarseGridPreconditioner::PointJacobi,
        ExcMessage(
          "Only PointJacobi preconditioner implemented for Chebyshev coarse grid solver."));

      mg_smoother[0].reset(new ChebyshevSmoother<Operator, VectorTypeMG>());
      initialize_chebyshev_smoother_coarse_grid(matrix,
                                                mg_data.coarse_problem.solver_data,
                                                operator_is_singular);

      mg_coarse.reset(new MGCoarseChebyshev<VectorTypeMG, SMOOTHER>(mg_smoother[0]));
      break;
    }
    case MultigridCoarseGridSolver::ChebyshevNonsymmetricOperator:
    {
      AssertThrow(
        mg_data.coarse_problem.preconditioner == MultigridCoarseGridPreconditioner::PointJacobi,
        ExcMessage(
          "Only PointJacobi preconditioner implemented for Chebyshev coarse grid solver."));

      mg_smoother[0].reset(new ChebyshevSmoother<Operator, VectorTypeMG>());
      initialize_chebyshev_smoother_nonsymmetric_operator_coarse_grid(
        matrix, mg_data.coarse_problem.solver_data, operator_is_singular);

      mg_coarse.reset(new MGCoarseChebyshev<VectorTypeMG, SMOOTHER>(mg_smoother[0]));
      break;
    }
    case MultigridCoarseGridSolver::CG:
    case MultigridCoarseGridSolver::GMRES:
    {
      typename MGCoarseKrylov<Operator>::AdditionalData additional_data;

      if(mg_data.coarse_problem.solver == MultigridCoarseGridSolver::CG)
        additional_data.solver_type = KrylovSolverType::CG;
      else if(mg_data.coarse_problem.solver == MultigridCoarseGridSolver::GMRES)
        additional_data.solver_type = KrylovSolverType::GMRES;
      else
        AssertThrow(false, ExcMessage("Not implemented."));

      additional_data.solver_data          = mg_data.coarse_problem.solver_data;
      additional_data.operator_is_singular = operator_is_singular;
      additional_data.preconditioner       = mg_data.coarse_problem.preconditioner;
      additional_data.amg_data             = mg_data.coarse_problem.amg_data;

      mg_coarse.reset(new MGCoarseKrylov<Operator>(matrix, additional_data));
      break;
    }
    case MultigridCoarseGridSolver::AMG:
    {
      mg_coarse.reset(new MGCoarseAMG<Operator>(matrix, mg_data.coarse_problem.amg_data));
      return;
    }
    default:
    {
      AssertThrow(false, ExcMessage("Unknown coarse-grid solver specified."));
    }
  }
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_mg_transfer()
{
  this->mg_transfer.template reinit<MultigridNumber>(mg_matrixfree,
                                                     mg_constraints,
                                                     mg_constrained_dofs);
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_multigrid_preconditioner()
{
  this->multigrid_preconditioner.reset(
    new MultigridPreconditioner<VectorTypeMG, Operator, SMOOTHER>(
      this->mg_matrices, *this->mg_coarse, this->mg_transfer, this->mg_smoother));
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_chebyshev_smoother(
  Operator &   matrix,
  unsigned int level)
{
  typedef ChebyshevSmoother<Operator, VectorTypeMG> CHEBYSHEV_SMOOTHER;
  typename CHEBYSHEV_SMOOTHER::AdditionalData       smoother_data;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  matrix.initialize_dof_vector(smoother_data.matrix_diagonal_inverse);
  matrix.calculate_inverse_diagonal(smoother_data.matrix_diagonal_inverse);
#pragma GCC diagnostic pop

  /*
  std::pair<double,double> eigenvalues = compute_eigenvalues(mg_matrices[level],
  smoother_data.matrix_diagonal_inverse);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << "Eigenvalues on level l = " << level << std::endl;
    std::cout << std::scientific << std::setprecision(3)
              <<"Max EV = " << eigenvalues.second << " : Min EV = " <<
  eigenvalues.first << std::endl;
  }
  */

  smoother_data.smoothing_range     = mg_data.smoother_data.smoothing_range;
  smoother_data.degree              = mg_data.smoother_data.iterations;
  smoother_data.eig_cg_n_iterations = mg_data.smoother_data.iterations_eigenvalue_estimation;

  std::shared_ptr<CHEBYSHEV_SMOOTHER> smoother =
    std::dynamic_pointer_cast<CHEBYSHEV_SMOOTHER>(mg_smoother[level]);
  smoother->initialize(matrix, smoother_data);
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::
  initialize_chebyshev_smoother_coarse_grid(Operator &         matrix,
                                            SolverData const & solver_data,
                                            bool const         operator_is_singular)
{
  // use Chebyshev smoother of high degree to solve the coarse grid problem approximately
  typedef ChebyshevSmoother<Operator, VectorTypeMG> CHEBYSHEV_SMOOTHER;
  typename CHEBYSHEV_SMOOTHER::AdditionalData       smoother_data;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  matrix.initialize_dof_vector(smoother_data.matrix_diagonal_inverse);
  matrix.calculate_inverse_diagonal(smoother_data.matrix_diagonal_inverse);
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  std::pair<double, double> eigenvalues =
    compute_eigenvalues(matrix, smoother_data.matrix_diagonal_inverse, operator_is_singular);
#pragma GCC diagnostic pop

  double const factor = 1.1;

  smoother_data.max_eigenvalue  = factor * eigenvalues.second;
  smoother_data.smoothing_range = eigenvalues.second / eigenvalues.first * factor;

  double sigma = (1. - std::sqrt(1. / smoother_data.smoothing_range)) /
                 (1. + std::sqrt(1. / smoother_data.smoothing_range));

  // calculate/estimate the number of Chebyshev iterations needed to reach a specified relative
  // solver tolerance
  double const eps = solver_data.rel_tol;

  smoother_data.degree = std::log(1. / eps + std::sqrt(1. / eps / eps - 1.)) / std::log(1. / sigma);
  smoother_data.eig_cg_n_iterations = 0;

  std::shared_ptr<CHEBYSHEV_SMOOTHER> smoother =
    std::dynamic_pointer_cast<CHEBYSHEV_SMOOTHER>(mg_smoother[0]);
  smoother->initialize(matrix, smoother_data);
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::
  initialize_chebyshev_smoother_nonsymmetric_operator(Operator & matrix, unsigned int level)
{
  typedef ChebyshevSmoother<Operator, VectorTypeMG> CHEBYSHEV_SMOOTHER;
  typename CHEBYSHEV_SMOOTHER::AdditionalData       smoother_data;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  matrix.initialize_dof_vector(smoother_data.matrix_diagonal_inverse);
  matrix.calculate_inverse_diagonal(smoother_data.matrix_diagonal_inverse);
#pragma GCC diagnostic pop

  /*
  std::pair<double,double> eigenvalues =
  compute_eigenvalues_gmres(mg_matrices[level],
  smoother_data.matrix_diagonal_inverse);
  std::cout<<"Max EW = "<< eigenvalues.second <<" : Min EW =
  "<<eigenvalues.first<<std::endl;
  */

  // use gmres to calculate eigenvalues for nonsymmetric problem
  unsigned int const eig_n_iter = 20;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  std::pair<std::complex<double>, std::complex<double>> eigenvalues =
    compute_eigenvalues_gmres(matrix, smoother_data.matrix_diagonal_inverse, eig_n_iter);
#pragma GCC diagnostic pop

  double const factor = 1.1;

  smoother_data.max_eigenvalue      = factor * std::abs(eigenvalues.second);
  smoother_data.smoothing_range     = mg_data.smoother_data.smoothing_range;
  smoother_data.degree              = mg_data.smoother_data.iterations;
  smoother_data.eig_cg_n_iterations = 0;

  std::shared_ptr<CHEBYSHEV_SMOOTHER> smoother =
    std::dynamic_pointer_cast<CHEBYSHEV_SMOOTHER>(mg_smoother[level]);
  smoother->initialize(matrix, smoother_data);
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::
  initialize_chebyshev_smoother_nonsymmetric_operator_coarse_grid(Operator &         matrix,
                                                                  SolverData const & solver_data,
                                                                  bool const operator_is_singular)
{
  // use Chebyshev smoother of high degree to solve the coarse grid problem approximately
  typedef ChebyshevSmoother<Operator, VectorTypeMG> CHEBYSHEV_SMOOTHER;
  typename CHEBYSHEV_SMOOTHER::AdditionalData       smoother_data;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  matrix.initialize_dof_vector(smoother_data.matrix_diagonal_inverse);
  matrix.calculate_inverse_diagonal(smoother_data.matrix_diagonal_inverse);
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  std::pair<std::complex<double>, std::complex<double>> eigenvalues =
    compute_eigenvalues_gmres(matrix, smoother_data.matrix_diagonal_inverse, operator_is_singular);
#pragma GCC diagnostic pop

  double const factor = 1.1;

  smoother_data.max_eigenvalue = factor * std::abs(eigenvalues.second);
  smoother_data.smoothing_range =
    factor * std::abs(eigenvalues.second) / std::abs(eigenvalues.first);

  double sigma = (1. - std::sqrt(1. / smoother_data.smoothing_range)) /
                 (1. + std::sqrt(1. / smoother_data.smoothing_range));

  // calculate/estimate the number of Chebyshev iterations needed to reach a specified relative
  // solver tolerance
  double const eps = solver_data.rel_tol;

  smoother_data.degree = std::log(1. / eps + std::sqrt(1. / eps / eps - 1)) / std::log(1. / sigma);
  smoother_data.eig_cg_n_iterations = 0;

  std::shared_ptr<CHEBYSHEV_SMOOTHER> smoother =
    std::dynamic_pointer_cast<CHEBYSHEV_SMOOTHER>(mg_smoother[0]);
  smoother->initialize(matrix, smoother_data);
}

template class MultigridPreconditionerBase<2, float, float>;
template class MultigridPreconditionerBase<2, double, float>;
template class MultigridPreconditionerBase<2, double, double>;

template class MultigridPreconditionerBase<3, float, float>;
template class MultigridPreconditionerBase<3, double, float>;
template class MultigridPreconditionerBase<3, double, double>;
