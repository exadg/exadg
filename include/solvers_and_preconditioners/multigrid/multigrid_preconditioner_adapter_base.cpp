#include "multigrid_preconditioner_adapter_base.h"

#include <navierstokes/config.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include "../mg_coarse/mg_coarse_ml.h"

template<int dim, typename value_type, typename Operator>
MyMultigridPreconditionerBase<dim, value_type, Operator>::MyMultigridPreconditionerBase(
  std::shared_ptr<Operator> underlying_operator)
  : underlying_operator(underlying_operator)
{
}

template<int dim, typename value_type, typename Operator>
MyMultigridPreconditionerBase<dim, value_type, Operator>::~MyMultigridPreconditionerBase()
{
}

template<int dim, typename value_type, typename Operator>
void
MyMultigridPreconditionerBase<dim, value_type, Operator>::initialize(
  const MultigridData &                                                mg_data_in,
  const DoFHandler<dim> &                                              dof_handler,
  const Mapping<dim> &                                                 mapping,
  void *                                                               operator_data_in)
{
  AssertThrow(mg_data_in.coarse_solver != MultigridCoarseGridSolver::AMG_ML, 
              ExcMessage("You have to provide Dirichlet BCs if you want to use AMG!"));
    
  // create emty vector for Dirichlet BC so that we can use the more general
  //  method which is written for continuous and discontinuous Galerkin methods
  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> dirichlet_bc;
  this->initialize(mg_data_in, dof_handler, mapping, dirichlet_bc, operator_data_in);
}

template<int dim, typename value_type, typename Operator>
void
MyMultigridPreconditionerBase<dim, value_type, Operator>::initialize(
  const MultigridData &                                                mg_data_in,
  const DoFHandler<dim> &                                              dof_handler,
  const Mapping<dim> &                                                 mapping,
  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const & dirichlet_bc,
  void *                                                               operator_data_in)
{
  // save mg-setup
  this->mg_data = mg_data_in;

  // get triangulation
  const parallel::Triangulation<dim> * tria =
    dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation());

  // extract paramters
  const auto   mg_type= this->mg_data.type;
  unsigned int degree = dof_handler.get_fe().degree;

  // setup sequence
  std::vector<unsigned int> h_levels, p_levels;
  std::vector<std::pair<unsigned int, unsigned int>> global_levels;
  this->initialize_mg_sequence(tria, global_levels, h_levels, p_levels, degree, mg_type);
  this->n_global_levels = global_levels.size(); // number of actual multigrid levels
  
  // setup-components
  this->initialize_mg_dof_handler_and_constraints(dof_handler, tria, global_levels, p_levels, dirichlet_bc,degree);
  this->initialize_mg_matrices(global_levels, mapping, operator_data_in);
  if(mg_data_in.coarse_solver == MultigridCoarseGridSolver::AMG_ML) // TODO: will be removed
    this->initialize_auxiliary_space(tria, global_levels, dirichlet_bc, mapping, operator_data_in);
  this->initialize_mg_matrices(global_levels, mapping, operator_data_in);
  this->initialize_smoothers();
  this->initialize_coarse_solver(global_levels[0].first);
  this->initialize_mg_transfer(tria, global_levels, h_levels, p_levels);
  this->initialize_multigrid_preconditioner();
}

/*
example: h_levels = [0 1 2], p_levels = [1 3 7]

p-MG:
global_levels  h_levels  p_levels
2              2         7
1              2         3
0              2         1

ph-MG:
global_levels  h_levels  p_levels
4              2         7
3              2         3
2              2         1
1              1         1
0              0         1

h-MG:
global_levels  h_levels  p_levels
2              2         7
1              1         7
0              0         7

hp-MG:
global_levels  h_levels  p_levels
4              2         7
3              1         7
2              0         7
1              0         3
0              0         1
*/

template<int dim, typename value_type, typename Operator>
void
MyMultigridPreconditionerBase<dim, value_type, Operator>::initialize_mg_sequence(
    const parallel::Triangulation<dim> * tria,  
    std::vector<std::pair<unsigned int, unsigned int>>& global_levels,
    std::vector<unsigned int> & h_levels,
    std::vector<unsigned int>& p_levels,
    unsigned int degree,
    MultigridType mg_type)
{

  for(unsigned int i = 0; i < tria->n_global_levels(); i++)
    h_levels.push_back(i);

  unsigned int temp = degree;
  do
  {
    p_levels.push_back(temp);
    temp = get_next_coarser_degree(temp);
  } while(temp != p_levels.back());
  std::reverse(std::begin(p_levels), std::end(p_levels));


  if(mg_type == MultigridType::pMG || mg_type == MultigridType::phMG)
  {
    // top level: p-gmg
    if(mg_type == MultigridType::phMG) // low level: h-gmg
      for(unsigned int i = 0; i < h_levels.size() - 1; i++)
        global_levels.push_back(std::pair<int, int>(h_levels[i], p_levels.front()));
    for(auto deg : p_levels)
      global_levels.push_back(std::pair<int, int>(h_levels.back(), deg));
  }
  else if(mg_type == MultigridType::hMG || mg_type == MultigridType::hpMG)
  {
    // top level: h-gmg
    if(mg_type == MultigridType::hpMG) // low level: p-gmg
      for(unsigned int i = 0; i < p_levels.size() - 1; i++)
        global_levels.push_back(std::pair<int, int>(h_levels.front(), p_levels[i]));
    for(auto geo : h_levels)
      global_levels.push_back(std::pair<int, int>(geo, p_levels.back()));
  }
  else
    AssertThrow(false, ExcMessage("This multigrid type does not exist!"));
}


template<int dim, typename value_type, typename Operator>
void
MyMultigridPreconditionerBase<dim, value_type, Operator>::initialize_auxiliary_space(
    const parallel::Triangulation<dim> * tria, 
    std::vector<std::pair<unsigned int, unsigned int>>& global_levels,
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const & dirichlet_bc,
    const Mapping<dim> & mapping,
    void * operator_data_in)
{
    // create coarse matrix with fe_q
    auto dof_handler_cg = new DoFHandler<dim>(*tria);
    dof_handler_cg->distribute_dofs(FE_Q<dim>(global_levels[0].second));
    dof_handler_cg->distribute_mg_dofs();
    this->cg_dofhandler.reset(dof_handler_cg);

    auto constrained_dofs_cg = new MGConstrainedDoFs();
    constrained_dofs_cg->clear();
    this->initialize_mg_constrained_dofs(*dof_handler_cg, *constrained_dofs_cg, dirichlet_bc);
    this->cg_constrained_dofs.reset(constrained_dofs_cg);

    // TODO: remove static cast
    auto matrix_cg = static_cast<Operator *>(underlying_operator->get_new(global_levels[0].second));
    matrix_cg->reinit(*dof_handler_cg, mapping, operator_data_in, *this->cg_constrained_dofs, global_levels[0].first);
    this->cg_matrices.reset(matrix_cg);    
}

template<int dim, typename value_type, typename Operator>
void
MyMultigridPreconditionerBase<dim, value_type, Operator>::initialize_mg_dof_handler_and_constraints(
    const DoFHandler<dim> & dof_handler,
    const parallel::Triangulation<dim> * tria, 
    std::vector<std::pair<unsigned int, unsigned int>>& global_levels,
    std::vector<unsigned int>& p_levels,
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const & dirichlet_bc,
    unsigned int degree)

{
  this->mg_constrained_dofs.resize(0, this->n_global_levels-1);
  this->mg_dofhandler.resize(0, this->n_global_levels-1);
  
  // determine number of components
  // n_components is needed so that also vector quantities can be handled
  // (note: since at the moment continuous space is only selectable as an auxiliary 
  // coarse space and vector quantities are not supported there, it is 
  // enough to determine this number for DG)
  const unsigned int n_components =
    dof_handler.n_dofs() / tria->n_global_active_cells() / std::pow(1 + degree, dim);
    
  // temporal storage for new dofhandlers and constraints on each p-level
  std::map<unsigned int, std::shared_ptr<const DoFHandler<dim>>> map_dofhandlers;
  std::map<unsigned int, std::shared_ptr<MGConstrainedDoFs>> map_constraints;
  
  // setup dof-handler and constrained dofs for each p-level
  for(unsigned int degree : p_levels)
  {
    // setup dof_handler: create dof_handler...
    auto dof_handler = new DoFHandler<dim>(*tria);
    // ... create FE and distrubute it
    dof_handler->distribute_dofs(FESystem<dim>(FE_DGQ<dim>(degree), n_components));
    dof_handler->distribute_mg_dofs();
    // setup constrained dofs:
    auto constrained_dofs = new MGConstrainedDoFs();
    constrained_dofs->clear();
    this->initialize_mg_constrained_dofs(*dof_handler, *constrained_dofs, dirichlet_bc);

    // put in temporal storage
    map_dofhandlers[degree] = std::shared_ptr<const DoFHandler<dim>>(dof_handler);
    map_constraints[degree] = std::shared_ptr<MGConstrainedDoFs>(constrained_dofs);
  }
  
  // populate dofhandler and constrained dofs to all hp-levels with the same degree
  for(unsigned int i = 0; i < global_levels.size(); i++)
  {
    int degree = global_levels[i].second;
    mg_dofhandler[i]       = map_dofhandlers[degree];
    mg_constrained_dofs[i] = map_constraints[degree];
  }
    
}

template<int dim, typename value_type, typename Operator>
void
MyMultigridPreconditionerBase<dim, value_type, Operator>::initialize_mg_matrices(
    std::vector<std::pair<unsigned int, unsigned int>>& global_levels,
    const Mapping<dim> & mapping,
    void * operator_data_in)
{
  this->mg_matrices.resize(0, this->n_global_levels-1);
    
  // create and setup operator on each level
  for(unsigned int i = 0; i < this->n_global_levels; i++)
  {
    auto matrix = static_cast<Operator *>(underlying_operator->get_new(global_levels[i].second));
    matrix->reinit(*mg_dofhandler[i], mapping, operator_data_in, *this->mg_constrained_dofs[i], global_levels[i].first);
    mg_matrices[i].reset(matrix);
  }
}

template<int dim, typename value_type, typename Operator>
void
MyMultigridPreconditionerBase<dim, value_type, Operator>::initialize_smoothers()
{
  this->mg_smoother.resize(0, this->n_global_levels-1);
    
  for(unsigned int i = 1; i < this->n_global_levels; i++)
    this->initialize_smoother(*this->mg_matrices[i], i);
}


template<int dim, typename value_type, typename Operator>
void
MyMultigridPreconditionerBase<dim, value_type, Operator>::initialize_mg_transfer(
    const parallel::Triangulation<dim> * tria,
    std::vector<std::pair<unsigned int, unsigned int>>& global_levels,
    std::vector<unsigned int> & /*h_levels*/,
    std::vector<unsigned int>& p_levels)
{
    
  this->mg_transfer.resize(0, this->n_global_levels-1);
    
#ifdef DEBUG
  unsigned int rank = Utilities::MPI::this_mpi_process(tria->get_communicator());
#else
  (void)tria;
#endif
    
  // setup transfer for h-MG: one h-transfer-operator is shared per p-level 
  for(unsigned int deg : p_levels)
  {
    // map: global level -> h level (is needed by the h-transfer operator, since it accesses the triangulation directly)
    std::map<unsigned int, unsigned int> map_global_level_to_h_level;

    // fill the map
    for(unsigned int i = 1; i < global_levels.size(); i++)
    {
      auto coarse_level = global_levels[i - 1];
      auto fine_level   = global_levels[i];
      if(coarse_level.first != fine_level.first && deg == coarse_level.second && deg == fine_level.second)
      {
#ifdef DEBUG
        if(!rank)
          printf("  h-MG (l=%2d,%2d) -> (k=%2d,%2d)\n", coarse_level.first, coarse_level.second, fine_level.first, fine_level.second);
#endif
        map_global_level_to_h_level[i] = fine_level.first;
      }
    }

    // there has been only one global level with this degree -> no h-transfer operator has to be created
    if(map_global_level_to_h_level.empty())
      continue;

    // create actual h-transfer-operator 
    std::shared_ptr<MGTransferMF<dim, typename Operator::value_type>> transfer(
      new MGTransferMF<dim, typename Operator::value_type>(map_global_level_to_h_level));
    transfer->initialize_constraints(*mg_constrained_dofs[map_global_level_to_h_level.begin()->first]);
    transfer->build(*mg_dofhandler[map_global_level_to_h_level.begin()->first]);

    // populate new h-transfer to levels sharing it
    for(auto i : map_global_level_to_h_level)
      mg_transfer[i.first] = transfer;
  }

  // setup transfer for p-MG
  for(unsigned int i = 1; i < global_levels.size(); i++)
  {
    auto coarse_level = global_levels[i - 1];
    auto fine_level  = global_levels[i];
    
    if(coarse_level.second != fine_level.second)
    {
      auto h_level = fine_level.first;
      AssertThrow(h_level == coarse_level.first, ExcMessage("The mesh level has to be the same for p-transfer."))
#ifdef DEBUG
      if(!rank)
        printf("  p-MG (l=%2d,k=%2d) -> (l=%2d,k=%2d)\n", coarse_level.first, coarse_level.second, h_level, fine_level.second);
#endif
      MGTransferBase<VECTOR_TYPE> * temp;

      const unsigned int from = fine_level.second, to = coarse_level.second;

// clang-format off      
#if DEGREE_15 && DEGREE_7
      if(from == 15 && to == 7)
        temp = new MGTransferMatrixFreeP<dim, 15, 7, typename Operator::value_type, VECTOR_TYPE>(*mg_dofhandler[i], *mg_dofhandler[i - 1], h_level);
      else
#endif
#if DEGREE_14 && DEGREE_7
      if(from == 14 && to == 7)
        temp = new MGTransferMatrixFreeP<dim, 14, 7, typename Operator::value_type, VECTOR_TYPE>(*mg_dofhandler[i], *mg_dofhandler[i - 1], h_level);
      else
#endif
#if DEGREE_13 && DEGREE_6
      if(from == 13 && to == 6)
        temp = new MGTransferMatrixFreeP<dim, 13, 6, typename Operator::value_type, VECTOR_TYPE>(*mg_dofhandler[i], *mg_dofhandler[i - 1], h_level);
      else
#endif
#if DEGREE_12 && DEGREE_6
      if(from == 12 && to == 6)
        temp = new MGTransferMatrixFreeP<dim, 12, 6, typename Operator::value_type, VECTOR_TYPE>(*mg_dofhandler[i], *mg_dofhandler[i - 1], h_level);
      else
#endif
#if DEGREE_11 && DEGREE_5
      if(from == 11 && to == 5)
        temp = new MGTransferMatrixFreeP<dim, 11, 5, typename Operator::value_type, VECTOR_TYPE>(*mg_dofhandler[i], *mg_dofhandler[i - 1], h_level);
      else
#endif
#if DEGREE_10 && DEGREE_5
      if(from == 10 && to == 5)
        temp = new MGTransferMatrixFreeP<dim, 9, 4, typename Operator::value_type, VECTOR_TYPE>(*mg_dofhandler[i], *mg_dofhandler[i - 1], h_level);
      else
#endif
#if DEGREE_9 && DEGREE_4
      if(from == 9 && to == 4)
        temp = new MGTransferMatrixFreeP<dim, 9, 4, typename Operator::value_type, VECTOR_TYPE>(*mg_dofhandler[i], *mg_dofhandler[i - 1], h_level);
      else
#endif
#if DEGREE_8 && DEGREE_4
      if(from == 8 && to == 4)
        temp = new MGTransferMatrixFreeP<dim, 8, 4, typename Operator::value_type, VECTOR_TYPE>(*mg_dofhandler[i], *mg_dofhandler[i - 1], h_level);
      else
#endif
#if DEGREE_7 && DEGREE_3
      if(from == 7 && to == 3)
        temp = new MGTransferMatrixFreeP<dim, 7, 3, typename Operator::value_type, VECTOR_TYPE>(*mg_dofhandler[i], *mg_dofhandler[i - 1], h_level);
      else
#endif
#if DEGREE_6 && DEGREE_3
      if(from == 6 && to == 3)
        temp = new MGTransferMatrixFreeP<dim, 6, 3, typename Operator::value_type, VECTOR_TYPE>(*mg_dofhandler[i], *mg_dofhandler[i - 1], h_level);
      else
#endif
#if DEGREE_5 && DEGREE_2
      if(from == 5 && to == 2)
        temp = new MGTransferMatrixFreeP<dim, 5, 2, typename Operator::value_type, VECTOR_TYPE>(*mg_dofhandler[i], *mg_dofhandler[i - 1], h_level);
      else
#endif
#if DEGREE_4 && DEGREE_2
      if(from == 4 && to == 2)
        temp = new MGTransferMatrixFreeP<dim, 4, 2, typename Operator::value_type, VECTOR_TYPE>(*mg_dofhandler[i], *mg_dofhandler[i - 1], h_level);
      else
#endif
#if DEGREE_3 && DEGREE_1
      if(from == 3 && to == 1)
        temp = new MGTransferMatrixFreeP<dim, 3, 1, typename Operator::value_type, VECTOR_TYPE>(*mg_dofhandler[i], *mg_dofhandler[i - 1], h_level);
      else
#endif
#if DEGREE_2 && DEGREE_1
      if(from == 2 && to == 1)
        temp = new MGTransferMatrixFreeP<dim, 2, 1, typename Operator::value_type, VECTOR_TYPE>(*mg_dofhandler[i], *mg_dofhandler[i - 1], h_level);
      else
#endif
// clang-format on
      {
        AssertThrow(false, ExcMessage("This type of p-transfer is not implemented"));
      }

      mg_transfer[i].reset(temp);
    }
  }
}


template<int dim, typename value_type, typename Operator>
void
MyMultigridPreconditionerBase<dim, value_type, Operator>::initialize_mg_constrained_dofs(
  const DoFHandler<dim> &                                              dof_handler,
  MGConstrainedDoFs &                                                  constrained_dofs,
  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const & dirichlet_bc)
{
  std::set<types::boundary_id> dirichlet_boundary;
  for(auto & it : dirichlet_bc)
    dirichlet_boundary.insert(it.first);
  constrained_dofs.initialize(dof_handler);
  constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary);
}

template<int dim, typename value_type, typename Operator>
void
MyMultigridPreconditionerBase<dim, value_type, Operator>::update(
  MatrixOperatorBase const * /*matrix_operator*/)
{
}

template<int dim, typename value_type, typename Operator>
void
MyMultigridPreconditionerBase<dim, value_type, Operator>::vmult(
  parallel::distributed::Vector<value_type> &       dst,
  const parallel::distributed::Vector<value_type> & src) const
{
  multigrid_preconditioner->vmult(dst, src);
}

template<int dim, typename value_type, typename Operator>
void
MyMultigridPreconditionerBase<dim, value_type, Operator>::apply_smoother_on_fine_level(
  parallel::distributed::Vector<typename Operator::value_type> &       dst,
  const parallel::distributed::Vector<typename Operator::value_type> & src) const
{
  this->mg_smoother[this->mg_smoother.max_level()]->vmult(dst, src);
}


template<int dim, typename value_type, typename Operator>
void
MyMultigridPreconditionerBase<dim, value_type, Operator>::update_smoother(unsigned int level)
{
  AssertThrow(level > 0, ExcMessage("Multigrid level is invalid when initializing multigrid smoother!"));

  switch(mg_data.smoother)
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
      typedef GMRESSmoother<Operator, VECTOR_TYPE> GMRES_SMOOTHER;
      std::shared_ptr<GMRES_SMOOTHER>              smoother =
        std::dynamic_pointer_cast<GMRES_SMOOTHER>(mg_smoother[level]);
      smoother->update();
      break;
    }
    case MultigridSmoother::CG:
    {
      typedef CGSmoother<Operator, VECTOR_TYPE> CG_SMOOTHER;
      std::shared_ptr<CG_SMOOTHER> smoother = std::dynamic_pointer_cast<CG_SMOOTHER>(mg_smoother[level]);
      smoother->update();
      break;
    }
    case MultigridSmoother::Jacobi:
    {
      typedef JacobiSmoother<Operator, VECTOR_TYPE> JACOBI_SMOOTHER;
      std::shared_ptr<JACOBI_SMOOTHER>              smoother =
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

template<int dim, typename value_type, typename Operator>
void
MyMultigridPreconditionerBase<dim, value_type, Operator>::update_coarse_solver()
{
  switch(mg_data.coarse_solver)
  {
    case MultigridCoarseGridSolver::Chebyshev:
    {
      initialize_chebyshev_smoother_coarse_grid(*mg_matrices[0]);
      break;
    }
    case MultigridCoarseGridSolver::ChebyshevNonsymmetricOperator:
    {
      initialize_chebyshev_smoother_nonsymmetric_operator_coarse_grid(*mg_matrices[0]);
      break;
    }
    case MultigridCoarseGridSolver::PCG_NoPreconditioner:
    {
      // do nothing
      break;
    }
    case MultigridCoarseGridSolver::PCG_PointJacobi:
    {
      std::shared_ptr<MGCoarsePCG<Operator>> coarse_solver =
        std::dynamic_pointer_cast<MGCoarsePCG<Operator>>(mg_coarse);
      coarse_solver->update_preconditioner(*this->mg_matrices[0]);

      break;
    }
    case MultigridCoarseGridSolver::PCG_BlockJacobi:
    {
      std::shared_ptr<MGCoarsePCG<Operator>> coarse_solver =
        std::dynamic_pointer_cast<MGCoarsePCG<Operator>>(mg_coarse);
      coarse_solver->update_preconditioner(*this->mg_matrices[0]);

      break;
    }
    case MultigridCoarseGridSolver::GMRES_NoPreconditioner:
    {
      // do nothing
      break;
    }
    case MultigridCoarseGridSolver::GMRES_PointJacobi:
    {
      std::shared_ptr<MGCoarseGMRES<Operator>> coarse_solver =
        std::dynamic_pointer_cast<MGCoarseGMRES<Operator>>(mg_coarse);
      coarse_solver->update_preconditioner(*this->mg_matrices[0]);
      break;
    }
    case MultigridCoarseGridSolver::GMRES_BlockJacobi:
    {
      std::shared_ptr<MGCoarseGMRES<Operator>> coarse_solver =
        std::dynamic_pointer_cast<MGCoarseGMRES<Operator>>(mg_coarse);
      coarse_solver->update_preconditioner(*this->mg_matrices[0]);
      break;
    }
    default:
    {
      AssertThrow(false, ExcMessage("Unknown coarse-grid solver given"));
    }
  }
}

template<int dim, typename value_type, typename Operator>
void
MyMultigridPreconditionerBase<dim, value_type, Operator>::initialize_smoother(Operator &   matrix,
                                                                              unsigned int level)
{
  AssertThrow(level > 0, ExcMessage("Multigrid level is invalid when initializing multigrid smoother!"));

  switch(mg_data.smoother)
  {
    case MultigridSmoother::Chebyshev:
    {
      mg_smoother[level].reset(new ChebyshevSmoother<Operator, VECTOR_TYPE>());
      initialize_chebyshev_smoother(matrix, level);
      break;
    }
    case MultigridSmoother::ChebyshevNonsymmetricOperator:
    {
      mg_smoother[level].reset(new ChebyshevSmoother<Operator, VECTOR_TYPE>());
      initialize_chebyshev_smoother_nonsymmetric_operator(matrix, level);
      break;
    }
    case MultigridSmoother::GMRES:
    {
      typedef GMRESSmoother<Operator, VECTOR_TYPE> GMRES_SMOOTHER;
      mg_smoother[level].reset(new GMRES_SMOOTHER());

      typename GMRES_SMOOTHER::AdditionalData smoother_data;
      smoother_data.preconditioner       = mg_data.gmres_smoother_data.preconditioner;
      smoother_data.number_of_iterations = mg_data.gmres_smoother_data.number_of_iterations;

      std::shared_ptr<GMRES_SMOOTHER> smoother =
        std::dynamic_pointer_cast<GMRES_SMOOTHER>(mg_smoother[level]);
      smoother->initialize(matrix, smoother_data);
      break;
    }
    case MultigridSmoother::CG:
    {
      typedef CGSmoother<Operator, VECTOR_TYPE> CG_SMOOTHER;
      mg_smoother[level].reset(new CG_SMOOTHER());

      typename CG_SMOOTHER::AdditionalData smoother_data;
      smoother_data.preconditioner       = mg_data.cg_smoother_data.preconditioner;
      smoother_data.number_of_iterations = mg_data.cg_smoother_data.number_of_iterations;

      std::shared_ptr<CG_SMOOTHER> smoother = std::dynamic_pointer_cast<CG_SMOOTHER>(mg_smoother[level]);
      smoother->initialize(matrix, smoother_data);
      break;
    }
    case MultigridSmoother::Jacobi:
    {
      typedef JacobiSmoother<Operator, VECTOR_TYPE> JACOBI_SMOOTHER;
      mg_smoother[level].reset(new JACOBI_SMOOTHER());

      typename JACOBI_SMOOTHER::AdditionalData smoother_data;
      smoother_data.preconditioner            = mg_data.jacobi_smoother_data.preconditioner;
      smoother_data.number_of_smoothing_steps = mg_data.jacobi_smoother_data.number_of_smoothing_steps;
      smoother_data.damping_factor            = mg_data.jacobi_smoother_data.damping_factor;

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

template<int dim, typename value_type, typename Operator>
void
MyMultigridPreconditionerBase<dim, value_type, Operator>::initialize_coarse_solver(
  const unsigned int coarse_level)
{
  Operator & matrix = *mg_matrices[0];
    
  switch(mg_data.coarse_solver)
  {
    case MultigridCoarseGridSolver::Chebyshev:
    {
      mg_smoother[0].reset(new ChebyshevSmoother<Operator, VECTOR_TYPE>());
      initialize_chebyshev_smoother_coarse_grid(matrix);

      mg_coarse.reset(
        new MGCoarseInverseOperator<parallel::distributed::Vector<typename Operator::value_type>, SMOOTHER>(
          mg_smoother[0]));
      break;
    }
    case MultigridCoarseGridSolver::ChebyshevNonsymmetricOperator:
    {
      mg_smoother[0].reset(new ChebyshevSmoother<Operator, VECTOR_TYPE>());
      initialize_chebyshev_smoother_nonsymmetric_operator_coarse_grid(matrix);

      mg_coarse.reset(
        new MGCoarseInverseOperator<parallel::distributed::Vector<typename Operator::value_type>, SMOOTHER>(
          mg_smoother[0]));
      break;
    }
    case MultigridCoarseGridSolver::PCG_NoPreconditioner:
    {
      typename MGCoarsePCG<Operator>::AdditionalData additional_data;
      additional_data.preconditioner = PreconditionerCoarseGridSolver::None;

      mg_coarse.reset(new MGCoarsePCG<Operator>(matrix, additional_data));
      break;
    }
    case MultigridCoarseGridSolver::PCG_PointJacobi:
    {
      typename MGCoarsePCG<Operator>::AdditionalData additional_data;
      additional_data.preconditioner = PreconditionerCoarseGridSolver::PointJacobi;

      mg_coarse.reset(new MGCoarsePCG<Operator>(matrix, additional_data));
      break;
    }
    case MultigridCoarseGridSolver::PCG_BlockJacobi:
    {
      typename MGCoarsePCG<Operator>::AdditionalData additional_data;
      additional_data.preconditioner = PreconditionerCoarseGridSolver::BlockJacobi;

      mg_coarse.reset(new MGCoarsePCG<Operator>(matrix, additional_data));
      break;
    }
    case MultigridCoarseGridSolver::GMRES_NoPreconditioner:
    {
      typename MGCoarseGMRES<Operator>::AdditionalData additional_data;
      additional_data.preconditioner = PreconditionerCoarseGridSolver::None;

      mg_coarse.reset(new MGCoarseGMRES<Operator>(matrix, additional_data));
      break;
    }
    case MultigridCoarseGridSolver::GMRES_PointJacobi:
    {
      typename MGCoarseGMRES<Operator>::AdditionalData additional_data;
      additional_data.preconditioner = PreconditionerCoarseGridSolver::PointJacobi;

      mg_coarse.reset(new MGCoarseGMRES<Operator>(matrix, additional_data));
      break;
    }
    case MultigridCoarseGridSolver::GMRES_BlockJacobi:
    {
      typename MGCoarseGMRES<Operator>::AdditionalData additional_data;
      additional_data.preconditioner = PreconditionerCoarseGridSolver::BlockJacobi;

      mg_coarse.reset(new MGCoarseGMRES<Operator>(matrix, additional_data));
      break;
    }
    case MultigridCoarseGridSolver::AMG_ML:
    {
      mg_coarse.reset(new MGCoarseML<Operator>(matrix, *cg_matrices, true, coarse_level, this->mg_data.coarse_ml_data));
      return;
    }
    default:
    {
      AssertThrow(false, ExcMessage("Unknown coarse-grid solver specified."));
    }
  }
}

template<int dim, typename value_type, typename Operator>
void
MyMultigridPreconditionerBase<dim, value_type, Operator>::initialize_multigrid_preconditioner()
{
  this->multigrid_preconditioner.reset(
    new MultigridPreconditioner<VECTOR_TYPE, Operator, MG_TRANSFER, SMOOTHER>(
      this->mg_matrices, *this->mg_coarse, this->mg_transfer, this->mg_smoother));
}

template<int dim, typename value_type, typename Operator>
void
MyMultigridPreconditionerBase<dim, value_type, Operator>::initialize_chebyshev_smoother(Operator &   matrix,
                                                                                        unsigned int level)
{
  typedef ChebyshevSmoother<Operator, VECTOR_TYPE> CHEBYSHEV_SMOOTHER;
  typename CHEBYSHEV_SMOOTHER::AdditionalData      smoother_data;

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

  smoother_data.smoothing_range     = mg_data.chebyshev_smoother_data.smoother_smoothing_range;
  smoother_data.degree              = mg_data.chebyshev_smoother_data.smoother_poly_degree;
  smoother_data.eig_cg_n_iterations = mg_data.chebyshev_smoother_data.eig_cg_n_iterations;

  std::shared_ptr<CHEBYSHEV_SMOOTHER> smoother =
    std::dynamic_pointer_cast<CHEBYSHEV_SMOOTHER>(mg_smoother[level]);
  smoother->initialize(matrix, smoother_data);
}

template<int dim, typename value_type, typename Operator>
void
MyMultigridPreconditionerBase<dim, value_type, Operator>::initialize_chebyshev_smoother_coarse_grid(
  Operator & matrix)
{
  // use Chebyshev smoother of high degree to solve the coarse grid problem
  // approximately
  typedef ChebyshevSmoother<Operator, VECTOR_TYPE> CHEBYSHEV_SMOOTHER;
  typename CHEBYSHEV_SMOOTHER::AdditionalData      smoother_data;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  matrix.initialize_dof_vector(smoother_data.matrix_diagonal_inverse);
  matrix.calculate_inverse_diagonal(smoother_data.matrix_diagonal_inverse);
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  std::pair<double, double> eigenvalues = compute_eigenvalues(matrix, smoother_data.matrix_diagonal_inverse);
#pragma GCC diagnostic pop
  double const factor           = 1.1;
  smoother_data.max_eigenvalue  = factor * eigenvalues.second;
  smoother_data.smoothing_range = eigenvalues.second / eigenvalues.first * factor;
  double sigma                  = (1. - std::sqrt(1. / smoother_data.smoothing_range)) /
                 (1. + std::sqrt(1. / smoother_data.smoothing_range));
  const double eps     = 1.e-3;
  smoother_data.degree = std::log(1. / eps + std::sqrt(1. / eps / eps - 1.)) / std::log(1. / sigma);
  smoother_data.eig_cg_n_iterations = 0;

  std::shared_ptr<CHEBYSHEV_SMOOTHER> smoother =
    std::dynamic_pointer_cast<CHEBYSHEV_SMOOTHER>(mg_smoother[0]);
  smoother->initialize(matrix, smoother_data);
}

template<int dim, typename value_type, typename Operator>
void
MyMultigridPreconditionerBase<dim, value_type, Operator>::initialize_chebyshev_smoother_nonsymmetric_operator(
  Operator &   matrix,
  unsigned int level)
{
  typedef ChebyshevSmoother<Operator, VECTOR_TYPE> CHEBYSHEV_SMOOTHER;
  typename CHEBYSHEV_SMOOTHER::AdditionalData      smoother_data;

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
  const unsigned int eig_n_iter = 20;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  std::pair<std::complex<double>, std::complex<double>> eigenvalues =
    compute_eigenvalues_gmres(matrix, smoother_data.matrix_diagonal_inverse, eig_n_iter);
#pragma GCC diagnostic pop
  const double factor               = 1.1;
  smoother_data.max_eigenvalue      = factor * std::abs(eigenvalues.second);
  smoother_data.smoothing_range     = mg_data.chebyshev_smoother_data.smoother_smoothing_range;
  smoother_data.degree              = mg_data.chebyshev_smoother_data.smoother_poly_degree;
  smoother_data.eig_cg_n_iterations = 0;

  std::shared_ptr<CHEBYSHEV_SMOOTHER> smoother =
    std::dynamic_pointer_cast<CHEBYSHEV_SMOOTHER>(mg_smoother[level]);
  smoother->initialize(matrix, smoother_data);
}

template<int dim, typename value_type, typename Operator>
void
MyMultigridPreconditionerBase<dim, value_type, Operator>::
  initialize_chebyshev_smoother_nonsymmetric_operator_coarse_grid(Operator & matrix)
{
  // use Chebyshev smoother of high degree to solve the coarse grid problem
  // approximately
  typedef ChebyshevSmoother<Operator, VECTOR_TYPE> CHEBYSHEV_SMOOTHER;
  typename CHEBYSHEV_SMOOTHER::AdditionalData      smoother_data;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  matrix.initialize_dof_vector(smoother_data.matrix_diagonal_inverse);
  matrix.calculate_inverse_diagonal(smoother_data.matrix_diagonal_inverse);
#pragma GCC diagnostic pop

  const double factor = 1.1;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  std::pair<std::complex<double>, std::complex<double>> eigenvalues =
    compute_eigenvalues_gmres(matrix, smoother_data.matrix_diagonal_inverse);
#pragma GCC diagnostic pop
  smoother_data.max_eigenvalue  = factor * std::abs(eigenvalues.second);
  smoother_data.smoothing_range = factor * std::abs(eigenvalues.second) / std::abs(eigenvalues.first);
  double sigma                  = (1. - std::sqrt(1. / smoother_data.smoothing_range)) /
                 (1. + std::sqrt(1. / smoother_data.smoothing_range));
  const double eps     = 1e-3;
  smoother_data.degree = std::log(1. / eps + std::sqrt(1. / eps / eps - 1)) / std::log(1. / sigma);
  smoother_data.eig_cg_n_iterations = 0;

  std::shared_ptr<CHEBYSHEV_SMOOTHER> smoother =
    std::dynamic_pointer_cast<CHEBYSHEV_SMOOTHER>(mg_smoother[0]);
  smoother->initialize(matrix, smoother_data);
}

#include "multigrid_preconditioner_adapter_base.hpp"