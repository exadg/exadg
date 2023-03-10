/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// deal.II
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <deal.II/fe/mapping_fe.h>
#include <exadg/grid/mapping_dof_vector.h>
#include <exadg/matrix_free/categorization.h>
#include <exadg/solvers_and_preconditioners/multigrid/coarse_grid_solvers.h>
#include <exadg/solvers_and_preconditioners/multigrid/constraints.h>
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_algorithm.h>
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h>
#include <exadg/solvers_and_preconditioners/multigrid/smoothers/cg_smoother.h>
#include <exadg/solvers_and_preconditioners/multigrid/smoothers/chebyshev_smoother.h>
#include <exadg/solvers_and_preconditioners/multigrid/smoothers/gmres_smoother.h>
#include <exadg/solvers_and_preconditioners/multigrid/smoothers/jacobi_smoother.h>
#include <exadg/solvers_and_preconditioners/multigrid/transfers/mg_transfer_global_coarsening.h>
#include <exadg/solvers_and_preconditioners/multigrid/transfers/mg_transfer_global_refinement.h>
#include <exadg/solvers_and_preconditioners/utilities/compute_eigenvalues.h>
#include <exadg/utilities/mpi.h>

namespace ExaDG
{
template<int dim, typename Number>
MultigridPreconditionerBase<dim, Number>::MultigridPreconditionerBase(MPI_Comm const & comm)
  : n_levels(1),
    coarse_level(0),
    fine_level(0),
    mpi_comm(comm),
    multigrid_variant(MultigridVariant::LocalSmoothing),
    triangulation(nullptr)
{
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::initialize(
  MultigridData const &                                                  data,
  MultigridVariant const &                                               multigrid_variant,
  dealii::Triangulation<dim> const *                                     tria,
  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> const & coarse_triangulations,
  dealii::FiniteElement<dim> const &                                     fe,
  std::shared_ptr<dealii::Mapping<dim> const>                            mapping,
  bool const                                                             operator_is_singular,
  Map_DBC const &                                                        dirichlet_bc,
  Map_DBC_ComponentMask const & dirichlet_bc_component_mask,
  PeriodicFacePairs const &     periodic_face_pairs)
{
  this->data = data;

  this->multigrid_variant = multigrid_variant;

  this->triangulation = tria;

  this->coarse_triangulations = coarse_triangulations;

  this->mapping = mapping;

  bool const is_dg = fe.dofs_per_vertex == 0;

  this->initialize_levels(this->triangulation, fe.degree, is_dg);

  this->initialize_mapping();

  this->initialize_dof_handler_and_constraints(operator_is_singular,
                                               periodic_face_pairs,
                                               fe,
                                               this->triangulation,
                                               dirichlet_bc,
                                               dirichlet_bc_component_mask);

  this->initialize_matrix_free();

  this->initialize_operators();

  this->initialize_smoothers();

  this->initialize_coarse_solver(operator_is_singular);

  this->initialize_transfer_operators();

  this->initialize_multigrid_algorithm();
}

/*
 *
 * example: h_levels = [0 1 2], p_levels = [1 3 7]
 *
 * p-MG:
 * levels  h_levels  p_levels
 * 2       2         7
 * 1       2         3
 * 0       2         1
 *
 * ph-MG:
 * levels  h_levels  p_levels
 * 4       2         7
 * 3       2         3
 * 2       2         1
 * 1       1         1
 * 0       0         1
 *
 * h-MG:
 * levels  h_levels  p_levels
 * 2       2         7
 * 1       1         7
 * 0       0         7
 *
 * hp-MG:
 * levels  h_levels  p_levels
 * 4       2         7
 * 3       1         7
 * 2       0         7
 * 1       0         3
 * 0       0         1
 *
 */

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::initialize_levels(dealii::Triangulation<dim> const * tria,
                                                            unsigned int const degree,
                                                            bool const         is_dg)
{
  MultigridType const mg_type = data.type;

  std::vector<unsigned int> h_levels;

  // setup h-levels
  if(mg_type == MultigridType::pMG || mg_type == MultigridType::cpMG ||
     mg_type == MultigridType::pcMG)
  {
    h_levels.push_back(tria->n_global_levels() - 1);
  }
  else // h-MG is involved working on all mesh levels
  {
    bool const use_global_coarsening = (multigrid_variant == MultigridVariant::GlobalCoarsening);

    if(use_global_coarsening)
    {
      AssertThrow(
        coarse_triangulations.size() > 0,
        dealii::ExcMessage(
          "You are using global coarsening multigrid, but the vector coarse_triangulations is empty. "
          "Most likely, you forgot to set the parameter GridData::create_coarse_triangulations."));
    }

    unsigned int const n_h_levels =
      (use_global_coarsening ? coarse_triangulations.size() : tria->n_global_levels());

    for(unsigned int h = 0; h < n_h_levels; h++)
      h_levels.push_back(h);
  }

  // setup p-levels
  if(mg_type == MultigridType::hMG)
  {
    p_levels.push_back({degree, is_dg});
  }
  else if(mg_type == MultigridType::cMG || mg_type == MultigridType::chMG ||
          mg_type == MultigridType::hcMG)
  {
    p_levels.push_back({degree, false});
    p_levels.push_back({degree, is_dg});
  }
  else // p-MG is involved with high- and low-order elements
  {
    unsigned int p = degree;

    bool discontinuous = is_dg;

    // c-transfer before p-coarsening
    if(is_dg)
    {
      if(mg_type == MultigridType::cpMG || mg_type == MultigridType::hcpMG ||
         mg_type == MultigridType::chpMG || mg_type == MultigridType::cphMG)
      {
        p_levels.push_back({p, discontinuous});
        discontinuous = false;
      }
    }

    do
    {
      p_levels.push_back({p, discontinuous});
      switch(data.p_sequence)
      {
          // clang-format off
        case PSequenceType::GoToOne:       p = 1;                                                break;
        case PSequenceType::DecreaseByOne: p = std::max(p-1, 1u);                                break;
        case PSequenceType::Bisect:        p = std::max(p/2, 1u);                                break;
        case PSequenceType::Manual:        p = (degree==3 && p==3) ? 2 : std::max(degree/2, 1u); break;
        default:
          AssertThrow(false, dealii::ExcMessage("No valid p-sequence selected!"));
          // clang-format on
      }
    } while(p != p_levels.back().degree);

    // c-transfer after p-coarsening
    if(is_dg)
    {
      if(mg_type == MultigridType::pcMG || mg_type == MultigridType::hpcMG ||
         mg_type == MultigridType::phcMG || mg_type == MultigridType::pchMG)
      {
        p_levels.push_back({p, false});
      }
    }

    // sort p levels from coarse to fine
    std::reverse(std::begin(p_levels), std::end(p_levels));
  }

  // setup global-levels from coarse to fine and inserting via push_back
  if(mg_type == MultigridType::hMG)
  {
    for(unsigned int h = 0; h < h_levels.size(); h++)
      level_info.push_back({h_levels[h], p_levels.front()});
  }
  else if(mg_type == MultigridType::cMG)
  {
    level_info.push_back({h_levels.back(), p_levels.front()});
    level_info.push_back({h_levels.back(), p_levels.back()});
  }
  else if(mg_type == MultigridType::chMG)
  {
    for(unsigned int h = 0; h < h_levels.size(); h++)
      level_info.push_back({h_levels[h], p_levels.front()});

    level_info.push_back({h_levels.back(), p_levels.back()});
  }
  else if(mg_type == MultigridType::hcMG)
  {
    level_info.push_back({h_levels.front(), p_levels.front()});

    for(unsigned int h = 0; h < h_levels.size(); h++)
      level_info.push_back({h_levels[h], p_levels.back()});
  }
  else if(mg_type == MultigridType::pMG || mg_type == MultigridType::pcMG ||
          mg_type == MultigridType::cpMG)
  {
    for(unsigned int p = 0; p < p_levels.size(); p++)
      level_info.push_back({h_levels.front(), p_levels[p]});
  }
  else if(mg_type == MultigridType::phMG || mg_type == MultigridType::cphMG ||
          mg_type == MultigridType::pchMG)
  {
    for(unsigned int h = 0; h < h_levels.size() - 1; h++)
      level_info.push_back({h_levels[h], p_levels.front()});

    for(auto p : p_levels)
      level_info.push_back({h_levels.back(), p});
  }
  else if(mg_type == MultigridType::hpMG || mg_type == MultigridType::hcpMG ||
          mg_type == MultigridType::hpcMG)
  {
    for(unsigned int p = 0; p < p_levels.size() - 1; p++)
      level_info.push_back({h_levels.front(), p_levels[p]});

    for(auto h : h_levels)
      level_info.push_back({h, p_levels.back()});
  }
  else if(mg_type == MultigridType::phcMG)
  {
    level_info.push_back({h_levels.front(), p_levels.front()});

    std::vector<MGDoFHandlerIdentifier>::iterator it = p_levels.begin();
    ++it;

    for(unsigned int h = 0; h < h_levels.size() - 1; h++)
      level_info.push_back({h_levels[h], *it});

    for(; it != p_levels.end(); ++it)
      level_info.push_back({h_levels.back(), *it});
  }
  else if(mg_type == MultigridType::chpMG)
  {
    for(unsigned int p = 0; p < p_levels.size() - 2; p++)
      level_info.push_back({h_levels.front(), p_levels[p]});

    for(auto h : h_levels)
      level_info.push_back({h, p_levels[p_levels.size() - 2]});

    level_info.push_back({h_levels.back(), p_levels.back()});
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("This multigrid type is not implemented!"));
  }

  this->n_levels     = level_info.size(); // number of actual multigrid levels
  this->coarse_level = 0;
  this->fine_level   = this->n_levels - 1;

  this->check_levels(level_info);
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::check_levels(std::vector<MGLevelInfo> const & level_info)
{
  AssertThrow(n_levels == level_info.size(),
              dealii::ExcMessage("Variable n_levels is not initialized correctly."));
  AssertThrow(coarse_level == 0,
              dealii::ExcMessage("Variable coarse_level is not initialized correctly."));
  AssertThrow(fine_level == n_levels - 1,
              dealii::ExcMessage("Variable fine_level is not initialized correctly."));

  for(unsigned int l = 1; l < level_info.size(); l++)
  {
    auto fine   = level_info[l];
    auto coarse = level_info[l - 1];

    AssertThrow(
      (fine.h_level() != coarse.h_level()) xor (fine.degree() != coarse.degree()) xor
        (fine.is_dg() != coarse.is_dg()),
      dealii::ExcMessage(
        "Between two consecutive multigrid levels, only one type of transfer is allowed."));
  }
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::initialize_mapping()
{
  // We only need to initialize the mapping for all multigrid h-levels if it is of type
  // dealii::MappingQCache (including MappingDoFVector as a derived class), while MappingQ is
  // unproblematic.
  std::shared_ptr<dealii::MappingQCache<dim> const> mapping_q_cache =
    std::dynamic_pointer_cast<dealii::MappingQCache<dim> const>(mapping);

  if(data.involves_h_transfer() && mapping_q_cache.get() != 0)
  {
    if(multigrid_variant == MultigridVariant::GlobalCoarsening)
    {
      MappingTools::initialize_multigrid(coarse_grid_mappings,
                                         mapping_q_cache,
                                         coarse_triangulations);
    }
    else if(multigrid_variant == MultigridVariant::LocalSmoothing)
    {
      mapping_dof_vector =
        std::make_shared<MappingDoFVector<dim, Number>>(mapping_q_cache->get_degree());

      MappingTools::initialize_multigrid(mapping_dof_vector, mapping_q_cache, *triangulation);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("not implemented."));
    }
  }
}

template<int dim, typename Number>
dealii::Mapping<dim> const &
MultigridPreconditionerBase<dim, Number>::get_mapping(unsigned int const h_level) const
{
  std::shared_ptr<dealii::MappingQCache<dim> const> mapping_q_cache =
    std::dynamic_pointer_cast<dealii::MappingQCache<dim> const>(mapping);

  if(data.involves_h_transfer() && mapping_q_cache.get() != 0)
  {
    if(multigrid_variant == MultigridVariant::GlobalCoarsening)
    {
      AssertThrow(h_level < coarse_grid_mappings.size(),
                  dealii::ExcMessage("coarse_grid_mappings are not initialized correctly."));

      return *(coarse_grid_mappings[h_level]);
    }
    else if(multigrid_variant == MultigridVariant::LocalSmoothing)
    {
      AssertThrow(mapping_dof_vector.get() != 0,
                  dealii::ExcMessage("mapping_dof_vector is not initialized correctly."));

      return *mapping_dof_vector;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("not implemented."));
    }
  }
  else
  {
    return *mapping;
  }
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::initialize_dof_handler_and_constraints(
  bool const                         operator_is_singular,
  PeriodicFacePairs const &          periodic_face_pairs,
  dealii::FiniteElement<dim> const & fe,
  dealii::Triangulation<dim> const * tria,
  Map_DBC const &                    dirichlet_bc,
  Map_DBC_ComponentMask const &      dirichlet_bc_component_mask)
{
  this->do_initialize_dof_handler_and_constraints(operator_is_singular,
                                                  periodic_face_pairs,
                                                  fe,
                                                  tria,
                                                  dirichlet_bc,
                                                  dirichlet_bc_component_mask,
                                                  this->level_info,
                                                  this->p_levels,
                                                  this->dof_handlers,
                                                  this->constrained_dofs,
                                                  this->constraints);
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::do_initialize_dof_handler_and_constraints(
  bool                                  is_singular,
  PeriodicFacePairs const &             periodic_face_pairs,
  dealii::FiniteElement<dim> const &    fe,
  dealii::Triangulation<dim> const *    tria,
  Map_DBC const &                       dirichlet_bc,
  Map_DBC_ComponentMask const &         dirichlet_bc_component_mask,
  std::vector<MGLevelInfo> &            level_info,
  std::vector<MGDoFHandlerIdentifier> & p_levels,
  dealii::MGLevelObject<std::shared_ptr<dealii::DoFHandler<dim> const>> & dof_handlers,
  dealii::MGLevelObject<std::shared_ptr<dealii::MGConstrainedDoFs>> &     constrained_dofs,
  dealii::MGLevelObject<std::shared_ptr<dealii::AffineConstraints<MultigridNumber>>> & constraints)
{
  constrained_dofs.resize(0, this->n_levels - 1);
  dof_handlers.resize(0, this->n_levels - 1);
  constraints.resize(0, this->n_levels - 1);

  // this type of transfer has to be used for triangulations with hanging nodes
  if(multigrid_variant == MultigridVariant::GlobalCoarsening)
  {
    // setup dof-handler and constrained dofs for all multigrid levels
    for(unsigned int i = 0; i < level_info.size(); i++)
    {
      auto const & level = level_info[i];

      auto dof_handler =
        new dealii::DoFHandler<dim>((level.h_level() + 1 == triangulation->n_global_levels()) ?
                                      *(dynamic_cast<dealii::Triangulation<dim> const *>(tria)) :
                                      *(coarse_triangulations[level.h_level()]));

      if(tria->all_reference_cells_are_hyper_cube())
      {
        if(level.is_dg())
          dof_handler->distribute_dofs(
            dealii::FESystem<dim>(dealii::FE_DGQ<dim>(level.degree()), fe.n_components()));
        else
          dof_handler->distribute_dofs(
            dealii::FESystem<dim>(dealii::FE_Q<dim>(level.degree()), fe.n_components()));
      }
      else if(tria->all_reference_cells_are_simplex())
      {
        if(level.is_dg())
          dof_handler->distribute_dofs(
            dealii::FESystem<dim>(dealii::FE_SimplexDGP<dim>(level.degree()), fe.n_components()));
        else
          dof_handler->distribute_dofs(
            dealii::FESystem<dim>(dealii::FE_SimplexP<dim>(level.degree()), fe.n_components()));
      }
      else
        AssertThrow(false, dealii::ExcMessage("Only hypercube or simplex elements are supported."));

      dof_handlers[i].reset(dof_handler);

      auto affine_constraints_own = new dealii::AffineConstraints<MultigridNumber>();

      // TODO: integrate periodic constraints into initialize_affine_constraints

      AssertThrow(is_singular == false, dealii::ExcNotImplemented());
      AssertThrow(periodic_face_pairs.empty(),
                  dealii::ExcMessage(
                    "Multigrid transfer option use_global_coarsening "
                    "is currently not available for problems with periodic boundaries."));

      dealii::IndexSet locally_relevant_dofs;
      dealii::DoFTools::extract_locally_relevant_dofs(*dof_handler, locally_relevant_dofs);
      affine_constraints_own->reinit(locally_relevant_dofs);

      dealii::DoFTools::make_hanging_node_constraints(*dof_handler, *affine_constraints_own);

      // collect all boundary functions and translate to format understood by
      // deal.II to cover all boundaries at once
      dealii::Functions::ZeroFunction<dim, MultigridNumber> zero_function(
        dof_handler->get_fe().n_components());

      std::map<dealii::types::boundary_id, dealii::Function<dim, MultigridNumber> const *>
        boundary_functions;
      for(auto & it : dirichlet_bc)
      {
        boundary_functions[it.first] = &zero_function;
      }

      auto const & mapping_dummy =
        dof_handler->get_fe().reference_cell().template get_default_linear_mapping<dim>();

      dealii::VectorTools::interpolate_boundary_values(mapping_dummy,
                                                       *dof_handler,
                                                       boundary_functions,
                                                       *affine_constraints_own);

      affine_constraints_own->close();

      constraints[i].reset(affine_constraints_own);
    }
  }
  else if(multigrid_variant == MultigridVariant::LocalSmoothing)
  {
    AssertThrow(tria->has_hanging_nodes() == false,
                dealii::ExcMessage("Hanging nodes are only supported with the option "
                                   "use_global_coarsening enabled."));
    AssertThrow(tria->all_reference_cells_are_hyper_cube(),
                dealii::ExcMessage("This multigrid implementation is currently only available for "
                                   "hyper-cube elements. Other grids need to enable the option "
                                   "use_global_coarsening."));

    unsigned int const n_components = fe.n_components();

    // temporal storage for new DoFHandlers and constraints on each p-level
    std::map<MGDoFHandlerIdentifier, std::shared_ptr<dealii::DoFHandler<dim> const>>
      map_dofhandlers;
    std::map<MGDoFHandlerIdentifier, std::shared_ptr<dealii::MGConstrainedDoFs>>
      map_constrained_dofs;

    // setup dof-handler and constrained dofs for each p-level
    for(auto level : p_levels)
    {
      // setup dof_handler
      auto dof_handler = new dealii::DoFHandler<dim>(*tria);

      // distribute dofs
      if(level.is_dg)
      {
        dof_handler->distribute_dofs(
          dealii::FESystem<dim>(dealii::FE_DGQ<dim>(level.degree), n_components));
      }
      else
      {
        dof_handler->distribute_dofs(
          dealii::FESystem<dim>(dealii::FE_Q<dim>(level.degree), n_components));
      }

      // distribute MG dofs
      dof_handler->distribute_mg_dofs();

      // constrained dofs
      auto constrained_dofs = new dealii::MGConstrainedDoFs();
      constrained_dofs->clear();

      if(level.is_dg)
      {
        std::set<dealii::types::boundary_id> dirichlet_boundary;
        for(auto & it : dirichlet_bc)
          dirichlet_boundary.insert(it.first);
        constrained_dofs->initialize(*dof_handler);
        constrained_dofs->make_zero_boundary_constraints(*dof_handler, dirichlet_boundary);
      }
      else
      {
        constrained_dofs->initialize(*dof_handler);
        for(auto it : dirichlet_bc)
        {
          std::set<dealii::types::boundary_id> dirichlet_boundary;
          dirichlet_boundary.insert(it.first);

          dealii::ComponentMask mask    = dealii::ComponentMask();
          auto                  it_mask = dirichlet_bc_component_mask.find(it.first);
          if(it_mask != dirichlet_bc_component_mask.end())
            mask = it_mask->second;

          constrained_dofs->make_zero_boundary_constraints(*dof_handler, dirichlet_boundary, mask);
        }
      }

      // put in temporal storage
      map_dofhandlers[level]      = std::shared_ptr<dealii::DoFHandler<dim> const>(dof_handler);
      map_constrained_dofs[level] = std::shared_ptr<dealii::MGConstrainedDoFs>(constrained_dofs);
    }

    // populate dof-handler and constrained dofs to all hp-levels with the same degree
    for(unsigned int level = 0; level < level_info.size(); level++)
    {
      auto p_level            = level_info[level].dof_handler_id();
      dof_handlers[level]     = map_dofhandlers[p_level];
      constrained_dofs[level] = map_constrained_dofs[p_level];
    }

    for(unsigned int level = coarse_level; level <= fine_level; level++)
    {
      auto affine_constraints_own = new dealii::AffineConstraints<MultigridNumber>;

      ConstraintUtil::add_constraints<dim>(level_info[level].is_dg(),
                                           is_singular,
                                           *dof_handlers[level],
                                           *affine_constraints_own,
                                           *constrained_dofs[level],
                                           periodic_face_pairs,
                                           level_info[level].h_level());

      constraints[level].reset(affine_constraints_own);
    }
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("not implemented."));
  }
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::initialize_matrix_free()
{
  matrix_free_data_objects.resize(0, n_levels - 1);
  matrix_free_objects.resize(0, n_levels - 1);

  for(unsigned int level = coarse_level; level <= fine_level; level++)
  {
    unsigned int const h_level = (multigrid_variant == MultigridVariant::GlobalCoarsening) ?
                                   dealii::numbers::invalid_unsigned_int :
                                   level_info[level].h_level();

    matrix_free_data_objects[level] = std::make_shared<MatrixFreeData<dim, MultigridNumber>>();
    fill_matrix_free_data(*matrix_free_data_objects[level], level, h_level);

    matrix_free_objects[level] = std::make_shared<dealii::MatrixFree<dim, MultigridNumber>>();

    matrix_free_objects[level]->reinit(get_mapping(level_info[level].h_level()),
                                       matrix_free_data_objects[level]->get_dof_handler_vector(),
                                       matrix_free_data_objects[level]->get_constraint_vector(),
                                       matrix_free_data_objects[level]->get_quadrature_vector(),
                                       matrix_free_data_objects[level]->data);
  }
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::update_matrix_free()
{
  for(unsigned int level = coarse_level; level <= fine_level; level++)
    matrix_free_objects[level]->update_mapping(get_mapping(level_info[level].h_level()));
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::initialize_operators()
{
  this->operators.resize(0, this->n_levels - 1);

  // create and setup operator on each level
  for(unsigned int level = coarse_level; level <= fine_level; level++)
    operators[level] = this->initialize_operator(level);
}

template<int dim, typename Number>
std::shared_ptr<
  MultigridOperatorBase<dim, typename MultigridPreconditionerBase<dim, Number>::MultigridNumber>>
MultigridPreconditionerBase<dim, Number>::initialize_operator(unsigned int const level)
{
  (void)level;

  AssertThrow(false,
              dealii::ExcMessage("This function needs to be implemented by derived classes."));

  std::shared_ptr<Operator> op;

  return op;
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::initialize_smoothers()
{
  this->smoothers.resize(0, this->n_levels - 1);

  // skip the coarsest level
  for(unsigned int level = coarse_level + 1; level <= fine_level; level++)
    this->initialize_smoother(*this->operators[level], level);
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::update()
{
  // do nothing in base class (has to be implemented by derived classes if necessary)
}

template<int dim, typename Number>
std::shared_ptr<TimerTree>
MultigridPreconditionerBase<dim, Number>::get_timings() const
{
  return multigrid_algorithm->get_timings();
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::vmult(VectorType & dst, VectorType const & src) const
{
  multigrid_algorithm->vmult(dst, src);
}

template<int dim, typename Number>
unsigned int
MultigridPreconditionerBase<dim, Number>::solve(VectorType & dst, VectorType const & src) const
{
  return multigrid_algorithm->solve(dst, src);
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::apply_smoother_on_fine_level(
  VectorTypeMG &       dst,
  VectorTypeMG const & src) const
{
  this->smoothers[this->smoothers.max_level()]->vmult(dst, src);
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::initialize_smoother(Operator &   mg_operator,
                                                              unsigned int level)
{
  AssertThrow(level > 0,
              dealii::ExcMessage(
                "Multigrid level is invalid when initializing multigrid smoother!"));

  switch(data.smoother_data.smoother)
  {
    case MultigridSmoother::Chebyshev:
    {
      if(data.smoother_data.preconditioner == PreconditionerSmoother::PointJacobi)
      {
        smoothers[level] = std::make_shared<
          ChebyshevSmoother<Operator, VectorTypeMG, dealii::DiagonalMatrix<VectorTypeMG>>>();
        initialize_chebyshev_smoother_point_jacobi(mg_operator, level);
      }
      else if(data.smoother_data.preconditioner == PreconditionerSmoother::BlockJacobi)
      {
        smoothers[level] = std::make_shared<
          ChebyshevSmoother<Operator, VectorTypeMG, BlockJacobiPreconditioner<Operator>>>();
        initialize_chebyshev_smoother_block_jacobi(mg_operator, level);
      }
      else
        AssertThrow(false, dealii::ExcNotImplemented());
      break;
    }
    case MultigridSmoother::GMRES:
    {
      typedef GMRESSmoother<Operator, VectorTypeMG> GMRES;
      smoothers[level] = std::make_shared<GMRES>();

      typename GMRES::AdditionalData smoother_data;
      smoother_data.preconditioner       = data.smoother_data.preconditioner;
      smoother_data.number_of_iterations = data.smoother_data.iterations;

      std::shared_ptr<GMRES> smoother = std::dynamic_pointer_cast<GMRES>(smoothers[level]);
      smoother->initialize(mg_operator, smoother_data);
      break;
    }
    case MultigridSmoother::CG:
    {
      typedef CGSmoother<Operator, VectorTypeMG> CG;
      smoothers[level] = std::make_shared<CG>();

      typename CG::AdditionalData smoother_data;
      smoother_data.preconditioner       = data.smoother_data.preconditioner;
      smoother_data.number_of_iterations = data.smoother_data.iterations;

      std::shared_ptr<CG> smoother = std::dynamic_pointer_cast<CG>(smoothers[level]);
      smoother->initialize(mg_operator, smoother_data);
      break;
    }
    case MultigridSmoother::Jacobi:
    {
      typedef JacobiSmoother<Operator, VectorTypeMG> Jacobi;
      smoothers[level] = std::make_shared<Jacobi>();

      typename Jacobi::AdditionalData smoother_data;
      smoother_data.preconditioner            = data.smoother_data.preconditioner;
      smoother_data.number_of_smoothing_steps = data.smoother_data.iterations;
      smoother_data.damping_factor            = data.smoother_data.relaxation_factor;

      std::shared_ptr<Jacobi> smoother = std::dynamic_pointer_cast<Jacobi>(smoothers[level]);
      smoother->initialize(mg_operator, smoother_data);
      break;
    }
    default:
    {
      AssertThrow(false, dealii::ExcMessage("Specified MultigridSmoother not implemented!"));
    }
  }
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::update_smoothers()
{
  // Skip coarsest level
  for(unsigned int level = this->coarse_level + 1; level <= this->fine_level; ++level)
  {
    this->update_smoother(level);
  }
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::update_smoother(unsigned int level)
{
  AssertThrow(level > 0,
              dealii::ExcMessage(
                "Multigrid level is invalid when initializing multigrid smoother!"));

  switch(data.smoother_data.smoother)
  {
    case MultigridSmoother::Chebyshev:
    {
      if(data.smoother_data.preconditioner == PreconditionerSmoother::PointJacobi)
      {
        smoothers[level] = std::make_shared<
          ChebyshevSmoother<Operator, VectorTypeMG, dealii::DiagonalMatrix<VectorTypeMG>>>();
        initialize_chebyshev_smoother_point_jacobi(*operators[level], level);
      }
      else if(data.smoother_data.preconditioner == PreconditionerSmoother::BlockJacobi)
      {
        smoothers[level] = std::make_shared<
          ChebyshevSmoother<Operator, VectorTypeMG, BlockJacobiPreconditioner<Operator>>>();
        initialize_chebyshev_smoother_block_jacobi(*operators[level], level);
      }
      else
        AssertThrow(false, dealii::ExcNotImplemented());
      break;
    }
    case MultigridSmoother::GMRES:
    {
      typedef GMRESSmoother<Operator, VectorTypeMG> GMRES;

      std::shared_ptr<GMRES> smoother = std::dynamic_pointer_cast<GMRES>(smoothers[level]);
      smoother->update();
      break;
    }
    case MultigridSmoother::CG:
    {
      typedef CGSmoother<Operator, VectorTypeMG> CG;

      std::shared_ptr<CG> smoother = std::dynamic_pointer_cast<CG>(smoothers[level]);
      smoother->update();
      break;
    }
    case MultigridSmoother::Jacobi:
    {
      typedef JacobiSmoother<Operator, VectorTypeMG> Jacobi;

      std::shared_ptr<Jacobi> smoother = std::dynamic_pointer_cast<Jacobi>(smoothers[level]);
      smoother->update();
      break;
    }
    default:
    {
      AssertThrow(false, dealii::ExcMessage("Specified MultigridSmoother not implemented!"));
    }
  }
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::update_coarse_solver(bool const operator_is_singular)
{
  switch(data.coarse_problem.solver)
  {
    case MultigridCoarseGridSolver::Chebyshev:
    {
      AssertThrow(
        data.coarse_problem.preconditioner == MultigridCoarseGridPreconditioner::PointJacobi,
        dealii::ExcMessage(
          "Only PointJacobi preconditioner implemented for Chebyshev coarse grid solver."));

      initialize_chebyshev_smoother_coarse_grid(*operators[0],
                                                data.coarse_problem.solver_data,
                                                operator_is_singular);
      break;
    }
    case MultigridCoarseGridSolver::CG:
    case MultigridCoarseGridSolver::GMRES:
    {
      if(data.coarse_problem.preconditioner != MultigridCoarseGridPreconditioner::None)
      {
        std::shared_ptr<MGCoarseKrylov<Operator>> coarse_solver =
          std::dynamic_pointer_cast<MGCoarseKrylov<Operator>>(coarse_grid_solver);
        coarse_solver->update();
      }

      break;
    }
    case MultigridCoarseGridSolver::AMG:
    {
      std::shared_ptr<MGCoarseAMG<Operator>> coarse_solver =
        std::dynamic_pointer_cast<MGCoarseAMG<Operator>>(coarse_grid_solver);
      coarse_solver->update();

      break;
    }
    default:
    {
      AssertThrow(false, dealii::ExcMessage("Unknown coarse-grid solver given"));
    }
  }
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::initialize_coarse_solver(bool const operator_is_singular)
{
  Operator & coarse_operator = *operators[0];

  switch(data.coarse_problem.solver)
  {
    case MultigridCoarseGridSolver::Chebyshev:
    {
      AssertThrow(
        data.coarse_problem.preconditioner == MultigridCoarseGridPreconditioner::PointJacobi,
        dealii::ExcMessage(
          "Only PointJacobi preconditioner implemented for Chebyshev coarse grid solver."));

      smoothers[0] = std::make_shared<
        ChebyshevSmoother<Operator, VectorTypeMG, dealii::DiagonalMatrix<VectorTypeMG>>>();
      initialize_chebyshev_smoother_coarse_grid(coarse_operator,
                                                data.coarse_problem.solver_data,
                                                operator_is_singular);

      coarse_grid_solver =
        std::make_shared<MGCoarseChebyshev<VectorTypeMG, Smoother>>(smoothers[0]);
      break;
    }
    case MultigridCoarseGridSolver::CG:
    case MultigridCoarseGridSolver::GMRES:
    {
      typename MGCoarseKrylov<Operator>::AdditionalData additional_data;

      if(data.coarse_problem.solver == MultigridCoarseGridSolver::CG)
        additional_data.solver_type = KrylovSolverType::CG;
      else if(data.coarse_problem.solver == MultigridCoarseGridSolver::GMRES)
        additional_data.solver_type = KrylovSolverType::GMRES;
      else
        AssertThrow(false, dealii::ExcMessage("Not implemented."));

      additional_data.solver_data          = data.coarse_problem.solver_data;
      additional_data.operator_is_singular = operator_is_singular;
      additional_data.preconditioner       = data.coarse_problem.preconditioner;
      additional_data.amg_data             = data.coarse_problem.amg_data;

      coarse_grid_solver =
        std::make_shared<MGCoarseKrylov<Operator>>(coarse_operator, additional_data, mpi_comm);
      break;
    }
    case MultigridCoarseGridSolver::AMG:
    {
      if(data.coarse_problem.amg_data.amg_type == AMGType::ML)
      {
        coarse_grid_solver =
          std::make_shared<MGCoarseAMG<Operator>>(coarse_operator, data.coarse_problem.amg_data);
      }
      else if(data.coarse_problem.amg_data.amg_type == AMGType::BoomerAMG)
      {
        coarse_grid_solver =
          std::make_shared<MGCoarseAMG<Operator>>(coarse_operator, data.coarse_problem.amg_data);
      }
      else
      {
        AssertThrow(false, dealii::ExcNotImplemented());
      }

      break;
    }
    default:
    {
      AssertThrow(false, dealii::ExcMessage("Unknown coarse-grid solver specified."));
    }
  }
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::initialize_transfer_operators()
{
  unsigned int const dof_index = 0;
  this->do_initialize_transfer_operators(transfers, constrained_dofs, dof_index);
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::do_initialize_transfer_operators(
  std::shared_ptr<MGTransfer<VectorTypeMG>> &                         transfers,
  dealii::MGLevelObject<std::shared_ptr<dealii::MGConstrainedDoFs>> & constrained_dofs,
  unsigned int const                                                  dof_index)
{
  if(multigrid_variant == MultigridVariant::GlobalCoarsening)
  {
    auto tmp = std::make_shared<MGTransferGlobalCoarsening<dim, MultigridNumber, VectorTypeMG>>();

    tmp->reinit(matrix_free_objects, dof_index);

    transfers = tmp;
  }
  else if(multigrid_variant == MultigridVariant::LocalSmoothing)
  {
    auto tmp = std::make_shared<MGTransferGlobalRefinement<dim, MultigridNumber, VectorTypeMG>>();

    tmp->reinit(*mapping, matrix_free_objects, constrained_dofs, dof_index);

    transfers = tmp;
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("not implemented."));
  }
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::initialize_multigrid_algorithm()
{
  this
    ->multigrid_algorithm = std::make_shared<MultigridAlgorithm<VectorTypeMG, Operator, Smoother>>(
    this->operators, *this->coarse_grid_solver, *this->transfers, this->smoothers, this->mpi_comm);
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::initialize_chebyshev_smoother_point_jacobi(
  Operator &         mg_operator,
  unsigned int const level)
{
  AssertThrow(data.smoother_data.preconditioner == PreconditionerSmoother::PointJacobi,
              dealii::ExcNotImplemented());

  typedef ChebyshevSmoother<Operator, VectorTypeMG, dealii::DiagonalMatrix<VectorTypeMG>> Chebyshev;
  typename Chebyshev::AdditionalData smoother_data;

  std::shared_ptr<dealii::DiagonalMatrix<VectorTypeMG>> diagonal_matrix =
    std::make_shared<dealii::DiagonalMatrix<VectorTypeMG>>();
  VectorTypeMG & diagonal_vector = diagonal_matrix->get_vector();

  mg_operator.initialize_dof_vector(diagonal_vector);
  mg_operator.calculate_inverse_diagonal(diagonal_vector);

  smoother_data.preconditioner = diagonal_matrix;

  smoother_data.smoothing_range     = data.smoother_data.smoothing_range;
  smoother_data.degree              = data.smoother_data.iterations;
  smoother_data.eig_cg_n_iterations = data.smoother_data.iterations_eigenvalue_estimation;

  std::shared_ptr<Chebyshev> smoother = std::dynamic_pointer_cast<Chebyshev>(smoothers[level]);
  smoother->initialize(mg_operator, smoother_data);
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::initialize_chebyshev_smoother_block_jacobi(
  Operator &         mg_operator,
  unsigned int const level)
{
  AssertThrow(data.smoother_data.preconditioner == PreconditionerSmoother::BlockJacobi,
              dealii::ExcNotImplemented());

  typedef ChebyshevSmoother<Operator, VectorTypeMG, BlockJacobiPreconditioner<Operator>> Chebyshev;
  typename Chebyshev::AdditionalData smoother_data;

  smoother_data.preconditioner = std::make_shared<BlockJacobiPreconditioner<Operator>>(mg_operator);

  smoother_data.smoothing_range     = data.smoother_data.smoothing_range;
  smoother_data.degree              = data.smoother_data.iterations;
  smoother_data.eig_cg_n_iterations = data.smoother_data.iterations_eigenvalue_estimation;

  std::shared_ptr<Chebyshev> smoother = std::dynamic_pointer_cast<Chebyshev>(smoothers[level]);
  smoother->initialize(mg_operator, smoother_data);
}

template<int dim, typename Number>
void
MultigridPreconditionerBase<dim, Number>::initialize_chebyshev_smoother_coarse_grid(
  Operator &         coarse_operator,
  SolverData const & solver_data,
  bool const         operator_is_singular)
{
  // use Chebyshev smoother of high degree to solve the coarse grid problem approximately
  typedef ChebyshevSmoother<Operator, VectorTypeMG, dealii::DiagonalMatrix<VectorTypeMG>> Chebyshev;
  typename Chebyshev::AdditionalData smoother_data;

  std::shared_ptr<dealii::DiagonalMatrix<VectorTypeMG>> diagonal_matrix =
    std::make_shared<dealii::DiagonalMatrix<VectorTypeMG>>();
  VectorTypeMG & diagonal_vector = diagonal_matrix->get_vector();

  coarse_operator.initialize_dof_vector(diagonal_vector);
  coarse_operator.calculate_inverse_diagonal(diagonal_vector);

  std::pair<double, double> eigenvalues =
    compute_eigenvalues(coarse_operator, diagonal_vector, operator_is_singular);

  double const factor = 1.1;

  smoother_data.preconditioner  = diagonal_matrix;
  smoother_data.max_eigenvalue  = factor * eigenvalues.second;
  smoother_data.smoothing_range = eigenvalues.second / eigenvalues.first * factor;

  double sigma = (1. - std::sqrt(1. / smoother_data.smoothing_range)) /
                 (1. + std::sqrt(1. / smoother_data.smoothing_range));

  // calculate/estimate the number of Chebyshev iterations needed to reach a specified relative
  // solver tolerance
  double const eps = solver_data.rel_tol;

  smoother_data.degree = static_cast<unsigned int>(
    std::log(1. / eps + std::sqrt(1. / eps / eps - 1.)) / std::log(1. / sigma));
  smoother_data.eig_cg_n_iterations = 0;

  std::shared_ptr<Chebyshev> smoother = std::dynamic_pointer_cast<Chebyshev>(smoothers[0]);
  smoother->initialize(coarse_operator, smoother_data);
}


template class MultigridPreconditionerBase<2, float>;
template class MultigridPreconditionerBase<2, double>;

template class MultigridPreconditionerBase<3, float>;
template class MultigridPreconditionerBase<3, double>;

} // namespace ExaDG
