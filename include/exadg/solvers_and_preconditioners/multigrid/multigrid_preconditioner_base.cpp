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
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/grid/grid_utilities.h>
#include <exadg/grid/mapping_dof_vector.h>
#include <exadg/matrix_free/categorization.h>
#include <exadg/operators/finite_element.h>
#include <exadg/solvers_and_preconditioners/multigrid/constraints.h>
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_algorithm.h>
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h>
#include <exadg/solvers_and_preconditioners/multigrid/smoothers/cg_smoother.h>
#include <exadg/solvers_and_preconditioners/multigrid/smoothers/chebyshev_smoother.h>
#include <exadg/solvers_and_preconditioners/multigrid/smoothers/gmres_smoother.h>
#include <exadg/solvers_and_preconditioners/multigrid/smoothers/jacobi_smoother.h>
#include <exadg/solvers_and_preconditioners/multigrid/transfer.h>
#include <exadg/solvers_and_preconditioners/utilities/compute_eigenvalues.h>
#include <exadg/utilities/mpi.h>

namespace ExaDG
{
template<int dim, typename Number, typename MultigridNumber>
MultigridPreconditionerBase<dim, Number, MultigridNumber>::MultigridPreconditionerBase(
  MPI_Comm const & comm)
  : mpi_comm(comm)
{
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize(
  MultigridData const &                       data,
  std::shared_ptr<Grid<dim> const>            grid,
  std::shared_ptr<dealii::Mapping<dim> const> mapping,
  dealii::FiniteElement<dim> const &          fe,
  bool const                                  operator_is_singular,
  Map_DBC const &                             dirichlet_bc,
  Map_DBC_ComponentMask const &               dirichlet_bc_component_mask,
  bool const                                  initialize_preconditioners)
{
  this->data = data;

  this->grid = grid;

  this->mapping = mapping;

  bool const is_dg = (fe.dofs_per_vertex == 0);

  this->initialize_levels(fe.degree, is_dg);

  this->initialize_mapping();

  this->initialize_dof_handler_and_constraints(operator_is_singular,
                                               fe.n_components(),
                                               dirichlet_bc,
                                               dirichlet_bc_component_mask);

  this->initialize_matrix_free_objects();

  this->initialize_transfer_operators();

  this->initialize_operators();

  this->initialize_smoothers(initialize_preconditioners);

  this->initialize_coarse_solver(operator_is_singular, initialize_preconditioners);

  this->initialize_multigrid_algorithm();
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_levels(
  unsigned int const degree,
  bool const         is_dg)
{
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

  MultigridType const mg_type = data.type;

  std::vector<unsigned int> h_levels;
  std::vector<unsigned int> dealii_tria_levels;


  // setup h-levels

  // In case only a single h-level exists
  if(not(data.involves_h_transfer()) or (grid->triangulation->n_global_levels() == 1))
  {
    h_levels.push_back(0);
    // the only h-level that exists is an active level
    dealii_tria_levels.push_back(dealii::numbers::invalid_unsigned_int);
  }
  else // involves_h_transfer == true and n_global_levels() > 1
  {
    // In case we have a separate Triangulation object for each h-level
    if(grid->coarse_triangulations.size() > 0)
    {
      for(unsigned int h = 0; h < grid->coarse_triangulations.size() + 1; h++)
      {
        h_levels.push_back(h);
        dealii_tria_levels.push_back(dealii::numbers::invalid_unsigned_int);
      }
    }
    else
    {
      for(unsigned int h = 0; h < grid->triangulation->n_global_levels(); h++)
      {
        h_levels.push_back(h);
        dealii_tria_levels.push_back(h);
      }
    }
  }


  // setup p-levels
  if(mg_type == MultigridType::hMG)
  {
    p_levels.push_back({degree, is_dg});
  }
  else if(mg_type == MultigridType::cMG or mg_type == MultigridType::chMG or
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
      if(mg_type == MultigridType::cpMG or mg_type == MultigridType::hcpMG or
         mg_type == MultigridType::chpMG or mg_type == MultigridType::cphMG)
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
        case PSequenceType::Manual:        p = (degree==3 and p==3) ? 2 : std::max(degree/2, 1u); break;
        default:
          AssertThrow(false, dealii::ExcMessage("No valid p-sequence selected!"));
          // clang-format on
      }
    } while(p != p_levels.back().degree);

    // c-transfer after p-coarsening
    if(is_dg)
    {
      if(mg_type == MultigridType::pcMG or mg_type == MultigridType::hpcMG or
         mg_type == MultigridType::phcMG or mg_type == MultigridType::pchMG)
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
      level_info.push_back({h_levels[h], dealii_tria_levels[h], p_levels.front()});
  }
  else if(mg_type == MultigridType::cMG)
  {
    level_info.push_back({h_levels.back(), dealii_tria_levels.back(), p_levels.front()});
    level_info.push_back({h_levels.back(), dealii_tria_levels.back(), p_levels.back()});
  }
  else if(mg_type == MultigridType::chMG)
  {
    for(unsigned int h = 0; h < h_levels.size(); h++)
      level_info.push_back({h_levels[h], dealii_tria_levels[h], p_levels.front()});

    level_info.push_back({h_levels.back(), dealii_tria_levels.back(), p_levels.back()});
  }
  else if(mg_type == MultigridType::hcMG)
  {
    level_info.push_back({h_levels.front(), dealii_tria_levels.front(), p_levels.front()});

    for(unsigned int h = 0; h < h_levels.size(); h++)
      level_info.push_back({h_levels[h], dealii_tria_levels[h], p_levels.back()});
  }
  else if(mg_type == MultigridType::pMG or mg_type == MultigridType::pcMG or
          mg_type == MultigridType::cpMG)
  {
    for(unsigned int p = 0; p < p_levels.size(); p++)
      level_info.push_back({h_levels.front(), dealii_tria_levels.front(), p_levels[p]});
  }
  else if(mg_type == MultigridType::phMG or mg_type == MultigridType::cphMG or
          mg_type == MultigridType::pchMG)
  {
    for(unsigned int h = 0; h < h_levels.size() - 1; h++)
      level_info.push_back({h_levels[h], dealii_tria_levels[h], p_levels.front()});

    for(auto p : p_levels)
      level_info.push_back({h_levels.back(), dealii_tria_levels.back(), p});
  }
  else if(mg_type == MultigridType::hpMG or mg_type == MultigridType::hcpMG or
          mg_type == MultigridType::hpcMG)
  {
    for(unsigned int p = 0; p < p_levels.size() - 1; p++)
      level_info.push_back({h_levels.front(), dealii_tria_levels.front(), p_levels[p]});

    for(unsigned int h = 0; h < h_levels.size(); h++)
      level_info.push_back({h_levels[h], dealii_tria_levels[h], p_levels.back()});
  }
  else if(mg_type == MultigridType::phcMG)
  {
    level_info.push_back({h_levels.front(), dealii_tria_levels.front(), p_levels.front()});

    std::vector<MGDoFHandlerIdentifier>::iterator it = p_levels.begin();
    ++it;

    for(unsigned int h = 0; h < h_levels.size() - 1; h++)
      level_info.push_back({h_levels[h], dealii_tria_levels[h], *it});

    for(; it != p_levels.end(); ++it)
      level_info.push_back({h_levels.back(), dealii_tria_levels.back(), *it});
  }
  else if(mg_type == MultigridType::chpMG)
  {
    for(unsigned int p = 0; p < p_levels.size() - 2; p++)
      level_info.push_back({h_levels.front(), dealii_tria_levels.front(), p_levels[p]});

    for(unsigned int h = 0; h < h_levels.size(); h++)
      level_info.push_back({h_levels[h], dealii_tria_levels[h], p_levels[p_levels.size() - 2]});

    level_info.push_back({h_levels.back(), dealii_tria_levels.back(), p_levels.back()});
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("This multigrid type is not implemented!"));
  }

  for(unsigned int l = 1; l < get_number_of_levels(); l++)
  {
    auto fine   = level_info[l];
    auto coarse = level_info[l - 1];

    AssertThrow(
      (fine.h_level() != coarse.h_level()) xor (fine.degree() != coarse.degree()) xor
        (fine.is_dg() != coarse.is_dg()),
      dealii::ExcMessage(
        "Between two consecutive multigrid levels, only one type of transfer is allowed."));
  }

  AssertThrow(h_levels.size() == dealii_tria_levels.size(),
              dealii::ExcMessage("h_levels and dealii_tria_levels have different size."));
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_mapping()
{
  unsigned int const n_h_levels = level_info.back().h_level() - level_info.front().h_level() + 1;

  multigrid_mappings->initialize_coarse_mappings(*grid, n_h_levels);
}

template<int dim, typename Number, typename MultigridNumber>
dealii::Mapping<dim> const &
MultigridPreconditionerBase<dim, Number, MultigridNumber>::get_mapping(
  unsigned int const h_level) const
{
  unsigned int const n_h_levels = level_info.back().h_level() - level_info.front().h_level() + 1;

  return multigrid_mappings->get_mapping(h_level, n_h_levels);
}

template<int dim, typename Number, typename MultigridNumber>
unsigned int
MultigridPreconditionerBase<dim, Number, MultigridNumber>::get_number_of_levels() const
{
  AssertThrow(level_info.size() > 0,
              dealii::ExcMessage(
                "MultigridPreconditionerBase: level_info seems to be uninitialized."));

  return level_info.size();
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_dof_handler_and_constraints(
  bool const                    operator_is_singular,
  unsigned int const            n_components,
  Map_DBC const &               dirichlet_bc,
  Map_DBC_ComponentMask const & dirichlet_bc_component_mask)
{
  this->do_initialize_dof_handler_and_constraints(operator_is_singular,
                                                  n_components,
                                                  dirichlet_bc,
                                                  dirichlet_bc_component_mask,
                                                  this->dof_handlers,
                                                  this->constraints);
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::
  do_initialize_dof_handler_and_constraints(
    bool                          is_singular,
    unsigned int const            n_components,
    Map_DBC const &               dirichlet_bc,
    Map_DBC_ComponentMask const & dirichlet_bc_component_mask,
    dealii::MGLevelObject<std::shared_ptr<dealii::DoFHandler<dim> const>> & dof_handlers,
    dealii::MGLevelObject<std::shared_ptr<dealii::AffineConstraints<MultigridNumber>>> &
      constraints)
{
  dealii::MGLevelObject<std::shared_ptr<dealii::MGConstrainedDoFs>> constrained_dofs;
  constrained_dofs.resize(0, get_number_of_levels() - 1);
  dof_handlers.resize(0, get_number_of_levels() - 1);
  constraints.resize(0, get_number_of_levels() - 1);

  bool const is_hypercube_mesh_without_hanging_nodes =
    grid->triangulation->all_reference_cells_are_hyper_cube() and
    not(grid->triangulation->has_hanging_nodes());

  if(grid->coarse_triangulations.size() > 0 or not(is_hypercube_mesh_without_hanging_nodes))
  {
    // setup dof-handler and constrained dofs for all multigrid levels
    for_all_levels([&](unsigned int const l) {
      auto const & level = level_info[l];

      std::shared_ptr<dealii::FiniteElement<dim>> fe = create_finite_element<dim>(
        get_element_type(*grid->triangulation), level.is_dg(), n_components, level.degree());

      std::shared_ptr<dealii::Triangulation<dim> const> triangulation;
      if(level.h_level() == level_info.back().h_level()) // fine-level triangulation
      {
        triangulation = grid->triangulation;
      }
      else
      {
        AssertThrow(level.h_level() < grid->coarse_triangulations.size(),
                    dealii::ExcMessage(
                      "The vector coarse_triangulations seems to have incorrect size."));

        triangulation = grid->coarse_triangulations[level.h_level()];
      }

      std::shared_ptr<dealii::DoFHandler<dim>> dof_handler =
        std::make_shared<dealii::DoFHandler<dim>>(*triangulation);

      dof_handler->distribute_dofs(*fe);

      dof_handlers[l] = dof_handler;

      auto affine_constraints_own = new dealii::AffineConstraints<MultigridNumber>();

      AssertThrow(is_singular == false, dealii::ExcNotImplemented());

      dealii::IndexSet locally_relevant_dofs;
      dealii::DoFTools::extract_locally_relevant_dofs(*dof_handler, locally_relevant_dofs);
      affine_constraints_own->reinit(locally_relevant_dofs);

      // hanging nodes (needs to be done before imposing periodicity constraints and boundary
      // conditions)
      dealii::DoFTools::make_hanging_node_constraints(*dof_handler, *affine_constraints_own);

      // constraints from periodic boundary conditions
      if(not(grid->periodic_face_pairs.empty()))
      {
        std::vector<
          dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator>>
          periodic_faces;
        if(level.h_level() == level_info.back().h_level()) // fine-level triangulation
        {
          periodic_faces = grid->periodic_face_pairs;
        }
        else
        {
          AssertThrow(
            grid->coarse_periodic_face_pairs.size() == grid->coarse_triangulations.size(),
            dealii::ExcMessage(
              "The size of coarse_periodic_face_pairs differs from the size of coarse_triangulations."));

          AssertThrow(level.h_level() < grid->coarse_periodic_face_pairs.size(),
                      dealii::ExcMessage(
                        "The vector coarse_periodic_face_pairs seems to have incorrect size."));

          periodic_faces = grid->coarse_periodic_face_pairs[level.h_level()];
        }

        // change type of dealii cell iterator
        std::vector<
          dealii::GridTools::PeriodicFacePair<typename dealii::DoFHandler<dim>::cell_iterator>>
          periodic_faces_dof =
            GridUtilities::transform_periodic_face_pairs_to_dof_cell_iterator(periodic_faces,
                                                                              *dof_handler);

        dealii::DoFTools::make_periodicity_constraints<dim, dim, MultigridNumber>(
          periodic_faces_dof, *affine_constraints_own);
      }

      // collect all boundary functions and translate to format understood by
      // deal.II to cover all boundaries at once
      dealii::Functions::ZeroFunction<dim, MultigridNumber> zero_function(
        dof_handler->get_fe().n_components());

      auto const & mapping_dummy =
        dof_handler->get_fe().reference_cell().template get_default_linear_mapping<dim>();

      for(auto & it : dirichlet_bc)
      {
        dealii::ComponentMask mask = dealii::ComponentMask();

        auto it_mask = dirichlet_bc_component_mask.find(it.first);
        if(it_mask != dirichlet_bc_component_mask.end())
          mask = it_mask->second;

        dealii::VectorTools::interpolate_boundary_values(
          mapping_dummy, *dof_handler, it.first, zero_function, *affine_constraints_own, mask);
      }

      affine_constraints_own->close();

      constraints[l].reset(affine_constraints_own);
    });
  }
  else
  {
    AssertThrow(is_hypercube_mesh_without_hanging_nodes,
                dealii::ExcMessage(
                  "This implementation only allows globally refined hypercube meshes."));

    // temporal storage for new DoFHandlers and constraints on each p-level
    std::map<MGDoFHandlerIdentifier, std::shared_ptr<dealii::DoFHandler<dim> const>>
      map_dofhandlers;
    std::map<MGDoFHandlerIdentifier, std::shared_ptr<dealii::MGConstrainedDoFs>>
      map_constrained_dofs;

    // setup dof-handler and constrained dofs for each p-level
    for(auto level : p_levels)
    {
      // create finite element
      std::shared_ptr<dealii::FiniteElement<dim>> fe =
        create_finite_element<dim>(ElementType::Hypercube, level.is_dg, n_components, level.degree);

      // create dof handler
      auto dof_handler = new dealii::DoFHandler<dim>(*grid->triangulation);

      // distribute dofs
      dof_handler->distribute_dofs(*fe);

      // distribute MG dofs
      dof_handler->distribute_mg_dofs();

      // constrained dofs
      auto constrained_dofs = new dealii::MGConstrainedDoFs();
      constrained_dofs->clear();
      constrained_dofs->initialize(*dof_handler);

      if(not(level.is_dg))
      {
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

    // populate dof-handler and constrained dofs of a certain p-levels to all multigrid levels with
    // the same FE / DoFHandler
    for_all_levels([&](unsigned int const level) {
      auto p_level            = level_info[level].dof_handler_id();
      dof_handlers[level]     = map_dofhandlers[p_level];
      constrained_dofs[level] = map_constrained_dofs[p_level];
    });

    for_all_levels([&](unsigned int const level) {
      auto affine_constraints_own = new dealii::AffineConstraints<MultigridNumber>;

      ConstraintUtil::add_constraints<dim>(level_info[level].is_dg(),
                                           is_singular,
                                           *dof_handlers[level],
                                           *affine_constraints_own,
                                           *constrained_dofs[level],
                                           grid->periodic_face_pairs,
                                           level_info[level].h_level());

      constraints[level].reset(affine_constraints_own);
    });
  }
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_matrix_free_objects()
{
  matrix_free_data_objects.resize(0, get_number_of_levels() - 1);
  matrix_free_objects.resize(0, get_number_of_levels() - 1);

  for_all_levels([&](unsigned int const level) {
    matrix_free_data_objects[level] = std::make_shared<MatrixFreeData<dim, MultigridNumber>>();
    fill_matrix_free_data(*matrix_free_data_objects[level],
                          level,
                          level_info[level].dealii_tria_level());

    matrix_free_objects[level] = std::make_shared<dealii::MatrixFree<dim, MultigridNumber>>();

    matrix_free_objects[level]->reinit(get_mapping(level_info[level].h_level()),
                                       matrix_free_data_objects[level]->get_dof_handler_vector(),
                                       matrix_free_data_objects[level]->get_constraint_vector(),
                                       matrix_free_data_objects[level]->get_quadrature_vector(),
                                       matrix_free_data_objects[level]->data);
  });
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::update_matrix_free_objects()
{
  for_all_levels([&](unsigned int const level) {
    matrix_free_objects[level]->update_mapping(get_mapping(level_info[level].h_level()));
  });
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_operators()
{
  this->operators.resize(0, this->get_number_of_levels() - 1);

  for_all_levels(
    [&](unsigned int const level) { operators[level] = this->initialize_operator(level); });
}

template<int dim, typename Number, typename MultigridNumber>
std::shared_ptr<MultigridOperatorBase<
  dim,
  typename MultigridPreconditionerBase<dim, Number, MultigridNumber>::MultigridNumber>>
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_operator(
  unsigned int const level)
{
  (void)level;

  AssertThrow(false,
              dealii::ExcMessage("This function needs to be implemented by derived classes."));

  std::shared_ptr<Operator> op;

  return op;
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_smoothers(
  bool const initialize_preconditioner)
{
  if(get_number_of_levels() >= 2)
    this->smoothers.resize(1, get_number_of_levels() - 1);

  for_all_smoothing_levels([&](unsigned int const level) {
    this->initialize_smoother(*this->operators[level], level, initialize_preconditioner);
  });
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::update()
{
  // do nothing in base class (has to be implemented by derived classes if necessary)
}

template<int dim, typename Number, typename MultigridNumber>
std::shared_ptr<TimerTree>
MultigridPreconditionerBase<dim, Number, MultigridNumber>::get_timings() const
{
  return multigrid_algorithm->get_timings();
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::vmult(VectorType &       dst,
                                                                 VectorType const & src) const
{
  AssertThrow(not this->update_needed,
              dealii::ExcMessage(
                "Multigrid preconditioner can not be applied because it needs to be updated."));

  multigrid_algorithm->vmult(dst, src);
}

template<int dim, typename Number, typename MultigridNumber>
unsigned int
MultigridPreconditionerBase<dim, Number, MultigridNumber>::solve(VectorType &       dst,
                                                                 VectorType const & src) const
{
  AssertThrow(not this->update_needed,
              dealii::ExcMessage(
                "Multigrid preconditioner can not be applied because it needs to be updated."));

  return multigrid_algorithm->solve(dst, src);
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::apply_smoother_on_fine_level(
  VectorTypeMG &       dst,
  VectorTypeMG const & src) const
{
  this->smoothers[this->smoothers.max_level()]->vmult(dst, src);
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_smoother(
  Operator &   mg_operator,
  unsigned int level,
  bool const   initialize_preconditioner)
{
  AssertThrow(level > 0 and level < this->get_number_of_levels(),
              dealii::ExcMessage(
                "Multigrid level is invalid when initializing multigrid smoother!"));

  switch(data.smoother_data.smoother)
  {
    case MultigridSmoother::Chebyshev:
    {
      typedef ChebyshevSmoother<Operator, VectorTypeMG> Chebyshev;
      smoothers[level] = std::make_shared<Chebyshev>();

      typename Chebyshev::AdditionalData smoother_data;
      smoother_data.preconditioner  = data.smoother_data.preconditioner;
      smoother_data.smoothing_range = data.smoother_data.smoothing_range;
      smoother_data.degree          = data.smoother_data.iterations;
      smoother_data.iterations_eigenvalue_estimation =
        data.smoother_data.iterations_eigenvalue_estimation;

      std::shared_ptr<Chebyshev> smoother = std::dynamic_pointer_cast<Chebyshev>(smoothers[level]);
      smoother->setup(mg_operator, initialize_preconditioner, smoother_data);
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
      smoother->setup(mg_operator, initialize_preconditioner, smoother_data);
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
      smoother->setup(mg_operator, initialize_preconditioner, smoother_data);
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
      smoother->setup(mg_operator, initialize_preconditioner, smoother_data);
      break;
    }
    default:
    {
      AssertThrow(false, dealii::ExcMessage("Specified MultigridSmoother not implemented!"));
    }
  }
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::update_smoothers()
{
  for_all_smoothing_levels([&](unsigned int const level) { smoothers[level]->update(); });
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::update_coarse_solver()
{
  coarse_grid_solver->update();
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_coarse_solver(
  bool const operator_is_singular,
  bool const initialize_preconditioners)
{
  Operator & coarse_operator = *operators[0];

  switch(data.coarse_problem.solver)
  {
    case MultigridCoarseGridSolver::Chebyshev:
    {
      coarse_grid_solver =
        std::make_shared<MGCoarseChebyshev<Operator>>(coarse_operator,
                                                      initialize_preconditioners,
                                                      data.coarse_problem.solver_data.rel_tol,
                                                      data.coarse_problem.preconditioner,
                                                      operator_is_singular);
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

      coarse_grid_solver = std::make_shared<MGCoarseKrylov<Operator>>(coarse_operator,
                                                                      initialize_preconditioners,
                                                                      additional_data,
                                                                      mpi_comm);
      break;
    }
    case MultigridCoarseGridSolver::AMG:
    {
      if(data.coarse_problem.amg_data.amg_type == AMGType::ML or
         data.coarse_problem.amg_data.amg_type == AMGType::BoomerAMG)
      {
        coarse_grid_solver = std::make_shared<MGCoarseAMG<Operator>>(coarse_operator,
                                                                     initialize_preconditioners,
                                                                     data.coarse_problem.amg_data);
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

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_transfer_operators()
{
  unsigned int const dof_index = 0;
  this->do_initialize_transfer_operators(transfers, dof_index);
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::do_initialize_transfer_operators(
  std::shared_ptr<MultigridTransfer<dim, MultigridNumber, VectorTypeMG>> & transfers,
  unsigned int const                                                       dof_index)
{
  transfers = std::make_shared<MultigridTransfer<dim, MultigridNumber, VectorTypeMG>>();

  transfers->reinit(matrix_free_objects, dof_index, level_info);
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_multigrid_algorithm()
{
  multigrid_algorithm = std::make_shared<MultigridAlgorithm<VectorTypeMG, Operator, Smoother>>(
    operators, *coarse_grid_solver, *transfers, smoothers, mpi_comm);
}

template class MultigridPreconditionerBase<2, float>;
template class MultigridPreconditionerBase<2, double>;
template class MultigridPreconditionerBase<2, double, double>;

template class MultigridPreconditionerBase<3, float>;
template class MultigridPreconditionerBase<3, double>;
template class MultigridPreconditionerBase<3, double, double>;

} // namespace ExaDG
