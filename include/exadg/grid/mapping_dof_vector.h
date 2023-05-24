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

#ifndef INCLUDE_FUNCTIONALITIES_MESH_H_
#define INCLUDE_FUNCTIONALITIES_MESH_H_

// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_cache.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

// ExaDG
#include <exadg/grid/grid_data.h>
#include <exadg/solvers_and_preconditioners/multigrid/transfers/mg_transfer_global_coarsening.h>

namespace ExaDG
{
/**
 * A mapping class based on dealii::MappingQCache equipped with practical interfaces that can be
 * used to initialize the mapping.
 *
 * The two functions fill_grid_coordinates_vector() and initialize_mapping_q_cache() should become
 * member functions of dealii::MappingQCache in dealii, rendering this class superfluous in ExaDG.
 */
template<int dim, typename Number>
class MappingDoFVector : public dealii::MappingQCache<dim>
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  /**
   * Constructor.
   */
  MappingDoFVector(unsigned int const mapping_degree_q_cache)
    : dealii::MappingQCache<dim>(mapping_degree_q_cache)
  {
    hierarchic_to_lexicographic_numbering =
      dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(mapping_degree_q_cache);
    lexicographic_to_hierarchic_numbering =
      dealii::Utilities::invert_permutation(hierarchic_to_lexicographic_numbering);
  }

  /**
   * Destructor.
   */
  virtual ~MappingDoFVector()
  {
  }

  /**
   * Extract the grid coordinates of the current mesh configuration described by the
   * dealii::MappingQCache object and fill a dof-vector given a corresponding dealii::DoFHandler
   * object.
   */
  void
  fill_grid_coordinates_vector(VectorType &                    grid_coordinates,
                               dealii::DoFHandler<dim> const & dof_handler) const
  {
    // use the deformed state described by the dealii::MappingQCache object (*this)
    fill_grid_coordinates_vector(*this, grid_coordinates, dof_handler);
  }

  /**
   * Extract the grid coordinates for a given external mapping and fill a
   * dof-vector given a corresponding dealii::DoFHandler object.
   */
  void
  fill_grid_coordinates_vector(dealii::Mapping<dim> const &    mapping,
                               VectorType &                    grid_coordinates,
                               dealii::DoFHandler<dim> const & dof_handler) const
  {
    if(grid_coordinates.size() != dof_handler.n_dofs())
    {
      dealii::IndexSet relevant_dofs_grid;
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs_grid);
      grid_coordinates.reinit(dof_handler.locally_owned_dofs(),
                              relevant_dofs_grid,
                              dof_handler.get_communicator());
    }
    else
    {
      grid_coordinates = 0;
    }

    dealii::FiniteElement<dim> const & fe = dof_handler.get_fe();

    // Set up dealii::FEValues with base element and the Gauss-Lobatto quadrature to
    // reduce setup cost, as we only use the geometry information (this means
    // we need to call fe_values.reinit(cell) with Triangulation::cell_iterator
    // rather than dealii::DoFHandler::cell_iterator).
    dealii::FE_Nothing<dim> dummy_fe;
    dealii::FEValues<dim>   fe_values(mapping,
                                    dummy_fe,
                                    dealii::QGaussLobatto<dim>(fe.degree + 1),
                                    dealii::update_quadrature_points);

    std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);

    std::vector<std::array<unsigned int, dim>> component_to_system_index(
      fe.base_element(0).dofs_per_cell);

    if(fe.dofs_per_vertex > 0) // dealii::FE_Q
    {
      for(unsigned int i = 0; i < fe.dofs_per_cell; ++i)
      {
        component_to_system_index
          [hierarchic_to_lexicographic_numbering[fe.system_to_component_index(i).second]]
          [fe.system_to_component_index(i).first] = i;
      }
    }
    else // dealii::FE_DGQ
    {
      for(unsigned int i = 0; i < fe.dofs_per_cell; ++i)
      {
        component_to_system_index[fe.system_to_component_index(i).second]
                                 [fe.system_to_component_index(i).first] = i;
      }
    }

    for(auto const & cell : dof_handler.active_cell_iterators())
    {
      if(not cell->is_artificial())
      {
        fe_values.reinit(typename dealii::Triangulation<dim>::cell_iterator(cell));
        cell->get_dof_indices(dof_indices);
        for(unsigned int i = 0; i < fe_values.n_quadrature_points; ++i)
        {
          dealii::Point<dim> const point = fe_values.quadrature_point(i);
          for(unsigned int d = 0; d < dim; ++d)
          {
            if(grid_coordinates.get_partitioner()->in_local_range(
                 dof_indices[component_to_system_index[i][d]]))
            {
              grid_coordinates(dof_indices[component_to_system_index[i][d]]) = point[d];
            }
          }
        }
      }
    }

    grid_coordinates.update_ghost_values();
  }

  /**
   * Initializes the dealii::MappingQCache object by providing a mapping that describes an
   * undeformed reference configuration and a displacement dof-vector (with a corresponding
   * dealii::DoFHandler object) that describes the displacement of the mesh compared to that
   * reference configuration. There are two special cases:
   *
   * If the mapping pointer is invalid, this implies that the reference coordinates are interpreted
   * as zero, i.e., the displacement vector describes the absolute coordinates of the grid points.
   *
   * If the displacement_vector is empty or uninitialized, this implies that no displacements will
   * be added to the grid coordinates of the reference configuration described by mapping.
   */
  void
  initialize_mapping_q_cache(std::shared_ptr<dealii::Mapping<dim> const> mapping,
                             VectorType const &                          displacement_vector,
                             dealii::DoFHandler<dim> const &             dof_handler)
  {
    AssertThrow(dealii::MultithreadInfo::n_threads() == 1, dealii::ExcNotImplemented());

    VectorType displacement_vector_ghosted;
    if(dof_handler.n_dofs() > 0 and displacement_vector.size() == dof_handler.n_dofs())
    {
      dealii::IndexSet locally_relevant_dofs;
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
      displacement_vector_ghosted.reinit(dof_handler.locally_owned_dofs(),
                                         locally_relevant_dofs,
                                         dof_handler.get_communicator());
      displacement_vector_ghosted.copy_locally_owned_data_from(displacement_vector);
      displacement_vector_ghosted.update_ghost_values();
    }

    std::shared_ptr<dealii::FEValues<dim>> fe_values;

    dealii::FE_Nothing<dim> fe_nothing;
    if(mapping.get() != 0)
    {
      fe_values =
        std::make_shared<dealii::FEValues<dim>>(*mapping,
                                                fe_nothing,
                                                dealii::QGaussLobatto<dim>(this->get_degree() + 1),
                                                dealii::update_quadrature_points);
    }

    // update mapping according to mesh deformation described by displacement vector
    this->initialize(
      dof_handler.get_triangulation(),
      [&](const typename dealii::Triangulation<dim>::cell_iterator & cell_tria)
        -> std::vector<dealii::Point<dim>> {
        unsigned int const scalar_dofs_per_cell =
          dealii::Utilities::pow(this->get_degree() + 1, dim);

        std::vector<dealii::Point<dim>> grid_coordinates(scalar_dofs_per_cell);

        if(mapping.get() != 0)
        {
          fe_values->reinit(cell_tria);
          // extract displacement and add to original position
          for(unsigned int i = 0; i < scalar_dofs_per_cell; ++i)
          {
            grid_coordinates[i] =
              fe_values->quadrature_point(this->hierarchic_to_lexicographic_numbering[i]);
          }
        }

        // if this function is called with an empty dof-vector, this indicates that the
        // displacements are zero and the points do not have to be moved
        if(dof_handler.n_dofs() > 0 and displacement_vector.size() > 0 and
           cell_tria->is_active() and not(cell_tria->is_artificial()))
        {
          dealii::FiniteElement<dim> const & fe = dof_handler.get_fe();
          AssertThrow(fe.element_multiplicity(0) == dim,
                      dealii::ExcMessage("Expected finite element with dim components."));

          typename dealii::DoFHandler<dim>::cell_iterator cell(&cell_tria->get_triangulation(),
                                                               cell_tria->level(),
                                                               cell_tria->index(),
                                                               &dof_handler);

          std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);
          cell->get_dof_indices(dof_indices);

          for(unsigned int i = 0; i < dof_indices.size(); ++i)
          {
            std::pair<unsigned int, unsigned int> const id = fe.system_to_component_index(i);

            if(fe.dofs_per_vertex > 0) // dealii::FE_Q
            {
              grid_coordinates[id.second][id.first] += displacement_vector_ghosted(dof_indices[i]);
            }
            else // dealii::FE_DGQ
            {
              grid_coordinates[this->lexicographic_to_hierarchic_numbering[id.second]][id.first] +=
                displacement_vector_ghosted(dof_indices[i]);
            }
          }
        }

        return grid_coordinates;
      });
  }

  std::vector<unsigned int> hierarchic_to_lexicographic_numbering;
  std::vector<unsigned int> lexicographic_to_hierarchic_numbering;
};


namespace MappingTools
{
/**
 * Use this function to initialize the mapping for use in multigrid.
 *
 * The second argument mapping_q_cache describes the mapping of the fine triangulation provided as
 * third argument.
 *
 * This function only takes the grid coordinates described by mapping_q_cache without adding
 * displacements in order to initialize mapping_multigrid for all multigrid h-levels.
 *
 * Prior to calling this function, the vector of coarse_mappings must have the correct size
 * according to the number of h-multigrid levels (excluding the finest level). The first entry
 * corresponds to the coarsest triangulation, the last element to the level below the fine
 * triangulation.
 */
template<int dim, typename Number>
void
initialize_coarse_mappings(
  std::vector<std::shared_ptr<dealii::Mapping<dim> const>> & coarse_mappings,
  std::shared_ptr<dealii::MappingQCache<dim> const> const &  mapping_q_cache,
  dealii::Triangulation<dim> const &                         triangulation)
{
  AssertThrow(dealii::MultithreadInfo::n_threads() == 1, dealii::ExcNotImplemented());

  std::shared_ptr<MappingDoFVector<dim, Number>> mapping_dof_vector =
    std::make_shared<MappingDoFVector<dim, Number>>(mapping_q_cache->get_degree());

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  dealii::FESystem<dim>   fe(dealii::FE_Q<dim>(mapping_q_cache->get_degree()), dim);
  dealii::DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();
  VectorType grid_coordinates_fine_level;
  mapping_dof_vector->fill_grid_coordinates_vector(*mapping_q_cache,
                                                   grid_coordinates_fine_level,
                                                   dof_handler);

  // we have to project the solution onto all coarse levels of the triangulation
  dealii::MGLevelObject<VectorType> grid_coordinates_all_levels,
    grid_coordinates_all_levels_ghosted;
  unsigned int const n_levels = triangulation.n_global_levels();
  grid_coordinates_all_levels.resize(0, n_levels - 1);
  grid_coordinates_all_levels_ghosted.resize(0, n_levels - 1);

  dealii::MGTransferMatrixFree<dim, Number> transfer;
  transfer.build(dof_handler);
  transfer.interpolate_to_mg(dof_handler, grid_coordinates_all_levels, grid_coordinates_fine_level);

  // ghosting
  for(unsigned int level = 0; level < n_levels; level++)
  {
    dealii::IndexSet relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, relevant_dofs);

    grid_coordinates_all_levels_ghosted[level].reinit(
      dof_handler.locally_owned_mg_dofs(level),
      relevant_dofs,
      grid_coordinates_all_levels[level].get_mpi_communicator());

    grid_coordinates_all_levels_ghosted[level].copy_locally_owned_data_from(
      grid_coordinates_all_levels[level]);

    grid_coordinates_all_levels_ghosted[level].update_ghost_values();
  }

  AssertThrow(fe.element_multiplicity(0) == dim,
              dealii::ExcMessage("Expected finite element with dim components."));

  // update mapping for all multigrid levels according to grid coordinates described by static
  // mapping
  mapping_dof_vector->initialize(
    dof_handler.get_triangulation(),
    [&](const typename dealii::Triangulation<dim>::cell_iterator & cell_tria)
      -> std::vector<dealii::Point<dim>> {
      unsigned int const level = cell_tria->level();

      typename dealii::DoFHandler<dim>::cell_iterator cell(&cell_tria->get_triangulation(),
                                                           level,
                                                           cell_tria->index(),
                                                           &dof_handler);

      unsigned int const scalar_dofs_per_cell = dealii::Utilities::pow(fe.degree + 1, dim);

      std::vector<dealii::Point<dim>> grid_coordinates(scalar_dofs_per_cell);

      if(cell->level_subdomain_id() != dealii::numbers::artificial_subdomain_id)
      {
        std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);
        cell->get_mg_dof_indices(dof_indices);

        for(unsigned int i = 0; i < dof_indices.size(); ++i)
        {
          std::pair<unsigned int, unsigned int> const id = fe.system_to_component_index(i);

          if(fe.dofs_per_vertex > 0) // dealii::FE_Q
          {
            grid_coordinates[id.second][id.first] =
              grid_coordinates_all_levels_ghosted[level](dof_indices[i]);
          }
          else // dealii::FE_DGQ
          {
            grid_coordinates[mapping_dof_vector->lexicographic_to_hierarchic_numbering[id.second]]
                            [id.first] = grid_coordinates_all_levels_ghosted[level](dof_indices[i]);
          }
        }
      }

      return grid_coordinates;
    });


  // let all coarse grid mappings point to the MappingDoFVector object
  for(unsigned int h_level = 0; h_level < coarse_mappings.size(); ++h_level)
  {
    // using the same Mapping object for all multigrid h-levels is some form of legacy code. The
    // class dealii::MatrixFree can internally extract the coarse-level mapping information
    // (provided through the fine-level Mapping object).
    coarse_mappings[h_level] = mapping_dof_vector;
  }
}

/**
 * Free function used to initialize the mapping for all multigrid h-levels of
 * coarse_triangulations.
 *
 * The second argument mapping_q_cache describes the mapping of the fine triangulation.
 *
 * This function only takes the grid coordinates described by mapping_q_cache without adding
 * displacements in order to initialize the mapping for all multigrid h-levels.
 *
 * Prior to calling this function, the vector of coarse_mappings must have the correct size
 * according to the number of h-multigrid levels (excluding the finest level). The first entry
 * corresponds to the coarsest triangulation, the last element to the level below the fine
 * triangulation.
 */
template<int dim, typename Number>
void
initialize_coarse_mappings(
  std::vector<std::shared_ptr<dealii::Mapping<dim> const>> &             coarse_mappings,
  std::shared_ptr<dealii::MappingQCache<dim> const> const &              mapping_q_cache,
  std::shared_ptr<dealii::Triangulation<dim> const> const &              fine_triangulation,
  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> const & coarse_triangulations)
{
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  dealii::FESystem<dim>                fe(dealii::FE_Q<dim>(mapping_q_cache->get_degree()), dim);
  unsigned int const                   n_h_levels = coarse_triangulations.size() + 1;
  std::vector<dealii::DoFHandler<dim>> dof_handlers_all_levels(n_h_levels);
  std::vector<dealii::AffineConstraints<Number>> constraints_all_levels(n_h_levels);
  for(unsigned int h_level = 0; h_level < n_h_levels; ++h_level)
  {
    if(h_level == n_h_levels - 1)
      dof_handlers_all_levels[h_level].reinit(*fine_triangulation);
    else
      dof_handlers_all_levels[h_level].reinit(*coarse_triangulations[h_level]);
    dof_handlers_all_levels[h_level].distribute_dofs(fe);
    // constraints are irrelevant for interpolation
    constraints_all_levels[h_level].close();
  }
  dealii::MGLevelObject<dealii::MGTwoLevelTransfer<dim, VectorType>> transfers(0, n_h_levels - 1);
  for(unsigned int h_level = 1; h_level < n_h_levels; ++h_level)
  {
    transfers[h_level].reinit_geometric_transfer(dof_handlers_all_levels[h_level],
                                                 dof_handlers_all_levels[h_level - 1],
                                                 constraints_all_levels[h_level],
                                                 constraints_all_levels[h_level - 1]);
  }

  // a function that initializes the dof-vector for a given level and dof_handler
  const std::function<void(unsigned int const, VectorType &)> initialize_dof_vector =
    [&](unsigned int const h_level, VectorType & vector) {
      dealii::IndexSet locally_relevant_dofs;
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handlers_all_levels[h_level],
                                                      locally_relevant_dofs);
      vector.reinit(dof_handlers_all_levels[h_level].locally_owned_dofs(),
                    locally_relevant_dofs,
                    dof_handlers_all_levels[h_level].get_communicator());
    };

  dealii::MGTransferGlobalCoarsening<dim, VectorType> mg_transfer_global_coarsening(
    transfers, initialize_dof_vector);

  dealii::MGLevelObject<VectorType> grid_coordinates_all_levels(0, n_h_levels - 1);

  std::shared_ptr<MappingDoFVector<dim, Number>> mapping_dof_vector =
    std::make_shared<MappingDoFVector<dim, Number>>(mapping_q_cache->get_degree());

  // get dof-vector with grid coordinates from the finest h-level
  mapping_dof_vector->fill_grid_coordinates_vector(*mapping_q_cache,
                                                   grid_coordinates_all_levels[n_h_levels - 1],
                                                   dof_handlers_all_levels[n_h_levels - 1]);

  // Transfer grid coordinates to coarser h-levels.
  // The dealii::DoFHandler object will not be used for global coarsening.
  dealii::DoFHandler<dim> dof_handler_dummy;
  VectorType              grid_coordinates_fine_level(grid_coordinates_all_levels[n_h_levels - 1]);
  mg_transfer_global_coarsening.interpolate_to_mg(dof_handler_dummy,
                                                  grid_coordinates_all_levels,
                                                  grid_coordinates_fine_level);

  // initialize mapping for all coarse h-levels using the dof-vectors with grid coordinates
  AssertThrow(coarse_mappings.size() == n_h_levels - 1,
              dealii::ExcMessage(
                "coarse_mappings does not have correct size relative to coarse_triangulations."));
  for(unsigned int h_level = 0; h_level < coarse_mappings.size(); ++h_level)
  {
    std::shared_ptr<MappingDoFVector<dim, Number>> mapping_dof_vector =
      std::make_shared<MappingDoFVector<dim, Number>>(mapping_q_cache->get_degree());

    // grid_coordinates_all_levels describes absolute coordinates -> use an uninitialized mapping
    // in order to interpret grid_coordinates_all_levels as absolute coordinates and not as
    // displacements.
    std::shared_ptr<dealii::Mapping<dim> const> mapping_dummy;
    mapping_dof_vector->initialize_mapping_q_cache(mapping_dummy,
                                                   grid_coordinates_all_levels[h_level],
                                                   dof_handlers_all_levels[h_level]);

    coarse_mappings[h_level] = mapping_dof_vector;
  }
}

/**
 * This function unifies the two functions above as determined by the parameter multigrid_variant.
 * In all cases, a vector of coarse grid mappings is filled.
 *
 * Prior to calling this function, the vector of coarse_mappings must have the correct size
 * according to the number of h-multigrid levels (excluding the finest level). The first entry
 * corresponds to the coarsest triangulation, the last element to the level below the fine
 * triangulation.
 *
 * Depending on the type of geometric multigrid coarsening, only one of the triangulation arguments
 * will be used.
 *
 * If the fine_mapping provided as second argument is not of type dealii::MappingQCache, all coarse
 * grid mappings will simply point to the fine_mapping.
 */
template<int dim, typename Number>
void
initialize_coarse_mappings(
  std::vector<std::shared_ptr<dealii::Mapping<dim> const>> &             coarse_mappings,
  std::shared_ptr<dealii::Mapping<dim> const> const &                    fine_mapping,
  MultigridVariant const &                                               multigrid_variant,
  std::shared_ptr<dealii::Triangulation<dim> const> const &              fine_triangulation,
  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> const & coarse_triangulations)
{
  // We only need to explicitly compute the mapping for all multigrid h-levels if it is of type
  // dealii::MappingQCache (including MappingDoFVector as a derived class)
  std::shared_ptr<dealii::MappingQCache<dim> const> mapping_q_cache =
    std::dynamic_pointer_cast<dealii::MappingQCache<dim> const>(fine_mapping);

  if(mapping_q_cache.get() != 0)
  {
    if(multigrid_variant == MultigridVariant::GlobalCoarsening)
    {
      MappingTools::initialize_coarse_mappings<dim, Number>(coarse_mappings,
                                                            mapping_q_cache,
                                                            coarse_triangulations);
    }
    else if(multigrid_variant == MultigridVariant::LocalSmoothing)
    {
      MappingTools::initialize_coarse_mappings<dim, Number>(coarse_mappings,
                                                            mapping_q_cache,
                                                            *fine_triangulation);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("not implemented."));
    }
  }
  else
  {
    for(unsigned int h_level = 0; h_level < coarse_mappings.size(); ++h_level)
    {
      coarse_mappings[h_level] = fine_mapping;
    }
  }
}


} // namespace MappingTools

} // namespace ExaDG

#endif /* INCLUDE_FUNCTIONALITIES_MESH_H_ */
