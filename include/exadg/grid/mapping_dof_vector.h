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
#include <exadg/solvers_and_preconditioners/multigrid/transfer.h>

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
class MappingDoFVector
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  /**
   * Constructor.
   */
  MappingDoFVector(dealii::Triangulation<dim> const & triangulation)
  {
    element_type = get_element_type(triangulation);

    if(element_type == ElementType::Hypercube)
    {
      // TODO
    }
    else if(element_type == ElementType::Simplex)
    {
      // TODO
    }
    else
    {
      AssertThrow(false,
                  dealii::ExcMessage(
                    "MappingDoFVector is currently not implemented for the given ElementType."));
    }
  }

  /**
   * Destructor.
   */
  virtual ~MappingDoFVector()
  {
  }

  /**
   * returns the deal.ii mapping object that describes the deformed mapping
   */
  std::shared_ptr<dealii::Mapping<dim> const>
  get_mapping() const
  {
    if(element_type == ElementType::Hypercube)
    {
      AssertThrow(mapping_q_cache.get(),
                  dealii::ExcMessage("Mapping object mapping_q_cache is not initialized."));

      return mapping_q_cache;
    }
    else if(element_type == ElementType::Simplex)
    {
      // TODO

      AssertThrow(false,
                  dealii::ExcMessage(
                    "MappingDoFVector is currently not implemented for ElementType::Simplex."));

      return mapping_q_cache;
    }
    else
    {
      AssertThrow(false,
                  dealii::ExcMessage(
                    "MappingDoFVector is currently not implemented for the given ElementType."));

      return mapping_q_cache;
    }
  }

  std::shared_ptr<dealii::MappingQCache<dim>>
  get_mapping_q_cache() const
  {
    if(element_type == ElementType::Hypercube)
    {
      AssertThrow(mapping_q_cache.get(),
                  dealii::ExcMessage("Mapping object mapping_q_cache is not initialized."));

      return mapping_q_cache;
    }
    else
    {
      AssertThrow(
        false,
        dealii::ExcMessage(
          "The function get_mapping_q_cache() may only be called for ElementType::Hypercube."));

      return mapping_q_cache;
    }
  }

  void
  create_mapping_q_cache(unsigned int const degree)
  {
    if(not mapping_q_cache.get())
    {
      mapping_q_cache = std::make_shared<dealii::MappingQCache<dim>>(degree);
    }
    else
    {
      AssertThrow(
        mapping_q_cache->get_degree() == degree,
        dealii::ExcMessage(
          "Cannot create MappingQCache(degree) because the object already exists with another degree."));
    }
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
    if(element_type == ElementType::Hypercube)
    {
      AssertThrow(mapping_q_cache.get(),
                  dealii::ExcMessage("Mapping object mapping_q_cache is not initialized."));

      // use the deformed state described by the dealii::MappingQCache object
      fill_grid_coordinates_vector(*mapping_q_cache, grid_coordinates, dof_handler);
    }
    else if(element_type == ElementType::Simplex)
    {
      // TODO

      AssertThrow(false,
                  dealii::ExcMessage(
                    "MappingDoFVector is currently not implemented for ElementType::Simplex."));
    }
    else
    {
      AssertThrow(false,
                  dealii::ExcMessage(
                    "MappingDoFVector is currently not implemented for the given ElementType."));
    }
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
    AssertThrow(get_element_type(dof_handler.get_triangulation()) == element_type,
                dealii::ExcMessage("MappingDoFVector detected inconsistent element types."));

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

    if(element_type == ElementType::Hypercube)
    {
      dealii::FiniteElement<dim> const & fe = dof_handler.get_fe();

      unsigned int dofs_per_cell = fe.base_element(0).dofs_per_cell;

      // The elements of this vector need to be in lexicographic ordering.
      std::vector<std::array<unsigned int, dim>> component_to_system_index(dofs_per_cell);

      if(fe.dofs_per_vertex > 0) // dealii::FE_Q
      {
        std::vector<unsigned int> hierarchic_to_lexicographic_numbering =
          dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(fe.degree);

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

      // Set up dealii::FEValues with FE_Nothing and the Gauss-Lobatto quadrature to
      // reduce setup cost, as we only use the geometry information (this means
      // we need to call fe_values.reinit(cell) with Triangulation::cell_iterator
      // rather than dealii::DoFHandler::cell_iterator).
      dealii::FE_Nothing<dim> fe_nothing;
      dealii::FEValues<dim>   fe_values(mapping,
                                      fe_nothing,
                                      dealii::QGaussLobatto<dim>(fe.degree + 1),
                                      dealii::update_quadrature_points);

      std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);

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
    }
    else if(element_type == ElementType::Simplex)
    {
      // TODO

      AssertThrow(false,
                  dealii::ExcMessage(
                    "MappingDoFVector is currently not implemented for ElementType::Simplex."));
    }
    else
    {
      AssertThrow(false,
                  dealii::ExcMessage(
                    "MappingDoFVector is currently not implemented for the given ElementType."));
    }

    grid_coordinates.update_ghost_values();
  }

  /**
   * Initializes the dealii::MappingQCache object by providing a mapping that describes an
   * undeformed reference configuration and a displacement dof-vector (with a corresponding
   * dealii::DoFHandler object) that describes the displacement of the mesh compared to that
   * reference configuration.
   *
   * If the mapping pointer is invalid, this implies that the reference coordinates are interpreted
   * as zero, i.e., the displacement vector describes the absolute coordinates of the grid points.
   */
  void
  initialize_mapping_from_dof_vector(std::shared_ptr<dealii::Mapping<dim> const> mapping,
                                     VectorType const &              displacement_vector,
                                     dealii::DoFHandler<dim> const & dof_handler)
  {
    AssertThrow(get_element_type(dof_handler.get_triangulation()) == element_type,
                dealii::ExcMessage("MappingDoFVector detected inconsistent element types."));

    AssertThrow(dof_handler.n_dofs() > 0 and displacement_vector.size() == dof_handler.n_dofs(),
                dealii::ExcMessage("Unitialized parameters displacement_vector or dof_handler."));

    if(element_type == ElementType::Hypercube)
    {
      unsigned int const degree = dof_handler.get_fe().degree;

      if(mapping_q_cache.get())
      {
        AssertThrow(
          degree == mapping_q_cache->get_degree(),
          dealii::ExcMessage(
            "Degree of finite element of dof_handler does not fit to the degree used to initialize MappingQCache."));
      }
      else
      {
        mapping_q_cache = std::make_shared<dealii::MappingQCache<dim>>(degree);
      }

      AssertThrow(dealii::MultithreadInfo::n_threads() == 1, dealii::ExcNotImplemented());

      VectorType displacement_vector_ghosted;
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

      // Set up dealii::FEValues with FE_Nothing and the Gauss-Lobatto quadrature to
      // reduce setup cost, as we only use the geometry information (this means
      // we need to call fe_values.reinit(cell) with Triangulation::cell_iterator
      // rather than dealii::DoFHandler::cell_iterator).
      dealii::FE_Nothing<dim> fe_nothing;

      if(mapping.get() != 0)
      {
        fe_values = std::make_shared<dealii::FEValues<dim>>(*mapping,
                                                            fe_nothing,
                                                            dealii::QGaussLobatto<dim>(degree + 1),
                                                            dealii::update_quadrature_points);
      }

      std::vector<unsigned int> hierarchic_to_lexicographic_numbering =
        dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(degree);
      std::vector<unsigned int> lexicographic_to_hierarchic_numbering =
        dealii::Utilities::invert_permutation(hierarchic_to_lexicographic_numbering);

      // take the grid coordinates described by mapping and add deformation described by
      // displacement vector
      mapping_q_cache->initialize(
        dof_handler.get_triangulation(),
        [&](const typename dealii::Triangulation<dim>::cell_iterator & cell_tria)
          -> std::vector<dealii::Point<dim>> {
          unsigned int dofs_per_cell = dof_handler.get_fe().base_element(0).dofs_per_cell;

          // dealii::MappingQCache::initialize() expects vector of points in hierarchical ordering
          std::vector<dealii::Point<dim>> grid_coordinates(dofs_per_cell);

          if(mapping.get() != 0)
          {
            fe_values->reinit(cell_tria);
            // extract displacement and add to original position
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              // access fe_values->quadrature_point() by lexicographic index
              grid_coordinates[i] =
                fe_values->quadrature_point(hierarchic_to_lexicographic_numbering[i]);
            }
          }

          if(cell_tria->is_active() and not(cell_tria->is_artificial()))
          {
            typename dealii::DoFHandler<dim>::cell_iterator cell(&cell_tria->get_triangulation(),
                                                                 cell_tria->level(),
                                                                 cell_tria->index(),
                                                                 &dof_handler);

            dealii::FiniteElement<dim> const & fe = dof_handler.get_fe();
            AssertThrow(fe.element_multiplicity(0) == dim,
                        dealii::ExcMessage("Expected finite element with dim components."));

            std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);
            cell->get_dof_indices(dof_indices);

            for(unsigned int i = 0; i < dof_indices.size(); ++i)
            {
              std::pair<unsigned int, unsigned int> const id = fe.system_to_component_index(i);

              if(fe.dofs_per_vertex > 0) // dealii::FE_Q
              {
                grid_coordinates[id.second][id.first] +=
                  displacement_vector_ghosted(dof_indices[i]);
              }
              else // dealii::FE_DGQ
              {
                grid_coordinates[lexicographic_to_hierarchic_numbering[id.second]][id.first] +=
                  displacement_vector_ghosted(dof_indices[i]);
              }
            }
          }

          return grid_coordinates;
        });
    }
    else if(element_type == ElementType::Simplex)
    {
      // TODO

      AssertThrow(false,
                  dealii::ExcMessage(
                    "MappingDoFVector is currently not implemented for ElementType::Simplex."));
    }
    else
    {
      AssertThrow(false,
                  dealii::ExcMessage(
                    "MappingDoFVector is currently not implemented for the given ElementType."));
    }
  }

protected:
  /**
   * For ElementType::Hypercube
   */
  std::shared_ptr<dealii::MappingQCache<dim>> mapping_q_cache;

private:
  ElementType element_type;
};


namespace MappingTools
{
/**
 * Use this function to initialize the coarse mappings for use in multigrid.
 *
 * The second argument describes the mapping of the fine triangulation.
 *
 * This function only takes the grid coordinates described by the fine mapping without adding
 * displacements in order to initialize the coarse mappings for all multigrid h-levels.
 *
 * Prior to calling this function, the vector of coarse_mappings must have the correct size
 * according to the number of h-multigrid levels (excluding the finest level). The first entry
 * corresponds to the coarsest triangulation, the last element to the level below the fine
 * triangulation.
 */
template<int dim, typename Number>
void
initialize_coarse_mappings_from_mapping_dof_vector(
  std::vector<std::shared_ptr<MappingDoFVector<dim, Number>>> & coarse_mappings,
  unsigned int const                                            degree_coarse_mappings,
  std::shared_ptr<MappingDoFVector<dim, Number> const> const &  fine_mapping,
  dealii::Triangulation<dim> const &                            triangulation)
{
  if(get_element_type(triangulation) == ElementType::Hypercube)
  {
    AssertThrow(dealii::MultithreadInfo::n_threads() == 1, dealii::ExcNotImplemented());

    std::shared_ptr<dealii::MappingQCache<dim>> mapping_q_cache =
      fine_mapping->get_mapping_q_cache();
    AssertThrow(mapping_q_cache.get(),
                dealii::ExcMessage("Shared pointer mapping_q_cache is invalid."));

    dealii::FESystem<dim>   fe(dealii::FE_Q<dim>(degree_coarse_mappings), dim);
    dealii::DoFHandler<dim> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();

    std::shared_ptr<MappingDoFVector<dim, Number>> mapping_dof_vector_all_levels =
      std::make_shared<MappingDoFVector<dim, Number>>(triangulation);

    // fill a dof vector with grid coordinates of the fine level using degree_coarse_mappings
    typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;
    VectorType                                                 grid_coordinates_fine_level;

    {
      mapping_dof_vector_all_levels->fill_grid_coordinates_vector(*fine_mapping->get_mapping(),
                                                                  grid_coordinates_fine_level,
                                                                  dof_handler);
    }

    // project the solution onto all coarse levels of the triangulation using degree_coarse_mappings
    dealii::MGLevelObject<VectorType> grid_coordinates_all_levels,
      grid_coordinates_all_levels_ghosted;
    unsigned int const n_levels = triangulation.n_global_levels();
    grid_coordinates_all_levels.resize(0, n_levels - 1);
    grid_coordinates_all_levels_ghosted.resize(0, n_levels - 1);

    dealii::MGTransferMatrixFree<dim, Number> transfer;
    transfer.build(dof_handler);
    transfer.interpolate_to_mg(dof_handler,
                               grid_coordinates_all_levels,
                               grid_coordinates_fine_level);

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

    std::vector<unsigned int> hierarchic_to_lexicographic_numbering =
      dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(degree_coarse_mappings);
    std::vector<unsigned int> lexicographic_to_hierarchic_numbering =
      dealii::Utilities::invert_permutation(hierarchic_to_lexicographic_numbering);

    // Call the initialize() function of dealii::MappingQCache, which initializes the mapping for
    // all levels according to grid_coordinates_all_levels_ghosted.
    mapping_dof_vector_all_levels->create_mapping_q_cache(fe.degree);
    mapping_dof_vector_all_levels->get_mapping_q_cache()->initialize(
      dof_handler.get_triangulation(),
      [&](const typename dealii::Triangulation<dim>::cell_iterator & cell_tria)
        -> std::vector<dealii::Point<dim>> {
        unsigned int const level = cell_tria->level();

        typename dealii::DoFHandler<dim>::cell_iterator cell(&cell_tria->get_triangulation(),
                                                             level,
                                                             cell_tria->index(),
                                                             &dof_handler);

        AssertThrow(fe.element_multiplicity(0) == dim,
                    dealii::ExcMessage("Expected finite element with dim components."));

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
              grid_coordinates[lexicographic_to_hierarchic_numbering[id.second]][id.first] =
                grid_coordinates_all_levels_ghosted[level](dof_indices[i]);
            }
          }
        }

        return grid_coordinates;
      });


    AssertThrow(
      coarse_mappings.size() == n_levels - 1,
      dealii::ExcMessage(
        "coarse_mappings does not have correct size relative to the number of levels of the triangulation."));

    // Finally, let all coarse grid mappings point to the same MappingDoFVector object. Using the
    // same Mapping object for all multigrid h-levels is some form of legacy code. The class
    // dealii::MatrixFree can internally extract the coarse-level mapping information (provided
    // through the fine-level Mapping object).
    for(unsigned int h_level = 0; h_level < coarse_mappings.size(); ++h_level)
    {
      coarse_mappings[h_level] = mapping_dof_vector_all_levels;
    }
  }
  else if(get_element_type(triangulation) == ElementType::Simplex)
  {
    // TODO

    AssertThrow(false,
                dealii::ExcMessage(
                  "The function initialize_coarse_mappings_from_mapping_dof_vector() "
                  "is currently not implemented for ElementType::Simplex."));
  }
  else
  {
    AssertThrow(false,
                dealii::ExcMessage(
                  "The function initialize_coarse_mappings_from_mapping_dof_vector() "
                  "is currently not implemented for the given ElementType."));
  }
}

/**
 *
 * Use this function to initialize the coarse mappings for use in multigrid in case the multigrid
 * algorithm uses a separate triangulation object for each multigrid h-level.
 *
 * The second argument describes the mapping of the fine triangulation.
 *
 * This function only takes the grid coordinates described by the fine mapping without adding
 * displacements in order to initialize the coarse mappings for all multigrid h-levels.
 *
 * Prior to calling this function, the vector of coarse_mappings must have the correct size
 * according to the number of h-multigrid levels (excluding the finest level). The first entry
 * corresponds to the coarsest triangulation, the last element to the level below the fine
 * triangulation.
 */
template<int dim, typename Number>
void
initialize_coarse_mappings_from_mapping_dof_vector(
  std::vector<std::shared_ptr<MappingDoFVector<dim, Number>>> &          coarse_mappings,
  unsigned int const                                                     degree_coarse_mappings,
  std::shared_ptr<MappingDoFVector<dim, Number> const> const &           fine_mapping,
  std::shared_ptr<dealii::Triangulation<dim> const> const &              fine_triangulation,
  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> const & coarse_triangulations)
{
  if(get_element_type(*fine_triangulation) == ElementType::Hypercube)
  {
    std::shared_ptr<dealii::MappingQCache<dim>> mapping_q_cache =
      fine_mapping->get_mapping_q_cache();
    AssertThrow(mapping_q_cache.get(),
                dealii::ExcMessage("Shared pointer mapping_q_cache is invalid."));

    // setup dof-handlers and constraints for all levels using degree_coarse_mappings
    dealii::FESystem<dim>                fe(dealii::FE_Q<dim>(degree_coarse_mappings), dim);
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

    // fill a dof vector with grid coordinates of the fine level using degree_coarse_mappings
    typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;
    VectorType                                                 grid_coordinates_fine_level;

    {
      std::shared_ptr<MappingDoFVector<dim, Number>> mapping_dof_vector_fine_level =
        std::make_shared<MappingDoFVector<dim, Number>>(*fine_triangulation);

      auto const & dof_handler_fine_level = dof_handlers_all_levels[n_h_levels - 1];
      mapping_dof_vector_fine_level->fill_grid_coordinates_vector(*fine_mapping->get_mapping(),
                                                                  grid_coordinates_fine_level,
                                                                  dof_handler_fine_level);
    }

    // create transfer objects
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

    // Transfer grid coordinates to coarser h-levels.
    // The dealii::DoFHandler object will not be used for global coarsening.
    dealii::DoFHandler<dim>           dof_handler_dummy;
    dealii::MGLevelObject<VectorType> grid_coordinates_all_levels(0, n_h_levels - 1);
    mg_transfer_global_coarsening.interpolate_to_mg(dof_handler_dummy,
                                                    grid_coordinates_all_levels,
                                                    grid_coordinates_fine_level);

    // initialize mapping for all coarse h-levels using the dof-vectors with grid coordinates
    AssertThrow(coarse_mappings.size() == n_h_levels - 1,
                dealii::ExcMessage(
                  "coarse_mappings does not have correct size relative to coarse_triangulations."));

    for(unsigned int h_level = 0; h_level < coarse_mappings.size(); ++h_level)
    {
      coarse_mappings[h_level] =
        std::make_shared<MappingDoFVector<dim, Number>>(*coarse_triangulations[h_level]);

      // grid_coordinates_all_levels describes absolute coordinates -> use an uninitialized mapping
      // in order to interpret the grid coordinates vector as absolute coordinates and not as
      // displacements.
      std::shared_ptr<dealii::Mapping<dim> const> mapping_dummy;
      coarse_mappings[h_level]->initialize_mapping_from_dof_vector(
        mapping_dummy, grid_coordinates_all_levels[h_level], dof_handlers_all_levels[h_level]);
    }
  }
  else if(get_element_type(*fine_triangulation) == ElementType::Simplex)
  {
    // TODO

    AssertThrow(false,
                dealii::ExcMessage(
                  "The function initialize_coarse_mappings_from_mapping_dof_vector() "
                  "is currently not implemented for ElementType::Simplex."));
  }
  else
  {
    AssertThrow(false,
                dealii::ExcMessage(
                  "The function initialize_coarse_mappings_from_mapping_dof_vector() "
                  "is currently not implemented for the given ElementType."));
  }
}

/**
 * A function to initalize mappings for coarse multigrid h-levels in case the mapping on the fine
 * level is of type MappingDoFVector.
 *
 * This function unifies the two functions above depending on whether the multigrid algorithm uses
 * separate triangulation objects for the coarse triangulations or just a single triangulation
 * object for all levels. In the latter case, the last argument of this function remains unused.
 *
 * In all cases, a vector of coarse grid mappings is filled.
 *
 * Prior to calling this function, the vector of coarse_mappings must have the correct size
 * according to the number of h-multigrid levels (excluding the finest level). The first entry
 * corresponds to the coarsest triangulation, the last element to the level below the fine
 * triangulation.
 */
template<int dim, typename Number>
void
initialize_coarse_mappings(
  std::vector<std::shared_ptr<MappingDoFVector<dim, Number>>> &          coarse_mappings,
  unsigned int const                                                     degree_coarse_mappings,
  std::shared_ptr<MappingDoFVector<dim, Number> const> const &           fine_mapping,
  std::shared_ptr<dealii::Triangulation<dim> const> const &              fine_triangulation,
  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> const & coarse_triangulations)
{
  if(fine_mapping.get())
  {
    if(coarse_triangulations.size() > 0)
    {
      MappingTools::initialize_coarse_mappings_from_mapping_dof_vector<dim, Number>(
        coarse_mappings,
        degree_coarse_mappings,
        fine_mapping,
        fine_triangulation,
        coarse_triangulations);
    }
    else
    {
      MappingTools::initialize_coarse_mappings_from_mapping_dof_vector<dim, Number>(
        coarse_mappings, degree_coarse_mappings, fine_mapping, *fine_triangulation);
    }
  }
}


} // namespace MappingTools

} // namespace ExaDG

#endif /* INCLUDE_FUNCTIONALITIES_MESH_H_ */
