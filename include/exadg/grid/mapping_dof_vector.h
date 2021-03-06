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
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_cache.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

namespace ExaDG
{
using namespace dealii;

/**
 * A mapping class based on MappingQCache equipped with practical interfaces that can be used to
 * initialize the mapping.
 */
template<int dim, typename Number>
class MappingDoFVector : public MappingQCache<dim>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  /**
   * Constructor.
   */
  MappingDoFVector(std::shared_ptr<Mapping<dim> const> mapping,
                   unsigned int const                  mapping_degree_q_cache,
                   Triangulation<dim> const &          triangulation)
    : MappingQCache<dim>(mapping_degree_q_cache),
      mapping(mapping),
      triangulation(triangulation),
      mpi_comm(triangulation.get_communicator())
  {
    hierarchic_to_lexicographic_numbering =
      FETools::hierarchic_to_lexicographic_numbering<dim>(mapping_degree_q_cache);
    lexicographic_to_hierarchic_numbering =
      Utilities::invert_permutation(hierarchic_to_lexicographic_numbering);

    // Make sure that MappingQCache is initialized correctly. An empty dof-vector is used and,
    // hence, no displacements are added to the reference configuration described by the static
    // mapping.
    FESystem<dim>   fe(FE_Q<dim>(this->get_degree()), dim);
    DoFHandler<dim> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);
    VectorType displacement_vector;
    initialize(displacement_vector, dof_handler);
  }

  /**
   * Destructor.
   */
  virtual ~MappingDoFVector()
  {
  }

  /**
   * Extract the grid coordinates of the current mesh configuration described by the MappingQCache
   * object and fill a dof-vector given a corresponding DoFHandler object.
   */
  void
  fill_grid_coordinates_vector(VectorType & grid_coordinates, DoFHandler<dim> const & dof_handler)
  {
    // use the deformed state described by the MappingQCache object (*this)
    fill_grid_coordinates_vector(*this, grid_coordinates, dof_handler);
  }

  /**
   * Extract the grid coordinates for a given external mapping and fill a
   * dof-vector given a corresponding DoFHandler object.
   */
  void
  fill_grid_coordinates_vector(Mapping<dim> const &    mapping,
                               VectorType &            grid_coordinates,
                               DoFHandler<dim> const & dof_handler)
  {
    if(grid_coordinates.size() != dof_handler.n_dofs())
    {
      IndexSet relevant_dofs_grid;
      DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs_grid);
      grid_coordinates.reinit(dof_handler.locally_owned_dofs(), relevant_dofs_grid, mpi_comm);
    }
    else
    {
      grid_coordinates = 0;
    }

    FiniteElement<dim> const & fe = dof_handler.get_fe();

    // Set up FEValues with base element and the Gauss-Lobatto quadrature to
    // reduce setup cost, as we only use the geometry information (this means
    // we need to call fe_values.reinit(cell) with Triangulation::cell_iterator
    // rather than DoFHandler::cell_iterator).
    FE_Nothing<dim> dummy_fe;
    FEValues<dim>   fe_values(mapping,
                            dummy_fe,
                            QGaussLobatto<dim>(fe.degree + 1),
                            update_quadrature_points);

    std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);

    std::vector<std::array<unsigned int, dim>> component_to_system_index(
      fe.base_element(0).dofs_per_cell);

    if(fe.dofs_per_vertex > 0) // FE_Q
    {
      for(unsigned int i = 0; i < fe.dofs_per_cell; ++i)
      {
        component_to_system_index
          [hierarchic_to_lexicographic_numbering[fe.system_to_component_index(i).second]]
          [fe.system_to_component_index(i).first] = i;
      }
    }
    else // FE_DGQ
    {
      for(unsigned int i = 0; i < fe.dofs_per_cell; ++i)
      {
        component_to_system_index[fe.system_to_component_index(i).second]
                                 [fe.system_to_component_index(i).first] = i;
      }
    }

    for(auto const & cell : dof_handler.active_cell_iterators())
    {
      if(!cell->is_artificial())
      {
        fe_values.reinit(typename Triangulation<dim>::cell_iterator(cell));
        cell->get_dof_indices(dof_indices);
        for(unsigned int i = 0; i < fe_values.n_quadrature_points; ++i)
        {
          Point<dim> const point = fe_values.quadrature_point(i);
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
   * Initializes the MappingQCache object by providing a displacement dof-vector (with a
   * corresponding DoFHandler object) that describes the displacement of the mesh compared to an
   * undeformed reference configuration. If the displacement dof-vector is empty or uninitialized,
   * this implies that no displacements will be added to the grid coordinates of the reference
   * configuration.
   */
  void
  initialize(VectorType const & displacement_vector, DoFHandler<dim> const & dof_handler)
  {
    AssertThrow(MultithreadInfo::n_threads() == 1, ExcNotImplemented());

    VectorType displacement_vector_ghosted;
    if(dof_handler.n_dofs() > 0 && displacement_vector.size() == dof_handler.n_dofs())
    {
      IndexSet locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
      displacement_vector_ghosted.reinit(dof_handler.locally_owned_dofs(),
                                         locally_relevant_dofs,
                                         dof_handler.get_communicator());
      displacement_vector_ghosted.copy_locally_owned_data_from(displacement_vector);
      displacement_vector_ghosted.update_ghost_values();
    }

    FE_Nothing<dim> fe_nothing;
    FEValues<dim>   fe_values(*mapping,
                            fe_nothing,
                            QGaussLobatto<dim>(this->get_degree() + 1),
                            update_quadrature_points);

    // update mapping according to mesh deformation described by displacement vector
    MappingQCache<dim>::initialize(
      dof_handler.get_triangulation(),
      [&](const typename Triangulation<dim>::cell_iterator & cell_tria) -> std::vector<Point<dim>> {
        unsigned int const scalar_dofs_per_cell = Utilities::pow(this->get_degree() + 1, dim);

        std::vector<Point<dim>> grid_coordinates(scalar_dofs_per_cell);

        fe_values.reinit(cell_tria);
        // extract displacement and add to original position
        for(unsigned int i = 0; i < scalar_dofs_per_cell; ++i)
        {
          grid_coordinates[i] =
            fe_values.quadrature_point(this->hierarchic_to_lexicographic_numbering[i]);
        }

        // if this function is called with an empty dof-vector, this indicates that the
        // displacements are zero and the points do not have to be moved
        if(dof_handler.n_dofs() > 0 && displacement_vector.size() > 0 && cell_tria->is_active() &&
           !cell_tria->is_artificial())
        {
          FiniteElement<dim> const & fe = dof_handler.get_fe();
          AssertThrow(fe.element_multiplicity(0) == dim,
                      ExcMessage("Expected finite element with dim components."));

          typename DoFHandler<dim>::cell_iterator cell(&cell_tria->get_triangulation(),
                                                       cell_tria->level(),
                                                       cell_tria->index(),
                                                       &dof_handler);

          std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
          cell->get_dof_indices(dof_indices);

          for(unsigned int i = 0; i < dof_indices.size(); ++i)
          {
            std::pair<unsigned int, unsigned int> const id = fe.system_to_component_index(i);

            if(fe.dofs_per_vertex > 0) // FE_Q
            {
              grid_coordinates[id.second][id.first] += displacement_vector_ghosted(dof_indices[i]);
            }
            else // FE_DGQ
            {
              grid_coordinates[this->lexicographic_to_hierarchic_numbering[id.second]][id.first] +=
                displacement_vector_ghosted(dof_indices[i]);
            }
          }
        }

        return grid_coordinates;
      });
  }

  /**
   * Use this function to initialize the mapping for use in multigrid with global refinement
   * transfer type. This function only takes the grid coordinates described by the static mapping
   * without adding displacements in order to initialize the MappingQCache object for all multigrid
   * levels.
   */
  void
  initialize_multigrid()
  {
    AssertThrow(MultithreadInfo::n_threads() == 1, ExcNotImplemented());

    // we have to project the solution onto all coarse levels of the triangulation
    MGLevelObject<VectorType> grid_coordinates_all_levels, grid_coordinates_all_levels_ghosted;
    unsigned int const        n_levels = triangulation.n_global_levels();
    grid_coordinates_all_levels.resize(0, n_levels - 1);
    grid_coordinates_all_levels_ghosted.resize(0, n_levels - 1);

    FESystem<dim>   fe(FE_Q<dim>(this->get_degree()), dim);
    DoFHandler<dim> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();
    VectorType grid_coordinates_fine_level;
    fill_grid_coordinates_vector(*this->mapping, grid_coordinates_fine_level, dof_handler);

    MGTransferMatrixFree<dim, Number> transfer;
    transfer.build(dof_handler);
    transfer.interpolate_to_mg(dof_handler,
                               grid_coordinates_all_levels,
                               grid_coordinates_fine_level);

    for(unsigned int level = 0; level < n_levels; level++)
    {
      IndexSet relevant_dofs;
      DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, relevant_dofs);

      grid_coordinates_all_levels_ghosted[level].reinit(
        dof_handler.locally_owned_mg_dofs(level),
        relevant_dofs,
        grid_coordinates_all_levels[level].get_mpi_communicator());

      grid_coordinates_all_levels_ghosted[level].copy_locally_owned_data_from(
        grid_coordinates_all_levels[level]);

      grid_coordinates_all_levels_ghosted[level].update_ghost_values();
    }

    AssertThrow(fe.element_multiplicity(0) == dim,
                ExcMessage("Expected finite element with dim components."));

    // update mapping for all multigrid levels according to grid coordinates described by static
    // mapping
    MappingQCache<dim>::initialize(
      dof_handler.get_triangulation(),
      [&](const typename Triangulation<dim>::cell_iterator & cell_tria) -> std::vector<Point<dim>> {
        unsigned int const level = cell_tria->level();

        typename DoFHandler<dim>::cell_iterator cell(&cell_tria->get_triangulation(),
                                                     level,
                                                     cell_tria->index(),
                                                     &dof_handler);

        unsigned int const scalar_dofs_per_cell = Utilities::pow(fe.degree + 1, dim);

        std::vector<Point<dim>> grid_coordinates(scalar_dofs_per_cell);

        if(cell->level_subdomain_id() != numbers::artificial_subdomain_id)
        {
          std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
          cell->get_mg_dof_indices(dof_indices);

          for(unsigned int i = 0; i < dof_indices.size(); ++i)
          {
            std::pair<unsigned int, unsigned int> const id = fe.system_to_component_index(i);

            if(fe.dofs_per_vertex > 0) // FE_Q
            {
              grid_coordinates[id.second][id.first] =
                grid_coordinates_all_levels_ghosted[level](dof_indices[i]);
            }
            else // FE_DGQ
            {
              grid_coordinates[this->lexicographic_to_hierarchic_numbering[id.second]][id.first] =
                grid_coordinates_all_levels_ghosted[level](dof_indices[i]);
            }
          }
        }

        return grid_coordinates;
      });
  }

protected:
  // static mapping describing undeformed state
  std::shared_ptr<Mapping<dim> const> mapping;

  Triangulation<dim> const & triangulation;

  std::vector<unsigned int> hierarchic_to_lexicographic_numbering;
  std::vector<unsigned int> lexicographic_to_hierarchic_numbering;

  // MPI communicator
  MPI_Comm const mpi_comm;
};

} // namespace ExaDG

#endif /* INCLUDE_FUNCTIONALITIES_MESH_H_ */
