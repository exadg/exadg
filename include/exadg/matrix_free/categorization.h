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

#ifndef OPERATOR_BASE_CATEGORIZATION_H
#define OPERATOR_BASE_CATEGORIZATION_H

// deal.II
#include <deal.II/grid/tria.h>

namespace ExaDG
{
namespace Categorization
{
/*
 * Adjust MatrixFree::AdditionalData such that
 *   1) cells which have the same boundary IDs for all faces are put into the
 *      same category
 *   2) cell based loops are enabled (incl. dealii::FEEvaluationBase::read_cell_data()
 *      for all neighboring cells)
 */
template<int dim, typename AdditionalData>
void
do_cell_based_loops(dealii::Triangulation<dim> const & tria,
                    AdditionalData &                   data,
                    unsigned int const level = dealii::numbers::invalid_unsigned_int)
{
  bool is_mg = (level != dealii::numbers::invalid_unsigned_int);

  // ... create list for the category of each cell
  if(is_mg)
    data.cell_vectorization_category.resize(std::distance(tria.begin(level), tria.end(level)));
  else
    data.cell_vectorization_category.resize(tria.n_active_cells());

  AssertThrow(tria.get_reference_cells().size() == 1,
              dealii::ExcMessage("No mixed meshes allowed."));
  unsigned int const n_faces_per_cell = tria.get_reference_cells()[0].n_faces();

  // ... setup scaling factor
  std::vector<unsigned int> factors(n_faces_per_cell);

  std::map<unsigned int, unsigned int> bid_map;
  for(unsigned int i = 0; i < tria.get_boundary_ids().size(); i++)
    bid_map[tria.get_boundary_ids()[i]] = i + 1;

  {
    unsigned int bids   = tria.get_boundary_ids().size() + 1;
    int          offset = 1;
    for(unsigned int i = 0; i < n_faces_per_cell; i++, offset = offset * bids)
      factors[i] = offset;
  }

  auto to_category = [&](auto & cell) {
    unsigned int c_num = 0;
    for(unsigned int i = 0; i < n_faces_per_cell; i++)
    {
      auto & face = *cell->face(i);
      if(face.at_boundary())
        c_num += factors[i] * bid_map[face.boundary_id()];
    }
    return c_num;
  };

  if(not is_mg)
  {
    for(auto cell = tria.begin_active(); cell != tria.end(); ++cell)
    {
      if(cell->is_locally_owned())
        data.cell_vectorization_category[cell->active_cell_index()] = to_category(cell);
    }
  }
  else
  {
    for(auto cell = tria.begin(level); cell != tria.end(level); ++cell)
    {
      if(cell->is_locally_owned_on_level())
        data.cell_vectorization_category[cell->index()] = to_category(cell);
    }
  }

  // ... finalize setup of matrix_free
  data.hold_all_faces_to_owned_cells        = true;
  data.cell_vectorization_categories_strict = true;
  data.mapping_update_flags_faces_by_cells =
    data.mapping_update_flags_inner_faces | data.mapping_update_flags_boundary_faces;
}

} // namespace Categorization
} // namespace ExaDG

#endif
