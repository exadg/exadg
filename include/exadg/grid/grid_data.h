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

#ifndef INCLUDE_EXADG_GRID_GRID_DATA_H_
#define INCLUDE_EXADG_GRID_GRID_DATA_H_

#include <exadg/grid/enum_types.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
struct GridData
{
  GridData()
    : triangulation_type(TriangulationType::Distributed),
      n_refine_global(0),
      n_subdivisions_1d_hypercube(1),
      mapping_degree(1)
  {
  }

  void
  check() const
  {
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    print_parameter(pcout, "Triangulation type", enum_to_string(triangulation_type));

    print_parameter(pcout, "Global refinements", n_refine_global);

    print_parameter(pcout, "Subdivisions hypercube", n_subdivisions_1d_hypercube);

    print_parameter(pcout, "Mapping degree", mapping_degree);
  }

  TriangulationType triangulation_type;

  unsigned int n_refine_global;

  // only relevant for hypercube geometry/mesh
  unsigned int n_subdivisions_1d_hypercube;

  unsigned int mapping_degree;

  // TODO: path to a grid file
  // std::string grid_file;
};

} // namespace ExaDG



#endif /* INCLUDE_EXADG_GRID_GRID_DATA_H_ */
