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

// deal.II
#include <deal.II/base/parameter_handler.h>

// ExaDG
#include <exadg/grid/enum_types.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
struct GridData
{
  GridData()
    : triangulation_type(TriangulationType::Distributed),
      element_type(ElementType::Hypercube),
      partitioning_type(PartitioningType::Metis),
      n_refine_global(0),
      n_subdivisions_1d_hypercube(1),
      create_coarse_triangulations(false),
      mapping_degree(1),
      file_name()
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

    print_parameter(pcout, "Element type", enum_to_string(element_type));

    print_parameter(pcout, "Create coarse triangulations", create_coarse_triangulations);

    if(triangulation_type == TriangulationType::FullyDistributed)
      print_parameter(pcout,
                      "Partitioning type (fully-distributed)",
                      enum_to_string(partitioning_type));

    print_parameter(pcout, "Global refinements", n_refine_global);

    print_parameter(pcout, "Subdivisions hypercube", n_subdivisions_1d_hypercube);

    print_parameter(pcout, "Mapping degree", mapping_degree);
  }

  void
  add_parameters(dealii::ParameterHandler & prm, std::string const & subsection_name = "Grid")
  {
    // clang-format off
    prm.enter_subsection(subsection_name);
      prm.add_parameter("FileName", file_name, "External input grid file.");
    prm.leave_subsection();
    // clang-format on
  }

  TriangulationType triangulation_type;

  ElementType element_type;

  PartitioningType partitioning_type;

  unsigned int n_refine_global;

  // only relevant for hypercube geometry/mesh
  unsigned int n_subdivisions_1d_hypercube;

  // this parameter needs to be activated to use global-coarsening multigrid
  // (which needs all the coarser triangulations)
  bool create_coarse_triangulations;

  unsigned int mapping_degree;

  // path to a grid file
  // the filename needs to include a proper filename ending/extension so that we can internally
  // deduce the correct type of file format
  std::string file_name;
};

} // namespace ExaDG



#endif /* INCLUDE_EXADG_GRID_GRID_DATA_H_ */
