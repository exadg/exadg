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

// C/C++
#include <string>

// deal.II
#include <deal.II/grid/tria.h>

// ExaDG
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
/*
 * Triangulation type
 */
enum class TriangulationType
{
  Serial,
  Distributed,
  FullyDistributed
};

/*
 * Element type
 */
enum class ElementType
{
  Hypercube,
  Simplex
};

/*
 * Partitioning type (relevant for fully-distributed triangulation)
 */
enum class PartitioningType
{
  Metis,
  z_order
};

/**
 * Returns the type of elements, where we currently only allow triangulations consisting of the same
 * type of elements.
 */
template<int dim>
ElementType
get_element_type(dealii::Triangulation<dim> const & tria)
{
  if(tria.all_reference_cells_are_simplex())
  {
    return ElementType::Simplex;
  }
  else if(tria.all_reference_cells_are_hyper_cube())
  {
    return ElementType::Hypercube;
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Invalid parameter element_type."));
    return ElementType::Hypercube;
  }
}

struct GridData
{
  GridData()
    : triangulation_type(TriangulationType::Distributed),
      element_type(ElementType::Hypercube),
      partitioning_type(PartitioningType::Metis),
      n_refine_global(0),
      file_name(),
      create_coarse_triangulations(false)
  {
  }

  void
  check() const
  {
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    print_parameter(pcout, "Triangulation type", triangulation_type);

    print_parameter(pcout, "Element type", element_type);

    if(triangulation_type == TriangulationType::FullyDistributed)
      print_parameter(pcout, "Partitioning type (fully-distributed)", partitioning_type);

    print_parameter(pcout, "Number of global refinements", n_refine_global);

    if(not file_name.empty())
      print_parameter(pcout, "Grid file name", file_name);

    print_parameter(pcout, "Create coarse triangulations", create_coarse_triangulations);
  }

  TriangulationType triangulation_type;

  ElementType element_type;

  // only relevant for TriangulationType::FullyDistributed
  PartitioningType partitioning_type;

  unsigned int n_refine_global;

  // path to a grid file
  // the filename needs to include a proper filename ending/extension so that we can internally
  // deduce the correct type of the file format
  std::string file_name;

  // In case of a hypercube mesh that is globally refined, i.e. without hanging nodes, the fine
  // triangulation can be used for all multigrid h-levels without the need to create coarse
  // triangulations explicitly. Hence, this parameter is typically set to false for globally-refined
  // hypercube meshes.
  // Nevertheless, it is possible to set this parameter to true for globally-refined hypercube
  // meshes. In that case, the coarse triangulations are created explicitly for use in
  // h-multigrid methods.
  // This parameter needs to be set to true if one wants to use h-multigrid methods for
  // locally-refined hypercube meshes or non-hypercube meshes.
  bool create_coarse_triangulations;
};

} // namespace ExaDG



#endif /* INCLUDE_EXADG_GRID_GRID_DATA_H_ */
