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

#ifndef INCLUDE_FUNCTIONALITIES_ENUM_TYPES_H_
#define INCLUDE_FUNCTIONALITIES_ENUM_TYPES_H_

#include <string>

namespace ExaDG
{
/**************************************************************************************/
/*                                                                                    */
/*                                         MESH                                       */
/*                                                                                    */
/**************************************************************************************/

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
 * deal.II provides different multigrid variants, which is to be specified by this parameter.
 */
enum class MultigridVariant
{
  LocalSmoothing,
  GlobalCoarsening
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

/*
 *  Mapping type (polynomial degree)
 */
enum class MappingType
{
  Affine,
  Quadratic,
  Cubic,
  Isoparametric
};

} // namespace ExaDG

#endif /* INCLUDE_FUNCTIONALITIES_ENUM_TYPES_H_ */
