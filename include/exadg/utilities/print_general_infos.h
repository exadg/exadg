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

#ifndef INCLUDE_EXADG_UTILITIES_PRINT_GENERAL_INFOS_H_
#define INCLUDE_EXADG_UTILITIES_PRINT_GENERAL_INFOS_H_

// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria_base.h>
#include <deal.II/fe/mapping_q.h>

// ExaDG
#include <exadg/grid/grid.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
void
print_exadg_header(dealii::ConditionalOStream const & pcout);

// print MPI info
void
print_MPI_info(dealii::ConditionalOStream const & pcout, MPI_Comm const & mpi_comm);

template<typename Number>
inline std::string get_type(Number)
{
  return "unknown type";
}

inline std::string
get_type(float)
{
  return "float";
}

inline std::string
get_type(double)
{
  return "double";
}

// print deal.II info
void
print_dealii_info(dealii::ConditionalOStream const & pcout);

// print ExaDG info
void
print_exadg_info(dealii::ConditionalOStream const & pcout);

template<typename Number>
inline void
print_matrixfree_info(dealii::ConditionalOStream const & pcout)
{
  unsigned int const n_vect_doubles = dealii::VectorizedArray<Number>::size();
  unsigned int const n_vect_bits    = 8 * sizeof(Number) * n_vect_doubles;
  std::string const  vect_level     = dealii::Utilities::System::get_current_vectorization_level();
  std::string const  type           = get_type(Number());

  // clang-format off
  pcout << std::endl
        << "MatrixFree info:" << std::endl
        << std::endl
        << "  vectorization level = " << DEAL_II_COMPILER_VECTORIZATION_LEVEL
        << std::endl
        << "  Vectorization over "
        << n_vect_doubles << " " << type << " = " << n_vect_bits << " bits (" << vect_level << ")"
        << std::endl;
  // clang-format on
}

template<int dim>
inline void
print_grid_info(dealii::ConditionalOStream const & pcout, Grid<dim> const & grid)
{
  pcout << std::endl
        << "Generating grid for " << dim << "-dimensional problem:" << std::endl
        << std::endl;

  print_parameter(pcout, "Max. number of refinements", grid.triangulation->n_global_levels() - 1);
  print_parameter(pcout, "Number of cells", grid.triangulation->n_global_active_cells());
}

template<typename Number>
inline void
print_general_info(dealii::ConditionalOStream const & pcout,
                   MPI_Comm const &                   mpi_comm,
                   bool const                         is_test)
{
  print_exadg_header(pcout);

  if(not(is_test))
  {
    print_exadg_info(pcout);
    print_dealii_info(pcout);
    print_matrixfree_info<Number>(pcout);
  }

  print_MPI_info(pcout, mpi_comm);
}

} // namespace ExaDG

#endif /* INCLUDE_EXADG_UTILITIES_PRINT_GENERAL_INFOS_H_ */
