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
#include <deal.II/base/revision.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria_base.h>

// ExaDG
#include <exadg/grid/grid.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
inline void
print_exadg_header(dealii::ConditionalOStream const & pcout)
{
  // clang-format off
  pcout << std::endl << std::endl << std::endl
  << print_horizontal_line() << std::endl
  << "                                                                                " << std::endl
  << "                ////////                      ///////   ////////                " << std::endl
  << "                ///                           ///  ///  ///                     " << std::endl
  << "                //////    ///  ///  ///////   ///  ///  /// ////                " << std::endl
  << "                ///         ////    //   //   ///  ///  ///  ///                " << std::endl
  << "                ////////  ///  ///  ///////// ///////   ////////                " << std::endl
  << "                                                                                " << std::endl
  << "               High-Order Discontinuous Galerkin for the Exa-Scale              " << std::endl
  << print_horizontal_line() << std::endl
  << std::endl;
  // clang-format on
}

// print MPI info
inline void
print_MPI_info(dealii::ConditionalOStream const & pcout, MPI_Comm const & mpi_comm)
{
  pcout << std::endl << "MPI info:" << std::endl << std::endl;
  print_parameter(pcout, "Number of processes", dealii::Utilities::MPI::n_mpi_processes(mpi_comm));
}

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
inline void
print_dealii_info(dealii::ConditionalOStream const & pcout)
{
  // clang-format off
  pcout << std::endl
        << "deal.II info:" << std::endl
        << std::endl
        << "  deal.II git version " << DEAL_II_GIT_SHORTREV << " on branch " << DEAL_II_GIT_BRANCH
        << std::endl;
  // clang-format on
}

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

  std::shared_ptr<dealii::MappingQ<dim>> mapping_q =
    std::dynamic_pointer_cast<dealii::MappingQ<dim>>(grid.mapping);
  if(mapping_q.get() != 0)
    print_parameter(pcout, "Mapping degree", mapping_q->get_degree());
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
    print_dealii_info(pcout);
    print_matrixfree_info<Number>(pcout);
  }

  print_MPI_info(pcout, mpi_comm);
}

} // namespace ExaDG

#endif /* INCLUDE_EXADG_UTILITIES_PRINT_GENERAL_INFOS_H_ */
