/*
 * print_general_infos.h
 *
 *  Created on: Feb 21, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_UTILITIES_PRINT_GENERAL_INFOS_H_
#define INCLUDE_UTILITIES_PRINT_GENERAL_INFOS_H_

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/revision.h>

#include "../utilities/print_functions.h"

namespace ExaDG
{
using namespace dealii;

// print MPI info
inline void
print_MPI_info(ConditionalOStream const & pcout, MPI_Comm const & mpi_comm)
{
  pcout << std::endl << "MPI info:" << std::endl << std::endl;
  print_parameter(pcout, "Number of processes", Utilities::MPI::n_mpi_processes(mpi_comm));
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
template<typename Number>
inline void
print_dealii_info(ConditionalOStream const & pcout)
{
  unsigned int const n_vect_doubles = VectorizedArray<Number>::size();
  unsigned int const n_vect_bits    = 8 * sizeof(Number) * n_vect_doubles;
  std::string const  vect_level     = Utilities::System::get_current_vectorization_level();
  std::string const  type           = get_type(Number());

  // clang-format off
  pcout << std::endl
        << "deal.II info:" << std::endl
        << std::endl
        << "  deal.II git version " << DEAL_II_GIT_SHORTREV << " on branch " << DEAL_II_GIT_BRANCH
        << std::endl
        << "  with vectorization level = " << DEAL_II_COMPILER_VECTORIZATION_LEVEL
        << std::endl;

  pcout << std::endl
        << "  Vectorization over "
        << n_vect_doubles << " " << type << " = " << n_vect_bits << " bits (" << vect_level << ")"
        << std::endl;
  // clang-format on
}

// print grid info
template<int dim>
inline void
print_grid_data(ConditionalOStream const &               pcout,
                unsigned int const                       n_refine_space,
                parallel::TriangulationBase<dim> const & triangulation)
{
  pcout << std::endl
        << "Generating grid for " << dim << "-dimensional problem:" << std::endl
        << std::endl;

  print_parameter(pcout, "Number of refinements", n_refine_space);
  print_parameter(pcout, "Number of cells", triangulation.n_global_active_cells());
}
} // namespace ExaDG

#endif /* INCLUDE_UTILITIES_PRINT_GENERAL_INFOS_H_ */
