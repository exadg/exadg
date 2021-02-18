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

#ifndef INCLUDE_EXADG_POSTPROCESSOR_WRITE_OUTPUT_H_
#define INCLUDE_EXADG_POSTPROCESSOR_WRITE_OUTPUT_H_

// C/C++
#include <fstream>

// deal.II
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>

namespace ExaDG
{
using namespace dealii;

template<int dim>
void
write_surface_mesh(Triangulation<dim> const & triangulation,
                   Mapping<dim> const &       mapping,
                   unsigned int               n_subdivisions,
                   std::string const &        folder,
                   std::string const &        file,
                   unsigned int const         counter,
                   MPI_Comm const &           mpi_comm)
{
  // write surface mesh only
  DataOutFaces<dim, DoFHandler<dim>> data_out_surface(true /*surface only*/);
  data_out_surface.attach_triangulation(triangulation);
  data_out_surface.build_patches(mapping, n_subdivisions);
  data_out_surface.write_vtu_with_pvtu_record(folder, file + "_surface", counter, mpi_comm, 4);
}

template<int dim>
void
write_boundary_IDs(Triangulation<dim> const & triangulation,
                   std::string const &        folder,
                   std::string const &        file,
                   MPI_Comm const &           mpi_communicator)
{
  unsigned int const rank    = Utilities::MPI::this_mpi_process(mpi_communicator);
  unsigned int const n_ranks = Utilities::MPI::n_mpi_processes(mpi_communicator);

  unsigned int const n_digits = static_cast<int>(std::ceil(std::log10(std::fabs(n_ranks))));

  std::string filename =
    folder + file + "_boundary_IDs" + "." + Utilities::int_to_string(rank, n_digits) + ".vtk";
  std::ofstream output(filename);

  GridOut           grid_out;
  GridOutFlags::Vtk flags;
  flags.output_cells         = false;
  flags.output_faces         = true;
  flags.output_edges         = false;
  flags.output_only_relevant = false;
  grid_out.set_flags(flags);
  grid_out.write_vtk(triangulation, output);
}

template<int dim>
void
write_grid(Triangulation<dim> const & triangulation,
           std::string const &        folder,
           std::string const &        file)
{
  std::string filename = folder + file + "_grid";

  GridOut grid_out;
  grid_out.write_mesh_per_processor_as_vtu(triangulation, filename);
}

} // namespace ExaDG

#endif /* INCLUDE_EXADG_POSTPROCESSOR_WRITE_OUTPUT_H_ */
