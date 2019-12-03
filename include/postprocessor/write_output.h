/*
 * write_output.h
 *
 *  Created on: Oct 11, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_POSTPROCESSOR_WRITE_OUTPUT_H_
#define INCLUDE_POSTPROCESSOR_WRITE_OUTPUT_H_

#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>

using namespace dealii;

template<int dim>
void
write_surface_mesh(Triangulation<dim> const & triangulation,
                   Mapping<dim> const &       mapping,
                   unsigned int               n_subdivisions,
                   std::string const &        folder,
                   std::string const &        file,
                   unsigned int const         counter)
{
  // write surface mesh only
  DataOutFaces<dim, DoFHandler<dim>> data_out_surface(true /*surface only*/);
  data_out_surface.attach_triangulation(triangulation);
  data_out_surface.build_patches(mapping, n_subdivisions);
  data_out_surface.write_vtu_with_pvtu_record(folder, file, counter, 4);
}

template<int dim>
void
write_boundary_IDs(Triangulation<dim> const & triangulation,
                   std::string const &        folder,
                   std::string const &        file,
                   MPI_Comm const &           mpi_communicator = MPI_COMM_WORLD)
{
  const unsigned int rank    = Utilities::MPI::this_mpi_process(mpi_communicator);
  const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(mpi_communicator);

  const unsigned int n_digits = static_cast<int>(std::ceil(std::log10(std::fabs(n_ranks))));

  std::string filename =
    folder + file + "_boundary_IDs" + "." + Utilities::int_to_string(rank, n_digits) + ".vtk";
  std::ofstream output(filename.c_str());

  GridOut           grid_out;
  GridOutFlags::Vtk flags;
  flags.output_cells         = false;
  flags.output_faces         = true;
  flags.output_edges         = false;
  flags.output_only_relevant = false;
  grid_out.set_flags(flags);
  grid_out.write_vtk(triangulation, output);
}


#endif /* INCLUDE_POSTPROCESSOR_WRITE_OUTPUT_H_ */
