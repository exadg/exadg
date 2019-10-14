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

template<typename DataOut>
void
write_pvtu_record_wrapper(DataOut const &     data_out,
                          std::string const & folder,
                          std::string const & file,
                          unsigned int const  counter,
                          unsigned int const  rank)
{
  if(rank == 0)
  {
    std::vector<std::string> vector;
    for(unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
    {
      std::string filename = file + "_Proc" + Utilities::int_to_string(i) + "_" +
                             Utilities::int_to_string(counter) + ".vtu";

      vector.push_back(filename.c_str());
    }
    std::string master_name = folder + file + "_" + Utilities::int_to_string(counter) + ".pvtu";

    std::ofstream master_output(master_name.c_str());
    data_out.write_pvtu_record(master_output, vector);
  }
}

template<int dim>
void
write_surface_mesh(Triangulation<dim> const & triangulation,
                   Mapping<dim> const &       mapping,
                   unsigned int               n_subdivisions,
                   std::string const &        folder,
                   std::string const &        file,
                   unsigned int const         counter,
                   unsigned int const         rank)
{
  // write surface mesh only
  DataOutFaces<dim, DoFHandler<dim>> data_out_surface(true /*surface only*/);
  data_out_surface.attach_triangulation(triangulation);
  data_out_surface.build_patches(mapping, n_subdivisions);

  std::string filename_surface = folder + file + "_Proc" + Utilities::int_to_string(rank) + "_" +
                                 Utilities::int_to_string(counter) + ".vtu";
  std::ofstream filename(filename_surface.c_str());
  data_out_surface.write_vtu(filename);

  write_pvtu_record_wrapper(data_out_surface, folder, file, counter, rank);
}

template<int dim>
void
write_boundary_IDs(Triangulation<dim> const & triangulation,
                   std::string const &        folder,
                   std::string const &        file)
{
  unsigned int rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  std::string filename =
    folder + file + "_grid" + "_Proc" + Utilities::int_to_string(rank) + ".vtk";
  std::ofstream output(filename.c_str());

  GridOut grid_out;
  grid_out.write_vtk(triangulation, output);
}


#endif /* INCLUDE_POSTPROCESSOR_WRITE_OUTPUT_H_ */
