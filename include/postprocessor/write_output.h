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
  data_out_surface.write_vtu_with_pvtu_record(folder, file, counter);
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
