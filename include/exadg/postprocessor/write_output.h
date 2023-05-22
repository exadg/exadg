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
#include <deal.II/base/bounding_box.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/particles/data_out.h>
#include <deal.II/particles/particle_handler.h>

namespace ExaDG
{
template<int dim>
void
write_surface_mesh(dealii::Triangulation<dim> const & triangulation,
                   dealii::Mapping<dim> const &       mapping,
                   unsigned int const                 n_subdivisions,
                   std::string const &                folder,
                   std::string const &                file,
                   unsigned int const                 counter,
                   MPI_Comm const &                   mpi_comm)
{
  // write surface mesh only
  dealii::DataOutFaces<dim> data_out_surface(true /*surface only*/);
  data_out_surface.attach_triangulation(triangulation);
  data_out_surface.build_patches(mapping, n_subdivisions);
  data_out_surface.write_vtu_with_pvtu_record(folder, file + "_surface", counter, mpi_comm, 4);
}

template<int dim>
void
write_boundary_IDs(dealii::Triangulation<dim> const & triangulation,
                   std::string const &                folder,
                   std::string const &                file,
                   MPI_Comm const &                   mpi_communicator)
{
  unsigned int const rank    = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
  unsigned int const n_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

  unsigned int const n_digits = static_cast<int>(std::ceil(std::log10(std::fabs(n_ranks))));

  std::string filename = folder + file + "_boundary_IDs" + "." +
                         dealii::Utilities::int_to_string(rank, n_digits) + ".vtk";
  std::ofstream output(filename);

  dealii::GridOut           grid_out;
  dealii::GridOutFlags::Vtk flags;
  flags.output_cells         = false;
  flags.output_faces         = true;
  flags.output_edges         = false;
  flags.output_only_relevant = false;
  grid_out.set_flags(flags);
  grid_out.write_vtk(triangulation, output);
}

template<int dim>
void
write_grid(dealii::Triangulation<dim> const & triangulation,
           dealii::Mapping<dim> const &       mapping,
           unsigned int const                 n_subdivisions,
           std::string const &                folder,
           std::string const &                file,
           unsigned int const &               counter,
           MPI_Comm const &                   mpi_comm)
{
  std::string filename = file + "_grid";

  dealii::DataOut<dim> data_out;

  dealii::DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = n_subdivisions > 1;
  data_out.set_flags(flags);

  data_out.attach_triangulation(triangulation);
  data_out.build_patches(mapping, n_subdivisions, dealii::DataOut<dim>::curved_inner_cells);
  data_out.write_vtu_with_pvtu_record(folder, filename, counter, mpi_comm, 4);
}

template<int dim>
void
write_points(dealii::Triangulation<dim> const &      triangulation,
             dealii::Mapping<dim> const &            mapping,
             std::vector<dealii::Point<dim>> const & points,
             std::string const &                     folder,
             std::string const &                     file,
             unsigned int const                      counter,
             MPI_Comm const &                        mpi_comm)
{
  std::string filename = file + "_points";

  dealii::Particles::ParticleHandler<dim, dim> particle_handler(triangulation, mapping);

  particle_handler.insert_particles(points);

  dealii::Particles::DataOut<dim, dim> particle_output;
  particle_output.build_patches(particle_handler);
  particle_output.write_vtu_with_pvtu_record(folder, filename, counter, mpi_comm);
}

template<int dim>
void
write_points_in_dummy_triangulation(std::vector<dealii::Point<dim>> const & points,
                                    std::string const &                     folder,
                                    std::string const &                     file,
                                    unsigned int const                      counter,
                                    MPI_Comm const &                        mpi_comm)
{
  dealii::BoundingBox<dim> bounding_box(points);
  auto const               boundary_points =
    bounding_box.create_extended(1e-3 * std::pow(bounding_box.volume(), 1.0 / ((double)dim)))
      .get_boundary_points();

  dealii::Triangulation<dim> particle_dummy_tria;
  dealii::GridGenerator::hyper_rectangle(particle_dummy_tria,
                                         boundary_points.first,
                                         boundary_points.second);

  dealii::MappingQGeneric<dim> particle_dummy_mapping(1 /* mapping_degree */);

  write_points(
    particle_dummy_tria, particle_dummy_mapping, points, folder, file, counter, mpi_comm);
}

} // namespace ExaDG

#endif /* INCLUDE_EXADG_POSTPROCESSOR_WRITE_OUTPUT_H_ */
