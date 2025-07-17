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

#ifndef EXADG_POSTPROCESSOR_WRITE_OUTPUT_H_
#define EXADG_POSTPROCESSOR_WRITE_OUTPUT_H_

// C/C++
#include <fstream>

// deal.II
#include <deal.II/base/bounding_box.h>
#include <deal.II/dofs/dof_tools.h>
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

template<int dim, typename VectorType>
void
write_vector(dealii::DoFHandler<dim> const & dof_handler,
             dealii::Mapping<dim> const &    mapping,
             VectorType const &              vector,
             std::string const &             folder,
             std::string const &             file,
             unsigned int const              n_subdivisions,
             unsigned int const              n_components = 1)
{
  // Write higher order output.
  dealii::DataOut<dim>          data_out;
  dealii::DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = n_subdivisions > 1;
  data_out.set_flags(flags);
  data_out.attach_dof_handler(dof_handler);

  // Get vector with locally relevant entries.
  VectorType       rel_vector;
  dealii::IndexSet rel_dofs;
  dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, rel_dofs);
  MPI_Comm const & mpi_comm = dof_handler.get_communicator();
  rel_vector.reinit(dof_handler.locally_owned_dofs(), rel_dofs, mpi_comm);
  rel_vector = vector;

  // Vector entries are to be interpreted as components of a vector.
  if(n_components > 1)
  {
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, dealii::DataComponentInterpretation::component_is_part_of_vector);
    std::vector<std::string> solution_names(n_components, "vector");
    data_out.add_data_vector(rel_vector,
                             "vector",
                             dealii::DataOut<dim>::type_dof_data,
                             data_component_interpretation);
  }
  else
  {
    data_out.add_data_vector(rel_vector, "vector");
  }

  auto const & triangulation = dof_handler.get_triangulation();

  // Add vector indicating subdomain.
  dealii::Vector<float> subdomain;
  if constexpr(true)
  {
    subdomain.reinit(triangulation.n_active_cells());
    for(unsigned int i = 0; i < subdomain.size(); ++i)
    {
      subdomain(i) = triangulation.locally_owned_subdomain();
    }
    data_out.add_data_vector(subdomain, "subdomain");
  }

  // Build patches, vectors to export must stay in scope until after this call.
  data_out.build_patches(mapping, n_subdivisions, dealii::DataOut<dim>::curved_inner_cells);

  // Create vtu files + pvtu record.
  std::string filename =
    folder + file + "_p" +
    dealii::Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4);
  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);

  // Combine outputs using rank 0.
  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    std::vector<std::string> filenames;
    for(unsigned int i = 0; i < dealii::Utilities::MPI::n_mpi_processes(mpi_comm); ++i)
    {
      filenames.push_back(folder + file + "_p" + dealii::Utilities::int_to_string(i, 4) + ".vtu");
    }

    // Combine outputs of individual threads.
    std::ofstream master_output((folder + file + ".pvtu").c_str());
    data_out.write_pvtu_record(master_output, filenames);
  }
}

} // namespace ExaDG

#endif /* EXADG_POSTPROCESSOR_WRITE_OUTPUT_H_ */
