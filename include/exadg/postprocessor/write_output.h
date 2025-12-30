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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef EXADG_POSTPROCESSOR_WRITE_OUTPUT_H_
#define EXADG_POSTPROCESSOR_WRITE_OUTPUT_H_

// C/C++
#include <fstream>

// deal.II
#include <deal.II/base/bounding_box.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/particles/data_out.h>
#include <deal.II/particles/particle_handler.h>

// ExaDG
#include <exadg/grid/grid_data.h>
#include <exadg/operators/quadrature.h>
#include <exadg/postprocessor/output_data_base.h>
#include <exadg/postprocessor/solution_field.h>

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

template<int dim, typename Number>
class VectorWriter
{
public:
  VectorWriter(OutputDataBase const & output_data,
               unsigned int const &   output_counter,
               MPI_Comm const &       mpi_comm)
    : output_data(output_data), output_counter(output_counter), mpi_comm(mpi_comm)
  {
    // Write higher order output.
    dealii::DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = output_data.write_higher_order;
    data_out.set_flags(flags);
  }

  // Note that the vectors must remain valid until we call `write_pvtu()`, which is not the
  // responsibility of this class.
  template<typename VectorType>
  void
  add_data_vector(VectorType const &               vector,
                  dealii::DoFHandler<dim> const &  dof_handler,
                  std::vector<std::string> const & component_names,
                  std::vector<bool> const &        component_is_part_of_vector = {false})
  {
    unsigned int n_components = component_names.size();
    AssertThrow(n_components > 0, dealii::ExcMessage("Provide names for each component."));

    AssertThrow(n_components == component_is_part_of_vector.size(),
                dealii::ExcMessage("Provide names and vector info for each component."));

    // Vector entries are to be interpreted as components of a vector.
    if(n_components > 1)
    {
      std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(n_components,
                                      dealii::DataComponentInterpretation::component_is_scalar);
      for(unsigned int i = 0; i < n_components; ++i)
      {
        if(component_is_part_of_vector[i])
        {
          data_component_interpretation[i] =
            dealii::DataComponentInterpretation::component_is_part_of_vector;
        }
      }
      data_out.add_data_vector(dof_handler, vector, component_names, data_component_interpretation);
    }
    else
    {
      data_out.add_data_vector(dof_handler, vector, component_names[0]);
    }
  }

  void
  add_fields(
    std::vector<dealii::ObserverPointer<SolutionField<dim, Number>>> const & additional_fields)
  {
    for(auto & additional_field : additional_fields)
    {
      if(additional_field->get_type() == SolutionFieldType::scalar)
      {
        data_out.add_data_vector(additional_field->get_dof_handler(),
                                 additional_field->get(),
                                 additional_field->get_name());
      }
      else if(additional_field->get_type() == SolutionFieldType::cellwise)
      {
        data_out.add_data_vector(additional_field->get(), additional_field->get_name());
      }
      else if(additional_field->get_type() == SolutionFieldType::vector)
      {
        std::vector<std::string> names(dim, additional_field->get_name());
        std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
          component_interpretation(
            dim, dealii::DataComponentInterpretation::component_is_part_of_vector);

        data_out.add_data_vector(additional_field->get_dof_handler(),
                                 additional_field->get(),
                                 names,
                                 component_interpretation);
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("This `SolutionFieldType` is not implemented."));
      }
    }
  }

  void
  write_aspect_ratio(dealii::DoFHandler<dim> const & dof_handler,
                     dealii::Mapping<dim> const &    mapping)
  {
    // Add aspect ratio. Vector needs to survive until build_patches.
    if(output_data.write_aspect_ratio)
    {
      dealii::Triangulation<dim> const & tria = dof_handler.get_triangulation();

      ElementType const element_type = get_element_type(tria);

      std::shared_ptr<dealii::Quadrature<dim>> quadrature = create_quadrature<dim>(element_type, 4);

      aspect_ratios = dealii::GridTools::compute_aspect_ratio_of_cells(mapping, tria, *quadrature);
      data_out.add_data_vector(aspect_ratios, "aspect_ratio");
    }
  }

  void
  write_pvtu(dealii::Mapping<dim> const * mapping = nullptr)
  {
    // Build patches, vectors to export must stay in scope until after this call.
    if(mapping == nullptr)
    {
      data_out.build_patches(output_data.degree);
    }
    else
    {
      data_out.build_patches(*mapping,
                             output_data.degree,
                             dealii::DataOut<dim>::curved_inner_cells);
    }

    unsigned int constexpr n_groups = 4;
    data_out.write_vtu_with_pvtu_record(
      output_data.directory, output_data.filename, output_counter, mpi_comm, n_groups);
  }

private:
  OutputDataBase const                                output_data;
  unsigned int                                        output_counter;
  dealii::ObserverPointer<dealii::Mapping<dim> const> mapping;
  MPI_Comm const                                      mpi_comm;

  dealii::Vector<double> aspect_ratios;

  dealii::DataOut<dim> data_out;
};

} // namespace ExaDG

#endif /* EXADG_POSTPROCESSOR_WRITE_OUTPUT_H_ */
