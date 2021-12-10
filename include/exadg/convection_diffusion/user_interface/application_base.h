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

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_USER_INTERFACE_APPLICATION_BASE_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_USER_INTERFACE_APPLICATION_BASE_H_

// deal.II
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

// ExaDG
#include <exadg/convection_diffusion/postprocessor/postprocessor.h>
#include <exadg/convection_diffusion/user_interface/boundary_descriptor.h>
#include <exadg/convection_diffusion/user_interface/field_functions.h>
#include <exadg/convection_diffusion/user_interface/input_parameters.h>
#include <exadg/grid/grid.h>

namespace ExaDG
{
namespace ConvDiff
{
using namespace dealii;

template<int dim, typename Number>
class ApplicationBase
{
public:
  virtual void
  add_parameters(ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("Output");
      prm.add_parameter("OutputDirectory",  output_directory, "Directory where output is written.");
      prm.add_parameter("OutputName",       output_name,      "Name of output files.");
      prm.add_parameter("WriteOutput",      write_output,     "Decides whether vtu output is written.");
    prm.leave_subsection();
    // clang-format on
  }

  ApplicationBase(std::string parameter_file, MPI_Comm const & comm)
    : mpi_comm(comm), parameter_file(parameter_file), n_subdivisions_1d_hypercube(1)
  {
    field_functions     = std::make_shared<FieldFunctions<dim>>();
    boundary_descriptor = std::make_shared<BoundaryDescriptor<dim>>();
  }

  virtual ~ApplicationBase()
  {
  }

  virtual void
  set_input_parameters(unsigned int const degree) = 0;

  virtual std::shared_ptr<Grid<dim, Number>>
  create_grid(GridData const & grid_data) = 0;

  virtual std::shared_ptr<Function<dim>>
  create_mesh_movement_function()
  {
    std::shared_ptr<Function<dim>> mesh_motion =
      std::make_shared<Functions::ZeroFunction<dim>>(dim);

    return mesh_motion;
  }

  virtual void
  set_boundary_descriptor() = 0;

  virtual void
  set_field_functions() = 0;

  virtual std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() = 0;

  void
  set_subdivisions_hypercube(unsigned int const n_subdivisions_1d)
  {
    n_subdivisions_1d_hypercube = n_subdivisions_1d;
  }

  InputParameters const &
  get_parameters() const
  {
    return param;
  }

  std::shared_ptr<BoundaryDescriptor<dim> const>
  get_boundary_descriptor() const
  {
    return boundary_descriptor;
  }

  std::shared_ptr<FieldFunctions<dim> const>
  get_field_functions() const
  {
    return field_functions;
  }

protected:
  MPI_Comm const & mpi_comm;

  InputParameters                          param;
  std::shared_ptr<FieldFunctions<dim>>     field_functions;
  std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor;

  std::string parameter_file;

  unsigned int n_subdivisions_1d_hypercube;

  std::string output_directory = "output/", output_name = "output";
  bool        write_output = false;
};

} // namespace ConvDiff
} // namespace ExaDG


#endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_USER_INTERFACE_APPLICATION_BASE_H_ */
