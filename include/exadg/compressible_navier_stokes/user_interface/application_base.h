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

#ifndef INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_APPLICATION_BASE_H_
#define INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_APPLICATION_BASE_H_

// deal.II
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_description.h>

// ExaDG
#include <exadg/compressible_navier_stokes/postprocessor/postprocessor.h>
#include <exadg/compressible_navier_stokes/user_interface/boundary_descriptor.h>
#include <exadg/compressible_navier_stokes/user_interface/field_functions.h>
#include <exadg/compressible_navier_stokes/user_interface/input_parameters.h>
#include <exadg/grid/grid.h>
#include <exadg/grid/mapping_degree.h>

namespace ExaDG
{
namespace CompNS
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
  }

  virtual ~ApplicationBase()
  {
  }

  void
  setup(unsigned int const         degree,
        unsigned int const         refine_space,
        ConditionalOStream const & pcout)
  {
    parameters.degree = degree;

    set_input_parameters();
    parameters.check_input_parameters();
    parameters.print(pcout, "List of input parameters:");

    // grid
    grid_data.triangulation_type = parameters.triangulation_type;
    grid_data.n_refine_global    = refine_space;
    grid_data.mapping_degree     = get_mapping_degree(parameters.mapping, parameters.degree);

    create_grid();
    print_grid_info(pcout, *grid);

    boundary_descriptor = std::make_shared<BoundaryDescriptor<dim>>();
    field_functions     = std::make_shared<FieldFunctions<dim>>();

    set_boundary_conditions();
    CompNS::verify_boundary_conditions<dim>(boundary_descriptor, *grid);

    set_field_functions();
  }

private:
  virtual void
  set_input_parameters() = 0;

  virtual void
  create_grid() = 0;

  virtual void
  set_boundary_conditions() = 0;

  virtual void
  set_field_functions() = 0;

public:
  virtual std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() = 0;

  void
  set_subdivisions_hypercube(unsigned int const n_subdivisions_1d)
  {
    n_subdivisions_1d_hypercube = n_subdivisions_1d;
  }

  MPI_Comm const & mpi_comm;

  InputParameters                          parameters;
  std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor;
  std::shared_ptr<FieldFunctions<dim>>     field_functions;
  GridData                                 grid_data;
  std::shared_ptr<Grid<dim, Number>>       grid;

protected:
  std::string parameter_file;

  unsigned int n_subdivisions_1d_hypercube;

  std::string output_directory = "output/", output_name = "output";
  bool        write_output = false;
};

} // namespace CompNS
} // namespace ExaDG


#endif /* INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_APPLICATION_BASE_H_ */
