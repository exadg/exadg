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

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_USER_INTERFACE_APPLICATION_BASE_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_USER_INTERFACE_APPLICATION_BASE_H_

// deal.II
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

// ExaDG
#include <exadg/grid/grid.h>

// Fluid
#include <exadg/incompressible_navier_stokes/postprocessor/postprocessor.h>
#include <exadg/incompressible_navier_stokes/user_interface/boundary_descriptor.h>
#include <exadg/incompressible_navier_stokes/user_interface/field_functions.h>
#include <exadg/incompressible_navier_stokes/user_interface/input_parameters.h>

// Structure
#include <exadg/structure/material/library/st_venant_kirchhoff.h>
#include <exadg/structure/postprocessor/postprocessor.h>
#include <exadg/structure/user_interface/boundary_descriptor.h>
#include <exadg/structure/user_interface/field_functions.h>
#include <exadg/structure/user_interface/input_parameters.h>
#include <exadg/structure/user_interface/material_descriptor.h>

// moving mesh
#include <exadg/convection_diffusion/user_interface/boundary_descriptor.h>
#include <exadg/poisson/user_interface/analytical_solution.h>
#include <exadg/poisson/user_interface/field_functions.h>
#include <exadg/poisson/user_interface/input_parameters.h>

namespace ExaDG
{
namespace FSI
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

  ApplicationBase(std::string parameter_file) : parameter_file(parameter_file)
  {
  }

  virtual ~ApplicationBase()
  {
  }

  // fluid
  virtual void
  set_input_parameters_fluid(IncNS::InputParameters & parameters) = 0;

  virtual std::shared_ptr<Grid<dim, Number>>
  create_grid_fluid(GridData const & data, MPI_Comm const & mpi_comm) = 0;

  virtual void
  set_boundary_conditions_fluid(
    std::shared_ptr<IncNS::BoundaryDescriptor<dim>> boundary_descriptor) = 0;

  virtual void
  set_field_functions_fluid(std::shared_ptr<IncNS::FieldFunctions<dim>> field_functions) = 0;

  virtual std::shared_ptr<IncNS::PostProcessorBase<dim, Number>>
  create_postprocessor_fluid(unsigned int const degree, MPI_Comm const & mpi_comm) = 0;

  // ALE

  // Poisson type mesh motion
  virtual void
  set_input_parameters_ale(Poisson::InputParameters & parameters) = 0;

  virtual void set_boundary_conditions_ale(
    std::shared_ptr<Poisson::BoundaryDescriptor<1, dim>> boundary_descriptor) = 0;

  virtual void
  set_field_functions_ale(std::shared_ptr<Poisson::FieldFunctions<dim>> field_functions) = 0;

  // elasticity type mesh motion
  virtual void
  set_input_parameters_ale(Structure::InputParameters & parameters) = 0;

  virtual void
  set_boundary_conditions_ale(
    std::shared_ptr<Structure::BoundaryDescriptor<dim>> boundary_descriptor) = 0;

  virtual void
  set_material_ale(Structure::MaterialDescriptor & material_descriptor) = 0;

  virtual void
  set_field_functions_ale(std::shared_ptr<Structure::FieldFunctions<dim>> field_functions) = 0;

  // Structure
  virtual void
  set_input_parameters_structure(Structure::InputParameters & parameters) = 0;

  virtual std::shared_ptr<Grid<dim, Number>>
  create_grid_structure(GridData const & data, MPI_Comm const & mpi_comm) = 0;

  virtual void
  set_boundary_conditions_structure(
    std::shared_ptr<Structure::BoundaryDescriptor<dim>> boundary_descriptor) = 0;

  virtual void
  set_material_structure(Structure::MaterialDescriptor & material_descriptor) = 0;

  virtual void
  set_field_functions_structure(
    std::shared_ptr<Structure::FieldFunctions<dim>> field_functions) = 0;

  virtual std::shared_ptr<Structure::PostProcessor<dim, Number>>
  create_postprocessor_structure(unsigned int const degree, MPI_Comm const & mpi_comm) = 0;

protected:
  std::string parameter_file;

  std::string output_directory = "output/", output_name = "output";
  bool        write_output = false;
};

} // namespace FSI
} // namespace ExaDG

#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_USER_INTERFACE_APPLICATION_BASE_H_ */
