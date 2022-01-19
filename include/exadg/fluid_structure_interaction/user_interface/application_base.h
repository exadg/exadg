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
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>

// Structure
#include <exadg/structure/material/library/st_venant_kirchhoff.h>
#include <exadg/structure/postprocessor/postprocessor.h>
#include <exadg/structure/user_interface/boundary_descriptor.h>
#include <exadg/structure/user_interface/field_functions.h>
#include <exadg/structure/user_interface/material_descriptor.h>
#include <exadg/structure/user_interface/parameters.h>

// moving mesh
#include <exadg/convection_diffusion/user_interface/boundary_descriptor.h>
#include <exadg/poisson/user_interface/analytical_solution.h>
#include <exadg/poisson/user_interface/field_functions.h>
#include <exadg/poisson/user_interface/parameters.h>

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

  ApplicationBase(std::string parameter_file, MPI_Comm const & comm)
    : mpi_comm(comm),
      pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0),
      parameter_file(parameter_file)
  {
  }

  virtual ~ApplicationBase()
  {
  }

  void
  set_parameters_convergence_study(unsigned int const degree_fluid,
                                   unsigned int const degree_structure,
                                   unsigned int const refine_space_fluid,
                                   unsigned int const refine_space_structure)
  {
    // fluid
    this->fluid_param.degree_u             = degree_fluid;
    this->fluid_param.grid.n_refine_global = refine_space_fluid;

    // structure
    this->structure_param.degree               = degree_structure;
    this->structure_param.grid.n_refine_global = refine_space_structure;
  }

  void
  setup()
  {
    /*
     * Structure
     */
    // parameters
    set_parameters_structure();
    structure_param.check();
    // Some FSI specific Asserts
    AssertThrow(structure_param.pull_back_traction == true,
                ExcMessage("Invalid parameter in context of fluid-structure interaction."));
    structure_param.print(pcout, "List of parameters for structure:");

    // grid
    structure_grid = std::make_shared<Grid<dim, Number>>(structure_param.grid, mpi_comm);
    create_grid_structure();
    print_grid_info(pcout, *structure_grid);

    // boundary conditions
    structure_boundary_descriptor = std::make_shared<Structure::BoundaryDescriptor<dim>>();
    set_boundary_descriptor_structure();
    verify_boundary_conditions(*structure_boundary_descriptor, *structure_grid);

    // material_descriptor
    structure_material_descriptor = std::make_shared<Structure::MaterialDescriptor>();
    set_material_descriptor_structure();

    // field functions
    structure_field_functions = std::make_shared<Structure::FieldFunctions<dim>>();
    set_field_functions_structure();


    /*
     * Fluid
     */
    // parameters
    set_parameters_fluid();
    fluid_param.check(pcout);
    fluid_param.print(pcout, "List of parameters for incompressible flow solver:");

    // Some FSI specific Asserts
    AssertThrow(fluid_param.problem_type == IncNS::ProblemType::Unsteady,
                ExcMessage("Invalid parameter in context of fluid-structure interaction."));
    AssertThrow(fluid_param.ale_formulation == true,
                ExcMessage("Invalid parameter in context of fluid-structure interaction."));

    // grid
    fluid_grid = std::make_shared<Grid<dim, Number>>(fluid_param.grid, mpi_comm);
    create_grid_fluid();
    print_grid_info(pcout, *fluid_grid);

    // boundary conditions
    fluid_boundary_descriptor = std::make_shared<IncNS::BoundaryDescriptor<dim>>();
    set_boundary_descriptor_fluid();
    IncNS::verify_boundary_conditions<dim, Number>(*fluid_boundary_descriptor, *fluid_grid);

    // field functions
    fluid_field_functions = std::make_shared<IncNS::FieldFunctions<dim>>();
    set_field_functions_fluid();

    /*
     * ALE
     */
    if(fluid_param.mesh_movement_type == IncNS::MeshMovementType::Poisson)
    {
      // parameters
      set_parameters_ale_poisson();
      ale_poisson_param.check();
      AssertThrow(ale_poisson_param.right_hand_side == false,
                  ExcMessage("Parameter does not make sense in context of FSI."));
      ale_poisson_param.print(pcout, "List of parameters for ALE solver (Poisson):");

      // boundary conditions
      ale_poisson_boundary_descriptor = std::make_shared<Poisson::BoundaryDescriptor<1, dim>>();
      set_boundary_descriptor_ale_poisson();
      verify_boundary_conditions(*ale_poisson_boundary_descriptor, *fluid_grid);

      // field functions
      ale_poisson_field_functions = std::make_shared<Poisson::FieldFunctions<dim>>();
      set_field_functions_ale_poisson();
    }
    else if(fluid_param.mesh_movement_type == IncNS::MeshMovementType::Elasticity)
    {
      // parameters
      set_parameters_ale_elasticity();
      ale_elasticity_param.check();
      AssertThrow(ale_elasticity_param.body_force == false,
                  ExcMessage("Parameter does not make sense in context of FSI."));
      ale_elasticity_param.print(pcout, "List of parameters for ALE solver (elasticity):");

      // boundary conditions
      ale_elasticity_boundary_descriptor = std::make_shared<Structure::BoundaryDescriptor<dim>>();
      set_boundary_descriptor_ale_elasticity();
      verify_boundary_conditions(*ale_elasticity_boundary_descriptor, *fluid_grid);

      // material_descriptor
      ale_elasticity_material_descriptor = std::make_shared<Structure::MaterialDescriptor>();
      set_material_descriptor_ale_elasticity();

      // field functions
      ale_elasticity_field_functions = std::make_shared<Structure::FieldFunctions<dim>>();
      set_field_functions_ale_elasticity();
    }
    else
    {
      AssertThrow(false, ExcMessage("not implemented."));
    }
  }

  virtual std::shared_ptr<IncNS::PostProcessorBase<dim, Number>>
  create_postprocessor_fluid() = 0;

  virtual std::shared_ptr<Structure::PostProcessor<dim, Number>>
  create_postprocessor_structure() = 0;

  IncNS::Parameters const &
  get_parameters_fluid() const
  {
    return fluid_param;
  }

  std::shared_ptr<Grid<dim, Number> const>
  get_grid_fluid() const
  {
    return fluid_grid;
  }

  std::shared_ptr<IncNS::BoundaryDescriptor<dim> const>
  get_boundary_descriptor_fluid() const
  {
    return fluid_boundary_descriptor;
  }

  std::shared_ptr<IncNS::FieldFunctions<dim> const>
  get_field_functions_fluid() const
  {
    return fluid_field_functions;
  }

  Structure::Parameters const &
  get_parameters_structure() const
  {
    return structure_param;
  }

  std::shared_ptr<Grid<dim, Number> const>
  get_grid_structure() const
  {
    return structure_grid;
  }

  std::shared_ptr<Structure::BoundaryDescriptor<dim> const>
  get_boundary_descriptor_structure() const
  {
    return structure_boundary_descriptor;
  }

  std::shared_ptr<Structure::MaterialDescriptor const>
  get_material_descriptor_structure() const
  {
    return structure_material_descriptor;
  }

  std::shared_ptr<Structure::FieldFunctions<dim> const>
  get_field_functions_structure() const
  {
    return structure_field_functions;
  }

  Poisson::Parameters const &
  get_parameters_ale_poisson() const
  {
    return ale_poisson_param;
  }

  std::shared_ptr<Poisson::BoundaryDescriptor<1, dim> const>
  get_boundary_descriptor_ale_poisson() const
  {
    return ale_poisson_boundary_descriptor;
  }

  std::shared_ptr<Poisson::FieldFunctions<dim> const>
  get_field_functions_ale_poisson() const
  {
    return ale_poisson_field_functions;
  }

  Structure::Parameters const &
  get_parameters_ale_elasticity() const
  {
    return ale_elasticity_param;
  }

  std::shared_ptr<Structure::BoundaryDescriptor<dim> const>
  get_boundary_descriptor_ale_elasticity() const
  {
    return ale_elasticity_boundary_descriptor;
  }

  std::shared_ptr<Structure::MaterialDescriptor const>
  get_material_descriptor_ale_elasticity() const
  {
    return ale_elasticity_material_descriptor;
  }

  std::shared_ptr<Structure::FieldFunctions<dim> const>
  get_field_functions_ale_elasticity() const
  {
    return ale_elasticity_field_functions;
  }

protected:
  MPI_Comm const & mpi_comm;

  ConditionalOStream pcout;

  // fluid
  IncNS::Parameters                               fluid_param;
  std::shared_ptr<Grid<dim, Number>>              fluid_grid;
  std::shared_ptr<IncNS::FieldFunctions<dim>>     fluid_field_functions;
  std::shared_ptr<IncNS::BoundaryDescriptor<dim>> fluid_boundary_descriptor;

  // ALE mesh motion

  // Poisson-type mesh motion
  Poisson::Parameters                                  ale_poisson_param;
  std::shared_ptr<Poisson::FieldFunctions<dim>>        ale_poisson_field_functions;
  std::shared_ptr<Poisson::BoundaryDescriptor<1, dim>> ale_poisson_boundary_descriptor;

  // elasticity-type mesh motion
  Structure::Parameters                               ale_elasticity_param;
  std::shared_ptr<Structure::FieldFunctions<dim>>     ale_elasticity_field_functions;
  std::shared_ptr<Structure::BoundaryDescriptor<dim>> ale_elasticity_boundary_descriptor;
  std::shared_ptr<Structure::MaterialDescriptor>      ale_elasticity_material_descriptor;

  // structure
  Structure::Parameters                               structure_param;
  std::shared_ptr<Grid<dim, Number>>                  structure_grid;
  std::shared_ptr<Structure::MaterialDescriptor>      structure_material_descriptor;
  std::shared_ptr<Structure::BoundaryDescriptor<dim>> structure_boundary_descriptor;
  std::shared_ptr<Structure::FieldFunctions<dim>>     structure_field_functions;

  std::string parameter_file;

  std::string output_directory = "output/", output_name = "output";
  bool        write_output = false;

private:
  // fluid
  virtual void
  set_parameters_fluid() = 0;

  virtual void
  create_grid_fluid() = 0;

  virtual void
  set_boundary_descriptor_fluid() = 0;

  virtual void
  set_field_functions_fluid() = 0;

  // ALE

  // Poisson type mesh motion
  virtual void
  set_parameters_ale_poisson() = 0;

  virtual void
  set_boundary_descriptor_ale_poisson() = 0;

  virtual void
  set_field_functions_ale_poisson() = 0;

  // elasticity type mesh motion
  virtual void
  set_parameters_ale_elasticity() = 0;

  virtual void
  set_boundary_descriptor_ale_elasticity() = 0;

  virtual void
  set_material_descriptor_ale_elasticity() = 0;

  virtual void
  set_field_functions_ale_elasticity() = 0;

  // Structure
  virtual void
  set_parameters_structure() = 0;

  virtual void
  create_grid_structure() = 0;

  virtual void
  set_boundary_descriptor_structure() = 0;

  virtual void
  set_material_descriptor_structure() = 0;

  virtual void
  set_field_functions_structure() = 0;
};

} // namespace FSI
} // namespace ExaDG

#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_USER_INTERFACE_APPLICATION_BASE_H_ */
