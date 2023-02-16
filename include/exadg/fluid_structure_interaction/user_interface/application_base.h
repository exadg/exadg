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
#include <exadg/postprocessor/output_parameters.h>
#include <exadg/utilities/resolution_parameters.h>

// Fluid
#include <exadg/incompressible_navier_stokes/postprocessor/postprocessor.h>
#include <exadg/incompressible_navier_stokes/user_interface/boundary_descriptor.h>
#include <exadg/incompressible_navier_stokes/user_interface/field_functions.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>

// Structure (and ALE elasticity)
#include <exadg/structure/material/library/st_venant_kirchhoff.h>
#include <exadg/structure/postprocessor/postprocessor.h>
#include <exadg/structure/user_interface/boundary_descriptor.h>
#include <exadg/structure/user_interface/field_functions.h>
#include <exadg/structure/user_interface/material_descriptor.h>
#include <exadg/structure/user_interface/parameters.h>

// ALE Poisson
#include <exadg/poisson/user_interface/analytical_solution.h>
#include <exadg/poisson/user_interface/boundary_descriptor.h>
#include <exadg/poisson/user_interface/field_functions.h>
#include <exadg/poisson/user_interface/parameters.h>

namespace ExaDG
{
namespace StructureFSI
{
template<int dim, typename Number>
class ApplicationBase
{
public:
  ApplicationBase(std::string parameter_file, MPI_Comm const & comm)
    : mpi_comm(comm),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
      parameter_file(parameter_file)
  {
  }

  virtual ~ApplicationBase()
  {
  }

  virtual void
  add_parameters(dealii::ParameterHandler & prm)
  {
    resolution.add_parameters(prm, "SpatialResolutionStructure");
    output_parameters.add_parameters(prm, "Output");
  }

  void
  parse_parameters()
  {
    dealii::ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(parameter_file, "", true, true);
  }

  void
  setup()
  {
    parse_parameters();

    set_resolution_parameters();

    // parameters
    set_parameters();
    param.check();
    // Some FSI specific Asserts
    AssertThrow(param.pull_back_traction == true,
                dealii::ExcMessage("Invalid parameter in context of fluid-structure interaction."));
    param.print(pcout, "List of parameters for structure:");

    // grid
    grid = std::make_shared<Grid<dim>>(param.grid, param.involves_h_multigrid(), mpi_comm);
    create_grid();
    print_grid_info(pcout, *grid);

    // boundary conditions
    boundary_descriptor = std::make_shared<Structure::BoundaryDescriptor<dim>>();
    set_boundary_descriptor();
    verify_boundary_conditions(*boundary_descriptor, *grid);

    // material_descriptor
    material_descriptor = std::make_shared<Structure::MaterialDescriptor>();
    set_material_descriptor();

    // field functions
    field_functions = std::make_shared<Structure::FieldFunctions<dim>>();
    set_field_functions();
  }

  Structure::Parameters const &
  get_parameters() const
  {
    return param;
  }

  std::shared_ptr<Grid<dim> const>
  get_grid() const
  {
    return grid;
  }

  std::shared_ptr<Structure::BoundaryDescriptor<dim> const>
  get_boundary_descriptor() const
  {
    return boundary_descriptor;
  }

  std::shared_ptr<Structure::MaterialDescriptor const>
  get_material_descriptor() const
  {
    return material_descriptor;
  }

  std::shared_ptr<Structure::FieldFunctions<dim> const>
  get_field_functions() const
  {
    return field_functions;
  }

  virtual std::shared_ptr<Structure::PostProcessor<dim, Number>>
  create_postprocessor() = 0;

protected:
  MPI_Comm const & mpi_comm;

  dealii::ConditionalOStream pcout;

  Structure::Parameters                               param;
  std::shared_ptr<Grid<dim>>                          grid;
  std::shared_ptr<Structure::MaterialDescriptor>      material_descriptor;
  std::shared_ptr<Structure::BoundaryDescriptor<dim>> boundary_descriptor;
  std::shared_ptr<Structure::FieldFunctions<dim>>     field_functions;

  std::string parameter_file;

  OutputParameters output_parameters;

private:
  void
  set_resolution_parameters()
  {
    param.degree               = resolution.degree;
    param.grid.n_refine_global = resolution.refine_space;
  }

  virtual void
  set_parameters() = 0;

  virtual void
  create_grid() = 0;

  virtual void
  set_boundary_descriptor() = 0;

  virtual void
  set_material_descriptor() = 0;

  virtual void
  set_field_functions() = 0;

  ResolutionParameters resolution;
};

} // namespace StructureFSI

namespace FluidFSI
{
template<int dim, typename Number>
class ApplicationBase
{
public:
  ApplicationBase(std::string parameter_file, MPI_Comm const & comm)
    : mpi_comm(comm),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
      parameter_file(parameter_file)
  {
  }

  virtual ~ApplicationBase()
  {
  }

  virtual void
  add_parameters(dealii::ParameterHandler & prm)
  {
    resolution.add_parameters(prm, "SpatialResolutionFluid");
    output_parameters.add_parameters(prm, "Output");
  }

  void
  parse_parameters()
  {
    dealii::ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(parameter_file, "", true, true);
  }

  void
  setup()
  {
    parse_parameters();

    set_resolution_parameters();

    // parameters
    set_parameters();
    param.check(pcout);
    param.print(pcout, "List of parameters for incompressible flow solver:");

    // Some FSI specific Asserts
    AssertThrow(param.problem_type == IncNS::ProblemType::Unsteady,
                dealii::ExcMessage("Invalid parameter in context of fluid-structure interaction."));
    AssertThrow(param.ale_formulation == true,
                dealii::ExcMessage("Invalid parameter in context of fluid-structure interaction."));

    // grid
    grid = std::make_shared<Grid<dim>>(param.grid, param.involves_h_multigrid(), mpi_comm);
    create_grid();
    print_grid_info(pcout, *grid);

    // boundary conditions
    boundary_descriptor = std::make_shared<IncNS::BoundaryDescriptor<dim>>();
    set_boundary_descriptor();
    IncNS::verify_boundary_conditions<dim, Number>(*boundary_descriptor, *grid);

    // field functions
    field_functions = std::make_shared<IncNS::FieldFunctions<dim>>();
    set_field_functions();

    /*
     * ALE
     */
    if(param.mesh_movement_type == IncNS::MeshMovementType::Poisson)
    {
      // parameters
      set_parameters_ale_poisson();
      ale_poisson_param.check();
      AssertThrow(ale_poisson_param.right_hand_side == false,
                  dealii::ExcMessage("Parameter does not make sense in context of FSI."));
      AssertThrow(
        ale_poisson_param.grid.multigrid == param.grid.multigrid,
        dealii::ExcMessage(
          "ALE and fluid use the same Grid, requiring the same settings in terms of multigrid coarsening."));

      ale_poisson_param.print(pcout, "List of parameters for ALE solver (Poisson):");

      // boundary conditions
      ale_poisson_boundary_descriptor = std::make_shared<Poisson::BoundaryDescriptor<1, dim>>();
      set_boundary_descriptor_ale_poisson();
      verify_boundary_conditions(*ale_poisson_boundary_descriptor, *grid);

      // field functions
      ale_poisson_field_functions = std::make_shared<Poisson::FieldFunctions<dim>>();
      set_field_functions_ale_poisson();
    }
    else if(param.mesh_movement_type == IncNS::MeshMovementType::Elasticity)
    {
      // parameters
      set_parameters_ale_elasticity();
      ale_elasticity_param.check();
      AssertThrow(ale_elasticity_param.body_force == false,
                  dealii::ExcMessage("Parameter does not make sense in context of FSI."));
      AssertThrow(
        ale_poisson_param.grid.multigrid == param.grid.multigrid,
        dealii::ExcMessage(
          "ALE and fluid use the same Grid, requiring the same settings in terms of multigrid coarsening."));

      ale_elasticity_param.print(pcout, "List of parameters for ALE solver (elasticity):");

      // boundary conditions
      ale_elasticity_boundary_descriptor = std::make_shared<Structure::BoundaryDescriptor<dim>>();
      set_boundary_descriptor_ale_elasticity();
      verify_boundary_conditions(*ale_elasticity_boundary_descriptor, *grid);

      // material_descriptor
      ale_elasticity_material_descriptor = std::make_shared<Structure::MaterialDescriptor>();
      set_material_descriptor_ale_elasticity();

      // field functions
      ale_elasticity_field_functions = std::make_shared<Structure::FieldFunctions<dim>>();
      set_field_functions_ale_elasticity();
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("not implemented."));
    }
  }

  IncNS::Parameters const &
  get_parameters() const
  {
    return param;
  }

  std::shared_ptr<Grid<dim> const>
  get_grid() const
  {
    return grid;
  }

  std::shared_ptr<IncNS::BoundaryDescriptor<dim> const>
  get_boundary_descriptor() const
  {
    return boundary_descriptor;
  }

  std::shared_ptr<IncNS::FieldFunctions<dim> const>
  get_field_functions() const
  {
    return field_functions;
  }

  virtual std::shared_ptr<IncNS::PostProcessorBase<dim, Number>>
  create_postprocessor() = 0;

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

  dealii::ConditionalOStream pcout;

  // fluid
  IncNS::Parameters                               param;
  std::shared_ptr<Grid<dim>>                      grid;
  std::shared_ptr<IncNS::FieldFunctions<dim>>     field_functions;
  std::shared_ptr<IncNS::BoundaryDescriptor<dim>> boundary_descriptor;

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

  std::string parameter_file;

  OutputParameters output_parameters;

private:
  void
  set_resolution_parameters()
  {
    param.degree_u             = resolution.degree;
    param.grid.n_refine_global = resolution.refine_space;
  }

  // fluid
  virtual void
  set_parameters() = 0;

  virtual void
  create_grid() = 0;

  virtual void
  set_boundary_descriptor() = 0;

  virtual void
  set_field_functions() = 0;

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

  ResolutionParameters resolution;
};
} // namespace FluidFSI

namespace FSI
{
template<int dim, typename Number>
class ApplicationBase
{
public:
  virtual void
  add_parameters(dealii::ParameterHandler & prm)
  {
    structure->add_parameters(prm);
    fluid->add_parameters(prm);
  }

  virtual ~ApplicationBase()
  {
  }

  void
  setup()
  {
    structure->setup();

    fluid->setup();
  }

  std::shared_ptr<StructureFSI::ApplicationBase<dim, Number>> structure;
  std::shared_ptr<FluidFSI::ApplicationBase<dim, Number>>     fluid;
};

} // namespace FSI
} // namespace ExaDG

#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_USER_INTERFACE_APPLICATION_BASE_H_ */
