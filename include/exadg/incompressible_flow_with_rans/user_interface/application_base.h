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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_FLOW_WITH_RANS_USER_INTERFACE_APPLICATION_BASE_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_FLOW_WITH_RANS_USER_INTERFACE_APPLICATION_BASE_H_

// deal.II
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_description.h>

// ExaDG
#include <exadg/grid/grid.h>
#include <exadg/grid/grid_utilities.h>

#include <exadg/incompressible_navier_stokes_for_rans/postprocessor/postprocessor.h>
#include <exadg/incompressible_navier_stokes_for_rans/user_interface/boundary_descriptor.h>
#include <exadg/incompressible_navier_stokes_for_rans/user_interface/field_functions.h>
#include <exadg/incompressible_navier_stokes_for_rans/user_interface/parameters.h>

#include <exadg/rans_equations/postprocessor/postprocessor.h>
#include <exadg/rans_equations/user_interface/boundary_descriptor.h>
#include <exadg/rans_equations/user_interface/field_functions.h>
#include <exadg/rans_equations/user_interface/parameters.h>

#include <exadg/operators/resolution_parameters.h>
#include <exadg/postprocessor/output_parameters.h>

namespace ExaDG
{
namespace NSRans
{
template<int dim, typename Number>
class FluidBase
{
public:
  virtual void
  add_parameters(dealii::ParameterHandler & prm, std::vector<std::string> const & subsection_names)
  {
    for(auto & name : subsection_names)
    {
      prm.enter_subsection(name);
    }

    resolution.add_parameters(prm);

    output_parameters.add_parameters(prm);

    for(auto & name : subsection_names)
    {
      (void)name;
      prm.leave_subsection();
    }
  }

  FluidBase(std::string parameter_file, MPI_Comm const & comm)
    : mpi_comm(comm),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
      parameter_file(parameter_file)
  {
  }

  virtual ~FluidBase()
  {
  }

  virtual void
  setup(std::shared_ptr<Grid<dim>> &                      grid,
        std::shared_ptr<dealii::Mapping<dim>> &           mapping,
        std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings,
        std::vector<std::string> const &                  subsection_names)
  {
    parse_parameters(subsection_names);

    // set resolution parameters
    param.degree_u             = resolution.degree;
    param.grid.n_refine_global = resolution.refine_space;

    set_parameters();
    param.check(pcout);
    param.print(pcout, "List of parameters:");

    // grid
    grid = std::make_shared<Grid<dim>>();
    create_grid(*grid, mapping, multigrid_mappings);
    print_grid_info(pcout, *grid);

    // boundary conditions
    boundary_descriptor = std::make_shared<IncRANS::BoundaryDescriptor<dim>>();
    set_boundary_descriptor();
    verify_boundary_conditions<dim>(*boundary_descriptor, *grid);

    // field functions
    field_functions = std::make_shared<IncRANS::FieldFunctions<dim>>();
    set_field_functions();
  }

  virtual std::shared_ptr<IncRANS::PostProcessorBase<dim, Number>>
  create_postprocessor() = 0;

  IncRANS::Parameters const &
  get_parameters() const
  {
    return param;
  }

  std::shared_ptr<IncRANS::BoundaryDescriptor<dim> const>
  get_boundary_descriptor() const
  {
    return boundary_descriptor;
  }

  std::shared_ptr<IncRANS::FieldFunctions<dim> const>
  get_field_functions() const
  {
    return field_functions;
  }

  // Analytical mesh motion
  virtual std::shared_ptr<dealii::Function<dim>>
  create_mesh_movement_function()
  {
    std::shared_ptr<dealii::Function<dim>> mesh_motion =
      std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim);

    return mesh_motion;
  }

protected:
  virtual void
  parse_parameters(std::vector<std::string> const & subsection_names)
  {
    dealii::ParameterHandler prm;
    this->add_parameters(prm, subsection_names);
    prm.parse_input(parameter_file, "", true, true);
  }

  MPI_Comm const mpi_comm;

  dealii::ConditionalOStream pcout;

  IncRANS::Parameters param;

  std::shared_ptr<IncRANS::FieldFunctions<dim>>     field_functions;
  std::shared_ptr<IncRANS::BoundaryDescriptor<dim>> boundary_descriptor;

  std::string parameter_file;

  OutputParameters output_parameters;

private:
  virtual void
  set_parameters() = 0;

  virtual void
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) = 0;

  virtual void
  set_boundary_descriptor() = 0;

  virtual void
  set_field_functions() = 0;

  SpatialResolutionParameters resolution;
};

template<int dim, typename Number>
class ScalarBase
{
public:
  virtual void
  add_parameters(dealii::ParameterHandler & prm, std::vector<std::string> const & subsection_names)
  {
    for(auto & name : subsection_names)
    {
      prm.enter_subsection(name);
    }

    prm.enter_subsection("SpatialResolution");
    {
      prm.add_parameter("Degree",
                        degree,
                        "Polynomial degree of shape functions.",
                        dealii::Patterns::Integer(1),
                        true);
    }
    prm.leave_subsection();

    output_parameters.add_parameters(prm);

    for(auto & name : subsection_names)
    {
      (void)name;
      prm.leave_subsection();
    }
  }

  ScalarBase(std::string parameter_file, MPI_Comm const & comm)
    : mpi_comm(comm),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0),
      parameter_file(parameter_file),
      degree(1)
  {
  }

  virtual ~ScalarBase()
  {
  }

  void
  setup(std::vector<std::string> const & subsection_names, Grid<dim> const & grid)
  {
    parse_parameters(subsection_names);

    // set degree of shape functions (note that the refinement level is identical to fluid domain)
    param.degree = degree;

    set_parameters();
    param.check();

    param.print(this->pcout, "List of parameters for quantity " + subsection_names.back() + ":");

    // boundary conditions
    boundary_descriptor = std::make_shared<RANS::BoundaryDescriptor<dim>>();
    set_boundary_descriptor();
    verify_boundary_conditions(*boundary_descriptor, grid);

    // field functions
    field_functions = std::make_shared<RANS::FieldFunctions<dim>>();
    set_field_functions();
  }

  virtual std::shared_ptr<RANS::PostProcessorBase<dim, Number>>
  create_postprocessor() = 0;

  RANS::Parameters const &
  get_parameters() const
  {
    return param;
  }

  std::shared_ptr<RANS::BoundaryDescriptor<dim> const>
  get_boundary_descriptor() const
  {
    return boundary_descriptor;
  }

  std::shared_ptr<RANS::FieldFunctions<dim> const>
  get_field_functions() const
  {
    return field_functions;
  }

protected:
  virtual void
  parse_parameters(std::vector<std::string> const & subsection_names)
  {
    dealii::ParameterHandler prm;
    this->add_parameters(prm, subsection_names);
    prm.parse_input(parameter_file, "", true, true);
  }

  MPI_Comm const mpi_comm;

  dealii::ConditionalOStream pcout;

  RANS::Parameters                               param;
  std::shared_ptr<RANS::FieldFunctions<dim>>     field_functions;
  std::shared_ptr<RANS::BoundaryDescriptor<dim>> boundary_descriptor;

  std::string parameter_file;

  OutputParameters output_parameters;

private:
  virtual void
  set_parameters() = 0;

  virtual void
  set_boundary_descriptor() = 0;

  virtual void
  set_field_functions() = 0;

  unsigned int degree;
};

template<int dim, typename Number>
class ApplicationBase
{
public:
  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    AssertThrow(fluid.get(), dealii::ExcMessage("fluid has not been initialized."));

    fluid->add_parameters(prm, {"Fluid"});

    for(unsigned int i = 0; i < scalars.size(); ++i)
    {
      AssertThrow(scalars[i].get(),
                  dealii::ExcMessage("scalar[" + std::to_string(i) +
                                     "] has not been initialized."));

      scalars[i]->add_parameters(prm, {"Scalar" + std::to_string(i)});
    }
  }

  ApplicationBase(std::string parameter_file, MPI_Comm const & comm)
    : mpi_comm(comm),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0),
      parameter_file(parameter_file)
  {
  }

  virtual ~ApplicationBase()
  {
  }

  void
  setup(std::shared_ptr<Grid<dim>> &                      grid,
        std::shared_ptr<dealii::Mapping<dim>> &           mapping,
        std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings)
  {
    AssertThrow(fluid.get(), dealii::ExcMessage("fluid has not been initialized."));

    // The fluid field is defined as the field that creates the grid and the mapping, while all
    // scalar fields use the same grid/mapping
    fluid->setup(grid, mapping, multigrid_mappings, {"Fluid"});

    for(unsigned int i = 0; i < scalars.size(); ++i)
    {
      AssertThrow(scalars[i].get(),
                  dealii::ExcMessage("scalar[" + std::to_string(i) +
                                     "] has not been initialized."));

      scalars[i]->setup({"Scalar" + std::to_string(i)}, *grid);

      // do additional parameter checks
      AssertThrow(scalars[i]->get_parameters().ale_formulation ==
                    fluid->get_parameters().ale_formulation,
                  dealii::ExcMessage(
                    "Parameter ale_formulation is different for fluid field and scalar field"));

      AssertThrow(
        scalars[i]->get_parameters().adaptive_time_stepping ==
          fluid->get_parameters().adaptive_time_stepping,
        dealii::ExcMessage(
          "The option adaptive_time_stepping has to be consistent for fluid and scalar transport solvers."));
    }
  }

  std::shared_ptr<FluidBase<dim, Number>>               fluid;
  std::vector<std::shared_ptr<ScalarBase<dim, Number>>> scalars;

protected:
  MPI_Comm const mpi_comm;

  dealii::ConditionalOStream pcout;

private:
  std::string parameter_file;
};

} // namespace NSRans
} // namespace ExaDG


#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_FLOW_WITH_RANS_USER_INTERFACE_APPLICATION_BASE_H_ */
