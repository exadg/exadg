/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef INCLUDE_EXADG_AERO_ACOUSTIC_USER_INTERFACE_APPLICATION_BASE_H_
#define INCLUDE_EXADG_AERO_ACOUSTIC_USER_INTERFACE_APPLICATION_BASE_H_

// deal.II
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

// ExaDG
#include <exadg/grid/grid.h>
#include <exadg/grid/grid_utilities.h>
#include <exadg/operators/resolution_parameters.h>
#include <exadg/postprocessor/output_parameters.h>

// Fluid
#include <exadg/incompressible_navier_stokes/postprocessor/postprocessor.h>
#include <exadg/incompressible_navier_stokes/user_interface/boundary_descriptor.h>
#include <exadg/incompressible_navier_stokes/user_interface/field_functions.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>

// Acoustic
#include <exadg/acoustic_conservation_equations/postprocessor/postprocessor.h>
#include <exadg/acoustic_conservation_equations/user_interface/boundary_descriptor.h>
#include <exadg/acoustic_conservation_equations/user_interface/field_functions.h>
#include <exadg/acoustic_conservation_equations/user_interface/parameters.h>

// AeroAcoustic
#include <exadg/aero_acoustic/user_interface/parameters.h>

namespace ExaDG
{
namespace AcousticsAeroAcoustic
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

  virtual ~ApplicationBase() = default;

  virtual void
  add_parameters(dealii::ParameterHandler & prm)
  {
    resolution.add_parameters(prm, "SpatialResolutionAcoustics");
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
  setup(std::shared_ptr<Grid<dim>> & grid, std::shared_ptr<dealii::Mapping<dim>> & mapping)
  {
    parse_parameters();

    set_resolution_parameters();

    // parameters
    set_parameters();
    param.check();

    // Some AeroAcoustic specific Asserts
    AssertThrow(param.adaptive_time_stepping == false,
                dealii::ExcMessage(
                  "Adaptive timestepping not yet implemented for aero-acoustics."));
    param.print(pcout, "List of parameters for acoustic conservation equations:");

    // grid
    grid = std::make_shared<Grid<dim>>();
    create_grid(*grid, mapping);
    print_grid_info(pcout, *grid);

    // boundary conditions
    boundary_descriptor = std::make_shared<Acoustics::BoundaryDescriptor<dim>>();
    set_boundary_descriptor();
    ExaDG::verify_boundary_conditions(*boundary_descriptor, *grid);

    // field functions
    field_functions = std::make_shared<Acoustics::FieldFunctions<dim>>();
    set_field_functions();
  }

  Acoustics::Parameters const &
  get_parameters() const
  {
    return param;
  }

  std::shared_ptr<Acoustics::BoundaryDescriptor<dim> const>
  get_boundary_descriptor() const
  {
    return boundary_descriptor;
  }

  std::shared_ptr<Acoustics::FieldFunctions<dim> const>
  get_field_functions() const
  {
    return field_functions;
  }

  virtual std::shared_ptr<Acoustics::PostProcessorBase<dim, Number>>
  create_postprocessor() = 0;

protected:
  MPI_Comm const mpi_comm;

  dealii::ConditionalOStream pcout;

  Acoustics::Parameters param;

  std::shared_ptr<Acoustics::FieldFunctions<dim>>     field_functions;
  std::shared_ptr<Acoustics::BoundaryDescriptor<dim>> boundary_descriptor;

  std::string parameter_file;

  OutputParameters output_parameters;

private:
  void
  set_resolution_parameters()
  {
    param.degree_u             = resolution.degree;
    param.grid.n_refine_global = resolution.refine_space;
  }

  virtual void
  set_parameters() = 0;

  virtual void
  create_grid(Grid<dim> & grid, std::shared_ptr<dealii::Mapping<dim>> & mapping) = 0;

  virtual void
  set_boundary_descriptor() = 0;

  virtual void
  set_field_functions() = 0;

  SpatialResolutionParameters resolution;
};

} // namespace AcousticsAeroAcoustic

namespace FluidAeroAcoustic
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

  virtual ~ApplicationBase() = default;

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
  setup(std::shared_ptr<Grid<dim>> &                      grid,
        std::shared_ptr<dealii::Mapping<dim>> &           mapping,
        std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings)
  {
    parse_parameters();

    set_resolution_parameters();

    // parameters
    set_parameters();
    param.check(pcout);
    param.print(pcout, "List of parameters for incompressible flow solver:");

    // Some AeroAcoustic specific Asserts
    AssertThrow(param.problem_type == IncNS::ProblemType::Unsteady,
                dealii::ExcMessage("Invalid parameter in context of aero-acoustic."));
    AssertThrow(param.ale_formulation == false,
                dealii::ExcMessage("ALE not yet implemented for aero-acoustic."));

    // grid
    grid = std::make_shared<Grid<dim>>();
    create_grid(*grid, mapping, multigrid_mappings);
    print_grid_info(pcout, *grid);

    // boundary conditions
    boundary_descriptor = std::make_shared<IncNS::BoundaryDescriptor<dim>>();
    set_boundary_descriptor();
    IncNS::verify_boundary_conditions<dim>(*boundary_descriptor, *grid);

    // field functions
    field_functions = std::make_shared<IncNS::FieldFunctions<dim>>();
    set_field_functions();
  }

  IncNS::Parameters const &
  get_parameters() const
  {
    return param;
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

protected:
  MPI_Comm const mpi_comm;

  dealii::ConditionalOStream pcout;

  // fluid
  IncNS::Parameters                               param;
  std::shared_ptr<IncNS::FieldFunctions<dim>>     field_functions;
  std::shared_ptr<IncNS::BoundaryDescriptor<dim>> boundary_descriptor;

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
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) = 0;

  virtual void
  set_boundary_descriptor() = 0;

  virtual void
  set_field_functions() = 0;

  SpatialResolutionParameters resolution;
};
} // namespace FluidAeroAcoustic

namespace AeroAcoustic
{
template<int dim, typename Number>
class ApplicationBase
{
public:
  ApplicationBase(std::string const & input_file, MPI_Comm const & comm)
    : parameter_file(input_file),
      mpi_comm(comm),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
  }

  virtual ~ApplicationBase() = default;

  void
  setup()
  {
    set_single_field_solvers(parameter_file, mpi_comm);

    parse_parameters();
    parameters.check();
    parameters.print(pcout, "List of parameters for aero-acoustic solver");
  }

  virtual void
  add_parameters(dealii::ParameterHandler & prm)
  {
    parameters.add_parameters(prm, "AeroAcoustic");

    acoustic->add_parameters(prm);
    fluid->add_parameters(prm);
  }

  Parameters parameters;

  std::shared_ptr<AcousticsAeroAcoustic::ApplicationBase<dim, Number>> acoustic;
  std::shared_ptr<FluidAeroAcoustic::ApplicationBase<dim, Number>>     fluid;

private:
  void
  parse_parameters()
  {
    dealii::ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(parameter_file, "", true, true);
  }

  virtual void
  set_single_field_solvers(std::string input_file, MPI_Comm const & comm) = 0;

  std::string const          parameter_file;
  MPI_Comm const             mpi_comm;
  dealii::ConditionalOStream pcout;
};

} // namespace AeroAcoustic
} // namespace ExaDG

#endif /* INCLUDE_EXADG_AERO_ACOUSTIC_USER_INTERFACE_APPLICATION_BASE_H_ */
