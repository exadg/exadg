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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_APPLICATION_BASE_PRECURSOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_APPLICATION_BASE_PRECURSOR_H_

// deal.II
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

// ExaDG
#include <exadg/convection_diffusion/user_interface/boundary_descriptor.h>
#include <exadg/grid/grid.h>
#include <exadg/grid/grid_utilities.h>
#include <exadg/incompressible_navier_stokes/postprocessor/postprocessor.h>
#include <exadg/incompressible_navier_stokes/user_interface/boundary_descriptor.h>
#include <exadg/incompressible_navier_stokes/user_interface/field_functions.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>
#include <exadg/operators/resolution_parameters.h>
#include <exadg/postprocessor/output_parameters.h>

namespace ExaDG
{
namespace IncNS
{
namespace Precursor
{
template<int dim, typename Number>
class Domain
{
public:
  virtual void
  add_parameters(dealii::ParameterHandler & prm, std::vector<std::string> const & subsection_names)
  {
    for(auto & name : subsection_names)
    {
      prm.enter_subsection(name);
    }

    resolution_parameters.add_parameters(prm);
    output_parameters.add_parameters(prm);

    for(auto & name : subsection_names)
    {
      (void)name;
      prm.leave_subsection();
    }
  }

  Domain(std::string parameter_file, MPI_Comm const & comm)
    : mpi_comm(comm),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
      parameter_file(parameter_file)
  {
  }

  virtual ~Domain()
  {
  }

  void
  setup(std::shared_ptr<Grid<dim>> &                      grid,
        std::shared_ptr<dealii::Mapping<dim>> &           mapping,
        std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings,
        std::vector<std::string> const &                  subsection_names)
  {
    parse_parameters(subsection_names);

    // set resolution parameters
    this->param.degree_u             = this->resolution_parameters.degree;
    this->param.grid.n_refine_global = this->resolution_parameters.refine_space;

    set_parameters();
    param.check(pcout);
    param.print(pcout, "List of parameters:");

    // grid
    GridUtilities::create_mapping(mapping, param.grid.element_type, param.mapping_degree);
    std::shared_ptr<std::vector<std::shared_ptr<dealii::Mapping<dim>>>> coarse_mappings;
    multigrid_mappings = std::make_shared<MultigridMappings<dim, Number>>(mapping, coarse_mappings);

    grid = std::make_shared<Grid<dim>>();
    create_grid(*grid, mapping, multigrid_mappings);
    print_grid_info(pcout, *grid);

    // boundary conditions
    boundary_descriptor = std::make_shared<BoundaryDescriptor<dim>>();
    set_boundary_descriptor();
    verify_boundary_conditions<dim>(*boundary_descriptor, *grid);

    // field functions
    field_functions = std::make_shared<FieldFunctions<dim>>();
    set_field_functions();
  }

  virtual std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() = 0;

  Parameters const &
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
  virtual void
  parse_parameters(std::vector<std::string> const & subsection_names)
  {
    dealii::ParameterHandler prm;
    this->add_parameters(prm, subsection_names);
    prm.parse_input(parameter_file, "", true, true);
  }

  MPI_Comm const & mpi_comm;

  dealii::ConditionalOStream pcout;

  std::string parameter_file;

  Parameters param;

  std::shared_ptr<FieldFunctions<dim>>     field_functions;
  std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor;

  SpatialResolutionParameters resolution_parameters;
  OutputParameters            output_parameters;

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
};

template<int dim, typename Number>
class ApplicationBase
{
public:
  ApplicationBase(std::string parameter_file, MPI_Comm const & comm)
    : mpi_comm(comm),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
      parameter_file(parameter_file),
      switch_off_precursor(false) // precursor is active by default
  {
  }

  virtual ~ApplicationBase()
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    AssertThrow(main.get(), dealii::ExcMessage("Domain main is uninitialized."));
    AssertThrow(precursor.get(), dealii::ExcMessage("Domain precursor is uninitialized."));

    main->add_parameters(prm, {"Main"});

    if(precursor_is_active())
      precursor->add_parameters(prm, {"Precursor"});
  }

  bool
  precursor_is_active() const
  {
    return not switch_off_precursor;
  }

  /**
   * Precursor and main domain. Make sure to create these objects in the constructor
   * of a derived class implementing a concrete precursor application with non-abstract
   * precursor domain and main domain.
   */
  std::shared_ptr<Domain<dim, Number>> precursor, main;

protected:
  MPI_Comm const & mpi_comm;

  dealii::ConditionalOStream pcout;

  std::string parameter_file;

  // This parameter allows to switch off the precursor (e.g. to test a simplified setup without
  // precursor and prescribed boundary conditions or inflow data). The default value is false, i.e.
  // to simulate with precursor. Set this variable to true in a derived class in order to switch off
  // the precursor.
  bool switch_off_precursor;
};

} // namespace Precursor
} // namespace IncNS
} // namespace ExaDG



#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_APPLICATION_BASE_PRECURSOR_H_ \
        */
