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
#include <exadg/postprocessor/output_parameters.h>
#include <exadg/utilities/resolution_parameters.h>

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
  Domain(std::string parameter_file, MPI_Comm const & comm)
    : mpi_comm(comm),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
      parameter_file(parameter_file)
  {
    grid = std::make_shared<Grid<dim>>();
  }

  virtual ~Domain()
  {
  }

  void
  setup()
  {
    set_parameters();
    param.check(pcout);
    param.print(pcout, "List of parameters:");

    // grid
    GridUtilities::create_mapping(mapping, param.grid.element_type, param.mapping_degree);
    create_grid();
    print_grid_info(pcout, *grid);

    // boundary conditions
    boundary_descriptor = std::make_shared<BoundaryDescriptor<dim>>();
    set_boundary_descriptor();
    verify_boundary_conditions<dim>(*boundary_descriptor, *grid);

    // field functions
    field_functions = std::make_shared<FieldFunctions<dim>>();
    set_field_functions();
  }

  void
  set_resolution_parameters(unsigned int const degree, unsigned int const refine_space)
  {
    param.degree_u             = degree;
    param.grid.n_refine_global = refine_space;
  }

  virtual std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() = 0;

  Parameters const &
  get_parameters() const
  {
    return param;
  }

  std::shared_ptr<Grid<dim> const>
  get_grid() const
  {
    return grid;
  }

  std::shared_ptr<dealii::Mapping<dim> const>
  get_mapping() const
  {
    return mapping;
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

  dealii::ConditionalOStream pcout;

  std::string parameter_file;

  Parameters param;

  std::shared_ptr<Grid<dim>> grid;

  std::shared_ptr<dealii::Mapping<dim>> mapping;

  std::shared_ptr<FieldFunctions<dim>>     field_functions;
  std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor;

private:
  virtual void
  set_parameters() = 0;

  virtual void
  create_grid() = 0;

  virtual void
  set_boundary_descriptor() = 0;

  virtual void
  set_field_functions() = 0;
};

template<int dim, typename Number>
class ApplicationBase
{
public:
  virtual void
  add_parameters(dealii::ParameterHandler & prm)
  {
    output_parameters.add_parameters(prm);
  }

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
  set_resolution_parameters(unsigned int const degree, unsigned int const refine_space)
  {
    main->set_resolution_parameters(degree, refine_space);
    precursor->set_resolution_parameters(degree, refine_space);
  }

  void
  setup()
  {
    parse_parameters();

    // main domain
    main->setup();

    // precursor domain
    if(precursor_is_active())
    {
      precursor->setup();

      // make some additional checks (i.e. enforce constraints between main and precursor
      // parameters)
      consistency_checks();
    }
  }

  bool
  precursor_is_active() const
  {
    return not switch_off_precursor;
  }

  std::shared_ptr<Domain<dim, Number>> precursor, main;

protected:
  virtual void
  parse_parameters()
  {
    dealii::ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(parameter_file, "", true, true);
  }

  MPI_Comm const & mpi_comm;

  dealii::ConditionalOStream pcout;

  std::string parameter_file;

  OutputParameters output_parameters;

  // This parameter allows to switch off the precursor (e.g. to test a simplified setup without
  // precursor and prescribed boundary conditions or inflow data). The default value is false, i.e.
  // to simulate with precursor. Set this variable to true in a derived class in order to switch off
  // the precursor.
  bool switch_off_precursor;

private:
  // performs some additional parameter checks
  void
  consistency_checks() const
  {
    AssertThrow(precursor->get_parameters().ale_formulation == false,
                dealii::ExcMessage("not implemented."));
    AssertThrow(main->get_parameters().ale_formulation == false,
                dealii::ExcMessage("not implemented."));

    AssertThrow(precursor->get_parameters().calculation_of_time_step_size ==
                  main->get_parameters().calculation_of_time_step_size,
                dealii::ExcMessage(
                  "Type of time step calculation has to be the same for both domains."));

    AssertThrow(precursor->get_parameters().adaptive_time_stepping ==
                  main->get_parameters().adaptive_time_stepping,
                dealii::ExcMessage(
                  "Type of time step calculation has to be the same for both domains."));

    AssertThrow(precursor->get_parameters().solver_type == SolverType::Unsteady and
                  main->get_parameters().solver_type == SolverType::Unsteady,
                dealii::ExcMessage("This is an unsteady solver. Check parameters."));

    // For the two-domain solver the parameter start_with_low_order has to be true.
    // This is due to the fact that the setup function of the time integrator initializes
    // the solution at previous time instants t_0 - dt, t_0 - 2*dt, ... in case of
    // start_with_low_order == false. However, the combined time step size
    // is not known at this point since the two domains have to first communicate with each other
    // in order to find the minimum time step size. Hence, the easiest way to avoid these kind of
    // inconsistencies is to preclude the case start_with_low_order == false.
    AssertThrow(precursor->get_parameters().start_with_low_order == true and
                  main->get_parameters().start_with_low_order == true,
                dealii::ExcMessage("start_with_low_order has to be true for two-domain solver."));
  }
};

} // namespace Precursor
} // namespace IncNS
} // namespace ExaDG



#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_APPLICATION_BASE_PRECURSOR_H_ \
        */
