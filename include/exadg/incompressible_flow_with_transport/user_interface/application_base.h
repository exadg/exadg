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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_FLOW_WITH_TRANSPORT_USER_INTERFACE_APPLICATION_BASE_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_FLOW_WITH_TRANSPORT_USER_INTERFACE_APPLICATION_BASE_H_

// ExaDG
#include <exadg/grid/grid.h>
#include <exadg/grid/grid_utilities.h>

#include <exadg/incompressible_navier_stokes/postprocessor/postprocessor.h>
#include <exadg/incompressible_navier_stokes/user_interface/boundary_descriptor.h>
#include <exadg/incompressible_navier_stokes/user_interface/field_functions.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>

#include <exadg/convection_diffusion/postprocessor/postprocessor.h>
#include <exadg/convection_diffusion/user_interface/boundary_descriptor.h>
#include <exadg/convection_diffusion/user_interface/field_functions.h>
#include <exadg/convection_diffusion/user_interface/parameters.h>

#include <exadg/postprocessor/output_parameters.h>
#include <exadg/utilities/resolution_parameters.h>

namespace ExaDG
{
namespace FTI
{
template<int dim, typename Number>
class Fluid
{
public:
  virtual void
  add_parameters(dealii::ParameterHandler & prm)
  {
    output_parameters.add_parameters(prm);
  }

  Fluid(std::string parameter_file, MPI_Comm const & comm)
    : mpi_comm(comm),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
      parameter_file(parameter_file)
  {
    grid = std::make_shared<Grid<dim>>();
  }

  virtual ~Fluid()
  {
  }

  virtual void
  setup()
  {
    parse_parameters();

    set_parameters();
    param.check(pcout);
    param.print(pcout, "List of parameters:");

    // grid
    GridUtilities::create_mapping(mapping, param.grid.element_type, param.mapping_degree);
    create_grid();
    print_grid_info(pcout, *grid);

    // boundary conditions
    boundary_descriptor = std::make_shared<IncNS::BoundaryDescriptor<dim>>();
    set_boundary_descriptor();
    verify_boundary_conditions<dim>(*boundary_descriptor, *grid);

    // field functions
    field_functions = std::make_shared<IncNS::FieldFunctions<dim>>();
    set_field_functions();
  }

  virtual std::shared_ptr<IncNS::PostProcessorBase<dim, Number>>
  create_postprocessor() = 0;

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

  std::shared_ptr<dealii::Mapping<dim> const>
  get_mapping() const
  {
    return mapping;
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
  parse_parameters()
  {
    dealii::ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(parameter_file, "", true, true);
  }

  MPI_Comm const & mpi_comm;

  dealii::ConditionalOStream pcout;

  IncNS::Parameters param;

  std::shared_ptr<Grid<dim>> grid;

  std::shared_ptr<dealii::Mapping<dim>> mapping;

  std::shared_ptr<IncNS::FieldFunctions<dim>>     field_functions;
  std::shared_ptr<IncNS::BoundaryDescriptor<dim>> boundary_descriptor;

  std::string parameter_file;

  OutputParameters output_parameters;

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
  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    fluid->add_parameters(prm);

    resolution.add_parameters(prm);
  }

  ApplicationBase(std::string parameter_file, MPI_Comm const & comm, unsigned int n_scalar_fields)
    : mpi_comm(comm),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0),
      parameter_file(parameter_file)
  {
    n_scalars = n_scalar_fields;
  }

  virtual ~ApplicationBase()
  {
  }

  void
  setup()
  {
    // TODO
    //    set_resolution_parameters();

    fluid->setup();

    scalar_param.resize(n_scalars);
    scalar_boundary_descriptor.resize(n_scalars);
    scalar_field_functions.resize(n_scalars);

    // parameters scalar
    for(unsigned int i = 0; i < n_scalars; ++i)
    {
      set_parameters_scalar(i);
      scalar_param[i].check();

      // some additional parameter checks
      AssertThrow(scalar_param[i].ale_formulation == fluid->get_parameters().ale_formulation,
                  dealii::ExcMessage(
                    "Parameter ale_formulation is different for fluid field and scalar field"));

      AssertThrow(
        scalar_param[i].adaptive_time_stepping == fluid->get_parameters().adaptive_time_stepping,
        dealii::ExcMessage(
          "The option adaptive_time_stepping has to be consistent for fluid and scalar transport solvers."));

      scalar_param[i].print(this->pcout,
                            "List of parameters for scalar quantity " +
                              dealii::Utilities::to_string(i) + ":");

      // boundary conditions
      scalar_boundary_descriptor[i] = std::make_shared<ConvDiff::BoundaryDescriptor<dim>>();
      set_boundary_descriptor_scalar(i);
      verify_boundary_conditions(*scalar_boundary_descriptor[i], *fluid->get_grid());

      // field functions
      scalar_field_functions[i] = std::make_shared<ConvDiff::FieldFunctions<dim>>();
      set_field_functions_scalar(i);
    }
  }

  unsigned int
  get_n_scalars()
  {
    return this->n_scalars;
  }

  virtual std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>>
  create_postprocessor_scalar(unsigned int const scalar_index = 0) = 0;

  ConvDiff::Parameters const &
  get_parameters_scalar(unsigned int const scalar_index = 0) const
  {
    return scalar_param[scalar_index];
  }

  std::shared_ptr<ConvDiff::BoundaryDescriptor<dim> const>
  get_boundary_descriptor_scalar(unsigned int const scalar_index = 0) const
  {
    return scalar_boundary_descriptor[scalar_index];
  }

  std::shared_ptr<ConvDiff::FieldFunctions<dim> const>
  get_field_functions_scalar(unsigned int const scalar_index = 0) const
  {
    return scalar_field_functions[scalar_index];
  }

  std::shared_ptr<Fluid<dim, Number>> fluid;

protected:
  MPI_Comm const & mpi_comm;

  dealii::ConditionalOStream pcout;

  std::vector<ConvDiff::Parameters>                               scalar_param;
  std::vector<std::shared_ptr<ConvDiff::FieldFunctions<dim>>>     scalar_field_functions;
  std::vector<std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>>> scalar_boundary_descriptor;

  unsigned int n_scalars = 1;

private:
  // TODO
  //  void
  //  set_resolution_parameters()
  //  {
  //    // fluid
  //    this->param.degree_u             = resolution.degree;
  //    this->param.grid.n_refine_global = resolution.refine_space;
  //
  //    // scalar transport
  //    for(unsigned int i = 0; i < scalar_param.size(); ++i)
  //      this->scalar_param[i].degree = resolution.degree;
  //  }

  virtual void
  set_parameters_scalar(unsigned int const scalar_index = 0) = 0;

  virtual void
  set_boundary_descriptor_scalar(unsigned int const scalar_index = 0) = 0;

  virtual void
  set_field_functions_scalar(unsigned int const scalar_index = 0) = 0;

  std::string parameter_file;

  ResolutionParameters resolution;
};

} // namespace FTI
} // namespace ExaDG


#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_FLOW_WITH_TRANSPORT_USER_INTERFACE_APPLICATION_BASE_H_ */
