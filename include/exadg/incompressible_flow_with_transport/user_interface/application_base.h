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
#include <exadg/convection_diffusion/postprocessor/postprocessor.h>
#include <exadg/convection_diffusion/user_interface/boundary_descriptor.h>
#include <exadg/convection_diffusion/user_interface/field_functions.h>
#include <exadg/convection_diffusion/user_interface/input_parameters.h>
#include <exadg/grid/grid.h>
#include <exadg/incompressible_navier_stokes/user_interface/application_base.h>

namespace ExaDG
{
template<int>
class Mesh;

namespace FTI
{
using namespace dealii;

template<int dim, typename Number>
class ApplicationBase : public IncNS::ApplicationBase<dim, Number>
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

  ApplicationBase(std::string parameter_file) : IncNS::ApplicationBase<dim, Number>(parameter_file)
  {
  }

  virtual ~ApplicationBase()
  {
  }

  unsigned int
  get_n_scalars()
  {
    return this->n_scalars;
  }

  virtual void
  set_input_parameters_scalar(ConvDiff::InputParameters & parameters,
                              unsigned int const          scalar_index = 0) = 0;

  virtual void
  set_boundary_conditions_scalar(
    std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> boundary_descriptor,
    unsigned int const                                 scalar_index = 0) = 0;

  virtual void
  set_field_functions_scalar(std::shared_ptr<ConvDiff::FieldFunctions<dim>> field_functions,
                             unsigned int const                             scalar_index = 0) = 0;

  virtual std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>>
  construct_postprocessor_scalar(unsigned int const degree,
                                 MPI_Comm const &   mpi_comm,
                                 unsigned int const scalar_index = 0) = 0;

protected:
  std::string  output_directory = "output/", output_name = "output";
  bool         write_output = false;
  unsigned int n_scalars    = 1;
};

} // namespace FTI
} // namespace ExaDG


#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_FLOW_WITH_TRANSPORT_USER_INTERFACE_APPLICATION_BASE_H_ */
