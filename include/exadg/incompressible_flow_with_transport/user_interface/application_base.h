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
#include <exadg/convection_diffusion/user_interface/parameters.h>
#include <exadg/grid/grid.h>
#include <exadg/incompressible_navier_stokes/user_interface/application_base.h>

namespace ExaDG
{
template<int>
class Mesh;

namespace FTI
{
template<int dim, typename Number>
class ApplicationBase : public IncNS::ApplicationBase<dim, Number>
{
public:
  virtual void
  add_parameters(dealii::ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("Output");
      prm.add_parameter("OutputDirectory",  output_directory, "Directory where output is written.");
      prm.add_parameter("OutputName",       output_name,      "Name of output files.");
      prm.add_parameter("WriteOutput",      write_output,     "Decides whether vtu output is written.");
    prm.leave_subsection();
    // clang-format on
  }

  ApplicationBase(std::string parameter_file, MPI_Comm const & comm, unsigned int n_scalar_fields)
    : IncNS::ApplicationBase<dim, Number>(parameter_file, comm)
  {
    n_scalars = n_scalar_fields;
  }

  virtual ~ApplicationBase()
  {
  }

  void
  set_parameters_convergence_study(unsigned int const degree, unsigned int const refine_space)
  {
    // fluid
    this->param.degree_u             = degree;
    this->param.grid.n_refine_global = refine_space;

    // scalar transport
    for(unsigned int i = 0; i < scalar_param.size(); ++i)
      this->scalar_param[i].degree = degree;
  }

  void
  setup() final
  {
    IncNS::ApplicationBase<dim, Number>::setup();

    scalar_param.resize(n_scalars);
    scalar_boundary_descriptor.resize(n_scalars);
    scalar_field_functions.resize(n_scalars);

    // parameters scalar
    for(unsigned int i = 0; i < n_scalars; ++i)
    {
      set_parameters_scalar(i);
      scalar_param[i].check();

      // some additional parameter checks
      AssertThrow(scalar_param[i].ale_formulation == this->param.ale_formulation,
                  dealii::ExcMessage(
                    "Parameter ale_formulation is different for fluid field and scalar field"));

      AssertThrow(
        scalar_param[i].adaptive_time_stepping == this->param.adaptive_time_stepping,
        dealii::ExcMessage(
          "The option adaptive_time_stepping has to be consistent for fluid and scalar transport solvers."));

      scalar_param[i].print(this->pcout,
                            "List of parameters for scalar quantity " +
                              dealii::Utilities::to_string(i) + ":");

      // boundary conditions
      scalar_boundary_descriptor[i] = std::make_shared<ConvDiff::BoundaryDescriptor<dim>>();
      set_boundary_descriptor_scalar(i);
      verify_boundary_conditions(*scalar_boundary_descriptor[i], *this->grid);

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

protected:
  std::vector<ConvDiff::Parameters>                               scalar_param;
  std::vector<std::shared_ptr<ConvDiff::FieldFunctions<dim>>>     scalar_field_functions;
  std::vector<std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>>> scalar_boundary_descriptor;

  std::string  output_directory = "output/", output_name = "output";
  bool         write_output = false;
  unsigned int n_scalars    = 1;

private:
  virtual void
  set_parameters_scalar(unsigned int const scalar_index = 0) = 0;

  virtual void
  set_boundary_descriptor_scalar(unsigned int const scalar_index = 0) = 0;

  virtual void
  set_field_functions_scalar(unsigned int const scalar_index = 0) = 0;
};

} // namespace FTI
} // namespace ExaDG


#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_FLOW_WITH_TRANSPORT_USER_INTERFACE_APPLICATION_BASE_H_ */
