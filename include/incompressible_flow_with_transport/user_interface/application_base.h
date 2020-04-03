/*
 * application_base.h
 *
 *  Created on: 31.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_FLOW_WITH_TRANSPORT_USER_INTERFACE_APPLICATION_BASE_H_
#define INCLUDE_INCOMPRESSIBLE_FLOW_WITH_TRANSPORT_USER_INTERFACE_APPLICATION_BASE_H_

// NAVIER-STOKES
#include "../../incompressible_navier_stokes/user_interface/application_base.h"

// CONVECTION-DIFFUSION

// postprocessor
#include "../../convection_diffusion/postprocessor/postprocessor.h"

// user interface
#include "../../convection_diffusion/user_interface/boundary_descriptor.h"
#include "../../convection_diffusion/user_interface/field_functions.h"
#include "../../convection_diffusion/user_interface/input_parameters.h"

using namespace dealii;

namespace FTI
{
template<int dim, typename Number>
class ApplicationBase : public IncNS::ApplicationBase<dim, Number>
{
public:
  virtual void
  add_parameters(ParameterHandler & prm)
  {
    (void)prm;

    // can be overwritten by derived classes and is for example necessary
    // in order to generate a default input file
  }

  ApplicationBase(std::string parameter_file) : IncNS::ApplicationBase<dim, Number>(parameter_file)
  {
  }

  virtual ~ApplicationBase()
  {
  }

  virtual void
  set_input_parameters_scalar(ConvDiff::InputParameters & parameters,
                              unsigned int const          scalar_index = 0) = 0;

  virtual void set_boundary_conditions_scalar(
    std::shared_ptr<ConvDiff::BoundaryDescriptor<0, dim>> boundary_descriptor,
    unsigned int const                                    scalar_index = 0) = 0;

  virtual void
  set_field_functions_scalar(std::shared_ptr<ConvDiff::FieldFunctions<dim>> field_functions,
                             unsigned int const                             scalar_index = 0) = 0;

  virtual std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>>
  construct_postprocessor_scalar(unsigned int const degree,
                                 MPI_Comm const &   mpi_comm,
                                 unsigned int const scalar_index = 0) = 0;
};

} // namespace FTI



#endif /* INCLUDE_INCOMPRESSIBLE_FLOW_WITH_TRANSPORT_USER_INTERFACE_APPLICATION_BASE_H_ */
