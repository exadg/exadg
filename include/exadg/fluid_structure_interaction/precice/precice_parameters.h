/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2022 by the ExaDG authors
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

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_PRECICE_PARAMETERS_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_PRECICE_PARAMETERS_H_

#include <deal.II/base/parameter_handler.h>

// ExaDG
#include <exadg/fluid_structure_interaction/precice/coupling_base.h>
#include <exadg/utilities/enum_utilities.h>

namespace ExaDG
{
namespace preCICE
{
/**
 * This class declares all preCICE parameters, which can be specified in the
 * parameter file. A lot of these information need to be consistent with the
 * precice-config.xml file.
 */
struct ConfigurationParameters
{
  ConfigurationParameters() = default;

  ConfigurationParameters(std::string const & input_file);

  std::string config_file              = "precice config-file";
  std::string physics                  = "undefined";
  std::string participant_name         = "exadg";
  std::string read_mesh_name           = "default";
  std::string write_mesh_name          = "default";
  std::string ale_mesh_name            = "default";
  std::string write_data_specification = "values_on_q_points";
  std::string velocity_data_name       = "default";
  std::string displacement_data_name   = "default";
  std::string stress_data_name         = "default";

  WriteDataType write_data_type = WriteDataType::undefined;

  void
  add_parameters(dealii::ParameterHandler & prm);
};



ConfigurationParameters::ConfigurationParameters(std::string const & input_file)
{
  dealii::ParameterHandler prm;
  add_parameters(prm);
  prm.parse_input(input_file, "", true, true);

  Utilities::string_to_enum(write_data_type, write_data_specification);
}



void
ConfigurationParameters::add_parameters(dealii::ParameterHandler & prm)
{
  prm.enter_subsection("preciceConfiguration");
  {
    prm.add_parameter("preciceConfigFile",
                      config_file,
                      "Name of the precice configuration file",
                      dealii::Patterns::Anything());
    prm.add_parameter("Physics",
                      physics,
                      "Specify the side you want to compute (Fluid vs Structure)",
                      dealii::Patterns::Selection("Structure|Fluid|undefined"));
    prm.add_parameter("ParticipantName",
                      participant_name,
                      "Name of the participant in the precice-config.xml file",
                      dealii::Patterns::Anything());
    prm.add_parameter("ReadMeshName",
                      read_mesh_name,
                      "Name of the read coupling mesh in the precice-config.xml file",
                      dealii::Patterns::Anything());
    prm.add_parameter("WriteMeshName",
                      write_mesh_name,
                      "Name of the write coupling mesh in the precice-config.xml file",
                      dealii::Patterns::Anything());
    prm.add_parameter("ALEMeshName",
                      ale_mesh_name,
                      "Name of the ale-mesh in the precice-config.xml file",
                      dealii::Patterns::Anything());
    prm.add_parameter("WriteDataSpecification",
                      write_data_specification,
                      "Specification of the write data location and the data type",
                      dealii::Patterns::Selection(
                        "values_on_dofs|values_on_q_points|normal_gradients_on_q_points|"
                        "values_on_other_mesh|gradients_on_other_mesh"));
    prm.add_parameter("VelocityDataName",
                      velocity_data_name,
                      "Name of the Velocity data in the precice-config.xml file",
                      dealii::Patterns::Anything());
    prm.add_parameter("DisplacementDataName",
                      displacement_data_name,
                      "Name of the Displacement data in the precice-config.xml file",
                      dealii::Patterns::Anything());
    prm.add_parameter("StressDataName",
                      stress_data_name,
                      "Name of the Stress data in the precice-config.xml file",
                      dealii::Patterns::Anything());
  }
  prm.leave_subsection();
}

} // namespace preCICE
} // namespace ExaDG

#endif
