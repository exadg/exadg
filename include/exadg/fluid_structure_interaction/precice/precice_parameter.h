#pragma once

#include <deal.II/base/parameter_handler.h>
#include <exadg/fluid_structure_interaction/precice/coupling_interface.h>

namespace ExaDG
{
namespace preCICE
{
using namespace dealii;

/**
 * This class declares all preCICE parameters, which can be specified in the
 * parameter file. The subsection abut preCICE configurations is directly
 * interlinked to the Adapter class.
 */

/**
 * @brief ConfigurationParameters: Specifies preCICE related information.
 *        A lot of these information need to be consistent with the
 *        precice-config.xml file.
 */
struct ConfigurationParameters
{
  ConfigurationParameters() = default;

  ConfigurationParameters(const std::string & input_file);

  std::string config_file              = "precice config-file";
  std::string physics                  = "undefined";
  std::string participant_name         = "exadg";
  std::string read_mesh_name           = "default";
  std::string write_mesh_name          = "default";
  int         write_quad_index         = 0;
  std::string write_data_specification = "values_on_q_points";
  std::string read_data_name           = "received-data";
  std::string write_data_name          = "calculated-data";

  WriteDataType write_data_type = WriteDataType::undefined;

  void
  add_parameters(ParameterHandler & prm);

  void
  string_to_enum(WriteDataType & enum_out, const std::string & string_in);
};



ConfigurationParameters::ConfigurationParameters(const std::string & input_file)
{
  dealii::ParameterHandler prm;
  add_parameters(prm);
  prm.parse_input(input_file, "", true, true);

  string_to_enum(write_data_type, write_data_specification);
}



void
ConfigurationParameters::add_parameters(ParameterHandler & prm)
{
  prm.enter_subsection("preciceConfiguration");
  {
    prm.add_parameter("preciceConfigFile",
                      config_file,
                      "Name of the precice configuration file",
                      Patterns::Anything());
    prm.add_parameter("Physics",
                      physics,
                      "Specify the side you want to compute (Fluid vs Structure)",
                      Patterns::Selection("Structure|Fluid|undefined"));
    prm.add_parameter("ParticipantName",
                      participant_name,
                      "Name of the participant in the precice-config.xml file",
                      Patterns::Anything());
    prm.add_parameter("ReadMeshName",
                      read_mesh_name,
                      "Name of the read coupling mesh in the precice-config.xml file",
                      Patterns::Anything());
    prm.add_parameter("WriteMeshName",
                      write_mesh_name,
                      "Name of the write coupling mesh in the precice-config.xml file",
                      Patterns::Anything());
    prm.add_parameter("WriteQuadratureIndex",
                      write_quad_index,
                      "Index of the quadrature formula in MatrixFree used for initialization",
                      Patterns::Integer(0));
    prm.add_parameter(
      "WriteDataSpecification",
      write_data_specification,
      "Specification of the write data location and the data type"
      "Available options are: values_on_dofs, values_on_q_points, normal_gradients_on_q_points",
      Patterns::Selection("values_on_dofs|values_on_q_points|normal_gradients_on_q_points|"
                          "values_on_other_mesh|gradients_on_other_mesh"));
    prm.add_parameter("ReadDataName",
                      read_data_name,
                      "Name of the read data in the precice-config.xml file",
                      Patterns::Anything());
    prm.add_parameter("WriteDataName",
                      write_data_name,
                      "Name of the write data in the precice-config.xml file",
                      Patterns::Anything());
  }
  prm.leave_subsection();
}

void
ConfigurationParameters::string_to_enum(WriteDataType & enum_out, const std::string & string_in)
{
  if(string_in == "values_on_dofs")
    enum_out = WriteDataType::values_on_dofs;
  else if(string_in == "values_on_other_mesh")
    enum_out = WriteDataType::values_on_other_mesh;
  else if(string_in == "gradients_on_other_mesh")
    enum_out = WriteDataType::gradients_on_other_mesh;
  else if(string_in == "values_on_q_points")
    enum_out = WriteDataType::values_on_q_points;
  else if(string_in == "normal_gradients_on_q_points")
    enum_out = WriteDataType::normal_gradients_on_q_points;
  else
    AssertThrow(false, ExcMessage("Unknwon write data type."));
}

} // namespace preCICE
} // namespace ExaDG