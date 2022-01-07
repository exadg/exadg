#pragma once

#include <deal.II/base/parameter_handler.h>

using namespace dealii;

/**
 * This class declares all preCICE parameters, which can be specified in the
 * parameter file. The subsection abut preCICE configurations is directly
 * interlinked to the Adapter class.
 */
namespace Parameters
{ /**
   * @brief PreciceAdapterConfiguration: Specifies preCICE related information.
   *        A lot of these information need to be consistent with the
   *        precice-config.xml file.
   */
struct PreciceAdapterConfiguration
{
  std::string config_file              = "precice config-file";
  std::string physics                  = "Structure";
  std::string participant_name         = "exadg";
  std::string read_mesh_name           = "default";
  std::string write_mesh_name          = "default";
  int         write_quad_index         = 0;
  std::string write_data_specification = "values_on_quads";
  std::string read_data_name           = "received-data";
  std::string write_data_name          = "calculated-data";

  void
  add_parameters(ParameterHandler & prm);
};


void
PreciceAdapterConfiguration::add_parameters(ParameterHandler & prm)
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
                      Patterns::Selection("Structure|Fluid"));
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
      "Available options are: values_on_dofs, values_on_quads, normal_gradients_on_quads",
      Patterns::Selection("values_on_dofs|values_on_quads|normal_gradients_on_quads|"
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
} // namespace Parameters
