/*
 * write_output.h
 *
 *  Created on: 2018
 *      Author: fehn
 */

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_WRITE_OUTPUT_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_WRITE_OUTPUT_H_

// postprocessor
#include "../../postprocessor/output_data.h"
#include "../../postprocessor/solution_field.h"

namespace CompNS
{
struct OutputData : public OutputDataBase
{
  OutputData()
    : write_velocity(false),
      write_pressure(false),
      write_temperature(false),
      write_vorticity(false),
      write_divergence(false),
      write_processor_id(false)
  {
  }

  void
  print(ConditionalOStream & pcout, bool unsteady)
  {
    OutputDataBase::print(pcout, unsteady);

    print_parameter(pcout, "Write velocity", write_velocity);
    print_parameter(pcout, "Write pressure", write_pressure);
    print_parameter(pcout, "Write temperature", write_temperature);
    print_parameter(pcout, "Write vorticity", write_vorticity);
    print_parameter(pcout, "Write divergence", write_divergence);
    print_parameter(pcout, "Write processor ID", write_processor_id);
  }

  // write velocity
  bool write_velocity;

  // write pressure
  bool write_pressure;

  // write temperature
  bool write_temperature;

  // write vorticity of velocity field
  bool write_vorticity;

  // write divergence of velocity field
  bool write_divergence;

  // write processor ID to scalar field in order to visualize the
  // distribution of cells to processors
  bool write_processor_id;
};

template<int dim, typename Number>
class OutputGenerator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  OutputGenerator();

  void
  setup(DoFHandler<dim> const & dof_handler_in,
        Mapping<dim> const &    mapping_in,
        OutputData const &      output_data_in);

  void
  evaluate(VectorType const &                              solution_conserved,
           std::vector<SolutionField<dim, Number>> const & additional_fields,
           double const &                                  time,
           int const &                                     time_step_number);

private:
  unsigned int output_counter;
  bool         reset_counter;

  SmartPointer<DoFHandler<dim> const> dof_handler;
  SmartPointer<Mapping<dim> const>    mapping;
  OutputData                          output_data;
};

} // namespace CompNS


#endif /* INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_WRITE_OUTPUT_H_ */
