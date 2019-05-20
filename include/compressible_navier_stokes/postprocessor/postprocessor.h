/*
 * postprocessor.h
 *
 *  Created on: 2018
 *      Author: fehn
 */

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_

#include "../../postprocessor/error_calculation.h"
#include "../../postprocessor/kinetic_energy_calculation.h"
#include "../../postprocessor/kinetic_energy_spectrum.h"
#include "../../postprocessor/lift_and_drag_calculation.h"
#include "../../postprocessor/pressure_difference_calculation.h"

#include "postprocessor_base.h"
#include "write_output.h"

namespace CompNS
{
template<int dim>
struct PostProcessorData
{
  PostProcessorData() : calculate_velocity(false), calculate_pressure(false)
  {
  }

  bool calculate_velocity;
  bool calculate_pressure;

  OutputData                  output_data;
  ErrorCalculationData<dim>   error_data;
  LiftAndDragData             lift_and_drag_data;
  PressureDifferenceData<dim> pressure_difference_data;
  KineticEnergyData           kinetic_energy_data;
  KineticEnergySpectrumData   kinetic_energy_spectrum_data;
};

template<int dim, typename Number>
class PostProcessor : public PostProcessorBase<dim, Number>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  PostProcessor(PostProcessorData<dim> const & postprocessor_data);

  virtual ~PostProcessor();

  virtual void
  setup(DGOperator<dim, Number> const & navier_stokes_operator_in,
        DoFHandler<dim> const &         dof_handler_in,
        DoFHandler<dim> const &         dof_handler_vector_in,
        DoFHandler<dim> const &         dof_handler_scalar_in,
        Mapping<dim> const &            mapping_in,
        MatrixFree<dim, Number> const & matrix_free_data_in);

  virtual void
  do_postprocessing(VectorType const & solution, double const time, int const time_step_number);

protected:
  // DoF vectors for derived quantities: (p, u, T)
  VectorType pressure;
  VectorType velocity;
  VectorType temperature;
  VectorType vorticity;
  VectorType divergence;

  std::vector<SolutionField<dim, Number>> additional_fields;

private:
  void
  initialize_additional_vectors();

  void
  calculate_additional_vectors(VectorType const & solution);

  PostProcessorData<dim> pp_data;

  SmartPointer<DGOperator<dim, Number> const> navier_stokes_operator;

  OutputGenerator<dim, Number>                 output_generator;
  ErrorCalculator<dim, Number>                 error_calculator;
  LiftAndDragCalculator<dim, Number>           lift_and_drag_calculator;
  PressureDifferenceCalculator<dim, Number>    pressure_difference_calculator;
  KineticEnergyCalculator<dim, Number>         kinetic_energy_calculator;
  KineticEnergySpectrumCalculator<dim, Number> kinetic_energy_spectrum_calculator;
};

} // namespace CompNS

#endif /* INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_ */
