/*
 * postprocessor.h
 *
 *  Created on: May, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_

#include "postprocessor_base.h"

#include "../../postprocessor/error_calculation.h"
#include "../../postprocessor/kinetic_energy_spectrum.h"
#include "../../postprocessor/lift_and_drag_calculation.h"
#include "../../postprocessor/pressure_difference_calculation.h"

#include "divergence_and_mass_error.h"
#include "kinetic_energy_dissipation_detailed.h"
#include "line_plot_calculation.h"
#include "output_generator.h"

namespace IncNS
{
template<int dim>
struct PostProcessorData
{
  PostProcessorData()
  {
  }

  OutputData                     output_data;
  ErrorCalculationData<dim>      error_data_u;
  ErrorCalculationData<dim>      error_data_p;
  LiftAndDragData                lift_and_drag_data;
  PressureDifferenceData<dim>    pressure_difference_data;
  MassConservationData           mass_data;
  KineticEnergyData              kinetic_energy_data;
  KineticEnergySpectrumData      kinetic_energy_spectrum_data;
  LinePlotDataInstantaneous<dim> line_plot_data;
};

template<int dim, typename Number>
class PostProcessor : public PostProcessorBase<dim, Number>
{
public:
  typedef PostProcessorBase<dim, Number> Base;

  typedef typename Base::VectorType VectorType;

  typedef typename Base::Operator Operator;

  PostProcessor(PostProcessorData<dim> const & postprocessor_data, MPI_Comm const & mpi_comm);

  virtual ~PostProcessor();

  virtual void
  setup(Operator const & pde_operator);

  virtual void
  do_postprocessing(VectorType const & velocity,
                    VectorType const & pressure,
                    double const       time             = 0.0,
                    int const          time_step_number = -1);

protected:
  MPI_Comm const & mpi_comm;

private:
  PostProcessorData<dim> pp_data;

  // write output for visualization of results (e.g., using paraview)
  OutputGenerator<dim, Number> output_generator;

  // calculate errors for verification purposes for problems with known analytical solution
  ErrorCalculator<dim, Number> error_calculator_u;
  ErrorCalculator<dim, Number> error_calculator_p;

  // calculate lift and drag forces for flow around bodies
  LiftAndDragCalculator<dim, Number> lift_and_drag_calculator;

  // calculate pressure difference between two points, e.g., the leading and trailing edge of a body
  PressureDifferenceCalculator<dim, Number> pressure_difference_calculator;

  // calculate divergence and continuity errors as a measure of mass conservation (particularly
  // relevant for turbulent flows)
  DivergenceAndMassErrorCalculator<dim, Number> div_and_mass_error_calculator;

  // calculate kinetic energy as well as dissipation rates (particularly relevant for turbulent
  // flows)
  KineticEnergyCalculatorDetailed<dim, Number> kinetic_energy_calculator;

  // evaluate kinetic energy in spectral space (i.e., as a function of the wavenumber)
  KineticEnergySpectrumCalculator<dim, Number> kinetic_energy_spectrum_calculator;

  // evaluate quantities along lines through the domain
  LinePlotCalculator<dim, Number> line_plot_calculator;
};


} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_ */
