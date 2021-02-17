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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_

#include <exadg/incompressible_navier_stokes/postprocessor/divergence_and_mass_error.h>
#include <exadg/incompressible_navier_stokes/postprocessor/kinetic_energy_dissipation_detailed.h>
#include <exadg/incompressible_navier_stokes/postprocessor/line_plot_calculation.h>
#include <exadg/incompressible_navier_stokes/postprocessor/output_generator.h>
#include <exadg/incompressible_navier_stokes/postprocessor/postprocessor_base.h>
#include <exadg/postprocessor/error_calculation.h>
#include <exadg/postprocessor/kinetic_energy_spectrum.h>
#include <exadg/postprocessor/lift_and_drag_calculation.h>
#include <exadg/postprocessor/pressure_difference_calculation.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

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
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_ */
