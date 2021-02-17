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

#ifndef INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_
#define INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_

#include <exadg/compressible_navier_stokes/postprocessor/output_generator.h>
#include <exadg/compressible_navier_stokes/postprocessor/postprocessor_base.h>
#include <exadg/postprocessor/error_calculation.h>
#include <exadg/postprocessor/kinetic_energy_calculation.h>
#include <exadg/postprocessor/kinetic_energy_spectrum.h>
#include <exadg/postprocessor/lift_and_drag_calculation.h>
#include <exadg/postprocessor/pressure_difference_calculation.h>

namespace ExaDG
{
namespace CompNS
{
using namespace dealii;

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

  PostProcessor(PostProcessorData<dim> const & postprocessor_data, MPI_Comm const & comm);

  virtual ~PostProcessor();

  virtual void
  setup(Operator<dim, Number> const & pde_operator);

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

  MPI_Comm const & mpi_comm;

  PostProcessorData<dim> pp_data;

  SmartPointer<Operator<dim, Number> const> navier_stokes_operator;

  OutputGenerator<dim, Number>                 output_generator;
  ErrorCalculator<dim, Number>                 error_calculator;
  LiftAndDragCalculator<dim, Number>           lift_and_drag_calculator;
  PressureDifferenceCalculator<dim, Number>    pressure_difference_calculator;
  KineticEnergyCalculator<dim, Number>         kinetic_energy_calculator;
  KineticEnergySpectrumCalculator<dim, Number> kinetic_energy_spectrum_calculator;
};

} // namespace CompNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_ */
