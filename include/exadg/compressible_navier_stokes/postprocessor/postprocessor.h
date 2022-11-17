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
#include <exadg/compressible_navier_stokes/postprocessor/pointwise_output_generator.h>
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
template<int dim>
struct PostProcessorData
{
  OutputData                  output_data;
  PointwiseOutputData<dim>    pointwise_output_data;
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
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  PostProcessor(PostProcessorData<dim> const & postprocessor_data, MPI_Comm const & comm);

  virtual ~PostProcessor();

  void
  setup(Operator<dim, Number> const & pde_operator) override;

  void
  do_postprocessing(VectorType const &     solution,
                    double const           time,
                    types::time_step const time_step_number) override;

protected:
  SolutionField<dim, Number> pressure;
  SolutionField<dim, Number> velocity;
  SolutionField<dim, Number> temperature;
  SolutionField<dim, Number> vorticity;
  SolutionField<dim, Number> divergence;

  std::vector<dealii::SmartPointer<SolutionField<dim, Number>>> additional_fields_vtu;

private:
  void
  initialize_additional_vectors();

  void
  reinit_additional_fields(VectorType const & solution);

  MPI_Comm const mpi_comm;

  PostProcessorData<dim> pp_data;

  dealii::SmartPointer<Operator<dim, Number> const> navier_stokes_operator;

  OutputGenerator<dim, Number>                 output_generator;
  PointwiseOutputGenerator<dim, Number>        pointwise_output_generator;
  ErrorCalculator<dim, Number>                 error_calculator;
  LiftAndDragCalculator<dim, Number>           lift_and_drag_calculator;
  PressureDifferenceCalculator<dim, Number>    pressure_difference_calculator;
  KineticEnergyCalculator<dim, Number>         kinetic_energy_calculator;
  KineticEnergySpectrumCalculator<dim, Number> kinetic_energy_spectrum_calculator;
};

} // namespace CompNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_ */
