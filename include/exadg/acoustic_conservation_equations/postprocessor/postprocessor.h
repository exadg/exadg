/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_POSTPROCESSOR_POSTPROCESSOR_H_
#define EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_POSTPROCESSOR_POSTPROCESSOR_H_

#include <exadg/acoustic_conservation_equations/postprocessor/output_generator.h>
#include <exadg/acoustic_conservation_equations/postprocessor/postprocessor_base.h>
#include <exadg/acoustic_conservation_equations/postprocessor/sound_energy_calculator.h>
#include <exadg/postprocessor/error_calculation.h>


namespace ExaDG
{
namespace Acoustics
{
template<int dim>
struct PostProcessorData
{
  PostProcessorData() = default;

  OutputData                output_data;
  ErrorCalculationData<dim> error_data_p;
  ErrorCalculationData<dim> error_data_u;
  SoundEnergyCalculatorData sound_energy_data;
};

template<int dim, typename Number>
class PostProcessor : public PostProcessorBase<dim, Number>
{
public:
  using Base = PostProcessorBase<dim, Number>;

  using BlockVectorType = typename Base::BlockVectorType;

  using AcousticsOperator = typename Base::AcousticsOperator;

  static unsigned int const block_index_pressure = AcousticsOperator::block_index_pressure;
  static unsigned int const block_index_velocity = AcousticsOperator::block_index_velocity;

  PostProcessor(PostProcessorData<dim> const & postprocessor_data, MPI_Comm const & mpi_comm);

  void
  setup(AcousticsOperator const & pde_operator) final;

  void
  do_postprocessing(BlockVectorType const & solution,
                    double const            time             = 0.0,
                    types::time_step const  time_step_number = numbers::steady_timestep) final;

protected:
  MPI_Comm const mpi_comm;

private:
  PostProcessorData<dim> pp_data;

  // write output for visualization of results (e.g., using paraview)
  OutputGenerator<dim, Number> output_generator;

  // calculate errors for verification purposes for problems with known analytical solution
  ErrorCalculator<dim, Number> error_calculator_p;
  ErrorCalculator<dim, Number> error_calculator_u;

  // calculates the sound energy in the computational domain
  SoundEnergyCalculator<dim, Number> sound_energy_calculator;
};



} // namespace Acoustics
} // namespace ExaDG


#endif /*EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_POSTPROCESSOR_POSTPROCESSOR_H_*/
