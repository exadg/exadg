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


#ifndef EXADG_ACOUSTIC_CONSERVATION_LAWS_POSTPROCESSOR_POINTWISE_SOUND_ENERGY_CALCULATIOR_H_
#define EXADG_ACOUSTIC_CONSERVATION_LAWS_POSTPROCESSOR_POINTWISE_SOUND_ENERGY_CALCULATIOR_H_

#include <deal.II/matrix_free/matrix_free.h>

#include <exadg/matrix_free/integrators.h>
#include <exadg/utilities/create_directories.h>
#include <exadg/utilities/print_functions.h>

#include <fstream>

namespace ExaDG
{
namespace Acoustics
{
struct SoundEnergyCalculatorData
{
  SoundEnergyCalculatorData()
    : directory("output/"),
      filename("sound_energy.csv"),
      clear_file(true),
      density(-1.0),
      speed_of_sound(-1.0)
  {
  }

  void
  print(dealii::ConditionalOStream & pcout)
  {
    if(time_control_data.is_active)
    {
      pcout << std::endl << "  Calculate sound energy:" << std::endl;

      // only implemented for unsteady problem
      time_control_data.print(pcout, true /*unsteady*/);

      print_parameter(pcout, "Directory of output files", directory);
      print_parameter(pcout, "Filename", filename);
    }
  }

  TimeControlData time_control_data;

  // directory and filename
  std::string directory;
  std::string filename;
  bool        clear_file;

  double density;
  double speed_of_sound;
};

template<int dim, typename Number>
class SoundEnergyCalculator
{
  using This = SoundEnergyCalculator<dim, Number>;

  using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<Number>;

  using scalar = dealii::VectorizedArray<Number>;
  using vector = dealii::Tensor<1, dim, scalar>;

  using CellIntegratorU = CellIntegrator<dim, dim, Number>;
  using CellIntegratorP = CellIntegrator<dim, 1, Number>;

public:
  SoundEnergyCalculator(MPI_Comm const & comm)
    : mpi_comm(comm),
      clear_files(true),
      matrix_free(nullptr),
      dof_index_pressure(0),
      dof_index_velocity(1),
      quad_index(0),
      block_index_pressure(0),
      block_index_velocity(1),
      rho(-1.0),
      c(-1.0),
      rhocc_inv(-1.0)
  {
  }

  void
  setup(dealii::MatrixFree<dim, Number> const & matrix_free_in,
        SoundEnergyCalculatorData const &       data_in,
        unsigned int const                      dof_index_pressure_in,
        unsigned int const                      dof_index_velocity_in,
        unsigned int const                      quad_index_in,
        unsigned int const                      block_index_pressure_in,
        unsigned int const                      block_index_velocity_in)
  {
    time_control.setup(data_in.time_control_data);

    matrix_free = &matrix_free_in;
    data        = data_in;

    dof_index_pressure   = dof_index_pressure_in;
    dof_index_velocity   = dof_index_velocity_in;
    quad_index           = quad_index_in;
    block_index_pressure = block_index_pressure_in;
    block_index_velocity = block_index_velocity_in;

    rho       = static_cast<Number>(data_in.density);
    c         = static_cast<Number>(data_in.speed_of_sound);
    rhocc_inv = static_cast<Number>(
      1.0 / (data_in.density * data_in.speed_of_sound * data_in.speed_of_sound));

    clear_files = data_in.clear_file;

    if(data_in.time_control_data.is_active)
    {
      AssertThrow(data_in.density > 0.0 && data_in.speed_of_sound > 0.0,
                  dealii::ExcMessage("Material parameters not set in SoundEnergyCalculatorData."));

      create_directories(data_in.directory, mpi_comm);
    }
  }

  void
  evaluate(BlockVectorType const & solution, double const & time, bool const unsteady)
  {
    AssertThrow(unsteady,
                dealii::ExcMessage(
                  "This postprocessing tool can only be used for unsteady problems."));

    do_evaluate(solution, time);
  }

  TimeControl time_control;

private:
  void
  do_evaluate(BlockVectorType const & solution, double const time)
  {
    double sound_energy = calculate_sound_energy(solution);

    // write output file
    if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      std::ostringstream filename;
      filename << data.directory + data.filename;

      std::ofstream f;
      if(clear_files == true)
      {
        f.open(filename.str().c_str(), std::ios::trunc);
        f << "Time, Sound energy: E = (1,p*p/(2*rho*c*c)+rho*u*u/2)_Omega" << std::endl;
        clear_files = false;
      }
      else
      {
        f.open(filename.str().c_str(), std::ios::app);
      }

      unsigned int precision = 12;
      f << std::scientific << std::setprecision(precision) << std::setw(precision + 8) << time
        << ", " << std::setw(precision + 8) << sound_energy << std::endl;
    }
  }


  /*
   *  This function calculates the sound energy
   *
   *  Sound energy: E = (1,p*p/(2*rho*c*c)+rho*u*u/2)_Omega
   *
   */
  double
  calculate_sound_energy(BlockVectorType const & solution) const
  {
    Number energy = 0.;
    matrix_free->cell_loop(&This::cell_loop, this, energy, solution);

    // sum over all MPI processes
    double const sound_energy = dealii::Utilities::MPI::sum(energy, mpi_comm);

    return sound_energy;
  }

  void
  cell_loop(dealii::MatrixFree<dim, Number> const &       matrix_free_in,
            Number &                                      dst,
            BlockVectorType const &                       src,
            std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    CellIntegratorP pressure(matrix_free_in, dof_index_pressure, quad_index);
    CellIntegratorU velocity(matrix_free_in, dof_index_velocity, quad_index);

    Number energy = 0.0;

    // Loop over all elements
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      pressure.reinit(cell);
      velocity.reinit(cell);
      pressure.read_dof_values(src.block(block_index_pressure));
      velocity.read_dof_values(src.block(block_index_velocity));
      pressure.evaluate(dealii::EvaluationFlags::values);
      velocity.evaluate(dealii::EvaluationFlags::values);

      scalar energy_batch = dealii::make_vectorized_array<Number>(0.);

      for(unsigned int q = 0; q < pressure.n_q_points; ++q)
      {
        vector u = velocity.get_value(q);
        scalar p = pressure.get_value(q);
        energy_batch += Number{0.5} * rho * u * u * velocity.JxW(q);
        energy_batch += Number{0.5} * rhocc_inv * p * p * pressure.JxW(q);
      }

      // sum over active entries of dealii::VectorizedArray
      for(unsigned int v = 0; v < matrix_free_in.n_active_entries_per_cell_batch(cell); ++v)
      {
        energy += energy_batch[v];
      }
    }

    dst += energy;
  }

  MPI_Comm const mpi_comm;

  bool clear_files;

  dealii::MatrixFree<dim, Number> const * matrix_free;
  SoundEnergyCalculatorData               data;

  unsigned int dof_index_pressure;
  unsigned int dof_index_velocity;
  unsigned int quad_index;
  unsigned int block_index_pressure;
  unsigned int block_index_velocity;

  Number rho;
  Number c;
  Number rhocc_inv;
};

} // namespace Acoustics
} // namespace ExaDG

#endif /*EXADG_ACOUSTIC_CONSERVATION_LAWS_POSTPROCESSOR_POINTWISE_SOUND_ENERGY_CALCULATIOR_H_*/
