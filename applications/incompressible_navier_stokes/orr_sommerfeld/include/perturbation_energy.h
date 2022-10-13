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

#ifndef SOLVERS_INCOMPRESSIBLE_NAVIER_STOKES_APPLICATIONS_ORR_SOMMERFELD_PERTURBATION_ENERGY_H_
#define SOLVERS_INCOMPRESSIBLE_NAVIER_STOKES_APPLICATIONS_ORR_SOMMERFELD_PERTURBATION_ENERGY_H_

// C/C++
#include <fstream>

// ExaDG
#include <exadg/matrix_free/integrators.h>
#include <exadg/postprocessor/time_control.h>
#include <exadg/utilities/create_directories.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace IncNS
{
struct PerturbationEnergyData
{
  PerturbationEnergyData()
    : directory("output/"), filename("orr_sommerfeld"), omega_i(0.0), h(1.0), U_max(1.0)
  {
  }

  void
  print(dealii::ConditionalOStream & pcout)
  {
    if(time_control_data.is_active)
    {
      pcout << "  Calculate perturbation energy:" << std::endl;
      // only implemented for unsteady case
      time_control_data.print(pcout, true /*unsteady*/);

      print_parameter(pcout, "Directory", directory);
      print_parameter(pcout, "Filename", filename);
      print_parameter(pcout, "Amplification omega_i", omega_i);
      print_parameter(pcout, "Channel height h", h);
      print_parameter(pcout, "Maximum velocity U_max", U_max);
    }
  }

  TimeControlData time_control_data;

  std::string directory;
  std::string filename;
  double      omega_i;
  double      h;
  double      U_max;
};

/*
 * Calculation of perturbation energy for Orr-Sommerfeld problem
 */
template<int dim, typename Number>
class PerturbationEnergyCalculator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef PerturbationEnergyCalculator<dim, Number> This;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;

  PerturbationEnergyCalculator(MPI_Comm const & comm)
    : mpi_comm(comm),
      clear_files(true),
      initial_perturbation_energy_has_been_calculated(false),
      initial_perturbation_energy(1.0),
      matrix_free(nullptr),
      dof_index(0),
      quad_index(0)
  {
  }

  void
  setup(dealii::MatrixFree<dim, Number> const & matrix_free_in,
        unsigned int const                      dof_index_in,
        unsigned int const                      quad_index_in,
        PerturbationEnergyData const &          data_in)
  {
    matrix_free = &matrix_free_in;
    dof_index   = dof_index_in;
    quad_index  = quad_index_in;
    data        = data_in;

    time_control.setup(data.time_control_data);

    if(data.time_control_data.is_active)
      create_directories(data.directory, mpi_comm);
  }

  void
  evaluate(VectorType const & velocity, double const time, bool const unsteady)
  {
    AssertThrow(unsteady,
                dealii::ExcMessage(
                  "Calculation of perturbation energy for "
                  "Orr-Sommerfeld problem only makes sense for unsteady problems."));
    do_evaluate(velocity, time);
  }

private:
  void
  do_evaluate(VectorType const & velocity, double const time)
  {
    Number perturbation_energy = 0.0;

    integrate(*matrix_free, velocity, perturbation_energy);

    if(!initial_perturbation_energy_has_been_calculated)
    {
      initial_perturbation_energy = perturbation_energy;

      initial_perturbation_energy_has_been_calculated = true;
    }

    // write output file
    if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      // clang-format off
        unsigned int l = matrix_free->get_dof_handler(dof_index)
                           .get_triangulation().n_global_levels() - 1;
      // clang-format on

      std::string filename =
        data.directory + data.filename + "_l" + dealii::Utilities::int_to_string(l);

      std::ofstream f;
      if(clear_files == true)
      {
        f.open(filename.c_str(), std::ios::trunc);
        f << "Perturbation energy: E = (1,(u-u_base)^2)_Omega" << std::endl
          << "Error:               e = |exp(2*omega_i*t) - E(t)/E(0)|" << std::endl;

        f << std::endl << "  Time                energy              error" << std::endl;

        clear_files = false;
      }
      else
      {
        f.open(filename.c_str(), std::ios::app);
      }

      Number const rel   = perturbation_energy / initial_perturbation_energy;
      Number const error = std::abs(std::exp(2 * data.omega_i * time) - rel);

      unsigned int const precision = 12;
      f << std::scientific << std::setprecision(precision) << std::setw(precision + 8) << time
        << std::setw(precision + 8) << perturbation_energy << std::setw(precision + 8) << error
        << std::endl;
    }
  }

  /*
   *  This function calculates the perturbation energy
   *
   *  Perturbation energy: E = (1,u*u)_Omega
   */
  void
  integrate(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType const &                      velocity,
            Number &                                energy)
  {
    std::vector<Number> dst(1, 0.0);
    matrix_free.cell_loop(&This::local_compute, this, dst, velocity);

    // sum over all MPI processes
    energy = dealii::Utilities::MPI::sum(dst.at(0), mpi_comm);
  }

  void
  local_compute(dealii::MatrixFree<dim, Number> const &       matrix_free,
                std::vector<Number> &                         dst,
                VectorType const &                            src,
                std::pair<unsigned int, unsigned int> const & cell_range)
  {
    CellIntegrator<dim, dim, Number> integrator(matrix_free, dof_index, quad_index);

    // Loop over all elements
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);
      integrator.read_dof_values(src);
      integrator.evaluate(dealii::EvaluationFlags::values);

      dealii::VectorizedArray<Number> energy_vec = dealii::make_vectorized_array<Number>(0.);
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        vector velocity = integrator.get_value(q);

        dealii::Point<dim, scalar> q_points = integrator.quadrature_point(q);

        scalar y = q_points[1] / data.h;

        vector velocity_base;
        velocity_base[0] = data.U_max * (1.0 - y * y);
        energy_vec += integrator.JxW(q) * (velocity - velocity_base) * (velocity - velocity_base);
      }

      // sum over entries of dealii::VectorizedArray, but only over those
      // that are "active"
      for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
      {
        dst.at(0) += energy_vec[v];
      }
    }
  }

  TimeControl time_control;

  MPI_Comm const mpi_comm;

  bool   clear_files;
  bool   initial_perturbation_energy_has_been_calculated;
  Number initial_perturbation_energy;

  dealii::MatrixFree<dim, Number> const * matrix_free;
  unsigned int                            dof_index, quad_index;
  PerturbationEnergyData                  data;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* SOLVERS_INCOMPRESSIBLE_NAVIER_STOKES_APPLICATIONS_ORR_SOMMERFELD_PERTURBATION_ENERGY_H_ \
        */
