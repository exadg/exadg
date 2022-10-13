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

// C/C++
#include <fstream>

// ExaDG
#include <exadg/postprocessor/kinetic_energy_calculation.h>
#include <exadg/utilities/create_directories.h>

namespace ExaDG
{
template<int dim, typename Number>
KineticEnergyCalculator<dim, Number>::KineticEnergyCalculator(MPI_Comm const & comm)
  : mpi_comm(comm), clear_files(true), matrix_free(nullptr), dof_index(0), quad_index(0)
{
}

template<int dim, typename Number>
void
KineticEnergyCalculator<dim, Number>::setup(dealii::MatrixFree<dim, Number> const & matrix_free_in,
                                            unsigned int const                      dof_index_in,
                                            unsigned int const                      quad_index_in,
                                            KineticEnergyData const & kinetic_energy_data_in)
{
  matrix_free = &matrix_free_in;
  dof_index   = dof_index_in;
  quad_index  = quad_index_in;
  data        = kinetic_energy_data_in;

  time_control.setup(kinetic_energy_data_in.time_control_data);

  clear_files = data.clear_file;

  if(data.time_control_data.is_active)
    create_directories(data.directory, mpi_comm);
}

template<int dim, typename Number>
void
KineticEnergyCalculator<dim, Number>::evaluate(VectorType const & velocity,
                                               double const       time,
                                               bool const         unsteady)
{
  AssertThrow(unsteady,
              dealii::ExcMessage(
                "This postprocessing tool can only be used for unsteady problems."));

  AssertThrow(data.evaluate_individual_terms == false,
              dealii::ExcMessage("Not implemented in this class."));

  calculate_basic(velocity, time);
}

template<int dim, typename Number>
void
KineticEnergyCalculator<dim, Number>::calculate_basic(VectorType const & velocity,
                                                      double const       time)
{
  Number kinetic_energy = 0.0, enstrophy = 0.0, dissipation = 0.0, max_vorticity = 0.0;

  integrate(*matrix_free, velocity, kinetic_energy, enstrophy, dissipation, max_vorticity);

  // write output file
  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    // clang-format off
      std::ostringstream filename;
      filename << data.directory + data.filename;

      std::ofstream f;
      if(clear_files == true)
      {
        f.open(filename.str().c_str(), std::ios::trunc);
        f << "Kinetic energy: E_k = 1/V * 1/2 * (u,u)_Omega, where V=(1,1)_Omega" << std::endl
          << "Dissipation rate: epsilon = nu/V * (grad(u),grad(u))_Omega, where V=(1,1)_Omega" << std::endl
          << "Enstrophy: E = 1/V * 1/2 * (rot(u),rot(u))_Omega, where V=(1,1)_Omega" << std::endl;

        f << std::endl
          << "  Time                Kin. energy         dissipation         enstrophy           max_vorticity"
          << std::endl;

        clear_files = false;
      }
      else
      {
        f.open(filename.str().c_str(), std::ios::app);
      }

      unsigned int precision = 12;
      f << std::scientific << std::setprecision(precision)
        << std::setw(precision + 8) << time
        << std::setw(precision + 8) << kinetic_energy
        << std::setw(precision + 8) << dissipation
        << std::setw(precision + 8) << enstrophy
        << std::setw(precision + 8) << max_vorticity
        << std::endl;
    // clang-format on
  }
}

template<int dim, typename Number>
Number
KineticEnergyCalculator<dim, Number>::integrate(dealii::MatrixFree<dim, Number> const & matrix_free,
                                                VectorType const &                      velocity,
                                                Number &                                energy,
                                                Number &                                enstrophy,
                                                Number &                                dissipation,
                                                Number & max_vorticity)
{
  std::vector<Number> dst(5, 0.0);
  matrix_free.cell_loop(&KineticEnergyCalculator<dim, Number>::cell_loop, this, dst, velocity);

  // sum over all MPI processes
  Number volume = 1.0;
  volume        = dealii::Utilities::MPI::sum(dst.at(0), mpi_comm);
  energy        = dealii::Utilities::MPI::sum(dst.at(1), mpi_comm);
  enstrophy     = dealii::Utilities::MPI::sum(dst.at(2), mpi_comm);
  dissipation   = dealii::Utilities::MPI::sum(dst.at(3), mpi_comm);

  energy /= volume;
  enstrophy /= volume;
  dissipation /= volume;

  max_vorticity = dealii::Utilities::MPI::max(dst.at(4), mpi_comm);

  return volume;
}

template<int dim, typename Number>
void
KineticEnergyCalculator<dim, Number>::cell_loop(
  dealii::MatrixFree<dim, Number> const &       matrix_free,
  std::vector<Number> &                         dst,
  VectorType const &                            src,
  std::pair<unsigned int, unsigned int> const & cell_range)
{
  CellIntegrator<dim, dim, Number> fe_eval(matrix_free, dof_index, quad_index);

  Number volume        = 0.;
  Number energy        = 0.;
  Number enstrophy     = 0.;
  Number dissipation   = 0.;
  Number max_vorticity = 0.;

  // Loop over all elements
  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    fe_eval.reinit(cell);
    fe_eval.read_dof_values(src);
    fe_eval.evaluate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

    scalar volume_vec        = dealii::make_vectorized_array<Number>(0.);
    scalar energy_vec        = dealii::make_vectorized_array<Number>(0.);
    scalar enstrophy_vec     = dealii::make_vectorized_array<Number>(0.);
    scalar dissipation_vec   = dealii::make_vectorized_array<Number>(0.);
    scalar max_vorticity_vec = dealii::make_vectorized_array<Number>(0.);

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      volume_vec += fe_eval.JxW(q);

      vector velocity = fe_eval.get_value(q);
      energy_vec +=
        fe_eval.JxW(q) * dealii::make_vectorized_array<Number>(0.5) * velocity * velocity;

      tensor velocity_gradient = fe_eval.get_gradient(q);
      dissipation_vec += fe_eval.JxW(q) *
                         dealii::make_vectorized_array<Number>(this->data.viscosity) *
                         scalar_product(velocity_gradient, velocity_gradient);

      dealii::Tensor<1, number_vorticity_components, scalar> omega = fe_eval.get_curl(q);

      scalar norm_omega = omega * omega;

      enstrophy_vec += fe_eval.JxW(q) * dealii::make_vectorized_array<Number>(0.5) * norm_omega;

      max_vorticity_vec = std::max(max_vorticity_vec, std::sqrt(norm_omega));
    }

    // sum over entries of dealii::VectorizedArray, but only over those
    // that are "active"
    for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
    {
      volume += volume_vec[v];
      energy += energy_vec[v];
      enstrophy += enstrophy_vec[v];
      dissipation += dissipation_vec[v];

      max_vorticity = std::max(max_vorticity, max_vorticity_vec[v]);
    }
  }

  dst.at(0) += volume;
  dst.at(1) += energy;
  dst.at(2) += enstrophy;
  dst.at(3) += dissipation;
  dst.at(4) = std::max(dst.at(4), max_vorticity);
}

template class KineticEnergyCalculator<2, float>;
template class KineticEnergyCalculator<2, double>;

template class KineticEnergyCalculator<3, float>;
template class KineticEnergyCalculator<3, double>;

} // namespace ExaDG
