/*
 * kinetic_energy_calculation.cpp
 *
 *  Created on: May 17, 2019
 *      Author: fehn
 */

// C/C++
#include <fstream>

// ExaDG
#include <exadg/postprocessor/kinetic_energy_calculation.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, typename Number>
KineticEnergyCalculator<dim, Number>::KineticEnergyCalculator(MPI_Comm const & comm)
  : mpi_comm(comm), clear_files(true), matrix_free(nullptr), dof_index(0), quad_index(0)
{
}

template<int dim, typename Number>
void
KineticEnergyCalculator<dim, Number>::setup(MatrixFree<dim, Number> const & matrix_free_in,
                                            unsigned int const              dof_index_in,
                                            unsigned int const              quad_index_in,
                                            KineticEnergyData const &       kinetic_energy_data_in)
{
  matrix_free = &matrix_free_in;
  dof_index   = dof_index_in;
  quad_index  = quad_index_in;
  data        = kinetic_energy_data_in;

  clear_files = data.clear_file;
}

template<int dim, typename Number>
void
KineticEnergyCalculator<dim, Number>::evaluate(VectorType const & velocity,
                                               double const &     time,
                                               int const &        time_step_number)
{
  if(data.calculate == true)
  {
    AssertThrow(time_step_number >= 0,
                ExcMessage("This postprocessing tool can only be used for unsteady problems."));

    AssertThrow(data.evaluate_individual_terms == false,
                ExcMessage("Not implemented in this class."));

    calculate_basic(velocity, time, time_step_number);
  }
}

template<int dim, typename Number>
void
KineticEnergyCalculator<dim, Number>::calculate_basic(VectorType const & velocity,
                                                      double const       time,
                                                      unsigned int const time_step_number)
{
  if((time_step_number - 1) % data.calculate_every_time_steps == 0)
  {
    Number kinetic_energy = 0.0, enstrophy = 0.0, dissipation = 0.0, max_vorticity = 0.0;

    integrate(*matrix_free, velocity, kinetic_energy, enstrophy, dissipation, max_vorticity);

    // write output file
    if(Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      // clang-format off
      std::ostringstream filename;
      filename << data.filename;

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
}

template<int dim, typename Number>
Number
KineticEnergyCalculator<dim, Number>::integrate(MatrixFree<dim, Number> const & matrix_free,
                                                VectorType const &              velocity,
                                                Number &                        energy,
                                                Number &                        enstrophy,
                                                Number &                        dissipation,
                                                Number &                        max_vorticity)
{
  std::vector<Number> dst(5, 0.0);
  matrix_free.cell_loop(&KineticEnergyCalculator<dim, Number>::cell_loop, this, dst, velocity);

  // sum over all MPI processes
  Number volume = 1.0;
  volume        = Utilities::MPI::sum(dst.at(0), mpi_comm);
  energy        = Utilities::MPI::sum(dst.at(1), mpi_comm);
  enstrophy     = Utilities::MPI::sum(dst.at(2), mpi_comm);
  dissipation   = Utilities::MPI::sum(dst.at(3), mpi_comm);

  energy /= volume;
  enstrophy /= volume;
  dissipation /= volume;

  max_vorticity = Utilities::MPI::max(dst.at(4), mpi_comm);

  return volume;
}

template<int dim, typename Number>
void
KineticEnergyCalculator<dim, Number>::cell_loop(
  MatrixFree<dim, Number> const &               matrix_free,
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
    fe_eval.evaluate(true, true);

    scalar volume_vec        = make_vectorized_array<Number>(0.);
    scalar energy_vec        = make_vectorized_array<Number>(0.);
    scalar enstrophy_vec     = make_vectorized_array<Number>(0.);
    scalar dissipation_vec   = make_vectorized_array<Number>(0.);
    scalar max_vorticity_vec = make_vectorized_array<Number>(0.);

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      volume_vec += fe_eval.JxW(q);

      vector velocity = fe_eval.get_value(q);
      energy_vec += fe_eval.JxW(q) * make_vectorized_array<Number>(0.5) * velocity * velocity;

      tensor velocity_gradient = fe_eval.get_gradient(q);
      dissipation_vec += fe_eval.JxW(q) * make_vectorized_array<Number>(this->data.viscosity) *
                         scalar_product(velocity_gradient, velocity_gradient);

      Tensor<1, number_vorticity_components, scalar> omega = fe_eval.get_curl(q);

      scalar norm_omega = omega * omega;

      enstrophy_vec += fe_eval.JxW(q) * make_vectorized_array<Number>(0.5) * norm_omega;

      max_vorticity_vec = std::max(max_vorticity_vec, std::sqrt(norm_omega));
    }

    // sum over entries of VectorizedArray, but only over those
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
