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
#include <exadg/incompressible_navier_stokes/postprocessor/kinetic_energy_dissipation_detailed.h>
#include <exadg/utilities/numbers.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
KineticEnergyCalculatorDetailed<dim, Number>::KineticEnergyCalculatorDetailed(MPI_Comm const & comm)
  : KineticEnergyCalculator<dim, Number>(comm)
{
}

template<int dim, typename Number>
void
KineticEnergyCalculatorDetailed<dim, Number>::setup(
  NavierStokesOperator const &            navier_stokes_operator_in,
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  unsigned int const                      dof_index_in,
  unsigned int const                      quad_index_in,
  KineticEnergyData const &               kinetic_energy_data_in)
{
  Base::setup(matrix_free_in, dof_index_in, quad_index_in, kinetic_energy_data_in);

  navier_stokes_operator = &navier_stokes_operator_in;
}

template<int dim, typename Number>
void
KineticEnergyCalculatorDetailed<dim, Number>::evaluate(VectorType const & velocity,
                                                       double const &     time,
                                                       int const &        time_step_number)
{
  if(this->data.calculate == true)
  {
    AssertThrow(Utilities::is_unsteady_timestep(time_step_number),
                dealii::ExcMessage(
                  "This postprocessing tool can only be used for unsteady problems."));

    if(this->data.evaluate_individual_terms)
      calculate_detailed(velocity, time, time_step_number);
    else
      this->calculate_basic(velocity, time, time_step_number);
  }
}

template<int dim, typename Number>
void
KineticEnergyCalculatorDetailed<dim, Number>::calculate_detailed(
  VectorType const & velocity,
  double const       time,
  unsigned int const time_step_number)
{
  if((time_step_number - 1) % this->data.calculate_every_time_steps == 0)
  {
    Number kinetic_energy = 0.0, enstrophy = 0.0, dissipation = 0.0, max_vorticity = 0.0;

    Number volume = this->integrate(
      *this->matrix_free, velocity, kinetic_energy, enstrophy, dissipation, max_vorticity);

    AssertThrow(navier_stokes_operator != nullptr, dealii::ExcMessage("Invalid pointer."));
    Number dissipation_convective =
      navier_stokes_operator->calculate_dissipation_convective_term(velocity, time) / volume;
    Number dissipation_viscous =
      navier_stokes_operator->calculate_dissipation_viscous_term(velocity) / volume;
    Number dissipation_divergence =
      navier_stokes_operator->calculate_dissipation_divergence_term(velocity) / volume;
    Number dissipation_continuity =
      navier_stokes_operator->calculate_dissipation_continuity_term(velocity) / volume;

    // write output file
    if(dealii::Utilities::MPI::this_mpi_process(this->mpi_comm) == 0)
    {
      // clang-format off
      std::ostringstream filename;
      filename << this->data.filename;

      std::ofstream f;
      if(this->clear_files == true)
      {
        f.open(filename.str().c_str(), std::ios::trunc);
        f << "Kinetic energy: E_k = 1/V * 1/2 * (u,u)_Omega, where V=(1,1)_Omega" << std::endl
          << "Dissipation rate: epsilon = nu/V * (grad(u),grad(u))_Omega, where V=(1,1)_Omega" << std::endl
          << "Enstrophy: E = 1/V * 1/2 * (rot(u),rot(u))_Omega, where V=(1,1)_Omega" << std::endl
          << "Dissipation convective term: eps_conv = 1/V * c(u,u)_Omega, where V=(1,1)_Omega" << std::endl
          << "Dissipation viscous term: eps_vis = 1/V * v(u,u)_Omega, where V=(1,1)_Omega" << std::endl
          << "Dissipation divergence penalty term: eps_div = 1/V * a_D(u,u)_Omega, where V=(1,1)_Omega" << std::endl
          << "Dissipation continuity penalty term: eps_conti = 1/V * a_C(u,u)_Omega, where V=(1,1)_Omega" << std::endl;

        f << std::endl
          << "  Time                Kin. energy         dissipation         enstrophy           max_vorticity       convective          viscous             divergence          continuity"
          << std::endl;

        this->clear_files = false;
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
        << std::setw(precision + 8) << dissipation_convective
        << std::setw(precision + 8) << dissipation_viscous
        << std::setw(precision + 8) << dissipation_divergence
        << std::setw(precision + 8) << dissipation_continuity
        << std::endl;
      // clang-format on
    }
  }
}

template class KineticEnergyCalculatorDetailed<2, float>;
template class KineticEnergyCalculatorDetailed<2, double>;

template class KineticEnergyCalculatorDetailed<3, float>;
template class KineticEnergyCalculatorDetailed<3, double>;

} // namespace IncNS
} // namespace ExaDG
