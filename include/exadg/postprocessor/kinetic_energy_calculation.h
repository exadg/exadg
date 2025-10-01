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

#ifndef EXADG_POSTPROCESSOR_KINETIC_ENERGY_CALCULATION_H_
#define EXADG_POSTPROCESSOR_KINETIC_ENERGY_CALCULATION_H_

// deal.II
#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/incompressible_navier_stokes/spatial_discretization/curl_compute.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/postprocessor/time_control.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
struct KineticEnergyData
{
  KineticEnergyData()
    : evaluate_individual_terms(false),
      viscosity(1.0),
      directory("output/"),
      filename("kinetic_energy"),
      clear_file(true)
  {
  }

  void
  print(dealii::ConditionalOStream & pcout)
  {
    if(time_control_data.is_active)
    {
      pcout << std::endl << "  Calculate kinetic energy:" << std::endl;

      // only implemented for unsteady problem
      time_control_data.print(pcout, true /*unsteady*/);

      print_parameter(pcout, "Evaluate individual terms", evaluate_individual_terms);
      print_parameter(pcout, "Directory of output files", directory);
      print_parameter(pcout, "Filename", filename);
      print_parameter(pcout, "Clear file", clear_file);
    }
  }

  TimeControlData time_control_data;

  // perform detailed analysis and evaluate contribution of individual terms (e.g., convective term,
  // viscous term) to overall kinetic energy dissipation?
  bool evaluate_individual_terms;

  // kinematic viscosity
  double viscosity;

  // directory and filename
  std::string directory;
  std::string filename;
  bool        clear_file;
};

template<int dim, typename Number>
class KineticEnergyCalculator
{
public:
  static unsigned int const number_vorticity_components = (dim == 2) ? 1 : dim;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> tensor;

  KineticEnergyCalculator(MPI_Comm const & comm);

  void
  setup(dealii::MatrixFree<dim, Number> const & matrix_free_in,
        unsigned int const                      dof_index_in,
        unsigned int const                      quad_index_in,
        KineticEnergyData const &               kinetic_energy_data_in);

  void
  evaluate(VectorType const & velocity, double const time, bool const unsteady);

  TimeControl time_control;

protected:
  void
  calculate_basic(VectorType const & velocity, double const time);

  /*
   *  This function calculates the kinetic energy
   *
   *  Kinetic energy: E_k = 1/V * 1/2 * (1,u*u)_Omega, V=(1,1)_Omega is the volume
   *
   *  Enstrophy: 1/V * 0.5 (1,rot(u)*rot(u))_Omega, V=(1,1)_Omega is the volume
   *
   *  Dissipation rate: epsilon = nu/V * (1, grad(u):grad(u))_Omega, V=(1,1)_Omega is the volume
   *
   *  Note that
   *
   *    epsilon = 2 * nu * Enstrophy
   *
   *  for incompressible flows (div(u)=0) and periodic boundary conditions.
   */
  Number
  integrate(dealii::MatrixFree<dim, Number> const & matrix_free_data,
            VectorType const &                      velocity,
            Number &                                energy,
            Number &                                enstrophy,
            Number &                                dissipation,
            Number &                                max_vorticity);

  void
  cell_loop(dealii::MatrixFree<dim, Number> const &       data,
            std::vector<Number> &                         dst,
            VectorType const &                            src,
            std::pair<unsigned int, unsigned int> const & cell_range);

  MPI_Comm const mpi_comm;

  bool clear_files;

  dealii::MatrixFree<dim, Number> const * matrix_free;
  unsigned int                            dof_index, quad_index;
  KineticEnergyData                       data;
};

} // namespace ExaDG

#endif /* EXADG_POSTPROCESSOR_KINETIC_ENERGY_CALCULATION_H_ */
