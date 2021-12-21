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
#include <exadg/postprocessor/pressure_difference_calculation.h>
#include <exadg/utilities/create_directories.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, typename Number>
PressureDifferenceCalculator<dim, Number>::PressureDifferenceCalculator(MPI_Comm const & comm)
  : mpi_comm(comm), clear_files(true)
{
}

template<int dim, typename Number>
void
PressureDifferenceCalculator<dim, Number>::setup(DoFHandler<dim> const & dof_handler_pressure_in,
                                                 Mapping<dim> const &    mapping_in,
                                                 PressureDifferenceData<dim> const & data_in)
{
  dof_handler_pressure = &dof_handler_pressure_in;
  mapping              = &mapping_in;
  data                 = data_in;

  if(data.calculate)
  {
    create_directories(data.directory, mpi_comm);
    evaluation_cache = std::make_shared<Utilities::MPI::RemotePointEvaluation<dim>>(
      1e-8 * dof_handler_pressure->begin()->diameter());

    // only request result on rank 0
    if(Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      std::vector<Point<dim>> points{data.point_1, data.point_2};
      evaluation_cache->reinit(points, dof_handler_pressure->get_triangulation(), mapping_in);
    }
    else
      evaluation_cache->reinit({}, dof_handler_pressure->get_triangulation(), mapping_in);
  }
}

template<int dim, typename Number>
void
PressureDifferenceCalculator<dim, Number>::evaluate(VectorType const & pressure,
                                                    double const &     time) const
{
  if(data.calculate)
  {
    AssertThrow(evaluation_cache.get() != nullptr, ExcNotInitialized());

    // convert to double vector to ensure we can evaluate with deal.II's function
    LinearAlgebra::distributed::Vector<double> ghosted_vector(pressure.get_partitioner());
    ghosted_vector.copy_locally_owned_data_from(pressure);
    ghosted_vector.update_ghost_values();

    // default flag combination computes average from all cells
    std::vector<double> point_values =
      VectorTools::point_values<1>(*evaluation_cache, *dof_handler_pressure, ghosted_vector);

    if(Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      double const pressure_difference = point_values[0] - point_values[1];

      std::string filename = data.directory + data.filename;

      unsigned int precision = 12;

      std::ofstream f;
      if(clear_files)
      {
        f.open(filename.c_str(), std::ios::trunc);

        // clang-format off
        f << std::setw(precision + 8) << std::left << "time t"
          << std::setw(precision + 8) << std::left << "pressure difference"
          << std::endl;
        // clang-format on

        clear_files = false;
      }
      else
      {
        f.open(filename.c_str(), std::ios::app);
      }

      // clang-format off
      f << std::scientific << std::setprecision(precision)
        << std::setw(precision + 8) << std::left << time
        << std::setw(precision + 8) << std::left << pressure_difference
        << std::endl;
      // clang-format on

      f.close();
    }
  }
}

template class PressureDifferenceCalculator<2, float>;
template class PressureDifferenceCalculator<2, double>;

template class PressureDifferenceCalculator<3, float>;
template class PressureDifferenceCalculator<3, double>;

} // namespace ExaDG
