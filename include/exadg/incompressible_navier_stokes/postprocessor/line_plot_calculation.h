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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_H_

#include <exadg/incompressible_navier_stokes/postprocessor/line_plot_data.h>

#include <deal.II/numerics/vector_tools_evaluate.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

/*
 *  Evaluate quantities along lines.
 *
 *  Assumptions/Restrictions:
 *
 *   - straight lines, points are distributed equidistantly along the line
 *
 *   - no statistical averaging, instantaneous quantities are calculated
 */
template<int dim, typename Number>
class LinePlotCalculator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  LinePlotCalculator(MPI_Comm const & comm);

  void
  setup(DoFHandler<dim> const &                dof_handler_velocity_in,
        DoFHandler<dim> const &                dof_handler_pressure_in,
        Mapping<dim> const &                   mapping_in,
        LinePlotDataInstantaneous<dim> const & line_plot_data_in);

  void
  evaluate(VectorType const & velocity, VectorType const & pressure) const;

private:
  MPI_Comm const mpi_comm;

  mutable bool clear_files;

  SmartPointer<DoFHandler<dim> const> dof_handler_velocity;
  SmartPointer<DoFHandler<dim> const> dof_handler_pressure;
  SmartPointer<Mapping<dim> const>    mapping;

  LinePlotDataInstantaneous<dim> data;

  std::vector<Point<dim>>                                     evaluation_points;
  std::shared_ptr<Utilities::MPI::RemotePointEvaluation<dim>> evaluation_cache;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_H_ */
