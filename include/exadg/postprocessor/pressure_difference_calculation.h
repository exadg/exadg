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

#ifndef INCLUDE_EXADG_POSTPROCESSOR_PRESSURE_DIFFERENCE_CALCULATION_H_
#define INCLUDE_EXADG_POSTPROCESSOR_PRESSURE_DIFFERENCE_CALCULATION_H_

// deal.II
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace ExaDG
{
using namespace dealii;

template<int dim>
struct PressureDifferenceData
{
  PressureDifferenceData() : calculate(false), directory("output"), filename("pressure_difference")
  {
  }

  /*
   *  active or not
   */
  bool calculate;

  /*
   *  Points:
   *  calculation of pressure difference: p(point_1) - p(point_2)
   */
  Point<dim> point_1;
  Point<dim> point_2;

  /*
   *  directory and filename
   */
  std::string directory;
  std::string filename;
};

template<int dim, typename Number>
class PressureDifferenceCalculator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  PressureDifferenceCalculator(MPI_Comm const & comm);

  void
  setup(DoFHandler<dim> const &             dof_handler_pressure_in,
        Mapping<dim> const &                mapping_in,
        PressureDifferenceData<dim> const & pressure_difference_data_in);

  void
  evaluate(VectorType const & pressure, double const & time) const;

private:
  MPI_Comm const mpi_comm;

  mutable bool clear_files;

  SmartPointer<DoFHandler<dim> const> dof_handler_pressure;
  SmartPointer<Mapping<dim> const>    mapping;

  PressureDifferenceData<dim> data;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_POSTPROCESSOR_PRESSURE_DIFFERENCE_CALCULATION_H_ */
