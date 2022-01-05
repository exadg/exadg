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

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_H_

// application
#include <exadg/fluid_structure_interaction_precice/driver.h>

namespace ExaDG
{
namespace FSI
{
using namespace dealii;

template<int dim, typename Number>
class Driver
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  static void
  add_parameters(dealii::ParameterHandler & prm, PartitionedData & fsi_data) = 0;

  void
  setup(std::shared_ptr<ApplicationBase<dim, Number>> application,
        unsigned int const                            degree_fluid,
        unsigned int const                            degree_structure,
        unsigned int const                            refine_space_fluid,
        unsigned int const                            refine_space_structure);

  void
  solve() const = 0;

  void
  print_performance_results(double const total_time) const = 0;
};

} // namespace FSI
} // namespace ExaDG


#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_H_ */
