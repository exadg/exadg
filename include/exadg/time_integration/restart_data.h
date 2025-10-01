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

#ifndef EXADG_TIME_INTEGRATION_RESTART_DATA_H_
#define EXADG_TIME_INTEGRATION_RESTART_DATA_H_

// C/C++
#include <limits>

// deal.II
#include <deal.II/base/conditional_ostream.h>

// ExaDG
#include <exadg/grid/grid_data.h>
#include <exadg/incompressible_navier_stokes/user_interface/enum_types.h>
#include <exadg/utilities/numbers.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
struct RestartData
{
  RestartData()
    : write_restart(false),
      interval_time(std::numeric_limits<double>::max()),
      interval_wall_time(std::numeric_limits<double>::max()),
      interval_time_steps(std::numeric_limits<unsigned int>::max()),
      directory("./output/"),
      filename("restart"),
      counter(1),
      degree_u(dealii::numbers::invalid_unsigned_int),
      degree_p(dealii::numbers::invalid_unsigned_int),
      triangulation_type(TriangulationType::Serial),
      spatial_discretization(IncNS::SpatialDiscretization::L2),
      discretization_identical(false),
      consider_mapping(false),
      mapping_degree(dealii::numbers::invalid_unsigned_int),
      rpe_rtree_level(0),
      rpe_tolerance_unit_cell(1e-12),
      rpe_enforce_unique_mapping(false)
  {
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    pcout << "  Restart:" << std::endl;
    print_parameter(pcout, "Write restart", write_restart);

    if(write_restart == true)
    {
      print_parameter(pcout, "Interval physical time", interval_time);
      print_parameter(pcout, "Interval wall time", interval_wall_time);
      print_parameter(pcout, "Interval time steps", interval_time_steps);
      print_parameter(pcout, "Directory", directory);
      print_parameter(pcout, "Filename", filename);
    }
  }

  bool
  do_restart(double const           wall_time,
             double const           time,
             types::time_step const time_step_number,
             bool const             reset_counter) const
  {
    // After a restart, the counter is reset to 1, but time = current_time - start time != 0 after a
    // restart. Hence, we have to explicitly reset the counter in that case. There is nothing to do
    // if the restart is controlled by the wall time or the time_step_number because these
    // variables are reinitialized after a restart anyway.
    if(reset_counter)
      counter += int((time + 1.e-10) / interval_time);

    bool do_restart = wall_time > interval_wall_time * counter or time > interval_time * counter or
                      time_step_number > interval_time_steps * counter;

    if(do_restart)
      ++counter;

    return do_restart;
  }

  bool write_restart;

  // physical time
  double interval_time;

  // wall time in seconds (= hours * 3600)
  double interval_wall_time;

  // number of time steps after which to write restart
  unsigned int interval_time_steps;

  // directory for restart files
  std::string directory;

  // filename for restart files
  std::string filename;

  // counter needed do decide when to write restart
  mutable unsigned int counter;

  // Finite element degree used when restart data was written (relevant for restart run only).
  unsigned int degree_u;
  unsigned int degree_p;

  // TriangulationType used when restart data was written (relevant for restart run only).
  TriangulationType triangulation_type;

  // Finite element space used when the restart data was written, relevant only for incompressible
  // flow.
  IncNS::SpatialDiscretization spatial_discretization;

  // The discretization used when writing the restart data was identical to the current one.
  // Note that this includes the finite element, uniform and adaptive refinement, and the
  // `TriangulationType`, *but* one might consider a different number of MPI ranks for
  // `dealii::parallel::distributed::Triangulation` without the need for the otherwise
  // necessary global projection.
  bool discretization_identical;

  // The mapping of the triangulation should be de-/serialized as well to consider for a mapped
  // geometry at serialization and during deserialization. This is option toggles storing the
  // mapping via a displacement vector *and* reading it back in. Hence, this parameter needs to
  // match in serialization/deserialization runs.
  bool consider_mapping;

  // The `mapping_degree` considered when storing or reading the grid.
  unsigned int mapping_degree;

  // Parameters for `dealii::Utilities::MPI::RemotePointEvaluation<dim>::RemotePointEvaluation`
  // used for grid-to-grid projection. Mirrored here to avoid `dim` template parameter.
  unsigned int rpr_rtree_level;
  double       rpe_tolerance_unit_cell;
  bool         rpe_enforce_unique_mapping;
};

} // namespace ExaDG

#endif /* EXADG_TIME_INTEGRATION_RESTART_DATA_H_ */
