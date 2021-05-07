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

#ifndef INCLUDE_EXADG_POSTPROCESSOR_OUTPUT_DATA_BASE_H_
#define INCLUDE_EXADG_POSTPROCESSOR_OUTPUT_DATA_BASE_H_

#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
using namespace dealii;

struct OutputDataBase
{
  OutputDataBase()
    : write_output(false),
      start_counter(0),
      directory("output/"),
      filename("name"),
      start_time(std::numeric_limits<double>::max()),
      interval_time(std::numeric_limits<double>::max()),
      write_surface_mesh(false),
      write_boundary_IDs(false),
      write_grid(false),
      write_processor_id(false),
      write_higher_order(true),
      degree(1)
  {
  }

  void
  print(ConditionalOStream & pcout, bool unsteady)
  {
    // output for visualization of results
    print_parameter(pcout, "Write output", write_output);

    if(write_output == true)
    {
      print_parameter(pcout, "Output counter start", start_counter);
      print_parameter(pcout, "Output directory", directory);
      print_parameter(pcout, "Name of output files", filename);

      if(unsteady == true)
      {
        print_parameter(pcout, "Output start time", start_time);
        print_parameter(pcout, "Output interval time", interval_time);
      }

      print_parameter(pcout, "Write surface mesh", write_surface_mesh);
      print_parameter(pcout, "Write boundary IDs", write_boundary_IDs);

      print_parameter(pcout, "Write processor ID", write_processor_id);

      print_parameter(pcout, "Write higher order", write_higher_order);
      print_parameter(pcout, "Polynomial degree", degree);
    }
  }

  // set write_output = true in order to write files for visualization
  bool write_output;

  unsigned int start_counter;

  // output directory
  std::string directory;

  // name of generated output files
  std::string filename;

  // before then no output will be written
  double start_time;

  // specifies the time interval in which output is written
  double interval_time;

  // this variable decides whether the surface mesh is written separately
  bool write_surface_mesh;

  // this variable decides whether a vtk-file is written that allows a visualization of boundary
  // IDs, e.g., to verify that boundary IDs have been set correctly. Note that in the current
  // version of deal.II, boundaries with ID = 0 (default) are not visible, but only those with
  // ID != 0.
  bool write_boundary_IDs;

  // write grid output for debug meshing
  bool write_grid;

  // write processor ID to scalar field in order to visualize the
  // distribution of cells to processors
  bool write_processor_id;

  // write higher order output (NOTE: requires at least ParaView version 5.5, switch off if ParaView
  // version is lower)
  bool write_higher_order;

  // defines polynomial degree used for output (for visualization in ParaView: Properties >
  // Miscellaneous > Nonlinear Subdivision Level (use a value > 1)) if write_higher_order = true. In
  // case of write_higher_order = false, this variable defines the number of subdivisions of a cell,
  // with ParaView using linear interpolation for visualization on these subdivided cells.
  unsigned int degree;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_POSTPROCESSOR_OUTPUT_DATA_BASE_H_ */
