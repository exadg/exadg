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

#ifndef EXADG_POSTPROCESSOR_OUTPUT_DATA_BASE_H_
#define EXADG_POSTPROCESSOR_OUTPUT_DATA_BASE_H_

// ExaDG
#include <exadg/postprocessor/time_control.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
struct OutputDataBase
{
  OutputDataBase()
    : directory("output/"),
      filename("name"),
      write_surface_mesh(false),
      write_boundary_IDs(false),
      write_grid(false),
      write_aspect_ratio(false),
      write_processor_id(false),
      write_higher_order(true),
      degree(1)
  {
  }

  void
  print(dealii::ConditionalOStream & pcout, bool unsteady)
  {
    if(time_control_data.is_active)
    {
      time_control_data.print(pcout, unsteady);

      print_parameter(pcout, "Output directory", directory);
      print_parameter(pcout, "Name of output files", filename);

      print_parameter(pcout, "Write surface mesh", write_surface_mesh);
      print_parameter(pcout, "Write boundary IDs", write_boundary_IDs);

      print_parameter(pcout, "Write aspect ratio", write_aspect_ratio);
      print_parameter(pcout, "Write processor ID", write_processor_id);

      print_parameter(pcout, "Write higher order", write_higher_order);
      print_parameter(pcout, "Polynomial degree", degree);
    }
  }

  TimeControlData time_control_data;

  // output directory
  std::string directory;

  // name of generated output files
  std::string filename;

  // this variable decides whether the surface mesh is written separately
  bool write_surface_mesh;

  // this variable decides whether a vtk-file is written that allows a visualization of boundary
  // IDs, e.g., to verify that boundary IDs have been set correctly. Note that in the current
  // version of deal.II, boundaries with ID = 0 (default) are not visible, but only those with
  // ID != 0.
  bool write_boundary_IDs;

  // write grid output for debug meshing
  bool write_grid;

  // write the aspect ratio to check the mesh quality
  bool write_aspect_ratio;

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

#endif /* EXADG_POSTPROCESSOR_OUTPUT_DATA_BASE_H_ */
