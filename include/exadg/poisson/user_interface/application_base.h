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

#ifndef INCLUDE_EXADG_POISSON_USER_INTERFACE_APPLICATION_BASE_H_
#define INCLUDE_EXADG_POISSON_USER_INTERFACE_APPLICATION_BASE_H_

// deal.II
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_description.h>

// ExaDG
#include <exadg/grid/grid.h>
#include <exadg/poisson/postprocessor/postprocessor.h>
#include <exadg/poisson/user_interface/boundary_descriptor.h>
#include <exadg/poisson/user_interface/field_functions.h>
#include <exadg/poisson/user_interface/parameters.h>

namespace ExaDG
{
namespace Poisson
{
using namespace dealii;

template<int dim, typename Number>
class ApplicationBase
{
public:
  typedef
    typename std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
      PeriodicFaces;

  virtual void
  add_parameters(ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("Output");
      prm.add_parameter("OutputDirectory",  output_directory, "Directory where output is written.");
      prm.add_parameter("OutputName",       output_name,      "Name of output files.");
      prm.add_parameter("WriteOutput",      write_output,     "Decides whether vtu output is written.");
    prm.leave_subsection();
    // clang-format on
  }

  ApplicationBase(std::string parameter_file, MPI_Comm const & comm)
    : mpi_comm(comm),
      pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0),
      parameter_file(parameter_file)
  {
  }

  virtual ~ApplicationBase()
  {
  }

  void
  set_parameters_refinement_study(unsigned int const degree,
                                  unsigned int const refine_space,
                                  unsigned int const n_subdivisions_1d_hypercube)
  {
    this->param.degree = degree;

    this->param.grid.n_refine_global             = refine_space;
    this->param.grid.n_subdivisions_1d_hypercube = n_subdivisions_1d_hypercube;
  }

  void
  setup()
  {
    // parameters
    set_parameters();
    param.check();
    param.print(pcout, "List of parameters:");

    // grid
    grid = std::make_shared<Grid<dim>>(param.grid, mpi_comm);
    create_grid();
    print_grid_info(pcout, *grid);

    // compute aspect ratio (TODO: shift this code to Grid class)
    if(false)
    {
      // this variant is only for comparison
      double AR = calculate_aspect_ratio_vertex_distance(*grid->triangulation, mpi_comm);
      pcout << std::endl << "Maximum aspect ratio vertex distance = " << AR << std::endl;

      QGauss<dim> quadrature(param.degree + 1);
      AR =
        GridTools::compute_maximum_aspect_ratio(*grid->mapping, *grid->triangulation, quadrature);
      pcout << std::endl << "Maximum aspect ratio Jacobian = " << AR << std::endl;
    }

    // boundary conditions
    boundary_descriptor = std::make_shared<BoundaryDescriptor<0, dim>>();
    set_boundary_descriptor();
    verify_boundary_conditions(*boundary_descriptor, *grid);

    // field functions
    field_functions = std::make_shared<FieldFunctions<dim>>();
    set_field_functions();
  }

  virtual std::shared_ptr<Poisson::PostProcessorBase<dim, Number>>
  create_postprocessor() = 0;

  Parameters const &
  get_parameters() const
  {
    return param;
  }

  std::shared_ptr<Grid<dim> const>
  get_grid() const
  {
    return grid;
  }

  std::shared_ptr<BoundaryDescriptor<0, dim> const>
  get_boundary_descriptor() const
  {
    return boundary_descriptor;
  }

  std::shared_ptr<FieldFunctions<dim> const>
  get_field_functions() const
  {
    return field_functions;
  }

protected:
  MPI_Comm const & mpi_comm;

  ConditionalOStream pcout;

  Parameters param;

  std::shared_ptr<Grid<dim>> grid;

  std::shared_ptr<BoundaryDescriptor<0, dim>> boundary_descriptor;
  std::shared_ptr<FieldFunctions<dim>>        field_functions;

  std::string parameter_file;

  std::string output_directory = "output/", output_name = "output";
  bool        write_output = false;

private:
  virtual void
  set_parameters() = 0;

  virtual void
  create_grid() = 0;

  virtual void
  set_boundary_descriptor() = 0;

  virtual void
  set_field_functions() = 0;
};

} // namespace Poisson
} // namespace ExaDG


#endif /* INCLUDE_EXADG_POISSON_USER_INTERFACE_APPLICATION_BASE_H_ */
