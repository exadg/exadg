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
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_description.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>
#include <exadg/grid/calculate_maximum_aspect_ratio.h>
#include <exadg/grid/grid.h>
#include <exadg/grid/grid_parameters.h>
#include <exadg/grid/grid_utilities.h>
#include <exadg/operators/resolution_parameters.h>
#include <exadg/poisson/postprocessor/postprocessor.h>
#include <exadg/poisson/user_interface/boundary_descriptor.h>
#include <exadg/poisson/user_interface/field_functions.h>
#include <exadg/poisson/user_interface/parameters.h>
#include <exadg/postprocessor/output_parameters.h>
#include <exadg/utilities/exceptions.h>

namespace ExaDG
{
namespace Poisson
{
template<int dim, int n_components, typename Number>
class ApplicationBase
{
public:
  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : dealii::numbers::invalid_unsigned_int);

  typedef typename std::vector<
    dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator>>
    PeriodicFaces;

  virtual void
  add_parameters(dealii::ParameterHandler & prm)
  {
    grid_parameters.add_parameters(prm);
    output_parameters.add_parameters(prm);
  }

  ApplicationBase(std::string parameter_file, MPI_Comm const & comm)
    : mpi_comm(comm),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
      parameter_file(parameter_file),
      n_subdivisions_1d_hypercube(1)
  {
    grid = std::make_shared<Grid<dim>>();
  }

  virtual ~ApplicationBase()
  {
  }

  void
  set_parameters_throughput_study(unsigned int const degree,
                                  unsigned int const refine_space,
                                  unsigned int const n_subdivisions_1d_hypercube)
  {
    this->param.degree                = degree;
    this->param.grid.n_refine_global  = refine_space;
    this->n_subdivisions_1d_hypercube = n_subdivisions_1d_hypercube;
  }

  void
  set_parameters_convergence_study(unsigned int const degree, unsigned int const refine_space)
  {
    this->param.degree               = degree;
    this->param.grid.n_refine_global = refine_space;
  }

  void
  setup()
  {
    // parameters
    parse_parameters();
    set_parameters();
    param.check();
    param.print(pcout, "List of parameters:");

    // grid
    GridUtilities::create_mapping(mapping, param.grid.element_type, param.mapping_degree);
    create_grid();
    print_grid_info(pcout, *grid);

    if(compute_aspect_ratio)
    {
      // this variant is only for comparison
      double AR = calculate_aspect_ratio_vertex_distance(*grid->triangulation, mpi_comm);
      pcout << std::endl << "Maximum aspect ratio (vertex distance) = " << AR << std::endl;

      auto const reference_cells = grid->triangulation->get_reference_cells();
      AssertThrow(reference_cells.size() == 1, dealii::ExcMessage("No mixed meshes allowed"));

      auto const quad =
        reference_cells[0].template get_gauss_type_quadrature<dim>(param.degree + 1);

      AR = dealii::GridTools::compute_maximum_aspect_ratio(*mapping, *grid->triangulation, quad);
      pcout << std::endl << "Maximum aspect ratio (Jacobian) = " << AR << std::endl;
    }

    // boundary conditions
    boundary_descriptor = std::make_shared<BoundaryDescriptor<rank, dim>>();
    set_boundary_descriptor();
    verify_boundary_conditions(*boundary_descriptor, *grid);

    // field functions
    field_functions = std::make_shared<FieldFunctions<dim>>();
    set_field_functions();
  }

  virtual std::shared_ptr<Poisson::PostProcessorBase<dim, n_components, Number>>
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

  std::shared_ptr<dealii::Mapping<dim> const>
  get_mapping() const
  {
    return mapping;
  }

  std::shared_ptr<BoundaryDescriptor<rank, dim> const>
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
  virtual void
  parse_parameters()
  {
    dealii::ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(parameter_file, "", true, true);
  }

  MPI_Comm const & mpi_comm;

  dealii::ConditionalOStream pcout;

  Parameters param;

  std::shared_ptr<Grid<dim>> grid;

  std::shared_ptr<dealii::Mapping<dim>> mapping;

  std::shared_ptr<BoundaryDescriptor<rank, dim>> boundary_descriptor;
  std::shared_ptr<FieldFunctions<dim>>           field_functions;

  std::string parameter_file;

  GridParameters grid_parameters;

  unsigned int n_subdivisions_1d_hypercube;

  OutputParameters output_parameters;

  bool compute_aspect_ratio = false;

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
