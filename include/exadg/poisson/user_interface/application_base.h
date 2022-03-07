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
#include <exadg/utilities/output_parameters.h>

namespace ExaDG
{
namespace Poisson
{
template<int dim, typename Number>
class ApplicationBase
{
public:
  typedef typename std::vector<
    dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator>>
    PeriodicFaces;

  virtual void
  add_parameters(dealii::ParameterHandler & prm)
  {
    output_parameters.add_parameters(prm);
  }

  ApplicationBase(std::string parameter_file, MPI_Comm const & comm)
    : mpi_comm(comm),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
      parameter_file(parameter_file)
  {
  }

  virtual ~ApplicationBase()
  {
  }

  virtual void
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
    parse_parameters();

    // parameters
    set_parameters();
    param.check();
    param.print(pcout, "List of parameters:");

    // grid
    grid = std::make_shared<Grid<dim>>(param.grid, mpi_comm);
    create_grid();
    print_grid_info(pcout, *grid);

    if(compute_aspect_ratio)
    {
      // this variant is only for comparison
      double AR = calculate_aspect_ratio_vertex_distance(*grid->triangulation, mpi_comm);
      pcout << std::endl << "Maximum aspect ratio (vertex distance) = " << AR << std::endl;

      dealii::QGauss<dim> quad(param.degree + 1);
      AR =
        dealii::GridTools::compute_maximum_aspect_ratio(*grid->mapping, *grid->triangulation, quad);
      pcout << std::endl << "Maximum aspect ratio (Jacobian) = " << AR << std::endl;
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

  std::shared_ptr<BoundaryDescriptor<0, dim>> boundary_descriptor;
  std::shared_ptr<FieldFunctions<dim>>        field_functions;

  std::string parameter_file;

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

template<int dim, typename Number>
class ApplicationOversetGridsBase
{
public:
  typedef typename std::vector<
    dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator>>
    PeriodicFaces;

  virtual void
  add_parameters(dealii::ParameterHandler & prm)
  {
    output_parameters.add_parameters(prm);
  }

  ApplicationOversetGridsBase(std::string parameter_file, MPI_Comm const & comm)
    : mpi_comm(comm),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
      parameter_file(parameter_file)
  {
  }

  virtual ~ApplicationOversetGridsBase()
  {
  }

  virtual void
  set_parameters_refinement_study(unsigned int const degree,
                                  unsigned int const refine_space,
                                  unsigned int const n_subdivisions_1d_hypercube)
  {
    this->param.degree = degree;

    this->param.grid.n_refine_global             = refine_space;
    this->param.grid.n_subdivisions_1d_hypercube = n_subdivisions_1d_hypercube;

    this->param_second.degree = degree;

    this->param_second.grid.n_refine_global             = refine_space;
    this->param_second.grid.n_subdivisions_1d_hypercube = n_subdivisions_1d_hypercube;
  }

  void
  setup()
  {
    parse_parameters();

    // parameters
    set_parameters();
    param.check();
    param.print(pcout, "List of parameters:");

    // grid
    grid = std::make_shared<Grid<dim>>(param.grid, mpi_comm);
    create_grid();
    print_grid_info(pcout, *grid);

    // boundary conditions
    boundary_descriptor = std::make_shared<BoundaryDescriptor<1, dim>>();
    set_boundary_descriptor();
    verify_boundary_conditions(*boundary_descriptor, *grid);

    // field functions
    field_functions = std::make_shared<FieldFunctions<dim>>();
    set_field_functions();

    // parameters
    set_parameters_second();
    param_second.check();
    param_second.print(this->pcout, "List of parameters second domain:");

    // grid
    grid_second = std::make_shared<Grid<dim>>(param_second.grid, this->mpi_comm);
    create_grid_second();
    print_grid_info(this->pcout, *grid_second);

    // boundary conditions
    boundary_descriptor_second = std::make_shared<BoundaryDescriptor<1, dim>>();
    set_boundary_descriptor_second();
    verify_boundary_conditions(*boundary_descriptor_second, *grid_second);

    // field functions
    field_functions_second = std::make_shared<FieldFunctions<dim>>();
    set_field_functions_second();
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

  std::shared_ptr<BoundaryDescriptor<1, dim> const>
  get_boundary_descriptor() const
  {
    return boundary_descriptor;
  }

  std::shared_ptr<FieldFunctions<dim> const>
  get_field_functions() const
  {
    return field_functions;
  }

  virtual std::shared_ptr<Poisson::PostProcessorBase<dim, Number>>
  create_postprocessor_second() = 0;

  Parameters const &
  get_parameters_second() const
  {
    return param_second;
  }

  std::shared_ptr<Grid<dim> const>
  get_grid_second() const
  {
    return grid_second;
  }

  std::shared_ptr<BoundaryDescriptor<1, dim> const>
  get_boundary_descriptor_second() const
  {
    return boundary_descriptor_second;
  }

  std::shared_ptr<FieldFunctions<dim> const>
  get_field_functions_second() const
  {
    return field_functions_second;
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

  Parameters param, param_second;

  std::shared_ptr<Grid<dim>> grid, grid_second;

  std::shared_ptr<BoundaryDescriptor<1, dim>> boundary_descriptor, boundary_descriptor_second;
  std::shared_ptr<FieldFunctions<dim>>        field_functions, field_functions_second;

  std::string parameter_file;

  OutputParameters output_parameters;

private:
  virtual void
  set_parameters() = 0;

  virtual void
  create_grid() = 0;

  virtual void
  set_boundary_descriptor() = 0;

  virtual void
  set_field_functions() = 0;

  virtual void
  set_parameters_second() = 0;

  virtual void
  create_grid_second() = 0;

  virtual void
  set_boundary_descriptor_second() = 0;

  virtual void
  set_field_functions_second() = 0;
};

} // namespace Poisson
} // namespace ExaDG


#endif /* INCLUDE_EXADG_POISSON_USER_INTERFACE_APPLICATION_BASE_H_ */
