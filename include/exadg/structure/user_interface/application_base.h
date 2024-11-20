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

#ifndef INCLUDE_EXADG_STRUCTURE_USER_INTERFACE_APPLICATION_BASE_H_
#define INCLUDE_EXADG_STRUCTURE_USER_INTERFACE_APPLICATION_BASE_H_

// deal.II
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

// ExaDG
#include <exadg/grid/grid.h>
#include <exadg/grid/grid_utilities.h>
#include <exadg/postprocessor/output_parameters.h>
#include <exadg/structure/material/library/st_venant_kirchhoff.h>
#include <exadg/structure/postprocessor/postprocessor.h>
#include <exadg/structure/user_interface/boundary_descriptor.h>
#include <exadg/structure/user_interface/field_functions.h>
#include <exadg/structure/user_interface/material_descriptor.h>
#include <exadg/structure/user_interface/parameters.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
class ApplicationBase
{
public:
  virtual void
  add_parameters(dealii::ParameterHandler & prm)
  {
    output_parameters.add_parameters(prm);
  }

  ApplicationBase(std::string parameter_file, MPI_Comm const & comm)
    : mpi_comm(comm),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
      parameter_file(parameter_file),
      n_subdivisions_1d_hypercube(1)
  {
  }

  virtual ~ApplicationBase()
  {
  }

  void
  setup(std::shared_ptr<Grid<dim>> &                      grid,
        std::shared_ptr<dealii::Mapping<dim>> &           mapping,
        std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings)
  {
    parse_parameters();

    // parameters
    set_parameters();
    param.check();
    param.print(pcout, "List of parameters:");

    // grid
    grid = std::make_shared<Grid<dim>>();
    create_grid(*grid, mapping, multigrid_mappings);
    print_grid_info(pcout, *grid);

    // boundary conditions
    boundary_descriptor = std::make_shared<BoundaryDescriptor<dim>>();
    set_boundary_descriptor();
    verify_boundary_conditions(*boundary_descriptor, *grid);

    // material
    material_descriptor = std::make_shared<MaterialDescriptor>();
    set_material_descriptor();

    // field functions
    field_functions = std::make_shared<FieldFunctions<dim>>();
    set_field_functions();
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
  set_parameters_convergence_study(unsigned int const degree,
                                   unsigned int const refine_space,
                                   unsigned int const refine_time)
  {
    this->param.degree               = degree;
    this->param.grid.n_refine_global = refine_space;
    this->param.n_refine_time        = refine_time;
  }

  virtual std::shared_ptr<PostProcessor<dim, Number>>
  create_postprocessor() = 0;

  Parameters const &
  get_parameters() const
  {
    return param;
  }

  std::shared_ptr<BoundaryDescriptor<dim> const>
  get_boundary_descriptor() const
  {
    return boundary_descriptor;
  }

  std::shared_ptr<MaterialDescriptor const>
  get_material_descriptor() const
  {
    return material_descriptor;
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

  MPI_Comm const mpi_comm;

  dealii::ConditionalOStream pcout;

  Parameters param;

  std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor;
  std::shared_ptr<MaterialDescriptor>      material_descriptor;
  std::shared_ptr<FieldFunctions<dim>>     field_functions;

  std::string parameter_file;

  unsigned int n_subdivisions_1d_hypercube;

  OutputParameters output_parameters;

private:
  virtual void
  set_parameters() = 0;

  virtual void
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) = 0;

  virtual void
  set_boundary_descriptor() = 0;

  virtual void
  set_material_descriptor() = 0;

  virtual void
  set_field_functions() = 0;
};

} // namespace Structure
} // namespace ExaDG

#endif
