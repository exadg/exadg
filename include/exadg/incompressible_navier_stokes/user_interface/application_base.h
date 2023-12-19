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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_APPLICATION_BASE_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_APPLICATION_BASE_H_

// deal.II
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

// ExaDG
#include <exadg/grid/grid.h>
#include <exadg/grid/grid_utilities.h>
#include <exadg/incompressible_navier_stokes/postprocessor/postprocessor.h>
#include <exadg/incompressible_navier_stokes/user_interface/boundary_descriptor.h>
#include <exadg/incompressible_navier_stokes/user_interface/field_functions.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>
#include <exadg/operators/resolution_parameters.h>
#include <exadg/poisson/user_interface/analytical_solution.h>
#include <exadg/poisson/user_interface/field_functions.h>
#include <exadg/poisson/user_interface/parameters.h>
#include <exadg/postprocessor/output_parameters.h>

namespace ExaDG
{
namespace IncNS
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
  set_parameters_throughput_study(unsigned int const degree,
                                  unsigned int const refine_space,
                                  unsigned int const n_subdivisions_1d_hypercube)
  {
    this->param.degree_u              = degree;
    this->param.grid.n_refine_global  = refine_space;
    this->n_subdivisions_1d_hypercube = n_subdivisions_1d_hypercube;
  }

  void
  set_parameters_convergence_study(unsigned int const degree,
                                   unsigned int const refine_space,
                                   unsigned int const refine_time)
  {
    this->param.degree_u             = degree;
    this->param.grid.n_refine_global = refine_space;
    this->param.n_refine_time        = refine_time;
  }

  virtual void
  setup(std::shared_ptr<Grid<dim>> &                      grid,
        std::shared_ptr<dealii::Mapping<dim>> &           mapping,
        std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings)
  {
    parse_parameters();

    set_parameters();
    param.check(pcout);
    param.print(pcout, "List of parameters:");

    // grid
    grid = std::make_shared<Grid<dim>>();
    create_grid(*grid, mapping, multigrid_mappings);
    print_grid_info(pcout, *grid);

    // boundary conditions
    boundary_descriptor = std::make_shared<BoundaryDescriptor<dim>>();
    set_boundary_descriptor();
    verify_boundary_conditions<dim>(*boundary_descriptor, *grid);

    // field functions
    field_functions = std::make_shared<FieldFunctions<dim>>();
    set_field_functions();
  }

  virtual std::shared_ptr<PostProcessorBase<dim, Number>>
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

  std::shared_ptr<FieldFunctions<dim> const>
  get_field_functions() const
  {
    return field_functions;
  }

  // Analytical mesh motion
  virtual std::shared_ptr<dealii::Function<dim>>
  create_mesh_movement_function()
  {
    std::shared_ptr<dealii::Function<dim>> mesh_motion =
      std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim);

    return mesh_motion;
  }

  // Poisson-type mesh motion (solve PDE problem)
  void
  setup_poisson(std::shared_ptr<Grid<dim> const> const & grid)
  {
    // Note that the grid parameters in Poisson::Parameters are ignored since
    // the grid is created using the parameters specified in IncNS::Parameters
    set_parameters_poisson();
    poisson_param.check();
    poisson_param.print(pcout, "List of parameters for Poisson solver (moving mesh):");

    poisson_boundary_descriptor = std::make_shared<Poisson::BoundaryDescriptor<1, dim>>();
    set_boundary_descriptor_poisson();
    verify_boundary_conditions(*poisson_boundary_descriptor, *grid);

    poisson_field_functions = std::make_shared<Poisson::FieldFunctions<dim>>();
    set_field_functions_poisson();

    AssertThrow(poisson_param.right_hand_side == false,
                dealii::ExcMessage("Poisson problem is used for mesh movement. Hence, "
                                   "the right-hand side has to be zero for the Poisson problem."));
  }

  Poisson::Parameters const &
  get_parameters_poisson() const
  {
    return poisson_param;
  }

  std::shared_ptr<Poisson::BoundaryDescriptor<1, dim> const>
  get_boundary_descriptor_poisson() const
  {
    return poisson_boundary_descriptor;
  }

  std::shared_ptr<Poisson::FieldFunctions<dim> const>
  get_field_functions_poisson() const
  {
    return poisson_field_functions;
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

  std::shared_ptr<FieldFunctions<dim>>     field_functions;
  std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor;

  // solve mesh deformation by a Poisson problem
  Poisson::Parameters                                  poisson_param;
  std::shared_ptr<Poisson::FieldFunctions<dim>>        poisson_field_functions;
  std::shared_ptr<Poisson::BoundaryDescriptor<1, dim>> poisson_boundary_descriptor;

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
  set_field_functions() = 0;

  // Poisson-type mesh motion
  virtual void
  set_parameters_poisson()
  {
    AssertThrow(false,
                dealii::ExcMessage("Has to be overwritten by derived classes in order "
                                   "to use Poisson solver for mesh movement."));
  }

  virtual void
  set_boundary_descriptor_poisson()
  {
    AssertThrow(false,
                dealii::ExcMessage("Has to be overwritten by derived classes in order "
                                   "to use Poisson solver for mesh movement."));
  }

  virtual void
  set_field_functions_poisson()
  {
    AssertThrow(false,
                dealii::ExcMessage("Has to be overwritten by derived classes in order "
                                   "to use Poisson solver for mesh movement."));
  }
};


} // namespace IncNS
} // namespace ExaDG


#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_APPLICATION_BASE_H_ */
