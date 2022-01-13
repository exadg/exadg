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
#include <exadg/convection_diffusion/user_interface/boundary_descriptor.h>
#include <exadg/grid/grid.h>
#include <exadg/incompressible_navier_stokes/postprocessor/postprocessor.h>
#include <exadg/incompressible_navier_stokes/user_interface/boundary_descriptor.h>
#include <exadg/incompressible_navier_stokes/user_interface/field_functions.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>
#include <exadg/poisson/user_interface/analytical_solution.h>
#include <exadg/poisson/user_interface/field_functions.h>
#include <exadg/poisson/user_interface/parameters.h>

namespace ExaDG
{
template<int>
class Mesh;

namespace IncNS
{
using namespace dealii;

template<int dim, typename Number>
class ApplicationBase
{
public:
  virtual void
  add_parameters(ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("Output");
      prm.add_parameter("OutputDirectory",  output_directory, "Directory where output is written.");
      prm.add_parameter("OutputName",       output_name,      "Name of output files.");
      prm.add_parameter("WriteOutput",  	  write_output,     "Decides whether vtu output is written.");
    prm.leave_subsection();
    // clang-format on
  }

  ApplicationBase(std::string parameter_file, MPI_Comm const & comm)
    : mpi_comm(comm), parameter_file(parameter_file)
  {
    field_functions     = std::make_shared<FieldFunctions<dim>>();
    boundary_descriptor = std::make_shared<BoundaryDescriptor<dim>>();

    poisson_field_functions     = std::make_shared<Poisson::FieldFunctions<dim>>();
    poisson_boundary_descriptor = std::make_shared<Poisson::BoundaryDescriptor<1, dim>>();
  }

  virtual ~ApplicationBase()
  {
  }

  void
  set_parameters(unsigned int const degree,
                 unsigned int const refine_space,
                 unsigned int const n_subdivisions_1d_hypercube,
                 unsigned int const refine_time)
  {
    this->param.degree_u = degree;

    this->param.grid.n_refine_global             = refine_space;
    this->param.grid.n_subdivisions_1d_hypercube = n_subdivisions_1d_hypercube;

    this->param.n_refine_time = refine_time;

    this->set_parameters();
  }

  virtual void
  set_parameters() = 0;

  virtual std::shared_ptr<Grid<dim, Number>>
  create_grid() = 0;

  virtual void
  set_boundary_descriptor() = 0;

  virtual void
  set_field_functions() = 0;

  virtual std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() = 0;

  // Moving mesh (analytical function)
  virtual std::shared_ptr<Function<dim>>
  create_mesh_movement_function()
  {
    std::shared_ptr<Function<dim>> mesh_motion =
      std::make_shared<Functions::ZeroFunction<dim>>(dim);

    return mesh_motion;
  }

  // Moving mesh (Poisson problem)
  void
  set_parameters_poisson(unsigned int const degree)
  {
    poisson_param.degree = degree;

    this->set_parameters_poisson();
  }

  virtual void
  set_parameters_poisson()
  {
    AssertThrow(false,
                ExcMessage("Has to be overwritten by derived classes in order "
                           "to use Poisson solver for mesh movement."));
  }

  virtual void
  set_boundary_descriptor_poisson()
  {
    AssertThrow(false,
                ExcMessage("Has to be overwritten by derived classes in order "
                           "to use Poisson solver for mesh movement."));
  }

  virtual void
  set_field_functions_poisson()
  {
    AssertThrow(false,
                ExcMessage("Has to be overwritten by derived classes in order "
                           "to use Poisson solver for mesh movement."));
  }

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
  MPI_Comm const & mpi_comm;

  Parameters                               param;
  std::shared_ptr<FieldFunctions<dim>>     field_functions;
  std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor;

  // solve mesh deformation by a Poisson problem
  Poisson::Parameters                                  poisson_param;
  std::shared_ptr<Poisson::FieldFunctions<dim>>        poisson_field_functions;
  std::shared_ptr<Poisson::BoundaryDescriptor<1, dim>> poisson_boundary_descriptor;

  std::string parameter_file;

  std::string output_directory = "output/", output_name = "output";
  bool        write_output = false;
};

template<int dim, typename Number>
class ApplicationBasePrecursor : public ApplicationBase<dim, Number>
{
public:
  ApplicationBasePrecursor(std::string parameter_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(parameter_file, comm)
  {
    field_functions_pre     = std::make_shared<FieldFunctions<dim>>();
    boundary_descriptor_pre = std::make_shared<BoundaryDescriptor<dim>>();
  }

  virtual ~ApplicationBasePrecursor()
  {
  }

  void
  set_parameters_precursor(unsigned int const degree, unsigned int const refine_space)
  {
    this->param_pre.degree_u             = degree;
    this->param_pre.grid.n_refine_global = refine_space;

    this->set_parameters_precursor();
  }

  virtual void
  set_parameters_precursor() = 0;

  virtual std::shared_ptr<Grid<dim, Number>>
  create_grid_precursor() = 0;

  virtual void
  set_boundary_descriptor_precursor() = 0;

  virtual void
  set_field_functions_precursor() = 0;

  virtual std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor_precursor() = 0;

  Parameters const &
  get_parameters_precursor() const
  {
    return param_pre;
  }

  std::shared_ptr<BoundaryDescriptor<dim> const>
  get_boundary_descriptor_precursor() const
  {
    return boundary_descriptor_pre;
  }

  std::shared_ptr<FieldFunctions<dim> const>
  get_field_functions_precursor() const
  {
    return field_functions_pre;
  }

protected:
  Parameters                               param_pre;
  std::shared_ptr<FieldFunctions<dim>>     field_functions_pre;
  std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor_pre;
};


} // namespace IncNS
} // namespace ExaDG


#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_APPLICATION_BASE_H_ */
