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
#include <exadg/grid/grid_utilities.h>
#include <exadg/incompressible_navier_stokes/postprocessor/postprocessor.h>
#include <exadg/incompressible_navier_stokes/user_interface/boundary_descriptor.h>
#include <exadg/incompressible_navier_stokes/user_interface/field_functions.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>
#include <exadg/poisson/user_interface/analytical_solution.h>
#include <exadg/poisson/user_interface/field_functions.h>
#include <exadg/poisson/user_interface/parameters.h>
#include <exadg/postprocessor/output_parameters.h>
#include <exadg/utilities/resolution_parameters.h>

namespace ExaDG
{
template<int>
class Mesh;

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
  setup()
  {
    parse_parameters();

    set_parameters();
    param.check(pcout);
    param.print(pcout, "List of parameters:");

    // grid
    grid->initialize(param.grid, mpi_comm);
    GridUtilities::create_mapping(mapping, param.grid.element_type, param.grid.mapping_degree);
    create_grid();
    print_grid_info(pcout, *grid);

    // boundary conditions
    boundary_descriptor = std::make_shared<BoundaryDescriptor<dim>>();
    set_boundary_descriptor();
    verify_boundary_conditions<dim, Number>(*boundary_descriptor, *grid);

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
  setup_poisson()
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

  MPI_Comm const & mpi_comm;

  dealii::ConditionalOStream pcout;

  Parameters param;

  std::shared_ptr<Grid<dim>> grid;

  std::shared_ptr<dealii::Mapping<dim>> mapping;

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
  create_grid() = 0;

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

template<int dim, typename Number>
class ApplicationBasePrecursor : public ApplicationBase<dim, Number>
{
public:
  ApplicationBasePrecursor(std::string parameter_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(parameter_file, comm)
  {
  }

  virtual ~ApplicationBasePrecursor()
  {
  }

  virtual void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    resolution.add_parameters(prm);
  }

  void
  setup() final
  {
    this->parse_parameters();

    // resolution parameters
    set_resolution_parameters();

    // actual domain
    ApplicationBase<dim, Number>::setup();

    // precursor domain

    // parameters
    set_parameters_precursor();
    param_pre.check(this->pcout);
    param_pre.print(this->pcout, "List of parameters for precursor domain:");

    // make some additional parameter checks
    AssertThrow(param_pre.ale_formulation == false, dealii::ExcMessage("not implemented."));
    AssertThrow(this->param.ale_formulation == false, dealii::ExcMessage("not implemented."));

    AssertThrow(
      param_pre.calculation_of_time_step_size == this->param.calculation_of_time_step_size,
      dealii::ExcMessage("Type of time step calculation has to be the same for both domains."));

    AssertThrow(param_pre.adaptive_time_stepping == this->param.adaptive_time_stepping,
                dealii::ExcMessage(
                  "Type of time step calculation has to be the same for both domains."));

    AssertThrow(param_pre.solver_type == SolverType::Unsteady and
                  this->param.solver_type == SolverType::Unsteady,
                dealii::ExcMessage("This is an unsteady solver. Check parameters."));

    // For the two-domain solver the parameter start_with_low_order has to be true.
    // This is due to the fact that the setup function of the time integrator initializes
    // the solution at previous time instants t_0 - dt, t_0 - 2*dt, ... in case of
    // start_with_low_order == false. However, the combined time step size
    // is not known at this point since the two domains have to first communicate with each other
    // in order to find the minimum time step size. Hence, the easiest way to avoid these kind of
    // inconsistencies is to preclude the case start_with_low_order == false.
    AssertThrow(param_pre.start_with_low_order == true and this->param.start_with_low_order == true,
                dealii::ExcMessage("start_with_low_order has to be true for two-domain solver."));

    // grid
    grid_pre->initialize(param_pre.grid, this->mpi_comm);
    GridUtilities::create_mapping(mapping_pre,
                                  param_pre.grid.element_type,
                                  param_pre.grid.mapping_degree);
    create_grid_precursor();
    print_grid_info(this->pcout, *grid_pre);

    // boundary conditions
    boundary_descriptor_pre = std::make_shared<BoundaryDescriptor<dim>>();
    set_boundary_descriptor_precursor();
    verify_boundary_conditions<dim, Number>(*boundary_descriptor_pre, *grid_pre);

    // field functions
    field_functions_pre = std::make_shared<FieldFunctions<dim>>();
    set_field_functions_precursor();
  }

  virtual std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor_precursor() = 0;

  Parameters const &
  get_parameters_precursor() const
  {
    return param_pre;
  }

  std::shared_ptr<Grid<dim> const>
  get_grid_precursor() const
  {
    return grid_pre;
  }

  std::shared_ptr<dealii::Mapping<dim> const>
  get_mapping_precursor() const
  {
    return mapping_pre;
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
  Parameters param_pre;

  std::shared_ptr<Grid<dim>> grid_pre;

  std::shared_ptr<dealii::Mapping<dim>> mapping_pre;

  std::shared_ptr<FieldFunctions<dim>>     field_functions_pre;
  std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor_pre;

private:
  void
  set_resolution_parameters()
  {
    this->param.degree_u             = resolution.degree;
    this->param.grid.n_refine_global = resolution.refine_space;

    this->param_pre.degree_u             = resolution.degree;
    this->param_pre.grid.n_refine_global = resolution.refine_space;
  }

  virtual void
  set_parameters_precursor() = 0;

  virtual void
  create_grid_precursor() = 0;

  virtual void
  set_boundary_descriptor_precursor() = 0;

  virtual void
  set_field_functions_precursor() = 0;

  ResolutionParameters resolution;
};


} // namespace IncNS
} // namespace ExaDG


#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_APPLICATION_BASE_H_ */
