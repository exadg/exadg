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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEMPLATE_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEMPLATE_H_

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

//  Example of a user defined function
template<int dim>
class MyFunction : public Function<dim>
{
public:
  MyFunction(unsigned int const n_components = dim, double const time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const component = 0) const
  {
    (void)p;
    (void)component;

    return 0.0;
  }
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  void
  set_input_parameters(unsigned int const degree) final
  {
    // Here, set all parameters differing from their default values as initialized in
    // IncNS::InputParameters::InputParameters()

    this->param.degree_u = degree;
  }

  std::shared_ptr<Grid<dim, Number>>
  create_grid(GridData const & grid_data) final
  {
    std::shared_ptr<Grid<dim, Number>> grid =
      std::make_shared<Grid<dim, Number>>(grid_data, this->mpi_comm);

    // create triangulation

    grid->triangulation->refine_global(grid_data.n_refine_global);

    return grid;
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // these lines show how the boundary descriptors are filled

    // velocity
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));
    this->boundary_descriptor->velocity->neumann_bc.insert(
      pair(1, new Functions::ZeroFunction<dim>(dim)));

    // pressure
    this->boundary_descriptor->pressure->neumann_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));
    this->boundary_descriptor->pressure->dirichlet_bc.insert(
      pair(1, new Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions() final
  {
    // these lines show how the field functions are filled
    this->field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    this->field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    this->field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    // these lines show how the postprocessor is constructed
    PostProcessorData<dim> pp_data;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEMPLATE_H_ */
