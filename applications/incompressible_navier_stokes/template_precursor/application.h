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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEMPLATE_PRECURSOR_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEMPLATE_PRECURSOR_H_

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim, typename Number>
class Application : public ApplicationBasePrecursor<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBasePrecursor<dim, Number>(input_file, comm)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  void
  set_input_parameters(unsigned int const degree) final
  {
    this->param.degree_u = degree;
  }

  void
  set_input_parameters_precursor(unsigned int const degree) final
  {
    this->param_pre.degree_u = degree;
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


  std::shared_ptr<Grid<dim, Number>>
  create_grid_precursor(GridData const & grid_data) final
  {
    std::shared_ptr<Grid<dim, Number>> grid =
      std::make_shared<Grid<dim, Number>>(grid_data, this->mpi_comm);

    // create triangulation

    grid->triangulation->refine_global(grid_data.n_refine_global);

    return grid;
  }


  void
  set_boundary_conditions() final
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;
  }

  void
  set_boundary_conditions_precursor() final
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    this->field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    this->field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  void
  set_field_functions_precursor() final
  {
    this->field_functions_pre->initial_solution_velocity.reset(
      new Functions::ZeroFunction<dim>(dim));
    this->field_functions_pre->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    this->field_functions_pre->analytical_solution_pressure.reset(
      new Functions::ZeroFunction<dim>(1));
    this->field_functions_pre->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    std::shared_ptr<PostProcessorBase<dim, Number>> pp;

    PostProcessorData<dim> pp_data;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor_precursor() final
  {
    std::shared_ptr<PostProcessorBase<dim, Number>> pp;

    PostProcessorData<dim> pp_data;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application_precursor.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEMPLATE_PRECURSOR_H_ */
