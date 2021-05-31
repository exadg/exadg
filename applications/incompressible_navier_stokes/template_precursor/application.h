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
  Application(std::string input_file) : ApplicationBasePrecursor<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  void
  set_input_parameters(InputParameters & param) final
  {
    (void)param;
  }

  void
  set_input_parameters_precursor(InputParameters & param)
  {
    (void)param;
  }

  std::shared_ptr<Grid<dim>>
  create_grid(GridData const & data, MPI_Comm const & mpi_comm) final
  {
    std::shared_ptr<Grid<dim>> grid = std::make_shared<Grid<dim>>(data, mpi_comm);

    // create triangulation

    grid->triangulation->refine_global(data.n_refine_global);

    return grid;
  }


  std::shared_ptr<Grid<dim>>
  create_grid_precursor(GridData const & data, MPI_Comm const & mpi_comm) final
  {
    std::shared_ptr<Grid<dim>> grid = std::make_shared<Grid<dim>>(data, mpi_comm);

    // create triangulation

    grid->triangulation->refine_global(data.n_refine_global);

    return grid;
  }


  void
  set_boundary_conditions(
    std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure) final
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    (void)boundary_descriptor_velocity;
    (void)boundary_descriptor_pressure;
  }

  void
  set_boundary_conditions_precursor(
    std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure) final
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    (void)boundary_descriptor_velocity;
    (void)boundary_descriptor_pressure;
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions) final
  {
    field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  void
  set_field_functions_precursor(std::shared_ptr<FieldFunctions<dim>> field_functions) final
  {
    field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm) final
  {
    (void)degree;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;

    PostProcessorData<dim> pp_data;
    pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor_precursor(unsigned int const degree, MPI_Comm const & mpi_comm) final
  {
    (void)degree;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;

    PostProcessorData<dim> pp_data;
    pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace IncNS

template<int dim, typename Number>
std::shared_ptr<IncNS::ApplicationBase<dim, Number>>
get_application(std::string input_file)
{
  return std::make_shared<IncNS::Application<dim, Number>>(input_file);
}

template<int dim, typename Number>
std::shared_ptr<IncNS::ApplicationBasePrecursor<dim, Number>>
get_application(std::string input_file)
{
  return std::make_shared<IncNS::Application<dim, Number>>(input_file);
}

} // namespace ExaDG


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEMPLATE_PRECURSOR_H_ */
