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

#ifndef APPLICATIONS_FSI_TEMPLATE_H_
#define APPLICATIONS_FSI_TEMPLATE_H_

namespace ExaDG
{
namespace FSI
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
  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  void
  set_input_parameters_fluid(IncNS::InputParameters & param) final
  {
    (void)param;

    // Here, set all parameters differing from their default values as initialized in
    // IncNS::InputParameters::InputParameters()
  }

  void
  set_input_parameters_ale(Poisson::InputParameters & param) final
  {
    (void)param;

    // Here, set all parameters differing from their default values as initialized in
    // Poisson::InputParameters::InputParameters()
  }

  std::shared_ptr<Grid<dim, Number>>
  create_grid_fluid(GridData const & data, MPI_Comm const & mpi_comm) final
  {
    std::shared_ptr<Grid<dim, Number>> grid = std::make_shared<Grid<dim, Number>>(data, mpi_comm);

    // create triangulation

    grid->triangulation->refine_global(data.n_refine_global);

    return grid;
  }

  void
  set_boundary_conditions_fluid(
    std::shared_ptr<IncNS::BoundaryDescriptor<dim>> boundary_descriptor) final
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // these lines show exemplarily how the boundary descriptors are filled

    // velocity
    boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->velocity->neumann_bc.insert(
      pair(1, new Functions::ZeroFunction<dim>(dim)));

    // pressure
    boundary_descriptor->pressure->neumann_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->pressure->dirichlet_bc.insert(
      pair(1, new Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions_fluid(std::shared_ptr<IncNS::FieldFunctions<dim>> field_functions) final
  {
    // these lines show exemplarily how the field functions are filled
    field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  void set_boundary_conditions_ale(
    std::shared_ptr<Poisson::BoundaryDescriptor<1, dim>> boundary_descriptor) final
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // these lines show exemplarily how the boundary descriptors are filled
    boundary_descriptor->dirichlet_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
    boundary_descriptor->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions_ale(std::shared_ptr<Poisson::FieldFunctions<dim>> field_functions) final
  {
    // these lines show exemplarily how the field functions are filled
    field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
  }

  void
  set_input_parameters_ale(Structure::InputParameters & parameters) final
  {
    (void)parameters;
  }

  void
  set_boundary_conditions_ale(
    std::shared_ptr<Structure::BoundaryDescriptor<dim>> boundary_descriptor) final
  {
    (void)boundary_descriptor;
  }

  void
  set_material_ale(Structure::MaterialDescriptor & material_descriptor) final
  {
    (void)material_descriptor;
  }

  void
  set_field_functions_ale(std::shared_ptr<Structure::FieldFunctions<dim>> field_functions) final
  {
    (void)field_functions;
  }


  std::shared_ptr<IncNS::PostProcessorBase<dim, Number>>
  create_postprocessor_fluid(unsigned int const degree, MPI_Comm const & mpi_comm) final
  {
    (void)degree;

    // these lines show exemplarily how the postprocessor is constructued
    IncNS::PostProcessorData<dim> pp_data;

    std::shared_ptr<IncNS::PostProcessorBase<dim, Number>> pp;
    pp.reset(new IncNS::PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }

  // Structure
  void
  set_input_parameters_structure(Structure::InputParameters & parameters) final
  {
    (void)parameters;
  }

  std::shared_ptr<Grid<dim, Number>>
  create_grid_structure(GridData const & data, MPI_Comm const & mpi_comm) final
  {
    std::shared_ptr<Grid<dim, Number>> grid = std::make_shared<Grid<dim, Number>>(data, mpi_comm);

    // create triangulation

    grid->triangulation->refine_global(data.n_refine_global);

    return grid;
  }

  void
  set_boundary_conditions_structure(
    std::shared_ptr<Structure::BoundaryDescriptor<dim>> boundary_descriptor) final
  {
    (void)boundary_descriptor;
  }

  void
  set_material_structure(Structure::MaterialDescriptor & material_descriptor) final
  {
    (void)material_descriptor;
  }

  void
  set_field_functions_structure(
    std::shared_ptr<Structure::FieldFunctions<dim>> field_functions) final
  {
    (void)field_functions;
  }

  std::shared_ptr<Structure::PostProcessor<dim, Number>>
  create_postprocessor_structure(unsigned int const degree, MPI_Comm const & mpi_comm) final
  {
    (void)degree;

    Structure::PostProcessorData<dim>                      pp_data;
    std::shared_ptr<Structure::PostProcessor<dim, Number>> pp;

    pp.reset(new Structure::PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace FSI

template<int dim, typename Number>
std::shared_ptr<FSI::ApplicationBase<dim, Number>>
get_application(std::string input_file)
{
  return std::make_shared<FSI::Application<dim, Number>>(input_file);
}

} // namespace ExaDG

#endif /* APPLICATIONS_FSI_TEMPLATE_H_ */
