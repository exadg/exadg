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
  MyFunction(const unsigned int n_components = dim, const double time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
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
  typedef typename ApplicationBase<dim, Number>::PeriodicFaces PeriodicFaces;

  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  void
  set_input_parameters_fluid(IncNS::InputParameters & param)
  {
    (void)param;

    // Here, set all parameters differing from their default values as initialized in
    // IncNS::InputParameters::InputParameters()
  }

  void
  set_input_parameters_ale(Poisson::InputParameters & param)
  {
    (void)param;

    // Here, set all parameters differing from their default values as initialized in
    // Poisson::InputParameters::InputParameters()
  }

  void
  create_grid_fluid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
                    PeriodicFaces &                                   periodic_faces,
                    unsigned int const                                n_refine_space,
                    std::shared_ptr<Mapping<dim>> &                   mapping,
                    unsigned int const                                mapping_degree)
  {
    // to avoid warnings (unused variable) use ...
    (void)triangulation;
    (void)periodic_faces;
    (void)n_refine_space;
    (void)mapping;
    (void)mapping_degree;
  }

  void
  set_boundary_conditions_fluid(
    std::shared_ptr<IncNS::BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
    std::shared_ptr<IncNS::BoundaryDescriptorP<dim>> boundary_descriptor_pressure)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // these lines show exemplarily how the boundary descriptors are filled

    // velocity
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_velocity->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(dim)));

    // pressure
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_pressure->dirichlet_bc.insert(pair(1, new Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions_fluid(std::shared_ptr<IncNS::FieldFunctions<dim>> field_functions)
  {
    // these lines show exemplarily how the field functions are filled
    field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  void set_boundary_conditions_ale(
    std::shared_ptr<Poisson::BoundaryDescriptor<1, dim>> boundary_descriptor)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // these lines show exemplarily how the boundary descriptors are filled
    boundary_descriptor->dirichlet_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
    boundary_descriptor->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions_ale(std::shared_ptr<Poisson::FieldFunctions<dim>> field_functions)
  {
    // these lines show exemplarily how the field functions are filled
    field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
  }

  void
  set_input_parameters_ale(Structure::InputParameters & parameters)
  {
    (void)parameters;
  }

  void
  set_boundary_conditions_ale(
    std::shared_ptr<Structure::BoundaryDescriptor<dim>> boundary_descriptor)
  {
    (void)boundary_descriptor;
  }

  void
  set_material_ale(Structure::MaterialDescriptor & material_descriptor)
  {
    (void)material_descriptor;
  }

  void
  set_field_functions_ale(std::shared_ptr<Structure::FieldFunctions<dim>> field_functions)
  {
    (void)field_functions;
  }


  std::shared_ptr<IncNS::PostProcessorBase<dim, Number>>
  construct_postprocessor_fluid(unsigned int const degree, MPI_Comm const & mpi_comm)
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
  set_input_parameters_structure(Structure::InputParameters & parameters)
  {
    (void)parameters;
  }

  void
  create_grid_structure(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
                        PeriodicFaces &                                   periodic_faces,
                        unsigned int const                                n_refine_space,
                        std::shared_ptr<Mapping<dim>> &                   mapping,
                        unsigned int const                                mapping_degree)
  {
    (void)triangulation;
    (void)periodic_faces;
    (void)n_refine_space;
    (void)mapping;
    (void)mapping_degree;
  }

  void
  set_boundary_conditions_structure(
    std::shared_ptr<Structure::BoundaryDescriptor<dim>> boundary_descriptor)
  {
    (void)boundary_descriptor;
  }

  void
  set_material_structure(Structure::MaterialDescriptor & material_descriptor)
  {
    (void)material_descriptor;
  }

  void
  set_field_functions_structure(std::shared_ptr<Structure::FieldFunctions<dim>> field_functions)
  {
    (void)field_functions;
  }

  std::shared_ptr<Structure::PostProcessor<dim, Number>>
  construct_postprocessor_structure(unsigned int const degree, MPI_Comm const & mpi_comm)
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
