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
//  Example of a user defined function
template<int dim>
class MyFunction : public dealii::Function<dim>
{
public:
  MyFunction(unsigned int const n_components = dim, double const time = 0.)
    : dealii::Function<dim>(n_components, time)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    (void)p;
    (void)component;

    return 0.0;
  }
};

namespace StructureFSI
{
template<int dim, typename Number>
class Application : public StructureFSI::ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : StructureFSI::ApplicationBase<dim, Number>(input_file, comm)
  {
  }

private:
  void
  set_parameters() final
  {
  }

  void
  create_grid() final
  {
    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> &                        tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & periodic_face_pairs,
          unsigned int const                                       global_refinements,
          std::vector<unsigned int> const &                        vector_local_refinements) {
        // create triangulation and perform local/global refinements
        (void)tria;
        (void)periodic_face_pairs;
        (void)global_refinements;
        (void)vector_local_refinements;
      };

    GridUtilities::create_triangulation_with_multigrid<dim>(*this->grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
                                                            {} /* no local refinements */);
  }

  void
  set_boundary_descriptor() final
  {
  }

  void
  set_material_descriptor() final
  {
  }

  void
  set_field_functions() final
  {
  }

  std::shared_ptr<Structure::PostProcessor<dim, Number>>
  create_postprocessor() final
  {
    Structure::PostProcessorData<dim>                      pp_data;
    std::shared_ptr<Structure::PostProcessor<dim, Number>> pp;

    pp.reset(new Structure::PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};
} // namespace StructureFSI

namespace FluidFSI
{
template<int dim, typename Number>
class Application : public FluidFSI::ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : FluidFSI::ApplicationBase<dim, Number>(input_file, comm)
  {
  }

private:
  void
  set_parameters() final
  {
    using namespace IncNS;

    Parameters & param = this->param;

    // Set parameters here
    (void)param;
  }

  void
  create_grid() final
  {
    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> &                        tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & periodic_face_pairs,
          unsigned int const                                       global_refinements,
          std::vector<unsigned int> const &                        vector_local_refinements) {
        // create triangulation and perform local/global refinements
        (void)tria;
        (void)periodic_face_pairs;
        (void)global_refinements;
        (void)vector_local_refinements;
      };

    GridUtilities::create_triangulation_with_multigrid<dim>(*this->grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
                                                            {} /* no local refinements */);
  }

  void
  set_boundary_descriptor() final
  {
    std::shared_ptr<IncNS::BoundaryDescriptor<dim>> boundary_descriptor = this->boundary_descriptor;

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    // these lines show exemplarily how the boundary descriptors are filled

    // velocity
    boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->velocity->neumann_bc.insert(
      pair(1, new dealii::Functions::ZeroFunction<dim>(dim)));

    // pressure
    boundary_descriptor->pressure->neumann_bc.insert(0);
    boundary_descriptor->pressure->dirichlet_bc.insert(
      pair(1, new dealii::Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions() final
  {
    std::shared_ptr<IncNS::FieldFunctions<dim>> field_functions = this->field_functions;

    // these lines show exemplarily how the field functions are filled
    field_functions->initial_solution_velocity.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    field_functions->initial_solution_pressure.reset(new dealii::Functions::ZeroFunction<dim>(1));
    field_functions->analytical_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<IncNS::PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    // these lines show exemplarily how the postprocessor is constructued
    IncNS::PostProcessorData<dim> pp_data;

    std::shared_ptr<IncNS::PostProcessorBase<dim, Number>> pp;
    pp.reset(new IncNS::PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  void
  set_parameters_ale_poisson() final
  {
    using namespace Poisson;

    Parameters & param = this->ale_poisson_param;

    // Set parameters here
    (void)param;
  }

  void
  set_boundary_descriptor_ale_poisson() final
  {
    std::shared_ptr<Poisson::BoundaryDescriptor<1, dim>> boundary_descriptor =
      this->ale_poisson_boundary_descriptor;

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    // these lines show exemplarily how the boundary descriptors are filled
    boundary_descriptor->dirichlet_bc.insert(pair(0, new dealii::Functions::ZeroFunction<dim>(1)));
    boundary_descriptor->neumann_bc.insert(pair(1, new dealii::Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions_ale_poisson() final
  {
    std::shared_ptr<Poisson::FieldFunctions<dim>> field_functions =
      this->ale_poisson_field_functions;

    // these lines show exemplarily how the field functions are filled
    field_functions->initial_solution.reset(new dealii::Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(1));
  }

  void
  set_parameters_ale_elasticity() final
  {
    using namespace Structure;

    Parameters & param = this->ale_elasticity_param;

    // Set parameters here
    (void)param;
  }

  void
  set_boundary_descriptor_ale_elasticity() final
  {
  }

  void
  set_material_descriptor_ale_elasticity() final
  {
  }

  void
  set_field_functions_ale_elasticity() final
  {
  }
};
} // namespace FluidFSI

namespace FSI
{
template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
  {
    this->structure = std::make_shared<StructureFSI::Application<dim, Number>>(input_file, comm);
    this->fluid     = std::make_shared<FluidFSI::Application<dim, Number>>(input_file, comm);
  }
};
} // namespace FSI

} // namespace ExaDG

#include <exadg/fluid_structure_interaction/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_FSI_TEMPLATE_H_ */
