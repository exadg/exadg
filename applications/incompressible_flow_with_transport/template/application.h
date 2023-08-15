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
namespace FTI
{
// Example of a user defined function
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

template<int dim, typename Number>
class Fluid : public FluidBase<dim, Number>
{
public:
  Fluid(std::string parameter_file, MPI_Comm const & comm)
    : FluidBase<dim, Number>(parameter_file, comm)
  {
  }

private:
  void
  set_parameters() final
  {
    using namespace IncNS;

    // Set parameters here
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
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    // velocity
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));
    this->boundary_descriptor->velocity->neumann_bc.insert(
      pair(1, new dealii::Functions::ZeroFunction<dim>(dim)));

    // pressure
    this->boundary_descriptor->pressure->neumann_bc.insert(0);
    this->boundary_descriptor->pressure->dirichlet_bc.insert(
      pair(1, new dealii::Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(
      new dealii::Functions::ZeroFunction<dim>(dim));
    this->field_functions->initial_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->analytical_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<IncNS::PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    IncNS::PostProcessorData<dim> pp_data;

    std::shared_ptr<IncNS::PostProcessorBase<dim, Number>> pp;
    pp.reset(new IncNS::PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

template<int dim, typename Number>
class Scalar : public ScalarBase<dim, Number>
{
public:
  Scalar(std::string parameter_file, MPI_Comm const & comm)
    : ScalarBase<dim, Number>(parameter_file, comm)
  {
  }

private:
  void
  set_parameters() final
  {
    using namespace ConvDiff;
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    this->boundary_descriptor->dirichlet_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(1)));
    this->boundary_descriptor->neumann_bc.insert(
      pair(1, new dealii::Functions::ZeroFunction<dim>(1)));
  }


  void
  set_field_functions() final
  {
    this->field_functions->initial_solution.reset(new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->velocity.reset(new dealii::Functions::ZeroFunction<dim>(1));
  }

  std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    ConvDiff::PostProcessorData<dim> pp_data;

    std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>> pp;
    pp.reset(new ConvDiff::PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
    this->fluid = std::make_shared<Fluid<dim, Number>>(input_file, comm);

    // create one (or even more) scalar fields
    this->scalars.resize(1);
    this->scalars[0] = std::make_shared<Scalar<dim, Number>>(input_file, comm);
  }
};

} // namespace FTI

} // namespace ExaDG

#include <exadg/incompressible_flow_with_transport/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEMPLATE_H_ */
