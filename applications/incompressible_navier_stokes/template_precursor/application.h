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
namespace Precursor
{
template<int dim, typename Number>
class PrecursorDomain : public Domain<dim, Number>
{
public:
  PrecursorDomain(std::string parameter_file, MPI_Comm const & comm)
    : Domain<dim, Number>(parameter_file, comm)
  {
  }

  void
  set_parameters() final
  {
    // set parameters here
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

    GridUtilities::create_fine_and_coarse_triangulations<dim>(*this->grid,
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

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    std::shared_ptr<PostProcessorBase<dim, Number>> pp;

    PostProcessorData<dim> pp_data;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

template<int dim, typename Number>
class MainDomain : public Domain<dim, Number>
{
public:
  MainDomain(std::string parameter_file, MPI_Comm const & comm)
    : Domain<dim, Number>(parameter_file, comm)
  {
  }

  void
  set_parameters() final
  {
    // set parameters here
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

    GridUtilities::create_fine_and_coarse_triangulations<dim>(*this->grid,
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

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    std::shared_ptr<PostProcessorBase<dim, Number>> pp;

    PostProcessorData<dim> pp_data;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

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
    this->precursor = std::make_shared<PrecursorDomain<dim, Number>>(input_file, comm);
    this->main      = std::make_shared<MainDomain<dim, Number>>(input_file, comm);
  }
};

} // namespace Precursor
} // namespace IncNS
} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/precursor/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEMPLATE_PRECURSOR_H_ */
