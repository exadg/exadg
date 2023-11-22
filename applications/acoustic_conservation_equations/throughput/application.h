/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef APPLICATIONS_ACOUSTIC_CONSERVATION_EQUATIONS_TEST_CASES_THROUGHPUT_H_
#define APPLICATIONS_ACOUSTIC_CONSERVATION_EQUATIONS_TEST_CASES_THROUGHPUT_H_

#include <exadg/grid/periodic_box.h>

namespace ExaDG
{
namespace Acoustics
{
enum class MeshType
{
  Cartesian,
  Curvilinear
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    prm.enter_subsection("Application");
    {
      prm.add_parameter("MeshType", mesh_type, "Type of mesh (Cartesian versus curvilinear).");
    }
    prm.leave_subsection();
  }

private:
  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.formulation = Formulation::SkewSymmetric;

    // PHYSICAL QUANTITIES
    this->param.start_time     = 0.0;
    this->param.end_time       = 1.0;
    this->param.speed_of_sound = 1.0;
    this->param.density        = 1.0;

    // TEMPORAL DISCRETIZATION
    this->param.calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
    this->param.time_step_size                = 1e-03;
    this->param.order_time_integrator         = 2;
    this->param.start_with_low_order          = false;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.mapping_degree          = 1;
    this->param.degree_p                = this->param.degree_u;
    this->param.degree_u                = this->param.degree_p;
  }

  void
  create_grid() final
  {
    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> &                        tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & periodic_face_pairs,
          unsigned int const                                       global_refinements,
          std::vector<unsigned int> const & /* vector_local_refinements*/) {
        double const left = -1.0, right = 1.0;
        double const deformation = 0.1;

        create_periodic_box(tria,
                            global_refinements,
                            periodic_face_pairs,
                            this->n_subdivisions_1d_hypercube,
                            left,
                            right,
                            mesh_type == MeshType::Curvilinear,
                            deformation);
      };

    GridUtilities::create_triangulation<dim>(
      *this->grid, this->mpi_comm, this->param.grid, lambda_create_triangulation, {});
  }

  void
  set_boundary_descriptor() final
  {
    // test case with purely periodic boundary conditions
    // boundary descriptors remain empty for velocity and pressure
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_pressure =
      std::make_shared<dealii::Functions::ZeroFunction<dim>>(1);

    this->field_functions->initial_solution_velocity =
      std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim);
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // no postprocessing

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  MeshType mesh_type = MeshType::Cartesian;
};

} // namespace Acoustics

} // namespace ExaDG

#include <exadg/acoustic_conservation_equations/user_interface/implement_get_application.h>

#endif /*APPLICATIONS_ACOUSTIC_CONSERVATION_EQUATIONS_TEST_CASES_THROUGHPUT_H_*/
