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

#ifndef APPLICATIONS_POISSON_TEST_CASES_PERIODIC_BOX_H_
#define APPLICATIONS_POISSON_TEST_CASES_PERIODIC_BOX_H_

// ExaDG
#include <exadg/grid/periodic_box.h>

namespace ExaDG
{
namespace Poisson
{
enum class MeshType
{
  Cartesian,
  Curvilinear
};

template<int dim, int n_components, typename Number>
class Application : public ApplicationBase<dim, n_components, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, n_components, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    ApplicationBase<dim, n_components, Number>::add_parameters(prm);

    // clang-format off
    prm.enter_subsection("Application");
      prm.add_parameter("MeshType", mesh_type_string, "Type of mesh (Cartesian versus curvilinear).", dealii::Patterns::Selection("Cartesian|Curvilinear"));
    prm.leave_subsection();
    // clang-format on
  }

private:
  void
  parse_parameters() final
  {
    ApplicationBase<dim, n_components, Number>::parse_parameters();

    Utilities::string_to_enum(mesh_type, mesh_type_string);
  }

  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.right_hand_side = false;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.grid.mapping_degree     = 1;
    this->param.spatial_discretization  = SpatialDiscretization::DG;
    this->param.IP_factor               = 1.0e0;

    // SOLVER
    this->param.solver         = Poisson::Solver::CG;
    this->param.preconditioner = Preconditioner::None;
  }

  void
  create_grid() final
  {
    double const left = -1.0, right = 1.0;
    double const deformation = 0.1;

    bool curvilinear_mesh = false;
    if(mesh_type == MeshType::Cartesian)
    {
      // do nothing
    }
    else if(mesh_type == MeshType::Curvilinear)
    {
      curvilinear_mesh = true;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    create_periodic_box(this->grid->triangulation,
                        this->param.grid.n_refine_global,
                        this->grid->periodic_faces,
                        this->n_subdivisions_1d_hypercube,
                        left,
                        right,
                        curvilinear_mesh,
                        deformation);
  }

  void
  set_boundary_descriptor() final
  {
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution.reset(new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(1));
  }

  std::shared_ptr<PostProcessorBase<dim, n_components, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    std::shared_ptr<PostProcessorBase<dim, n_components, Number>> pp;
    pp.reset(new PostProcessor<dim, n_components, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  std::string mesh_type_string = "Cartesian";
  MeshType    mesh_type        = MeshType::Cartesian;
};

} // namespace Poisson

} // namespace ExaDG

#include <exadg/poisson/user_interface/implement_get_application.h>

#endif
