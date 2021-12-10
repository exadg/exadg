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

#ifndef APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_TEMPLATE_H_
#define APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_TEMPLATE_H_

#include <exadg/grid/periodic_box.h>

namespace ExaDG
{
namespace ConvDiff
{
using namespace dealii;

template<int dim>
class Velocity : public Function<dim>
{
public:
  Velocity(unsigned int const n_components = 1, double const time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const component = 0) const
  {
    return p[component];
  }
};

enum class MeshType
{
  Cartesian,
  Curvilinear
};

void
string_to_enum(MeshType & enum_type, std::string const & string_type)
{
  // clang-format off
  if     (string_type == "Cartesian")   enum_type = MeshType::Cartesian;
  else if(string_type == "Curvilinear") enum_type = MeshType::Curvilinear;
  else AssertThrow(false, ExcMessage("Not implemented."));
  // clang-format on
}

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(input_file, "", true, true);

    string_to_enum(mesh_type, mesh_type_string);
  }

  void
  add_parameters(ParameterHandler & prm)
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    // clang-format off
    prm.enter_subsection("Application");
      prm.add_parameter("MeshType",  mesh_type_string, "Type of mesh (Cartesian versus curvilinear).", Patterns::Selection("Cartesian|Curvilinear"));
    prm.leave_subsection();
    // clang-format on
  }

  std::string mesh_type_string = "Cartesian";
  MeshType    mesh_type        = MeshType::Cartesian;

  void
  set_input_parameters(unsigned int const degree) final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type    = ProblemType::Unsteady;
    this->param.equation_type   = EquationType::ConvectionDiffusion;
    this->param.right_hand_side = false;
    // Note: set parameter store_analytical_velocity_in_dof_vector to test different implementation
    // variants
    this->param.analytical_velocity_field = true;

    // PHYSICAL QUANTITIES
    this->param.start_time  = 0.0;
    this->param.end_time    = 1.0;
    this->param.diffusivity = 1.0;

    // TEMPORAL DISCRETIZATION
    this->param.temporal_discretization       = TemporalDiscretization::BDF;
    this->param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Implicit;
    this->param.calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
    this->param.time_step_size                = 1.e-2;

    // SPATIAL DISCRETIZATION
    this->param.triangulation_type = TriangulationType::Distributed;
    this->param.mapping            = MappingType::Affine;
    this->param.degree             = degree;

    // convective term
    this->param.numerical_flux_convective_operator =
      NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

    // viscous term
    this->param.IP_factor = 1.0;

    // SOLVER
    this->param.solver         = Solver::GMRES;
    this->param.preconditioner = Preconditioner::None;

    // NUMERICAL PARAMETERS
    this->param.use_cell_based_face_loops               = false;
    this->param.use_combined_operator                   = true;
    this->param.store_analytical_velocity_in_dof_vector = true;
  }

  std::shared_ptr<Grid<dim, Number>>
  create_grid(GridData const & grid_data) final
  {
    std::shared_ptr<Grid<dim, Number>> grid =
      std::make_shared<Grid<dim, Number>>(grid_data, this->mpi_comm);

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
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    create_periodic_box(grid->triangulation,
                        grid_data.n_refine_global,
                        grid->periodic_faces,
                        this->n_subdivisions_1d_hypercube,
                        left,
                        right,
                        curvilinear_mesh,
                        deformation);

    return grid;
  }

  void
  set_boundary_descriptor() final
  {
  }

  void
  set_field_functions() final
  {
    // these lines show exemplarily how the field functions are filled
    this->field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
    this->field_functions->velocity.reset(new Velocity<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

} // namespace ConvDiff

} // namespace ExaDG

#include <exadg/convection_diffusion/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_TEMPLATE_H_ */
