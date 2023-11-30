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
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, n_components, Number>::add_parameters(prm);

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
    this->param.right_hand_side = false;

    // SPATIAL DISCRETIZATION
    this->param.grid.element_type = ElementType::Hypercube;
    if(this->param.grid.element_type == ElementType::Simplex)
    {
      this->param.grid.triangulation_type           = TriangulationType::FullyDistributed;
      this->param.grid.create_coarse_triangulations = false;
    }
    else if(this->param.grid.element_type == ElementType::Hypercube)
    {
      this->param.grid.triangulation_type           = TriangulationType::Distributed;
      this->param.grid.create_coarse_triangulations = false;
    }

    this->param.mapping_degree         = 1;
    this->param.spatial_discretization = SpatialDiscretization::DG;
    this->param.IP_factor              = 1.0e0;

    // SOLVER
    this->param.solver         = LinearSolver::CG;
    this->param.preconditioner = Preconditioner::None;
  }

  void
  create_grid(Grid<dim> & grid, std::shared_ptr<dealii::Mapping<dim>> & mapping) final
  {
    (void)mapping;

    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> &                        tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & periodic_face_pairs,
          unsigned int const                                       global_refinements,
          std::vector<unsigned int> const &                        vector_local_refinements) {
        (void)vector_local_refinements;

        double const left = -1.0, right = 1.0;

        // periodic boundary conditions are currently not available in deal.II for simplex meshes

        if(this->param.grid.element_type == ElementType::Hypercube)
        {
          double const deformation = 0.1;

          create_periodic_box(tria,
                              global_refinements,
                              periodic_face_pairs,
                              this->n_subdivisions_1d_hypercube,
                              left,
                              right,
                              mesh_type == MeshType::Curvilinear,
                              deformation);
        }
        else if(this->param.grid.element_type == ElementType::Simplex)
        {
          dealii::GridGenerator::subdivided_hyper_cube_with_simplices(
            tria, this->n_subdivisions_1d_hypercube, left, right);

          tria.refine_global(global_refinements);
        }
        else
        {
          AssertThrow(false, ExcNotImplemented());
        }
      };


    GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
                                                            {} /* no local refinements */);
  }

  void
  set_boundary_descriptor() final
  {
    // periodic boundary conditions are currently not available in deal.II for simplex meshes.
    // Hence, we set homogeneous Dirichlet boundary conditions in case of simplex meshes
    if(this->param.grid.element_type == ElementType::Simplex)
    {
      typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        pair;

      this->boundary_descriptor->dirichlet_bc.insert(
        pair(0, new dealii::Functions::ZeroFunction<dim>(1)));
    }
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
