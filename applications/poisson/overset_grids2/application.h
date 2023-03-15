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
#ifndef APPLICATIONS_POISSON_TEST_CASES_TEMPLATE_H_
#define APPLICATIONS_POISSON_TEST_CASES_TEMPLATE_H_

namespace ExaDG
{
namespace Poisson
{
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

    prm.add_parameter("MeshType",
                      mesh_type,
                      "Type of mesh (Matching, ArtificialNonmatching, Nonmatching or Overset).",
                      dealii::Patterns::Selection(
                        "Matching|ArtificialNonmatching|Nonmatching|Overset"));
    prm.add_parameter("Preconditioner",
                      preconditioner,
                      "Which preconditioner is used.",
                      dealii::Patterns::Selection("None|PointJacobi|BlockJacobi|Multigrid"));

    prm.leave_subsection();
  }

private:
  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.right_hand_side = true;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.spatial_discretization  = SpatialDiscretization::DG;
    this->param.IP_factor               = 1.0e0;

    if(mesh_type == "Matching" || mesh_type == "ArtificialNonmatching")
    {
      // in case of ArtificialNonmatching the system is symmetrical, generally this is not the case
      // for non-matching or overset meshes
      this->param.solver = Solver::CG;
    }
    else
    {
      // we need GMRES since the system is not symmetrical
      this->param.solver = Solver::GMRES;
    }

    this->param.solver_data.abs_tol         = 1.e-20;
    this->param.solver_data.rel_tol         = 1.e-10;
    this->param.solver_data.max_iter        = 1e4;
    this->param.compute_performance_metrics = true;

    if(preconditioner == "None")
    {
      this->param.preconditioner = Preconditioner::None;
    }
    else if(preconditioner == "PointJacobi")
    {
      this->param.preconditioner = Preconditioner::PointJacobi;
    }
    else if(preconditioner == "BlockJacobi")
    {
      this->param.preconditioner = Preconditioner::BlockJacobi;
    }
    else if(preconditioner == "Multigrid")
    {
      this->param.preconditioner            = Preconditioner::Multigrid;
      this->param.multigrid_data.type       = MultigridType::cphMG;
      this->param.multigrid_data.p_sequence = PSequenceType::Bisect;
      // MG smoother
      this->param.multigrid_data.smoother_data.smoother        = MultigridSmoother::Chebyshev;
      this->param.multigrid_data.smoother_data.iterations      = 5;
      this->param.multigrid_data.smoother_data.smoothing_range = 20;
      // MG coarse grid solver
      this->param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::CG;
      this->param.multigrid_data.coarse_problem.preconditioner =
        MultigridCoarseGridPreconditioner::AMG;
      this->param.multigrid_data.coarse_problem.solver_data.rel_tol = 1.e-3;
    }
  }

  void
  create_grid() final
  {
    if(mesh_type == "Overset")
    {
      dealii::Triangulation<dim> domain1;
      {
        double const       right = 1.0;
        dealii::Point<dim> p1, p2;
        p1[0] = 0.0;
        p1[1] = 0.0;
        p2[0] = right;
        p2[1] = 1.0;
        dealii::GridGenerator::subdivided_hyper_rectangle(domain1, {3, 3}, p1, p2);

        for(const auto & face : domain1.active_face_iterators())
          if(face->at_boundary())
          {
            if(face->center()[0] > right - 1e-6)
              face->set_boundary_id(0 + 10);
            else
              face->set_boundary_id(0);
          }
      }

      dealii::Triangulation<dim> domain2;
      {
        double const       left = 0.95;
        dealii::Point<dim> p1, p2;
        p1[0] = left;
        p1[1] = 0.0;
        p2[0] = left + 1.0;
        p2[1] = 1.0;
        dealii::GridGenerator::subdivided_hyper_rectangle(domain2, {2, 2}, p1, p2);

        for(const auto & face : domain2.active_face_iterators())
          if(face->at_boundary())
          {
            if(face->center()[0] < left + 1e-6)
              face->set_boundary_id(0 + 11);
            else
              face->set_boundary_id(0);
          }
      }

      dealii::GridGenerator::merge_triangulations(
        domain1, domain2, *this->grid->triangulation, 0.0, true, true);
    }
    else if(mesh_type == "Nonmatching")
    {
      dealii::Triangulation<dim> domain1;
      {
        double const       right = 1.0;
        dealii::Point<dim> p1, p2;
        p1[0] = 0.0;
        p1[1] = 0.0;
        p2[0] = right;
        p2[1] = 1.0;
        dealii::GridGenerator::subdivided_hyper_rectangle(domain1, {2, 2}, p1, p2);

        for(const auto & face : domain1.active_face_iterators())
          if(face->at_boundary())
          {
            if(face->center()[0] > right - 1e-6)
              face->set_boundary_id(0 + 10);
            else
              face->set_boundary_id(0);
          }
      }

      dealii::Triangulation<dim> domain2;
      {
        double const       left = 1.0;
        dealii::Point<dim> p1, p2;
        p1[0] = left;
        p1[1] = 0.0;
        p2[0] = left + 1.0;
        p2[1] = 1.0;
        dealii::GridGenerator::subdivided_hyper_rectangle(domain2, {3, 3}, p1, p2);

        for(const auto & face : domain2.active_face_iterators())
          if(face->at_boundary())
          {
            if(face->center()[0] < left + 1e-6)
              face->set_boundary_id(0 + 11);
            else
              face->set_boundary_id(0);
          }
      }

      dealii::GridGenerator::merge_triangulations(
        domain1, domain2, *this->grid->triangulation, 0.0, true, true);
    }
    else if(mesh_type == "ArtificialNonmatching")
    {
      // ArtificialNonmatching is designed, such that the same mesh is used as in the Matching case.
      // Therefore, results and iterations should be equivalent even though the overset
      // implementation is uesd

      dealii::Triangulation<dim> domain1;
      {
        double const       right = 1.0;
        dealii::Point<dim> p1, p2;
        p1[0] = 0.0;
        p1[1] = 0.0;
        p2[0] = right;
        p2[1] = 1.0;
        dealii::GridGenerator::subdivided_hyper_rectangle(domain1, {3, 3}, p1, p2);

        for(const auto & face : domain1.active_face_iterators())
          if(face->at_boundary())
          {
            if(face->center()[0] > right - 1e-6)
              face->set_boundary_id(0 + 10);
            else
              face->set_boundary_id(0);
          }
      }

      dealii::Triangulation<dim> domain2;
      {
        double const       left = 1.0;
        dealii::Point<dim> p1, p2;
        p1[0] = left;
        p1[1] = 0.0;
        p2[0] = left + 1.0;
        p2[1] = 1.0;
        dealii::GridGenerator::subdivided_hyper_rectangle(domain2, {3, 3}, p1, p2);

        for(const auto & face : domain2.active_face_iterators())
          if(face->at_boundary())
          {
            if(face->center()[0] < left + 1e-6)
              face->set_boundary_id(0 + 11);
            else
              face->set_boundary_id(0);
          }
      }

      dealii::GridGenerator::merge_triangulations(
        domain1, domain2, *this->grid->triangulation, 0.0, true, true);
    }
    else if(mesh_type == "Matching")
    {
      dealii::Point<dim> p1, p2;
      p1[0] = 0.0;
      p1[1] = 0.0;
      p2[0] = 2.0;
      p2[1] = 1.0;
      dealii::GridGenerator::subdivided_hyper_rectangle(*this->grid->triangulation, {6, 3}, p1, p2);

      for(const auto & face : this->grid->triangulation->active_face_iterators())
        if(face->at_boundary())
          face->set_boundary_id(0);
    }


    this->grid->triangulation->refine_global(this->param.grid.n_refine_global);
  }

  void
  set_boundary_descriptor() final
  {
    this->boundary_descriptor->dirichlet_bc.insert(
      std::make_pair(0, std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim)));

    if(mesh_type == "Nonmatching" || mesh_type == "Overset" || mesh_type == "ArtificialNonmatching")
    {
      this->boundary_descriptor->overset_boundaries.insert(std::make_pair(10, 11));
      this->boundary_descriptor->overset_boundaries.insert(std::make_pair(11, 10));
    }
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution.reset(
      new dealii::Functions::ZeroFunction<dim>(n_components));
    this->field_functions->right_hand_side.reset(
      new dealii::Functions::ConstantFunction<dim>(1.0, n_components));
  }

  std::shared_ptr<PostProcessorBase<dim, n_components, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.time_control_data.is_active = this->output_parameters.write;
    pp_data.output_data.directory                   = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = this->output_parameters.filename + "_" + mesh_type;
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.degree             = this->param.degree;
    pp_data.output_data.write_boundary_IDs = true;

    std::shared_ptr<PostProcessorBase<dim, n_components, Number>> pp;
    pp.reset(new PostProcessor<dim, n_components, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  std::string mesh_type      = "Overset";
  std::string preconditioner = "Multigrid";
};


} // namespace Poisson
} // namespace ExaDG

#include <exadg/poisson/user_interface/implement_get_application.h>

#endif /*APPLICATIONS_POISSON_TEST_CASES_TEMPLATE_H_*/
