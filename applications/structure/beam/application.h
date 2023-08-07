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

#ifndef STRUCTURE_BEAM
#define STRUCTURE_BEAM

namespace ExaDG
{
namespace Structure
{
template<int dim>
class BendingMoment : public dealii::Function<dim>
{
public:
  BendingMoment(double force, double height, bool incremental_loading)
    : dealii::Function<dim>(dim),
      force_max(force / (height / 2)),
      incremental_loading(incremental_loading)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const final
  {
    double factor = 1.0;
    if(incremental_loading)
      factor = this->get_time();

    if(c == 0)
      return factor * force_max * p[1];
    else
      return 0.0;
  }

private:
  double const force_max;
  bool const   incremental_loading;
};

template<int dim>
class SingleForce : public dealii::Function<dim>
{
public:
  SingleForce(double force, double length, bool incremental_loading)
    : dealii::Function<dim>(dim),
      force_per_length(force / length),
      incremental_loading(incremental_loading)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const final
  {
    (void)p;

    double factor = 1.0;
    if(incremental_loading)
      factor = this->get_time();

    if(c == 1)
      return -factor * force_per_length;
    else
      return 0.0;
  }

private:
  double const force_per_length;
  bool const   incremental_loading;
};

template<int dim>
class SolutionSF : public dealii::Function<dim>
{
public:
  SolutionSF(double length, double height, double width, double singleforce)
    : dealii::Function<dim>(dim),
      length(length),
      height(height),
      width(width),
      lineforce(singleforce * width)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const final
  {
    (void)p;
    (void)c;

    if(c == 1)
    {
      return -(length * length * length * lineforce /
               (6 * 200e3 * width * height * height * height / 12)) *
             (-p[0] * p[0] * p[0] / (length * length * length) +
              3 * p[0] * p[0] / (length * length));
    }
    else
      return 0.0;
  }

private:
  double const length;
  double const height;
  double const width;
  double const lineforce;
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
      prm.add_parameter("Length", length, "Length of domain.");
      prm.add_parameter("Height", height, "Height of domain.");
      prm.add_parameter("Width", width, "Width of domain.");
      prm.add_parameter("BoundaryType", boundary_type, "Type of Neumann BC at right boundary.");
      prm.add_parameter("Force", force, "Value of force on right boundary.");
    }
    prm.leave_subsection();
  }

private:
  void
  set_parameters() final
  {
    this->param.problem_type      = ProblemType::QuasiStatic;
    this->param.body_force        = false;
    this->param.large_deformation = true;

    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.mapping_degree          = 1;

    this->param.load_increment = 0.1;

    this->param.newton_solver_data                     = Newton::SolverData(1e3, 1.e-10, 1.e-6);
    this->param.solver                                 = Solver::FGMRES;
    this->param.solver_data                            = SolverData(1e3, 1.e-14, 1.e-6, 100);
    this->param.preconditioner                         = Preconditioner::Multigrid;
    this->param.update_preconditioner                  = true;
    this->param.update_preconditioner_every_time_steps = 1;
    this->param.update_preconditioner_every_newton_iterations =
      this->param.newton_solver_data.max_iter;
    this->param.update_preconditioner_once_newton_converged = true;
    this->param.multigrid_data.type                         = MultigridType::hpMG;
    this->param.multigrid_data.coarse_problem.solver        = MultigridCoarseGridSolver::CG;
    this->param.multigrid_data.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;
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
        (void)periodic_face_pairs;
        (void)vector_local_refinements;

        dealii::Point<dim> p1, p2;
        p1[0] = 0;
        p1[1] = -(this->height / 2);
        if(dim == 3)
          p1[2] = -(this->width / 2);

        p2[0] = this->length;
        p2[1] = +(this->height / 2);
        if(dim == 3)
          p2[2] = (this->width / 2);

        std::vector<unsigned int> repetitions(dim);
        repetitions[0] = this->repetitions0;
        repetitions[1] = this->repetitions1;
        if(dim == 3)
          repetitions[2] = this->repetitions2;

        dealii::GridGenerator::subdivided_hyper_rectangle(*this->grid->triangulation,
                                                          repetitions,
                                                          p1,
                                                          p2);

        element_length = this->length / (this->repetitions0 * pow(2, global_refinements));

        double const tol = 1.e-8;
        for(auto cell : tria)
        {
          for(auto const & face : cell.face_indices())
          {
            // left face
            if(std::fabs(cell.face(face)->center()(0) - 0) < tol)
            {
              cell.face(face)->set_all_boundary_ids(1);
            }
            // right face
            else if(std::fabs(cell.face(face)->center()(0) - this->length) < tol)
            {
              cell.face(face)->set_all_boundary_ids(2);
            }
            // top-right edge
            else if(std::fabs(cell.face(face)->center()(0) - this->length) < element_length and
                    std::fabs(cell.face(face)->center()(1) - this->height / 2) < tol)
            {
              if(boundary_type == BoundaryType::SingleForce)
              {
                cell.face(face)->set_all_boundary_ids(3);
              }
              else
              {
                AssertThrow(boundary_type == BoundaryType::BendingMoment,
                            dealii::ExcMessage("Not implemented."));
              }
            }
          }
        }

        tria.refine_global(global_refinements);
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
    typedef typename std::pair<dealii::types::boundary_id, dealii::ComponentMask> pair_mask;

    this->boundary_descriptor->neumann_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));

    // left side
    this->boundary_descriptor->dirichlet_bc.insert(
      pair(1, new dealii::Functions::ZeroFunction<dim>(dim)));
    this->boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
      pair(1, new dealii::Functions::ZeroFunction<dim>(dim)));

    this->boundary_descriptor->dirichlet_bc_component_mask.insert(
      pair_mask(1, dealii::ComponentMask()));

    // right side
    bool const incremental_loading = (this->param.problem_type == ProblemType::QuasiStatic);

    if(boundary_type == BoundaryType::BendingMoment)
    {
      this->boundary_descriptor->neumann_bc.insert(
        pair(2, new BendingMoment<dim>(force, height, incremental_loading)));
    }
    else if(boundary_type == BoundaryType::SingleForce)
    {
      this->boundary_descriptor->neumann_bc.insert(
        pair(2, new dealii::Functions::ZeroFunction<dim>(dim)));

      this->boundary_descriptor->neumann_bc.insert(
        pair(3, new SingleForce<dim>(force, element_length, incremental_loading)));
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("not implemented."));
    }
  }

  void
  set_material_descriptor() final
  {
    typedef std::pair<dealii::types::material_id, std::shared_ptr<MaterialData>> Pair;

    MaterialType const type = MaterialType::StVenantKirchhoff;
    // E-Modulus of Steel in unit = [N/mm^2]
    double const E = 200e3, nu = 0.3;
    Type2D const two_dim_type = Type2D::PlaneStress;

    this->material_descriptor->insert(
      Pair(0, new StVenantKirchhoffData<dim>(type, E, nu, two_dim_type)));
  }

  void
  set_field_functions() final
  {
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    this->field_functions->initial_displacement.reset(
      new dealii::Functions::ZeroFunction<dim>(dim));
    this->field_functions->initial_velocity.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessor<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.time_control_data.is_active = this->output_parameters.write;
    pp_data.output_data.directory                   = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename                    = this->output_parameters.filename;
    pp_data.output_data.write_higher_order          = false;
    pp_data.output_data.degree                      = this->param.degree;

    if(boundary_type == BoundaryType::SingleForce)
    {
      pp_data.error_data.time_control_data.is_active = true;
      pp_data.error_data.calculate_relative_errors   = true;
      pp_data.error_data.analytical_solution.reset(
        new SolutionSF<dim>(length, height, width, force));
    }
    else
    {
      AssertThrow(boundary_type == BoundaryType::BendingMoment,
                  dealii::ExcMessage("Not implemented."));
    }

    std::shared_ptr<PostProcessor<dim, Number>> post(
      new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return post;
  }

  // size of geometry
  double length = 1.0, height = 1.0, width = 1.0;

  // single force or bending moment
  enum class BoundaryType
  {
    SingleForce,
    BendingMoment
  };
  BoundaryType boundary_type = BoundaryType::SingleForce;


  double force = 2500;

  double element_length = 1.0;

  // number of subdivisions in each direction
  unsigned int const repetitions0 = 20, repetitions1 = 4, repetitions2 = 1;
};

} // namespace Structure

} // namespace ExaDG

#include <exadg/structure/user_interface/implement_get_application.h>

#endif
