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

#ifndef STRUCTURE_CAN
#define STRUCTURE_CAN

namespace ExaDG
{
namespace Structure
{
template<int dim>
class DisplacementDBC : public dealii::Function<dim>
{
public:
  DisplacementDBC(double displacement, bool incremental_loading)
    : dealii::Function<dim>(dim),
      displacement(displacement),
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

    if(c == 2)
      return factor * displacement;
    else
      return 0.0;
  }

private:
  double const displacement;
  bool const   incremental_loading;
};

template<int dim>
class AreaForce : public dealii::Function<dim>
{
public:
  AreaForce(double force, bool incremental_loading)
    : dealii::Function<dim>(dim), force(force), incremental_loading(incremental_loading)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const final
  {
    (void)p;

    double factor = 1.0;
    if(incremental_loading)
      factor = this->get_time();

    if(c == 2)
      return factor * force; // area_force in z-direction
    else
      return 0.0;
  }

private:
  double const force;
  bool const   incremental_loading;
};

template<int dim>
class VolumeForce : public dealii::Function<dim>
{
public:
  VolumeForce(double volume_force) : dealii::Function<dim>(dim), volume_force(volume_force)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const final
  {
    (void)p;

    if(c == 2)
      return volume_force; // volume_force in z-direction
    else
      return 0.0;
  }

private:
  double const volume_force;
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
      prm.add_parameter("InnerRadius", inner_radius, "Inner radius.");
      prm.add_parameter("OuterRadius", outer_radius, "Outer radius.");
      prm.add_parameter("Height", height, "Height.");
      prm.add_parameter("UseVolumeForce", use_volume_force, "Use volume force.");
      prm.add_parameter("VolumeForce", volume_force, "Volume force.");
      prm.add_parameter("BoundaryType",
                        boundary_type,
                        "Type of boundary condition, Dirichlet vs Neumann.");
      prm.add_parameter("Displacement",
                        displacement,
                        "Diplacement of right boundary in case of Dirichlet BC.");
      prm.add_parameter("Traction",
                        area_force,
                        "Traction acting on right boundary in case of Neumann BC.");
    }
    prm.leave_subsection();
  }

private:
  void
  set_parameters() final
  {
    this->param.problem_type         = ProblemType::QuasiStatic; // Steady;
    this->param.body_force           = use_volume_force;
    this->param.large_deformation    = true;
    this->param.pull_back_body_force = false;
    this->param.pull_back_traction   = false;

    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.mapping_degree          = 1;

    this->param.load_increment = 0.1;

    this->param.newton_solver_data                     = Newton::SolverData(1e3, 1.e-10, 1.e-6);
    this->param.solver                                 = Solver::CG;
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
    AssertThrow(dim == 3, dealii::ExcMessage("This application only makes sense for dim=3."));

    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> &                        tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & periodic_face_pairs,
          unsigned int const                                       global_refinements,
          std::vector<unsigned int> const &                        vector_local_refinements) {
        (void)periodic_face_pairs;
        (void)vector_local_refinements;

        dealii::GridGenerator::cylinder_shell(tria, height, inner_radius, outer_radius);

        // 0 = bottom ; 1 = top ; 2 = inner and outer radius
        for(auto cell : tria.cell_iterators())
        {
          for(auto const & f : cell->face_indices())
          {
            if(cell->face(f)->at_boundary())
            {
              dealii::Point<dim> const face_center = cell->face(f)->center();

              // bottom
              if(dim == 3 and std::abs(face_center[dim - 1] - 0.0) < 1.e-8)
              {
                cell->face(f)->set_boundary_id(0);
              }
              // top
              else if(dim == 3 and std::abs(face_center[dim - 1] - height) < 1.e-8)
              {
                cell->face(f)->set_boundary_id(1);
              }
              // inner radius and outer radius
              else
              {
                cell->face(f)->set_boundary_id(2);
              }
            }
          }
        }

        tria.refine_global(global_refinements);
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
    typedef typename std::pair<dealii::types::boundary_id, dealii::ComponentMask> pair_mask;

    // Dirichlet BC at the bottom (boundary_id = 0)
    //     std::vector<bool> mask_lower = {false, false, true}; // let boundary slide in x-y-plane
    std::vector<bool> mask_lower = {true, true, true}; // clamp boundary, i.e., fix all directions
    this->boundary_descriptor->dirichlet_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));
    this->boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));

    this->boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(0, mask_lower));

    // BC at the top (boundary_id = 1)
    bool const incremental_loading = (this->param.problem_type == ProblemType::QuasiStatic);
    if(boundary_type == BoundaryType::Dirichlet)
    {
      std::vector<bool> mask_upper = {false, false, true}; // let boundary slide in x-y-plane
      //      std::vector<bool> mask_upper = {true, true, true}; // clamp boundary, i.e., fix all
      //      directions
      this->boundary_descriptor->dirichlet_bc.insert(
        pair(1, new DisplacementDBC<dim>(displacement, incremental_loading)));
      // DisplacementDBC is a linearly increasing function, so the acceleration is zero.
      this->boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
        pair(1, new dealii::Functions::ZeroFunction<dim>(dim)));

      this->boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(1, mask_upper));
    }
    else if(boundary_type == BoundaryType::Neumann)
    {
      this->boundary_descriptor->neumann_bc.insert(
        pair(1, new AreaForce<dim>(area_force, incremental_loading)));
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("not implemented."));
    }

    // Neumann BC at the inner and outer radius (boundary_id = 2)
    this->boundary_descriptor->neumann_bc.insert(
      pair(2, new dealii::Functions::ZeroFunction<dim>(dim)));
  }

  void
  set_material_descriptor() final
  {
    typedef std::pair<dealii::types::material_id, std::shared_ptr<MaterialData>> Pair;

    MaterialType const type = MaterialType::StVenantKirchhoff;
    double const       E = 200.0e9, nu = 0.3;
    Type2D const       two_dim_type = Type2D::PlaneStress;

    this->material_descriptor->insert(
      Pair(0, new StVenantKirchhoffData<dim>(type, E, nu, two_dim_type)));
  }

  void
  set_field_functions() final
  {
    if(use_volume_force)
      this->field_functions->right_hand_side.reset(new VolumeForce<dim>(volume_force));
    else
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

    std::shared_ptr<PostProcessor<dim, Number>> post(
      new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return post;
  }

  double inner_radius = 0.8, outer_radius = 1.0, height = 1.0;

  bool use_volume_force = true;

  double volume_force = 1.0;

  enum class BoundaryType
  {
    Dirichlet,
    Neumann
  };
  BoundaryType boundary_type = BoundaryType::Dirichlet;


  double displacement = 0.2; // "Dirichlet"
  double area_force   = 1.0; // "Neumann"
};

} // namespace Structure

} // namespace ExaDG

#include <exadg/structure/user_interface/implement_get_application.h>

#endif
