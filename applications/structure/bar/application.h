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

#ifndef STRUCTURE_BAR
#define STRUCTURE_BAR

namespace ExaDG
{
namespace Structure
{
template<int dim>
class DisplacementDBC : public dealii::Function<dim>
{
public:
  DisplacementDBC(double const displacement,
                  bool const   quasistatic_solver,
                  bool const   unsteady,
                  double const end_time)
    : dealii::Function<dim>(dim),
      displacement(displacement),
      quasistatic(quasistatic_solver),
      unsteady(unsteady),
      end_time(end_time)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const
  {
    (void)p;

    double factor = 1.0;
    if(quasistatic)
      factor *= this->get_time();

    if(unsteady)
      factor = std::pow(std::sin(this->get_time() * 2.0 * dealii::numbers::PI / end_time), 2.0);

    if(c == 0)
      return displacement * factor;
    else
      return 0.0;
  }

private:
  double const displacement;
  bool const   quasistatic;
  bool const   unsteady;
  double const end_time;
};

template<int dim>
class VolumeForce : public dealii::Function<dim>
{
public:
  VolumeForce(double volume_force, bool quasistatic_solver)
    : dealii::Function<dim>(dim), volume_force(volume_force), quasistatic(quasistatic_solver)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const
  {
    (void)p;

    double factor = 1.0;
    if(quasistatic)
      factor *= this->get_time();

    if(c == 0)
      return volume_force * factor; // volume force in x-direction
    else
      return 0.0;
  }

private:
  double const volume_force;
  bool const   quasistatic;
};

template<int dim>
class AreaForce : public dealii::Function<dim>
{
public:
  AreaForce(double areaforce, bool const quasistatic_solver)
    : dealii::Function<dim>(dim), areaforce(areaforce), quasistatic(quasistatic_solver)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const
  {
    (void)p;

    double factor = 1.0;
    if(quasistatic)
      factor *= this->get_time();

    if(c == 0)
      return areaforce * factor; // area force  in x-direction
    else
      return 0.0;
  }

private:
  double const areaforce;
  bool const   quasistatic;
};

// analytical solution (if a Neumann BC is used at the right boundary)
template<int dim>
class SolutionNBC : public dealii::Function<dim>
{
public:
  SolutionNBC(double length, double area_force, double volume_force, double E_modul)
    : dealii::Function<dim>(dim),
      A(-volume_force / 2 / E_modul),
      B(+area_force / E_modul - length * 2 * this->A)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const
  {
    if(c == 0)
      return A * p[0] * p[0] + B * p[0]; // displacement in x-direction
    else
      return 0.0;
  }

private:
  double const A;
  double const B;
};

// analytical solution (if a Dirichlet BC is used at the right boundary)
template<int dim>
class SolutionDBC : public dealii::Function<dim>
{
public:
  SolutionDBC(double length, double displacement, double volume_force, double E_modul)
    : dealii::Function<dim>(dim),
      A(-volume_force / 2 / E_modul),
      B(+displacement / length - this->A * length)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const
  {
    if(c == 0)
      return A * p[0] * p[0] + B * p[0]; // displacement in x-direction
    else
      return 0.0;
  }

private:
  double const A;
  double const B;
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
  add_parameters(dealii::ParameterHandler & prm)
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    // clang-format off
    prm.enter_subsection("Application");
    prm.add_parameter("Length",           length,           "Length of domain.");
    prm.add_parameter("Height",           height,           "Height of domain.");
    prm.add_parameter("Width",            width,            "Width of domain.");
    prm.add_parameter("UseVolumeForce",   use_volume_force, "Use volume force.");
    prm.add_parameter("VolumeForce",      volume_force,     "Volume force.");
    prm.add_parameter("BoundaryType",     boundary_type,    "Type of boundary conditin, Dirichlet vs Neumann.", dealii::Patterns::Selection("Dirichlet|Neumann"));
    prm.add_parameter("Displacement",     displacement,     "Diplacement of right boundary in case of Dirichlet BC.");
    prm.add_parameter("Traction",         area_force,       "Traction acting on right boundary in case of Neumann BC.");
    prm.leave_subsection();
    // clang-format on
  }

private:
  void
  set_parameters() final
  {
    this->param.problem_type         = ProblemType::QuasiStatic; //Steady;
    this->param.body_force           = use_volume_force;
    this->param.large_deformation    = true;
    this->param.pull_back_body_force = false;
    this->param.pull_back_traction   = false;

    this->param.density = density;

    this->param.start_time                           = start_time;
    this->param.end_time                             = end_time;
    this->param.time_step_size                       = end_time / 200.;
    this->param.gen_alpha_type                       = GenAlphaType::BossakAlpha;
    this->param.spectral_radius                      = 0.8;
    this->param.solver_info_data.interval_time_steps = 2;

    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.grid.mapping_degree     = 1;

    this->param.load_increment            = 0.025;
    this->param.adjust_load_increment     = false;
    this->param.desired_newton_iterations = 20;

    this->param.newton_solver_data                   = Newton::SolverData(1e4, 1.e-10, 1.e-10);
    this->param.solver                               = Solver::FGMRES;
    this->param.solver_data                          = SolverData(1e4, 1.e-12, 1.e-6, 100);
    this->param.preconditioner                       = Preconditioner::Multigrid;
    this->param.multigrid_data.type                  = MultigridType::phMG;
    this->param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::CG;
    this->param.multigrid_data.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;

    this->param.update_preconditioner                         = true;
    this->param.update_preconditioner_every_time_steps        = 1;
    this->param.update_preconditioner_every_newton_iterations = 1;
  }

  void
  create_grid() final
  {
    // left-bottom-front and right-top-back point
    dealii::Point<dim> p1, p2;

    for(unsigned d = 0; d < dim; d++)
      p1[d] = 0.0;

    p2[0] = this->length;
    p2[1] = this->height;
    if(dim == 3)
      p2[2] = this->width;

    std::vector<unsigned int> repetitions(dim);
    repetitions[0] = this->repetitions0;
    repetitions[1] = this->repetitions1;
    if(dim == 3)
      repetitions[2] = this->repetitions2;

    dealii::GridGenerator::subdivided_hyper_rectangle(*this->grid->triangulation,
                                                      repetitions,
                                                      p1,
                                                      p2);

    for(auto cell : *this->grid->triangulation)
    {
      for(auto const & face : cell.face_indices())
      {
        // left face
        if(std::fabs(cell.face(face)->center()(0) - 0.0) < 1e-8)
        {
          cell.face(face)->set_all_boundary_ids(1);
        }

        // right face
        if(std::fabs(cell.face(face)->center()(0) - this->length) < 1e-8)
        {
          cell.face(face)->set_all_boundary_ids(2);
        }

        // lower face
        if(std::fabs(cell.face(face)->center()(1) - 0.0) < 1e-8)
        {
          cell.face(face)->set_all_boundary_ids(3);
        }

        // back face
        if(dim == 3)
          if(std::fabs(cell.face(face)->center()(2) - 0.0) < 1e-8)
          {
            cell.face(face)->set_all_boundary_ids(4);
          }
      }
    }

    this->grid->triangulation->refine_global(this->param.grid.n_refine_global);
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
                                                                                  pair;
    typedef typename std::pair<dealii::types::boundary_id, dealii::ComponentMask> pair_mask;

    this->boundary_descriptor->neumann_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));

    // left face
    {
      std::vector<bool> mask = {true, false};
      if(dim == 3)
      {
        mask.resize(3);
        mask[2] = false;
      }
      this->boundary_descriptor->dirichlet_bc.insert(
        pair(1, new dealii::Functions::ZeroFunction<dim>(dim)));
      this->boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(1, mask));
    }

    bool quasistatic_solver = false;
    if(this->param.problem_type == ProblemType::QuasiStatic)
      quasistatic_solver = true;

    // right face
    if(boundary_type == "Dirichlet")
    {
      bool const        clamp_at_right_boundary = false;
      std::vector<bool> mask_right              = {true, clamp_at_right_boundary};
      if(dim == 3)
      {
        mask_right.resize(3);
        mask_right[2] = clamp_at_right_boundary;
      }

      bool const unsteady = (this->param.problem_type == ProblemType::Unsteady);
      this->boundary_descriptor->dirichlet_bc.insert(
        pair(2, new DisplacementDBC<dim>(displacement, quasistatic_solver, unsteady, end_time)));
      this->boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(2, mask_right));

      if(dim == 2)
      {
        if(mask_right[1] == false)
        {
          this->boundary_descriptor->dirichlet_bc.insert(
            pair(3, new dealii::Functions::ZeroFunction<dim>(dim)));
          std::vector<bool> mask_y = {false, true};
          this->boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(3, mask_y));
        }
        else
        {
          this->boundary_descriptor->neumann_bc.insert(
            pair(3, new dealii::Functions::ZeroFunction<dim>(dim)));
        }
      }

      if(dim == 3)
      {
        if(mask_right[1] == false)
        {
          this->boundary_descriptor->dirichlet_bc.insert(
            pair(3, new dealii::Functions::ZeroFunction<dim>(dim)));
          std::vector<bool> mask_y = {false, true, false};
          this->boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(3, mask_y));
        }
        else
        {
          this->boundary_descriptor->neumann_bc.insert(
            pair(3, new dealii::Functions::ZeroFunction<dim>(dim)));
        }

        if(mask_right[2] == false)
        {
          this->boundary_descriptor->dirichlet_bc.insert(
            pair(4, new dealii::Functions::ZeroFunction<dim>(dim)));
          std::vector<bool> mask_z = {false, false, true};
          this->boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(4, mask_z));
        }
        else
        {
          this->boundary_descriptor->neumann_bc.insert(
            pair(4, new dealii::Functions::ZeroFunction<dim>(dim)));
        }
      }
    }
    else if(boundary_type == "Neumann")
    {
      this->boundary_descriptor->neumann_bc.insert(
        pair(2, new AreaForce<dim>(area_force, quasistatic_solver)));

      if(dim == 2)
      {
        this->boundary_descriptor->dirichlet_bc.insert(
          pair(3, new dealii::Functions::ZeroFunction<dim>(dim)));
        std::vector<bool> mask_y = {false, true};
        this->boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(3, mask_y));
      }
      if(dim == 3)
      {
        this->boundary_descriptor->dirichlet_bc.insert(
          pair(3, new dealii::Functions::ZeroFunction<dim>(dim)));
        std::vector<bool> mask_y = {false, true, false};
        this->boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(3, mask_y));

        this->boundary_descriptor->dirichlet_bc.insert(
          pair(4, new dealii::Functions::ZeroFunction<dim>(dim)));
        std::vector<bool> mask_z = {false, false, true};
        this->boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(4, mask_z));
      }
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
    double const       E = E_modul, nu = 0.3;
    Type2D const       two_dim_type = Type2D::PlaneStress;

    this->material_descriptor->insert(
      Pair(0, new StVenantKirchhoffData<dim>(type, E, nu, two_dim_type)));
  }

  void
  set_field_functions() final
  {
    bool quasistatic_solver = false;
    if(this->param.problem_type == ProblemType::QuasiStatic)
      quasistatic_solver = true;

    if(use_volume_force)
      this->field_functions->right_hand_side.reset(
        new VolumeForce<dim>(this->volume_force, quasistatic_solver));
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
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = start_time;
    pp_data.output_data.time_control_data.trigger_interval = (end_time - start_time) / 20.0;
    pp_data.output_data.directory          = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = this->output_parameters.filename;
    pp_data.output_data.write_higher_order = false;
    pp_data.output_data.degree             = this->param.degree;

    pp_data.error_data.time_control_data.is_active = true;
    pp_data.error_data.calculate_relative_errors   = true;
    double const vol_force                         = use_volume_force ? this->volume_force : 0.0;
    if(boundary_type == "Dirichlet")
    {
      pp_data.error_data.analytical_solution.reset(
        new SolutionDBC<dim>(this->length, this->displacement, vol_force, this->E_modul));
    }
    else if(boundary_type == "Neumann")
    {
      pp_data.error_data.analytical_solution.reset(
        new SolutionNBC<dim>(this->length, this->area_force, vol_force, this->E_modul));
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("not implemented."));
    }

    std::shared_ptr<PostProcessor<dim, Number>> post(
      new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return post;
  }

  double length = 1.0, height = 1.0, width = 1.0;

  bool use_volume_force = true;

  double volume_force = 1.0;

  std::string boundary_type = "Dirichlet";

  double displacement = 1.0; // "Dirichlet"
  double area_force   = 1.0; // "Neumann"

  // mesh parameters
  unsigned int const repetitions0 = 4, repetitions1 = 1, repetitions2 = 1;

  double const E_modul = 200.0;

  double const start_time = 0.0;
  double const end_time   = 100.0;

  double const density = 0.001;
};

} // namespace Structure

} // namespace ExaDG

#include <exadg/structure/user_interface/implement_get_application.h>

#endif
