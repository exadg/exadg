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

#include <exadg/grid/deformed_cube_manifold.h>

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
  value(dealii::Point<dim> const & p, unsigned int const c) const final
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

// TODO: only the time factor is different compared to the above function -> refactor and unify the
// code
template<int dim>
class AccelerationDBC : public dealii::Function<dim>
{
public:
  AccelerationDBC(double const displacement,
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
  value(dealii::Point<dim> const & p, unsigned int const c) const final
  {
    (void)p;

    double factor = 1.0;
    if(quasistatic)
      factor *= this->get_time();

    if(unsteady)
    {
      factor =
        2 * std::pow(2.0 * dealii::numbers::PI / end_time, 2.0) *
        (1.0 -
         2.0 * std::pow(std::sin(this->get_time() * 2.0 * dealii::numbers::PI / end_time), 2.0));
    }

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
  value(dealii::Point<dim> const & p, unsigned int const c) const final
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
  value(dealii::Point<dim> const & p, unsigned int const c) const final
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
  value(dealii::Point<dim> const & p, unsigned int const c) const final
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
  value(dealii::Point<dim> const & p, unsigned int const c) const final
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
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    prm.enter_subsection("Application");
    {
      prm.add_parameter("Length", length, "Length of domain.");
      prm.add_parameter("Height", height, "Height of domain.");
      prm.add_parameter("Width", width, "Width of domain.");
      prm.add_parameter("UseVolumeForce", use_volume_force, "Use volume force.");
      prm.add_parameter("SpatialIntegration", spatial_integration, "Use spatial integration.");
      prm.add_parameter("ForceMaterialResidual",
                        force_material_residual,
                        "Use undeformed configuration to evaluate the residual.");
      prm.add_parameter("LoadIncrement",
                        load_increment,
                        "Load increment used in QuasiStatic solver.");
      prm.add_parameter("CacheLevel", cache_level, "Cache level: 0 none, 1 scalars, 2 tensors.");
      prm.add_parameter("CheckType", check_type, "Check type for deformation gradient.");
      prm.add_parameter("MappingStrength", mapping_strength, "Strength of the mapping applied.");
      prm.add_parameter("VolumeForce", volume_force, "Volume force.");
      prm.add_parameter("BoundaryType",
                        boundary_type,
                        "Type of boundary condition, Dirichlet vs Neumann.");
      prm.add_parameter("ProblemType",
                        problem_type,
                        "Problem type considered, QuasiStatic vs Unsteady vs. Steady");
      prm.add_parameter("MaterialType",
                        material_type,
                        "StVenantKirchhoff vs. IncompressibleNeoHookean");
      prm.add_parameter("WeakDamping",
                        weak_damping_coefficient,
                        "Weak damping coefficient for unsteady problems.");
      prm.add_parameter("Displacement",
                        displacement,
                        "Displacement of right boundary in case of Dirichlet BC.");
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
    this->param.problem_type            = problem_type;
    this->param.body_force              = use_volume_force;
    this->param.large_deformation       = true;
    this->param.pull_back_body_force    = false;
    this->param.pull_back_traction      = false;
    this->param.spatial_integration     = spatial_integration;
    this->param.cache_level             = cache_level;
    this->param.check_type              = check_type;
    this->param.force_material_residual = force_material_residual;

    this->param.density = density;
    if(this->param.problem_type == ProblemType::Unsteady and weak_damping_coefficient > 0.0)
    {
      this->param.weak_damping_active      = true;
      this->param.weak_damping_coefficient = weak_damping_coefficient;
    }

    this->param.start_time      = start_time;
    this->param.end_time        = end_time;
    this->param.time_step_size  = end_time / 200.;
    this->param.gen_alpha_type  = GenAlphaType::BossakAlpha;
    this->param.spectral_radius = 0.8;
    this->param.solver_info_data.interval_time_steps =
      problem_type == ProblemType::Unsteady ? 200 : 2;

    this->param.mapping_degree =
      this->param.degree; // spatial_integration ? this->param.degree : 1;
    this->param.grid.element_type = ElementType::Hypercube; // Simplex;
    if(this->param.grid.element_type == ElementType::Simplex)
    {
      this->param.grid.triangulation_type           = TriangulationType::FullyDistributed;
      this->param.grid.create_coarse_triangulations = true;
    }
    else if(this->param.grid.element_type == ElementType::Hypercube)
    {
      this->param.grid.triangulation_type           = TriangulationType::Distributed;
      this->param.grid.create_coarse_triangulations = false; // can also be set to true if desired
    }

    this->param.load_increment = load_increment;

    this->param.newton_solver_data  = Newton::SolverData(1e2, 1.e-9, 1.e-9);
    this->param.solver              = Solver::FGMRES;
    this->param.solver_data         = SolverData(1e3, 1.e-12, 1.e-8, 100);
    this->param.preconditioner      = Preconditioner::Multigrid;
    this->param.multigrid_data.type = MultigridType::phMG;

    this->param.multigrid_data.p_sequence             = PSequenceType::DecreaseByOne; // Bisect;
    this->param.multigrid_data.smoother_data.smoother = MultigridSmoother::Chebyshev;
    this->param.multigrid_data.smoother_data.preconditioner = PreconditionerSmoother::PointJacobi;
    this->param.multigrid_data.smoother_data.iterations     = 5;

    this->param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;
    this->param.multigrid_data.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;

    this->param.update_preconditioner                  = true;
    this->param.update_preconditioner_every_time_steps = 1;
    this->param.update_preconditioner_every_newton_iterations =
      this->param.newton_solver_data.max_iter;
    this->param.update_preconditioner_once_newton_converged = true;
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

        if(this->param.grid.element_type == ElementType::Hypercube)
        {
          dealii::GridGenerator::subdivided_hyper_rectangle(tria, repetitions, p1, p2);
        }
        else if(this->param.grid.element_type == ElementType::Simplex)
        {
          dealii::Triangulation<dim, dim> tria_hypercube;
          dealii::GridGenerator::subdivided_hyper_rectangle(tria_hypercube, repetitions, p1, p2);

          dealii::GridGenerator::convert_hypercube_to_simplex_mesh(tria_hypercube, tria);
        }
        else
        {
          AssertThrow(false, dealii::ExcMessage("Not implemented."));
        }

        /*
         * illustration of 2d geometry / boundary ids:
         *
         *                  bid = 0
         *      ________________________________
         *     |                                |
         *     | bid = 1                        | bid = 2
         *     |________________________________|
         *
         *                  bid = 3
         *
         * in the 3d case: face at z = 0 has bid = 4, the other face has bid = 0 (as the top face in
         * the figure above).
         *
         */
        for(auto cell : tria)
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
            {
              if(std::fabs(cell.face(face)->center()(2) - 0.0) < 1e-8)
              {
                cell.face(face)->set_all_boundary_ids(4);
              }
            }
          }
        }

        if(vector_local_refinements.size() > 0)
          refine_local(tria, vector_local_refinements);

        if(global_refinements > 0)
          tria.refine_global(global_refinements);

        // Apply manifold map on a uniform cube
        unsigned int const frequency = 1;
        if(std::abs(this->length - this->height) < 1e-12)
          if(dim == 2 or std::abs(this->length - this->width) < 1e-12)
            apply_deformed_cube_manifold(tria, 0.0, this->length, mapping_strength, frequency);
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

    // left face
    std::vector<bool> mask_left = {true, clamp_at_left_boundary};
    if(dim == 3)
    {
      mask_left.resize(3);
      mask_left[2] = clamp_at_left_boundary;
    }
    this->boundary_descriptor->dirichlet_bc.insert(
      pair(1, new dealii::Functions::ZeroFunction<dim>(dim)));
    this->boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
      pair(1, new dealii::Functions::ZeroFunction<dim>(dim)));

    this->boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(1, mask_left));

    // right face: Dirichlet or Neumann BC

    bool quasistatic_solver = false;
    if(this->param.problem_type == ProblemType::QuasiStatic)
      quasistatic_solver = true;

    if(boundary_type == BoundaryType::Dirichlet)
    {
      std::vector<bool> mask_right = {true, clamp_at_right_boundary};
      if(dim == 3)
      {
        mask_right.resize(3);
        mask_right[2] = clamp_at_right_boundary;
      }

      bool const unsteady = (this->param.problem_type == ProblemType::Unsteady);
      this->boundary_descriptor->dirichlet_bc.insert(
        pair(2, new DisplacementDBC<dim>(displacement, quasistatic_solver, unsteady, end_time)));
      this->boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
        pair(2, new AccelerationDBC<dim>(displacement, quasistatic_solver, unsteady, end_time)));

      this->boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(2, mask_right));
    }
    else if(boundary_type == BoundaryType::Neumann)
    {
      this->boundary_descriptor->neumann_bc.insert(
        pair(2, new AreaForce<dim>(area_force, quasistatic_solver)));
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("not implemented."));
    }

    // lower/upper face (3d: front/back face)
    if(clamp_at_left_boundary or clamp_at_right_boundary)
    {
      this->boundary_descriptor->neumann_bc.insert(
        pair(3, new dealii::Functions::ZeroFunction<dim>(dim)));

      if(dim == 3)
      {
        this->boundary_descriptor->neumann_bc.insert(
          pair(4, new dealii::Functions::ZeroFunction<dim>(dim)));
      }
    }
    else // we need to apply additional boundary conditions to suppress rigid body motion
    {
      if(dim == 2)
      {
        this->boundary_descriptor->dirichlet_bc.insert(
          pair(3, new dealii::Functions::ZeroFunction<dim>(dim)));
        this->boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
          pair(3, new dealii::Functions::ZeroFunction<dim>(dim)));

        std::vector<bool> mask_y = {false, true};
        this->boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(3, mask_y));
      }
      if(dim == 3)
      {
        this->boundary_descriptor->dirichlet_bc.insert(
          pair(3, new dealii::Functions::ZeroFunction<dim>(dim)));
        this->boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
          pair(3, new dealii::Functions::ZeroFunction<dim>(dim)));

        std::vector<bool> mask_y = {false, true, false};
        this->boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(3, mask_y));

        this->boundary_descriptor->dirichlet_bc.insert(
          pair(4, new dealii::Functions::ZeroFunction<dim>(dim)));
        this->boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
          pair(4, new dealii::Functions::ZeroFunction<dim>(dim)));

        std::vector<bool> mask_z = {false, false, true};
        this->boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(4, mask_z));
      }
    }
  }

  void
  set_material_descriptor() final
  {
    typedef std::pair<dealii::types::material_id, std::shared_ptr<MaterialData>> Pair;

    if(material_type == MaterialType::StVenantKirchhoff)
    {
      Type2D const two_dim_type = Type2D::PlaneStress;
      double const nu           = 0.3;
      this->material_descriptor->insert(
        Pair(0, new StVenantKirchhoffData<dim>(material_type, E_modul, nu, two_dim_type)));
    }
    else if(material_type == MaterialType::IncompressibleNeoHookean)
    {
      Type2D const two_dim_type  = Type2D::Undefined;
      double const shear_modulus = 1.0e2;
      double const nu            = 0.49;
      double const bulk_modulus  = shear_modulus * 2.0 * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu));

      this->material_descriptor->insert(Pair(0,
                                             new IncompressibleNeoHookeanData<dim>(material_type,
                                                                                   shear_modulus,
                                                                                   bulk_modulus,
                                                                                   two_dim_type)));
    }
    else if(material_type == MaterialType::CompressibleNeoHookean)
    {
      Type2D const two_dim_type  = Type2D::Undefined;
      double const shear_modulus = 1.0e2;
      double const nu            = 0.3;
      double const lambda        = shear_modulus * 2.0 * nu / (1.0 - 2.0 * nu);

      this->material_descriptor->insert(Pair(
        0,
        new CompressibleNeoHookeanData<dim>(material_type, shear_modulus, lambda, two_dim_type)));
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage(""));
    }
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
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.degree             = this->param.degree;

    pp_data.error_data.time_control_data.start_time       = start_time;
    pp_data.error_data.time_control_data.trigger_interval = (end_time - start_time);

    pp_data.error_data.time_control_data.is_active = true;
    pp_data.error_data.calculate_relative_errors   = true;
    double const vol_force                         = use_volume_force ? this->volume_force : 0.0;
    if(boundary_type == BoundaryType::Dirichlet)
    {
      pp_data.error_data.analytical_solution.reset(
        new SolutionDBC<dim>(this->length, this->displacement, vol_force, this->E_modul));
    }
    else if(boundary_type == BoundaryType::Neumann)
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

  bool use_volume_force        = true;
  bool spatial_integration     = false;
  bool force_material_residual = false;

  unsigned int check_type  = 0;
  unsigned int cache_level = 0;

  bool const clamp_at_right_boundary = false;
  bool const clamp_at_left_boundary  = false;

  double volume_force = 1.0;

  enum class BoundaryType
  {
    Dirichlet,
    Neumann
  };
  BoundaryType boundary_type = BoundaryType::Dirichlet;
  ProblemType  problem_type  = ProblemType::Unsteady;

  double weak_damping_coefficient = 0.0;

  double displacement = 1.0; // "Dirichlet"
  double area_force   = 1.0; // "Neumann"

  double load_increment = 0.1;

  // mesh parameters
  unsigned int const repetitions0 = 1, repetitions1 = 1, repetitions2 = 1;

  MaterialType material_type = MaterialType::Undefined;
  double const E_modul       = 200.0;

  double const start_time = 0.0;
  double const end_time   = 100.0;

  double const density = 0.001;

  double mapping_strength = 0.0;
};

} // namespace Structure

} // namespace ExaDG

#include <exadg/structure/user_interface/implement_get_application.h>

#endif
