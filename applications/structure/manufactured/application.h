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

#ifndef APPLICATIONS_STRUCTURE_MANUFACTURED_APPLICATION_H_
#define APPLICATIONS_STRUCTURE_MANUFACTURED_APPLICATION_H_

/*
 * Manufactured solution for nonlinear elasticity problem with St. Venant-Kirchhoff
 * material. The test case can be used in both 2d (plane strain!) and 3d, as well as for testing
 * both steady and unsteady solvers.
 *
 * Consider the following displacement field
 *
 *     / X \   / f(t) g(X) \
 * x = | Y | + |   0       |
 *     \ Z /   \   0       /
 *
 * with function f(t) in time, and g(X) a one-dimensional displacement constant in Y, Z.
 *
 * If g(X) is a linear function, the Green-Lagrange strains and Piola-Kirchhoff stresses
 * are constant in space, so that the stress term vanishes in the balance equation. Moreover,
 * a linear function can be represented exactly by piecewise linear shape functions (or
 * shape functions of higher degree). Hence, this setup is well suited to measure temporal
 * discretization errors.
 */

namespace ExaDG
{
namespace Structure
{
/*
 * Different formulations of function f(t).
 */
enum class FunctionTypeTime
{
  Sine,
  SineSquared
};

FunctionTypeTime function_type_time = FunctionTypeTime::SineSquared;

/*
 * Different formulations of function g(X).
 */
enum class FunctionTypeSpace
{
  Linear,
  Sine,
  Exponential
};

FunctionTypeSpace function_type_space = FunctionTypeSpace::Exponential;

double
time_function(double const time, double const frequency)
{
  double time_factor = 1.0;

  if(function_type_time == FunctionTypeTime::Sine)
    time_factor = std::sin(time * frequency);
  else if(function_type_time == FunctionTypeTime::SineSquared)
    time_factor = std::pow(std::sin(time * frequency), 2.0);
  else
    AssertThrow(false, dealii::ExcMessage("not implemented"));

  return time_factor;
}

double
time_derivative(double const time, double const frequency)
{
  double time_factor = 1.0;

  if(function_type_time == FunctionTypeTime::Sine)
    time_factor = frequency * std::cos(time * frequency);
  else if(function_type_time == FunctionTypeTime::SineSquared)
    time_factor = 2.0 * std::sin(time * frequency) * frequency * std::cos(time * frequency);
  else
    AssertThrow(false, dealii::ExcMessage("not implemented"));

  return time_factor;
}

double
time_2nd_derivative(double const time, double const frequency)
{
  double time_factor = 1.0;

  if(function_type_time == FunctionTypeTime::Sine)
    time_factor = -std::pow(frequency, 2.0) * std::sin(time * frequency);
  else if(function_type_time == FunctionTypeTime::SineSquared)
    time_factor =
      2.0 * frequency * frequency *
      (std::pow(std::cos(time * frequency), 2.0) - std::pow(std::sin(time * frequency), 2.0));
  else
    AssertThrow(false, dealii::ExcMessage("not implemented"));

  return time_factor;
}

double
space_function(double const x, double const length, double const max_displacement)
{
  double space_factor = 1.0;

  if(function_type_space == FunctionTypeSpace::Linear)
    space_factor = max_displacement * x / length;
  else if(function_type_space == FunctionTypeSpace::Sine)
    space_factor = max_displacement * std::sin(x * length * 2.0 * dealii::numbers::PI);
  else if(function_type_space == FunctionTypeSpace::Exponential)
    space_factor = max_displacement * (std::exp(x / length) - 1.0) / (std::exp(1.0) - 1.0);
  else
    AssertThrow(false, dealii::ExcMessage("not implemented"));

  return space_factor;
}

double
space_derivative(double const x, double const length, double const max_displacement)
{
  double space_factor = 1.0;

  if(function_type_space == FunctionTypeSpace::Linear)
    space_factor = max_displacement / length;
  else if(function_type_space == FunctionTypeSpace::Sine)
    space_factor = max_displacement * 2.0 * dealii::numbers::PI / length *
                   std::cos(x / length * 2.0 * dealii::numbers::PI);
  else if(function_type_space == FunctionTypeSpace::Exponential)
    space_factor = max_displacement / length * (std::exp(x / length)) / (std::exp(1.0) - 1.0);
  else
    AssertThrow(false, dealii::ExcMessage("not implemented"));

  return space_factor;
}

double
space_2nd_derivative(double const x, double const length, double const max_displacement)
{
  double space_factor = 1.0;

  if(function_type_space == FunctionTypeSpace::Linear)
    space_factor = 0.0;
  else if(function_type_space == FunctionTypeSpace::Sine)
    space_factor = -max_displacement * std::pow(2.0 * dealii::numbers::PI / length, 2.0) *
                   std::sin(x / length * 2.0 * dealii::numbers::PI);
  else if(function_type_space == FunctionTypeSpace::Exponential)
    space_factor =
      max_displacement / (length * length) * (std::exp(x / length)) / (std::exp(1.0) - 1.0);
  else
    AssertThrow(false, dealii::ExcMessage("not implemented"));

  return space_factor;
}

template<int dim>
class Solution : public dealii::Function<dim>
{
public:
  Solution(double const max_displacement,
           double const length,
           bool const   unsteady,
           double const frequency)
    : dealii::Function<dim>(dim),
      max_displacement(max_displacement),
      length(length),
      unsteady(unsteady),
      frequency(frequency)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const final
  {
    if(c == 0)
    {
      double const time_factor = time_function(this->get_time(), frequency);

      return space_function(p[0], length, max_displacement) * (unsteady ? time_factor : 1.0);
    }
    else
      return 0.0;
  }

private:
  double const max_displacement, length;
  bool const   unsteady;
  double const frequency;
};

template<int dim>
class InitialVelocity : public dealii::Function<dim>
{
public:
  InitialVelocity(double const max_displacement,
                  double const length,
                  bool const   unsteady,
                  double const frequency)
    : dealii::Function<dim>(dim),
      max_displacement(max_displacement),
      length(length),
      unsteady(unsteady),
      frequency(frequency)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const final
  {
    if(c == 0)
    {
      double time_factor = 0.0;
      if(unsteady)
        time_factor = time_derivative(this->get_time(), frequency);

      return space_function(p[0], length, max_displacement) * time_factor;
    }
    else
      return 0.0;
  }

private:
  double const max_displacement, length;
  bool const   unsteady;
  double const frequency;
};

template<int dim>
class InitialAcceleration : public dealii::Function<dim>
{
public:
  InitialAcceleration(double const max_displacement,
                      double const length,
                      bool const   unsteady,
                      double const frequency)
    : dealii::Function<dim>(dim),
      max_displacement(max_displacement),
      length(length),
      unsteady(unsteady),
      frequency(frequency)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const final
  {
    if(c == 0)
    {
      double time_factor = 0.0;
      if(unsteady)
        time_factor = time_2nd_derivative(this->get_time(), frequency);

      return space_function(p[0], length, max_displacement) * time_factor;
    }
    else
      return 0.0;
  }

private:
  double const max_displacement, length;
  bool const   unsteady;
  double const frequency;
};

template<int dim>
class VolumeForce : public dealii::Function<dim>
{
public:
  VolumeForce(double const max_displacement,
              double const length,
              double const density,
              bool const   unsteady,
              double const frequency,
              double const f0)
    : dealii::Function<dim>(dim),
      max_displacement(max_displacement),
      length(length),
      density(density),
      unsteady(unsteady),
      frequency(frequency),
      f0(f0)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const final
  {
    if(c == 0)
    {
      double const time_2nd    = time_2nd_derivative(this->get_time(), frequency);
      double const time_factor = time_function(this->get_time(), frequency);

      double const acceleration_term =
        density * space_function(p[0], length, max_displacement) * (unsteady ? time_2nd : 0.0);
      double const elasticity_term =
        f0 / 2.0 *
        (3.0 * std::pow(1.0 + (unsteady ? time_factor : 1.0) *
                                space_derivative(p[0], length, max_displacement),
                        2.0) -
         1.0) *
        (unsteady ? time_factor : 1.0) * space_2nd_derivative(p[0], length, max_displacement);

      return (acceleration_term - elasticity_term);
    }
    else
      return 0.0;
  }

private:
  double const max_displacement, length;
  double const density;
  bool const   unsteady;
  double const frequency;
  double const f0;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

private:
  void
  set_parameters() final
  {
    this->param.problem_type         = unsteady ? ProblemType::Unsteady : ProblemType::Steady;
    this->param.body_force           = true;
    this->param.large_deformation    = true;
    this->param.pull_back_body_force = false;
    this->param.pull_back_traction   = false;

    this->param.density = density;

    this->param.start_time                           = start_time;
    this->param.end_time                             = end_time;
    this->param.time_step_size                       = end_time;
    this->param.gen_alpha_type                       = GenAlphaType::BossakAlpha;
    this->param.spectral_radius                      = 0.8;
    this->param.solver_info_data.interval_time_steps = 1e4;

    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.mapping_degree          = 1;

    this->param.newton_solver_data  = Newton::SolverData(1e4, 1.e-10, 1.e-10);
    this->param.solver              = Solver::CG;
    this->param.solver_data         = SolverData(1e4, 1.e-12, 1.e-6, 100);
    this->param.preconditioner      = Preconditioner::Multigrid;
    this->param.multigrid_data.type = MultigridType::phMG;

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
        (void)vector_local_refinements;

        // left-bottom-front and right-top-back point
        dealii::Point<dim> p1, p2;

        for(unsigned d = 0; d < dim; d++)
          p1[d] = 0.0;

        p2[0] = this->length;
        p2[1] = this->height;
        if(dim == 3)
          p2[2] = this->width;

        std::vector<unsigned int> repetitions(dim);
        repetitions[0] = this->n_subdivisions_1d_hypercube;
        repetitions[1] = this->n_subdivisions_1d_hypercube;
        if(dim == 3)
          repetitions[2] = this->n_subdivisions_1d_hypercube;

        dealii::GridGenerator::subdivided_hyper_rectangle(tria, repetitions, p1, p2);

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

    this->boundary_descriptor->dirichlet_bc.insert(
      pair(0, new Solution<dim>(max_displacement, length, unsteady, frequency)));
    this->boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
      pair(0, new InitialAcceleration<dim>(max_displacement, length, unsteady, frequency)));

    this->boundary_descriptor->dirichlet_bc_component_mask.insert(
      pair_mask(0, dealii::ComponentMask()));
  }

  void
  set_material_descriptor() final
  {
    typedef std::pair<dealii::types::material_id, std::shared_ptr<MaterialData>> Pair;

    MaterialType const type         = MaterialType::StVenantKirchhoff;
    Type2D const       two_dim_type = Type2D::PlaneStrain;

    this->material_descriptor->insert(
      Pair(0, new StVenantKirchhoffData<dim>(type, E, nu, two_dim_type)));
  }

  void
  set_field_functions() final
  {
    this->field_functions->right_hand_side.reset(
      new VolumeForce<dim>(max_displacement, length, density, unsteady, frequency, f0));
    this->field_functions->initial_displacement.reset(
      new dealii::Functions::ZeroFunction<dim>(dim));
    this->field_functions->initial_velocity.reset(
      new InitialVelocity<dim>(max_displacement, length, unsteady, frequency));

    if(prescribe_initial_acceleration_as_field_function)
    {
      this->field_functions->initial_acceleration.reset(
        new InitialAcceleration<dim>(max_displacement, length, unsteady, frequency));
    }
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

    pp_data.error_data.time_control_data.is_active        = true;
    pp_data.error_data.time_control_data.start_time       = start_time;
    pp_data.error_data.time_control_data.trigger_interval = (end_time - start_time);
    pp_data.error_data.calculate_relative_errors          = false;
    pp_data.error_data.analytical_solution.reset(
      new Solution<dim>(max_displacement, length, unsteady, frequency));

    std::shared_ptr<PostProcessor<dim, Number>> post(
      new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return post;
  }

  double length = 1.0, height = 1.0, width = 1.0;

  double const E  = 1.0;
  double const nu = 0.3;
  double const f0 = E * (1.0 - nu) / (1 + nu) / (1.0 - 2.0 * nu); // plane strain

  double const density = 1.0;

  bool const   unsteady         = true;
  double const max_displacement = 0.1 * length;
  double const start_time       = 0.0;
  double const end_time         = 1.0;
  double const frequency        = 3.0 / 2.0 * dealii::numbers::PI / end_time;

  bool const prescribe_initial_acceleration_as_field_function = false;
};

} // namespace Structure

} // namespace ExaDG

#include <exadg/structure/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_STRUCTURE_MANUFACTURED_APPLICATION_H_ */
