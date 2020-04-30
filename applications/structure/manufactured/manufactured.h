/*
 * manufactured.h
 *
 *  Created on: 30.04.2020
 *      Author: fehn
 */

#ifndef APPLICATIONS_STRUCTURE_MANUFACTURED_MANUFACTURED_H_
#define APPLICATIONS_STRUCTURE_MANUFACTURED_MANUFACTURED_H_

#include "../../../include/structure/user_interface/application_base.h"

/*
 * Manufactured solution for nonlinear elasticity problem with St. Venant Kirchhoff
 * material. The test case can be used in both 2d and 3d, as well as for testing both
 * steady and unsteady solvers.
 */
namespace Structure
{
namespace Manufactured
{
double
time_function(double const time, double const frequency)
{
  double time_factor = std::sin(time * frequency);

  return time_factor;
}

double
time_derivative(double const time, double const frequency)
{
  double time_factor = frequency * std::cos(time * frequency);

  return time_factor;
}

double
time_2nd_derivative(double const time, double const frequency)
{
  double time_factor = -std::pow(frequency, 2.0) * std::sin(time * frequency);

  return time_factor;
}

template<int dim>
class Solution : public Function<dim>
{
public:
  Solution(double const max_displacement,
           double const length,
           bool const   unsteady,
           double const end_time)
    : Function<dim>(dim),
      max_displacement(max_displacement),
      length(length),
      unsteady(unsteady),
      end_time(end_time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int c) const
  {
    double time_factor = 1.0;

    if(unsteady)
      time_factor = time_function(this->get_time(), 2.0 * numbers::PI / end_time);

    if(c == 0)
      return max_displacement * (p[0] / length) * time_factor;
    else
      return 0.0;
  }

private:
  double const max_displacement, length;
  bool const   unsteady;
  double const end_time;
};

template<int dim>
class InitialVelocity : public Function<dim>
{
public:
  InitialVelocity(double const max_displacement,
                  double const length,
                  bool const   unsteady,
                  double const end_time)
    : Function<dim>(dim),
      max_displacement(max_displacement),
      length(length),
      unsteady(unsteady),
      end_time(end_time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int c) const
  {
    double time_factor = 0.0;

    if(unsteady)
      time_factor = time_derivative(this->get_time(), 2.0 * numbers::PI / end_time);

    if(c == 0)
      return max_displacement * (p[0] / length) * time_factor;
    else
      return 0.0;
  }

private:
  double const max_displacement, length;
  bool const   unsteady;
  double const end_time;
};

template<int dim>
class VolumeForce : public Function<dim>
{
public:
  VolumeForce(double const max_displacement,
              double const length,
              double const density,
              bool const   unsteady,
              double const end_time)
    : Function<dim>(dim),
      max_displacement(max_displacement),
      length(length),
      density(density),
      unsteady(unsteady),
      end_time(end_time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int c) const
  {
    double time_factor = 0.0;

    if(unsteady)
      time_factor = time_2nd_derivative(this->get_time(), 2.0 * numbers::PI / end_time);

    if(c == 0)
      return +density * max_displacement * (p[0] / length) * time_factor;
    else
      return 0.0;
  }

private:
  double const max_displacement, length;
  double const density;
  bool const   unsteady;
  double const end_time;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  void
  add_parameters(ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("Application");
    prm.add_parameter("OutputDirectory",  output_directory, "Directory where output is written.");
    prm.add_parameter("OutputName",       output_name,      "Name of output files.");
    prm.leave_subsection();
    // clang-format on
  }

  std::string output_directory = "output/bar/vtu/", output_name = "test";

  double length = 1.0, height = 1.0, width = 1.0;

  // mesh parameters
  unsigned int const repetitions0 = 1, repetitions1 = 1, repetitions2 = 1;

  double const E_modul = 200.0;

  double const density = 1.0;

  bool const   unsteady         = true;
  double const max_displacement = 0.1 * length;
  double const end_time         = 1.0;

  Application() : ApplicationBase<dim, Number>("")
  {
  }

  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    parse_input(input_file, prm, true, true);
  }

  void
  set_input_parameters(InputParameters & parameters)
  {
    parameters.problem_type         = unsteady ? ProblemType::Unsteady : ProblemType::Steady;
    parameters.body_force           = true;
    parameters.large_deformation    = true;
    parameters.pull_back_body_force = false;
    parameters.pull_back_traction   = false;

    parameters.density = density;

    parameters.start_time                           = 0.0;
    parameters.end_time                             = end_time;
    parameters.time_step_size                       = end_time / 10.;
    parameters.gen_alpha_type                       = GenAlphaType::BossakAlpha;
    parameters.spectral_radius                      = 0.8;
    parameters.solver_info_data.interval_time_steps = 2;

    parameters.triangulation_type = TriangulationType::Distributed;
    parameters.mapping            = MappingType::Affine;

    parameters.newton_solver_data  = Newton::SolverData(1e4, 1.e-12, 1.e-10);
    parameters.solver              = Solver::CG;
    parameters.solver_data         = SolverData(1e4, 1.e-12, 1.e-6, 100);
    parameters.preconditioner      = Preconditioner::Multigrid;
    parameters.multigrid_data.type = MultigridType::phMG;

    parameters.update_preconditioner                         = true;
    parameters.update_preconditioner_every_time_steps        = 1;
    parameters.update_preconditioner_every_newton_iterations = 10;

    this->param = parameters;
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    (void)periodic_faces;

    // left-bottom-front and right-top-back point
    Point<dim> p1, p2;

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

    GridGenerator::subdivided_hyper_rectangle(*triangulation, repetitions, p1, p2);

    triangulation->refine_global(n_refine_space);
  }

  void
  set_boundary_conditions(std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;
    typedef typename std::pair<types::boundary_id, ComponentMask>                  pair_mask;

    boundary_descriptor->dirichlet_bc.insert(
      pair(0, new Solution<dim>(max_displacement, length, unsteady, end_time)));
    boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(0, ComponentMask()));
  }

  void
  set_material(MaterialDescriptor & material_descriptor)
  {
    typedef std::pair<types::material_id, std::shared_ptr<MaterialData>> Pair;

    MaterialType const type = MaterialType::StVenantKirchhoff;
    double const       E = E_modul, nu = 0.3;
    Type2D const       two_dim_type = Type2D::PlainStress;

    material_descriptor.insert(Pair(0, new StVenantKirchhoffData(type, E, nu, two_dim_type)));
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    field_functions->right_hand_side.reset(
      new VolumeForce<dim>(max_displacement, length, density, unsteady, end_time));
    field_functions->initial_displacement.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->initial_velocity.reset(
      new InitialVelocity<dim>(max_displacement, length, unsteady, end_time));
  }

  std::shared_ptr<PostProcessor<dim, Number>>
  construct_postprocessor(InputParameters & param, MPI_Comm const & mpi_comm)
  {
    (void)param;

    PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output         = true;
    pp_data.output_data.output_folder        = output_directory;
    pp_data.output_data.output_name          = output_name;
    pp_data.output_data.output_start_time    = param.start_time;
    pp_data.output_data.output_interval_time = (param.end_time - param.start_time) / 20;
    pp_data.output_data.write_higher_order   = false;

    pp_data.error_data.analytical_solution_available = true;
    pp_data.error_data.calculate_relative_errors     = false;
    pp_data.error_data.error_calc_start_time         = param.start_time;
    pp_data.error_data.error_calc_interval_time      = param.end_time - param.start_time;
    pp_data.error_data.analytical_solution.reset(
      new Solution<dim>(max_displacement, length, unsteady, end_time));

    std::shared_ptr<PostProcessor<dim, Number>> post(
      new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return post;
  }
};

} // namespace Manufactured

} // namespace Structure

#endif /* APPLICATIONS_STRUCTURE_MANUFACTURED_MANUFACTURED_H_ */
