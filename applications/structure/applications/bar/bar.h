/*
 * application.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef STRUCTURE_BAR
#define STRUCTURE_BAR

#include "../../../include/structure/user_interface/application_base.h"

namespace ExaDG
{
namespace Structure
{
namespace Bar
{
using namespace dealii;

template<int dim>
class DisplacementDBC : public Function<dim>
{
public:
  DisplacementDBC(double const displacement, bool const unsteady, double const end_time)
    : Function<dim>(dim), displacement(displacement), unsteady(unsteady), end_time(end_time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int c) const
  {
    (void)p;

    double factor = 1.0;

    if(unsteady)
      factor = std::pow(std::sin(this->get_time() * 2.0 * numbers::PI / end_time), 2.0);

    if(c == 0)
      return displacement * factor;
    else
      return 0.0;
  }

private:
  double const displacement;
  bool const   unsteady;
  double const end_time;
};

template<int dim>
class VolumeForce : public Function<dim>
{
public:
  VolumeForce(double volume_force) : Function<dim>(dim), volume_force(volume_force)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int c) const
  {
    (void)p;

    if(c == 0)
      return volume_force; // volume force in x-direction
    else
      return 0.0;
  }

private:
  double const volume_force;
};

template<int dim>
class AreaForce : public Function<dim>
{
public:
  AreaForce(double areaforce) : Function<dim>(dim), areaforce(areaforce)
  {
  }

  const double areaforce;

  double
  value(const Point<dim> & p, const unsigned int c) const
  {
    (void)p;

    if(c == 0)
      return areaforce; // area force  in x-direction
    else
      return 0.0;
  }
};

// analytical solution (if a Neumann BC is used at the right boundary)
template<int dim>
class SolutionNBC : public Function<dim>
{
public:
  SolutionNBC(double length, double area_force, double volume_force, double E_modul)
    : Function<dim>(dim),
      A(-volume_force / 2 / E_modul),
      B(+area_force / E_modul - length * 2 * this->A)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int c) const
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
class SolutionDBC : public Function<dim>
{
public:
  SolutionDBC(double length, double displacement, double volume_force, double E_modul)
    : Function<dim>(dim),
      A(-volume_force / 2 / E_modul),
      B(+displacement / length - this->A * length)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int c) const
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
  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  void
  add_parameters(ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("Application");
    prm.add_parameter("OutputDirectory",  output_directory, "Directory where output is written.");
    prm.add_parameter("OutputName",       output_name,      "Name of output files.");
    prm.add_parameter("Length",           length,           "Length of domain.");
    prm.add_parameter("Height",           height,           "Height of domain.");
    prm.add_parameter("Width",            width,            "Width of domain.");
    prm.add_parameter("UseVolumeForce",   use_volume_force, "Use volume force.");
    prm.add_parameter("VolumeForce",      volume_force,     "Volume force.");
    prm.add_parameter("BoundaryType",     boundary_type,    "Type of boundary conditin, Dirichlet vs Neumann.", Patterns::Selection("Dirichlet|Neumann"));
    prm.add_parameter("Displacement",     displacement,     "Diplacement of right boundary in case of Dirichlet BC.");
    prm.add_parameter("Traction",         area_force,       "Traction acting on right boundary in case of Neumann BC.");
    prm.leave_subsection();
    // clang-format on
  }

  std::string output_directory = "output/bar/vtu/", output_name = "test";

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

  void
  set_input_parameters(InputParameters & parameters)
  {
    parameters.problem_type         = ProblemType::Steady;
    parameters.body_force           = use_volume_force;
    parameters.large_deformation    = true;
    parameters.pull_back_body_force = false;
    parameters.pull_back_traction   = false;

    parameters.density = density;

    parameters.start_time                           = start_time;
    parameters.end_time                             = end_time;
    parameters.time_step_size                       = end_time / 200.;
    parameters.gen_alpha_type                       = GenAlphaType::BossakAlpha;
    parameters.spectral_radius                      = 0.8;
    parameters.solver_info_data.interval_time_steps = 2;

    parameters.triangulation_type = TriangulationType::Distributed;
    parameters.mapping            = MappingType::Affine;

    parameters.load_increment            = 0.1;
    parameters.adjust_load_increment     = false;
    parameters.desired_newton_iterations = 20;

    parameters.newton_solver_data                   = Newton::SolverData(1e4, 1.e-10, 1.e-10);
    parameters.solver                               = Solver::FGMRES;
    parameters.solver_data                          = SolverData(1e4, 1.e-12, 1.e-6, 100);
    parameters.preconditioner                       = Preconditioner::Multigrid;
    parameters.multigrid_data.type                  = MultigridType::phMG;
    parameters.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::CG;
    parameters.multigrid_data.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;

    parameters.update_preconditioner                         = true;
    parameters.update_preconditioner_every_time_steps        = 1;
    parameters.update_preconditioner_every_newton_iterations = 1;

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

    for(auto cell : *triangulation)
    {
      for(unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
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
      }
    }

    triangulation->refine_global(n_refine_space);
  }

  void
  set_boundary_conditions(std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;
    typedef typename std::pair<types::boundary_id, ComponentMask>                  pair_mask;

    boundary_descriptor->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));

    // left face
    std::vector<bool> mask = {true, false};
    if(dim == 3)
    {
      mask.resize(3);
      mask[2] = true;
    }
    boundary_descriptor->dirichlet_bc.insert(pair(1, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(1, mask));

    // right face
    if(boundary_type == "Dirichlet")
    {
      bool const unsteady = (this->param.problem_type == ProblemType::Unsteady);
      boundary_descriptor->dirichlet_bc.insert(
        pair(2, new DisplacementDBC<dim>(displacement, unsteady, end_time)));
      boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(2, ComponentMask()));
    }
    else if(boundary_type == "Neumann")
    {
      boundary_descriptor->neumann_bc.insert(pair(2, new AreaForce<dim>(area_force)));
    }
    else
    {
      AssertThrow(false, ExcMessage("not implemented."));
    }
  }

  void
  set_material(MaterialDescriptor & material_descriptor)
  {
    typedef std::pair<types::material_id, std::shared_ptr<MaterialData>> Pair;

    MaterialType const type = MaterialType::StVenantKirchhoff;
    double const       E = E_modul, nu = 0.3;
    Type2D const       two_dim_type = Type2D::PlainStress;

    material_descriptor.insert(Pair(0, new StVenantKirchhoffData<dim>(type, E, nu, two_dim_type)));
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    if(use_volume_force)
      field_functions->right_hand_side.reset(new VolumeForce<dim>(this->volume_force));
    else
      field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));

    field_functions->initial_displacement.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->initial_velocity.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessor<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output         = true;
    pp_data.output_data.output_folder        = output_directory;
    pp_data.output_data.output_name          = output_name;
    pp_data.output_data.output_start_time    = start_time;
    pp_data.output_data.output_interval_time = (end_time - start_time) / 20;
    pp_data.output_data.write_higher_order   = false;
    pp_data.output_data.degree               = degree;

    pp_data.error_data.analytical_solution_available = true;
    pp_data.error_data.calculate_relative_errors     = true;
    double const vol_force                           = use_volume_force ? this->volume_force : 0.0;
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
      AssertThrow(false, ExcMessage("not implemented."));
    }

    std::shared_ptr<PostProcessor<dim, Number>> post(
      new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return post;
  }
};

} // namespace Bar
} // namespace Structure
} // namespace ExaDG


#endif
