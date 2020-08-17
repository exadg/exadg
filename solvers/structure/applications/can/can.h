/*
 * application.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef STRUCTURE_CAN
#define STRUCTURE_CAN

#include "structure/user_interface/application_base.h"

namespace ExaDG
{
namespace Structure
{
namespace Can
{
using namespace dealii;

template<int dim>
class DisplacementDBC : public Function<dim>
{
public:
  DisplacementDBC(double displacement, bool incremental_loading)
    : Function<dim>(dim), displacement(displacement), incremental_loading(incremental_loading)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int c) const
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
class AreaForce : public Function<dim>
{
public:
  AreaForce(double force, bool incremental_loading)
    : Function<dim>(dim), force(force), incremental_loading(incremental_loading)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int c) const
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
    prm.add_parameter("InnerRadius",      inner_radius,     "Inner radius.");
    prm.add_parameter("OuterRadius",      outer_radius,     "Outer radius.");
    prm.add_parameter("Height",           height,           "Height.");
    prm.add_parameter("UseVolumeForce",   use_volume_force, "Use volume force.");
    prm.add_parameter("VolumeForce",      volume_force,     "Volume force.");
    prm.add_parameter("BoundaryType",     boundary_type,    "Type of boundary condition, Dirichlet vs Neumann.", Patterns::Selection("Dirichlet|Neumann"));
    prm.add_parameter("Displacement",     displacement,     "Diplacement of right boundary in case of Dirichlet BC.");
    prm.add_parameter("Traction",         area_force,       "Traction acting on right boundary in case of Neumann BC.");
    prm.leave_subsection();
    // clang-format on
  }

  std::string output_directory = "output/can/vtu/", output_name = "test";

  double inner_radius = 0.8, outer_radius = 1.0, height = 1.0;

  bool use_volume_force = true;

  double volume_force = 1.0;

  std::string boundary_type = "Dirichlet";

  double displacement = 0.2; // "Dirichlet"
  double area_force   = 1.0; // "Neumann"

  void
  set_input_parameters(InputParameters & parameters)
  {
    parameters.problem_type         = ProblemType::QuasiStatic; // Steady;
    parameters.body_force           = use_volume_force;
    parameters.large_deformation    = true;
    parameters.pull_back_body_force = false;
    parameters.pull_back_traction   = false;

    parameters.triangulation_type = TriangulationType::Distributed;
    parameters.mapping            = MappingType::Affine;

    parameters.load_increment            = 0.01;
    parameters.adjust_load_increment     = false; // true;
    parameters.desired_newton_iterations = 10;

    parameters.newton_solver_data                     = Newton::SolverData(1e3, 1.e-10, 1.e-6);
    parameters.solver                                 = Solver::CG;
    parameters.solver_data                            = SolverData(1e3, 1.e-14, 1.e-6, 100);
    parameters.preconditioner                         = Preconditioner::Multigrid;
    parameters.update_preconditioner                  = true;
    parameters.update_preconditioner_every_time_steps = 1;
    parameters.update_preconditioner_every_newton_iterations = 1;
    parameters.multigrid_data.type                           = MultigridType::hpMG;
    parameters.multigrid_data.coarse_problem.solver          = MultigridCoarseGridSolver::CG;
    parameters.multigrid_data.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;

    this->param = parameters;
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    AssertThrow(dim == 3, ExcMessage("Application can only makes sense for dim=3."));

    (void)periodic_faces;

    GridGenerator::cylinder_shell(*triangulation, height, inner_radius, outer_radius);

    // 0 = bottom ; 1 = top ; 2 = inner and outer radius
    for(auto cell : triangulation->active_cell_iterators())
    {
      for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      {
        if(cell->face(f)->at_boundary())
        {
          const Point<dim> face_center = cell->face(f)->center();

          // bottom
          if(dim == 3 && std::abs(face_center[dim - 1] - 0.0) < 1.e-8)
          {
            cell->face(f)->set_boundary_id(0);
          }
          // top
          else if(dim == 3 && std::abs(face_center[dim - 1] - height) < 1.e-8)
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

    triangulation->refine_global(n_refine_space);
  }

  void
  set_boundary_conditions(std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;
    typedef typename std::pair<types::boundary_id, ComponentMask>                  pair_mask;

    // Dirichlet BC at the bottom (boundary_id = 0)
    //     std::vector<bool> mask_lower = {false, false, true}; // let boundary slide in x-y-plane
    std::vector<bool> mask_lower = {true, true, true}; // clamp boundary, i.e., fix all directions
    boundary_descriptor->dirichlet_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(0, mask_lower));

    // BC at the top (boundary_id = 1)
    bool const incremental_loading = (this->param.problem_type == ProblemType::QuasiStatic);
    if(boundary_type == "Dirichlet")
    {
      std::vector<bool> mask_upper = {false, false, true}; // let boundary slide in x-y-plane
      //      std::vector<bool> mask_upper = {true, true, true}; // clamp boundary, i.e., fix all
      //      directions
      boundary_descriptor->dirichlet_bc.insert(
        pair(1, new DisplacementDBC<dim>(displacement, incremental_loading)));
      boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(1, mask_upper));
    }
    else if(boundary_type == "Neumann")
    {
      boundary_descriptor->neumann_bc.insert(
        pair(1, new AreaForce<dim>(area_force, incremental_loading)));
    }
    else
    {
      AssertThrow(false, ExcMessage("not implemented."));
    }

    // Neumann BC at the inner and outer radius (boundary_id = 2)
    boundary_descriptor->neumann_bc.insert(pair(2, new Functions::ZeroFunction<dim>(dim)));
  }

  void
  set_material(MaterialDescriptor & material_descriptor)
  {
    typedef std::pair<types::material_id, std::shared_ptr<MaterialData>> Pair;

    MaterialType const type = MaterialType::StVenantKirchhoff;
    double const       E = 200.0e9, nu = 0.3;
    Type2D const       two_dim_type = Type2D::PlainStress;

    material_descriptor.insert(Pair(0, new StVenantKirchhoffData<dim>(type, E, nu, two_dim_type)));
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    if(use_volume_force)
      field_functions->right_hand_side.reset(new VolumeForce<dim>(volume_force));
    else
      field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));

    field_functions->initial_displacement.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->initial_velocity.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessor<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output       = true;
    pp_data.output_data.output_folder      = output_directory;
    pp_data.output_data.output_name        = output_name;
    pp_data.output_data.write_higher_order = false;
    pp_data.output_data.degree             = degree;

    std::shared_ptr<PostProcessor<dim, Number>> post(
      new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return post;
  }
};

} // namespace Can
} // namespace Structure
} // namespace ExaDG

#endif
