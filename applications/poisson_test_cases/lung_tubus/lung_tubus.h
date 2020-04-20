/*
 * lung_tubus.h
 *
 *  Created on: May, 2019
 *      Author: fehn
 */

#include "../../../include/functionalities/one_sided_cylindrical_manifold.h"

namespace Poisson
{
namespace LungTubus
{
template<int dim>
void
set_manifolds_pipe(Triangulation<dim> & triangulation, double const radius)
{
  /*
   *  MANIFOLDS
   */
  triangulation.set_all_manifold_ids(0);

  // first fill vectors of manifold_ids and face_ids
  std::vector<unsigned int> manifold_ids;
  std::vector<unsigned int> face_ids;

  for(typename Triangulation<dim>::cell_iterator cell = triangulation.begin();
      cell != triangulation.end();
      ++cell)
  {
    for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
    {
      bool face_at_sphere_boundary = true;
      for(unsigned int v = 0; v < GeometryInfo<dim - 1>::vertices_per_cell; ++v)
      {
        Point<dim> point = Point<dim>(0, 0, cell->face(f)->vertex(v)[2]);

        if(std::abs((cell->face(f)->vertex(v) - point).norm() - radius) > 1e-12)
          face_at_sphere_boundary = false;
      }
      if(face_at_sphere_boundary)
      {
        face_ids.push_back(f);
        unsigned int manifold_id = manifold_ids.size() + 1;
        cell->set_all_manifold_ids(manifold_id);
        manifold_ids.push_back(manifold_id);
      }
    }
  }

  // generate vector of manifolds and apply manifold to all cells that have been marked
  static std::vector<std::shared_ptr<Manifold<dim>>> manifold_vec;
  manifold_vec.resize(manifold_ids.size());

  for(unsigned int i = 0; i < manifold_ids.size(); ++i)
  {
    for(typename Triangulation<dim>::cell_iterator cell = triangulation.begin();
        cell != triangulation.end();
        ++cell)
    {
      if(cell->manifold_id() == manifold_ids[i])
      {
        manifold_vec[i] = std::shared_ptr<Manifold<dim>>(static_cast<Manifold<dim> *>(
          new OneSidedCylindricalManifold<dim>(cell, face_ids[i], Point<dim>())));
        triangulation.set_manifold(manifold_ids[i], *(manifold_vec[i]));
      }
    }
  }
}

template<int dim>
void
create_grid(
  Triangulation<dim> & triangulation,
  unsigned int const   n_refine_space,
  std::vector<
    GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> & /*periodic_faces*/)
{
  // the default value of 1.e-12 is too small for this problem!
  double const merge_tolerance = 1.e-10; // test 1.e-10, 1.e-12, 1.e-14, 1.e-20

  double const r = 0.5, R = 1.5;

  Triangulation<dim> torus_1, torus_1_flatten, torus_2, torus_2_flatten, pipe, pipe_flatten, temp;
  GridGenerator::torus(torus_1, R, r, 6, numbers::PI);
  GridGenerator::torus(torus_2, R, r, 6, numbers::PI);

  // do refinement in a first step since manifolds will be gone after
  // rotate/shift/merge_triangulation
  torus_1.refine_global(n_refine_space);
  torus_2.refine_global(n_refine_space);

  // forget about the manifolds because we need to rotate later
  torus_1.reset_all_manifolds();
  torus_2.reset_all_manifolds();

  // flatten triangulation since merge_triangulation is unable to deal with refined triangulations
  GridGenerator::flatten_triangulation(torus_1, torus_1_flatten);
  GridGenerator::flatten_triangulation(torus_2, torus_2_flatten);

  // shift and rotate torus 2
  Tensor<1, dim, double> direction;
  direction[0] = 2.0 * R;
  GridTools::shift(direction, torus_2_flatten);
  GridTools::rotate(numbers::PI, 1, torus_2_flatten);

  // merge torus 1 and 2
  GridGenerator::merge_triangulations(torus_1_flatten, torus_2_flatten, temp, merge_tolerance);

  // create cylinder
  Triangulation<2> tria_2d;
  GridGenerator::hyper_ball(tria_2d, Point<2>(), r);
  GridGenerator::extrude_triangulation(tria_2d,
                                       11 /* = N_cells_axial + 1*/,
                                       4 * R /* length of pipe */,
                                       pipe);

  // manifold needs to be set before refinement
  set_manifolds_pipe(pipe, r);

  // do refinement in a first step since manifolds will be gone after
  // shift/rotate/merge_triangulation
  pipe.refine_global(n_refine_space);

  // shift and subsequently apply manifold
  Tensor<1, dim> offset = Tensor<1, dim>();
  offset[0]             = -3.0 * R;
  GridTools::shift(offset, pipe);

  // flatten triangulation since merge_triangulation is unable to deal with refined triangulations
  GridGenerator::flatten_triangulation(pipe, pipe_flatten);

  // finally merge torus with cylinder
  GridGenerator::merge_triangulations(temp, pipe_flatten, triangulation, merge_tolerance);
}

void do_create_grid(
  std::shared_ptr<parallel::TriangulationBase<2>> /*triangulation*/,
  unsigned int const /*n_refine_space*/,
  std::vector<
    GridTools::PeriodicFacePair<typename Triangulation<2>::cell_iterator>> & /*periodic_faces*/)
{
  AssertThrow(false, ExcMessage("This test case is only implemented for dim=3."));
}

template<int dim>
void
do_create_grid(
  std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
  unsigned int const                                n_refine_space,
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
    periodic_faces)
{
  if(auto tria_fully_dist =
       dynamic_cast<parallel::fullydistributed::Triangulation<dim> *>(&*triangulation))
  {
    const auto construction_data =
      TriangulationDescription::Utilities::create_description_from_triangulation_in_groups<3, 3>(
        [&](dealii::Triangulation<3, 3> & tria) mutable {
          create_grid(tria, n_refine_space, periodic_faces);
        },
        [](dealii::Triangulation<3, 3> & tria,
           const MPI_Comm                comm,
           const unsigned int /* group_size */) {
          GridTools::partition_triangulation_zorder(Utilities::MPI::n_mpi_processes(comm), tria);
        },
        tria_fully_dist->get_communicator(),
        1 /* group size */,
        true /* construct multigrid levels */);
    tria_fully_dist->create_triangulation(construction_data);
  }
  else if(auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> *>(&*triangulation))
  {
    create_grid(*tria, n_refine_space, periodic_faces);
  }
  else
  {
    AssertThrow(false, ExcMessage("Unknown triangulation!"));
  }
}

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
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
  add_parameters(ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("Application");
      prm.add_parameter("OutputDirectory",  output_directory, "Directory where output is written.");
      prm.add_parameter("OutputName",       output_name,      "Name of output files.");
    prm.leave_subsection();
    // clang-format on
  }

  std::string output_directory = "output/poisson/vtu/", output_name = "lung_tubus";

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.right_hand_side = true;

    // SPATIAL DISCRETIZATION
    param.triangulation_type     = TriangulationType::Distributed;
    param.mapping                = MappingType::Isoparametric;
    param.spatial_discretization = SpatialDiscretization::DG;
    param.IP_factor              = 1.0;

    // SOLVER
    param.solver                      = Solver::CG;
    param.solver_data                 = SolverData(1e4, 1.e-20, 1.e-8);
    param.compute_performance_metrics = true;
    param.preconditioner              = Preconditioner::Multigrid;
    param.multigrid_data.type         = MultigridType::cphMG;
    // MG smoother
    param.multigrid_data.smoother_data.smoother = MultigridSmoother::Chebyshev;
    // MG coarse grid solver
    param.multigrid_data.coarse_problem.solver         = MultigridCoarseGridSolver::CG;
    param.multigrid_data.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::AMG;
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    do_create_grid(triangulation, n_refine_space, periodic_faces);
  }

  void set_boundary_conditions(std::shared_ptr<BoundaryDescriptor<0, dim>> boundary_descriptor)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    boundary_descriptor->dirichlet_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ConstantFunction<dim>(1.0, 1));
  }

  std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    ConvDiff::PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output       = true;
    pp_data.output_data.output_folder      = output_directory;
    pp_data.output_data.output_name        = output_name;
    pp_data.output_data.degree             = degree;
    pp_data.output_data.write_higher_order = false;

    std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>> pp;
    pp.reset(new ConvDiff::PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace LungTubus
} // namespace Poisson
