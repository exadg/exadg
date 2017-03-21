
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/distributed/tria.h>

#include <fe_navierstokes_solver.h>
#include <fstream>
#include <sstream>

#include "../include/incompressible_navier_stokes/postprocessor/statistics_manager.h"


using namespace dealii;



template <int dim>
class InitialChannel : public Function<dim>
{
public:
  InitialChannel (const double time = 0.)
    :
    Function<dim>(dim, time)
  {}

  virtual void vector_value (const Point<dim> &p,
                             Vector<double>   &values) const
  {
    values(0) = 22. * (1.-p[1]*p[1]*p[1]*p[1]*p[1]*p[1]);
    values(1) = 6.*(1-p[1]*p[1]*p[1]*p[1])*std::cos(p[dim-1]*3.+1./2.*p[0]);
    if (dim == 3)
      values(2) =  6.*(1-p[1]*p[1]*p[1]*p[1]*p[1]*p[1])*std::sin(p[dim-1]*3+1./2.*p[0]);
  }
};



template <int dim>
Point<dim> grid_transform (const Point<dim> &in)
{
  Point<dim> out = in;
  out[1] =  std::tanh(1.5*(2.*in(1)-1))/std::tanh(1.5);
  return out;
}



template <int dim>
void create_grid(parallel::distributed::Triangulation<dim> &triangulation,
                 const unsigned int elements_in_y_dir)
{
  Point<dim> coordinates;
  coordinates[0] = 4.*numbers::PI;
  coordinates[1] = 1.;
  if (dim == 3)
    coordinates[2] = 2.*numbers::PI;
  unsigned int base_refinements = elements_in_y_dir;
  unsigned int n_global_refinements = 0;
  while (base_refinements > 1 && base_refinements % 2 == 0)
    {
      ++n_global_refinements;
      base_refinements /= 2;
    }
  std::vector<unsigned int> refinements(dim, base_refinements);
  refinements[0] = 3 * base_refinements;
  if (dim == 3)
    refinements[2] = 2 * base_refinements;

  GridGenerator::subdivided_hyper_rectangle (triangulation, refinements,
                                             Point<dim>(), coordinates);
  std::vector<unsigned int> face_to_indicator_list(6);
  face_to_indicator_list[0] = 1;
  face_to_indicator_list[1] = 3;
  face_to_indicator_list[2] = face_to_indicator_list[3] = 0;
  face_to_indicator_list[4] = 2;
  face_to_indicator_list[5] = 4;
  for (typename Triangulation<dim>::cell_iterator cell = triangulation.begin();
       cell!= triangulation.end(); ++cell)
    for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary())
        cell->face(f)->set_all_boundary_ids(face_to_indicator_list[f]);

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> >
    periodic_faces;
  GridTools::collect_periodic_faces(triangulation, 1, 3, 0, periodic_faces);
  if (dim == 3)
    GridTools::collect_periodic_faces(triangulation, 2, 4, 2, periodic_faces);
  triangulation.add_periodicity(periodic_faces);

  triangulation.refine_global(n_global_refinements);
  GridTools::transform (&grid_transform<dim>, triangulation);
}



bool is_at_tick(const double time,
                const double time_step,
                const double tick)
{
  const int position = int(time * 1.0000000001 / tick);
  const double slot = position * tick;
  return (time - slot) < time_step*0.99999999999;
}



template <int dim>
void run_channel_flow ()
{
  const unsigned int elements_in_y_dir = 32;
  const unsigned int u_degree = 5;

  // create mesh and solver
  parallel::distributed::Triangulation<dim> triangulation
    (MPI_COMM_WORLD, Triangulation<dim>::none,
     parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);
  create_grid(triangulation, elements_in_y_dir);
  FENavierStokesSolver<dim> solver(triangulation, u_degree);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << "Running on " << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
                << " MPI ranks" << std::endl;
      std::cout << "Mesh: " << elements_in_y_dir
                << " elements in wall-normal direction, "
                << triangulation.n_global_active_cells() << " elements in total"
                << std::endl;
      std::cout << "Finite element pair: " << solver.fe_u.base_element(0).get_name()
                << "^" << dim << " x " << solver.fe_p.get_name()
                << std::endl << std::endl;
    }

  // Settings for 180 channel
  solver.set_viscosity(1./180);
  Tensor<1,dim> body_force;
  body_force[0] = 1.;
  std::string output_base = "output/cha-180-" + Utilities::to_string(elements_in_y_dir)
    + "-q" + Utilities::to_string(u_degree) + "-";
  const double end_time = 100;
  const double time_step = 0.00015;

  solver.set_body_force(body_force);
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> >
    periodic_faces;
  GridTools::collect_periodic_faces(triangulation, 1, 3, 0, periodic_faces);
  if (dim == 3)
    GridTools::collect_periodic_faces(triangulation, 2, 4, 2, periodic_faces);
  solver.set_periodic_boundaries(periodic_faces);
  solver.set_no_slip_boundary(0);

  solver.setup_problem(InitialChannel<dim>(0.));
  solver.set_time_step(time_step*0.03);

//  StatisticsManager<dim> statistics (solver.dof_handler_u, grid_transform<dim>);

  StatisticsManager<dim> statistics (solver.dof_handler_u);
      statistics.setup(grid_transform<dim>);

  solver.output_solution(output_base + Utilities::to_string(0, 4), 1);
  solver.time_step_output_frequency = 1;
  unsigned int count = 0;
  const double out_tick = 0.5;
  const double statistics_tick = end_time / 10;
  unsigned int slice = 0;
  for ( ; solver.time < end_time; )
    {
      solver.advance_time_step();

      if (is_at_tick(solver.time, solver.get_time_step(), out_tick)
          || solver.time >= end_time)
        solver.output_solution(output_base + Utilities::to_string(++count, 4), 1);

      solver.solution.update_ghost_values();
      if (solver.step_number % 10 == 0)
        statistics.evaluate(solver.solution.block(0));
      if (solver.step_number % 100 == 0 || solver.time >= end_time ||
          is_at_tick(solver.time, solver.get_time_step(), statistics_tick))
        statistics.write_output(output_base + "slice-" + Utilities::to_string(slice),
                                solver.get_viscosity());
      if (is_at_tick(solver.time, solver.get_time_step(), statistics_tick))
        {
          statistics.reset();
          slice++;
        }

      // Start with small time steps to capture the initial pressure
      // distribution more robustly and later increase them...
      if (solver.step_number == 5)
        solver.set_time_step(time_step);
    }
}



int main (int argc, char** argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi(argc,argv,1);
      run_channel_flow<3>();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
