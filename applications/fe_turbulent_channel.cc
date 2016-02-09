
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

#include "fe_navierstokes_solver.h"

#include <fstream>
#include <sstream>


using namespace dealii;



template <int dim>
class ChannelFlowProblem
{
public:
  static const unsigned int u_degree = 4;

  ChannelFlowProblem ();
  void run ();

private:

  parallel::distributed::Triangulation<dim>   triangulation;
  FENavierStokesSolver<dim> solver;
};



template <int dim>
class InitialChannel : public Function<dim>
{
public:
  InitialChannel (const double time = 0.,
                  const unsigned int selected_component = numbers::invalid_unsigned_int)
    :
    Function<dim>(dim, time)
  {}

  virtual void vector_value (const Point<dim> &p,
                             Vector<double>   &values) const
  {
    values(0) = (1.-p[1]*p[1]*p[1]*p[1]*p[1]*p[1])*1.15 + 0.1 * (-1. + 2.*(1.*rand()/RAND_MAX));
    values(1) = 0.1 * (-1. + 2.*(1.*rand()/RAND_MAX));
    if (dim == 3)
      values(2) =  0.1 * (-1. + 2.*(1.*rand()/RAND_MAX));
  }
};



template <int dim>
class BodyForce : public TensorFunction<1,dim>
{
  virtual Tensor<1,dim> value (const Point<dim> &p) const
  {
    // body force for 395 channel
    Tensor<1,dim> out;
    out[0] = 0.00337204;
    return out;
  }
};



template <int dim>
ChannelFlowProblem<dim>::ChannelFlowProblem () :
  triangulation (MPI_COMM_WORLD, Triangulation<dim>::none,
                 parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  solver(triangulation, u_degree)
{}



template <int dim>
Point<dim> grid_transform (const Point<dim> &in)
{
  Point<dim> out = in;
  out[1] =  std::tanh(1.5*(2.*in(1)-1))/std::tanh(1.5);
  return out;
}



template <int dim>
void create_grid(parallel::distributed::Triangulation<dim> &triangulation,
                 const unsigned int n_global_refinements)
{
  const unsigned int base_refinements = 1;
  Point<dim> coordinates;
  coordinates[0] = 2*numbers::PI;
  coordinates[1] = 1.;
  if (dim == 3)
    coordinates[2] = 2./3.*numbers::PI;
  std::vector<unsigned int> refinements(dim, base_refinements);
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
void ChannelFlowProblem<dim>::run ()
{
  create_grid(triangulation, 5);
  const double time_step = 0.0035;
  solver.set_viscosity(0.0001472);
  solver.set_body_force(std_cxx11::shared_ptr<TensorFunction<1,dim> >(new BodyForce<dim>()));

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> >
    periodic_faces;
  GridTools::collect_periodic_faces(triangulation, 1, 3, 0, periodic_faces);
  if (dim == 3)
    GridTools::collect_periodic_faces(triangulation, 2, 4, 2, periodic_faces);
  solver.set_periodic_boundaries(periodic_faces);
  solver.set_no_slip_boundary(0);

  solver.setup_problem(InitialChannel<dim>(0.));

  solver.set_time_step(time_step*0.03);

  solver.output_solution("solution" + Utilities::to_string(0, 4));
  solver.time_step_output_frequency = 1;
  const double end_time = 1000;
  unsigned int count = 0;
  const double tick = 1;
  for ( ; solver.time < end_time; )
    {
      solver.advance_time_step();

      const int position = int(solver.time * 1.0000000001 / tick);
      const double slot = position * tick;
      if (((solver.time - slot) < (solver.get_time_step()*0.99)) || solver.time >= end_time)
        solver.output_solution("solution" + Utilities::to_string(++count, 4));

      // Start with small time steps to capture the initial pressure
      // distribution more robustly and later increase them...
      if (solver.step_number == 19)
        solver.set_time_step(time_step * 0.3);
      if (solver.step_number == 169)
        solver.set_time_step(time_step);
    }
}



int main (int argc, char** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc,argv,1);
  deallog.depth_console(0);

  ChannelFlowProblem<3> solver;
  solver.run ();

  return 0;
}
