
#include <deal.II/base/function.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/numerics/vector_tools.h>

#include "fe_navierstokes_solver.h"

#include <fstream>
#include <sstream>


using namespace dealii;


template<int dim>
class CylinderFlowProblem
{
public:
  CylinderFlowProblem();
  void
  run();

private:
  parallel::distributed::Triangulation<dim> triangulation;

  FENavierStokesSolver<dim> solver; // std::shared_ptr<FluidBaseAlgorithm<dim> > solver;
};

template<int dim>
class QuadraticVelocityProfile : public Function<dim>
{
public:
  QuadraticVelocityProfile() : Function<dim>(dim)
  {
  }

  virtual void
  vector_value(const Point<dim> & p, Vector<double> & values) const
  {
    {
      const double Um = 1.5;
      const double H  = 4.1;
      if(dim == 2)
        values(0) = 4. * Um * p[1] * (H - p[1]) / (H * H);
      else if(dim == 3)
        values(0) = 1.; // 16. * Um * p[1] * (H-p[1]) * p[2] * (H-p[2]) / (H*H*H*H);
    }
    for(unsigned int d = 1; d < dim; ++d)
      values(d) = 0;
  }
};



void create_grid(Triangulation<2> & tria)
{
  GridIn<2> grid_in;
  grid_in.attach_triangulation(tria);
  {
    std::string   filename = "cylinder_mesh.inp";
    std::ifstream file(filename.c_str());
    Assert(file, ExcFileNotOpen(filename.c_str()));
    grid_in.read_ucd(file);
  }
}



void create_grid(Triangulation<3> & tria)
{
  Triangulation<2> tria_2d;
  create_grid(tria_2d);
  GridGenerator::extrude_triangulation(tria_2d, 3, 4.1, tria);
}



template<int dim>
CylinderFlowProblem<dim>::CylinderFlowProblem()
  : triangulation(MPI_COMM_WORLD,
                  Triangulation<dim>::none,
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
    solver(triangulation, 4)
{
}



template<int dim>
void
CylinderFlowProblem<dim>::run()
{
  create_grid(triangulation);
  triangulation.refine_global(6 - dim);

  solver.set_no_slip_boundary(1);
  solver.set_no_slip_boundary(4);
  solver.set_no_slip_boundary(5);
  solver.set_no_slip_boundary(6);
  solver.set_open_boundary(3);
  solver.set_velocity_dirichlet_boundary(
    2, std::shared_ptr<Function<dim>>(new QuadraticVelocityProfile<dim>()));

  solver.setup_problem(QuadraticVelocityProfile<dim>());

  solver.set_viscosity(0.01);
  solver.set_time_step(0.0001);

  solver.output_solution("solution" + Utilities::to_string(0, 4), 3);
  solver.time_step_output_frequency = 10;
  double       tick                 = 0.1;
  unsigned int count                = 0;
  const double end_time             = 30;
  for(; solver.time < end_time;)
  {
    solver.advance_time_step();

    const int    position = int(solver.time * 1.0000000001 / tick);
    const double slot     = position * tick;
    if(((solver.time - slot) < (solver.get_time_step() * 0.99)) || solver.time >= end_time)
      solver.output_solution("solution" + Utilities::to_string(++count, 4), 3);
  }
}



int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deallog.depth_console(0);

  unsigned int dim = 2;
  if(argc > 1)
    dim = std::atoi(argv[1]);
  if(dim == 2)
  {
    CylinderFlowProblem<2> solver;
    solver.run();
  }
  else if(dim == 3)
  {
    CylinderFlowProblem<3> solver;
    solver.run();
  }
  else
    AssertThrow(false, ExcMessage("Only dimensions 2 and 3 implemented."));

  return 0;
}
