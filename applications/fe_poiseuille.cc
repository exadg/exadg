
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
class ExactSolutionU : public Function<dim>
{
public:
  ExactSolutionU (const double viscosity = 1.,
                  const double time = 0.)
    :
    Function<dim>(dim, time),
    nu(viscosity)
  {}

  virtual void vector_value (const Point<dim> &p,
                             Vector<double>   &values) const
  {
    AssertDimension (values.size(), dim);

    values(0) = 0.5/nu * (1-p[1])*(1+p[1]);
    for (unsigned int d=1; d<dim; ++d)
      values(d) = 0;
  }

private:
  const double nu;
};



template <int dim>
class ExactSolutionP : public Function<dim>
{
public:
  ExactSolutionP ()
    :
    Function<dim>(1, 0)
  {}

  virtual double value (const Point<dim> &p,
                        const unsigned int) const
  {
    // choose zero pressure at outlet because we have not implemented the
    // other variants
    return (2-p[0]);
  }
};



template <int dim>
class ChannelFlowProblem
{
public:
  ChannelFlowProblem ();
  void run ();

private:
  parallel::distributed::Triangulation<dim>   triangulation;
};



template <int dim>
ChannelFlowProblem<dim>::ChannelFlowProblem () :
  triangulation (MPI_COMM_WORLD, Triangulation<dim>::none,
                 parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy)
{}



bool is_at_tick(const double time,
                const double time_step,
                const double tick)
{
  const int position = int(time * 1.0000000001 / tick);
  const double slot = position * tick;
  return (time - slot) < time_step*0.99999999999;
}



template <int dim>
void compute_error(FENavierStokesSolver<dim> &solver)
{
  // Compute the error to the analytic solution of the Poiseuille flow

  Vector<float> cellwise_errors (solver.dof_handler_u.get_triangulation().n_active_cells());
  const unsigned int u_degree = solver.fe_u.degree;

  // use high order quadrature to avoid underestimation of errors because of
  // superconvergence effects
  QGauss<dim>  quadrature(u_degree+2);

  VectorTools::integrate_difference (solver.dof_handler_p,
                                     solver.solution.block(1),
                                     ExactSolutionP<dim> (),
                                     cellwise_errors, quadrature,
                                     VectorTools::L2_norm);
  const double p_l2_error = std::sqrt(Utilities::MPI::sum(cellwise_errors.norm_sqr(),
                                                          MPI_COMM_WORLD));

  VectorTools::integrate_difference (solver.dof_handler_u,
                                     solver.solution.block(0),
                                     ExactSolutionU<dim> (solver.get_viscosity(),
                                                          solver.time),
                                     cellwise_errors, quadrature,
                                     VectorTools::L2_norm);
  const double u_l2_error = std::sqrt(Utilities::MPI::sum(cellwise_errors.norm_sqr(),
                                                          MPI_COMM_WORLD));

  std::cout.precision(4);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "  L2-Errors: ||e_p||_L2 = " << p_l2_error
              << ",   ||e_u||_L2 = " << u_l2_error
              << std::endl;

}



template <int dim>
void ChannelFlowProblem<dim>::run ()
{
  // create mesh and solver
  {
    std::vector<unsigned int> subdivisions (dim, 1);
    subdivisions[0] = 4;

    const Point<dim> bottom_left = (dim == 2 ?
                                    Point<dim>(-2,-1) :
                                    Point<dim>(-2,-1,-1));
    const Point<dim> top_right   = (dim == 2 ?
                                    Point<dim>(2,0) :
                                    Point<dim>(2,0,0));

    GridGenerator::subdivided_hyper_rectangle (triangulation,
                                               subdivisions,
                                               bottom_left,
                                               top_right);

    // no need to check for owned cells here: on level 0 everything is locally
    // owned
    for (typename Triangulation<dim>::active_cell_iterator it=triangulation.begin();
         it != triangulation.end(); ++it)
      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        if (it->face(face)->at_boundary() &&
            std::abs(it->face(face)->center()[0]-2)<1e-13)
          it->face(face)->set_boundary_id(1);
        else if (it->face(face)->at_boundary() &&
                 std::abs(it->face(face)->center()[0]+2)<1e-13)
          it->face(face)->set_boundary_id(2);
        else if (it->face(face)->at_boundary() &&
                 std::abs(it->face(face)->center()[1])<1e-13)
          it->face(face)->set_boundary_id(3);
  }
  triangulation.refine_global(3);
  const unsigned int u_degree = 4;

  FENavierStokesSolver<dim> solver(triangulation, u_degree);

  solver.set_viscosity(0.1);
  std::string output_base = "output/poiseuille-q" + Utilities::to_string(u_degree) + "-";
  const double end_time = 1;

  solver.set_no_slip_boundary(0);
  solver.set_symmetry_boundary(3);
  solver.set_open_boundary_with_normal_flux(1, std::shared_ptr<Function<dim> > (new ExactSolutionP<dim>()));
  solver.set_velocity_dirichlet_boundary(2, std::shared_ptr<Function<dim> > (new ExactSolutionU<dim>(0.1)));
  //solver.set_open_boundary_with_normal_flux(2, std::shared_ptr<Function<dim> > (new ExactSolutionP<dim>()));

  solver.setup_problem(ExactSolutionU<dim>(0.1,0));
  solver.set_time_step(0.0005);

  solver.output_solution(output_base + Utilities::to_string(0, 4), 1);
  solver.time_step_output_frequency = 10;
  unsigned int count = 0;
  const double out_tick = 0.1;
  for ( ; solver.time < end_time; )
    {
      solver.advance_time_step();

      if (is_at_tick(solver.time, solver.get_time_step(), out_tick)
          || solver.time >= end_time)
        {
          solver.output_solution(output_base + Utilities::to_string(++count, 4), 1);
          compute_error(solver);
        }
    }
}



int main (int argc, char** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc,argv,1);
  deallog.depth_console(0);

  ChannelFlowProblem<2> solver;
  solver.run ();

  return 0;
}
