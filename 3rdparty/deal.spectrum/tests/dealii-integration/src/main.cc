
// Program for explicit time integration of the advection problem

#include <fftw3-mpi.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/point_value_history.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <fstream>
#include <iomanip>
#include <iostream>

#include "../../../src/deal-spectrum.h"
#include "energy_spectrum_calculation.h"

namespace DGAdvection
{
using namespace dealii;

const unsigned int dimension            = 2;
const unsigned int fe_degree            = 2;
const unsigned int n_global_refinements = 3;

template<int dim>
class InitialSolutionVelocity : public Function<dim>
{
public:
  InitialSolutionVelocity(const unsigned int n_components = dim, const double time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  virtual ~InitialSolutionVelocity(){};

  virtual double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    if(component == 0)
      return std::cos(1 * p[0]) * std::cos(2 * p[1]);
    else if(component == 1)
      return std::cos(4 * p[0]) * std::cos(3 * p[1]);
    else
      return 0.0;
  }
};


template<int dim>
class AdvectionProblem
{
public:
  typedef double value_type;

  AdvectionProblem()
    : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
      triangulation(MPI_COMM_WORLD),
      mapping(fe_degree),
      fe_u(new FESystem<dim>(FE_DGQ<dim>(fe_degree), dim)),
      dof_handler_u(triangulation)
  {
  }

  void
  run()
  {
    // init grid and dofs ...
    make_grid();
    setup_dofs();

    // init vector field with analytical function:
    LinearAlgebra::distributed::Vector<double> velocity_double(dof_handler_u.locally_owned_dofs(),
                                                               MPI_COMM_WORLD);
    VectorTools::interpolate(mapping,
                             dof_handler_u,
                             InitialSolutionVelocity<dim>(),
                             velocity_double);
    velocity = velocity_double;

    //                if (false){
    //                    int       cells =
    //                    Utilities::fixed_int_power<2,n_global_refinements>::value; int local_cells
    //                    = triangulation.n_locally_owned_active_cells();
    //
    //                    DealSpectrumWrapper dsw(true,false);
    //                    dsw.init(dim, cells,fe_degree+1,fe_degree+1,local_cells);
    //                    double& temp = velocity.local_element(0);
    //                    dsw.execute(&temp);
    //                } else
    {
      KineticEnergySpectrumData                                                     kesd(true, 1);
      KineticEnergySpectrumCalculator<dim, n_global_refinements, fe_degree, double> kesp;

      kesp.setup(kesd, triangulation);
      kesp.evaluate(velocity, 0.0, 0);
    }
  }

private:
  void
  make_grid()
  {
    GridGenerator::hyper_cube(triangulation, 0, 2 * numbers::PI);

    // set periodic boundary conditions in all directions
    for(typename Triangulation<dim>::cell_iterator cell = triangulation.begin();
        cell != triangulation.end();
        ++cell)
      for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        if(cell->at_boundary(f))
          cell->face(f)->set_all_boundary_ids(f);

    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
      periodic_faces;
    for(unsigned int d = 0; d < dim; ++d)
      GridTools::collect_periodic_faces(triangulation, 2 * d, 2 * d + 1, d, periodic_faces);

    triangulation.refine_global(n_global_refinements);

    pcout << "   Number of global active cells: " << triangulation.n_global_active_cells()
          << std::endl;
  }

  void
  setup_dofs()
  {
    dof_handler_u.distribute_dofs(*fe_u);
    pcout << "   Number of degrees of freedom:  " << dof_handler_u.n_dofs() << std::endl;
    previous_velocity = velocity;
  }

  LinearAlgebra::distributed::Vector<value_type> velocity, previous_velocity;
  ConditionalOStream                             pcout;
  parallel::distributed::Triangulation<dim>      triangulation;
  MappingQGeneric<dim>                           mapping;
  std::shared_ptr<FESystem<dim>>                 fe_u;
  DoFHandler<dim>                                dof_handler_u;
};

} // namespace DGAdvection


int
main(int argc, char ** argv)
{
  using namespace DGAdvection;
  using namespace dealii;

  // change mode for rounding: denormals are flushed to zero to avoid computing
  // on denormals which can slow down things.
#define MXCSR_DAZ (1 << 6)  /* Enable denormals are zero mode */
#define MXCSR_FTZ (1 << 15) /* Enable flush to zero mode */

  unsigned int mxcsr = __builtin_ia32_stmxcsr();
  mxcsr |= MXCSR_DAZ | MXCSR_FTZ;
  __builtin_ia32_ldmxcsr(mxcsr);


  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  fftw_mpi_init();

  try
  {
    deallog.depth_console(0);

    AdvectionProblem<dimension> advect_problem;
    advect_problem.run();
  }
  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;

    return 1;
  }
  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}