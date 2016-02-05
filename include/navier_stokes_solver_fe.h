
#ifndef __indexa_navierstokes_solver_h_
#define __indexa_navierstokes_solver_h_

#include <deal.II/base/timer.h>
#include <deal.II/base/function.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/distributed/tria.h>


using namespace dealii;

#include "poisson_solver.h"
#include "fluid_base_algorithm.h"


template <int dim>
class NavierStokesSolverFE : public FluidBaseAlgorithm<dim>
{
public:
  NavierStokesSolverFE (const parallel::distributed::Triangulation<dim> &triangulation);

  ~NavierStokesSolverFE()
  {
    print_computing_times();
  }

  /**
   * Setup of problem. Initializes the degrees of freedom and solver-related
   * variables (vectors, matrices, etc.) and interpolates the initial velocity
   * field to the velocity variable. Must be called after setting the boundary
   * conditions.
   */
  virtual void setup_problem (const Function<dim> &initial_velocity_field);

  /**
   * Performs one complete time step of the problem. Returns the number of
   * linear iterations in the pressure Poisson solver.
   */
  virtual unsigned int advance_time_step ();

  /**
   * Generic output interface. Allows to write the complete solution field to
   * a vtu file. Derived classes decide which variables need to be written and
   * how often this is about to happen.
   *
   * The optional argument @p n_subdivisions lets the user override the
   * default value (0, taking the velocity degree) the sub-refinement used for
   * representing higher order solutions.
   */
  virtual void output_results (const std::string  filename_base,
                               const unsigned int n_subdivisions = 0) const;

  void compute_vorticity() const;

  void print_computing_times() const;

  FESystem<dim>        fe_u;
  DoFHandler<dim>      dof_handler_u;
  FE_Q<dim>            fe_p;
  DoFHandler<dim>      dof_handler_p;

  parallel::distributed::BlockVector<double> solution;
  mutable parallel::distributed::Vector<double> updates_u1, updates_u2, updates_p1, updates_p2;

  ConstraintMatrix     constraints_u;
  ConstraintMatrix     constraints_p;

  double        time;
  double        time_step;
  double        viscosity;

private:
  template <int fe_degree>
  void local_advect (const MatrixFree<dim>              &data,
                     std::vector<parallel::distributed::Vector<double> > &dst,
                     const std::vector<parallel::distributed::Vector<double> > &src,
                     const std::pair<unsigned int,unsigned int> &cell_range) const;

  template <int fe_degree>
  void local_divergence(const MatrixFree<dim>              &data,
                        parallel::distributed::Vector<double> &dst,
                        const std::vector<parallel::distributed::Vector<double> > &src,
                        const std::pair<unsigned int,unsigned int> &cell_range) const;

  void apply_velocity_operator(const double current_time,
                               const std::vector<parallel::distributed::Vector<double> > &src,
                               std::vector<parallel::distributed::Vector<double> > &dst) const;

  MatrixFree<dim>          matrix_free;
  PoissonSolver<dim>       poisson_solver;

  parallel::distributed::Vector<double> velocity_diagonal_mass;
  parallel::distributed::Vector<double> pressure_diagonal_mass;

  std::vector<std::vector<std::pair<types::global_dof_index, double> > > velocity_boundary;

  ConditionalOStream   pcout;

  std::vector<double> computing_times;
};


#endif  // ifndef __indexa_fluid_base_algorithm_h
