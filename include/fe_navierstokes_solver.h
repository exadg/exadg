
#ifndef __indexa_fe_navierstokes_solver_h_
#define __indexa_fe_navierstokes_solver_h_

#include <deal.II/base/timer.h>
#include <deal.II/base/function.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/parallel_block_vector.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/distributed/tria.h>


using namespace dealii;

#include "poisson_solver.h"
#include "fluid_base_algorithm.h"


template <int dim>
class FENavierStokesSolver : public FluidBaseAlgorithm<dim>
{
public:
  FENavierStokesSolver (const parallel::distributed::Triangulation<dim> &triangulation,
                        const unsigned int velocity_degree);

  virtual ~FENavierStokesSolver()
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
  virtual void output_solution (const std::string  filename_base,
                                const unsigned int n_subdivisions = 0) const;

  void compute_vorticity() const;

  void print_computing_times() const;

  FESystem<dim>        fe_u;
  DoFHandler<dim>      dof_handler_u;
  FE_Q<dim>            fe_p;
  DoFHandler<dim>      dof_handler_p;

  parallel::distributed::BlockVector<double> solution;
  mutable parallel::distributed::BlockVector<double> updates1, updates2;

  ConstraintMatrix     constraints_u;
  ConstraintMatrix     constraints_p;
  ConstraintMatrix     constraints_p_solve;

  double        time;
  unsigned int  step_number;
  unsigned int  time_step_output_frequency;

private:
  void apply_inhomogeneous_velocity_boundary_conditions
  (const parallel::distributed::Vector<double> &in_vec,
   const double current_time) const;

  void apply_velocity_operator(const double current_time,
                               const parallel::distributed::Vector<double> &src,
                               parallel::distributed::Vector<double> &dst) const;

  MatrixFree<dim>          matrix_free;
  PoissonSolver<dim>       poisson_solver;

  parallel::distributed::Vector<double> velocity_diagonal_mass;
  parallel::distributed::Vector<double> pressure_diagonal_mass;

  std::vector<std::vector<std::pair<types::global_dof_index, double> > > velocity_boundary;

  MPI_Comm            communicator;

  ConditionalOStream  pcout;

  Timer               global_timer;
  std::vector<double> computing_times;
};


#endif  // ifndef __indexa_fe_navierstokes_solver_h
