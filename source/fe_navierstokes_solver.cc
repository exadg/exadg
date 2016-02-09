
#include "fe_navierstokes_solver.h"
#include "fe_navierstokes_evaluator.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/parallel.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/distributed/tria.h>

#include <fstream>
#include <sstream>


using namespace dealii;


template <int dim>
FENavierStokesSolver<dim>::FENavierStokesSolver(const parallel::distributed::Triangulation<dim> &triangulation,
                                                const unsigned int velocity_degree)
  :
  FluidBaseAlgorithm<dim> (velocity_degree),
  fe_u (FE_Q<dim>(QGaussLobatto<1>(velocity_degree+1)), dim),
  dof_handler_u (triangulation),
  fe_p (QGaussLobatto<1>(velocity_degree)),
  dof_handler_p (triangulation),
  time (0),
  step_number (0),
  time_step_output_frequency (1),
  communicator (triangulation.get_communicator()),
  pcout (std::cout,
         Utilities::MPI::this_mpi_process(communicator) == 0)
{}



namespace
{
  template <int dim>
  void compute_diagonal_mass_inverse(const Mapping<dim> &mapping,
                                     const DoFHandler<dim> &dof_handler,
                                     const ConstraintMatrix &constraints,
                                     parallel::distributed::Vector<double> &vector)
  {
    QGaussLobatto<dim> quad(dof_handler.get_fe().degree+1);
    FEValues<dim> fe_values(mapping, dof_handler.get_fe(), quad,
                            update_values | update_JxW_values);

    Vector<double> values(dof_handler.get_fe().dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(values.size());

    for (typename DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          for (unsigned int i=0; i<values.size(); ++i)
            {
              double sum = 0;
              const double* shape_vals_i = &fe_values.shape_value(i,0);
              for (unsigned int q=0; q<quad.size(); ++q)
                sum += (shape_vals_i[q] * shape_vals_i[q]) * fe_values.JxW(q);
              values(i) = sum;
            }
          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(values, local_dof_indices, vector);
        }
    vector.compress(VectorOperation::add);
    for (unsigned int i=0; i<vector.local_size(); ++i)
      if (std::abs(vector.local_element(i)) > 1e-20)
        vector.local_element(i) = 1./vector.local_element(i);
      else
        vector.local_element(i) = 1.;
  }

}



template <int dim>
void FENavierStokesSolver<dim>::setup_problem
(const Function<dim> &initial_velocity_field)
{
  Timer timer;

  double shortest_edge = std::numeric_limits<double>::max();
  for (typename Triangulation<dim>::active_cell_iterator it =
         dof_handler_u.get_triangulation().begin_active();
       it != dof_handler_u.get_triangulation().end(); ++it)
    shortest_edge = std::min(shortest_edge,
                             it->minimum_vertex_distance());
  shortest_edge = Utilities::MPI::min(shortest_edge, communicator);

  const double cfl = 0.08/fe_u.degree;
  this->time_step_size = std::min(shortest_edge * cfl,
                                  shortest_edge * shortest_edge * cfl * 0.5 / this->viscosity);

  pcout << "Shortest edge: " << shortest_edge << ", time step estimate: "
        << this->time_step_size << std::endl;

  timer.restart();

  dof_handler_u.distribute_dofs (fe_u);
  dof_handler_p.distribute_dofs (fe_p);
  dof_handler_p.distribute_mg_dofs (fe_p);
  pcout << "Time distribute dofs: " << timer.wall_time() << std::endl;
  timer.restart();

  pcout << "Number of degrees of freedom: "
        << dof_handler_u.n_dofs() + dof_handler_p.n_dofs()
        << " (" << dim << "*" << dof_handler_u.n_dofs()/dim << " + "
        << dof_handler_p.n_dofs()  << ")"
        << std::endl;

  PoissonSolverData<dim> poisson_data;

  // no-slip boundaries directly filled into velocity system
  constraints_u.clear();
  constraints_p.clear();

  IndexSet relevant_dofs_u, relevant_dofs_p;
  DoFTools::extract_locally_relevant_dofs(dof_handler_u, relevant_dofs_u);
  constraints_u.reinit(relevant_dofs_u);
  DoFTools::extract_locally_relevant_dofs(dof_handler_p, relevant_dofs_p);
  constraints_p.reinit(relevant_dofs_p);

  DoFTools::make_hanging_node_constraints(dof_handler_u, constraints_u);
  DoFTools::make_hanging_node_constraints(dof_handler_p, constraints_p);

  std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator> > periodic_faces_u, periodic_faces_p;
  for (typename std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> >::iterator
         it = this->boundary->periodic_face_pairs_level0.begin();
       it != this->boundary->periodic_face_pairs_level0.end(); ++it)
    {
      const types::boundary_id in = it->cell[0]->face(it->face_idx[0])->boundary_id();
      const types::boundary_id out = it->cell[1]->face(it->face_idx[1])->boundary_id();
      AssertThrow (this->boundary->open_conditions_p.find(in) ==
                   this->boundary->open_conditions_p.end() &&
                   this->boundary->open_conditions_p.find(out) ==
                   this->boundary->open_conditions_p.end() &&
                   this->boundary->dirichlet_conditions_u.find(in) ==
                   this->boundary->dirichlet_conditions_u.end() &&
                   this->boundary->dirichlet_conditions_u.find(out) ==
                   this->boundary->dirichlet_conditions_u.end() &&
                   this->boundary->no_slip.find(in) ==
                   this->boundary->no_slip.end() &&
                   this->boundary->no_slip.find(out) ==
                   this->boundary->no_slip.end() &&
                   this->boundary->symmetry.find(in) ==
                   this->boundary->symmetry.end() &&
                   this->boundary->symmetry.find(out) ==
                   this->boundary->symmetry.end(),
                   ExcMessage("Cannot mix periodic boundary conditions with "
                              "other types of boundary conditions on same "
                              "boundary!"));

      GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator> periodic_u;
      GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator> periodic_p;
      for (unsigned int i=0; i<2; ++i)
        {
          periodic_u.cell[i] = typename DoFHandler<dim>::cell_iterator
            (&dof_handler_u.get_triangulation(),
             it->cell[i]->level(), it->cell[i]->index(), &dof_handler_u);
          periodic_u.face_idx[i] = it->face_idx[i];

          periodic_p.cell[i] = typename DoFHandler<dim>::cell_iterator
            (&dof_handler_p.get_triangulation(),
             it->cell[i]->level(), it->cell[i]->index(), &dof_handler_p);
          periodic_p.face_idx[i] = it->face_idx[i];
        }
      periodic_u.orientation = it->orientation;
      periodic_u.matrix = it->matrix;
      periodic_faces_u.push_back(periodic_u);
      periodic_p.orientation = it->orientation;
      periodic_p.matrix = it->matrix;
      periodic_faces_p.push_back(periodic_p);
    }
  DoFTools::make_periodicity_constraints<DoFHandler<dim> > (periodic_faces_u,
                                                            constraints_u);
  DoFTools::make_periodicity_constraints<DoFHandler<dim> > (periodic_faces_p,
                                                            constraints_p);
  poisson_data.periodic_face_pairs_level0 = this->boundary->periodic_face_pairs_level0;

  for (typename std::set<types::boundary_id>::const_iterator it =
         this->boundary->symmetry.begin();
       it != this->boundary->symmetry.end(); ++it)
    {
      AssertThrow (this->boundary->open_conditions_p.find(*it) == this->boundary->open_conditions_p.end() &&
                   this->boundary->no_slip.find(*it) == this->boundary->no_slip.end() &&
                   this->boundary->dirichlet_conditions_u.find(*it) == this->boundary->dirichlet_conditions_u.end(),
                   ExcMessage("Cannot mix symmetry boundary conditions with "
                              "other boundary conditions on same boundary!"));
      poisson_data.neumann_boundaries.insert(*it);
    }

  VectorTools::compute_no_normal_flux_constraints (dof_handler_u, 0,
                                                   this->boundary->symmetry,
                                                   constraints_u);
  VectorTools::compute_normal_flux_constraints (dof_handler_u, 0,
                                                this->boundary->normal_flux,
                                                constraints_u);

  {
    ZeroFunction<dim> zero_func(dim);
    typename FunctionMap<dim>::type homogeneous_dirichlet;
    for (typename std::map<types::boundary_id,
         std_cxx11::shared_ptr<Function<dim> > >::
         const_iterator it = this->boundary->dirichlet_conditions_u.begin();
         it != this->boundary->dirichlet_conditions_u.end(); ++it)
      {
        AssertThrow (this->boundary->open_conditions_p.find(it->first) ==
                     this->boundary->open_conditions_p.end(),
                     ExcMessage("Cannot mix velocity Dirichlet conditions with "
                                "open/pressure boundary conditions on same "
                                "boundary!"));
        poisson_data.neumann_boundaries.insert(it->first);

        // we don't add this to the list of Dirichlet boundaries on the
        // velocity because we will manually set the appropriate boundary
        // values before evaluation
      }

    // no-slip boundaries
    for (typename std::set<types::boundary_id>::const_iterator it =
           this->boundary->no_slip.begin();
         it != this->boundary->no_slip.end(); ++it)
      {
        AssertThrow (this->boundary->open_conditions_p.find(*it) ==
                     this->boundary->open_conditions_p.end(),
                     ExcMessage("Cannot mix velocity Dirichlet conditions with "
                                "open/pressure boundary conditions on same "
                                "boundary!"));
        homogeneous_dirichlet[*it] = &zero_func;
        poisson_data.neumann_boundaries.insert(*it);
      }

    VectorTools::interpolate_boundary_values(this->mapping, dof_handler_u,
                                             homogeneous_dirichlet,
                                             constraints_u);
  }

  constraints_u.close ();
  AssertThrow (constraints_u.has_inhomogeneities() == false,
               ExcMessage("Constraint matrix for u has inhomogeneities which "
                          "is not allowed."));

  {
    ZeroFunction<dim> zero_func(1);
    typename FunctionMap<dim>::type homogeneous_dirichlet;
    // open boundaries with prescribed pressure values
    for (typename std::map<types::boundary_id,
         std_cxx11::shared_ptr<Function<dim> > >::
         const_iterator it = this->boundary->open_conditions_p.begin();
         it != this->boundary->open_conditions_p.end(); ++it)
      {
        homogeneous_dirichlet[it->first] = &zero_func;
        poisson_data.dirichlet_boundaries.insert(it->first);
      }

    VectorTools::interpolate_boundary_values(this->mapping, dof_handler_p,
                                             homogeneous_dirichlet,
                                             constraints_p);
  }

  // constrain the zeroth pressure degree of freedom in case we have a pure
  // Neumann problem in pressure
  if (LaplaceOperator<dim,double>::verify_boundary_conditions(dof_handler_p, poisson_data) &&
      constraints_p.can_store_line(0))
    {
      // if dof 0 is constrained, it must be a periodic dof, so we take the
      // value on the other side
      types::global_dof_index line_index = 0;
      while (true)
        {
          const std::vector<std::pair<types::global_dof_index,double> >* lines =
            constraints_p.get_constraint_entries(line_index);
          if (lines == 0)
            {
              constraints_p.add_line(line_index);
              break;
            }
          else
            {
              Assert(lines->size() == 1 && std::abs((*lines)[0].second-1.)<1e-15,
                     ExcMessage("Periodic index expected, bailing out"));
              line_index = (*lines)[0].first;
            }
        }
    }

  constraints_p.close();

  // setup matrix-free object (integrator for all FE-related stuff)
  {
    std::vector<const DoFHandler<dim>*> dofs;
    dofs.push_back(&dof_handler_u);
    dofs.push_back(&dof_handler_p);
    std::vector<const ConstraintMatrix *> constraints;
    constraints.push_back (&constraints_u);
    constraints.push_back (&constraints_p);
    std::vector<Quadrature<1> > quadratures;
    // Choose enough points to avoid aliasing effects
    quadratures.push_back(QGauss<1>(fe_u.degree+1+fe_u.degree/2));
    quadratures.push_back(QGauss<1>(fe_u.degree));
    typename MatrixFree<dim>::AdditionalData data (communicator);
    data.tasks_parallel_scheme = MatrixFree<dim>::AdditionalData::none;
    data.tasks_block_size = 16;
    data.mapping_update_flags |= update_quadrature_points;
    matrix_free.reinit (this->mapping, dofs, constraints, quadratures, data);
  }

  poisson_data.poisson_dof_index = 1;
  poisson_data.poisson_quad_index = 1;
  poisson_data.smoother_smoothing_range = 25;
  poisson_solver.initialize(this->mapping, matrix_free, poisson_data);

  // compute diagonal vectors of velocity/pressure mass matrix needed for time
  // stepping
  matrix_free.initialize_dof_vector(velocity_diagonal_mass, 0);
  matrix_free.initialize_dof_vector(pressure_diagonal_mass, 1);

  compute_diagonal_mass_inverse(this->mapping, dof_handler_p, constraints_p,
                                pressure_diagonal_mass);
  compute_diagonal_mass_inverse(this->mapping, dof_handler_u, constraints_u,
                                velocity_diagonal_mass);
  {
    // for evaluating inhomogeneous boundary conditions, we choose to always
    // set the correct velocity values before evaluating the velocity
    // function. In order not to create spurious values during time
    // integration, we simply zero those entries when multiplying by the
    // inverse diagonal matrix, which is implemented here
    ZeroFunction<dim> zero(dim);
    typename FunctionMap<dim>::type inhom_dirichlet;
    for (typename std::map<types::boundary_id,
           std_cxx11::shared_ptr<Function<dim> > >::iterator it =
         this->boundary->dirichlet_conditions_u.begin();
       it != this->boundary->dirichlet_conditions_u.end(); ++it)
    {
      inhom_dirichlet[it->first] = &zero;
    }

    std::map<types::global_dof_index,double> boundary_values;
    VectorTools::interpolate_boundary_values (this->mapping, dof_handler_u,
                                              inhom_dirichlet,
                                              boundary_values);

    const Utilities::MPI::Partitioner &vel_part = *velocity_diagonal_mass.get_partitioner();
    for (std::map<types::global_dof_index,double>::const_iterator
           it = boundary_values.begin(); it != boundary_values.end(); ++it)
      if (vel_part.in_local_range(it->first))
        velocity_diagonal_mass.local_element(it->first-vel_part.local_range().first) = 0;
  }

  solution.reinit(2);
  matrix_free.initialize_dof_vector(solution.block(0), 0);
  matrix_free.initialize_dof_vector(solution.block(1), 1);
  solution.collect_sizes();
  updates1 = solution;
  updates2 = solution;

  VectorTools::interpolate(this->mapping, dof_handler_u, initial_velocity_field,
                           solution.block(0));
  constraints_u.distribute(solution.block(0));

  pcout << "Time vectors + integrator: " << timer.wall_time() << std::endl;

  computing_times.resize(4);
}



template <int dim>
void
FENavierStokesSolver<dim>::apply_inhomogeneous_velocity_boundary_conditions
(const parallel::distributed::Vector<double> &in_vec,
 const double current_time) const
{
  if (this->boundary->dirichlet_conditions_u.empty())
    return;

  typename FunctionMap<dim>::type dirichlet_u;

  // set the correct time in the function
  for (typename std::map<types::boundary_id,
         std_cxx11::shared_ptr<Function<dim> > >::iterator it =
         this->boundary->dirichlet_conditions_u.begin();
       it != this->boundary->dirichlet_conditions_u.end(); ++it)
    {
      it->second->set_time(current_time);
      dirichlet_u[it->first] = it->second.get();
    }

  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values (this->mapping, dof_handler_u,
                                            dirichlet_u,
                                            boundary_values);

  parallel::distributed::Vector<double> &src_vel =
    const_cast<parallel::distributed::Vector<double> &>(in_vec);
  const Utilities::MPI::Partitioner &vel_part = *src_vel.get_partitioner();
  for (std::map<types::global_dof_index,double>::const_iterator
         it = boundary_values.begin(); it != boundary_values.end(); ++it)
    if (vel_part.in_local_range(it->first))
      src_vel.local_element(it->first-vel_part.local_range().first) = it->second;
}


template <int dim>
void
FENavierStokesSolver<dim>
::apply_velocity_operator(const double                                 current_time,
                          const parallel::distributed::Vector<double> &src,
                          parallel::distributed::Vector<double>       &dst) const
{
  // apply inhomogeneous velocity boundary values (no-slip is applied via
  // MatrixFree)
  apply_inhomogeneous_velocity_boundary_conditions(src, current_time);

  if (this->body_force.get())
    this->body_force->set_time(current_time);
  FENavierStokesEvaluator<dim> evaluator(matrix_free, solution.block(1),
                                         updates1.block(1), *this);
  evaluator.advection_integrals(src, dst);
}



namespace
{
  template <int dim>
  struct RKVectorUpdater
  {
    RKVectorUpdater (const parallel::distributed::Vector<double> &matrix_diagonal_inverse,
                     const double  factor1,
                     const double  factor2,
                     const bool    is_last,
                     parallel::distributed::Vector<double> &vector1,
                     parallel::distributed::Vector<double> &vector2,
                     parallel::distributed::Vector<double> &vector3)
      :
      matrix_diagonal_inverse (matrix_diagonal_inverse),
      factor1 (factor1),
      factor2 (factor2),
      is_last (is_last),
      vector1 (vector1),
      vector2 (vector2),
      vector3 (vector3)
    {
      AssertDimension(vector1.size(), matrix_diagonal_inverse.size());
      AssertDimension(vector2.size(), matrix_diagonal_inverse.size());
      AssertDimension(vector3.size(), matrix_diagonal_inverse.size());
    }

    void
    apply_to_subrange (const std::size_t begin,
                       const std::size_t end) const
    {
      const double factor1 = this->factor1;
      const double factor2 = this->factor2;
      double *vector1 = this->vector1.begin(),
        *vector2 = this->vector2.begin(),
        *vector3 = this->vector3.begin();
      const double* matrix_diagonal_inverse = this->matrix_diagonal_inverse.begin();
      if (is_last)
        {
          DEAL_II_OPENMP_SIMD_PRAGMA
            for (std::size_t i=begin; i<end; ++i)
              {
                const double update = vector1[i] * matrix_diagonal_inverse[i];
                vector2[i] += factor1 * update;
                vector1[i] = 0;
              }
        }
      else
        {
          DEAL_II_OPENMP_SIMD_PRAGMA
            for (std::size_t i=begin; i<end; ++i)
              {
                const double update = vector1[i] * matrix_diagonal_inverse[i];
                vector2[i] += factor1 * update;
                vector3[i] = vector2[i] + factor2 * update;
                vector1[i] = 0;
              }
        }
    }

    const parallel::distributed::Vector<double> &matrix_diagonal_inverse;
    const double factor1;
    const double factor2;
    const bool   is_last;
    parallel::distributed::Vector<double> &vector1;
    parallel::distributed::Vector<double> &vector2;
    parallel::distributed::Vector<double> &vector3;
  };

  template<int dim>
  struct RKVectorUpdatesRange : public parallel::ParallelForInteger
  {
    RKVectorUpdatesRange(const parallel::distributed::Vector<double> &matrix_diagonal_inverse,
                         const double  factor1,
                         const double  factor2,
                         const bool    is_last,
                         parallel::distributed::Vector<double> &vector1,
                         parallel::distributed::Vector<double> &vector2,
                         parallel::distributed::Vector<double> &vector3)
      :
      updater (matrix_diagonal_inverse, factor1, factor2, is_last,
               vector1, vector2, vector3)
    {
      const std::size_t size = vector1.local_size();
      if (size < internal::Vector::minimum_parallel_grain_size)
        apply_to_subrange (0, size);
      else
        apply_parallel (0, size,
                        internal::Vector::minimum_parallel_grain_size);
    }

    ~RKVectorUpdatesRange() {}

    virtual void
    apply_to_subrange (const std::size_t begin,
                       const std::size_t end) const
    {
      updater.apply_to_subrange(begin, end);
    }

    const RKVectorUpdater<dim> updater;
  };
}



template <int dim>
unsigned int
FENavierStokesSolver<dim>::advance_time_step()
{
  const bool output_info = step_number % time_step_output_frequency == (time_step_output_frequency-1);

  if (output_info)
    pcout << "Step " << std::setw(5) << step_number+1 << " to t = "
          << std::setw(8) << time+this->time_step_size;

  Timer time;

  // step 1: advect velocity by 3-stage Runge-Kutta

  const double a21 = 0.755726351946097;
  const double a32 = 0.386954477304099;
  const double b1 = 0.245170287303492;
  const double b2 = 0.184896052186740;
  const double b3 = 0.569933660509768;

  apply_velocity_operator(this->time, solution.block(0), updates1.block(0));
  RKVectorUpdatesRange<dim>(velocity_diagonal_mass, -a21*this->time_step_size,
                            (a21-b1)*this->time_step_size, false, updates1.block(0),
                            solution.block(0), updates2.block(0));
  apply_velocity_operator(this->time+a21*this->time_step_size, solution.block(0), updates1.block(0));
  RKVectorUpdatesRange<dim>(velocity_diagonal_mass, -a32*this->time_step_size,
                            (a32-b2)*this->time_step_size, false, updates1.block(0),
                            updates2.block(0), solution.block(0));
  apply_velocity_operator(this->time+(b1+a32)*this->time_step_size, updates2.block(0), updates1.block(0));
  RKVectorUpdatesRange<dim>(velocity_diagonal_mass, -b3*this->time_step_size,
                            0,                  true,  updates1.block(0),
                            solution.block(0), updates2.block(0));

  this->time += this->time_step_size;

  computing_times[0] += time.wall_time();
  time.restart();

  // step 2: compute divergence for pressure Poisson equation and solve
  updates2.block(1) = 0;
  FENavierStokesEvaluator<dim> evaluator(matrix_free, solution.block(1),
                                         updates1.block(1), *this);
  evaluator.divergence_integrals(solution.block(0), updates2.block(1));

  // get pressure boundary values
  typename FunctionMap<dim>::type dirichlet_p;

  // set the correct time in the function
  for (typename std::map<types::boundary_id,
         std_cxx11::shared_ptr<Function<dim> > >::iterator it =
         this->boundary->open_conditions_p.begin();
       it != this->boundary->open_conditions_p.end(); ++it)
    {
      it->second->set_time(this->time);
      dirichlet_p[it->first] = it->second.get();
    }

  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values (this->mapping, dof_handler_p,
                                            dirichlet_p,
                                            boundary_values);

  const Utilities::MPI::Partitioner &pres_part = *updates2.block(1).get_partitioner();
  for (std::map<types::global_dof_index,double>::const_iterator
         it = boundary_values.begin(); it != boundary_values.end(); ++it)
    if (pres_part.in_local_range(it->first))
      updates2.block(1).local_element(it->first-pres_part.local_range().first) = 0;

  computing_times[1] += time.wall_time();
  time.restart();

  // solve the pressure Poisson equation
  updates1.block(1) = 0;
  const unsigned int n_iter = poisson_solver.solve(updates1.block(1), updates2.block(1));
  if (output_info)
    {
      std::cout.precision(3);
      pcout << ", div norm: " << std::setw(8) << updates2.block(1).l2_norm()
            << ", cg its: " << n_iter << std::endl;
    }

  computing_times[2] += time.wall_time();
  time.restart();

  // step 3: update pressure; for pressure, use the solution from Poisson
  // equation and a rotational part (to prescribe consistent boundary values
  // for the pressure)
  updates2.block(1).scale(pressure_diagonal_mass);
  solution.block(1).add(-this->viscosity, updates2.block(1),
                        1./this->time_step_size, updates1.block(1));

  // apply pressure boundary values
  for (std::map<types::global_dof_index,double>::const_iterator
         it = boundary_values.begin(); it != boundary_values.end(); ++it)
    if (pres_part.in_local_range(it->first))
      solution.block(1).local_element(it->first-pres_part.local_range().first) = it->second;

  solution.block(1).update_ghost_values();
  updates1.block(1).update_ghost_values();

  computing_times[3] += time.wall_time();

  ++step_number;
  return n_iter;
}



template <int dim>
void
FENavierStokesSolver<dim>::compute_vorticity() const
{
  updates1.block(0) = 0;
  FENavierStokesEvaluator<dim> evaluator(matrix_free, solution.block(1),
                                         updates1.block(1), *this);
  evaluator.curl_integrals(solution.block(0), updates1.block(0));

  updates1.block(0).scale(velocity_diagonal_mass);
}



template <int dim>
void
FENavierStokesSolver<dim>::output_solution (const std::string  filename_base,
                                            const unsigned int n_patches) const
{
  compute_vorticity();

  constraints_u.distribute(const_cast<parallel::distributed::Vector<double> &>(solution.block(0)));
  constraints_p.distribute(const_cast<parallel::distributed::Vector<double> &>(solution.block(1)));

  solution.update_ghost_values();
  updates1.update_ghost_values();
  updates2.update_ghost_values();

  std::vector<std::string> velocity_name (dim, "velocity");
  std::vector<std::string> vorticity_name (dim, "vorticity");
  DataOut<dim> data_out;
  data_out.attach_triangulation (dof_handler_u.get_triangulation());
  std::vector< DataComponentInterpretation::DataComponentInterpretation >
    component_interpretation (dim,
                              DataComponentInterpretation::component_is_part_of_vector);
  data_out.add_data_vector (dof_handler_u, solution.block(0),
                            velocity_name, component_interpretation);
  data_out.add_data_vector (dof_handler_p, solution.block(1), "pressure");
  data_out.add_data_vector (dof_handler_p, updates2.block(1), "velocity_div");
  data_out.add_data_vector (dof_handler_u, updates1.block(0),
                            vorticity_name, component_interpretation);
  data_out.build_patches (n_patches);

  std::ostringstream filename;
  filename << filename_base
           << "_Proc"
           << Utilities::MPI::this_mpi_process(communicator)
           << ".vtu";

  std::ofstream output (filename.str().c_str());
  data_out.write_vtu (output);

  if ( Utilities::MPI::this_mpi_process(communicator) == 0)
    {

      std::vector<std::string> filenames;
      for (unsigned int i=0;i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);++i)
        {
          std::ostringstream filename;
          filename << filename_base
                   << "_Proc"
                   << i
                   << ".vtu";

          filenames.push_back(filename.str().c_str());
        }
      std::string master_name = filename_base +  ".pvtu";
      std::ofstream master_output (master_name.c_str());
      data_out.write_pvtu_record (master_output, filenames);
    }

  updates1.block(0) = 0;

}



template <int dim>
void
FENavierStokesSolver<dim>::print_computing_times() const
{
  std::string names [4] = {"Advection","Velocity div","Pressure","Other   "};
  pcout << std::endl << "Computing times:    \t [min/avg/max] \t\t [p_min/p_max]" << std::endl;
  double total_avg_time = 0;
  for (unsigned int i=0; i<computing_times.size(); ++i)
    {
      Utilities::MPI::MinMaxAvg data =
        Utilities::MPI::min_max_avg (computing_times[i], communicator);
      pcout << "Step " << i+1 <<  ": " << names[i] << "\t " << data.min << "/" << data.avg << "/" << data.max << " \t " << data.min_index << "/" << data.max_index << std::endl;
      total_avg_time += data.avg;
    }
  pcout  <<"Time (Step 1-" << computing_times.size() << "):\t "<<total_avg_time<<std::endl;
}



template class FENavierStokesSolver<2>;
template class FENavierStokesSolver<3>;
