#include "poisson_operation.h"


#include "../preconditioner/multigrid_preconditioner.h"

namespace Poisson
{
template<int dim, int degree, typename Number>
DGOperation<dim, degree, Number>::DGOperation(
  parallel::distributed::Triangulation<dim> const & triangulation,
  Poisson::InputParameters const &                  param_in)
  : dealii::Subscriptor(),
    fe_dgq(degree),
    fe_q(degree),
    mapping(param_in.degree_mapping),
    dof_handler(triangulation),
    param(param_in)
{
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::setup(
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> const
                                                    periodic_face_pairs_in,
  std::shared_ptr<Poisson::BoundaryDescriptor<dim>> boundary_descriptor_in,
  std::shared_ptr<Poisson::FieldFunctions<dim>>     field_functions_in)
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Setup Poisson operation ..." << std::endl;

  periodic_face_pairs = periodic_face_pairs_in;
  boundary_descriptor = boundary_descriptor_in;
  field_functions     = field_functions_in;

  create_dofs();

  initialize_matrix_free();

  setup_operators();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::setup_solver()
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Setup solver ..." << std::endl;

  // initialize preconditioner
  if(param.preconditioner == Poisson::Preconditioner::PointJacobi)
  {
    preconditioner.reset(
      new JacobiPreconditioner<Poisson::LaplaceOperator<dim, degree, Number>>(laplace_operator));
  }
  else if(param.preconditioner == Poisson::Preconditioner::BlockJacobi)
  {
    preconditioner.reset(
      new BlockJacobiPreconditioner<Poisson::LaplaceOperator<dim, degree, Number>>(
        laplace_operator));
  }
  else if(param.preconditioner == Poisson::Preconditioner::Multigrid)
  {
    MultigridData mg_data;
    mg_data = param.multigrid_data;

    typedef float MultigridNumber;

    typedef PreconditionableOperator<dim, MultigridNumber>                         MG_BASE;
    typedef Poisson::LaplaceOperator<dim, degree, MultigridNumber>                 MG_OPERATOR;
    typedef Poisson::MultigridPreconditioner<dim, degree, Number, MultigridNumber> MULTIGRID;

    preconditioner.reset(new MULTIGRID());

    std::shared_ptr<MULTIGRID> mg_preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(preconditioner);
    mg_preconditioner->initialize(mg_data,
                                  dof_handler,
                                  mapping,
                                  laplace_operator.get_operator_data(),
                                  &laplace_operator.get_operator_data().bc->dirichlet_bc,
                                  &this->periodic_face_pairs);
  }
  else
  {
    AssertThrow(param.preconditioner == Poisson::Preconditioner::None ||
                  param.preconditioner == Poisson::Preconditioner::PointJacobi ||
                  param.preconditioner == Poisson::Preconditioner::BlockJacobi ||
                  param.preconditioner == Poisson::Preconditioner::Multigrid,
                ExcMessage("Specified preconditioner is not implemented!"));
  }

  if(param.solver == Poisson::Solver::PCG)
  {
    // initialize solver_data
    CGSolverData solver_data;
    solver_data.solver_tolerance_abs        = param.abs_tol;
    solver_data.solver_tolerance_rel        = param.rel_tol;
    solver_data.max_iter                    = param.max_iter;
    solver_data.compute_performance_metrics = param.compute_performance_metrics;

    if(param.preconditioner != Poisson::Preconditioner::None)
      solver_data.use_preconditioner = true;

    // initialize solver
    iterative_solver.reset(
      new CGSolver<Poisson::LaplaceOperator<dim, degree, Number>,
                   PreconditionerBase<Number>,
                   VectorType>(laplace_operator, *preconditioner, solver_data));
  }
  else
  {
    AssertThrow(param.solver == Poisson::Solver::PCG,
                ExcMessage("Specified solver is not implemented!"));
  }

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::initialize_dof_vector(VectorType & src) const
{
  data.initialize_dof_vector(src);
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::rhs(VectorType & dst, double const evaluation_time) const
{
  dst = 0;
  if(param.spatial_discretization == SpatialDiscretization::DG)
    laplace_operator.rhs_add(dst, evaluation_time);
  if(param.right_hand_side == true)
    rhs_operator.evaluate_add(dst, evaluation_time);
}

template<int dim, int degree, typename Number>
unsigned int
DGOperation<dim, degree, Number>::solve(VectorType & sol, VectorType const & rhs)
{
  unsigned int iterations = iterative_solver->solve(sol, rhs);

  return iterations;
}

template<int dim, int degree, typename Number>
MatrixFree<dim, Number> const &
DGOperation<dim, degree, Number>::get_data() const
{
  return data;
}

template<int dim, int degree, typename Number>
Mapping<dim> const &
DGOperation<dim, degree, Number>::get_mapping() const
{
  return mapping;
}

template<int dim, int degree, typename Number>
DoFHandler<dim> const &
DGOperation<dim, degree, Number>::get_dof_handler() const
{
  return dof_handler;
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::create_dofs()
{
  // enumerate degrees of freedom
  if(param.spatial_discretization == SpatialDiscretization::DG)
    dof_handler.distribute_dofs(fe_dgq);
  else
    dof_handler.distribute_dofs(fe_q);
  dof_handler.distribute_mg_dofs();

  unsigned int ndofs_per_cell = Utilities::pow(degree + 1, dim);

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  pcout << std::endl
        << "Discontinuous Galerkin finite element discretization:" << std::endl
        << std::endl;

  print_parameter(pcout, "degree of 1D polynomials", degree);
  print_parameter(pcout, "number of dofs per cell", ndofs_per_cell);
  print_parameter(pcout, "number of dofs (total)", dof_handler.n_dofs());
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::initialize_matrix_free()
{
  // quadrature formula used to perform integrals
  QGauss<1> quadrature(degree + 1);

  // initialize matrix_free_data
  typename MatrixFree<dim, Number>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
    MatrixFree<dim, Number>::AdditionalData::partition_partition;

  additional_data.mapping_update_flags =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
     update_values);
  if(param.spatial_discretization == SpatialDiscretization::DG)
  {
    additional_data.mapping_update_flags_inner_faces =
      (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);

    additional_data.mapping_update_flags_boundary_faces =
      (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);

    if(param.enable_cell_based_face_loops)
    {
      auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> const *>(
        &dof_handler.get_triangulation());
      Categorization::do_cell_based_loops(*tria, additional_data);
    }
  }
  else
  {
    MGConstrainedDoFs            mg_constrained_dofs;
    std::set<types::boundary_id> dirichlet_boundary;
    for(auto it : boundary_descriptor->dirichlet_bc)
      dirichlet_boundary.insert(it.first);
    mg_constrained_dofs.initialize(dof_handler);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary);
    constraint_matrix.add_lines(mg_constrained_dofs.get_boundary_indices(
      dof_handler.get_triangulation().n_global_levels() - 1));
  }

  constraint_matrix.close();
  data.reinit(mapping, dof_handler, constraint_matrix, quadrature, additional_data);
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::setup_operators()
{
  // laplace operator
  Poisson::LaplaceOperatorData<dim> laplace_operator_data;
  laplace_operator_data.dof_index            = 0;
  laplace_operator_data.quad_index           = 0;
  laplace_operator_data.IP_factor            = param.IP_factor;
  laplace_operator_data.bc                   = boundary_descriptor;
  laplace_operator_data.use_cell_based_loops = param.enable_cell_based_face_loops;
  laplace_operator_data.mapping.reset(new MappingQGeneric<dim>(param.degree_mapping));
  laplace_operator.reinit(data, constraint_matrix, laplace_operator_data);

  // rhs operator
  ConvDiff::RHSOperatorData<dim> rhs_operator_data;
  rhs_operator_data.dof_index  = 0;
  rhs_operator_data.quad_index = 0;
  rhs_operator_data.rhs        = field_functions->right_hand_side;
  rhs_operator.reinit(data, rhs_operator_data);
}
} // namespace Poisson
