#include "operator.h"
#include "../../functionalities/calculate_maximum_aspect_ratio.h"
#include "../../solvers_and_preconditioners/util/check_multigrid.h"
#include "../preconditioner/multigrid_preconditioner.h"

namespace Poisson
{
template<int dim, typename Number>
DGOperator<dim, Number>::DGOperator(
  parallel::TriangulationBase<dim> const &                  triangulation,
  Poisson::InputParameters const &                          param_in,
  std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>> postprocessor_in)
  : dealii::Subscriptor(),
    param(param_in),
    fe_dgq(param.degree),
    fe_q(param.degree),
    mapping_degree(1),
    dof_handler(triangulation),
    postprocessor(postprocessor_in)
{
  if(param.mapping == MappingType::Affine)
  {
    mapping_degree = 1;
  }
  else if(param.mapping == MappingType::Quadratic)
  {
    mapping_degree = 2;
  }
  else if(param.mapping == MappingType::Cubic)
  {
    mapping_degree = 3;
  }
  else if(param.mapping == MappingType::Isoparametric)
  {
    mapping_degree = param.degree;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented"));
  }

  mapping.reset(new MappingQGeneric<dim>(mapping_degree));
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::setup(
  PeriodicFaces const                                     periodic_face_pairs_in,
  std::shared_ptr<Poisson::BoundaryDescriptor<dim>> const boundary_descriptor_in,
  std::shared_ptr<Poisson::FieldFunctions<dim>> const     field_functions_in)
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Setup Poisson operation ..." << std::endl;

  periodic_face_pairs = periodic_face_pairs_in;
  boundary_descriptor = boundary_descriptor_in;
  field_functions     = field_functions_in;

  create_dofs();

  initialize_matrix_free();

  setup_operators();

  setup_postprocessor();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::setup_solver()
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Setup solver ..." << std::endl;

  // initialize preconditioner
  if(param.preconditioner == Poisson::Preconditioner::PointJacobi)
  {
    preconditioner.reset(
      new JacobiPreconditioner<Poisson::LaplaceOperator<dim, Number>>(laplace_operator));
  }
  else if(param.preconditioner == Poisson::Preconditioner::BlockJacobi)
  {
    preconditioner.reset(
      new BlockJacobiPreconditioner<Poisson::LaplaceOperator<dim, Number>>(laplace_operator));
  }
  else if(param.preconditioner == Poisson::Preconditioner::Multigrid)
  {
    MultigridData mg_data;
    mg_data = param.multigrid_data;

    typedef Poisson::MultigridPreconditioner<dim, Number, MultigridNumber> MULTIGRID;

    preconditioner.reset(new MULTIGRID());

    std::shared_ptr<MULTIGRID> mg_preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(preconditioner);

    parallel::TriangulationBase<dim> const * tria =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(
        &this->dof_handler.get_triangulation());
    const FiniteElement<dim> & fe = this->dof_handler.get_fe();

    mg_preconditioner->initialize(mg_data,
                                  tria,
                                  fe,
                                  *mapping,
                                  laplace_operator.get_data(),
                                  &laplace_operator.get_data().bc->dirichlet_bc,
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

  if(param.solver == Poisson::Solver::CG)
  {
    // initialize solver_data
    CGSolverData solver_data;
    solver_data.solver_tolerance_abs        = param.solver_data.abs_tol;
    solver_data.solver_tolerance_rel        = param.solver_data.rel_tol;
    solver_data.max_iter                    = param.solver_data.max_iter;
    solver_data.compute_performance_metrics = param.compute_performance_metrics;

    if(param.preconditioner != Poisson::Preconditioner::None)
      solver_data.use_preconditioner = true;

    // initialize solver
    iterative_solver.reset(
      new CGSolver<Poisson::LaplaceOperator<dim, Number>, PreconditionerBase<Number>, VectorType>(
        laplace_operator, *preconditioner, solver_data));
  }
  else if(param.solver == Solver::FGMRES)
  {
    // initialize solver_data
    FGMRESSolverData solver_data;
    solver_data.solver_tolerance_abs        = param.solver_data.abs_tol;
    solver_data.solver_tolerance_rel        = param.solver_data.rel_tol;
    solver_data.max_iter                    = param.solver_data.max_iter;
    solver_data.max_n_tmp_vectors           = param.solver_data.max_krylov_size;
    solver_data.compute_performance_metrics = param.compute_performance_metrics;

    if(param.preconditioner != Preconditioner::None)
      solver_data.use_preconditioner = true;

    // initialize solver
    iterative_solver.reset(
      new FGMRESSolver<Poisson::LaplaceOperator<dim, Number>,
                       PreconditionerBase<Number>,
                       VectorType>(laplace_operator, *preconditioner, solver_data));
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified solver is not implemented!"));
  }

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::initialize_dof_vector(VectorType & src) const
{
  matrix_free.initialize_dof_vector(src);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::prescribe_initial_conditions(VectorType & src) const
{
  field_functions->initial_solution->set_time(0.0);

  // This is necessary if Number == float
  typedef LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

  VectorTypeDouble src_double;
  src_double = src;

  VectorTools::interpolate(dof_handler, *(field_functions->initial_solution), src_double);

  src = src_double;
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::rhs(VectorType & dst, double const time) const
{
  dst = 0;

  laplace_operator.set_time(time);
  laplace_operator.rhs_add(dst);

  if(param.right_hand_side == true)
    rhs_operator.evaluate_add(dst, time);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::vmult(VectorType & dst, VectorType const & src) const
{
  laplace_operator.vmult(dst, src);
}

template<int dim, typename Number>
unsigned int
DGOperator<dim, Number>::solve(VectorType & sol, VectorType const & rhs) const
{
  // only activate if desired
  if(false)
  {
    typedef Poisson::MultigridPreconditioner<dim, Number, MultigridNumber> MULTIGRID;

    std::shared_ptr<MULTIGRID> mg_preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(preconditioner);

    CheckMultigrid<dim, Number, LaplaceOperator<dim, Number>, MULTIGRID, MultigridNumber>
      check_multigrid(laplace_operator, mg_preconditioner);

    check_multigrid.check();
  }

  unsigned int iterations = iterative_solver->solve(sol, rhs, /* update_preconditioner = */ false);

  return iterations;
}

template<int dim, typename Number>
Mapping<dim> const &
DGOperator<dim, Number>::get_mapping() const
{
  return *mapping;
}

template<int dim, typename Number>
DoFHandler<dim> const &
DGOperator<dim, Number>::get_dof_handler() const
{
  return dof_handler;
}

template<int dim, typename Number>
types::global_dof_index
DGOperator<dim, Number>::get_number_of_dofs() const
{
  return dof_handler.n_dofs();
}

template<int dim, typename Number>
double
DGOperator<dim, Number>::get_n10() const
{
  return iterative_solver->n10;
}

template<int dim, typename Number>
double
DGOperator<dim, Number>::get_average_convergence_rate() const
{
  return iterative_solver->rho;
}

template<int dim, typename Number>
double
DGOperator<dim, Number>::calculate_maximum_aspect_ratio() const
{
  return calculate_aspect_ratio_jacobian(matrix_free, dof_handler, *mapping);
}

#ifdef DEAL_II_WITH_TRILINOS
template<int dim, typename Number>
void
DGOperator<dim, Number>::init_system_matrix(TrilinosWrappers::SparseMatrix & system_matrix) const
{
  laplace_operator.init_system_matrix(system_matrix);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::calculate_system_matrix(
  TrilinosWrappers::SparseMatrix & system_matrix) const
{
  laplace_operator.calculate_system_matrix(system_matrix);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::vmult_matrix_based(VectorTypeDouble &                     dst,
                                            TrilinosWrappers::SparseMatrix const & system_matrix,
                                            VectorTypeDouble const &               src) const
{
  system_matrix.vmult(dst, src);
}
#endif

template<int dim, typename Number>
void
DGOperator<dim, Number>::create_dofs()
{
  // enumerate degrees of freedom
  if(param.spatial_discretization == SpatialDiscretization::DG)
    dof_handler.distribute_dofs(fe_dgq);
  else
    dof_handler.distribute_dofs(fe_q);
  dof_handler.distribute_mg_dofs();

  unsigned int const ndofs_per_cell = Utilities::pow(param.degree + 1, dim);

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  pcout << std::endl
        << "Discontinuous Galerkin finite element discretization:" << std::endl
        << std::endl;

  print_parameter(pcout, "degree of 1D polynomials", param.degree);
  print_parameter(pcout, "number of dofs per cell", ndofs_per_cell);
  print_parameter(pcout, "number of dofs (total)", dof_handler.n_dofs());
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::initialize_matrix_free()
{
  // quadrature formula used to perform integrals
  QGauss<1> quadrature(param.degree + 1);

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
      auto tria =
        dynamic_cast<parallel::TriangulationBase<dim> const *>(&dof_handler.get_triangulation());
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
  matrix_free.reinit(*mapping, dof_handler, constraint_matrix, quadrature, additional_data);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::setup_operators()
{
  // Laplace operator
  Poisson::LaplaceOperatorData<dim> laplace_operator_data;
  laplace_operator_data.dof_index            = 0;
  laplace_operator_data.quad_index           = 0;
  laplace_operator_data.bc                   = boundary_descriptor;
  laplace_operator_data.use_cell_based_loops = param.enable_cell_based_face_loops;

  laplace_operator_data.kernel_data.IP_factor      = param.IP_factor;
  laplace_operator_data.kernel_data.degree         = param.degree;
  laplace_operator_data.kernel_data.degree_mapping = mapping_degree;

  laplace_operator.reinit(matrix_free, constraint_matrix, laplace_operator_data);

  // rhs operator
  ConvDiff::RHSOperatorData<dim> rhs_operator_data;
  rhs_operator_data.dof_index     = 0;
  rhs_operator_data.quad_index    = 0;
  rhs_operator_data.kernel_data.f = field_functions->right_hand_side;
  rhs_operator.reinit(matrix_free, rhs_operator_data);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::setup_postprocessor()
{
  postprocessor->setup(dof_handler, *mapping);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::do_postprocessing(VectorType const & solution) const
{
  postprocessor->do_postprocessing(solution);
}

template class DGOperator<2, float>;
template class DGOperator<2, double>;

template class DGOperator<3, float>;
template class DGOperator<3, double>;

} // namespace Poisson
