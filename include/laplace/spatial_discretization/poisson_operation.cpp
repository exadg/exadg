#include "poisson_operation.h"


#include "../../solvers_and_preconditioners/multigrid/multigrid_preconditioner_dg.h"

namespace Poisson
{
template<int dim, int fe_degree, typename value_type>
DGOperation<dim, fe_degree, value_type>::DGOperation(
  parallel::distributed::Triangulation<dim> const & triangulation,
  Poisson::InputParameters const &                  param_in)
  : fe(fe_degree), mapping(fe_degree), dof_handler(triangulation), param(param_in)
{
}

template<int dim, int fe_degree, typename value_type>
void
DGOperation<dim, fe_degree, value_type>::setup(
  const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
                                                    periodic_face_pairs,
  std::shared_ptr<Poisson::BoundaryDescriptor<dim>> boundary_descriptor_in,
  std::shared_ptr<Poisson::FieldFunctions<dim>>     field_functions_in)
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Setup Poisson operation ..." << std::endl;

  this->periodic_face_pairs = periodic_face_pairs;
  boundary_descriptor       = boundary_descriptor_in;
  field_functions           = field_functions_in;

  create_dofs();

  initialize_matrix_free();

  setup_operators();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int fe_degree, typename value_type>
void
DGOperation<dim, fe_degree, value_type>::setup_solver()
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Setup solver ..." << std::endl;

  // initialize preconditioner
  if(param.preconditioner == Poisson::Preconditioner::PointJacobi)
  {
    preconditioner.reset(
      new JacobiPreconditioner<Poisson::LaplaceOperator<dim, fe_degree, value_type>>(laplace_operator));
  }
  else if(param.preconditioner == Poisson::Preconditioner::BlockJacobi)
  {
    preconditioner.reset(
      new BlockJacobiPreconditioner<Poisson::LaplaceOperator<dim, fe_degree, value_type>>(laplace_operator));
  }
  else if(param.preconditioner == Poisson::Preconditioner::Multigrid)
  {
    MultigridData mg_data;
    mg_data = param.multigrid_data;

    typedef float Number;

    typedef MyMultigridPreconditionerDG<dim,
                                        value_type,
                                        Poisson::LaplaceOperator<dim, fe_degree, Number>>
      MULTIGRID;

    preconditioner.reset(new MULTIGRID());
    std::shared_ptr<MULTIGRID> mg_preconditioner = std::dynamic_pointer_cast<MULTIGRID>(preconditioner);
    mg_preconditioner->initialize(mg_data,
                                  dof_handler,
                                  mapping,
                                  laplace_operator.get_operator_data().bc->dirichlet_bc,
                                  (void *)&laplace_operator.get_operator_data());
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
    solver_data.solver_tolerance_abs = param.abs_tol;
    solver_data.solver_tolerance_rel = param.rel_tol;
    solver_data.max_iter             = param.max_iter;

    if(param.preconditioner != Poisson::Preconditioner::None)
      solver_data.use_preconditioner = true;

    // initialize solver
    iterative_solver.reset(new CGSolver<Poisson::LaplaceOperator<dim, fe_degree, value_type>,
                                        PreconditionerBase<value_type>,
                                        VectorType>(laplace_operator, *preconditioner, solver_data));
  }
  else
  {
    AssertThrow(param.solver == Poisson::Solver::PCG,
                ExcMessage("Specified solver is not implemented!"));
  }

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int fe_degree, typename value_type>
void
DGOperation<dim, fe_degree, value_type>::initialize_dof_vector(VectorType & src) const
{
  data.initialize_dof_vector(src);
}

template<int dim, int fe_degree, typename value_type>
void
DGOperation<dim, fe_degree, value_type>::rhs(VectorType & dst, double const evaluation_time) const
{
  dst = 0;
  laplace_operator.rhs_add(dst, evaluation_time);
  if(param.right_hand_side == true)
    rhs_operator.evaluate_add(dst, evaluation_time);
}

template<int dim, int fe_degree, typename value_type>
unsigned int
DGOperation<dim, fe_degree, value_type>::solve(VectorType & sol, VectorType const & rhs)
{
  unsigned int iterations = iterative_solver->solve(sol, rhs);

  return iterations;
}

template<int dim, int fe_degree, typename value_type>
MatrixFree<dim, value_type> const &
DGOperation<dim, fe_degree, value_type>::get_data() const
{
  return data;
}

template<int dim, int fe_degree, typename value_type>
Mapping<dim> const &
DGOperation<dim, fe_degree, value_type>::get_mapping() const
{
  return mapping;
}

template<int dim, int fe_degree, typename value_type>
DoFHandler<dim> const &
DGOperation<dim, fe_degree, value_type>::get_dof_handler() const
{
  return dof_handler;
}

template<int dim, int fe_degree, typename value_type>
void
DGOperation<dim, fe_degree, value_type>::create_dofs()
{
  // enumerate degrees of freedom
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();

  unsigned int ndofs_per_cell = Utilities::pow(fe_degree + 1, dim);

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  pcout << std::endl << "Discontinuous Galerkin finite element discretization:" << std::endl << std::endl;

  print_parameter(pcout, "degree of 1D polynomials", fe_degree);
  print_parameter(pcout, "number of dofs per cell", ndofs_per_cell);
  print_parameter(pcout, "number of dofs (total)", dof_handler.n_dofs());
}

template<int dim, int fe_degree, typename value_type>
void
DGOperation<dim, fe_degree, value_type>::initialize_matrix_free()
{
  // quadrature formula used to perform integrals
  QGauss<1> quadrature(fe_degree + 1);

  // initialize matrix_free_data
  typename MatrixFree<dim, value_type>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme = MatrixFree<dim, value_type>::AdditionalData::partition_partition;
  additional_data.build_face_info       = true;
  additional_data.mapping_update_flags =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors | update_values);

  additional_data.mapping_update_flags_inner_faces =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors | update_values);

  additional_data.mapping_update_flags_boundary_faces =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors | update_values);

  ConstraintMatrix dummy;
  dummy.close();
  data.reinit(mapping, dof_handler, dummy, quadrature, additional_data);
}

template<int dim, int fe_degree, typename value_type>
void
DGOperation<dim, fe_degree, value_type>::setup_operators()
{
  // laplace operator
  Poisson::LaplaceOperatorData<dim> laplace_operator_data;
  laplace_operator_data.dof_index                  = 0;
  laplace_operator_data.quad_index                 = 0;
  laplace_operator_data.IP_factor                  = param.IP_factor;
  laplace_operator_data.bc                         = boundary_descriptor;
  laplace_operator_data.periodic_face_pairs_level0 = periodic_face_pairs;
  laplace_operator.initialize(mapping, data, laplace_operator_data);

  // rhs operator
  ConvDiff::RHSOperatorData<dim> rhs_operator_data;
  rhs_operator_data.dof_index  = 0;
  rhs_operator_data.quad_index = 0;
  rhs_operator_data.rhs        = field_functions->right_hand_side;
  rhs_operator.initialize(data, rhs_operator_data);
}
} // namespace Poisson
