#include "operator.h"

#include "../../functionalities/calculate_maximum_aspect_ratio.h"
#include "../../solvers_and_preconditioners/util/check_multigrid.h"

namespace Poisson
{
template<int dim, typename Number, int n_components>
Operator<dim, Number, n_components>::Operator(
  parallel::TriangulationBase<dim> const &                triangulation_in,
  Mapping<dim> const &                                    mapping_in,
  PeriodicFaces const                                     periodic_face_pairs_in,
  std::shared_ptr<Poisson::BoundaryDescriptor<dim>> const boundary_descriptor_in,
  std::shared_ptr<Poisson::FieldFunctions<dim>> const     field_functions_in,
  Poisson::InputParameters const &                        param_in,
  MPI_Comm const &                                        mpi_comm_in)
  : dealii::Subscriptor(),
    mapping(mapping_in),
    periodic_face_pairs(periodic_face_pairs_in),
    boundary_descriptor(boundary_descriptor_in),
    field_functions(field_functions_in),
    param(param_in),
    dof_handler(triangulation_in),
    mpi_comm(mpi_comm_in),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm_in) == 0)
{
  pcout << std::endl << "Construct Poisson operator ..." << std::endl;

  distribute_dofs();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number, int n_components>
void
Operator<dim, Number, n_components>::append_data_structures(
  MatrixFreeWrapper<dim, Number> & matrix_free_wrapper,
  std::string const &              field) const
{
  this->field = field;

  // append mapping flags
  matrix_free_wrapper.append_mapping_flags(
    Operators::LaplaceKernel<dim, Number, n_components>::get_mapping_flags(
      param.spatial_discretization == SpatialDiscretization::DG));

  if(param.right_hand_side)
    matrix_free_wrapper.append_mapping_flags(
      ConvDiff::Operators::RHSKernel<dim, Number, n_components>::get_mapping_flags());

  // DoFHandler
  matrix_free_wrapper.insert_dof_handler(&dof_handler, field + dof_index);

  // AffineConstraints
  if(param.spatial_discretization == SpatialDiscretization::CG)
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
  matrix_free_wrapper.insert_constraint(&constraint_matrix, field + dof_index);

  // Quadrature
  matrix_free_wrapper.insert_quadrature(QGauss<1>(param.degree + 1), field + quad_index);
}

template<int dim, typename Number, int n_components>
void
Operator<dim, Number, n_components>::setup(
  std::shared_ptr<MatrixFreeWrapper<dim, Number>> matrix_free_wrapper_in)
{
  pcout << std::endl << "Setup Poisson operator ..." << std::endl;

  matrix_free_wrapper = matrix_free_wrapper_in;
  matrix_free         = matrix_free_wrapper->get_matrix_free();

  setup_operators();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number, int n_components>
void
Operator<dim, Number, n_components>::setup_solver()
{
  pcout << std::endl << "Setup Poisson solver ..." << std::endl;

  // initialize preconditioner
  if(param.preconditioner == Poisson::Preconditioner::PointJacobi)
  {
    preconditioner.reset(new JacobiPreconditioner<Laplace>(laplace_operator));
  }
  else if(param.preconditioner == Poisson::Preconditioner::BlockJacobi)
  {
    preconditioner.reset(new BlockJacobiPreconditioner<Laplace>(laplace_operator));
  }
  else if(param.preconditioner == Poisson::Preconditioner::Multigrid)
  {
    MultigridData mg_data;
    mg_data = param.multigrid_data;

    preconditioner.reset(new Multigrid(this->mpi_comm));

    std::shared_ptr<Multigrid> mg_preconditioner =
      std::dynamic_pointer_cast<Multigrid>(preconditioner);

    parallel::TriangulationBase<dim> const * tria =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(
        &this->dof_handler.get_triangulation());
    const FiniteElement<dim> & fe = this->dof_handler.get_fe();

    mg_preconditioner->initialize(mg_data,
                                  tria,
                                  fe,
                                  mapping,
                                  laplace_operator.get_data(),
                                  false /* moving_mesh */,
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
    iterative_solver.reset(new CGSolver<Laplace, PreconditionerBase<Number>, VectorType>(
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
    iterative_solver.reset(new FGMRESSolver<Laplace, PreconditionerBase<Number>, VectorType>(
      laplace_operator, *preconditioner, solver_data));
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified solver is not implemented!"));
  }

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number, int n_components>
void
Operator<dim, Number, n_components>::initialize_dof_vector(VectorType & src) const
{
  matrix_free->initialize_dof_vector(src, get_dof_index());
}

template<int dim, typename Number, int n_components>
void
Operator<dim, Number, n_components>::prescribe_initial_conditions(VectorType & src) const
{
  field_functions->initial_solution->set_time(0.0);

  // This is necessary if Number == float
  typedef LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

  VectorTypeDouble src_double;
  src_double = src;

  VectorTools::interpolate(dof_handler, *(field_functions->initial_solution), src_double);

  src = src_double;
}

template<int dim, typename Number, int n_components>
void
Operator<dim, Number, n_components>::rhs(VectorType & dst, double const time) const
{
  dst = 0;

  laplace_operator.set_time(time);
  laplace_operator.rhs_add(dst);

  if(param.right_hand_side)
    rhs_operator.evaluate_add(dst, time);
}

template<int dim, typename Number, int n_components>
void
Operator<dim, Number, n_components>::vmult(VectorType & dst, VectorType const & src) const
{
  laplace_operator.vmult(dst, src);
}

template<int dim, typename Number, int n_components>
unsigned int
Operator<dim, Number, n_components>::solve(VectorType & sol, VectorType const & rhs) const
{
  // only activate if desired
  if(false)
  {
    std::shared_ptr<Multigrid> mg_preconditioner =
      std::dynamic_pointer_cast<Multigrid>(preconditioner);

    CheckMultigrid<dim, Number, Laplace, Multigrid, MultigridNumber> check_multigrid(
      laplace_operator, mg_preconditioner, mpi_comm);

    check_multigrid.check();
  }

  unsigned int iterations = iterative_solver->solve(sol, rhs, /* update_preconditioner = */ false);

  return iterations;
}

template<int dim, typename Number, int n_components>
DoFHandler<dim> const &
Operator<dim, Number, n_components>::get_dof_handler() const
{
  return dof_handler;
}

template<int dim, typename Number, int n_components>
types::global_dof_index
Operator<dim, Number, n_components>::get_number_of_dofs() const
{
  return dof_handler.n_dofs();
}

template<int dim, typename Number, int n_components>
double
Operator<dim, Number, n_components>::get_n10() const
{
  AssertThrow(iterative_solver->n10 != 0,
              ExcMessage("Make sure to activate param.compute_performance_metrics!"));

  return iterative_solver->n10;
}

template<int dim, typename Number, int n_components>
double
Operator<dim, Number, n_components>::get_average_convergence_rate() const
{
  return iterative_solver->rho;
}

#ifdef DEAL_II_WITH_TRILINOS
template<int dim, typename Number, int n_components>
void
Operator<dim, Number, n_components>::init_system_matrix(
  TrilinosWrappers::SparseMatrix & system_matrix) const
{
  laplace_operator.init_system_matrix(system_matrix);
}

template<int dim, typename Number, int n_components>
void
Operator<dim, Number, n_components>::calculate_system_matrix(
  TrilinosWrappers::SparseMatrix & system_matrix) const
{
  laplace_operator.calculate_system_matrix(system_matrix);
}

template<int dim, typename Number, int n_components>
void
Operator<dim, Number, n_components>::vmult_matrix_based(
  VectorTypeDouble &                     dst,
  TrilinosWrappers::SparseMatrix const & system_matrix,
  VectorTypeDouble const &               src) const
{
  system_matrix.vmult(dst, src);
}
#endif

template<int dim, typename Number, int n_components>
unsigned int
Operator<dim, Number, n_components>::get_dof_index() const
{
  return matrix_free_wrapper->get_dof_index(field + dof_index);
}

template<int dim, typename Number, int n_components>
unsigned int
Operator<dim, Number, n_components>::get_quad_index() const
{
  return matrix_free_wrapper->get_quad_index(field + quad_index);
}

template<int dim, typename Number, int n_components>
void
Operator<dim, Number, n_components>::distribute_dofs()
{
  if(n_components == 1)
  {
    if(param.spatial_discretization == SpatialDiscretization::DG)
      fe.reset(new FE_DGQ<dim>(param.degree));
    else if(param.spatial_discretization == SpatialDiscretization::CG)
      fe.reset(new FE_Q<dim>(param.degree));
    else
      AssertThrow(false, ExcMessage("not implemented."));
  }
  else if(n_components == dim)
  {
    if(param.spatial_discretization == SpatialDiscretization::DG)
      fe.reset(new FESystem<dim>(FE_DGQ<dim>(param.degree), dim));
    else if(param.spatial_discretization == SpatialDiscretization::CG)
      fe.reset(new FESystem<dim>(FE_Q<dim>(param.degree), dim));
    else
      AssertThrow(false, ExcMessage("not implemented."));
  }
  else
  {
    AssertThrow(false, ExcMessage("not implemented."));
  }

  dof_handler.distribute_dofs(*fe);

  dof_handler.distribute_mg_dofs();

  unsigned int const ndofs_per_cell = Utilities::pow(param.degree + 1, dim);

  pcout << std::endl
        << "Discontinuous Galerkin finite element discretization:" << std::endl
        << std::endl;

  print_parameter(pcout, "degree of 1D polynomials", param.degree);
  print_parameter(pcout, "number of dofs per cell", ndofs_per_cell);
  print_parameter(pcout, "number of dofs (total)", dof_handler.n_dofs());
}

template<int dim, typename Number, int n_components>
void
Operator<dim, Number, n_components>::setup_operators()
{
  // Laplace operator
  Poisson::LaplaceOperatorData<dim> laplace_operator_data;
  laplace_operator_data.dof_index             = get_dof_index();
  laplace_operator_data.quad_index            = get_quad_index();
  laplace_operator_data.bc                    = boundary_descriptor;
  laplace_operator_data.use_cell_based_loops  = param.enable_cell_based_face_loops;
  laplace_operator_data.kernel_data.IP_factor = param.IP_factor;
  laplace_operator.reinit(*matrix_free, constraint_matrix, laplace_operator_data);

  // rhs operator
  if(param.right_hand_side)
  {
    ConvDiff::RHSOperatorData<dim> rhs_operator_data;
    rhs_operator_data.dof_index     = get_dof_index();
    rhs_operator_data.quad_index    = get_quad_index();
    rhs_operator_data.kernel_data.f = field_functions->right_hand_side;
    rhs_operator.reinit(*matrix_free, rhs_operator_data);
  }
}

template class Operator<2, float, 1>;
template class Operator<2, double, 1>;
template class Operator<2, float, 2>;
template class Operator<2, double, 2>;

template class Operator<3, float, 1>;
template class Operator<3, double, 1>;
template class Operator<3, float, 3>;
template class Operator<3, double, 3>;
} // namespace Poisson
