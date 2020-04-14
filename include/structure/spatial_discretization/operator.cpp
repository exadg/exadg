/*
 * operator.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#include "operator.h"

#include "../../solvers_and_preconditioners/preconditioner/jacobi_preconditioner.h"
#include "../../solvers_and_preconditioners/preconditioner/preconditioner_amg.h"
#include "../preconditioners/multigrid_preconditioner.h"

namespace Structure
{
// TODO this functionality should be replaced by MovingMesh functionality developed recently
template<int dim, typename VectorType>
void
do_move_mesh(DoFHandler<dim, dim> & dof_handler, const VectorType & solution)
{
  std::vector<bool> vertex_touched(dof_handler.get_triangulation().n_vertices(), false);

  for(typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
      cell != dof_handler.end();
      ++cell)
  {
    if(cell->is_locally_owned())
    {
      for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
      {
        if(vertex_touched[cell->vertex_index(v)] == false)
        {
          vertex_touched[cell->vertex_index(v)] = true;

          Point<dim> vertex_displacement;

          for(unsigned int d = 0; d < dim; ++d)
            vertex_displacement[d] = solution(cell->vertex_dof_index(v, d));
          cell->vertex(v) += vertex_displacement;
        }
      }
    }
  }
}

template<int dim, typename Number>
Operator<dim, Number>::Operator(
  parallel::TriangulationBase<dim> &             triangulation_in,
  Mapping<dim> const &                           mapping_in,
  unsigned int const &                           degree_in,
  PeriodicFaces const &                          periodic_face_pairs_in,
  std::shared_ptr<BoundaryDescriptor<dim>> const boundary_descriptor_in,
  std::shared_ptr<FieldFunctions<dim>> const     field_functions_in,
  std::shared_ptr<MaterialDescriptor> const      material_descriptor_in,
  InputParameters const &                        param_in,
  MPI_Comm const &                               mpi_comm_in)
  : dealii::Subscriptor(),
    mapping(mapping_in),
    periodic_face_pairs(periodic_face_pairs_in),
    boundary_descriptor(boundary_descriptor_in),
    field_functions(field_functions_in),
    material_descriptor(material_descriptor_in),
    param(param_in),
    mpi_comm(mpi_comm_in),
    degree(degree_in),
    fe(FE_Q<dim>(degree_in), dim),
    dof_handler(triangulation_in),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm_in) == 0)
{
  pcout << std::endl << "Construct elasticity operator ..." << std::endl;

  distribute_dofs();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
Operator<dim, Number>::setup()
{
  pcout << std::endl << "Setup spatial discretization operator ..." << std::endl;

  initialize_matrix_free();

  setup_operators();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
Operator<dim, Number>::distribute_dofs()
{
  // enumerate degrees of freedom
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();

  // print some statistics on the finest grid
  if(fe.dofs_per_vertex == 0)
  {
    pcout << std::endl
          << "Discontinuous Galerkin finite element discretization:" << std::endl
          << std::endl;
  }
  else
  {
    pcout << std::endl
          << "Continuous Galerkin finite element discretization:" << std::endl
          << std::endl;
  }

  print_parameter(pcout, "degree of 1D polynomials", degree);
  print_parameter(pcout, "number of dofs per cell", Utilities::pow(degree + 1, dim));
  print_parameter(pcout, "number of dofs (total)", dof_handler.n_dofs());
}

// TODO use matrix_free_wrapper concept
template<int dim, typename Number>
void
Operator<dim, Number>::initialize_matrix_free()
{
  // initialize matrix_free_data
  typename MatrixFree<dim, Number>::AdditionalData additional_data;

  MappingFlags flags;
  if(param.large_deformation)
    flags = flags || NonLinearOperator<dim, Number>::get_mapping_flags();
  else
    flags = flags || LinearOperator<dim, Number>::get_mapping_flags();

  flags = flags || RHSOperator<dim, Number>::get_mapping_flags();

  additional_data.mapping_update_flags                = flags.cells;
  additional_data.mapping_update_flags_inner_faces    = flags.inner_faces;
  additional_data.mapping_update_flags_boundary_faces = flags.boundary_faces;

  // determine constrained dofs
  constraint_matrix.clear();
  for(auto it : this->boundary_descriptor->dirichlet_bc)
  {
    ComponentMask mask =
      this->boundary_descriptor->dirichlet_bc_component_mask.find(it.first)->second;

    DoFTools::make_zero_boundary_constraints(this->dof_handler, it.first, constraint_matrix, mask);
  }
  constraint_matrix.close();

  // quadrature formula used to perform integrals
  QGauss<1> quadrature(degree + 1);

  std::vector<const DoFHandler<dim> *>           dof_handlers{&dof_handler};
  std::vector<const AffineConstraints<double> *> constraints{&constraint_matrix};

  matrix_free.reinit(mapping, dof_handlers, constraints, quadrature, additional_data);
}

// TODO use matrix_free_wrapper concept
template<int dim, typename Number>
void
Operator<dim, Number>::reinitialize_matrix_free()
{
  // initialize matrix_free_data
  typename MatrixFree<dim, Number>::AdditionalData additional_data;

  MappingFlags flags;
  if(param.large_deformation)
    flags = flags || NonLinearOperator<dim, Number>::get_mapping_flags();
  else
    flags = flags || LinearOperator<dim, Number>::get_mapping_flags();

  flags = flags || RHSOperator<dim, Number>::get_mapping_flags();

  additional_data.mapping_update_flags                = flags.cells;
  additional_data.mapping_update_flags_inner_faces    = flags.inner_faces;
  additional_data.mapping_update_flags_boundary_faces = flags.boundary_faces;

  // note: here we can skip setting up the constraint matrix
  //       since it does not change

  // quadrature formula used to perform integrals
  QGauss<1> quadrature(degree + 1);

  std::vector<const DoFHandler<dim> *>           dof_handlers{&dof_handler};
  std::vector<const AffineConstraints<double> *> constraints{&constraint_matrix};

  matrix_free.reinit(mapping, dof_handlers, constraints, quadrature, additional_data);
}


template<int dim, typename Number>
void
Operator<dim, Number>::setup_operators()
{
  // pass boundary conditions to operator
  operator_data.bc                  = boundary_descriptor;
  operator_data.material_descriptor = material_descriptor;
  operator_data.updated_formulation = param.updated_formulation;

  // setup operator
  if(param.large_deformation)
  {
    elasticity_operator_nonlinear.reinit(matrix_free, constraint_matrix, operator_data);

    residual_operator.initialize(*this);
    linearized_operator.initialize(*this);
  }
  else
  {
    elasticity_operator_linear.reinit(matrix_free, constraint_matrix, operator_data);
  }

  // setup rhs operator
  RHSOperatorData<dim> rhs_operator_data;
  rhs_operator_data.dof_index  = 0;
  rhs_operator_data.quad_index = 0;
  rhs_operator_data.degree     = degree;
  rhs_operator_data.rhs        = field_functions->right_hand_side;
  rhs_operator_data.do_rhs     = param.right_hand_side;
  rhs_operator_data.bc         = boundary_descriptor;
  rhs_operator.reinit(matrix_free, dof_handler, constraint_matrix, mapping, rhs_operator_data);
}

template<int dim, typename Number>
void
Operator<dim, Number>::setup_solver()
{
  pcout << std::endl << "Setup solver ..." << std::endl;

  initialize_preconditioner();

  initialize_solver();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
Operator<dim, Number>::initialize_preconditioner()
{
  if(param.preconditioner == Preconditioner::None)
  {
    // do nothing
  }
  else if(param.preconditioner == Preconditioner::PointJacobi)
  {
    AssertThrow(!param.large_deformation,
                ExcMessage("PointJacobi is not implemented yet for non-linear operator!"));

    preconditioner.reset(
      new JacobiPreconditioner<LinearOperator<dim, Number>>(elasticity_operator_linear));
  }
  else if(param.preconditioner == Preconditioner::Multigrid)
  {
    AssertThrow(!param.large_deformation,
                ExcMessage("Geometric multigrid is not implemented for non-linear operator!"));

    typedef MultigridPreconditioner<dim, Number, MultigridNumber> Multigrid;

    preconditioner.reset(new Multigrid(mpi_comm));
    std::shared_ptr<Multigrid> mg_preconditioner =
      std::dynamic_pointer_cast<Multigrid>(preconditioner);

    parallel::TriangulationBase<dim> const * tria =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(&dof_handler.get_triangulation());
    const FiniteElement<dim> & fe = dof_handler.get_fe();

    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> dirichlet;
    mg_preconditioner->initialize(param.multigrid_data,
                                  tria,
                                  fe,
                                  mapping,
                                  elasticity_operator_linear,
                                  &elasticity_operator_linear.get_data().bc->dirichlet_bc,
                                  &this->periodic_face_pairs);
  }
  else if(param.preconditioner == Preconditioner::AMG)
  {
    if(param.large_deformation)
      preconditioner_amg.reset(
        new PreconditionerAMG<NonLinearOperator<dim, Number>, double>(elasticity_operator_nonlinear,
                                                                      AMGData()));
    else
      preconditioner_amg.reset(
        new PreconditionerAMG<LinearOperator<dim, Number>, double>(elasticity_operator_linear,
                                                                   AMGData()));
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified preconditioner is not implemented!"));
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::initialize_solver()
{
  // initialize linear solvers
  if(param.solver == Solver::CG)
  {
    // initialize solver_data
    CGSolverData solver_data;
    solver_data.solver_tolerance_abs = param.solver_data.abs_tol;
    solver_data.solver_tolerance_rel = param.solver_data.rel_tol;
    solver_data.max_iter             = param.solver_data.max_iter;

    if(param.preconditioner != Preconditioner::None)
      solver_data.use_preconditioner = true;

    // initialize solver
    if(param.large_deformation)
      iterative_solver.reset(
        new CGSolver<NonLinearOperator<dim, Number>, PreconditionerBase<Number>, VectorType>(
          elasticity_operator_nonlinear, *preconditioner, solver_data));
    else
      iterative_solver.reset(
        new CGSolver<LinearOperator<dim, Number>, PreconditionerBase<Number>, VectorType>(
          elasticity_operator_linear, *preconditioner, solver_data));
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified solver is not implemented!"));
  }

  // initialize non-linear solver
  if(param.large_deformation)
  {
    // TODO: move to parameters
    NewtonSolverData newton_data;
    newton_data.rel_tol = 1.e-5;

    // initialize Newton
    non_linear_solver.reset(new NewtonSolver<VectorType,
                                             ResidualOperator<dim, Number>,
                                             LinearizedOperator<dim, Number>,
                                             IterativeSolverBase<VectorType>>(
      newton_data, residual_operator, linearized_operator, *iterative_solver));
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::initialize_dof_vector(VectorType & src) const
{
  matrix_free.initialize_dof_vector(src, 0);
}

template<int dim, typename Number>
void
Operator<dim, Number>::prescribe_initial_conditions(VectorType & src,
                                                    double const evaluation_time) const
{
  (void)evaluation_time;
  src = 0.0; // start with initial guess 0
  src.update_ghost_values();
}

template<int dim, typename Number>
void
Operator<dim, Number>::rhs(VectorType & dst, double const evaluation_time) const
{
  // compute volume force
  dst = 0.0;
  rhs_operator.evaluate_add(dst, evaluation_time);

  // nonlinear problem: evaluate NBC-contribution
  if(param.large_deformation)
  {
    // TODO: implement Neumann boundary conditions as inhomogeneous
    // contributions of nonlinear operator and not as part of
    // rhs_operator (rhs_operator should only account for body forces)
    rhs_operator.evaluate_add_nbc(dst, evaluation_time);
  }

  // linear problem: Neumann BCs and inhomogeneous Dirichlet BCs
  if(param.large_deformation == false)
  {
    elasticity_operator_linear.set_time(evaluation_time);
    elasticity_operator_linear.rhs_add(dst);
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::evaluate_nonlinear_residual(VectorType &       dst,
                                                   VectorType const & src,
                                                   VectorType const & rhs_vector,
                                                   double const       time) const
{
  // TODO dynamic problems

  elasticity_operator_nonlinear.set_time(time);
  elasticity_operator_nonlinear.evaluate(dst, src);

  dst -= rhs_vector;
}

template<int dim, typename Number>
void
Operator<dim, Number>::set_solution_linearization(VectorType const & vector) const
{
  elasticity_operator_nonlinear.set_solution_linearization(vector);
}

template<int dim, typename Number>
void
Operator<dim, Number>::apply_linearized_operator(VectorType &       dst,
                                                 VectorType const & src,
                                                 double const       time) const
{
  // TODO dynamic problems

  elasticity_operator_nonlinear.set_time(time);
  elasticity_operator_nonlinear.vmult(dst, src);
}

template<int dim, typename Number>
void
Operator<dim, Number>::solve_nonlinear(VectorType &       sol,
                                       VectorType const & rhs,
                                       double const       time,
                                       bool const         update_preconditioner,
                                       unsigned int &     newton_iterations,
                                       unsigned int &     linear_iterations)
{
  // update operators
  residual_operator.update(rhs, time);
  linearized_operator.update(time);

  // compute values at Dirichlet boundaries
  std::map<types::global_dof_index, double> boundary_values;
  for(auto dbc : boundary_descriptor->dirichlet_bc)
  {
    dbc.second->set_time(time);
    ComponentMask mask =
      this->boundary_descriptor->dirichlet_bc_component_mask.find(dbc.first)->second;

    VectorTools::interpolate_boundary_values(
      this->mapping, this->dof_handler, dbc.first, *dbc.second, boundary_values, mask);
  }

  // set Dirichlet values in solution vector
  for(auto m : boundary_values)
    if(sol.get_partitioner()->in_local_range(m.first))
      sol[m.first] = m.second;

  // call Newton solver
  non_linear_solver->solve(
    sol, newton_iterations, linear_iterations, update_preconditioner, 1 /* TODO */);
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::solve_linear(VectorType &       sol,
                                    VectorType const & rhs,
                                    double const       time,
                                    bool const         update_preconditioner)
{
  unsigned int iterations = 0;

  if(param.preconditioner == Preconditioner::AMG)
  {
#ifdef DEAL_II_WITH_TRILINOS
    TrilinosWrappers::SparseMatrix const * system_matrix = nullptr;

    std::shared_ptr<PreconditionerAMG<LinearOperator<dim, Number>, double>> preconditioner =
      std::dynamic_pointer_cast<PreconditionerAMG<LinearOperator<dim, Number>, double>>(
        preconditioner_amg);
    if(update_preconditioner)
      preconditioner->update();
    system_matrix = &preconditioner->get_system_matrix();

    ReductionControl solver_control(param.solver_data.max_iter,
                                    param.solver_data.abs_tol,
                                    param.solver_data.rel_tol);

    // create temporal vectors of type VectorTypeTrilinos (double)
    typedef LinearAlgebra::distributed::Vector<double> VectorTypeTrilinos;
    VectorTypeTrilinos                                 sol_trilinos;
    sol_trilinos.reinit(sol, false);
    VectorTypeTrilinos rhs_trilinos;
    rhs_trilinos.reinit(rhs, true);
    rhs_trilinos = rhs;

    SolverCG<VectorTypeTrilinos> solver(solver_control);
    solver.solve(*system_matrix, sol_trilinos, rhs_trilinos, *preconditioner_amg);

    // convert: VectorTypeTrilinos -> VectorTypeMultigrid
    sol.copy_locally_owned_data_from(sol_trilinos);

    iterations = solver_control.last_step();
#else
    AssertThrow(false, ExcMessage("deal.II is not compiled with Trilinos!"));
#endif
  }
  else
  {
    iterations = iterative_solver->solve(sol, rhs, update_preconditioner);
  }

  // set Dirichlet values
  elasticity_operator_linear.set_dirichlet_values_continuous(sol, time);

  return iterations;
}

template<int dim, typename Number>
void
Operator<dim, Number>::move_mesh(const VectorType & solution)
{
  // move vertices according to the vector
  do_move_mesh(this->dof_handler, solution);

  // reinitialize the mapping with matrix_free
  this->reinitialize_matrix_free();

  // update preconditioner
  this->preconditioner->update();
}

template<int dim, typename Number>
MatrixFree<dim, Number> const &
Operator<dim, Number>::get_matrix_free() const
{
  return matrix_free;
}

template<int dim, typename Number>
Mapping<dim> const &
Operator<dim, Number>::get_mapping() const
{
  return mapping;
}

template<int dim, typename Number>
DoFHandler<dim> const &
Operator<dim, Number>::get_dof_handler() const
{
  return dof_handler;
}

template class Operator<2, float>;
template class Operator<2, double>;

template class Operator<3, float>;
template class Operator<3, double>;

} // namespace Structure
