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

  flags = flags || BodyForceOperator<dim, Number>::get_mapping_flags();

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

template<int dim, typename Number>
void
Operator<dim, Number>::setup_operators()
{
  // pass boundary conditions to operator
  operator_data.bc                  = boundary_descriptor;
  operator_data.material_descriptor = material_descriptor;
  if(param.large_deformation)
    operator_data.pull_back_traction = param.pull_back_traction;
  else
    operator_data.pull_back_traction = false;

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
  BodyForceData<dim> body_force_data;
  body_force_data.dof_index  = 0;
  body_force_data.quad_index = 0;
  body_force_data.function   = field_functions->right_hand_side;
  if(param.large_deformation)
    body_force_data.pull_back_body_force = param.pull_back_body_force;
  else
    body_force_data.pull_back_body_force = false;
  body_force_operator.reinit(matrix_free, body_force_data);
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
    if(param.large_deformation)
    {
      preconditioner.reset(
        new JacobiPreconditioner<NonLinearOperator<dim, Number>>(elasticity_operator_nonlinear));
    }
    else
    {
      preconditioner.reset(
        new JacobiPreconditioner<LinearOperator<dim, Number>>(elasticity_operator_linear));
    }
  }
  else if(param.preconditioner == Preconditioner::Multigrid)
  {
    parallel::TriangulationBase<dim> const * tria =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(&dof_handler.get_triangulation());
    const FiniteElement<dim> & fe = dof_handler.get_fe();

    if(param.large_deformation)
    {
      typedef MultigridPreconditioner<dim, Number, MultigridNumber, true /* nonlinear */> Multigrid;

      preconditioner.reset(new Multigrid(mpi_comm));
      std::shared_ptr<Multigrid> mg_preconditioner =
        std::dynamic_pointer_cast<Multigrid>(preconditioner);

      mg_preconditioner->initialize(param.multigrid_data,
                                    tria,
                                    fe,
                                    mapping,
                                    elasticity_operator_nonlinear,
                                    &elasticity_operator_nonlinear.get_data().bc->dirichlet_bc,
                                    &this->periodic_face_pairs);
    }
    else
    {
      typedef MultigridPreconditioner<dim, Number, MultigridNumber, false /* linear */> Multigrid;

      preconditioner.reset(new Multigrid(mpi_comm));
      std::shared_ptr<Multigrid> mg_preconditioner =
        std::dynamic_pointer_cast<Multigrid>(preconditioner);

      mg_preconditioner->initialize(param.multigrid_data,
                                    tria,
                                    fe,
                                    mapping,
                                    elasticity_operator_linear,
                                    &elasticity_operator_linear.get_data().bc->dirichlet_bc,
                                    &this->periodic_face_pairs);
    }
  }
  else if(param.preconditioner == Preconditioner::AMG)
  {
    if(param.large_deformation)
    {
      typedef AlgebraicMultigridPreconditioner<NonLinearOperator<dim, Number>, Number> AMG;
      preconditioner.reset(new AMG(elasticity_operator_nonlinear, AMGData()));
    }
    else
    {
      typedef AlgebraicMultigridPreconditioner<LinearOperator<dim, Number>, Number> AMG;
      preconditioner.reset(new AMG(elasticity_operator_linear, AMGData()));
    }
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
  // initialize linear solver
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
      linear_solver.reset(
        new CGSolver<NonLinearOperator<dim, Number>, PreconditionerBase<Number>, VectorType>(
          elasticity_operator_nonlinear, *preconditioner, solver_data));
    else
      linear_solver.reset(
        new CGSolver<LinearOperator<dim, Number>, PreconditionerBase<Number>, VectorType>(
          elasticity_operator_linear, *preconditioner, solver_data));
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified solver is not implemented!"));
  }

  // initialize Newton solver
  if(param.large_deformation)
  {
    newton_solver.reset(new NewtonSolver(
      param.newton_solver_data, residual_operator, linearized_operator, *linear_solver));
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
Operator<dim, Number>::compute_rhs_linear(VectorType & dst, double const time) const
{
  dst = 0.0;

  // body force
  if(param.body_force)
  {
    // src is irrelevant for linear problem, since
    // pull_back_body_force = false in this case.
    VectorType src;
    body_force_operator.evaluate_add(dst, src, time);
  }

  // Neumann BCs and inhomogeneous Dirichlet BCs
  if(param.large_deformation == false)
  {
    elasticity_operator_linear.set_time(time);
    elasticity_operator_linear.rhs_add(dst);
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::evaluate_nonlinear_residual(VectorType &       dst,
                                                   VectorType const & src,
                                                   VectorType const & const_vector,
                                                   double const       time) const
{
  // TODO dynamic problems
  (void)const_vector;

  // elasticity operator
  elasticity_operator_nonlinear.set_time(time);
  elasticity_operator_nonlinear.evaluate_nonlinear(dst, src);

  // body forces
  if(param.body_force)
  {
    VectorType body_forces;
    body_forces.reinit(dst);
    body_force_operator.evaluate_add(body_forces, src, time);
    dst -= body_forces;
  }
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

  // set inhomogeneous Dirichlet values
  elasticity_operator_nonlinear.set_dirichlet_values_continuous(sol, time);

  // call Newton solver
  Newton::UpdateData update;
  update.do_update             = update_preconditioner;
  update.threshold_newton_iter = param.update_preconditioner_every_newton_iterations;

  std::tuple<unsigned int, unsigned int> iter = newton_solver->solve(sol, update);

  newton_iterations = std::get<0>(iter);
  linear_iterations = std::get<1>(iter);
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::solve_linear(VectorType & sol, VectorType const & rhs, double const time)
{
  unsigned int const iterations = linear_solver->solve(sol, rhs, false);

  // set Dirichlet values
  elasticity_operator_linear.set_dirichlet_values_continuous(sol, time);

  return iterations;
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
