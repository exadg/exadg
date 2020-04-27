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
  std::string const &                            field_in,
  MPI_Comm const &                               mpi_comm_in)
  : dealii::Subscriptor(),
    mapping(mapping_in),
    periodic_face_pairs(periodic_face_pairs_in),
    boundary_descriptor(boundary_descriptor_in),
    field_functions(field_functions_in),
    material_descriptor(material_descriptor_in),
    param(param_in),
    field(field_in),
    degree(degree_in),
    fe(FE_Q<dim>(degree_in), dim),
    dof_handler(triangulation_in),
    mpi_comm(mpi_comm_in),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm_in) == 0)
{
  pcout << std::endl << "Construct elasticity operator ..." << std::endl;

  distribute_dofs();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
Operator<dim, Number>::setup(std::shared_ptr<MatrixFree<dim, Number>>     matrix_free_in,
                             std::shared_ptr<MatrixFreeData<dim, Number>> matrix_free_data_in)
{
  pcout << std::endl << "Setup spatial discretization operator ..." << std::endl;

  matrix_free      = matrix_free_in;
  matrix_free_data = matrix_free_data_in;

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

  // determine constrained dofs
  constraint_matrix.clear();
  for(auto it : this->boundary_descriptor->dirichlet_bc)
  {
    ComponentMask mask =
      this->boundary_descriptor->dirichlet_bc_component_mask.find(it.first)->second;

    DoFTools::make_zero_boundary_constraints(this->dof_handler, it.first, constraint_matrix, mask);
  }
  constraint_matrix.close();

  pcout << std::endl
        << "Continuous Galerkin finite element discretization:" << std::endl
        << std::endl;

  print_parameter(pcout, "degree of 1D polynomials", degree);
  print_parameter(pcout, "number of dofs per cell", Utilities::pow(degree + 1, dim));
  print_parameter(pcout, "number of dofs (total)", dof_handler.n_dofs());
}

template<int dim, typename Number>
std::string
Operator<dim, Number>::get_dof_name() const
{
  return field + "_" + dof_index;
}

template<int dim, typename Number>
std::string
Operator<dim, Number>::get_quad_name() const
{
  return field + "_" + quad_index;
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::get_dof_index() const
{
  return matrix_free_data->get_dof_index(get_dof_name());
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::get_quad_index() const
{
  return matrix_free_data->get_quad_index(get_quad_name());
}

template<int dim, typename Number>
void
Operator<dim, Number>::fill_matrix_free_data(MatrixFreeData<dim, Number> & matrix_free_data) const
{
  if(param.large_deformation)
    matrix_free_data.append_mapping_flags(NonLinearOperator<dim, Number>::get_mapping_flags());
  else
    matrix_free_data.append_mapping_flags(LinearOperator<dim, Number>::get_mapping_flags());

  if(param.body_force)
    matrix_free_data.append_mapping_flags(BodyForceOperator<dim, Number>::get_mapping_flags());

  // DoFHandler, AffineConstraints
  matrix_free_data.insert_dof_handler(&dof_handler, get_dof_name());
  matrix_free_data.insert_constraint(&constraint_matrix, get_dof_name());

  // Quadrature
  matrix_free_data.insert_quadrature(QGauss<1>(degree + 1), get_quad_name());
}

template<int dim, typename Number>
void
Operator<dim, Number>::setup_operators()
{
  // elasticity operator
  operator_data.bc                  = boundary_descriptor;
  operator_data.material_descriptor = material_descriptor;
  operator_data.unsteady            = (param.problem_type == ProblemType::Unsteady);
  operator_data.density             = param.density;
  if(param.large_deformation)
  {
    operator_data.pull_back_traction = param.pull_back_traction;
  }
  else
  {
    operator_data.pull_back_traction = false;
  }

  if(param.large_deformation)
  {
    elasticity_operator_nonlinear.reinit(*matrix_free, constraint_matrix, operator_data);
  }
  else
  {
    elasticity_operator_linear.reinit(*matrix_free, constraint_matrix, operator_data);
  }

  // mass matrix operator and related solver for inversion
  if(param.problem_type == ProblemType::Unsteady)
  {
    MassMatrixOperatorData<dim> mass_data;
    mass_data.dof_index  = get_dof_index();
    mass_data.quad_index = get_quad_index();
    mass.reinit(*matrix_free, constraint_matrix, mass_data);
    mass.set_scaling_factor(param.density);

    // preconditioner and solver for mass matrix have to be initialized in
    // setup_operators() since the mass matrix solver is already needed in
    // setup() function of time integration scheme.

    // preconditioner
    mass_preconditioner.reset(new JacobiPreconditioner<MassMatrixOperator<dim, dim, Number>>(mass));

    // initialize solver
    CGSolverData solver_data;
    solver_data.use_preconditioner = true;
    // use the same solver tolerances as for solving the momentum equation
    if(param.large_deformation)
    {
      solver_data.solver_tolerance_abs = param.newton_solver_data.abs_tol;
      solver_data.solver_tolerance_rel = param.newton_solver_data.rel_tol;
      solver_data.max_iter             = param.newton_solver_data.max_iter;
    }
    else
    {
      solver_data.solver_tolerance_abs = param.solver_data.abs_tol;
      solver_data.solver_tolerance_rel = param.solver_data.rel_tol;
      solver_data.max_iter             = param.solver_data.max_iter;
    }

    mass_solver.reset(
      new CGSolver<MassMatrixOperator<dim, dim, Number>, PreconditionerBase<Number>, VectorType>(
        mass, *mass_preconditioner, solver_data));
  }

  // setup rhs operator
  BodyForceData<dim> body_force_data;
  body_force_data.dof_index  = get_dof_index();
  body_force_data.quad_index = get_quad_index();
  body_force_data.function   = field_functions->right_hand_side;
  if(param.large_deformation)
    body_force_data.pull_back_body_force = param.pull_back_body_force;
  else
    body_force_data.pull_back_body_force = false;
  body_force_operator.reinit(*matrix_free, body_force_data);
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
    residual_operator.initialize(*this);
    linearized_operator.initialize(*this);

    newton_solver.reset(new NewtonSolver(
      param.newton_solver_data, residual_operator, linearized_operator, *linear_solver));
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::initialize_dof_vector(VectorType & src) const
{
  matrix_free->initialize_dof_vector(src, get_dof_index());
}

template<int dim, typename Number>
void
Operator<dim, Number>::prescribe_initial_displacement(VectorType & displacement,
                                                      double const time) const
{
  // This is necessary if Number == float
  typedef LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

  VectorTypeDouble src_double;
  src_double = displacement;

  field_functions->initial_displacement->set_time(time);
  VectorTools::interpolate(dof_handler, *field_functions->initial_displacement, src_double);

  displacement = src_double;
}

template<int dim, typename Number>
void
Operator<dim, Number>::prescribe_initial_velocity(VectorType & velocity, double const time) const
{
  // This is necessary if Number == float
  typedef LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

  VectorTypeDouble src_double;
  src_double = velocity;

  field_functions->initial_velocity->set_time(time);
  VectorTools::interpolate(dof_handler, *field_functions->initial_velocity, src_double);

  velocity = src_double;
}

template<int dim, typename Number>
void
Operator<dim, Number>::compute_initial_acceleration(VectorType &       acceleration,
                                                    VectorType const & displacement,
                                                    double const       time) const
{
  VectorType rhs(acceleration);
  rhs = 0.0;

  if(param.large_deformation) // nonlinear case
  {
    // elasticity operator
    elasticity_operator_nonlinear.set_time(time);
    // NB: we have to deactivate the mass matrix term
    elasticity_operator_nonlinear.set_scaling_factor_mass(0.0);
    // evaluate nonlinear operator including Neumann BCs
    elasticity_operator_nonlinear.evaluate_nonlinear(rhs, displacement);
    // shift to right-hand side
    rhs *= -1.0;

    // body forces
    if(param.body_force)
    {
      body_force_operator.evaluate_add(rhs, displacement, time);
    }
  }
  else // linear case
  {
    // elasticity operator
    elasticity_operator_linear.set_time(time);
    // NB: we have to deactivate the mass matrix term
    elasticity_operator_linear.set_scaling_factor_mass(0.0);

    // compute action of homogeneous operator
    elasticity_operator_linear.apply(rhs, displacement);
    // shift to right-hand side
    rhs *= -1.0;

    // Neumann BCs and inhomogeneous Dirichlet BCs
    // (has already the correction sign, since rhs_add())
    elasticity_operator_linear.rhs_add(rhs);

    // body force
    if(param.body_force)
    {
      // displacement is irrelevant for linear problem, since
      // pull_back_body_force = false in this case.
      body_force_operator.evaluate_add(rhs, displacement, time);
    }
  }

  // invert mass matrix to get acceleration
  mass_solver->solve(acceleration, rhs, false);
}

template<int dim, typename Number>
void
Operator<dim, Number>::apply_mass_matrix(VectorType & dst, VectorType const & src) const
{
  mass.apply(dst, src);
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
  elasticity_operator_linear.set_time(time);
  elasticity_operator_linear.rhs_add(dst);
}

template<int dim, typename Number>
void
Operator<dim, Number>::evaluate_nonlinear_residual(VectorType &       dst,
                                                   VectorType const & src,
                                                   VectorType const & const_vector,
                                                   double const       factor,
                                                   double const       time) const
{
  // elasticity operator
  elasticity_operator_nonlinear.set_scaling_factor_mass(factor);
  elasticity_operator_nonlinear.set_time(time);
  elasticity_operator_nonlinear.evaluate_nonlinear(dst, src);

  // dynamic problems
  if(param.problem_type == ProblemType::Unsteady)
  {
    dst.add(1.0, const_vector);
  }

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
                                                 double const       factor,
                                                 double const       time) const
{
  elasticity_operator_nonlinear.set_scaling_factor_mass(factor);
  elasticity_operator_nonlinear.set_time(time);
  elasticity_operator_nonlinear.vmult(dst, src);
}

template<int dim, typename Number>
void
Operator<dim, Number>::set_constrained_values_to_zero(VectorType & vector) const
{
  if(param.large_deformation)
    elasticity_operator_nonlinear.set_constrained_values_to_zero(vector);
  else
    elasticity_operator_linear.set_constrained_values_to_zero(vector);
}

template<int dim, typename Number>
std::tuple<unsigned int, unsigned int>
Operator<dim, Number>::solve_nonlinear(VectorType &       sol,
                                       VectorType const & rhs,
                                       double const       factor,
                                       double const       time,
                                       bool const         update_preconditioner)
{
  // update operators
  residual_operator.update(rhs, factor, time);
  linearized_operator.update(factor, time);

  // set inhomogeneous Dirichlet values
  elasticity_operator_nonlinear.set_dirichlet_values_continuous(sol, time);

  // call Newton solver
  Newton::UpdateData update;
  update.do_update             = update_preconditioner;
  update.threshold_newton_iter = param.update_preconditioner_every_newton_iterations;

  // solve nonlinear problem
  auto const iter = newton_solver->solve(sol, update);

  // set inhomogeneous Dirichlet values
  elasticity_operator_nonlinear.set_dirichlet_values_continuous(sol, time);

  return iter;
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::solve_linear(VectorType &       sol,
                                    VectorType const & rhs,
                                    double const       factor,
                                    double const       time)
{
  // unsteady problems
  elasticity_operator_linear.set_scaling_factor_mass(factor);
  elasticity_operator_linear.set_time(time);

  // solve linear system of equations
  unsigned int const iterations = linear_solver->solve(sol, rhs, false);

  // set Dirichlet values
  elasticity_operator_linear.set_dirichlet_values_continuous(sol, time);

  return iterations;
}

template<int dim, typename Number>
MatrixFree<dim, Number> const &
Operator<dim, Number>::get_matrix_free() const
{
  return *matrix_free;
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

template<int dim, typename Number>
types::global_dof_index
Operator<dim, Number>::get_number_of_dofs() const
{
  return dof_handler.n_dofs();
}

template class Operator<2, float>;
template class Operator<2, double>;

template class Operator<3, float>;
template class Operator<3, double>;

} // namespace Structure
