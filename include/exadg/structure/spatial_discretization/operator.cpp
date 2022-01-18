/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// deal.II
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_amg.h>
#include <exadg/solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h>
#include <exadg/structure/preconditioners/amg_preconditioner.h>
#include <exadg/structure/preconditioners/multigrid_preconditioner.h>
#include <exadg/structure/spatial_discretization/operator.h>

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

template<int dim, typename Number>
Operator<dim, Number>::Operator(
  std::shared_ptr<Grid<dim, Number> const>       grid_in,
  std::shared_ptr<BoundaryDescriptor<dim> const> boundary_descriptor_in,
  std::shared_ptr<FieldFunctions<dim> const>     field_functions_in,
  std::shared_ptr<MaterialDescriptor const>      material_descriptor_in,
  Parameters const &                             param_in,
  std::string const &                            field_in,
  MPI_Comm const &                               mpi_comm_in)
  : dealii::Subscriptor(),
    grid(grid_in),
    boundary_descriptor(boundary_descriptor_in),
    field_functions(field_functions_in),
    material_descriptor(material_descriptor_in),
    param(param_in),
    field(field_in),
    fe(FE_Q<dim>(param_in.degree), dim),
    dof_handler(*grid_in->triangulation),
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
  pcout << std::endl << "Setup elasticity operator ..." << std::endl;

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

  // affine constraints
  affine_constraints.clear();

  // standard Dirichlet boundaries
  for(auto it : this->boundary_descriptor->dirichlet_bc)
  {
    ComponentMask mask =
      this->boundary_descriptor->dirichlet_bc_component_mask.find(it.first)->second;

    DoFTools::make_zero_boundary_constraints(this->dof_handler, it.first, affine_constraints, mask);
  }

  // mortar type Dirichlet boundaries
  for(auto it : this->boundary_descriptor->dirichlet_mortar_bc)
  {
    ComponentMask mask = ComponentMask();
    DoFTools::make_zero_boundary_constraints(dof_handler, it.first, affine_constraints, mask);
  }

  affine_constraints.close();

  // no constraints for mass operator
  constraints_mass.clear();
  constraints_mass.close();

  pcout << std::endl
        << "Continuous Galerkin finite element discretization:" << std::endl
        << std::endl;

  print_parameter(pcout, "degree of 1D polynomials", param.degree);
  print_parameter(pcout, "number of dofs per cell", Utilities::pow(param.degree + 1, dim));
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
Operator<dim, Number>::get_dof_name_mass() const
{
  return field + "_" + dof_index_mass;
}

template<int dim, typename Number>
std::string
Operator<dim, Number>::get_quad_name() const
{
  return field + "_" + quad_index;
}

template<int dim, typename Number>
std::string
Operator<dim, Number>::get_quad_gauss_lobatto_name() const
{
  return field + "_" + quad_index_gauss_lobatto;
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::get_dof_index() const
{
  return matrix_free_data->get_dof_index(get_dof_name());
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::get_dof_index_mass() const
{
  return matrix_free_data->get_dof_index(get_dof_name_mass());
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::get_quad_index() const
{
  return matrix_free_data->get_quad_index(get_quad_name());
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::get_quad_index_gauss_lobatto() const
{
  return matrix_free_data->get_quad_index(get_quad_gauss_lobatto_name());
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
  matrix_free_data.insert_constraint(&affine_constraints, get_dof_name());

  if(param.problem_type == ProblemType::Unsteady)
  {
    matrix_free_data.insert_dof_handler(&dof_handler, get_dof_name_mass());
    matrix_free_data.insert_constraint(&constraints_mass, get_dof_name_mass());
  }

  // Quadrature
  matrix_free_data.insert_quadrature(QGauss<1>(param.degree + 1), get_quad_name());

  // In order to set constrained degrees of freedom for continuous Galerkin
  // discretizations with Dirichlet mortar boundary conditions, a Gauss-Lobatto
  // quadrature rule has to be constructed for the mortar type boundary conditions
  // (so that the values stored in the mortar boundary condition can be directly
  // injected into the DoF vector)
  if(not(boundary_descriptor->dirichlet_mortar_bc.empty()))
  {
    matrix_free_data.insert_quadrature(QGaussLobatto<1>(param.degree + 1),
                                       get_quad_gauss_lobatto_name());
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::setup_operators()
{
  // elasticity operator
  operator_data.dof_index  = get_dof_index();
  operator_data.quad_index = get_quad_index();
  if(not(boundary_descriptor->dirichlet_mortar_bc.empty()))
    operator_data.quad_index_gauss_lobatto = get_quad_index_gauss_lobatto();
  operator_data.bc                  = boundary_descriptor;
  operator_data.material_descriptor = material_descriptor;
  operator_data.n_q_points_1d       = param.degree + 1;
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
    elasticity_operator_nonlinear.initialize(*matrix_free, affine_constraints, operator_data);
  }
  else
  {
    elasticity_operator_linear.initialize(*matrix_free, affine_constraints, operator_data);
  }

  // mass operator and related solver for inversion
  if(param.problem_type == ProblemType::Unsteady)
  {
    MassOperatorData<dim> mass_data;
    mass_data.dof_index  = get_dof_index_mass();
    mass_data.quad_index = get_quad_index();
    mass_operator.initialize(*matrix_free, constraints_mass, mass_data);

    mass_operator.set_scaling_factor(param.density);

    // preconditioner and solver for mass operator have to be initialized in
    // setup_operators() since the mass solver is already needed in
    // setup() function of time integration scheme.

    // preconditioner
    mass_preconditioner =
      std::make_shared<JacobiPreconditioner<MassOperator<dim, dim, Number>>>(mass_operator);

    // initialize solver
    Krylov::SolverDataCG solver_data;
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

    typedef Krylov::SolverCG<MassOperator<dim, dim, Number>, PreconditionerBase<Number>, VectorType>
      CG;
    mass_solver = std::make_shared<CG>(mass_operator, *mass_preconditioner, solver_data);
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
  body_force_operator.initialize(*matrix_free, body_force_data);
}

template<int dim, typename Number>
void
Operator<dim, Number>::setup_solver()
{
  pcout << std::endl << "Setup elasticity solver ..." << std::endl;

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
      preconditioner = std::make_shared<JacobiPreconditioner<NonLinearOperator<dim, Number>>>(
        elasticity_operator_nonlinear);
    }
    else
    {
      preconditioner = std::make_shared<JacobiPreconditioner<LinearOperator<dim, Number>>>(
        elasticity_operator_linear);
    }
  }
  else if(param.preconditioner == Preconditioner::Multigrid)
  {
    if(param.large_deformation)
    {
      typedef MultigridPreconditioner<dim, Number> Multigrid;

      preconditioner = std::make_shared<Multigrid>(mpi_comm);
      std::shared_ptr<Multigrid> mg_preconditioner =
        std::dynamic_pointer_cast<Multigrid>(preconditioner);

      mg_preconditioner->initialize(param.multigrid_data,
                                    &dof_handler.get_triangulation(),
                                    dof_handler.get_fe(),
                                    grid->mapping,
                                    elasticity_operator_nonlinear,
                                    true,
                                    &elasticity_operator_nonlinear.get_data().bc->dirichlet_bc,
                                    &grid->periodic_faces);
    }
    else
    {
      typedef MultigridPreconditioner<dim, Number> Multigrid;

      preconditioner = std::make_shared<Multigrid>(mpi_comm);
      std::shared_ptr<Multigrid> mg_preconditioner =
        std::dynamic_pointer_cast<Multigrid>(preconditioner);

      mg_preconditioner->initialize(param.multigrid_data,
                                    &dof_handler.get_triangulation(),
                                    dof_handler.get_fe(),
                                    grid->mapping,
                                    elasticity_operator_linear,
                                    false,
                                    &elasticity_operator_linear.get_data().bc->dirichlet_bc,
                                    &grid->periodic_faces);
    }
  }
  else if(param.preconditioner == Preconditioner::AMG)
  {
    if(param.large_deformation)
    {
      typedef PreconditionerAMG<NonLinearOperator<dim, Number>, Number> AMG;
      preconditioner = std::make_shared<AMG>(elasticity_operator_nonlinear,
                                             param.multigrid_data.coarse_problem.amg_data);
    }
    else
    {
      typedef PreconditionerAMG<LinearOperator<dim, Number>, Number> AMG;
      preconditioner = std::make_shared<AMG>(elasticity_operator_linear,
                                             param.multigrid_data.coarse_problem.amg_data);
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
    Krylov::SolverDataCG solver_data;
    solver_data.solver_tolerance_abs = param.solver_data.abs_tol;
    solver_data.solver_tolerance_rel = param.solver_data.rel_tol;
    solver_data.max_iter             = param.solver_data.max_iter;

    if(param.preconditioner != Preconditioner::None)
      solver_data.use_preconditioner = true;

    // initialize solver
    if(param.large_deformation)
    {
      typedef Krylov::
        SolverCG<NonLinearOperator<dim, Number>, PreconditionerBase<Number>, VectorType>
          CG;
      linear_solver =
        std::make_shared<CG>(elasticity_operator_nonlinear, *preconditioner, solver_data);
    }
    else
    {
      typedef Krylov::SolverCG<LinearOperator<dim, Number>, PreconditionerBase<Number>, VectorType>
        CG;
      linear_solver =
        std::make_shared<CG>(elasticity_operator_linear, *preconditioner, solver_data);
    }
  }
  else if(param.solver == Solver::FGMRES)
  {
    // initialize solver_data
    Krylov::SolverDataFGMRES solver_data;
    solver_data.solver_tolerance_abs = param.solver_data.abs_tol;
    solver_data.solver_tolerance_rel = param.solver_data.rel_tol;
    solver_data.max_iter             = param.solver_data.max_iter;
    solver_data.max_n_tmp_vectors    = param.solver_data.max_krylov_size;

    if(param.preconditioner != Preconditioner::None)
      solver_data.use_preconditioner = true;

    // initialize solver
    if(param.large_deformation)
    {
      typedef Krylov::
        SolverFGMRES<NonLinearOperator<dim, Number>, PreconditionerBase<Number>, VectorType>
          FGMRES;
      linear_solver =
        std::make_shared<FGMRES>(elasticity_operator_nonlinear, *preconditioner, solver_data);
    }
    else
    {
      typedef Krylov::
        SolverFGMRES<LinearOperator<dim, Number>, PreconditionerBase<Number>, VectorType>
          FGMRES;
      linear_solver =
        std::make_shared<FGMRES>(elasticity_operator_linear, *preconditioner, solver_data);
    }
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

    newton_solver = std::make_shared<NewtonSolver>(param.newton_solver_data,
                                                   residual_operator,
                                                   linearized_operator,
                                                   *linear_solver);
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
    // NB: we have to deactivate the mass operator term
    elasticity_operator_nonlinear.set_scaling_factor_mass_operator(0.0);
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
    // NB: we have to deactivate the mass operator
    elasticity_operator_linear.set_scaling_factor_mass_operator(0.0);

    // compute action of homogeneous operator
    elasticity_operator_linear.apply(rhs, displacement);
    // shift to right-hand side
    rhs *= -1.0;

    // Neumann BCs and inhomogeneous Dirichlet BCs
    // (has already the correct sign, since rhs_add())
    elasticity_operator_linear.rhs_add(rhs);

    // body force
    if(param.body_force)
    {
      // displacement is irrelevant for linear problem, since
      // pull_back_body_force = false in this case.
      body_force_operator.evaluate_add(rhs, displacement, time);
    }
  }

  // invert mass operator to get acceleration
  mass_solver->solve(acceleration, rhs, false);
}

template<int dim, typename Number>
void
Operator<dim, Number>::apply_mass_operator(VectorType & dst, VectorType const & src) const
{
  mass_operator.apply(dst, src);
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
  elasticity_operator_nonlinear.set_scaling_factor_mass_operator(factor);
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
  elasticity_operator_nonlinear.set_scaling_factor_mass_operator(factor);
  elasticity_operator_nonlinear.set_time(time);
  elasticity_operator_nonlinear.vmult(dst, src);
}

template<int dim, typename Number>
void
Operator<dim, Number>::apply_nonlinear_operator(VectorType &       dst,
                                                VectorType const & src,
                                                double const       factor,
                                                double const       time) const
{
  elasticity_operator_nonlinear.set_scaling_factor_mass_operator(factor);
  elasticity_operator_nonlinear.set_time(time);
  elasticity_operator_nonlinear.evaluate_nonlinear(dst, src);
}

template<int dim, typename Number>
void
Operator<dim, Number>::apply_linear_operator(VectorType &       dst,
                                             VectorType const & src,
                                             double const       factor,
                                             double const       time) const
{
  elasticity_operator_linear.set_scaling_factor_mass_operator(factor);
  elasticity_operator_linear.set_time(time);
  elasticity_operator_linear.vmult(dst, src);
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
                                       bool const         update_preconditioner) const
{
  // update operators
  residual_operator.update(rhs, factor, time);
  linearized_operator.update(factor, time);

  // set inhomogeneous Dirichlet values
  elasticity_operator_nonlinear.set_constrained_values(sol, time);

  // call Newton solver
  Newton::UpdateData update;
  update.do_update             = update_preconditioner;
  update.threshold_newton_iter = param.update_preconditioner_every_newton_iterations;

  // solve nonlinear problem
  auto const iter = newton_solver->solve(sol, update);

  // set inhomogeneous Dirichlet values
  elasticity_operator_nonlinear.set_constrained_values(sol, time);

  return iter;
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::solve_linear(VectorType &       sol,
                                    VectorType const & rhs,
                                    double const       factor,
                                    double const       time) const
{
  // unsteady problems
  elasticity_operator_linear.set_scaling_factor_mass_operator(factor);
  elasticity_operator_linear.set_time(time);

  // solve linear system of equations
  unsigned int const iterations = linear_solver->solve(sol, rhs, false);

  // set Dirichlet values
  elasticity_operator_linear.set_constrained_values(sol, time);

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
  return *grid->mapping;
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
} // namespace ExaDG
