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
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/grid/grid_utilities.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_amg.h>
#include <exadg/solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h>
#include <exadg/structure/preconditioners/multigrid_preconditioner.h>
#include <exadg/structure/spatial_discretization/operator.h>
#include <exadg/utilities/exceptions.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
Operator<dim, Number>::Operator(
  std::shared_ptr<Grid<dim> const>               grid_in,
  std::shared_ptr<dealii::Mapping<dim> const>    mapping_in,
  std::shared_ptr<BoundaryDescriptor<dim> const> boundary_descriptor_in,
  std::shared_ptr<FieldFunctions<dim> const>     field_functions_in,
  std::shared_ptr<MaterialDescriptor const>      material_descriptor_in,
  Parameters const &                             param_in,
  std::string const &                            field_in,
  MPI_Comm const &                               mpi_comm_in)
  : dealii::Subscriptor(),
    grid(grid_in),
    mapping(mapping_in),
    boundary_descriptor(boundary_descriptor_in),
    field_functions(field_functions_in),
    material_descriptor(material_descriptor_in),
    param(param_in),
    field(field_in),
    dof_handler(*grid_in->triangulation),
    mpi_comm(mpi_comm_in),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm_in) == 0)
{
  pcout << std::endl << "Construct elasticity operator ..." << std::endl;

  distribute_dofs();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
Operator<dim, Number>::setup(std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free_in,
                             std::shared_ptr<MatrixFreeData<dim, Number> const> matrix_free_data_in)
{
  pcout << std::endl << "Setup elasticity operator ..." << std::endl;

  matrix_free      = matrix_free_in;
  matrix_free_data = matrix_free_data_in;

  if(not(boundary_descriptor->dirichlet_cached_bc.empty()))
  {
    interface_data_dirichlet_cached = std::make_shared<ContainerInterfaceData<1, dim, double>>();
    std::vector<unsigned int> quad_indices;
    // Gauss-Lobatto quadrature rule for DirichletCached boundary conditions!
    quad_indices.emplace_back(get_quad_index_gauss_lobatto());

    interface_data_dirichlet_cached->setup(*matrix_free,
                                           get_dof_index(),
                                           quad_indices,
                                           boundary_descriptor->dirichlet_cached_bc);

    boundary_descriptor->set_dirichlet_cached_data(interface_data_dirichlet_cached);
  }

  if(not(boundary_descriptor->neumann_cached_bc.empty()))
  {
    interface_data_neumann_cached = std::make_shared<ContainerInterfaceData<1, dim, double>>();
    std::vector<unsigned int> quad_indices;
    quad_indices.emplace_back(get_quad_index());

    interface_data_neumann_cached->setup(*matrix_free,
                                         get_dof_index(),
                                         quad_indices,
                                         boundary_descriptor->neumann_cached_bc);

    boundary_descriptor->set_neumann_cached_data(interface_data_neumann_cached);
  }

  setup_operators();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
Operator<dim, Number>::distribute_dofs()
{
  // setup finite element
  if(this->grid->triangulation->all_reference_cells_are_hyper_cube())
    fe = std::make_shared<dealii::FESystem<dim>>(dealii::FE_Q<dim>(param.degree), dim);
  else if(this->grid->triangulation->all_reference_cells_are_simplex())
    fe = std::make_shared<dealii::FESystem<dim>>(dealii::FE_SimplexP<dim>(param.degree), dim);
  else
    AssertThrow(false, ExcNotImplemented());

  // enumerate degrees of freedom
  dof_handler.distribute_dofs(*fe);

  // The AffineConstraints object is used to initialize MatrixFree. Here, we apply homogeneous
  // boundary conditions as needed by vmult() in iterative solvers for linear(ized) systems of
  // equations, implemented via dealii::MatrixFree and FEEvaluation::read_dof_values() (or
  // gather_evaluate()). The actual inhomogeneous boundary data needs to be imposed separately due
  // to the implementation in deal.II that can not handle multiple AffineConstraints. Hence, we need
  // to do this explicitly in ExaDG.

  // affine constraints
  affine_constraints.clear();

  dealii::IndexSet locally_relevant_dofs;
  dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  affine_constraints.reinit(locally_relevant_dofs);

  // hanging nodes (needs to be done before imposing periodicity constraints and boundary
  // conditions)
  if(this->grid->triangulation->has_hanging_nodes())
    dealii::DoFTools::make_hanging_node_constraints(dof_handler, affine_constraints);

  // constraints from periodic boundary conditions
  if(not(this->grid->periodic_face_pairs.empty()))
  {
    auto periodic_faces_dof = GridUtilities::transform_periodic_face_pairs_to_dof_cell_iterator(
      this->grid->periodic_face_pairs, dof_handler);

    dealii::DoFTools::make_periodicity_constraints<dim, dim, Number>(periodic_faces_dof,
                                                                     affine_constraints);
  }

  // standard Dirichlet boundaries
  for(auto it : this->boundary_descriptor->dirichlet_bc)
  {
    dealii::ComponentMask mask = dealii::ComponentMask();

    auto it_mask = this->boundary_descriptor->dirichlet_bc_component_mask.find(it.first);
    if(it_mask != this->boundary_descriptor->dirichlet_bc_component_mask.end())
      mask = it_mask->second;

    dealii::DoFTools::make_zero_boundary_constraints(this->dof_handler,
                                                     it.first,
                                                     affine_constraints,
                                                     mask);
  }

  // DirichletCached boundaries
  for(auto it : this->boundary_descriptor->dirichlet_cached_bc)
  {
    dealii::ComponentMask mask = dealii::ComponentMask();
    dealii::DoFTools::make_zero_boundary_constraints(dof_handler, it, affine_constraints, mask);
  }

  affine_constraints.close();

  // affine constraints for mass operator
  if(param.problem_type == ProblemType::Unsteady)
  {
    constraints_mass.clear();

    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    constraints_mass.reinit(locally_relevant_dofs);

    // hanging nodes (needs to be done before imposing periodicity constraints and boundary
    // conditions)
    if(this->grid->triangulation->has_hanging_nodes())
      dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints_mass);

    // constraints from periodic boundary conditions
    if(not(this->grid->periodic_face_pairs.empty()))
    {
      auto periodic_faces_dof = GridUtilities::transform_periodic_face_pairs_to_dof_cell_iterator(
        this->grid->periodic_face_pairs, dof_handler);

      dealii::DoFTools::make_periodicity_constraints<dim, dim, Number>(periodic_faces_dof,
                                                                       constraints_mass);
    }

    // no constraints from Dirichlet boundary conditions for mass operator

    constraints_mass.close();
  }

  pcout << std::endl
        << "Continuous Galerkin finite element discretization:" << std::endl
        << std::endl;

  print_parameter(pcout, "degree of 1D polynomials", param.degree);
  print_parameter(pcout, "number of dofs per cell", fe->n_dofs_per_cell());
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
double
Operator<dim, Number>::compute_scaling_factor_mass(
  double const scaling_factor_mass_from_acceleration,
  double const scaling_factor_mass_from_velocity) const
{
  double scaling_factor_mass = scaling_factor_mass_from_acceleration;
  if(param.weak_damping_active)
  {
    scaling_factor_mass += param.weak_damping_coefficient * scaling_factor_mass_from_velocity;
  }
  return scaling_factor_mass;
}

template<int dim, typename Number>
std::shared_ptr<ContainerInterfaceData<1, dim, double>>
Operator<dim, Number>::get_container_interface_data_neumann() const
{
  return interface_data_neumann_cached;
}

template<int dim, typename Number>
std::shared_ptr<ContainerInterfaceData<1, dim, double>>
Operator<dim, Number>::get_container_interface_data_dirichlet() const
{
  return interface_data_dirichlet_cached;
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

  // dealii::DoFHandler, dealii::AffineConstraints
  matrix_free_data.insert_dof_handler(&dof_handler, get_dof_name());
  matrix_free_data.insert_constraint(&affine_constraints, get_dof_name());

  if(param.problem_type == ProblemType::Unsteady)
  {
    matrix_free_data.insert_dof_handler(&dof_handler, get_dof_name_mass());
    matrix_free_data.insert_constraint(&constraints_mass, get_dof_name_mass());
  }

  // dealii::Quadrature
  if(this->grid->triangulation->all_reference_cells_are_hyper_cube())
    matrix_free_data.insert_quadrature(dealii::QGauss<1>(param.degree + 1), get_quad_name());
  else if(this->grid->triangulation->all_reference_cells_are_simplex())
    matrix_free_data.insert_quadrature(dealii::QGaussSimplex<dim>(param.degree + 1),
                                       get_quad_name());
  else
    AssertThrow(false, ExcNotImplemented());

  // Create a Gauss-Lobatto quadrature rule for DirichletCached boundary conditions.
  // These quadrature points coincide with the nodes of the discretization, so that
  // the values stored in the DirichletCached boundary condition can be directly
  // injected into the DoF vector. This allows to set constrained degrees of freedom
  // in case of continuous Galerkin discretizations with DirichletCached boundary
  // conditions.
  if(not(boundary_descriptor->dirichlet_cached_bc.empty()))
  {
    AssertThrow(this->grid->triangulation->all_reference_cells_are_hyper_cube(),
                ExcNotImplemented());

    matrix_free_data.insert_quadrature(dealii::QGaussLobatto<1>(param.degree + 1),
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
  if(not(boundary_descriptor->dirichlet_cached_bc.empty()))
  {
    AssertThrow(this->grid->triangulation->all_reference_cells_are_hyper_cube(),
                ExcNotImplemented());

    operator_data.quad_index_gauss_lobatto = get_quad_index_gauss_lobatto();
  }
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
Operator<dim, Number>::setup_solver(double const & scaling_factor_mass_from_acceleration,
                                    double const & scaling_factor_mass_from_velocity)
{
  pcout << std::endl << "Setup elasticity solver ..." << std::endl;

  double const scaling_factor_mass =
    compute_scaling_factor_mass(scaling_factor_mass_from_acceleration,
                                scaling_factor_mass_from_velocity);
  if(param.large_deformation)
    elasticity_operator_nonlinear.set_scaling_factor_mass_operator(scaling_factor_mass);
  else
    elasticity_operator_linear.set_scaling_factor_mass_operator(scaling_factor_mass);

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

      std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        dirichlet_boundary_conditions = elasticity_operator_nonlinear.get_data().bc->dirichlet_bc;

      typedef std::map<dealii::types::boundary_id, dealii::ComponentMask> Map_DBC_ComponentMask;
      Map_DBC_ComponentMask dirichlet_bc_component_mask =
        elasticity_operator_nonlinear.get_data().bc->dirichlet_bc_component_mask;

      // We also need to add DirichletCached boundary conditions. From the
      // perspective of multigrid, there is no difference between standard
      // and cached Dirichlet BCs. Since multigrid does not need information
      // about inhomogeneous boundary data, we simply fill the map with
      // dealii::Functions::ZeroFunction for DirichletCached BCs.
      for(auto iter : elasticity_operator_nonlinear.get_data().bc->dirichlet_cached_bc)
      {
        typedef
          typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
            pair;

        dirichlet_boundary_conditions.insert(
          pair(iter, new dealii::Functions::ZeroFunction<dim>(dim)));

        typedef typename std::pair<dealii::types::boundary_id, dealii::ComponentMask> pair_mask;

        std::vector<bool> default_mask = std::vector<bool>(dim, true);
        dirichlet_bc_component_mask.insert(pair_mask(iter, default_mask));
      }

      mg_preconditioner->initialize(param.multigrid_data,
                                    param.grid.multigrid,
                                    grid,
                                    mapping,
                                    dof_handler.get_fe(),
                                    elasticity_operator_nonlinear,
                                    param.large_deformation,
                                    dirichlet_boundary_conditions,
                                    dirichlet_bc_component_mask);
    }
    else
    {
      typedef MultigridPreconditioner<dim, Number> Multigrid;

      preconditioner = std::make_shared<Multigrid>(mpi_comm);
      std::shared_ptr<Multigrid> mg_preconditioner =
        std::dynamic_pointer_cast<Multigrid>(preconditioner);

      std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        dirichlet_boundary_conditions = elasticity_operator_linear.get_data().bc->dirichlet_bc;

      typedef std::map<dealii::types::boundary_id, dealii::ComponentMask> Map_DBC_ComponentMask;
      Map_DBC_ComponentMask dirichlet_bc_component_mask =
        elasticity_operator_linear.get_data().bc->dirichlet_bc_component_mask;

      // We also need to add DirichletCached boundary conditions. From the
      // perspective of multigrid, there is no difference between standard
      // and cached Dirichlet BCs. Since multigrid does not need information
      // about inhomogeneous boundary data, we simply fill the map with
      // dealii::Functions::ZeroFunction for DirichletCached BCs.
      for(auto iter : elasticity_operator_linear.get_data().bc->dirichlet_cached_bc)
      {
        typedef
          typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
            pair;

        dirichlet_boundary_conditions.insert(
          pair(iter, new dealii::Functions::ZeroFunction<dim>(dim)));

        typedef typename std::pair<dealii::types::boundary_id, dealii::ComponentMask> pair_mask;

        std::vector<bool> default_mask = std::vector<bool>(dim, true);
        dirichlet_bc_component_mask.insert(pair_mask(iter, default_mask));
      }

      mg_preconditioner->initialize(param.multigrid_data,
                                    param.grid.multigrid,
                                    grid,
                                    mapping,
                                    dof_handler.get_fe(),
                                    elasticity_operator_linear,
                                    param.large_deformation,
                                    dirichlet_boundary_conditions,
                                    dirichlet_bc_component_mask);
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
    AssertThrow(false, dealii::ExcMessage("Specified preconditioner is not implemented!"));
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
    AssertThrow(false, dealii::ExcMessage("Specified solver is not implemented!"));
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
  typedef dealii::LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

  VectorTypeDouble src_double;
  src_double = displacement;

  field_functions->initial_displacement->set_time(time);
  dealii::VectorTools::interpolate(dof_handler, *field_functions->initial_displacement, src_double);

  displacement = src_double;
}

template<int dim, typename Number>
void
Operator<dim, Number>::prescribe_initial_velocity(VectorType & velocity, double const time) const
{
  // This is necessary if Number == float
  typedef dealii::LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

  VectorTypeDouble src_double;
  src_double = velocity;

  field_functions->initial_velocity->set_time(time);
  dealii::VectorTools::interpolate(dof_handler, *field_functions->initial_velocity, src_double);

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
    double const scaling_factor_mass =
      elasticity_operator_nonlinear.get_scaling_factor_mass_operator();
    elasticity_operator_nonlinear.set_scaling_factor_mass_operator(0.0);

    // evaluate nonlinear operator including Neumann BCs
    elasticity_operator_nonlinear.evaluate_nonlinear(rhs, displacement);
    // shift to right-hand side
    rhs *= -1.0;

    // revert scaling factor to initialized value
    elasticity_operator_nonlinear.set_scaling_factor_mass_operator(scaling_factor_mass);

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
    double const scaling_factor_mass =
      elasticity_operator_linear.get_scaling_factor_mass_operator();
    elasticity_operator_linear.set_scaling_factor_mass_operator(0.0);

    // compute action of homogeneous operator
    elasticity_operator_linear.apply(rhs, displacement);
    // shift to right-hand side
    rhs *= -1.0;

    // revert scaling factor to initialized value
    elasticity_operator_linear.set_scaling_factor_mass_operator(scaling_factor_mass);

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
  mass_solver->solve(acceleration, rhs);
}

template<int dim, typename Number>
void
Operator<dim, Number>::apply_mass_operator(VectorType & dst, VectorType const & src) const
{
  mass_operator.apply(dst, src);
}

template<int dim, typename Number>
void
Operator<dim, Number>::apply_add_weak_damping_operator(VectorType & dst, VectorType & src) const
{
  src *= param.weak_damping_coefficient;
  mass_operator.apply_add(dst, src);
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
  // elasticity operator: make sure that constrained degrees of freedom have been set correctly
  // before evaluating the elasticity operator.
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

  // To ensure convergence of the Newton solver, the residual has to be zero
  // for constrained degrees of freedom as well, which might not be the case
  // in general, e.g. due to const_vector. Hence, we set the constrained
  // degrees of freedom explicitly to zero.
  elasticity_operator_nonlinear.set_constrained_values_to_zero(dst);
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
std::tuple<unsigned int, unsigned int>
Operator<dim, Number>::solve_nonlinear(VectorType &       sol,
                                       VectorType const & rhs,
                                       double const       scaling_factor_mass_from_acceleration,
                                       double const       scaling_factor_mass_from_velocity,
                                       double const       time,
                                       bool const         update_preconditioner) const
{
  // update operators
  double const scaling_factor_mass =
    compute_scaling_factor_mass(scaling_factor_mass_from_acceleration,
                                scaling_factor_mass_from_velocity);
  residual_operator.update(rhs, scaling_factor_mass, time);
  linearized_operator.update(scaling_factor_mass, time);

  // set inhomogeneous Dirichlet values (this is necessary since we use
  // FEEvaluation::read_dof_values_plain() to evaluate the operator)
  elasticity_operator_nonlinear.set_constrained_values(sol, time);

  // call Newton solver
  Newton::UpdateData update;
  update.do_update                = update_preconditioner;
  update.update_every_newton_iter = param.update_preconditioner_every_newton_iterations;
  update.update_once_converged    = param.update_preconditioner_once_newton_converged;

  // solve nonlinear problem
  auto const iter = newton_solver->solve(sol, update);

  // This step should actually be optional: The constraints have already been set
  // before the nonlinear solver is called and no contributions to the constrained
  // degrees of freedom should be added in the Newton solver by the linearized solver
  // (because the residual vector forming the rhs of the linearized problem is zero
  // for constrained degrees of freedom, the initial solution of the linearized
  // solver is also zero, and the linearized operator contains values of 1 on the
  // diagonal for constrained degrees of freedom).
  elasticity_operator_nonlinear.set_constrained_values(sol, time);

  return iter;
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::solve_linear(VectorType &       sol,
                                    VectorType const & rhs,
                                    double const       scaling_factor_mass_from_acceleration,
                                    double const       scaling_factor_mass_from_velocity,
                                    double const       time,
                                    bool const         update_preconditioner) const
{
  // unsteady problems
  double const scaling_factor_mass =
    compute_scaling_factor_mass(scaling_factor_mass_from_acceleration,
                                scaling_factor_mass_from_velocity);
  elasticity_operator_linear.set_scaling_factor_mass_operator(scaling_factor_mass);
  elasticity_operator_linear.set_time(time);

  linear_solver->update_preconditioner(update_preconditioner);

  // Set constrained degrees of freedom of rhs vector according to the prescribed
  // Dirichlet boundary conditions.
  VectorType & rhs_mutable = const_cast<VectorType &>(rhs);
  elasticity_operator_linear.set_constrained_values(rhs_mutable, time);

  // solve linear system of equations
  unsigned int const iterations = linear_solver->solve(sol, rhs_mutable);

  // This step should actually be optional: The constrained degrees of freedom of the
  // rhs vector contain the Dirichlet boundary values and the linear operator contains
  // values of 1 on the diagonal. Hence, sol should already contain the correct
  // Dirichlet boundary values for constrained degrees of freedom.
  elasticity_operator_linear.set_constrained_values(sol, time);

  return iterations;
}

template<int dim, typename Number>
dealii::MatrixFree<dim, Number> const &
Operator<dim, Number>::get_matrix_free() const
{
  return *matrix_free;
}

template<int dim, typename Number>
dealii::Mapping<dim> const &
Operator<dim, Number>::get_mapping() const
{
  return *mapping;
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
Operator<dim, Number>::get_dof_handler() const
{
  return dof_handler;
}

template<int dim, typename Number>
dealii::types::global_dof_index
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
