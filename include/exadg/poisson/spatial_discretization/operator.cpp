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
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/grid/grid_data.h>
#include <exadg/operators/constraints.h>
#include <exadg/operators/finite_element.h>
#include <exadg/operators/quadrature.h>
#include <exadg/poisson/preconditioners/multigrid_preconditioner.h>
#include <exadg/poisson/spatial_discretization/operator.h>
#include <exadg/solvers_and_preconditioners/preconditioners/block_jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/inverse_mass_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_amg.h>
#include <exadg/solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h>
#include <exadg/solvers_and_preconditioners/utilities/check_multigrid.h>
#include <exadg/solvers_and_preconditioners/utilities/petsc_operation.h>
#include <exadg/utilities/exceptions.h>

namespace ExaDG
{
namespace Poisson
{
template<int dim, int n_components, typename Number>
Operator<dim, n_components, Number>::Operator(
  std::shared_ptr<Grid<dim> const>                     grid_in,
  std::shared_ptr<dealii::Mapping<dim> const>          mapping_in,
  std::shared_ptr<BoundaryDescriptor<rank, dim> const> boundary_descriptor_in,
  std::shared_ptr<FieldFunctions<dim> const>           field_functions_in,
  Parameters const &                                   param_in,
  std::string const &                                  field_in,
  MPI_Comm const &                                     mpi_comm_in)
  : dealii::Subscriptor(),
    grid(grid_in),
    mapping(mapping_in),
    boundary_descriptor(boundary_descriptor_in),
    field_functions(field_functions_in),
    param(param_in),
    field(field_in),
    dof_handler(*grid_in->triangulation),
    mpi_comm(mpi_comm_in),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm_in) == 0)
{
  pcout << std::endl << "Construct Poisson operator ..." << std::endl;

  distribute_dofs();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int n_components, typename Number>
void
Operator<dim, n_components, Number>::distribute_dofs()
{
  fe = create_finite_element<dim>(param.grid.element_type,
                                  param.spatial_discretization == SpatialDiscretization::DG,
                                  n_components,
                                  param.degree);

  dof_handler.distribute_dofs(*fe);

  // Affine constraints are only relevant for continuous Galerkin discretizations.
  if(param.spatial_discretization == SpatialDiscretization::CG)
  {
    affine_constraints_periodicity_and_hanging_nodes.clear();

    add_hanging_node_and_periodicity_constraints(affine_constraints_periodicity_and_hanging_nodes,
                                                 *this->grid,
                                                 dof_handler);

    affine_constraints_periodicity_and_hanging_nodes.close();

    // copy periodicity and hanging node constraints, and add further constraints stemming from
    // Dirichlet boundary conditions
    affine_constraints.clear();

    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    affine_constraints.reinit(locally_relevant_dofs);

    affine_constraints.copy_from(affine_constraints_periodicity_and_hanging_nodes);

    // use all the component masks defined by the user
    std::map<dealii::types::boundary_id, dealii::ComponentMask> map_bid_to_mask =
      boundary_descriptor->dirichlet_bc_component_mask;

    // collect all Dirichlet boundary IDs in a set:
    // DirichletCached boundary IDs are already provided as a set
    std::set<dealii::types::boundary_id> all_dirichlet_bids =
      boundary_descriptor->dirichlet_cached_bc;

    // standard Dirichlet boundaries: extract keys from map
    fill_keys_of_map_into_set(all_dirichlet_bids, boundary_descriptor->dirichlet_bc);

    // fill with default mask if no mask has been defined
    fill_map_bid_to_mask_with_default_mask(map_bid_to_mask, all_dirichlet_bids);

    // call deal.II utility function to add Dirichlet constraints
    add_homogeneous_dirichlet_constraints(affine_constraints, dof_handler, map_bid_to_mask);

    affine_constraints.close();
  }

  pcout << std::endl;

  if(param.spatial_discretization == SpatialDiscretization::DG)
    pcout << std::endl
          << "Discontinuous Galerkin finite element discretization:" << std::endl
          << std::endl;
  else if(param.spatial_discretization == SpatialDiscretization::CG)
    pcout << std::endl
          << "Continuous Galerkin finite element discretization:" << std::endl
          << std::endl;
  else
    AssertThrow(false, dealii::ExcMessage("Not implemented."));

  print_parameter(pcout, "degree of 1D polynomials", param.degree);
  print_parameter(pcout, "number of dofs per cell", fe->n_dofs_per_cell());
  print_parameter(pcout, "number of dofs (total)", dof_handler.n_dofs());
}

template<int dim, int n_components, typename Number>
void
Operator<dim, n_components, Number>::fill_matrix_free_data(
  MatrixFreeData<dim, Number> & matrix_free_data) const
{
  // append mapping flags

  // for continuous FE discretizations, we need to evaluate inhomogeneous Neumann
  // boundary conditions or set constrained Dirichlet values, which is why the
  // second argument is always true
  matrix_free_data.append_mapping_flags(
    Operators::LaplaceKernel<dim, Number, n_components>::get_mapping_flags(
      param.spatial_discretization == SpatialDiscretization::DG, true));

  if(param.right_hand_side)
  {
    matrix_free_data.append_mapping_flags(
      ExaDG::Operators::RHSKernel<dim, Number, n_components>::get_mapping_flags());
  }

  matrix_free_data.insert_dof_handler(&dof_handler, get_dof_name());
  matrix_free_data.insert_constraint(&affine_constraints, get_dof_name());

  // inhomogeneous Dirichlet boundary conditions: use additional AffineConstraints object, but the
  // same DoFHandler
  matrix_free_data.insert_dof_handler(&dof_handler,
                                      get_dof_name_periodicity_and_hanging_node_constraints());
  matrix_free_data.insert_constraint(&affine_constraints_periodicity_and_hanging_nodes,
                                     get_dof_name_periodicity_and_hanging_node_constraints());

  std::shared_ptr<dealii::Quadrature<dim>> quadrature =
    create_quadrature<dim>(param.grid.element_type, param.degree + 1);
  matrix_free_data.insert_quadrature(*quadrature, get_quad_name());

  // Create a Gauss-Lobatto quadrature rule for DirichletCached boundary conditions.
  // These quadrature points coincide with the nodes of the discretization, so that
  // the values stored in the DirichletCached boundary condition can be directly
  // injected into the DoF vector. This allows to set constrained degrees of freedom
  // in case of continuous Galerkin discretizations with DirichletCached boundary
  // conditions. This is not needed in case of discontinuous Galerkin discretizations
  // where boundary conditions are imposed weakly via integrals over the domain
  // boundaries.
  if(param.spatial_discretization == SpatialDiscretization::CG and
     not(boundary_descriptor->dirichlet_cached_bc.empty()))
  {
    AssertThrow(this->grid->triangulation->all_reference_cells_are_hyper_cube(),
                ExcNotImplemented());

    matrix_free_data.insert_quadrature(dealii::QGaussLobatto<1>(param.degree + 1),
                                       get_quad_gauss_lobatto_name());
  }
}

template<int dim, int n_components, typename Number>
void
Operator<dim, n_components, Number>::setup_operators()
{
  // Laplace operator
  Poisson::LaplaceOperatorData<rank, dim> laplace_operator_data;
  laplace_operator_data.dof_index = get_dof_index();
  if(param.spatial_discretization == SpatialDiscretization::CG)
  {
    laplace_operator_data.dof_index_inhomogeneous =
      get_dof_index_periodicity_and_hanging_node_constraints();
  }
  laplace_operator_data.quad_index = get_quad_index();
  if(param.spatial_discretization == SpatialDiscretization::CG and
     not(boundary_descriptor->dirichlet_cached_bc.empty()))
  {
    AssertThrow(this->grid->triangulation->all_reference_cells_are_hyper_cube(),
                ExcNotImplemented());

    laplace_operator_data.quad_index_gauss_lobatto = get_quad_index_gauss_lobatto();
  }
  laplace_operator_data.bc                    = boundary_descriptor;
  laplace_operator_data.use_cell_based_loops  = param.enable_cell_based_face_loops;
  laplace_operator_data.kernel_data.IP_factor = param.IP_factor;
  laplace_operator.initialize(*matrix_free, affine_constraints, laplace_operator_data);

  // rhs operator
  if(param.right_hand_side)
  {
    RHSOperatorData<dim> rhs_operator_data;
    rhs_operator_data.dof_index     = get_dof_index();
    rhs_operator_data.quad_index    = get_quad_index();
    rhs_operator_data.kernel_data.f = field_functions->right_hand_side;
    rhs_operator.initialize(*matrix_free, rhs_operator_data);
  }
}

template<int dim, int n_components, typename Number>
void
Operator<dim, n_components, Number>::setup(
  std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free_in,
  std::shared_ptr<MatrixFreeData<dim, Number> const>     matrix_free_data_in)
{
  pcout << std::endl << "Setup Poisson operator ..." << std::endl;

  matrix_free      = matrix_free_in;
  matrix_free_data = matrix_free_data_in;

  if(not(boundary_descriptor->dirichlet_cached_bc.empty()))
  {
    interface_data_dirichlet_cached = std::make_shared<ContainerInterfaceData<rank, dim, double>>();
    std::vector<unsigned int> quad_indices;
    if(param.spatial_discretization == SpatialDiscretization::DG)
      quad_indices.emplace_back(get_quad_index());
    else if(param.spatial_discretization == SpatialDiscretization::CG)
      quad_indices.emplace_back(get_quad_index_gauss_lobatto());
    else
      AssertThrow(false, dealii::ExcMessage("not implemented."));

    interface_data_dirichlet_cached->setup(*matrix_free,
                                           get_dof_index(),
                                           quad_indices,
                                           boundary_descriptor->dirichlet_cached_bc);

    boundary_descriptor->set_dirichlet_cached_data(interface_data_dirichlet_cached);
  }

  setup_operators();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int n_components, typename Number>
void
Operator<dim, n_components, Number>::setup_solver()
{
  pcout << std::endl << "Setup Poisson solver ..." << std::endl;

  // initialize preconditioner
  if(param.preconditioner == Poisson::Preconditioner::None)
  {
    // do nothing
  }
  else if(param.preconditioner == Poisson::Preconditioner::PointJacobi)
  {
    preconditioner = std::make_shared<JacobiPreconditioner<Laplace>>(laplace_operator);
  }
  else if(param.preconditioner == Poisson::Preconditioner::BlockJacobi)
  {
    preconditioner = std::make_shared<BlockJacobiPreconditioner<Laplace>>(laplace_operator);
  }
  else if(param.preconditioner == Poisson::Preconditioner::AMG)
  {
    preconditioner = std::make_shared<PreconditionerAMG<Laplace, Number>>(
      laplace_operator, param.multigrid_data.coarse_problem.amg_data);
  }
  else if(param.preconditioner == Poisson::Preconditioner::Multigrid)
  {
    MultigridData mg_data;
    mg_data = param.multigrid_data;

    typedef MultigridPreconditioner<dim, Number, n_components> Multigrid;

    preconditioner = std::make_shared<Multigrid>(this->mpi_comm);

    std::shared_ptr<Multigrid> mg_preconditioner =
      std::dynamic_pointer_cast<Multigrid>(preconditioner);

    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      dirichlet_boundary_conditions = laplace_operator.get_data().bc->dirichlet_bc;

    typedef std::map<dealii::types::boundary_id, dealii::ComponentMask> Map_DBC_ComponentMask;
    Map_DBC_ComponentMask dirichlet_bc_component_mask =
      laplace_operator.get_data().bc->dirichlet_bc_component_mask;

    // We also need to add DirichletCached boundary conditions. From the
    // perspective of multigrid, there is no difference between standard
    // and cached Dirichlet BCs. Since multigrid does not need information
    // about inhomogeneous boundary data, we simply fill the map with
    // dealii::Functions::ZeroFunction for DirichletCached BCs.
    for(auto iter : laplace_operator.get_data().bc->dirichlet_cached_bc)
    {
      typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        pair;

      dirichlet_boundary_conditions.insert(
        pair(iter, new dealii::Functions::ZeroFunction<dim>(n_components)));

      typedef typename std::pair<dealii::types::boundary_id, dealii::ComponentMask> pair_mask;

      std::vector<bool> default_mask = std::vector<bool>(n_components, true);
      dirichlet_bc_component_mask.insert(pair_mask(iter, default_mask));
    }

    mg_preconditioner->initialize(mg_data,
                                  grid,
                                  mapping,
                                  dof_handler.get_fe(),
                                  laplace_operator.get_data(),
                                  false /* moving_mesh */,
                                  dirichlet_boundary_conditions,
                                  dirichlet_bc_component_mask);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Specified preconditioner is not implemented!"));
  }

  if(param.solver == LinearSolver::CG)
  {
    // initialize solver_data
    Krylov::SolverDataCG solver_data;
    solver_data.solver_tolerance_abs        = param.solver_data.abs_tol;
    solver_data.solver_tolerance_rel        = param.solver_data.rel_tol;
    solver_data.max_iter                    = param.solver_data.max_iter;
    solver_data.compute_performance_metrics = param.compute_performance_metrics;

    if(param.preconditioner != Poisson::Preconditioner::None)
      solver_data.use_preconditioner = true;

    // initialize solver
    iterative_solver =
      std::make_shared<Krylov::SolverCG<Laplace, PreconditionerBase<Number>, VectorType>>(
        laplace_operator, *preconditioner, solver_data);
  }
  else if(param.solver == LinearSolver::FGMRES)
  {
    // initialize solver_data
    Krylov::SolverDataFGMRES solver_data;
    solver_data.solver_tolerance_abs        = param.solver_data.abs_tol;
    solver_data.solver_tolerance_rel        = param.solver_data.rel_tol;
    solver_data.max_iter                    = param.solver_data.max_iter;
    solver_data.max_n_tmp_vectors           = param.solver_data.max_krylov_size;
    solver_data.compute_performance_metrics = param.compute_performance_metrics;

    if(param.preconditioner != Preconditioner::None)
      solver_data.use_preconditioner = true;

    // initialize solver
    iterative_solver =
      std::make_shared<Krylov::SolverFGMRES<Laplace, PreconditionerBase<Number>, VectorType>>(
        laplace_operator, *preconditioner, solver_data);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Specified solver is not implemented!"));
  }

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int n_components, typename Number>
void
Operator<dim, n_components, Number>::initialize_dof_vector(VectorType & src) const
{
  matrix_free->initialize_dof_vector(src, get_dof_index());
}

template<int dim, int n_components, typename Number>
void
Operator<dim, n_components, Number>::prescribe_initial_conditions(VectorType & src) const
{
  field_functions->initial_solution->set_time(0.0);

  // This is necessary if Number == float
  typedef dealii::LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

  VectorTypeDouble src_double;
  src_double = src;

  dealii::VectorTools::interpolate(dof_handler, *(field_functions->initial_solution), src_double);

  src = src_double;
}

template<int dim, int n_components, typename Number>
void
Operator<dim, n_components, Number>::rhs(VectorType & dst, double const time) const
{
  laplace_operator.set_time(time);
  laplace_operator.rhs(dst);

  if(param.right_hand_side)
    rhs_operator.evaluate_add(dst, time);
}

template<int dim, int n_components, typename Number>
void
Operator<dim, n_components, Number>::vmult(VectorType & dst, VectorType const & src) const
{
  laplace_operator.vmult(dst, src);
}

template<int dim, int n_components, typename Number>
void
Operator<dim, n_components, Number>::evaluate(VectorType &       dst,
                                              VectorType const & src,
                                              double const       time) const
{
  laplace_operator.set_time(time);
  laplace_operator.evaluate(dst, src);
}

template<int dim, int n_components, typename Number>
unsigned int
Operator<dim, n_components, Number>::solve(VectorType &       sol,
                                           VectorType const & rhs,
                                           double const       time) const
{
  // only activate if desired
  if(false)
  {
    typedef MultigridPreconditioner<dim, Number, n_components> Multigrid;

    std::shared_ptr<Multigrid> mg_preconditioner =
      std::dynamic_pointer_cast<Multigrid>(preconditioner);

    CheckMultigrid<dim, Number, Laplace, Multigrid> check_multigrid(laplace_operator,
                                                                    mg_preconditioner,
                                                                    mpi_comm);

    check_multigrid.check();
  }

  unsigned int n_iterations = iterative_solver->solve(sol, rhs);

  // Set Dirichlet degrees of freedom according to Dirichlet boundary condition.
  if(param.spatial_discretization == SpatialDiscretization::CG)
  {
    laplace_operator.set_time(time);
    laplace_operator.set_inhomogeneous_boundary_values(sol);
  }

  return n_iterations;
}

template<int dim, int n_components, typename Number>
dealii::MatrixFree<dim, Number> const &
Operator<dim, n_components, Number>::get_matrix_free() const
{
  return *matrix_free;
}

template<int dim, int n_components, typename Number>
dealii::DoFHandler<dim> const &
Operator<dim, n_components, Number>::get_dof_handler() const
{
  return dof_handler;
}

template<int dim, int n_components, typename Number>
dealii::types::global_dof_index
Operator<dim, n_components, Number>::get_number_of_dofs() const
{
  return dof_handler.n_dofs();
}

template<int dim, int n_components, typename Number>
double
Operator<dim, n_components, Number>::get_n10() const
{
  return iterative_solver->n10;
}

template<int dim, int n_components, typename Number>
double
Operator<dim, n_components, Number>::get_average_convergence_rate() const
{
  return iterative_solver->rho;
}

template<int dim, int n_components, typename Number>
std::string
Operator<dim, n_components, Number>::get_dof_name() const
{
  return field + "_" + dof_index;
}

template<int dim, int n_components, typename Number>
std::string
Operator<dim, n_components, Number>::get_dof_name_periodicity_and_hanging_node_constraints() const
{
  return field + "_" + dof_index_periodicity_and_handing_node_constraints;
}

template<int dim, int n_components, typename Number>
std::string
Operator<dim, n_components, Number>::get_quad_name() const
{
  return field + "_" + quad_index;
}

template<int dim, int n_components, typename Number>
std::string
Operator<dim, n_components, Number>::get_quad_gauss_lobatto_name() const
{
  return field + "_" + quad_index_gauss_lobatto;
}

template<int dim, int n_components, typename Number>
unsigned int
Operator<dim, n_components, Number>::get_dof_index() const
{
  return matrix_free_data->get_dof_index(get_dof_name());
}

template<int dim, int n_components, typename Number>
unsigned int
Operator<dim, n_components, Number>::get_dof_index_periodicity_and_hanging_node_constraints() const
{
  return matrix_free_data->get_dof_index(get_dof_name_periodicity_and_hanging_node_constraints());
}

template<int dim, int n_components, typename Number>
unsigned int
Operator<dim, n_components, Number>::get_quad_index() const
{
  return matrix_free_data->get_quad_index(get_quad_name());
}

template<int dim, int n_components, typename Number>
unsigned int
Operator<dim, n_components, Number>::get_quad_index_gauss_lobatto() const
{
  return matrix_free_data->get_quad_index(get_quad_gauss_lobatto_name());
}

template<int dim, int n_components, typename Number>
std::shared_ptr<ContainerInterfaceData<Operator<dim, n_components, Number>::rank, dim, double>>
Operator<dim, n_components, Number>::get_container_interface_data() const
{
  return interface_data_dirichlet_cached;
}

template<int dim, int n_components, typename Number>
std::shared_ptr<TimerTree>
Operator<dim, n_components, Number>::get_timings() const
{
  return iterative_solver->get_timings();
}

template<int dim, int n_components, typename Number>
std::shared_ptr<dealii::Mapping<dim> const>
Operator<dim, n_components, Number>::get_mapping() const
{
  return mapping;
}

template class Operator<2, 1, float>;
template class Operator<2, 1, double>;
template class Operator<2, 2, float>;
template class Operator<2, 2, double>;

template class Operator<3, 1, float>;
template class Operator<3, 1, double>;
template class Operator<3, 3, float>;
template class Operator<3, 3, double>;
} // namespace Poisson
} // namespace ExaDG
