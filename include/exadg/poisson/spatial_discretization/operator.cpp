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
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#ifdef DEAL_II_WITH_PETSC
#  include <deal.II/lac/petsc_vector.h>
#endif
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/poisson/preconditioners/multigrid_preconditioner.h>
#include <exadg/poisson/spatial_discretization/operator.h>
#include <exadg/solvers_and_preconditioners/preconditioners/block_jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/inverse_mass_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>
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
  std::shared_ptr<BoundaryDescriptor<rank, dim> const> boundary_descriptor_in,
  std::shared_ptr<FieldFunctions<dim> const>           field_functions_in,
  Parameters const &                                   param_in,
  std::string const &                                  field_in,
  MPI_Comm const &                                     mpi_comm_in)
  : dealii::Subscriptor(),
    grid(grid_in),
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
  if(n_components == 1)
  {
    if(param.spatial_discretization == SpatialDiscretization::DG)
    {
      if(this->grid->triangulation->all_reference_cells_are_hyper_cube())
        fe = std::make_shared<dealii::FE_DGQ<dim>>(param.degree);
      else if(this->grid->triangulation->all_reference_cells_are_simplex())
        fe = std::make_shared<dealii::FE_SimplexDGP<dim>>(param.degree);
      else
        AssertThrow(false, ExcNotImplemented());
    }
    else if(param.spatial_discretization == SpatialDiscretization::CG)
    {
      if(this->grid->triangulation->all_reference_cells_are_hyper_cube())
        fe = std::make_shared<dealii::FE_Q<dim>>(param.degree);
      else if(this->grid->triangulation->all_reference_cells_are_simplex())
        fe = std::make_shared<dealii::FE_SimplexP<dim>>(param.degree);
      else
        AssertThrow(false, ExcNotImplemented());
    }
    else
    {
      AssertThrow(false, ExcNotImplemented());
    }
  }
  else if(n_components == dim)
  {
    if(param.spatial_discretization == SpatialDiscretization::DG)
    {
      if(this->grid->triangulation->all_reference_cells_are_hyper_cube())
        fe = std::make_shared<dealii::FESystem<dim>>(dealii::FE_DGQ<dim>(param.degree), dim);
      else if(this->grid->triangulation->all_reference_cells_are_simplex())
        fe = std::make_shared<dealii::FESystem<dim>>(dealii::FE_SimplexDGP<dim>(param.degree), dim);
      else
        AssertThrow(false, ExcNotImplemented());
    }
    else if(param.spatial_discretization == SpatialDiscretization::CG)
    {
      if(this->grid->triangulation->all_reference_cells_are_hyper_cube())
        fe = std::make_shared<dealii::FESystem<dim>>(dealii::FE_Q<dim>(param.degree), dim);
      else if(this->grid->triangulation->all_reference_cells_are_simplex())
        fe = std::make_shared<dealii::FESystem<dim>>(dealii::FE_SimplexP<dim>(param.degree), dim);
      else
        AssertThrow(false, ExcNotImplemented());
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("not implemented."));
    }
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("not implemented."));
  }

  dof_handler.distribute_dofs(*fe);

  // affine constraints only relevant for continuous FE discretization
  if(param.spatial_discretization == SpatialDiscretization::CG)
  {
    affine_constraints.clear();

    // standard Dirichlet boundaries
    for(auto it : this->boundary_descriptor->dirichlet_bc)
    {
      dealii::ComponentMask mask = dealii::ComponentMask();
      auto it_mask               = boundary_descriptor->dirichlet_bc_component_mask.find(it.first);
      if(it_mask != boundary_descriptor->dirichlet_bc_component_mask.end())
        mask = it_mask->second;

      dealii::DoFTools::make_zero_boundary_constraints(dof_handler,
                                                       it.first,
                                                       affine_constraints,
                                                       mask);
    }

    // DirichletCached boundaries
    for(auto it : this->boundary_descriptor->dirichlet_cached_bc)
    {
      dealii::ComponentMask mask = dealii::ComponentMask();
      dealii::DoFTools::make_zero_boundary_constraints(dof_handler,
                                                       it.first,
                                                       affine_constraints,
                                                       mask);
    }

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
  // conditions. This is not needed in case of discontinuous Galerkin discretizations
  // where boundary conditions are imposed weakly via integrals over the domain
  // boundaries.
  if(param.spatial_discretization == SpatialDiscretization::CG &&
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
  laplace_operator_data.dof_index  = get_dof_index();
  laplace_operator_data.quad_index = get_quad_index();
  if(param.spatial_discretization == SpatialDiscretization::CG &&
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
  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free_in,
  std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data_in)
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

    std::set<dealii::types::boundary_id> bids;
    for(auto const & bc : boundary_descriptor->dirichlet_cached_bc)
      bids.insert(bc.first);

    interface_data_dirichlet_cached->setup(matrix_free, get_dof_index(), quad_indices, bids);

    for(auto const & bc : boundary_descriptor->dirichlet_cached_bc)
      bc.second->set_data_pointer(interface_data_dirichlet_cached);
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
  else if(param.preconditioner == Poisson::Preconditioner::Multigrid)
  {
    MultigridData mg_data;
    mg_data = param.multigrid_data;

    typedef MultigridPreconditioner<dim, Number, n_components> Multigrid;

    preconditioner = std::make_shared<Multigrid>(this->mpi_comm);

    std::shared_ptr<Multigrid> mg_preconditioner =
      std::dynamic_pointer_cast<Multigrid>(preconditioner);

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      dirichlet_boundary_conditions = laplace_operator.get_data().bc->dirichlet_bc;

    // We also need to add DirichletCached boundary conditions. From the
    // perspective of multigrid, there is no difference between standard
    // and cached Dirichlet BCs. Since multigrid does not need information
    // about inhomogeneous boundary data, we simply fill the map with
    // dealii::Functions::ZeroFunction for DirichletCached BCs.
    for(auto iter : laplace_operator.get_data().bc->dirichlet_cached_bc)
      dirichlet_boundary_conditions.insert(
        pair(iter.first, new dealii::Functions::ZeroFunction<dim>(n_components)));

    mg_preconditioner->initialize(mg_data,
                                  &dof_handler.get_triangulation(),
                                  grid->coarse_triangulations,
                                  dof_handler.get_fe(),
                                  grid->mapping,
                                  laplace_operator.get_data(),
                                  false /* moving_mesh */,
                                  dirichlet_boundary_conditions,
                                  grid->periodic_faces);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Specified preconditioner is not implemented!"));
  }

  if(param.solver == Poisson::Solver::CG)
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
  else if(param.solver == Solver::FGMRES)
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
  dst = 0;

  laplace_operator.set_time(time);
  laplace_operator.rhs_add(dst);

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

  // Set constrained degrees of freedom of rhs vector according to the prescribed
  // Dirichlet boundary conditions.
  VectorType & rhs_mutable = const_cast<VectorType &>(rhs);
  if(param.spatial_discretization == SpatialDiscretization::CG)
  {
    laplace_operator.set_constrained_values(rhs_mutable, time);
  }

  unsigned int iterations =
    iterative_solver->solve(sol, rhs_mutable, /* update_preconditioner = */ false);

  // This step should actually be optional: The constrained degrees of freedom of the
  // rhs vector contain the Dirichlet boundary values and the linear operator contains
  // values of 1 on the diagonal. Hence, sol should already contain the correct
  // Dirichlet boundary values for constrained degrees of freedom.
  if(param.spatial_discretization == SpatialDiscretization::CG)
  {
    laplace_operator.set_constrained_values(sol, time);
  }

  return iterations;
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

#ifdef DEAL_II_WITH_TRILINOS
template<int dim, int n_components, typename Number>
void
Operator<dim, n_components, Number>::init_system_matrix(
  dealii::TrilinosWrappers::SparseMatrix & system_matrix,
  MPI_Comm const &                         mpi_comm) const
{
  laplace_operator.init_system_matrix(system_matrix, mpi_comm);
}

template<int dim, int n_components, typename Number>
void
Operator<dim, n_components, Number>::calculate_system_matrix(
  dealii::TrilinosWrappers::SparseMatrix & system_matrix) const
{
  laplace_operator.calculate_system_matrix(system_matrix);
}

template<int dim, int n_components, typename Number>
void
Operator<dim, n_components, Number>::vmult_matrix_based(
  VectorTypeDouble &                             dst,
  dealii::TrilinosWrappers::SparseMatrix const & system_matrix,
  VectorTypeDouble const &                       src) const
{
  system_matrix.vmult(dst, src);
}
#endif

#ifdef DEAL_II_WITH_PETSC
template<int dim, int n_components, typename Number>
void
Operator<dim, n_components, Number>::init_system_matrix(
  dealii::PETScWrappers::MPI::SparseMatrix & system_matrix,
  MPI_Comm const &                           mpi_comm) const
{
  laplace_operator.init_system_matrix(system_matrix, mpi_comm);
}

template<int dim, int n_components, typename Number>
void
Operator<dim, n_components, Number>::calculate_system_matrix(
  dealii::PETScWrappers::MPI::SparseMatrix & system_matrix) const
{
  laplace_operator.calculate_system_matrix(system_matrix);
}

template<int dim, int n_components, typename Number>
void
Operator<dim, n_components, Number>::vmult_matrix_based(
  VectorTypeDouble &                               dst,
  dealii::PETScWrappers::MPI::SparseMatrix const & system_matrix,
  VectorTypeDouble const &                         src) const
{
  apply_petsc_operation(dst,
                        src,
                        system_matrix.get_mpi_communicator(),
                        [&](dealii::PETScWrappers::VectorBase &       petsc_dst,
                            dealii::PETScWrappers::VectorBase const & petsc_src) {
                          system_matrix.vmult(petsc_dst, petsc_src);
                        });
}
#endif

template<int dim, int n_components, typename Number>
std::string
Operator<dim, n_components, Number>::get_dof_name() const
{
  return field + "_" + dof_index;
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
Operator<dim, n_components, Number>::get_container_interface_data()
{
  return interface_data_dirichlet_cached;
}

template<int dim, int n_components, typename Number>
std::shared_ptr<TimerTree>
Operator<dim, n_components, Number>::get_timings() const
{
  return iterative_solver->get_timings();
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
