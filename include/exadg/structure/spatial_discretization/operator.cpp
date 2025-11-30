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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// ExaDG
#include <exadg/functions_and_boundary_conditions/interpolate.h>
#include <exadg/grid/grid_data.h>
#include <exadg/operators/constraints.h>
#include <exadg/operators/finite_element.h>
#include <exadg/operators/quadrature.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_amg.h>
#include <exadg/solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h>
#include <exadg/structure/preconditioners/multigrid_preconditioner.h>
#include <exadg/structure/spatial_discretization/operator.h>
#include <exadg/structure/time_integration/time_int_gen_alpha.h>
#include <exadg/utilities/exceptions.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
Operator<dim, Number>::Operator(
  std::shared_ptr<Grid<dim> const>                      grid_in,
  std::shared_ptr<dealii::Mapping<dim> const>           mapping_in,
  std::shared_ptr<MultigridMappings<dim, Number>> const multigrid_mappings_in,
  std::shared_ptr<BoundaryDescriptor<dim> const>        boundary_descriptor_in,
  std::shared_ptr<FieldFunctions<dim> const>            field_functions_in,
  std::shared_ptr<MaterialDescriptor const>             material_descriptor_in,
  Parameters const &                                    param_in,
  std::string const &                                   field_in,
  bool const                                            setup_scalar_field_in,
  MPI_Comm const &                                      mpi_comm_in)
  : dealii::EnableObserverPointer(),
    grid(grid_in),
    mapping(mapping_in),
    multigrid_mappings(multigrid_mappings_in),
    boundary_descriptor(boundary_descriptor_in),
    field_functions(field_functions_in),
    material_descriptor(material_descriptor_in),
    param(param_in),
    field(field_in),
    dof_handler(*grid_in->triangulation),
    setup_scalar_field(setup_scalar_field_in),
    mpi_comm(mpi_comm_in),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm_in) == 0)
{
  pcout << std::endl << "Construct elasticity operator ..." << std::endl;

  initialize_dof_handler_and_constraints();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
Operator<dim, Number>::initialize_dof_handler_and_constraints()
{
  // create finite element
  fe = create_finite_element<dim>(param.grid.element_type, false /* is_dg */, dim, param.degree);

  // enumerate degrees of freedom
  dof_handler.distribute_dofs(*fe);

  // affine constraints
  affine_constraints_periodicity_and_hanging_nodes.clear();

  add_hanging_node_and_periodicity_constraints(affine_constraints_periodicity_and_hanging_nodes,
                                               *this->grid,
                                               dof_handler);

  // copy periodicity and hanging node constraints, and add further constraints stemming from
  // Dirichlet boundary conditions
  affine_constraints.copy_from(affine_constraints_periodicity_and_hanging_nodes);

  // use all the component masks defined by the user
  std::map<dealii::types::boundary_id, dealii::ComponentMask> map_bid_to_mask =
    boundary_descriptor->dirichlet_bc_component_mask;

  // Dirichlet constraints
  std::map<dealii::types::boundary_id, dealii::ComponentMask> map_boundary_id_to_component_mask;

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

  // compress constraints *once after* complete setup, since affine constraints to copy from need to
  // be non-closed to add further constraints
  affine_constraints_periodicity_and_hanging_nodes.close();
  affine_constraints.close();

  pcout << std::endl
        << "Continuous Galerkin finite element discretization:" << std::endl
        << std::endl;

  print_parameter(pcout, "degree of 1D polynomials", param.degree);
  print_parameter(pcout, "number of dofs per cell", fe->n_dofs_per_cell());
  print_parameter(pcout, "number of dofs (total)", dof_handler.n_dofs());

  // Set up finite element, DoF handler and constraints of scalar field.
  if(setup_scalar_field)
  {
    bool constexpr scalar_field_is_dg = true;
    fe_scalar                         = create_finite_element<dim>(param.grid.element_type,
                                           scalar_field_is_dg /* is_dg */,
                                           1 /* n_components */,
                                           param.degree);

    dof_handler_scalar = std::make_shared<dealii::DoFHandler<dim>>(*grid->triangulation);
    dof_handler_scalar->distribute_dofs(*fe_scalar);

    // Set up hanging node and periodicity constraints for scalar field.
    affine_constraints_periodicity_and_hanging_nodes_scalar.clear();
    if(not scalar_field_is_dg)
    {
      add_hanging_node_and_periodicity_constraints(
        affine_constraints_periodicity_and_hanging_nodes_scalar, *this->grid, *dof_handler_scalar);
    }
    affine_constraints_periodicity_and_hanging_nodes_scalar.close();

    pcout << std::endl
          << "Discontinuous Galerkin finite element discretization of scalar field:" << std::endl
          << std::endl;

    print_parameter(pcout, "degree of 1D polynomials", param.degree);
    print_parameter(pcout, "number of dofs per cell", fe_scalar->n_dofs_per_cell());
    print_parameter(pcout, "number of dofs (total)", dof_handler_scalar->n_dofs());
  }
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

  // Insert `dealii::DoFHandler` and `dealii::AffineConstraints`, where the constraints include
  // periodicity and hanging node constraints *as well as* inhomogeneous Dirichlet boundary
  // conditions.
  matrix_free_data.insert_dof_handler(&dof_handler, get_dof_name());
  matrix_free_data.insert_constraint(&affine_constraints, get_dof_name());

  // Insert `dealii::DoFHandler` and `dealii::AffineConstraints`, where the constraints include
  // periodicity and hanging node constraints *only*. Note that the `dealii::DoFHandler` is
  // identical to the one used above, but the constraints differ.
  matrix_free_data.insert_dof_handler(&dof_handler,
                                      get_dof_name_periodicity_and_hanging_node_constraints());
  matrix_free_data.insert_constraint(&affine_constraints_periodicity_and_hanging_nodes,
                                     get_dof_name_periodicity_and_hanging_node_constraints());

  // Insert `dealii::DoFHandler` and `dealii::AffineConstraints` for scalar field.
  if(setup_scalar_field)
  {
    matrix_free_data.insert_dof_handler(dof_handler_scalar.get(), get_dof_name_scalar());
    matrix_free_data.insert_constraint(&affine_constraints_periodicity_and_hanging_nodes_scalar,
                                       get_dof_name_scalar());
  }

  // Set up and insert `dealii::Quadrature` objects.
  std::shared_ptr<dealii::Quadrature<dim>> quadrature =
    create_quadrature<dim>(param.grid.element_type, param.degree + 1);
  matrix_free_data.insert_quadrature(*quadrature, get_quad_name());

  // Create a Gauss-Lobatto quadrature rule for `DirichletCached` boundary conditions.
  // These quadrature points coincide with the nodes of the discretization, so that
  // the values stored in the DirichletCached boundary condition can be directly
  // injected into the DoF vector. This allows to set constrained degrees of freedom
  // in case of continuous Galerkin discretizations with `DirichletCached` boundary
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
Operator<dim, Number>::setup_coupling_boundary_conditions()
{
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
}

template<int dim, typename Number>
void
Operator<dim, Number>::setup_operators()
{
  // elasticity operator
  operator_data.dof_index               = get_dof_index();
  operator_data.quad_index              = get_quad_index();
  operator_data.dof_index_inhomogeneous = get_dof_index_periodicity_and_hanging_node_constraints();
  operator_data.use_matrix_based_operator_level = param.use_matrix_based_operator;
  operator_data.sparse_matrix_type              = param.sparse_matrix_type;

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
  operator_data.large_deformation   = param.large_deformation;
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
    elasticity_operator_nonlinear.initialize(*matrix_free,
                                             affine_constraints,
                                             operator_data,
                                             false /* assemble_matrix */);
  }
  else
  {
    elasticity_operator_linear.initialize(*matrix_free,
                                          affine_constraints,
                                          operator_data,
                                          true /* assemble_matrix */);
  }

  // Mass operator and inverse mass operator for vector-valued space
  if(param.problem_type == ProblemType::Unsteady)
  {
    // vector-valued mass operator
    Structure::MassOperatorData<dim, Number> mass_data;
    mass_data.dof_index               = get_dof_index();
    mass_data.dof_index_inhomogeneous = get_dof_index_periodicity_and_hanging_node_constraints();
    mass_data.quad_index              = get_quad_index();
    mass_data.bc                      = boundary_descriptor;

    mass_operator.initialize(*matrix_free, affine_constraints, mass_data);

    mass_operator.set_scaling_factor(param.density);

    // vector-valued inverse mass operator for initial acceleration
    InverseMassOperatorData<Number> inverse_mass_operator_data;
    // Copy the relevant settings from the (non-)linear solvers for to reach the same tolerance as
    // the outermost solver.
    SolverData & solver_data = inverse_mass_operator_data.parameters.solver_data;
    if(param.large_deformation)
    {
      solver_data.abs_tol  = param.newton_solver_data.abs_tol;
      solver_data.rel_tol  = param.newton_solver_data.rel_tol;
      solver_data.max_iter = param.newton_solver_data.max_iter;
    }
    else
    {
      solver_data.abs_tol  = param.solver_data.abs_tol;
      solver_data.rel_tol  = param.solver_data.rel_tol;
      solver_data.max_iter = param.solver_data.max_iter;
    }
    inverse_mass_operator_data.dof_index  = get_dof_index();
    inverse_mass_operator_data.quad_index = get_quad_index();

    // For a continuous Galerkin discretization a global Krylov solver is needed. The default
    // `PointJacobi` preconditioner is usually sufficient.
    inverse_mass_operator_data.parameters.implementation_type =
      inverse_mass_operator_data.get_optimal_inverse_mass_type(*fe);
    inverse_mass_operator_data.parameters.preconditioner = PreconditionerMass::PointJacobi;

    inverse_mass.initialize(*matrix_free, inverse_mass_operator_data, &affine_constraints);
  }

  // scalar inverse mass operator
  if(setup_scalar_field)
  {
    InverseMassOperatorData<Number> inverse_mass_operator_data_scalar;
    inverse_mass_operator_data_scalar.parameters.solver_data = param.solver_data;
    inverse_mass_operator_data_scalar.dof_index              = get_dof_index_scalar();
    inverse_mass_operator_data_scalar.quad_index             = get_quad_index();
    inverse_mass_operator_data_scalar.parameters.implementation_type =
      inverse_mass_operator_data_scalar.get_optimal_inverse_mass_type(*fe_scalar);

    inverse_mass_scalar.initialize(*matrix_free,
                                   inverse_mass_operator_data_scalar,
                                   &affine_constraints_periodicity_and_hanging_nodes_scalar);
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
Operator<dim, Number>::setup_calculators_for_derived_quantities()
{
  if(setup_scalar_field)
  {
    vector_magnitude_calculator.initialize(*matrix_free,
                                           get_dof_index(),
                                           get_dof_index_scalar(),
                                           get_quad_index());

    displacement_jacobian_calculator.initialize(*matrix_free,
                                                get_dof_index(),
                                                get_dof_index_scalar(),
                                                get_quad_index());

    ElasticityOperatorBase<dim, Number> const & elasticity_operator_base =
      param.large_deformation ?
        static_cast<ElasticityOperatorBase<dim, Number> const &>(elasticity_operator_nonlinear) :
        static_cast<ElasticityOperatorBase<dim, Number> const &>(elasticity_operator_linear);

    max_principal_stress_calculator.initialize(*matrix_free,
                                               get_dof_index(),
                                               get_dof_index_scalar(),
                                               get_quad_index(),
                                               elasticity_operator_base);
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::setup()
{
  // initialize MatrixFree and MatrixFreeData
  std::shared_ptr<dealii::MatrixFree<dim, Number>> mf =
    std::make_shared<dealii::MatrixFree<dim, Number>>();
  std::shared_ptr<MatrixFreeData<dim, Number>> mf_data =
    std::make_shared<MatrixFreeData<dim, Number>>();

  fill_matrix_free_data(*mf_data);

  mf->reinit(get_mapping(),
             mf_data->get_dof_handler_vector(),
             mf_data->get_constraint_vector(),
             mf_data->get_quadrature_vector(),
             mf_data->data);

  // Subsequently, call the other setup function with MatrixFree/MatrixFreeData objects as
  // arguments.
  this->setup(mf, mf_data);
}

template<int dim, typename Number>
void
Operator<dim, Number>::setup(std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free_in,
                             std::shared_ptr<MatrixFreeData<dim, Number> const> matrix_free_data_in)
{
  pcout << std::endl << "Setup elasticity operator ..." << std::endl;

  matrix_free      = matrix_free_in;
  matrix_free_data = matrix_free_data_in;

  setup_coupling_boundary_conditions();

  setup_operators();

  setup_calculators_for_derived_quantities();

  setup_preconditioner();

  setup_solver();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
Operator<dim, Number>::setup_preconditioner()
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
        elasticity_operator_nonlinear, false);
    }
    else
    {
      preconditioner = std::make_shared<JacobiPreconditioner<LinearOperator<dim, Number>>>(
        elasticity_operator_linear, false);
    }
  }
  else if(param.preconditioner == Preconditioner::AdditiveSchwarz)
  {
    if(param.large_deformation)
    {
      preconditioner =
        std::make_shared<AdditiveSchwarzPreconditioner<NonLinearOperator<dim, Number>>>(
          elasticity_operator_nonlinear, false);
    }
    else
    {
      preconditioner = std::make_shared<AdditiveSchwarzPreconditioner<LinearOperator<dim, Number>>>(
        elasticity_operator_linear, false);
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
                                    grid,
                                    multigrid_mappings,
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
                                    grid,
                                    multigrid_mappings,
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
                                             false /* initialize */,
                                             param.multigrid_data.coarse_problem.amg_data);
    }
    else
    {
      typedef PreconditionerAMG<LinearOperator<dim, Number>, Number> AMG;
      preconditioner = std::make_shared<AMG>(elasticity_operator_linear,
                                             false /* initialize */,
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
Operator<dim, Number>::setup_solver()
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
std::string
Operator<dim, Number>::get_dof_name() const
{
  return field + "_" + dof_index;
}

template<int dim, typename Number>
std::string
Operator<dim, Number>::get_dof_name_periodicity_and_hanging_node_constraints() const
{
  return field + "_" + dof_index_periodicity_and_hanging_node_constraints;
}

template<int dim, typename Number>
std::string
Operator<dim, Number>::get_dof_name_scalar() const
{
  AssertThrow(setup_scalar_field, dealii::ExcMessage("Scalar field should not be set up."));

  return field + "_" + dof_index_scalar;
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
void
Operator<dim, Number>::compute_displacement_magnitude(VectorType &       dst_scalar_valued,
                                                      VectorType const & src_vector_valued) const
{
  AssertThrow(setup_scalar_field,
              dealii::ExcMessage("Scalar field not set up. "
                                 "Cannot compute displacement magnitude."));

  vector_magnitude_calculator.compute_projection_rhs(dst_scalar_valued, src_vector_valued);

  inverse_mass_scalar.apply(dst_scalar_valued, dst_scalar_valued);
}

template<int dim, typename Number>
void
Operator<dim, Number>::compute_displacement_jacobian(VectorType &       dst_scalar_valued,
                                                     VectorType const & src_vector_valued) const
{
  AssertThrow(setup_scalar_field,
              dealii::ExcMessage("Scalar field not set up. "
                                 "Cannot compute Jacobian of the displacement field."));

  displacement_jacobian_calculator.compute_projection_rhs(dst_scalar_valued, src_vector_valued);

  inverse_mass_scalar.apply(dst_scalar_valued, dst_scalar_valued);
}

template<int dim, typename Number>
void
Operator<dim, Number>::compute_max_principal_stress(VectorType &       dst_scalar_valued,
                                                    VectorType const & src_vector_valued) const
{
  AssertThrow(setup_scalar_field,
              dealii::ExcMessage("Scalar field not set up. "
                                 "Cannot compute maximum principal stress."));

  max_principal_stress_calculator.compute_projection_rhs(dst_scalar_valued, src_vector_valued);

  inverse_mass_scalar.apply(dst_scalar_valued, dst_scalar_valued);
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::get_dof_index_periodicity_and_hanging_node_constraints() const
{
  return matrix_free_data->get_dof_index(get_dof_name_periodicity_and_hanging_node_constraints());
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::get_dof_index_scalar() const
{
  AssertThrow(setup_scalar_field, dealii::ExcMessage("Scalar field should not be set up."));

  return matrix_free_data->get_dof_index(get_dof_name_scalar());
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
Operator<dim, Number>::compute_scaling_factor_mass(double const scaling_factor_acceleration,
                                                   double const scaling_factor_velocity) const
{
  double scaling_factor_mass = scaling_factor_acceleration;
  if(param.weak_damping_active)
  {
    scaling_factor_mass += param.weak_damping_coefficient * scaling_factor_velocity;
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
Operator<dim, Number>::initialize_dof_vector(VectorType & src) const
{
  matrix_free->initialize_dof_vector(src, get_dof_index());
}

template<int dim, typename Number>
void
Operator<dim, Number>::initialize_dof_vector_scalar(VectorType & src) const
{
  AssertThrow(setup_scalar_field,
              dealii::ExcMessage("Scalar field not set up. "
                                 "Cannot initialize scalar dof vector."));

  matrix_free->initialize_dof_vector(src, get_dof_index_scalar());
}

template<int dim, typename Number>
void
Operator<dim, Number>::prescribe_initial_displacement(VectorType & displacement,
                                                      double const time) const
{
  Utilities::interpolate(dof_handler, *field_functions->initial_displacement, displacement, time);
}

template<int dim, typename Number>
void
Operator<dim, Number>::prescribe_initial_velocity(VectorType & velocity, double const time) const
{
  Utilities::interpolate(dof_handler, *field_functions->initial_velocity, velocity, time);
}

template<int dim, typename Number>
void
Operator<dim, Number>::compute_initial_acceleration(VectorType &       initial_acceleration,
                                                    VectorType const & initial_displacement,
                                                    double const       time) const
{
  if(field_functions->initial_acceleration.get())
  {
    Utilities::interpolate(dof_handler,
                           *field_functions->initial_acceleration,
                           initial_acceleration,
                           time);
  }
  else
  {
    VectorType rhs(initial_acceleration);
    rhs = 0.0;

    if(param.large_deformation) // nonlinear case
    {
      // elasticity operator

      // deactivate the mass operator term
      double const scaling_factor_mass =
        elasticity_operator_nonlinear.get_scaling_factor_mass_operator();
      elasticity_operator_nonlinear.set_scaling_factor_mass_operator(0.0);

      // evaluate elasticity operator including inhomogeneous Dirichlet/Neumann boundary
      // conditions: Note that we do not have to set inhomogeneous Dirichlet degrees of freedom
      // explicitly since the function prescribe_initial_displacement() sets the initial
      // displacement for all dofs (including Dirichlet dofs) and since the initial condition for
      // the displacements needs to be consistent with the Dirichlet boundary data g(t=t0) at
      // initial time, i.e. the vector initial_displacement already contains the correct Dirichlet
      // data.
      elasticity_operator_nonlinear.set_time(time);
      elasticity_operator_nonlinear.evaluate_nonlinear(rhs, initial_displacement);
      // shift to right-hand side
      rhs *= -1.0;

      // revert scaling factor to initialized value
      elasticity_operator_nonlinear.set_scaling_factor_mass_operator(scaling_factor_mass);

      // body forces
      if(param.body_force)
      {
        body_force_operator.evaluate_add(rhs, initial_displacement, time);
      }
    }
    else // linear case
    {
      // elasticity operator

      // deactivate the mass operator
      double const scaling_factor_mass =
        elasticity_operator_linear.get_scaling_factor_mass_operator();
      elasticity_operator_linear.set_scaling_factor_mass_operator(0.0);

      // evaluate elasticity operator including inhomogeneous Dirichlet/Neumann boundary
      // conditions: Note that we do not have to set inhomogeneous Dirichlet degrees of freedom
      // explicitly since the function prescribe_initial_displacement() sets the initial
      // displacement for all dofs (including Dirichlet dofs) and since the initial condition for
      // the displacements needs to be consistent with the Dirichlet boundary data g(t=t0) at
      // initial time, i.e. the vector initial_displacement already contains the correct Dirichlet
      // data.
      elasticity_operator_linear.set_time(time);
      elasticity_operator_linear.evaluate(rhs, initial_displacement);
      // shift to right-hand side
      rhs *= -1.0;

      // revert scaling factor to initialized value
      elasticity_operator_linear.set_scaling_factor_mass_operator(scaling_factor_mass);

      // body force
      if(param.body_force)
      {
        // The displacement is irrelevant for linear problem, since
        // pull_back_body_force = false in this case.
        body_force_operator.evaluate_add(rhs, initial_displacement, time);
      }
    }

    // Shift inhomogeneous part of mass matrix operator (i.e. mass matrix applied to a dof vector
    // with the initial acceleration in Dirichlet degrees of freedom) to the right-hand side.
    mass_operator.rhs_add(rhs);

    // Apply inverse mass operator to compute the initial acceleration. Note that the mass
    // operator is scaled with `density`, hence scale the right-hand side, as the
    // `InverseMassOperator` has its own `MassOperator`.
    rhs /= param.density;
    inverse_mass.apply(initial_acceleration, rhs);

    // Set initial acceleration for the Dirichlet degrees of freedom so that the initial
    // acceleration is also correct on the Dirichlet boundary
    mass_operator.set_inhomogeneous_constrained_values(initial_acceleration);
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::evaluate_mass_operator(VectorType & dst, VectorType const & src) const
{
  mass_operator.evaluate(dst, src);
}

template<int dim, typename Number>
void
Operator<dim, Number>::apply_add_damping_operator(VectorType & dst, VectorType const & src) const
{
  if(param.weak_damping_active)
  {
    VectorType tmp;
    tmp.reinit(src);
    tmp.equ(param.weak_damping_coefficient, src);
    mass_operator.apply_add(dst, tmp);
  }
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
  update_elasticity_operator(factor, time);

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
  elasticity_operator_nonlinear.set_constrained_dofs_to_zero(dst);
}

template<int dim, typename Number>
void
Operator<dim, Number>::set_solution_linearization(VectorType const & vector) const
{
  elasticity_operator_nonlinear.set_solution_linearization(vector);
}

template<int dim, typename Number>
void
Operator<dim, Number>::assemble_matrix_if_matrix_based() const
{
  if(param.large_deformation)
  {
    elasticity_operator_nonlinear.assemble_matrix_if_matrix_based();
  }
  else
  {
    elasticity_operator_linear.assemble_matrix_if_matrix_based();
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::evaluate_elasticity_operator(VectorType &       dst,
                                                    VectorType const & src,
                                                    double const       factor,
                                                    double const       time) const
{
  update_elasticity_operator(factor, time);

  if(param.large_deformation)
  {
    elasticity_operator_nonlinear.evaluate_nonlinear(dst, src);
  }
  else
  {
    elasticity_operator_linear.evaluate(dst, src);
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::update_elasticity_operator(double const factor, double const time) const
{
  if(param.large_deformation)
  {
    elasticity_operator_nonlinear.set_scaling_factor_mass_operator(factor);
    elasticity_operator_nonlinear.set_time(time);
  }
  else
  {
    elasticity_operator_linear.set_scaling_factor_mass_operator(factor);
    elasticity_operator_linear.set_time(time);
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::apply_elasticity_operator(VectorType & dst, VectorType const & src) const
{
  if(param.large_deformation)
  {
    elasticity_operator_nonlinear.vmult(dst, src);
  }
  else
  {
    elasticity_operator_linear.vmult(dst, src);
  }
}

template<int dim, typename Number>
std::tuple<unsigned int, unsigned int>
Operator<dim, Number>::solve_nonlinear(VectorType &       sol,
                                       VectorType const & const_vector,
                                       double const       scaling_factor_acceleration,
                                       double const       scaling_factor_velocity,
                                       double const       time,
                                       bool const         update_preconditioner) const
{
  // update operators
  double const scaling_factor_mass =
    compute_scaling_factor_mass(scaling_factor_acceleration, scaling_factor_velocity);
  residual_operator.update(const_vector, scaling_factor_mass, time);

  linearized_operator.update(scaling_factor_mass, time);

  // Matrix-based implementation: note that the re-assembly of the matrix is done in the function
  // set_solution_linearization() called by the Newton solver.

  // set inhomogeneous Dirichlet values, hanging node and periodicity constraints in order to
  // evaluate the nonlinear residual correctly
  elasticity_operator_nonlinear.set_time(time);
  elasticity_operator_nonlinear.set_inhomogeneous_constrained_values(sol);

  // call Newton solver
  Newton::UpdateData update;
  update.do_update                = update_preconditioner;
  update.update_every_newton_iter = param.update_preconditioner_every_newton_iterations;
  update.update_once_converged    = param.update_preconditioner_once_newton_converged;

  // solve nonlinear problem
  auto const iter = newton_solver->solve(sol, update);

  return iter;
}

template<int dim, typename Number>
void
Operator<dim, Number>::rhs(VectorType & dst, double const time) const
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
unsigned int
Operator<dim, Number>::solve_linear(VectorType &       sol,
                                    VectorType const & rhs,
                                    double const       scaling_factor_acceleration,
                                    double const       scaling_factor_velocity,
                                    double const       time,
                                    bool const         update_preconditioner) const
{
  // unsteady problems
  double const scaling_factor_mass =
    compute_scaling_factor_mass(scaling_factor_acceleration, scaling_factor_velocity);

  update_elasticity_operator(scaling_factor_mass, time);

  // In case of a matrix-based implementation, we assemble the matrix once at initialization,
  // since it remains constant, and avoid calling `assemble_matrix_if_matrix_based()` here.

  linear_solver->update_preconditioner(update_preconditioner);

  // solve linear system of equations
  unsigned int const iterations = linear_solver->solve(sol, rhs);

  // Set Dirichlet degrees of freedom according to Dirichlet boundary condition.
  elasticity_operator_linear.set_time(time);
  elasticity_operator_linear.set_inhomogeneous_constrained_values(sol);

  return iterations;
}

template<int dim, typename Number>
std::shared_ptr<dealii::MatrixFree<dim, Number> const>
Operator<dim, Number>::get_matrix_free() const
{
  return matrix_free;
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
dealii::DoFHandler<dim> const &
Operator<dim, Number>::get_dof_handler_scalar() const
{
  AssertThrow(setup_scalar_field, dealii::ExcMessage("Scalar dof handler has not been set up."));

  return *dof_handler_scalar;
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
