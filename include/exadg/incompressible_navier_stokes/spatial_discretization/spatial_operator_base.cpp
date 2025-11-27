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
#include <exadg/grid/grid_utilities.h>
#include <exadg/grid/mapping_dof_vector.h>
#include <exadg/incompressible_navier_stokes/preconditioners/multigrid_preconditioner_projection.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/spatial_operator_base.h>
#include <exadg/operators/finite_element.h>
#include <exadg/operators/grid_related_time_step_restrictions.h>
#include <exadg/operators/quadrature.h>
#include <exadg/operators/solution_projection_between_triangulations.h>
#include <exadg/solvers_and_preconditioners/preconditioners/block_jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/inverse_mass_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>
#include <exadg/time_integration/restart.h>
#include <exadg/utilities/exceptions.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
SpatialOperatorBase<dim, Number>::SpatialOperatorBase(
  std::shared_ptr<Grid<dim> const>                      grid_in,
  std::shared_ptr<dealii::Mapping<dim> const>           mapping_in,
  std::shared_ptr<MultigridMappings<dim, Number>> const multigrid_mappings_in,
  std::shared_ptr<BoundaryDescriptor<dim> const>        boundary_descriptor_in,
  std::shared_ptr<FieldFunctions<dim> const>            field_functions_in,
  Parameters const &                                    parameters_in,
  std::string const &                                   field_in,
  MPI_Comm const &                                      mpi_comm_in)
  : dealii::EnableObserverPointer(),
    grid(grid_in),
    mapping(mapping_in),
    multigrid_mappings(multigrid_mappings_in),
    boundary_descriptor(boundary_descriptor_in),
    field_functions(field_functions_in),
    param(parameters_in),
    field(field_in),
    dof_index_first_point(0),
    evaluation_time(0.0),
    dof_handler_u(*grid_in->triangulation),
    dof_handler_p(*grid_in->triangulation),
    dof_handler_u_scalar(*grid_in->triangulation),
    pressure_level_is_undefined(false),
    mpi_comm(mpi_comm_in),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
    velocity_ptr(nullptr),
    pressure_ptr(nullptr)
{
  pcout << std::endl
        << "Construct incompressible Navier-Stokes operator ..." << std::endl
        << std::flush;

  initialize_dof_handler_and_constraints();

  initialize_boundary_descriptor_laplace();

  initialization_pure_dirichlet_bc();

  pcout << std::endl << "... done!" << std::endl << std::flush;
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::initialize_dof_handler_and_constraints()
{
  fe_p = create_finite_element<dim>(param.grid.element_type,
                                    true /* is_dg */,
                                    1 /* n_components */,
                                    param.get_degree_p(param.degree_u));

  fe_u_scalar = create_finite_element<dim>(param.grid.element_type,
                                           true /* is_dg */,
                                           1 /* n_components */,
                                           param.degree_u);

  fe_u = setup_fe_u(param.spatial_discretization, param.grid.element_type, param.degree_u);

  if((param.restart_data.consider_mapping_write and param.restart_data.write_restart) or
     (param.restart_data.consider_mapping_read_source and param.restarted_simulation))
  {
    fe_mapping          = create_finite_element<dim>(param.grid.element_type,
                                            true /* is_dg */,
                                            dim /* n_components */,
                                            param.mapping_degree);
    dof_handler_mapping = std::make_shared<dealii::DoFHandler<dim>>(*grid->triangulation);
    dof_handler_mapping->distribute_dofs(*fe_mapping);
  }

  // enumerate degrees of freedom
  dof_handler_u.distribute_dofs(*fe_u);
  dof_handler_p.distribute_dofs(*fe_p);
  dof_handler_u_scalar.distribute_dofs(*fe_u_scalar);

  // Strong imposition of boundary conditions in the case of HDIV
  if(param.spatial_discretization == SpatialDiscretization::HDIV)
  {
    // Periodic boundaries
    // We need to make sure the normal dofs are shared between cells on the periodic boundaries,
    // since these are continuous for HDIV.
    dealii::IndexSet relevant_dofs = dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_u);
    constraint_u.reinit(dof_handler_u.locally_owned_dofs(), relevant_dofs);

    for(auto const & face : grid->periodic_face_pairs)
      dealii::DoFTools::make_periodicity_constraints(
        dof_handler_u,
        face.cell[0]->face(face.face_idx[0])->boundary_id(),
        face.cell[1]->face(face.face_idx[1])->boundary_id(),
        face.face_idx[0] / 2,
        constraint_u);

    // Symmetry boundaries
    // Constraints the normal components of the velocity, where "0" as second argument indicates the
    // first component in the dof_handler.
    if(not(boundary_descriptor->velocity->symmetry_bc.empty()))
    {
      for(auto bc : boundary_descriptor->velocity->symmetry_bc)
        dealii::VectorTools::project_boundary_values_div_conforming(
          dof_handler_u, 0, *(bc.second), bc.first, constraint_u, *get_mapping());
    }

    // Dirichlet boundaries
    if(not(boundary_descriptor->velocity->dirichlet_bc.empty()))
    {
      AssertThrow(
        false,
        dealii::ExcMessage(
          "Dirichlet BCs are not properly implemented for HDIV. The normal component of the velocity field needs to be strongly applied."));
      // We would like to do something similar to above. Probably would work with the same function
      // i.e dealii::VectorTools::project_boundary_values_div_conforming(). Otherwise one might want
      // to look into: dealii::VectorTools::interpolate_boundary_values
      // dealii::VectorTools::project_boundary_values
      // dealii::DoFTools::make_zero_boundary_constraints
    }
  }

  constraint_u.close();
  constraint_p.close();
  constraint_u_scalar.close();

  // Output to pcout
  pcout << "Velocity:" << std::endl;
  if(param.spatial_discretization == SpatialDiscretization::L2)
  {
    print_parameter(pcout, "degree of 1D polynomials", param.degree_u);
  }
  else if(param.spatial_discretization == SpatialDiscretization::HDIV)
  {
    print_parameter(pcout, "degree of 1D polynomials (normal)", param.degree_u);
    print_parameter(pcout, "degree of 1D polynomials (tangential)", (param.degree_u - 1));
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("FE not implemented."));
  }
  print_parameter(pcout, "number of dofs per cell", fe_u->n_dofs_per_cell());
  print_parameter(pcout, "number of dofs (total)", dof_handler_u.n_dofs());

  pcout << "Pressure:" << std::endl;
  print_parameter(pcout, "degree of 1D polynomials", param.get_degree_p(param.degree_u));
  print_parameter(pcout, "number of dofs per cell", fe_p->n_dofs_per_cell());
  print_parameter(pcout, "number of dofs (total)", dof_handler_p.n_dofs());

  pcout << "Velocity and pressure:" << std::endl;
  print_parameter(pcout,
                  "number of dofs per cell",
                  fe_u->n_dofs_per_cell() + fe_p->n_dofs_per_cell());
  print_parameter(pcout, "number of dofs (total)", get_number_of_dofs());

  pcout << std::flush;
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::initialize_boundary_descriptor_laplace()
{
  boundary_descriptor_laplace = std::make_shared<Poisson::BoundaryDescriptor<0, dim>>();

  // Dirichlet BCs for pressure
  boundary_descriptor_laplace->dirichlet_bc = boundary_descriptor->pressure->dirichlet_bc;

  // Neumann BCs for pressure: These boundary conditions are empty.
  // However, when using projection methods with the solution of a pressure Poisson
  // equation, the interface of the Laplace operator requires to set functions on
  // Neumann boundaries, which we simply fill by ZeroFunction. In case that a
  // projection method prescribes inhomogeneous Neumann boundary conditions for the
  // pressure (e.g. dual splitting projection scheme), this is done by separate
  // routines.
  for(typename std::set<dealii::types::boundary_id>::const_iterator it =
        boundary_descriptor->pressure->neumann_bc.begin();
      it != boundary_descriptor->pressure->neumann_bc.end();
      ++it)
  {
    std::shared_ptr<dealii::Function<dim>> zero_function;
    zero_function = std::make_shared<dealii::Functions::ZeroFunction<dim>>(1);
    boundary_descriptor_laplace->neumann_bc.insert(
      std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>(*it,
                                                                                    zero_function));
  }
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::initialization_pure_dirichlet_bc()
{ // Erroneously, the boundary descriptor might contain too many boundary IDs which
  // do not even exist in the triangulation. Here, we make sure that each entry of
  // the boundary descriptor has indeed a counterpart in the triangulation.
  std::vector<dealii::types::boundary_id> boundary_ids = grid->triangulation->get_boundary_ids();
  for(auto it = boundary_descriptor->pressure->dirichlet_bc.begin();
      it != boundary_descriptor->pressure->dirichlet_bc.end();
      ++it)
  {
    bool const triangulation_has_boundary_id =
      std::find(boundary_ids.begin(), boundary_ids.end(), it->first) != boundary_ids.end();

    AssertThrow(triangulation_has_boundary_id,
                dealii::ExcMessage("The boundary descriptor for the pressure contains boundary IDs "
                                   "that are not part of the triangulation."));
  }

  pressure_level_is_undefined = boundary_descriptor->pressure->dirichlet_bc.empty();

  if(is_pressure_level_undefined())
  {
    if(param.adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalSolutionInPoint)
    {
      dof_index_first_point = 0;
      for(unsigned int d = 0; d < dim; ++d)
        first_point[d] = 0.0;

      if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
      {
        typename dealii::DoFHandler<dim>::active_cell_iterator first_cell;

        bool processor_has_active_cells = false;
        for(auto const & cell : dof_handler_p.active_cell_iterators())
        {
          if(cell->is_locally_owned())
          {
            first_cell = cell;

            processor_has_active_cells = true;
            break;
          }
        }

        AssertThrow(processor_has_active_cells == true,
                    dealii::ExcMessage("No active cells on Processor with ID=0"));

        dealii::FEValues<dim> fe_values(dof_handler_p.get_fe(),
                                        dealii::Quadrature<dim>(
                                          dof_handler_p.get_fe().get_unit_support_points()),
                                        dealii::update_quadrature_points);

        fe_values.reinit(first_cell);

        first_point = fe_values.quadrature_point(0);
        std::vector<dealii::types::global_dof_index> dof_indices(
          dof_handler_p.get_fe().dofs_per_cell);
        first_cell->get_dof_indices(dof_indices);
        dof_index_first_point = dof_indices[0];
      }
      dof_index_first_point = dealii::Utilities::MPI::sum(dof_index_first_point, mpi_comm);
      for(unsigned int d = 0; d < dim; ++d)
      {
        first_point[d] = dealii::Utilities::MPI::sum(first_point[d], mpi_comm);
      }
    }
  }
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::fill_matrix_free_data(
  MatrixFreeData<dim, Number> & matrix_free_data) const
{
  // append mapping flags
  matrix_free_data.append_mapping_flags(MassKernel<dim, Number>::get_mapping_flags());
  matrix_free_data.append_mapping_flags(
    Operators::DivergenceKernel<dim, Number>::get_mapping_flags());
  matrix_free_data.append_mapping_flags(
    Operators::GradientKernel<dim, Number>::get_mapping_flags());

  if(param.convective_problem())
    matrix_free_data.append_mapping_flags(
      Operators::ConvectiveKernel<dim, Number>::get_mapping_flags());

  if(param.viscous_problem())
    matrix_free_data.append_mapping_flags(
      Operators::ViscousKernel<dim, Number>::get_mapping_flags(true, true));

  if(param.right_hand_side)
    matrix_free_data.append_mapping_flags(Operators::RHSKernel<dim, Number>::get_mapping_flags());

  if(param.use_divergence_penalty)
    matrix_free_data.append_mapping_flags(
      Operators::DivergencePenaltyKernel<dim, Number>::get_mapping_flags());

  if(param.use_continuity_penalty)
    matrix_free_data.append_mapping_flags(
      Operators::ContinuityPenaltyKernel<dim, Number>::get_mapping_flags());

  // mapping flags required for CFL condition
  MappingFlags flags_cfl;
  flags_cfl.cells = dealii::update_quadrature_points;
  matrix_free_data.append_mapping_flags(flags_cfl);

  // dof handler
  matrix_free_data.insert_dof_handler(&dof_handler_u, field + dof_index_u);
  matrix_free_data.insert_dof_handler(&dof_handler_p, field + dof_index_p);
  matrix_free_data.insert_dof_handler(&dof_handler_u_scalar, field + dof_index_u_scalar);

  // constraint
  matrix_free_data.insert_constraint(&constraint_u, field + dof_index_u);
  matrix_free_data.insert_constraint(&constraint_p, field + dof_index_p);
  matrix_free_data.insert_constraint(&constraint_u_scalar, field + dof_index_u_scalar);

  // quadrature
  std::shared_ptr<dealii::Quadrature<dim>> quadrature_u =
    create_quadrature<dim>(param.grid.element_type, param.degree_u + 1);
  matrix_free_data.insert_quadrature(*quadrature_u, field + quad_index_u);
  std::shared_ptr<dealii::Quadrature<dim>> quadrature_p =
    create_quadrature<dim>(param.grid.element_type, param.get_degree_p(param.degree_u) + 1);
  matrix_free_data.insert_quadrature(*quadrature_p, field + quad_index_p);
  std::shared_ptr<dealii::Quadrature<dim>> quadrature_u_overintegration =
    create_quadrature<dim>(param.grid.element_type, param.degree_u + (param.degree_u + 2) / 2);
  matrix_free_data.insert_quadrature(*quadrature_u_overintegration,
                                     field + quad_index_u_overintegration);

  // TODO create these quadrature rules only when needed
  if(param.grid.element_type == ElementType::Hypercube)
  {
    matrix_free_data.insert_quadrature(dealii::QGaussLobatto<1>(param.degree_u + 1),
                                       field + quad_index_u_nodal_points);
    matrix_free_data.insert_quadrature(dealii::QGaussLobatto<1>(param.get_degree_p(param.degree_u) +
                                                                1),
                                       field + quad_index_p_nodal_points);
  }
  else if(param.grid.element_type == ElementType::Simplex)
  {
    matrix_free_data.insert_quadrature(dealii::Quadrature<dim>(
                                         dof_handler_u_scalar.get_fe().get_unit_support_points()),
                                       field + quad_index_u_nodal_points);
    matrix_free_data.insert_quadrature(dealii::Quadrature<dim>(
                                         dof_handler_p.get_fe().get_unit_support_points()),
                                       field + quad_index_p_nodal_points);
  }
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::initialize_dirichlet_cached_bc()
{
  // initialize data container for DirichletCached boundary conditions
  if(not(boundary_descriptor->velocity->dirichlet_cached_bc.empty()))
  {
    std::vector<unsigned int> quad_indices;
    quad_indices.emplace_back(get_quad_index_velocity_standard());
    quad_indices.emplace_back(get_quad_index_velocity_overintegration());
    quad_indices.emplace_back(get_quad_index_velocity_nodal_points());

    interface_data_dirichlet_cached = std::make_shared<ContainerInterfaceData<1, dim, double>>();
    interface_data_dirichlet_cached->setup(*matrix_free,
                                           get_dof_index_velocity(),
                                           quad_indices,
                                           boundary_descriptor->velocity->dirichlet_cached_bc);

    boundary_descriptor->velocity->set_dirichlet_cached_data(interface_data_dirichlet_cached);
  }
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::initialize_operators(std::string const & dof_index_temperature)
{
  // mass operator
  MassOperatorData<dim, Number> mass_operator_data;
  mass_operator_data.dof_index  = get_dof_index_velocity();
  mass_operator_data.quad_index = get_quad_index_velocity_standard();
  mass_operator.initialize(*matrix_free, constraint_u, mass_operator_data);

  // inverse mass operator velocity
  InverseMassOperatorData<Number> inverse_mass_operator_data_velocity;
  inverse_mass_operator_data_velocity.dof_index  = get_dof_index_velocity();
  inverse_mass_operator_data_velocity.quad_index = get_quad_index_velocity_standard();
  inverse_mass_operator_data_velocity.parameters = param.inverse_mass_operator;
  // avoid invalid settings for HDIV, preserving settings if admissible
  if(param.spatial_discretization == SpatialDiscretization::HDIV)
    inverse_mass_operator_data_velocity.parameters.implementation_type =
      InverseMassType::GlobalKrylovSolver;
  inverse_mass_velocity.initialize(*matrix_free,
                                   inverse_mass_operator_data_velocity,
                                   param.spatial_discretization == SpatialDiscretization::L2 ?
                                     nullptr :
                                     &constraint_u);

  // inverse mass operator velocity scalar
  InverseMassOperatorData<Number> inverse_mass_operator_data_velocity_scalar;
  inverse_mass_operator_data_velocity_scalar.dof_index  = get_dof_index_velocity_scalar();
  inverse_mass_operator_data_velocity_scalar.quad_index = get_quad_index_velocity_standard();
  inverse_mass_operator_data_velocity_scalar.parameters = param.inverse_mass_operator;
  // always use optimal inverse mass type for velocity scalar
  inverse_mass_operator_data_velocity_scalar.parameters.implementation_type =
    inverse_mass_operator_data_velocity_scalar.get_optimal_inverse_mass_type(
      matrix_free->get_dof_handler(get_dof_index_velocity_scalar()).get_fe());
  inverse_mass_velocity_scalar.initialize(*matrix_free, inverse_mass_operator_data_velocity_scalar);

  // body force operator
  RHSOperatorData<dim> rhs_data;
  rhs_data.dof_index = get_dof_index_velocity();
  if(param.boussinesq_term)
    rhs_data.dof_index_scalar = matrix_free_data->get_dof_index(dof_index_temperature);
  rhs_data.quad_index                                = get_quad_index_velocity_standard();
  rhs_data.kernel_data.f                             = field_functions->right_hand_side;
  rhs_data.kernel_data.boussinesq_term               = param.boussinesq_term;
  rhs_data.kernel_data.boussinesq_dynamic_part_only  = param.boussinesq_dynamic_part_only;
  rhs_data.kernel_data.thermal_expansion_coefficient = param.thermal_expansion_coefficient;
  rhs_data.kernel_data.reference_temperature         = param.reference_temperature;
  rhs_data.kernel_data.gravitational_force           = field_functions->gravitational_force;

  rhs_operator.initialize(*matrix_free, rhs_data);

  // gradient operator
  GradientOperatorData<dim> gradient_operator_data;
  gradient_operator_data.dof_index_velocity   = get_dof_index_velocity();
  gradient_operator_data.dof_index_pressure   = get_dof_index_pressure();
  gradient_operator_data.quad_index           = get_quad_index_velocity_standard();
  gradient_operator_data.integration_by_parts = param.gradp_integrated_by_parts;
  gradient_operator_data.formulation          = param.gradp_formulation;
  gradient_operator_data.use_boundary_data    = param.gradp_use_boundary_data;
  gradient_operator_data.bc                   = boundary_descriptor->pressure;
  gradient_operator.initialize(*matrix_free, gradient_operator_data);

  // divergence operator
  DivergenceOperatorData<dim> divergence_operator_data;
  divergence_operator_data.dof_index_velocity   = get_dof_index_velocity();
  divergence_operator_data.dof_index_pressure   = get_dof_index_pressure();
  divergence_operator_data.quad_index           = get_quad_index_velocity_standard();
  divergence_operator_data.integration_by_parts = param.divu_integrated_by_parts;
  divergence_operator_data.formulation          = param.divu_formulation;
  divergence_operator_data.use_boundary_data    = param.divu_use_boundary_data;
  divergence_operator_data.bc                   = boundary_descriptor->velocity;
  divergence_operator.initialize(*matrix_free, divergence_operator_data);

  // convective operator
  convective_kernel_data.formulation        = param.formulation_convective_term;
  convective_kernel_data.temporal_treatment = param.treatment_of_convective_term;
  convective_kernel_data.upwind_factor      = param.upwind_factor;
  convective_kernel_data.use_outflow_bc     = param.use_outflow_bc_convective_term;
  convective_kernel_data.type_dirichlet_bc  = param.type_dirichlet_bc_convective;
  convective_kernel_data.ale                = param.ale_formulation;
  convective_kernel = std::make_shared<Operators::ConvectiveKernel<dim, Number>>();
  convective_kernel->reinit(*matrix_free,
                            convective_kernel_data,
                            get_dof_index_velocity(),
                            get_quad_index_velocity_linearized(),
                            false /* is_mg =  use_velocity_own_storage */);

  dealii::AffineConstraints<Number> constraint_dummy;
  constraint_dummy.close();

  ConvectiveOperatorData<dim> convective_operator_data;
  convective_operator_data.kernel_data          = convective_kernel_data;
  convective_operator_data.dof_index            = get_dof_index_velocity();
  convective_operator_data.quad_index           = this->get_quad_index_velocity_linearized();
  convective_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
  convective_operator_data.quad_index_nonlinear = get_quad_index_velocity_overintegration();
  convective_operator_data.bc                   = boundary_descriptor->velocity;
  convective_operator.initialize(*matrix_free,
                                 constraint_dummy,
                                 convective_operator_data,
                                 convective_kernel);

  // viscous operator
  viscous_kernel_data.IP_factor                    = param.IP_factor_viscous;
  viscous_kernel_data.viscosity                    = param.viscosity;
  viscous_kernel_data.formulation_viscous_term     = param.formulation_viscous_term;
  viscous_kernel_data.penalty_term_div_formulation = param.penalty_term_div_formulation;
  viscous_kernel_data.IP_formulation               = param.IP_formulation_viscous;
  viscous_kernel_data.viscosity_is_variable        = param.viscosity_is_variable();
  viscous_kernel_data.variable_normal_vector       = param.neumann_with_variable_normal_vector;

  bool const use_velocity_own_storage_viscous_kernel = param.viscous_problem() and
                                                       param.viscosity_is_variable() and
                                                       not param.non_explicit_convective_problem();

  viscous_kernel = std::make_shared<Operators::ViscousKernel<dim, Number>>();
  viscous_kernel->reinit(*matrix_free,
                         viscous_kernel_data,
                         get_dof_index_velocity(),
                         get_quad_index_velocity_standard(),
                         use_velocity_own_storage_viscous_kernel);

  // initialize and check turbulence model data
  if(param.turbulence_model_data.is_active)
  {
    turbulence_model.initialize(*matrix_free,
                                *get_mapping(),
                                viscous_kernel,
                                param.turbulence_model_data,
                                get_dof_index_velocity());
  }

  // initialize and check generalized Newtonian model data
  if(param.generalized_newtonian_model_data.is_active)
  {
    generalized_newtonian_model.initialize(*matrix_free,
                                           viscous_kernel,
                                           param.generalized_newtonian_model_data,
                                           get_dof_index_velocity());
  }

  ViscousOperatorData<dim> viscous_operator_data;
  viscous_operator_data.kernel_data          = viscous_kernel_data;
  viscous_operator_data.bc                   = boundary_descriptor->velocity;
  viscous_operator_data.dof_index            = get_dof_index_velocity();
  viscous_operator_data.quad_index           = get_quad_index_velocity_standard();
  viscous_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
  viscous_operator.initialize(*matrix_free,
                              constraint_dummy,
                              viscous_operator_data,
                              viscous_kernel);

  // Momentum operator
  MomentumOperatorData<dim> data;

  data.unsteady_problem   = unsteady_problem_has_to_be_solved();
  data.convective_problem = param.non_explicit_convective_problem();
  data.viscous_problem    = param.viscous_problem();

  data.convective_kernel_data = convective_kernel_data;
  data.viscous_kernel_data    = viscous_kernel_data;

  data.turbulence_model_data            = param.turbulence_model_data;
  data.generalized_newtonian_model_data = param.generalized_newtonian_model_data;

  data.bc = boundary_descriptor->velocity;

  data.dof_index  = get_dof_index_velocity();
  data.quad_index = get_quad_index_velocity_linearized();

  data.use_cell_based_loops = param.use_cell_based_face_loops;
  data.implement_block_diagonal_preconditioner_matrix_free =
    param.implement_block_diagonal_preconditioner_matrix_free;
  if(data.convective_problem)
    data.solver_block_diagonal = Elementwise::Solver::GMRES;
  else
    data.solver_block_diagonal = Elementwise::Solver::CG;
  data.preconditioner_block_diagonal = Elementwise::Preconditioner::InverseMassMatrix;
  data.solver_data_block_diagonal    = param.solver_data_block_diagonal;

  momentum_operator.initialize(
    *matrix_free, constraint_dummy, data, viscous_kernel, convective_kernel);

  if(param.use_divergence_penalty)
  {
    // Kernel
    Operators::DivergencePenaltyKernelData div_penalty_data;
    div_penalty_data.type_penalty_parameter = param.type_penalty_parameter;
    div_penalty_data.viscosity              = param.viscosity;
    div_penalty_data.degree                 = param.degree_u;
    div_penalty_data.penalty_factor         = param.divergence_penalty_factor;

    div_penalty_kernel = std::make_shared<Operators::DivergencePenaltyKernel<dim, Number>>();
    div_penalty_kernel->reinit(*matrix_free,
                               get_dof_index_velocity(),
                               get_quad_index_velocity_standard(),
                               div_penalty_data);

    // Operator
    DivergencePenaltyData operator_data;
    operator_data.dof_index  = get_dof_index_velocity();
    operator_data.quad_index = get_quad_index_velocity_standard();

    div_penalty_operator.initialize(*matrix_free, operator_data, div_penalty_kernel);
  }

  if(param.use_continuity_penalty)
  {
    // Kernel
    Operators::ContinuityPenaltyKernelData kernel_data;

    kernel_data.type_penalty_parameter = param.type_penalty_parameter;
    kernel_data.which_components       = param.continuity_penalty_components;
    kernel_data.viscosity              = param.viscosity;
    kernel_data.degree                 = param.degree_u;
    kernel_data.penalty_factor         = param.continuity_penalty_factor;

    conti_penalty_kernel = std::make_shared<Operators::ContinuityPenaltyKernel<dim, Number>>();
    conti_penalty_kernel->reinit(*matrix_free,
                                 get_dof_index_velocity(),
                                 get_quad_index_velocity_standard(),
                                 kernel_data);

    // Operator
    ContinuityPenaltyData<dim> operator_data;
    operator_data.dof_index         = get_dof_index_velocity();
    operator_data.quad_index        = get_quad_index_velocity_standard();
    operator_data.use_boundary_data = param.continuity_penalty_use_boundary_data;
    operator_data.bc                = this->boundary_descriptor->velocity;

    conti_penalty_operator.initialize(*matrix_free, operator_data, conti_penalty_kernel);
  }

  if(param.use_divergence_penalty or param.use_continuity_penalty)
  {
    if(param.temporal_discretization == TemporalDiscretization::BDFDualSplitting or
       param.temporal_discretization == TemporalDiscretization::BDFConsistentSplitting or
       param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection or
       (param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution and
        param.apply_penalty_terms_in_postprocessing_step == true))
    {
      // setup projection operator
      ProjectionOperatorData<dim> data;
      data.use_divergence_penalty = param.use_divergence_penalty;
      data.use_continuity_penalty = param.use_continuity_penalty;
      data.use_boundary_data      = param.continuity_penalty_use_boundary_data;
      data.bc                     = this->boundary_descriptor->velocity;
      data.dof_index              = get_dof_index_velocity();
      data.quad_index             = get_quad_index_velocity_standard();
      data.use_cell_based_loops   = param.use_cell_based_face_loops;
      data.implement_block_diagonal_preconditioner_matrix_free =
        param.implement_block_diagonal_preconditioner_matrix_free;
      data.solver_block_diagonal         = Elementwise::Solver::CG;
      data.preconditioner_block_diagonal = param.preconditioner_block_diagonal_projection;
      data.solver_data_block_diagonal    = param.solver_data_block_diagonal_projection;

      projection_operator = std::make_shared<ProjOperator>();

      projection_operator->initialize(
        *matrix_free, constraint_dummy, data, div_penalty_kernel, conti_penalty_kernel);
    }
  }
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::initialize_calculators_for_derived_quantities()
{
  vorticity_calculator.initialize(*matrix_free,
                                  get_dof_index_velocity(),
                                  get_quad_index_velocity_standard());
  divergence_calculator.initialize(*matrix_free,
                                   get_dof_index_velocity(),
                                   get_dof_index_velocity_scalar(),
                                   get_quad_index_velocity_standard());
  shear_rate_calculator.initialize(*matrix_free,
                                   get_dof_index_velocity(),
                                   get_dof_index_velocity_scalar(),
                                   get_quad_index_velocity_standard());
  viscosity_calculator.initialize(*matrix_free,
                                  get_dof_index_velocity_scalar(),
                                  get_quad_index_velocity_standard(),
                                  *viscous_kernel);
  magnitude_calculator.initialize(*matrix_free,
                                  get_dof_index_velocity(),
                                  get_dof_index_velocity_scalar(),
                                  get_quad_index_velocity_standard());
  q_criterion_calculator.initialize(*matrix_free,
                                    get_dof_index_velocity(),
                                    get_dof_index_velocity_scalar(),
                                    get_quad_index_velocity_standard(),
                                    false /*compressible_flow*/);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::setup()
{
  // initialize MatrixFree and MatrixFreeData
  std::shared_ptr<dealii::MatrixFree<dim, Number>> mf =
    std::make_shared<dealii::MatrixFree<dim, Number>>();
  std::shared_ptr<MatrixFreeData<dim, Number>> mf_data =
    std::make_shared<MatrixFreeData<dim, Number>>();

  fill_matrix_free_data(*mf_data);

  if(param.use_cell_based_face_loops)
    Categorization::do_cell_based_loops(*grid->triangulation, mf_data->data);
  mf->reinit(*get_mapping(),
             mf_data->get_dof_handler_vector(),
             mf_data->get_constraint_vector(),
             mf_data->get_quadrature_vector(),
             mf_data->data);

  if(param.ale_formulation)
    matrix_free_own_storage = mf;

  // Subsequently, call the other setup function with MatrixFree/MatrixFreeData objects as
  // arguments.
  this->setup(mf, mf_data);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::setup(
  std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free_in,
  std::shared_ptr<MatrixFreeData<dim, Number> const>     matrix_free_data_in,
  std::string const &                                    dof_index_temperature)
{
  pcout << std::endl
        << "Setup incompressible Navier-Stokes operator ..." << std::endl
        << std::flush;

  // MatrixFree
  matrix_free      = matrix_free_in;
  matrix_free_data = matrix_free_data_in;

  // Next, initialize data structures depending on MatrixFree:

  initialize_dirichlet_cached_bc();

  initialize_operators(dof_index_temperature);

  initialize_calculators_for_derived_quantities();

  // Finally, do set up of derived classes
  setup_derived();

  setup_preconditioners_and_solvers();

  pcout << std::endl << "... done!" << std::endl << std::flush;
}

template<int dim, typename Number>
dealii::types::global_dof_index
SpatialOperatorBase<dim, Number>::get_number_of_dofs() const
{
  return dof_handler_u.n_dofs() + dof_handler_p.n_dofs();
}

template<int dim, typename Number>
dealii::MatrixFree<dim, Number> const &
SpatialOperatorBase<dim, Number>::get_matrix_free() const
{
  return *matrix_free;
}

template<int dim, typename Number>
std::string
SpatialOperatorBase<dim, Number>::get_dof_name_velocity() const
{
  return field + dof_index_u;
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::get_dof_index_velocity() const
{
  return matrix_free_data->get_dof_index(get_dof_name_velocity());
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::get_dof_index_pressure() const
{
  return matrix_free_data->get_dof_index(field + dof_index_p);
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::get_dof_index_velocity_scalar() const
{
  return matrix_free_data->get_dof_index(field + dof_index_u_scalar);
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::get_quad_index_velocity_standard() const
{
  return matrix_free_data->get_quad_index(field + quad_index_u);
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::get_quad_index_pressure() const
{
  return matrix_free_data->get_quad_index(field + quad_index_p);
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::get_quad_index_velocity_overintegration() const
{
  return matrix_free_data->get_quad_index(field + quad_index_u_overintegration);
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::get_quad_index_velocity_nodal_points() const
{
  return matrix_free_data->get_quad_index(field + quad_index_u_nodal_points);
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::get_quad_index_pressure_nodal_points() const
{
  return matrix_free_data->get_quad_index(field + quad_index_p_nodal_points);
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::get_quad_index_velocity_linearized() const
{
  if(param.quad_rule_linearization == QuadratureRuleLinearization::Standard)
  {
    return get_quad_index_velocity_standard();
  }
  else if(param.quad_rule_linearization == QuadratureRuleLinearization::Overintegration32k)
  {
    if(param.non_explicit_convective_problem())
      return get_quad_index_velocity_overintegration();
    else
      return get_quad_index_velocity_standard();
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented"));
    return get_quad_index_velocity_overintegration();
  }
}

template<int dim, typename Number>
std::shared_ptr<dealii::Mapping<dim> const>
SpatialOperatorBase<dim, Number>::get_mapping() const
{
  return mapping;
}

template<int dim, typename Number>
dealii::FiniteElement<dim> const &
SpatialOperatorBase<dim, Number>::get_fe_u() const
{
  return *fe_u;
}

template<int dim, typename Number>
dealii::FiniteElement<dim> const &
SpatialOperatorBase<dim, Number>::get_fe_p() const
{
  return *fe_p;
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
SpatialOperatorBase<dim, Number>::get_dof_handler_u() const
{
  return dof_handler_u;
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
SpatialOperatorBase<dim, Number>::get_dof_handler_u_scalar() const
{
  return dof_handler_u_scalar;
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
SpatialOperatorBase<dim, Number>::get_dof_handler_p() const
{
  return dof_handler_p;
}

template<int dim, typename Number>
dealii::AffineConstraints<Number> const &
SpatialOperatorBase<dim, Number>::get_constraint_p() const
{
  return constraint_p;
}

template<int dim, typename Number>
dealii::AffineConstraints<Number> const &
SpatialOperatorBase<dim, Number>::get_constraint_u() const
{
  return constraint_u;
}

template<int dim, typename Number>
dealii::VectorizedArray<Number>
SpatialOperatorBase<dim, Number>::get_viscosity_boundary_face(unsigned int const face,
                                                              unsigned int const q) const
{
  if(param.viscosity_is_variable())
  {
    return viscous_kernel->get_coefficient_face(face, q);
  }
  else
  {
    return dealii::make_vectorized_array<Number>(param.viscosity);
  }
}

template<int dim, typename Number>
std::shared_ptr<ContainerInterfaceData<1, dim, double>>
SpatialOperatorBase<dim, Number>::get_container_interface_data()
{
  return interface_data_dirichlet_cached;
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::set_velocity_ptr(VectorType const & velocity) const
{
  convective_kernel->set_velocity_ptr(velocity);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::initialize_vector_velocity(VectorType & src) const
{
  matrix_free->initialize_dof_vector(src, get_dof_index_velocity());
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::initialize_vector_velocity_scalar(VectorType & src) const
{
  matrix_free->initialize_dof_vector(src, get_dof_index_velocity_scalar());
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::initialize_vector_pressure(VectorType & src) const
{
  matrix_free->initialize_dof_vector(src, get_dof_index_pressure());
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::initialize_block_vector_velocity_pressure(
  BlockVectorType & src) const
{
  // velocity (1st block) + pressure (2nd block)
  src.reinit(2);

  matrix_free->initialize_dof_vector(src.block(0), get_dof_index_velocity());
  matrix_free->initialize_dof_vector(src.block(1), get_dof_index_pressure());

  src.collect_sizes();
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::interpolate_functions(
  VectorType &                                   velocity,
  std::shared_ptr<dealii::Function<dim>> const & f_velocity,
  VectorType &                                   pressure,
  std::shared_ptr<dealii::Function<dim>> const & f_pressure,
  double const                                   time) const
{
  AssertThrow(f_velocity, dealii::ExcMessage("Function not set"));
  AssertThrow(f_pressure, dealii::ExcMessage("Function not set"));

  Utilities::interpolate(*get_mapping(), dof_handler_u, *f_velocity, velocity, time);
  Utilities::interpolate(*get_mapping(), dof_handler_p, *f_pressure, pressure, time);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::prescribe_initial_conditions(VectorType & velocity,
                                                               VectorType & pressure,
                                                               double const time) const
{
  interpolate_functions(velocity,
                        field_functions->initial_solution_velocity,
                        pressure,
                        field_functions->initial_solution_pressure,
                        time);

  // Compute initial variable viscosity using the initial velocity field.
  if(this->param.viscous_problem() and this->param.viscosity_is_variable())
  {
    this->update_viscosity(velocity);
  }
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::serialize_vectors(
  std::vector<VectorType const *> & vectors_velocity,
  std::vector<VectorType const *> & vectors_pressure) const
{
  // Write deserialization parameters. These do not change during the simulation, but the data are
  // small and we want to make sure to overwrite them.
  DeserializationParameters deserialization_parameters;
  deserialization_parameters.degree_u               = param.degree_u;
  deserialization_parameters.degree_p               = param.get_degree_p(param.degree_u);
  deserialization_parameters.mapping_degree         = param.mapping_degree;
  deserialization_parameters.consider_mapping_write = param.restart_data.consider_mapping_write;
  deserialization_parameters.triangulation_type     = param.grid.triangulation_type;
  deserialization_parameters.spatial_discretization = param.spatial_discretization;
  write_deserialization_parameters(mpi_comm, param.restart_data, deserialization_parameters);

  // Attach vectors to triangulation and serialize.
  std::vector<dealii::DoFHandler<dim> const *> dof_handlers{&dof_handler_u, &dof_handler_p};
  std::vector<std::vector<VectorType const *>> vectors_per_dof_handler{vectors_velocity,
                                                                       vectors_pressure};
  if(param.restart_data.consider_mapping_write)
  {
    store_vectors_in_triangulation_and_serialize(param.restart_data,
                                                 dof_handlers,
                                                 vectors_per_dof_handler,
                                                 *this->get_mapping(),
                                                 dof_handler_mapping.get(),
                                                 param.mapping_degree);
  }
  else
  {
    store_vectors_in_triangulation_and_serialize(param.restart_data,
                                                 dof_handlers,
                                                 vectors_per_dof_handler);
  }
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::deserialize_vectors(std::vector<VectorType *> & vectors_velocity,
                                                      std::vector<VectorType *> & vectors_pressure)
{
  // Store ghost state to recover after deserialization.
  std::vector<bool> const has_ghost_elements_velocity = get_ghost_state(vectors_velocity);
  std::vector<bool> const has_ghost_elements_pressure = get_ghost_state(vectors_pressure);

  // Load the deserialization parameters.
  DeserializationParameters const deserialization_parameters =
    read_deserialization_parameters(mpi_comm, param.restart_data);

  // Load potentially unfitting checkpoint triangulation of TriangulationType.
  std::shared_ptr<dealii::Triangulation<dim>> checkpoint_triangulation =
    deserialize_triangulation<dim>(param.restart_data,
                                   deserialization_parameters.triangulation_type,
                                   mpi_comm);

  // Set up DoFHandlers *as checkpointed*, sequence matches `this->serialize_vectors()`.
  dealii::DoFHandler<dim> checkpoint_dof_handler_u(*checkpoint_triangulation);
  dealii::DoFHandler<dim> checkpoint_dof_handler_p(*checkpoint_triangulation);

  ElementType const checkpoint_element_type = get_element_type(*checkpoint_triangulation);

  std::shared_ptr<dealii::FiniteElement<dim>> checkpoint_fe_u =
    setup_fe_u(deserialization_parameters.spatial_discretization,
               checkpoint_element_type,
               deserialization_parameters.degree_u);

  std::shared_ptr<dealii::FiniteElement<dim>> checkpoint_fe_p =
    create_finite_element<dim>(checkpoint_element_type,
                               true /* is_dg */,
                               1 /* n_components */,
                               deserialization_parameters.degree_p);

  checkpoint_dof_handler_u.distribute_dofs(*checkpoint_fe_u);
  checkpoint_dof_handler_p.distribute_dofs(*checkpoint_fe_p);

  std::vector<dealii::DoFHandler<dim> const *> checkpoint_dof_handlers{&checkpoint_dof_handler_u,
                                                                       &checkpoint_dof_handler_p};

  // Deserialize vectors stored in triangulation, sequence matches `this->serialize_vectors()`.
  std::vector<VectorType>                checkpoint_vectors_velocity(vectors_velocity.size());
  std::vector<VectorType>                checkpoint_vectors_pressure(vectors_pressure.size());
  std::vector<std::vector<VectorType *>> checkpoint_vectors(2);

  for(unsigned int i = 0; i < vectors_velocity.size(); ++i)
  {
    checkpoint_vectors_velocity[i].reinit(checkpoint_dof_handler_u.locally_owned_dofs(), mpi_comm);
    checkpoint_vectors[0 /* velocity */].push_back(&checkpoint_vectors_velocity[i]);
  }
  for(unsigned int i = 0; i < vectors_pressure.size(); ++i)
  {
    checkpoint_vectors_pressure[i].reinit(checkpoint_dof_handler_p.locally_owned_dofs(), mpi_comm);
    checkpoint_vectors[1 /* pressure */].push_back(&checkpoint_vectors_pressure[i]);
  }

  if(param.restart_data.discretization_identical)
  {
    // DoFHandlers need to be setup with `checkpoint_triangulation`, otherwise
    // they are identical. We can simply copy the vector contents.
    load_vectors(checkpoint_vectors, checkpoint_dof_handlers);
    for(unsigned int i = 0; i < vectors_velocity.size(); ++i)
    {
      vectors_velocity[i]->copy_locally_owned_data_from(checkpoint_vectors_velocity[i]);
    }
    for(unsigned int i = 0; i < vectors_pressure.size(); ++i)
    {
      vectors_pressure[i]->copy_locally_owned_data_from(checkpoint_vectors_pressure[i]);
    }
  }
  else
  {
    // Perform global projection in case of a non-matching discretization.
    std::vector<dealii::DoFHandler<dim> const *> dof_handlers{&this->get_dof_handler_u(),
                                                              &this->get_dof_handler_p()};

    std::vector<std::vector<VectorType *>> vectors_per_dof_handler(2);
    for(unsigned int i = 0; i < vectors_velocity.size(); ++i)
    {
      vectors_per_dof_handler[0 /* velocity */].push_back(vectors_velocity[i]);
    }
    for(unsigned int i = 0; i < vectors_pressure.size(); ++i)
    {
      vectors_per_dof_handler[1 /* pressure */].push_back(vectors_pressure[i]);
    }

    // Deserialize mapping from vector or project on reference triangulations.
    check_mapping_deserialization(param.restart_data.consider_mapping_read_source,
                                  deserialization_parameters.consider_mapping_write);
    std::shared_ptr<dealii::Mapping<dim> const> checkpoint_mapping;
    std::shared_ptr<MappingDoFVector<dim, typename VectorType::value_type>>
      checkpoint_mapping_dof_vector;
    if(param.restart_data.consider_mapping_read_source)
    {
      dealii::DoFHandler<dim> checkpoint_dof_handler_mapping(*checkpoint_triangulation);
      std::shared_ptr<dealii::FiniteElement<dim>> checkpoint_fe_mapping =
        create_finite_element<dim>(checkpoint_element_type,
                                   true /* is_dg */,
                                   dim /* n_components */,
                                   deserialization_parameters.mapping_degree);
      checkpoint_dof_handler_mapping.distribute_dofs(*checkpoint_fe_mapping);

      checkpoint_mapping_dof_vector = load_vectors(checkpoint_vectors,
                                                   checkpoint_dof_handlers,
                                                   &checkpoint_dof_handler_mapping,
                                                   deserialization_parameters.mapping_degree);

      checkpoint_mapping = checkpoint_mapping_dof_vector->get_mapping();
    }
    else
    {
      load_vectors(checkpoint_vectors, checkpoint_dof_handlers);

      // Create dummy linear mapping since we have no mapping serialized to restore.
      std::shared_ptr<dealii::Mapping<dim>> tmp;
      GridUtilities::create_mapping(tmp,
                                    get_element_type(*checkpoint_triangulation),
                                    1 /* mapping_degree */);
      checkpoint_mapping = std::const_pointer_cast<dealii::Mapping<dim> const>(tmp);
    }

    ExaDG::GridToGridProjection::GridToGridProjectionData<dim> data;
    data.rpe_data.rtree_level            = param.restart_data.rpe_rtree_level;
    data.rpe_data.tolerance              = param.restart_data.rpe_tolerance_unit_cell;
    data.rpe_data.enforce_unique_mapping = param.restart_data.rpe_enforce_unique_mapping;

    ExaDG::GridToGridProjection::do_grid_to_grid_projection<dim, Number, VectorType>(
      checkpoint_mapping,
      checkpoint_dof_handlers,
      checkpoint_vectors,
      dof_handlers,
      *matrix_free,
      vectors_per_dof_handler,
      data);
  }

  // Recover ghost vector state.
  set_ghost_state(vectors_velocity, has_ghost_elements_velocity);
  set_ghost_state(vectors_pressure, has_ghost_elements_pressure);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::interpolate_analytical_solution(VectorType & velocity,
                                                                  VectorType & pressure,
                                                                  double const time) const
{
  interpolate_functions(velocity,
                        field_functions->analytical_solution_velocity,
                        pressure,
                        field_functions->analytical_solution_pressure,
                        time);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::interpolate_stress_bc(VectorType &       stress,
                                                        VectorType const & velocity,
                                                        VectorType const & pressure) const
{
  velocity_ptr = &velocity;
  pressure_ptr = &pressure;

  stress = 0.0;

  VectorType src_dummy;
  matrix_free->loop(&This::cell_loop_empty,
                    &This::face_loop_empty,
                    &This::local_interpolate_stress_bc_boundary_face,
                    this,
                    stress,
                    src_dummy);

  velocity_ptr = nullptr;
  pressure_ptr = nullptr;
}

template<int dim, typename Number>
std::shared_ptr<dealii::FiniteElement<dim>>
SpatialOperatorBase<dim, Number>::setup_fe_u(SpatialDiscretization const spatial_discretization,
                                             ElementType const           element_type,
                                             unsigned int const          degree) const
{
  std::shared_ptr<dealii::FiniteElement<dim>> fe;
  if(spatial_discretization == SpatialDiscretization::L2)
  {
    fe = create_finite_element<dim>(element_type, true /* is_dg */, dim /* n_components */, degree);
  }
  else if(spatial_discretization == SpatialDiscretization::HDIV)
  {
    AssertThrow(
      element_type == ElementType::Hypercube,
      dealii::ExcMessage(
        "SpatialDiscretization::HDIV is currently only implemented for hypercube elements. "
        "You might want to change the element type of the grid, or the function space, "
        "or implement HDIV for element types other than hypercube."));

    // The constructor of FE_RaviartThomas takes the degree in tangential direction as an
    // argument.
    fe = std::make_shared<dealii::FE_RaviartThomasNodal<dim>>(degree - 1);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("FE not implemented."));
  }

  return fe;
}

template<int dim, typename Number>
double
SpatialOperatorBase<dim, Number>::calculate_minimum_element_length() const
{
  return calculate_minimum_vertex_distance(dof_handler_u.get_triangulation(),
                                           *get_mapping(),
                                           mpi_comm);
}

template<int dim, typename Number>
double
SpatialOperatorBase<dim, Number>::calculate_time_step_max_efficiency(
  unsigned int const order_time_integrator) const
{
  double const h_min = calculate_minimum_element_length();

  return ExaDG::calculate_time_step_max_efficiency(h_min, param.degree_u, order_time_integrator);
}

template<int dim, typename Number>
double
SpatialOperatorBase<dim, Number>::calculate_time_step_cfl_global() const
{
  std::shared_ptr<dealii::Function<dim>> const velocity_field =
    std::make_shared<dealii::Functions::ConstantFunction<dim>>(param.max_velocity, dim);

  return calculate_time_step_cfl_local<dim, Number>(
    *matrix_free,
    get_dof_index_velocity(),
    get_quad_index_velocity_standard(),
    velocity_field,
    param.start_time /* will not be used (ConstantFunction) */,
    param.degree_u,
    param.cfl_exponent_fe_degree_velocity,
    CFLConditionType::VelocityComponents,
    mpi_comm);
}

template<int dim, typename Number>
double
SpatialOperatorBase<dim, Number>::calculate_time_step_cfl(VectorType const & velocity) const
{
  // Need to update ghost values in the case of continuity constraints.
  if(param.spatial_discretization == SpatialDiscretization::HDIV)
    velocity.update_ghost_values();

  return calculate_time_step_cfl_local<dim, Number>(*matrix_free,
                                                    get_dof_index_velocity(),
                                                    get_quad_index_velocity_standard(),
                                                    velocity,
                                                    param.degree_u,
                                                    param.cfl_exponent_fe_degree_velocity,
                                                    param.adaptive_time_stepping_cfl_type,
                                                    mpi_comm);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::calculate_cfl_from_time_step(VectorType &       cfl,
                                                               VectorType const & velocity,
                                                               double const time_step_size) const
{
  calculate_cfl<dim, Number>(cfl,
                             *grid->triangulation,
                             *matrix_free,
                             get_dof_index_velocity(),
                             get_quad_index_velocity_standard(),
                             velocity,
                             time_step_size,
                             param.degree_u,
                             param.cfl_exponent_fe_degree_velocity);
}

template<int dim, typename Number>
double
SpatialOperatorBase<dim, Number>::get_characteristic_element_length() const
{
  double const h_min = calculate_minimum_element_length();

  return calculate_high_order_element_length(h_min, param.degree_u, true /* is_dg */);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::apply_mass_operator(VectorType &       dst,
                                                      VectorType const & src) const
{
  mass_operator.apply(dst, src);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::apply_mass_operator_add(VectorType &       dst,
                                                          VectorType const & src) const
{
  mass_operator.apply_add(dst, src);
}

template<int dim, typename Number>
bool
SpatialOperatorBase<dim, Number>::is_pressure_level_undefined() const
{
  return pressure_level_is_undefined;
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::adjust_pressure_level_if_undefined(VectorType &   pressure,
                                                                     double const & time) const
{
  if(is_pressure_level_undefined())
  {
    // If an analytical solution is available: shift pressure so that the numerical pressure
    // solution coincides with the analytical pressure solution in an arbitrary point. Note that the
    // parameter 'time' is only needed for unsteady problems.
    if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalSolutionInPoint)
    {
      field_functions->analytical_solution_pressure->set_time(time);
      double const exact = field_functions->analytical_solution_pressure->value(first_point);

      double current = -std::numeric_limits<double>::max();
      if(pressure.locally_owned_elements().is_element(dof_index_first_point))
        current = pressure(dof_index_first_point);
      current = dealii::Utilities::MPI::max(current, mpi_comm);

      pressure.add(exact - current);
    }
    else if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyZeroMeanValue)
    {
      dealii::VectorTools::subtract_mean_value(pressure);
    }
    // If an analytical solution is available: shift pressure so that the numerical pressure
    // solution has a mean value identical to the "exact pressure solution" obtained by
    // interpolation of analytical solution. Note that the parameter 'time' is only needed for
    // unsteady problems.
    else if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalMeanValue)
    {
      // one cannot use Number as template here since Number might be float
      // while analytical_solution_pressure is of type dealii::Function<dim,double>
      typedef dealii::LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

      VectorTypeDouble vec_double;
      vec_double = pressure; // initialize

      field_functions->analytical_solution_pressure->set_time(time);
      dealii::VectorTools::interpolate(*get_mapping(),
                                       dof_handler_p,
                                       *(field_functions->analytical_solution_pressure),
                                       vec_double);

      double const exact   = vec_double.mean_value();
      double const current = pressure.mean_value();

      pressure.add(exact - current);
    }
    else
    {
      AssertThrow(
        false, dealii::ExcMessage("Specified method to adjust pressure level is not implemented."));
    }
  }
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::set_temperature(VectorType const & temperature)
{
  AssertThrow(param.boussinesq_term, dealii::ExcMessage("Invalid parameters detected."));

  rhs_operator.set_temperature(temperature);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::compute_vorticity(VectorType & dst, VectorType const & src) const
{
  vorticity_calculator.compute_projection_rhs(dst, src);
  this->apply_inverse_mass_operator(dst, dst);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::compute_divergence(VectorType & dst, VectorType const & src) const
{
  divergence_calculator.compute_projection_rhs(dst, src);
  inverse_mass_velocity_scalar.apply(dst, dst);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::compute_shear_rate(VectorType & dst, VectorType const & src) const
{
  shear_rate_calculator.compute_projection_rhs(dst, src);
  inverse_mass_velocity_scalar.apply(dst, dst);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::access_viscosity(VectorType & dst) const
{
  if(param.viscosity_is_variable())
  {
    viscosity_calculator.compute_projection_rhs(dst);
    inverse_mass_velocity_scalar.apply(dst, dst);
  }
  else
  {
    dst = param.viscosity;
  }
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::compute_velocity_magnitude(VectorType &       dst,
                                                             VectorType const & src) const
{
  magnitude_calculator.compute_projection_rhs(dst, src);
  inverse_mass_velocity_scalar.apply(dst, dst);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::compute_vorticity_magnitude(VectorType &       dst,
                                                              VectorType const & src) const
{
  magnitude_calculator.compute_projection_rhs(dst, src);
  inverse_mass_velocity_scalar.apply(dst, dst);
}

/*
 *  Streamfunction psi (2D only): defined as u1 = d(psi)/dx2, u2 = - d(psi)/dx1
 *
 *  Vorticity: omega = du2/dx1 - du1/dx2
 *
 *  --> laplace(psi) = (d²/dx1²+d²/dx2²)(psi)
 *                   = d(d(psi)/dx1)/dx1 + d(d(psi)/dx2)/dx2
 *                   = d(-u2)/dx1 + d(u1)/dx2 = - omega
 *
 *  or
 *      - laplace(psi) = omega
 *
 *  with homogeneous Dirichlet BC's (assumption: whole boundary == streamline)
 */
template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::compute_streamfunction(VectorType &       dst,
                                                         VectorType const & src) const
{
  AssertThrow(dim == 2,
              dealii::ExcMessage("Calculation of streamfunction can only be used for dim==2."));

  // compute rhs vector
  StreamfunctionCalculatorRHSOperator<dim, Number> rhs_operator;
  rhs_operator.initialize(*matrix_free,
                          get_dof_index_velocity(),
                          get_dof_index_velocity_scalar(),
                          get_quad_index_velocity_standard());
  VectorType rhs;
  initialize_vector_velocity_scalar(rhs);
  rhs_operator.apply(rhs, src);

  // setup Laplace operator for scalar velocity vector
  Poisson::LaplaceOperatorData<0, dim> laplace_operator_data;
  laplace_operator_data.dof_index  = get_dof_index_velocity_scalar();
  laplace_operator_data.quad_index = get_quad_index_velocity_standard();

  std::shared_ptr<Poisson::BoundaryDescriptor<0, dim>> boundary_descriptor_streamfunction;
  boundary_descriptor_streamfunction = std::make_shared<Poisson::BoundaryDescriptor<0, dim>>();

  // fill boundary descriptor: Assumption: only Dirichlet BC's
  boundary_descriptor_streamfunction->dirichlet_bc = boundary_descriptor->velocity->dirichlet_bc;

  AssertThrow(boundary_descriptor->velocity->neumann_bc.empty() == true,
              dealii::ExcMessage("Assumption is not fulfilled. Streamfunction calculator is "
                                 "not implemented for this type of boundary conditions."));
  AssertThrow(boundary_descriptor->velocity->symmetry_bc.empty() == true,
              dealii::ExcMessage("Assumption is not fulfilled. Streamfunction calculator is "
                                 "not implemented for this type of boundary conditions."));

  laplace_operator_data.bc = boundary_descriptor_streamfunction;

  laplace_operator_data.kernel_data.IP_factor = 1.0;

  typedef Poisson::LaplaceOperator<dim, Number, 1> Laplace;
  Laplace                                          laplace_operator;
  dealii::AffineConstraints<Number>                constraint_dummy;
  laplace_operator.initialize(*matrix_free,
                              constraint_dummy,
                              laplace_operator_data,
                              true /* assemble_matrix */);

  // setup preconditioner
  std::shared_ptr<PreconditionerBase<Number>> preconditioner;

  // use multigrid preconditioner with Chebyshev smoother
  MultigridData mg_data;

  preconditioner = std::make_shared<MultigridPoisson>(mpi_comm);

  std::shared_ptr<MultigridPoisson> mg_preconditioner =
    std::dynamic_pointer_cast<MultigridPoisson>(preconditioner);

  typedef std::map<dealii::types::boundary_id, dealii::ComponentMask> Map_DBC_ComponentMask;
  Map_DBC_ComponentMask                                               dirichlet_bc_component_mask;

  mg_preconditioner->initialize(mg_data,
                                grid,
                                multigrid_mappings,
                                dof_handler_u_scalar.get_fe(),
                                laplace_operator.get_data(),
                                param.ale_formulation,
                                laplace_operator.get_data().bc->dirichlet_bc,
                                dirichlet_bc_component_mask);

  // setup solver
  Krylov::SolverDataCG solver_data;
  solver_data.solver_tolerance_rel = 1.e-10;
  solver_data.use_preconditioner   = true;

  Krylov::SolverCG<Laplace, PreconditionerBase<Number>, VectorType> poisson_solver(laplace_operator,
                                                                                   *preconditioner,
                                                                                   solver_data);

  // solve Poisson problem
  poisson_solver.solve(dst, rhs);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::compute_q_criterion(VectorType &       dst,
                                                      VectorType const & src) const
{
  q_criterion_calculator.compute_projection_rhs(dst, src);
  inverse_mass_velocity_scalar.apply(dst, dst);
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::apply_inverse_mass_operator(VectorType &       dst,
                                                              VectorType const & src) const
{
  inverse_mass_velocity.apply(dst, src);
  return 0;
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::evaluate_add_body_force_term(VectorType & dst,
                                                               double const time) const
{
  this->rhs_operator.evaluate_add(dst, time);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::evaluate_convective_term(VectorType &       dst,
                                                           VectorType const & src,
                                                           Number const       time) const
{
  convective_operator.evaluate_nonlinear_operator(dst, src, time);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::evaluate_pressure_gradient_term(VectorType &       dst,
                                                                  VectorType const & src,
                                                                  double const       time) const
{
  gradient_operator.evaluate(dst, src, time);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::evaluate_velocity_divergence_term(VectorType &       dst,
                                                                    VectorType const & src,
                                                                    double const       time) const
{
  divergence_operator.evaluate(dst, src, time);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::update_viscosity(VectorType const & velocity) const
{
  AssertThrow(param.viscous_problem() and param.viscosity_is_variable(),
              dealii::ExcMessage(
                "Updating viscosity reasonable for variable viscosity models only."));

  // update linearization velocity vector in `viscous_kernel`
  if(viscous_kernel->get_use_velocity_own_storage())
  {
    viscous_kernel->set_velocity_copy(velocity);
  }

  // reset the viscosity stored
  // viscosity = viscosity_newtonian_limit
  viscous_kernel->set_constant_coefficient(viscous_kernel_data.viscosity);

  // add contribution from generalized Newtonian model
  // viscosity += generalized_newtonian_viscosity(viscosity_newtonian_limit)
  if(param.generalized_newtonian_model_data.is_active)
  {
    generalized_newtonian_model.add_viscosity(velocity);
  }

  // add contribution from turbulence model
  // viscosity += turbulent_viscosity(viscosity)
  // note that the apparent viscosity is used to compute the turbulent viscosity, such that the
  // *sequence of calls matters*, i.e., we can only compute the turbulent viscosity once the laminar
  // viscosity has been computed
  if(param.turbulence_model_data.is_active)
  {
    turbulence_model.add_viscosity(velocity);
  }
}

template<int dim, typename Number>
double
SpatialOperatorBase<dim, Number>::calculate_dissipation_convective_term(VectorType const & velocity,
                                                                        double const time) const
{
  if(param.convective_problem())
  {
    VectorType dst;
    dst.reinit(velocity, false);
    convective_operator.evaluate_nonlinear_operator(dst, velocity, time);
    return velocity * dst;
  }
  else
  {
    return 0.0;
  }
}

template<int dim, typename Number>
double
SpatialOperatorBase<dim, Number>::calculate_dissipation_viscous_term(
  VectorType const & velocity) const
{
  if(param.viscous_problem())
  {
    VectorType dst;
    dst.reinit(velocity, false);
    viscous_operator.apply(dst, velocity);
    return velocity * dst;
  }
  else
  {
    return 0.0;
  }
}

template<int dim, typename Number>
double
SpatialOperatorBase<dim, Number>::calculate_dissipation_divergence_term(
  VectorType const & velocity) const
{
  if(param.use_divergence_penalty == true)
  {
    VectorType dst;
    dst.reinit(velocity, false);
    div_penalty_operator.apply(dst, velocity);
    return velocity * dst;
  }
  else
  {
    return 0.0;
  }
}

template<int dim, typename Number>
double
SpatialOperatorBase<dim, Number>::calculate_dissipation_continuity_term(
  VectorType const & velocity) const
{
  if(param.use_continuity_penalty == true)
  {
    VectorType dst;
    dst.reinit(velocity, false);
    conti_penalty_operator.apply(dst, velocity);
    return velocity * dst;
  }
  else
  {
    return 0.0;
  }
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::update_after_grid_motion(bool const update_matrix_free)
{
  if(update_matrix_free)
  {
    // Since matrix_free points to matrix_free_own_storage, we also update the actual/main
    // MatrixFree object called matrix_free.
    matrix_free_own_storage->update_mapping(*get_mapping());
  }

  if(param.turbulence_model_data.is_active)
  {
    // the mesh (and hence the filter width) changes in case of an ALE formulation
    turbulence_model.calculate_filter_width(*get_mapping());
  }

  if(this->param.viscous_problem())
  {
    // update SIPG penalty parameter of viscous operator which depends on the deformation
    // of elements
    viscous_kernel->calculate_penalty_parameter(*matrix_free, get_dof_index_velocity());
  }

  // The inverse mass operator might contain matrix-based components, in which cases it needs to be
  // updated after the grid has been deformed.
  inverse_mass_velocity.update();
  inverse_mass_velocity_scalar.update();

  // note that the update of div-div and continuity penalty terms is done separately
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::set_grid_velocity(VectorType const & u_grid_in)
{
  convective_kernel->set_grid_velocity_ptr(u_grid_in);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::setup_projection_solver()
{
  // setup projection solver

  // divergence penalty only -> local, elementwise problem
  if(param.use_divergence_penalty == true and param.use_continuity_penalty == false)
  {
    // elementwise operator
    elementwise_projection_operator =
      std::make_shared<ElementwiseProjOperator>(*projection_operator);

    // elementwise preconditioner
    if(param.preconditioner_projection == PreconditionerProjection::None)
    {
      typedef Elementwise::PreconditionerIdentity<dealii::VectorizedArray<Number>>
        ElementwiseIdentityPreconditioner;

      elementwise_preconditioner_projection = std::make_shared<ElementwiseIdentityPreconditioner>(
        elementwise_projection_operator->get_problem_size());
    }
    else if(param.preconditioner_projection == PreconditionerProjection::InverseMassMatrix)
    {
      typedef Elementwise::InverseMassPreconditioner<dim, dim, Number>
        ElementwiseInverseMassPreconditioner;

      elementwise_preconditioner_projection =
        std::make_shared<ElementwiseInverseMassPreconditioner>(
          projection_operator->get_matrix_free(),
          projection_operator->get_dof_index(),
          projection_operator->get_quad_index());
    }
    else if(param.preconditioner_projection == PreconditionerProjection::PointJacobi)
    {
      typedef Elementwise::JacobiPreconditioner<dim, dim, Number, ProjOperator>
        ElementwiseJacobiPreconditioner;

      elementwise_preconditioner_projection =
        std::make_shared<ElementwiseJacobiPreconditioner>(projection_operator->get_matrix_free(),
                                                          projection_operator->get_dof_index(),
                                                          projection_operator->get_quad_index(),
                                                          *projection_operator,
                                                          false);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("The specified preconditioner is not implemented."));
    }

    // elementwise solver
    if(param.solver_projection == SolverProjection::CG)
    {
      Elementwise::IterativeSolverData projection_solver_data;
      projection_solver_data.solver_type         = Elementwise::Solver::CG;
      projection_solver_data.solver_data.abs_tol = param.solver_data_projection.abs_tol;
      projection_solver_data.solver_data.rel_tol = param.solver_data_projection.rel_tol;

      typedef Elementwise::
        IterativeSolver<dim, dim, Number, ElementwiseProjOperator, ElementwisePreconditionerBase>
          ElementwiseSolver;

      projection_solver = std::make_shared<ElementwiseSolver>(
        *std::dynamic_pointer_cast<ElementwiseProjOperator>(elementwise_projection_operator),
        *std::dynamic_pointer_cast<ElementwisePreconditionerBase>(
          elementwise_preconditioner_projection),
        projection_solver_data);
    }
    else
    {
      AssertThrow(param.solver_projection == SolverProjection::CG,
                  dealii::ExcMessage("Specified projection solver not implemented."));
    }
  }
  // continuity penalty term with/without divergence penalty term -> globally coupled problem
  else if(param.use_continuity_penalty == true)
  {
    // preconditioner
    if(param.preconditioner_projection == PreconditionerProjection::None)
    {
      // do nothing, preconditioner will not be used
    }
    else if(param.preconditioner_projection == PreconditionerProjection::InverseMassMatrix)
    {
      InverseMassOperatorData<Number> inverse_mass_operator_data;
      inverse_mass_operator_data.dof_index  = get_dof_index_velocity();
      inverse_mass_operator_data.quad_index = get_quad_index_velocity_standard();
      inverse_mass_operator_data.parameters = param.inverse_mass_preconditioner;

      preconditioner_projection =
        std::make_shared<InverseMassPreconditioner<dim, dim, Number>>(*matrix_free,
                                                                      inverse_mass_operator_data);
    }
    else if(param.preconditioner_projection == PreconditionerProjection::PointJacobi)
    {
      // Note that at this point (when initializing the Jacobi preconditioner and calculating the
      // diagonal) the penalty parameter of the projection operator has not been calculated and the
      // time step size has not been set. Hence, 'update_preconditioner = true' should be used for
      // the Jacobi preconditioner in order to use to correct diagonal for preconditioning.
      preconditioner_projection =
        std::make_shared<JacobiPreconditioner<ProjOperator>>(*projection_operator, false);
    }
    else if(param.preconditioner_projection == PreconditionerProjection::BlockJacobi)
    {
      // Note that at this point (when initializing the Jacobi preconditioner)
      // the penalty parameter of the projection operator has not been calculated and the time step
      // size has not been set. Hence, 'update_preconditioner = true' should be used for the Jacobi
      // preconditioner in order to use to correct diagonal blocks for preconditioning.
      preconditioner_projection =
        std::make_shared<BlockJacobiPreconditioner<ProjOperator>>(*projection_operator, false);
    }
    else if(param.preconditioner_projection == PreconditionerProjection::Multigrid)
    {
      typedef MultigridPreconditionerProjection<dim, Number> Multigrid;

      preconditioner_projection = std::make_shared<Multigrid>(this->mpi_comm);

      std::shared_ptr<Multigrid> mg_preconditioner =
        std::dynamic_pointer_cast<Multigrid>(preconditioner_projection);

      typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        pair;

      std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        dirichlet_boundary_conditions = this->projection_operator->get_data().bc->dirichlet_bc;

      // We also need to add DirichletCached boundary conditions. From the
      // perspective of multigrid, there is no difference between standard
      // and cached Dirichlet BCs. Since multigrid does not need information
      // about inhomogeneous boundary data, we simply fill the map with
      // dealii::Functions::ZeroFunction for DirichletCached BCs.
      for(auto iter : this->projection_operator->get_data().bc->dirichlet_cached_bc)
        dirichlet_boundary_conditions.insert(
          pair(iter, new dealii::Functions::ZeroFunction<dim>(dim)));

      typedef std::map<dealii::types::boundary_id, dealii::ComponentMask> Map_DBC_ComponentMask;
      Map_DBC_ComponentMask dirichlet_bc_component_mask;

      auto const & dof_handler = this->get_dof_handler_u();
      mg_preconditioner->initialize(this->param.multigrid_data_projection,
                                    grid,
                                    multigrid_mappings,
                                    dof_handler.get_fe(),
                                    *this->projection_operator,
                                    this->param.ale_formulation,
                                    dirichlet_boundary_conditions,
                                    dirichlet_bc_component_mask);
    }
    else
    {
      AssertThrow(false,
                  dealii::ExcMessage(
                    "Preconditioner specified for projection step is not implemented."));
    }

    // solver
    if(param.solver_projection == SolverProjection::CG)
    {
      // setup solver data
      Krylov::SolverDataCG solver_data;
      solver_data.max_iter             = param.solver_data_projection.max_iter;
      solver_data.solver_tolerance_abs = param.solver_data_projection.abs_tol;
      solver_data.solver_tolerance_rel = param.solver_data_projection.rel_tol;
      // default value of use_preconditioner = false
      if(param.preconditioner_projection != PreconditionerProjection::None)
      {
        solver_data.use_preconditioner = true;
      }

      // setup solver
      projection_solver =
        std::make_shared<Krylov::SolverCG<ProjOperator, PreconditionerBase<Number>, VectorType>>(
          *projection_operator, *preconditioner_projection, solver_data);
    }
    else if(param.solver_projection == SolverProjection::FGMRES)
    {
      // setup solver data
      Krylov::SolverDataFGMRES solver_data;
      solver_data.max_iter             = param.solver_data_projection.max_iter;
      solver_data.solver_tolerance_abs = param.solver_data_projection.abs_tol;
      solver_data.solver_tolerance_rel = param.solver_data_projection.rel_tol;
      solver_data.max_n_tmp_vectors    = param.solver_data_projection.max_krylov_size;

      // default value of use_preconditioner = false
      if(param.preconditioner_projection != PreconditionerProjection::None)
      {
        solver_data.use_preconditioner = true;
      }

      // setup solver
      projection_solver = std::make_shared<
        Krylov::SolverFGMRES<ProjOperator, PreconditionerBase<Number>, VectorType>>(
        *projection_operator, *preconditioner_projection, solver_data);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Specified projection solver not implemented."));
    }
  }
  else
  {
    AssertThrow(
      param.use_divergence_penalty == false and param.use_continuity_penalty == false,
      dealii::ExcMessage(
        "Specified combination of divergence and continuity penalty operators not implemented."));
  }
}

template<int dim, typename Number>
bool
SpatialOperatorBase<dim, Number>::unsteady_problem_has_to_be_solved() const
{
  return (this->param.solver_type == SolverType::Unsteady);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::update_projection_operator(VectorType const & velocity,
                                                             double const time_step_size) const
{
  AssertThrow(projection_operator.get() != 0,
              dealii::ExcMessage("Projection operator is not initialized."));

  // Update projection operator, i.e., the penalty parameters that depend on the velocity field
  // and the time step size
  projection_operator->update(velocity, time_step_size);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::rhs_add_projection_operator(VectorType & dst,
                                                              double const time) const
{
  projection_operator->set_time(time);
  projection_operator->rhs_add(dst);
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::solve_projection(VectorType &       dst,
                                                   VectorType const & src,
                                                   bool const &       update_preconditioner) const
{
  Assert(projection_solver.get() != 0,
         dealii::ExcMessage("Projection solver has not been initialized."));

  projection_solver->update_preconditioner(update_preconditioner);

  unsigned int n_iter = projection_solver->solve(dst, src);

  return n_iter;
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::local_interpolate_stress_bc_boundary_face(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &,
  Range const & face_range) const
{
  unsigned int const dof_index_u = this->get_dof_index_velocity();
  unsigned int const dof_index_p = this->get_dof_index_pressure();
  unsigned int const quad_index  = this->get_quad_index_velocity_nodal_points();

  FaceIntegratorU integrator_u(matrix_free, true, dof_index_u, quad_index);
  FaceIntegratorP integrator_p(matrix_free, true, dof_index_p, quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    dealii::types::boundary_id const boundary_id = matrix_free.get_boundary_id(face);

    BoundaryTypeU const boundary_type =
      this->boundary_descriptor->velocity->get_boundary_type(boundary_id);

    // a Dirichlet boundary for the fluid is a stress boundary for the structure
    if(boundary_type == BoundaryTypeU::DirichletCached)
    {
      integrator_u.reinit(face);
      integrator_u.gather_evaluate(*velocity_ptr, dealii::EvaluationFlags::gradients);

      integrator_p.reinit(face);
      integrator_p.gather_evaluate(*pressure_ptr, dealii::EvaluationFlags::values);

      for(unsigned int q = 0; q < integrator_u.n_q_points; ++q)
      {
        unsigned int const local_face_number = matrix_free.get_face_info(face).interior_face_no;

        unsigned int const index = matrix_free.get_shape_info(dof_index_u, quad_index)
                                     .face_to_cell_index_nodal[local_face_number][q];

        // compute traction acting on structure with normal vector in opposite direction
        // as compared to the fluid domain
        vector normal = integrator_u.normal_vector(q);
        tensor grad_u = integrator_u.get_gradient(q);
        scalar p      = integrator_p.get_value(q);

        scalar viscosity = get_viscosity_boundary_face(face, q);

        // incompressible flow solver is formulated in terms of kinematic viscosity and kinematic
        // pressure
        // -> multiply by density to get true traction in N/m^2.
        vector traction =
          param.density * (viscosity * (grad_u + transpose(grad_u)) * normal - p * normal);

        integrator_u.submit_dof_value(traction, index);
      }

      integrator_u.set_dof_values(dst);
    }
    else
    {
      AssertThrow(boundary_type == BoundaryTypeU::Dirichlet or
                    boundary_type == BoundaryTypeU::Neumann or
                    boundary_type == BoundaryTypeU::Symmetry,
                  dealii::ExcMessage("BoundaryTypeU not implemented."));
    }
  }
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::distribute_constraint_u(VectorType & velocity) const
{
  if(param.spatial_discretization == SpatialDiscretization::HDIV)
  {
    constraint_u.distribute(velocity);
  }
  else if(param.spatial_discretization == SpatialDiscretization::L2)
  {
    // Do nothing
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }
}

template class SpatialOperatorBase<2, float>;
template class SpatialOperatorBase<3, float>;

template class SpatialOperatorBase<2, double>;
template class SpatialOperatorBase<3, double>;

} // namespace IncNS
} // namespace ExaDG
