/*
 * dg_navier_stokes_base.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include "dg_navier_stokes_base.h"

#include "../../poisson/preconditioner/multigrid_preconditioner.h"
#include "../preconditioners/multigrid_preconditioner_projection.h"

namespace IncNS
{
template<int dim, typename Number>
DGNavierStokesBase<dim, Number>::DGNavierStokesBase(
  parallel::TriangulationBase<dim> const & triangulation_in,
  std::shared_ptr<Mesh<dim>> const         mesh_in,
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> const
                                                  periodic_face_pairs_in,
  std::shared_ptr<BoundaryDescriptorU<dim>> const boundary_descriptor_velocity_in,
  std::shared_ptr<BoundaryDescriptorP<dim>> const boundary_descriptor_pressure_in,
  std::shared_ptr<FieldFunctions<dim>> const      field_functions_in,
  InputParameters const &                         parameters_in,
  std::shared_ptr<Postprocessor>                  postprocessor_in,
  MPI_Comm const &                                mpi_comm_in)
  : dealii::Subscriptor(),
    mesh(mesh_in),
    periodic_face_pairs(periodic_face_pairs_in),
    boundary_descriptor_velocity(boundary_descriptor_velocity_in),
    boundary_descriptor_pressure(boundary_descriptor_pressure_in),
    field_functions(field_functions_in),
    param(parameters_in),
    dof_index_first_point(0),
    evaluation_time(0.0),
    fe_u(new FESystem<dim>(FE_DGQ<dim>(parameters_in.degree_u), dim)),
    fe_p(parameters_in.get_degree_p()),
    fe_u_scalar(parameters_in.degree_u),
    dof_handler_u(triangulation_in),
    dof_handler_p(triangulation_in),
    dof_handler_u_scalar(triangulation_in),
    postprocessor(postprocessor_in),
    mpi_comm(mpi_comm_in),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0)
{
  pcout << std::endl << "Construct Navier-Stokes operator ..." << std::endl << std::flush;

  if(param.ale_formulation)
  {
    moving_mesh = std::dynamic_pointer_cast<MovingMesh<dim, Number>>(mesh_in);

    AssertThrow(moving_mesh.get() != 0,
                ExcMessage("moving_mesh is not correctly initialized for ALE formulation"));
  }

  initialize_boundary_descriptor_laplace();

  distribute_dofs();

  constraint_u.close();
  constraint_p.close();
  constraint_u_scalar.close();

  if(param.pure_dirichlet_bc == true &&
     param.adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalSolutionInPoint)
  {
    initialization_pure_dirichlet_bc();
  }

  pcout << std::endl << "... done!" << std::endl << std::flush;
}

template<int dim, typename Number>
DGNavierStokesBase<dim, Number>::~DGNavierStokesBase()
{
  matrix_free->clear();
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::append_data_structures(
  std::shared_ptr<MatrixFreeWrapper<dim, Number>> matrix_free_wrapper)
{
  MappingFlags flags;

  // get current state
  flags.cells          = matrix_free_wrapper->data.mapping_update_flags;
  flags.inner_faces    = matrix_free_wrapper->data.mapping_update_flags_inner_faces;
  flags.boundary_faces = matrix_free_wrapper->data.mapping_update_flags_boundary_faces;

  // append
  flags = flags || MassMatrixKernel<dim, Number>::get_mapping_flags();
  flags = flags || Operators::DivergenceKernel<dim, Number>::get_mapping_flags();
  flags = flags || Operators::GradientKernel<dim, Number>::get_mapping_flags();

  if(param.convective_problem())
    flags = flags || Operators::ConvectiveKernel<dim, Number>::get_mapping_flags();

  if(param.viscous_problem())
    flags = flags || Operators::ViscousKernel<dim, Number>::get_mapping_flags();

  if(param.right_hand_side)
    flags = flags || Operators::RHSKernel<dim, Number>::get_mapping_flags();

  if(param.use_divergence_penalty)
    flags = flags || Operators::DivergencePenaltyKernel<dim, Number>::get_mapping_flags();

  if(param.use_continuity_penalty)
    flags = flags || Operators::ContinuityPenaltyKernel<dim, Number>::get_mapping_flags();

  // write back into additional_data
  matrix_free_wrapper->data.mapping_update_flags                = flags.cells;
  matrix_free_wrapper->data.mapping_update_flags_inner_faces    = flags.inner_faces;
  matrix_free_wrapper->data.mapping_update_flags_boundary_faces = flags.boundary_faces;

  // dof handler
  matrix_free_wrapper->dof_handler_vec.resize(
    static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
      DofHandlerSelector::n_variants));
  matrix_free_wrapper->dof_handler_vec[dof_index_u]        = &dof_handler_u;
  matrix_free_wrapper->dof_handler_vec[dof_index_p]        = &dof_handler_p;
  matrix_free_wrapper->dof_handler_vec[dof_index_u_scalar] = &dof_handler_u_scalar;

  // constraint
  matrix_free_wrapper->constraint_vec.resize(
    static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
      DofHandlerSelector::n_variants));

  matrix_free_wrapper->constraint_vec[dof_index_u]        = &constraint_u;
  matrix_free_wrapper->constraint_vec[dof_index_p]        = &constraint_p;
  matrix_free_wrapper->constraint_vec[dof_index_u_scalar] = &constraint_u_scalar;

  // quadrature
  matrix_free_wrapper->quadrature_vec.resize(
    static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::n_variants));
  // velocity
  matrix_free_wrapper->quadrature_vec[quad_index_u] = QGauss<1>(param.degree_u + 1);
  // pressure
  matrix_free_wrapper->quadrature_vec[quad_index_p] = QGauss<1>(param.get_degree_p() + 1);
  // exact integration of nonlinear convective term
  matrix_free_wrapper->quadrature_vec[quad_index_u_nonlinear] =
    QGauss<1>(param.degree_u + (param.degree_u + 2) / 2);
  // velocity: Gauss-Lobatto quadrature points
  matrix_free_wrapper->quadrature_vec[quad_index_u_gauss_lobatto] =
    QGaussLobatto<1>(param.degree_u + 1);
  // pressure: Gauss-Lobatto quadrature points
  matrix_free_wrapper->quadrature_vec[quad_index_p_gauss_lobatto] =
    QGaussLobatto<1>(param.get_degree_p() + 1);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::setup(
  std::shared_ptr<MatrixFreeWrapper<dim, Number>> matrix_free_wrapper_in)
{
  pcout << std::endl << "Setup Navier-Stokes operator ..." << std::endl << std::flush;

  // MatrixFree
  matrix_free_wrapper = matrix_free_wrapper_in;
  matrix_free         = matrix_free_wrapper->get_matrix_free();

  // initialize data structures depending on MatrixFree
  initialize_operators();

  initialize_calculators_for_derived_quantities();

  initialize_postprocessor();

  // turbulence model depends on MatrixFree and ViscousOperator
  if(param.use_turbulence_model == true)
  {
    initialize_turbulence_model();
  }

  pcout << std::endl << "... done!" << std::endl << std::flush;
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::setup_solvers(double const & scaling_factor_time_derivative_term,
                                               VectorType const & velocity)
{
  // depending on MatrixFree
  initialize_momentum_operator(scaling_factor_time_derivative_term, velocity);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::initialize_boundary_descriptor_laplace()
{
  boundary_descriptor_laplace.reset(new Poisson::BoundaryDescriptor<dim>());

  // Dirichlet BCs for pressure
  boundary_descriptor_laplace->dirichlet_bc = boundary_descriptor_pressure->dirichlet_bc;

  // Neumann BCs for pressure
  // Note: for the dual splitting scheme, neumann_bc contains functions corresponding
  //       to dudt term required in pressure Neumann boundary condition.
  // Here: set this functions explicitly to ZeroFunction when filling the boundary
  //       descriptor for the Laplace operator because these inhomogeneous
  //       boundary conditions have to be implemented separately
  //       and can not be applied by the Laplace operator.
  for(typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::const_iterator it =
        boundary_descriptor_pressure->neumann_bc.begin();
      it != boundary_descriptor_pressure->neumann_bc.end();
      ++it)
  {
    std::shared_ptr<Function<dim>> zero_function;
    zero_function.reset(new Functions::ZeroFunction<dim>(1));
    boundary_descriptor_laplace->neumann_bc.insert(
      std::pair<types::boundary_id, std::shared_ptr<Function<dim>>>(it->first, zero_function));
  }
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::distribute_dofs()
{
  // enumerate degrees of freedom
  dof_handler_u.distribute_dofs(*fe_u);
  dof_handler_u.distribute_mg_dofs();
  dof_handler_p.distribute_dofs(fe_p);
  dof_handler_p.distribute_mg_dofs();
  dof_handler_u_scalar.distribute_dofs(fe_u_scalar);
  dof_handler_u_scalar.distribute_mg_dofs(); // probably, we don't need this

  unsigned int const ndofs_per_cell_velocity = Utilities::pow(param.degree_u + 1, dim) * dim;
  unsigned int const ndofs_per_cell_pressure = Utilities::pow(param.get_degree_p() + 1, dim);

  pcout << std::endl
        << "Discontinuous Galerkin finite element discretization:" << std::endl
        << std::endl
        << std::flush;

  pcout << "Velocity:" << std::endl;
  print_parameter(pcout, "degree of 1D polynomials", param.degree_u);
  print_parameter(pcout, "number of dofs per cell", ndofs_per_cell_velocity);
  print_parameter(pcout, "number of dofs (total)", dof_handler_u.n_dofs());

  pcout << "Pressure:" << std::endl;
  print_parameter(pcout, "degree of 1D polynomials", param.get_degree_p());
  print_parameter(pcout, "number of dofs per cell", ndofs_per_cell_pressure);
  print_parameter(pcout, "number of dofs (total)", dof_handler_p.n_dofs());

  pcout << "Velocity and pressure:" << std::endl;
  print_parameter(pcout,
                  "number of dofs per cell",
                  ndofs_per_cell_velocity + ndofs_per_cell_pressure);
  print_parameter(pcout, "number of dofs (total)", dof_handler_u.n_dofs() + dof_handler_p.n_dofs());

  pcout << std::flush;
}

template<int dim, typename Number>
types::global_dof_index
DGNavierStokesBase<dim, Number>::get_number_of_dofs() const
{
  return dof_handler_u.n_dofs() + dof_handler_p.n_dofs();
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::initialize_operators()
{
  // operator kernels
  Operators::ConvectiveKernelData convective_kernel_data;
  convective_kernel_data.formulation       = param.formulation_convective_term;
  convective_kernel_data.upwind_factor     = param.upwind_factor;
  convective_kernel_data.use_outflow_bc    = param.use_outflow_bc_convective_term;
  convective_kernel_data.type_dirichlet_bc = param.type_dirichlet_bc_convective;
  convective_kernel_data.ale               = param.ale_formulation;
  convective_kernel.reset(new Operators::ConvectiveKernel<dim, Number>());
  convective_kernel->reinit(*matrix_free,
                            convective_kernel_data,
                            dof_index_u,
                            get_quad_index_velocity_linearized(),
                            false /* is_mg */);

  Operators::ViscousKernelData viscous_kernel_data;
  viscous_kernel_data.IP_factor                    = param.IP_factor_viscous;
  viscous_kernel_data.viscosity                    = param.viscosity;
  viscous_kernel_data.formulation_viscous_term     = param.formulation_viscous_term;
  viscous_kernel_data.penalty_term_div_formulation = param.penalty_term_div_formulation;
  viscous_kernel_data.IP_formulation               = param.IP_formulation_viscous;
  viscous_kernel_data.viscosity_is_variable        = param.use_turbulence_model;
  viscous_kernel_data.variable_normal_vector       = param.neumann_with_variable_normal_vector;
  viscous_kernel.reset(new Operators::ViscousKernel<dim, Number>());
  viscous_kernel->reinit(*matrix_free, viscous_kernel_data, dof_index_u);

  AffineConstraints<double> constraint_dummy;
  constraint_dummy.close();

  // mass matrix operator
  MassMatrixOperatorData mass_matrix_operator_data;
  mass_matrix_operator_data.dof_index  = dof_index_u;
  mass_matrix_operator_data.quad_index = quad_index_u;
  mass_matrix_operator.reinit(*matrix_free, constraint_dummy, mass_matrix_operator_data);

  // inverse mass matrix operator
  inverse_mass_velocity.initialize(*matrix_free, param.degree_u, dof_index_u, quad_index_u);

  // inverse mass matrix operator velocity scalar
  inverse_mass_velocity_scalar.initialize(*matrix_free,
                                          param.degree_u,
                                          dof_index_u_scalar,
                                          quad_index_u);

  // body force operator
  RHSOperatorData<dim> rhs_data;
  rhs_data.dof_index                                 = dof_index_u;
  rhs_data.dof_index_scalar                          = dof_index_u_scalar;
  rhs_data.quad_index                                = quad_index_u;
  rhs_data.kernel_data.f                             = field_functions->right_hand_side;
  rhs_data.kernel_data.boussinesq_term               = param.boussinesq_term;
  rhs_data.kernel_data.thermal_expansion_coefficient = param.thermal_expansion_coefficient;
  rhs_data.kernel_data.reference_temperature         = param.reference_temperature;
  rhs_data.kernel_data.gravitational_force           = field_functions->gravitational_force;

  rhs_operator.reinit(*matrix_free, rhs_data);

  // gradient operator
  GradientOperatorData<dim> gradient_operator_data;
  gradient_operator_data.dof_index_velocity   = dof_index_u;
  gradient_operator_data.dof_index_pressure   = dof_index_p;
  gradient_operator_data.quad_index           = quad_index_u;
  gradient_operator_data.integration_by_parts = param.gradp_integrated_by_parts;
  gradient_operator_data.formulation          = param.gradp_formulation;
  gradient_operator_data.use_boundary_data    = param.gradp_use_boundary_data;
  gradient_operator_data.bc                   = boundary_descriptor_pressure;
  gradient_operator.reinit(*matrix_free, gradient_operator_data);

  // divergence operator
  DivergenceOperatorData<dim> divergence_operator_data;
  divergence_operator_data.dof_index_velocity   = dof_index_u;
  divergence_operator_data.dof_index_pressure   = dof_index_p;
  divergence_operator_data.quad_index           = quad_index_u;
  divergence_operator_data.integration_by_parts = param.divu_integrated_by_parts;
  divergence_operator_data.formulation          = param.divu_formulation;
  divergence_operator_data.use_boundary_data    = param.divu_use_boundary_data;
  divergence_operator_data.bc                   = boundary_descriptor_velocity;
  divergence_operator.reinit(*matrix_free, divergence_operator_data);

  // convective operator
  ConvectiveOperatorData<dim> convective_operator_data;
  convective_operator_data.kernel_data          = convective_kernel_data;
  convective_operator_data.dof_index            = dof_index_u;
  convective_operator_data.quad_index           = this->get_quad_index_velocity_linearized();
  convective_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
  convective_operator_data.quad_index_nonlinear = quad_index_u_nonlinear;
  convective_operator_data.bc                   = boundary_descriptor_velocity;
  convective_operator.reinit(*matrix_free,
                             constraint_dummy,
                             convective_operator_data,
                             convective_kernel);

  // viscous operator
  ViscousOperatorData<dim> viscous_operator_data;
  viscous_operator_data.kernel_data          = viscous_kernel_data;
  viscous_operator_data.bc                   = boundary_descriptor_velocity;
  viscous_operator_data.dof_index            = dof_index_u;
  viscous_operator_data.quad_index           = quad_index_u;
  viscous_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
  viscous_operator.reinit(*matrix_free, constraint_dummy, viscous_operator_data, viscous_kernel);

  if(param.use_divergence_penalty)
  {
    // Kernel
    Operators::DivergencePenaltyKernelData div_penalty_data;
    div_penalty_data.type_penalty_parameter = param.type_penalty_parameter;
    div_penalty_data.viscosity              = param.viscosity;
    div_penalty_data.degree                 = param.degree_u;
    div_penalty_data.penalty_factor         = param.divergence_penalty_factor;

    div_penalty_kernel.reset(new Operators::DivergencePenaltyKernel<dim, Number>());
    div_penalty_kernel->reinit(*matrix_free,
                               get_dof_index_velocity(),
                               get_quad_index_velocity_linear(),
                               div_penalty_data);

    // Operator
    DivergencePenaltyData operator_data;
    operator_data.dof_index  = get_dof_index_velocity();
    operator_data.quad_index = get_quad_index_velocity_linear();

    div_penalty_operator.reinit(*matrix_free, operator_data, div_penalty_kernel);
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

    conti_penalty_kernel.reset(new Operators::ContinuityPenaltyKernel<dim, Number>());
    conti_penalty_kernel->reinit(*matrix_free,
                                 get_dof_index_velocity(),
                                 get_quad_index_velocity_linear(),
                                 kernel_data);

    // Operator
    ContinuityPenaltyData<dim> operator_data;
    operator_data.dof_index         = get_dof_index_velocity();
    operator_data.quad_index        = get_quad_index_velocity_linear();
    operator_data.use_boundary_data = param.continuity_penalty_use_boundary_data;
    operator_data.bc                = this->boundary_descriptor_velocity;

    conti_penalty_operator.reinit(*matrix_free, operator_data, conti_penalty_kernel);
  }

  if(param.use_divergence_penalty || param.use_continuity_penalty)
  {
    if(param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme ||
       param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection ||
       (param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution &&
        param.apply_penalty_terms_in_postprocessing_step == true))
    {
      // setup projection operator
      ProjectionOperatorData<dim> data;
      data.use_divergence_penalty = param.use_divergence_penalty;
      data.use_continuity_penalty = param.use_continuity_penalty;
      data.use_boundary_data      = param.continuity_penalty_use_boundary_data;
      data.bc                     = this->boundary_descriptor_velocity;
      data.dof_index              = get_dof_index_velocity();
      data.quad_index             = get_quad_index_velocity_linear();
      data.use_cell_based_loops   = param.use_cell_based_face_loops;
      data.implement_block_diagonal_preconditioner_matrix_free =
        param.implement_block_diagonal_preconditioner_matrix_free;
      data.solver_block_diagonal         = Elementwise::Solver::CG;
      data.preconditioner_block_diagonal = param.preconditioner_block_diagonal_projection;
      data.solver_data_block_diagonal    = param.solver_data_block_diagonal_projection;

      projection_operator.reset(new PROJ_OPERATOR());

      projection_operator->reinit(
        *matrix_free, constraint_dummy, data, div_penalty_kernel, conti_penalty_kernel);
    }
  }
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::initialize_momentum_operator(
  double const &     scaling_factor_time_derivative_term,
  VectorType const & velocity)
{
  // Momentum operator
  MomentumOperatorData<dim> data;

  data.unsteady_problem           = unsteady_problem_has_to_be_solved();
  data.scaling_factor_mass_matrix = scaling_factor_time_derivative_term;

  if(param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
    data.convective_problem = false;
  else
    data.convective_problem = param.nonlinear_problem_has_to_be_solved();

  data.viscous_problem = param.viscous_problem();

  data.formulation_convective_term = param.formulation_convective_term;

  data.bc = boundary_descriptor_velocity;

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

  AffineConstraints<double> constraint_dummy;
  constraint_dummy.close();

  momentum_operator.reinit(*matrix_free, constraint_dummy, data, viscous_kernel, convective_kernel);

  if(data.convective_problem)
    set_velocity_ptr(velocity);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::initialize_turbulence_model()
{
  // initialize turbulence model
  TurbulenceModelData model_data;
  model_data.turbulence_model    = param.turbulence_model;
  model_data.constant            = param.turbulence_model_constant;
  model_data.kinematic_viscosity = param.viscosity;
  model_data.dof_index           = dof_index_u;
  model_data.quad_index          = quad_index_u;
  model_data.degree              = param.degree_u;
  turbulence_model.initialize(*matrix_free, get_mapping(), viscous_kernel, model_data);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::initialize_calculators_for_derived_quantities()
{
  vorticity_calculator.initialize(*matrix_free, dof_index_u, quad_index_u);
  divergence_calculator.initialize(*matrix_free, dof_index_u, dof_index_u_scalar, quad_index_u);
  velocity_magnitude_calculator.initialize(*matrix_free,
                                           dof_index_u,
                                           dof_index_u_scalar,
                                           quad_index_u);
  q_criterion_calculator.initialize(*matrix_free, dof_index_u, dof_index_u_scalar, quad_index_u);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::initialization_pure_dirichlet_bc()
{
  dof_index_first_point = 0;
  for(unsigned int d = 0; d < dim; ++d)
    first_point[d] = 0.0;

  if(Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    typename DoFHandler<dim>::active_cell_iterator first_cell;
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_p.begin_active(),
                                                   endc = dof_handler_p.end();

    bool processor_has_active_cells = false;
    for(; cell != endc; ++cell)
    {
      if(cell->is_locally_owned())
      {
        first_cell = cell;

        processor_has_active_cells = true;
        break;
      }
    }

    AssertThrow(processor_has_active_cells == true,
                ExcMessage("No active cells on Processor with ID=0"));

    FEValues<dim> fe_values(dof_handler_p.get_fe(),
                            Quadrature<dim>(dof_handler_p.get_fe().get_unit_support_points()),
                            update_quadrature_points);

    fe_values.reinit(first_cell);

    first_point = fe_values.quadrature_point(0);
    std::vector<types::global_dof_index> dof_indices(dof_handler_p.get_fe().dofs_per_cell);
    first_cell->get_dof_indices(dof_indices);
    dof_index_first_point = dof_indices[0];
  }
  dof_index_first_point = Utilities::MPI::sum(dof_index_first_point, mpi_comm);
  for(unsigned int d = 0; d < dim; ++d)
  {
    first_point[d] = Utilities::MPI::sum(first_point[d], mpi_comm);
  }
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::initialize_postprocessor()
{
  postprocessor->setup(*this);
}

template<int dim, typename Number>
MatrixFree<dim, Number> const &
DGNavierStokesBase<dim, Number>::get_matrix_free() const
{
  return *matrix_free;
}

template<int dim, typename Number>
unsigned int
DGNavierStokesBase<dim, Number>::get_dof_index_velocity() const
{
  return dof_index_u;
}

template<int dim, typename Number>
unsigned int
DGNavierStokesBase<dim, Number>::get_dof_index_velocity_scalar() const
{
  return dof_index_u_scalar;
}

template<int dim, typename Number>
unsigned int
DGNavierStokesBase<dim, Number>::get_quad_index_velocity_linear() const
{
  return quad_index_u;
}

template<int dim, typename Number>
unsigned int
DGNavierStokesBase<dim, Number>::get_quad_index_velocity_nonlinear() const
{
  return quad_index_u_nonlinear;
}

template<int dim, typename Number>
unsigned int
DGNavierStokesBase<dim, Number>::get_quad_index_velocity_gauss_lobatto() const
{
  return quad_index_u_gauss_lobatto;
}

template<int dim, typename Number>
unsigned int
DGNavierStokesBase<dim, Number>::get_quad_index_pressure_gauss_lobatto() const
{
  return quad_index_p_gauss_lobatto;
}

template<int dim, typename Number>
unsigned int
DGNavierStokesBase<dim, Number>::get_quad_index_velocity_linearized() const
{
  if(param.quad_rule_linearization == QuadratureRuleLinearization::Standard)
  {
    return quad_index_u;
  }
  else if(param.quad_rule_linearization == QuadratureRuleLinearization::Overintegration32k)
  {
    if(param.nonlinear_problem_has_to_be_solved())
      return quad_index_u_nonlinear;
    else
      return quad_index_u;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented"));
    return quad_index_u_nonlinear;
  }
}

template<int dim, typename Number>
unsigned int
DGNavierStokesBase<dim, Number>::get_dof_index_pressure() const
{
  return dof_index_p;
}

template<int dim, typename Number>
unsigned int
DGNavierStokesBase<dim, Number>::get_quad_index_pressure() const
{
  return quad_index_p;
}

template<int dim, typename Number>
Mapping<dim> const &
DGNavierStokesBase<dim, Number>::get_mapping() const
{
  return mesh->get_mapping();
}

template<int dim, typename Number>
FESystem<dim> const &
DGNavierStokesBase<dim, Number>::get_fe_u() const
{
  return *fe_u;
}

template<int dim, typename Number>
FE_DGQ<dim> const &
DGNavierStokesBase<dim, Number>::get_fe_p() const
{
  return fe_p;
}

template<int dim, typename Number>
DoFHandler<dim> const &
DGNavierStokesBase<dim, Number>::get_dof_handler_u() const
{
  return dof_handler_u;
}

template<int dim, typename Number>
DoFHandler<dim> const &
DGNavierStokesBase<dim, Number>::get_dof_handler_u_scalar() const
{
  return dof_handler_u_scalar;
}

template<int dim, typename Number>
DoFHandler<dim> const &
DGNavierStokesBase<dim, Number>::get_dof_handler_p() const
{
  return dof_handler_p;
}

template<int dim, typename Number>
AffineConstraints<double> const &
DGNavierStokesBase<dim, Number>::get_constraint_p() const
{
  return constraint_p;
}

template<int dim, typename Number>
double
DGNavierStokesBase<dim, Number>::get_viscosity() const
{
  return param.viscosity;
}

template<int dim, typename Number>
VectorizedArray<Number>
DGNavierStokesBase<dim, Number>::get_viscosity_boundary_face(unsigned int const face,
                                                             unsigned int const q) const
{
  VectorizedArray<Number> viscosity = make_vectorized_array<Number>(get_viscosity());

  bool const viscosity_is_variable = param.use_turbulence_model;
  if(viscosity_is_variable)
    viscous_kernel->get_coefficient_face(face, q);

  return viscosity;
}

// Polynomial degree required for CFL condition, e.g., CFL_k = CFL / k^{exp}.
template<int dim, typename Number>
unsigned int
DGNavierStokesBase<dim, Number>::get_polynomial_degree() const
{
  return param.degree_u;
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::set_velocity_ptr(VectorType const & velocity) const
{
  convective_kernel->set_velocity_ptr(velocity);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::initialize_vector_velocity(VectorType & src) const
{
  matrix_free->initialize_dof_vector(src, dof_index_u);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::initialize_vector_velocity_scalar(VectorType & src) const
{
  matrix_free->initialize_dof_vector(src, dof_index_u_scalar);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::initialize_vector_pressure(VectorType & src) const
{
  matrix_free->initialize_dof_vector(src, dof_index_p);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::prescribe_initial_conditions(VectorType & velocity,
                                                              VectorType & pressure,
                                                              double const time) const
{
  // make sure that the mesh fits to the time at which we want to evaluate the solution
  if(param.ale_formulation)
    moving_mesh->move_mesh_analytical(time);

  field_functions->initial_solution_velocity->set_time(time);
  field_functions->initial_solution_pressure->set_time(time);

  // This is necessary if Number == float
  typedef LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

  VectorTypeDouble velocity_double;
  VectorTypeDouble pressure_double;
  velocity_double = velocity;
  pressure_double = pressure;

  VectorTools::interpolate(get_mapping(),
                           dof_handler_u,
                           *(field_functions->initial_solution_velocity),
                           velocity_double);

  VectorTools::interpolate(get_mapping(),
                           dof_handler_p,
                           *(field_functions->initial_solution_pressure),
                           pressure_double);

  velocity = velocity_double;
  pressure = pressure_double;
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::move_mesh_and_interpolate_velocity_dirichlet_bc(
  VectorType &   dst,
  double const & time)
{
  // make sure that the mesh fits to the time at which we want to evaluate the solution
  if(param.ale_formulation)
  {
    moving_mesh->move_mesh_analytical(time);

    // we also need to update matrix_free
    update_after_mesh_movement();
  }

  interpolate_velocity_dirichlet_bc(dst, time);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::move_mesh_and_interpolate_pressure_dirichlet_bc(
  VectorType &   dst,
  double const & time)
{
  // make sure that the mesh fits to the time at which we want to evaluate the solution
  if(param.ale_formulation)
  {
    moving_mesh->move_mesh_analytical(time);

    // we also need to update matrix_free
    update_after_mesh_movement();
  }

  interpolate_pressure_dirichlet_bc(dst, time);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::interpolate_velocity_dirichlet_bc(VectorType &   dst,
                                                                   double const & time)
{
  this->evaluation_time = time;

  dst = 0.0;

  VectorType src_dummy;
  matrix_free->loop(&This::cell_loop_empty,
                    &This::face_loop_empty,
                    &This::local_interpolate_velocity_dirichlet_bc_boundary_face,
                    this,
                    dst,
                    src_dummy);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::interpolate_pressure_dirichlet_bc(VectorType &   dst,
                                                                   double const & time)
{
  this->evaluation_time = time;

  dst = 0.0;

  VectorType src_dummy;
  matrix_free->loop(&This::cell_loop_empty,
                    &This::face_loop_empty,
                    &This::local_interpolate_pressure_dirichlet_bc_boundary_face,
                    this,
                    dst,
                    src_dummy);
}

template<int dim, typename Number>
double
DGNavierStokesBase<dim, Number>::calculate_minimum_element_length() const
{
  return calculate_minimum_vertex_distance(dof_handler_u.get_triangulation(), mpi_comm);
}

template<int dim, typename Number>
double
DGNavierStokesBase<dim, Number>::calculate_time_step_cfl(VectorType const & velocity,
                                                         double const       cfl,
                                                         double const       exponent_degree) const
{
  return calculate_time_step_cfl_local<dim, Number>(*matrix_free,
                                                    dof_index_u,
                                                    quad_index_u,
                                                    velocity,
                                                    cfl,
                                                    param.degree_u,
                                                    exponent_degree,
                                                    param.adaptive_time_stepping_cfl_type,
                                                    mpi_comm);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::apply_mass_matrix(VectorType & dst, VectorType const & src) const
{
  mass_matrix_operator.apply(dst, src);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::apply_mass_matrix_add(VectorType &       dst,
                                                       VectorType const & src) const
{
  mass_matrix_operator.apply_add(dst, src);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::shift_pressure(VectorType & pressure, double const & time) const
{
  VectorType vec1(pressure);
  for(unsigned int i = 0; i < vec1.local_size(); ++i)
    vec1.local_element(i) = 1.;
  field_functions->analytical_solution_pressure->set_time(time);
  double const exact   = field_functions->analytical_solution_pressure->value(first_point);
  double       current = 0.;
  if(pressure.locally_owned_elements().is_element(dof_index_first_point))
    current = pressure(dof_index_first_point);
  current = Utilities::MPI::sum(current, mpi_comm);
  pressure.add(exact - current, vec1);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::shift_pressure_mean_value(VectorType &   pressure,
                                                           double const & time) const
{
  // one cannot use Number as template here since Number might be float
  // while analytical_solution_pressure is of type Function<dim,double>
  typedef LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

  VectorTypeDouble vec_double;
  vec_double = pressure; // initialize

  field_functions->analytical_solution_pressure->set_time(time);
  VectorTools::interpolate(get_mapping(),
                           dof_handler_p,
                           *(field_functions->analytical_solution_pressure),
                           vec_double);

  double const exact   = vec_double.mean_value();
  double const current = pressure.mean_value();

  VectorType vec_temp2(pressure);
  for(unsigned int i = 0; i < vec_temp2.local_size(); ++i)
    vec_temp2.local_element(i) = 1.;

  pressure.add(exact - current, vec_temp2);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::set_temperature(VectorType const & temperature)
{
  AssertThrow(param.boussinesq_term, ExcMessage("Invalid parameters detected."));

  rhs_operator.set_temperature(temperature);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::compute_vorticity(VectorType & dst, VectorType const & src) const
{
  vorticity_calculator.compute_vorticity(dst, src);

  inverse_mass_velocity.apply(dst, dst);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::compute_divergence(VectorType & dst, VectorType const & src) const
{
  divergence_calculator.compute_divergence(dst, src);

  inverse_mass_velocity_scalar.apply(dst, dst);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::compute_velocity_magnitude(VectorType &       dst,
                                                            VectorType const & src) const
{
  velocity_magnitude_calculator.compute(dst, src);

  inverse_mass_velocity_scalar.apply(dst, dst);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::compute_vorticity_magnitude(VectorType &       dst,
                                                             VectorType const & src) const
{
  velocity_magnitude_calculator.compute(dst, src);

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
DGNavierStokesBase<dim, Number>::compute_streamfunction(VectorType &       dst,
                                                        VectorType const & src) const
{
  AssertThrow(dim == 2, ExcMessage("Calculation of streamfunction can only be used for dim==2."));

  // compute rhs vector
  StreamfunctionCalculatorRHSOperator<dim, Number> rhs_operator;
  rhs_operator.initialize(*matrix_free, dof_index_u, dof_index_u_scalar, quad_index_u);
  VectorType rhs;
  initialize_vector_velocity_scalar(rhs);
  rhs_operator.apply(rhs, src);

  // setup Laplace operator for scalar velocity vector
  Poisson::LaplaceOperatorData<dim> laplace_operator_data;
  laplace_operator_data.dof_index  = get_dof_index_velocity_scalar();
  laplace_operator_data.quad_index = get_quad_index_velocity_linear();

  std::shared_ptr<Poisson::BoundaryDescriptor<dim>> boundary_descriptor_streamfunction;
  boundary_descriptor_streamfunction.reset(new Poisson::BoundaryDescriptor<dim>());

  // fill boundary descriptor: Assumption: only Dirichlet BC's
  boundary_descriptor_streamfunction->dirichlet_bc = boundary_descriptor_velocity->dirichlet_bc;

  AssertThrow(boundary_descriptor_velocity->neumann_bc.empty() == true,
              ExcMessage("Assumption is not fulfilled. Streamfunction calculator is "
                         "not implemented for this type of boundary conditions."));
  AssertThrow(boundary_descriptor_velocity->symmetry_bc.empty() == true,
              ExcMessage("Assumption is not fulfilled. Streamfunction calculator is "
                         "not implemented for this type of boundary conditions."));

  laplace_operator_data.bc = boundary_descriptor_streamfunction;

  laplace_operator_data.kernel_data.IP_factor = 1.0;

  typedef Poisson::LaplaceOperator<dim, Number> Laplace;
  Laplace                                       laplace_operator;
  AffineConstraints<double>                     constraint_dummy;
  laplace_operator.reinit(*matrix_free, constraint_dummy, laplace_operator_data);

  // setup preconditioner
  std::shared_ptr<PreconditionerBase<Number>> preconditioner;

  // use multigrid preconditioner with Chebyshev smoother
  MultigridData mg_data;

  typedef Poisson::MultigridPreconditioner<dim, Number, MultigridNumber> MULTIGRID;

  preconditioner.reset(new MULTIGRID(this->mpi_comm));

  std::shared_ptr<MULTIGRID> mg_preconditioner =
    std::dynamic_pointer_cast<MULTIGRID>(preconditioner);

  // explicit copy needed since function is called on const
  auto periodic_face_pairs = this->periodic_face_pairs;

  parallel::TriangulationBase<dim> const * tria =
    dynamic_cast<const parallel::TriangulationBase<dim> *>(
      &dof_handler_u_scalar.get_triangulation());
  const FiniteElement<dim> & fe = dof_handler_u_scalar.get_fe();

  mg_preconditioner->initialize(mg_data,
                                tria,
                                fe,
                                get_mapping(),
                                laplace_operator.get_data(),
                                this->param.ale_formulation,
                                &laplace_operator.get_data().bc->dirichlet_bc,
                                &periodic_face_pairs);

  // setup solver
  CGSolverData solver_data;
  solver_data.solver_tolerance_rel = 1.e-10;
  solver_data.use_preconditioner   = true;

  CGSolver<Laplace, PreconditionerBase<Number>, VectorType> poisson_solver(laplace_operator,
                                                                           *preconditioner,
                                                                           solver_data);

  // solve Poisson problem
  poisson_solver.solve(dst, rhs, /* update preconditioner = */ false);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::compute_q_criterion(VectorType & dst, VectorType const & src) const
{
  q_criterion_calculator.compute(dst, src);

  inverse_mass_velocity_scalar.apply(dst, dst);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::apply_inverse_mass_matrix(VectorType &       dst,
                                                           VectorType const & src) const
{
  inverse_mass_velocity.apply(dst, src);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::evaluate_add_body_force_term(VectorType & dst,
                                                              double const time) const
{
  this->rhs_operator.evaluate_add(dst, time);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::evaluate_convective_term(VectorType &       dst,
                                                          VectorType const & src,
                                                          Number const       time) const
{
  convective_operator.evaluate_nonlinear_operator(dst, src, time);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::evaluate_pressure_gradient_term(VectorType &       dst,
                                                                 VectorType const & src,
                                                                 double const       time) const
{
  gradient_operator.evaluate(dst, src, time);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::evaluate_velocity_divergence_term(VectorType &       dst,
                                                                   VectorType const & src,
                                                                   double const       time) const
{
  divergence_operator.evaluate(dst, src, time);
}

// OIF splitting
template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::evaluate_negative_convective_term_and_apply_inverse_mass_matrix(
  VectorType &       dst,
  VectorType const & src,
  Number const       time) const
{
  convective_operator.evaluate_nonlinear_operator(dst, src, time);

  // shift convective term to the rhs of the equation
  dst *= -1.0;

  inverse_mass_velocity.apply(dst, dst);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::evaluate_negative_convective_term_and_apply_inverse_mass_matrix(
  VectorType &       dst,
  VectorType const & src,
  Number const       time,
  VectorType const & velocity_transport) const
{
  convective_operator.evaluate_linear_transport(dst, src, time, velocity_transport);

  // shift convective term to the rhs of the equation
  dst *= -1.0;

  inverse_mass_velocity.apply(dst, dst);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::update_turbulence_model(VectorType const & velocity)
{
  // calculate turbulent viscosity locally in each cell and face quadrature point
  turbulence_model.calculate_turbulent_viscosity(velocity);
}

template<int dim, typename Number>
double
DGNavierStokesBase<dim, Number>::calculate_dissipation_convective_term(VectorType const & velocity,
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
DGNavierStokesBase<dim, Number>::calculate_dissipation_viscous_term(
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
DGNavierStokesBase<dim, Number>::calculate_dissipation_divergence_term(
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
DGNavierStokesBase<dim, Number>::calculate_dissipation_continuity_term(
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
DGNavierStokesBase<dim, Number>::update_after_mesh_movement()
{
  // TODO: problems occur if the mesh is not deformed (displacement = 0)
  matrix_free_wrapper->update_geometry();

  // TODO: this should not be necessary for a good design. MatrixFree has to care about that.
  inverse_mass_velocity.reinit();
  inverse_mass_velocity_scalar.reinit();

  if(this->param.use_turbulence_model)
  {
    // the mesh (and hence the filter width) changes in case of ALE formulation
    turbulence_model.calculate_filter_width(get_mapping());
  }

  if(this->param.viscous_problem())
  {
    // update SIPG penalty parameter of viscous operator which depends on the deformation
    // of elements
    viscous_operator.update();
  }

  // note that the update of div-div and continuity penalty terms is done separately
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::set_grid_velocity(VectorType u_grid_in)
{
  convective_kernel->set_grid_velocity_ptr(u_grid_in);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::move_mesh(double const time)
{
  moving_mesh->move_mesh_analytical(time);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::move_mesh_and_fill_grid_coordinates_vector(VectorType & vector,
                                                                            double const time)
{
  moving_mesh->move_mesh_analytical(time);

  moving_mesh->fill_grid_coordinates_vector(vector, get_dof_handler_u());
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::setup_projection_solver()
{
  // setup projection solver

  // divergence penalty only -> local, elementwise problem
  if(param.use_divergence_penalty == true && param.use_continuity_penalty == false)
  {
    if(param.solver_projection == SolverProjection::CG)
    {
      // projection operator
      elementwise_projection_operator.reset(new ELEMENTWISE_PROJ_OPERATOR(*projection_operator));

      // preconditioner
      typedef Elementwise::PreconditionerBase<VectorizedArray<Number>> PROJ_PRECONDITIONER;

      if(param.preconditioner_projection == PreconditionerProjection::None)
      {
        typedef Elementwise::PreconditionerIdentity<VectorizedArray<Number>> IDENTITY;

        elementwise_preconditioner_projection.reset(
          new IDENTITY(elementwise_projection_operator->get_problem_size()));
      }
      else if(param.preconditioner_projection == PreconditionerProjection::InverseMassMatrix)
      {
        typedef Elementwise::InverseMassMatrixPreconditioner<dim, dim, Number> INVERSE_MASS;

        elementwise_preconditioner_projection.reset(
          new INVERSE_MASS(projection_operator->get_matrix_free(),
                           projection_operator->get_dof_index(),
                           projection_operator->get_quad_index()));
      }
      else
      {
        AssertThrow(false, ExcMessage("The specified preconditioner is not implemented."));
      }

      // solver
      Elementwise::IterativeSolverData projection_solver_data;
      projection_solver_data.solver_type         = Elementwise::Solver::CG;
      projection_solver_data.solver_data.abs_tol = param.solver_data_projection.abs_tol;
      projection_solver_data.solver_data.rel_tol = param.solver_data_projection.rel_tol;

      typedef Elementwise::
        IterativeSolver<dim, dim, Number, ELEMENTWISE_PROJ_OPERATOR, PROJ_PRECONDITIONER>
          PROJ_SOLVER;

      projection_solver.reset(new PROJ_SOLVER(
        *std::dynamic_pointer_cast<ELEMENTWISE_PROJ_OPERATOR>(elementwise_projection_operator),
        *std::dynamic_pointer_cast<PROJ_PRECONDITIONER>(elementwise_preconditioner_projection),
        projection_solver_data));
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified projection solver not implemented."));
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
      preconditioner_projection.reset(new InverseMassMatrixPreconditioner<dim, dim, Number>(
        *matrix_free, param.degree_u, get_dof_index_velocity(), get_quad_index_velocity_linear()));
    }
    else if(param.preconditioner_projection == PreconditionerProjection::PointJacobi)
    {
      // Note that at this point (when initializing the Jacobi preconditioner and calculating the
      // diagonal) the penalty parameter of the projection operator has not been calculated and the
      // time step size has not been set. Hence, 'update_preconditioner = true' should be used for
      // the Jacobi preconditioner in order to use to correct diagonal for preconditioning.
      preconditioner_projection.reset(new JacobiPreconditioner<PROJ_OPERATOR>(
        *std::dynamic_pointer_cast<PROJ_OPERATOR>(projection_operator)));
    }
    else if(param.preconditioner_projection == PreconditionerProjection::BlockJacobi)
    {
      // Note that at this point (when initializing the Jacobi preconditioner)
      // the penalty parameter of the projection operator has not been calculated and the time step
      // size has not been set. Hence, 'update_preconditioner = true' should be used for the Jacobi
      // preconditioner in order to use to correct diagonal blocks for preconditioning.
      preconditioner_projection.reset(new BlockJacobiPreconditioner<PROJ_OPERATOR>(
        *std::dynamic_pointer_cast<PROJ_OPERATOR>(projection_operator)));
    }
    else if(param.preconditioner_projection == PreconditionerProjection::Multigrid)
    {
      typedef MultigridPreconditionerProjection<dim, Number, MultigridNumber> MULTIGRID;

      preconditioner_projection.reset(new MULTIGRID(this->mpi_comm));

      std::shared_ptr<MULTIGRID> mg_preconditioner =
        std::dynamic_pointer_cast<MULTIGRID>(preconditioner_projection);

      auto const & dof_handler = this->get_dof_handler_u();

      parallel::TriangulationBase<dim> const * tria =
        dynamic_cast<parallel::TriangulationBase<dim> const *>(&dof_handler.get_triangulation());

      FiniteElement<dim> const & fe = dof_handler.get_fe();

      mg_preconditioner->initialize(this->param.multigrid_data_projection,
                                    tria,
                                    fe,
                                    this->get_mapping(),
                                    *this->projection_operator,
                                    this->param.ale_formulation,
                                    &this->projection_operator->get_data().bc->dirichlet_bc,
                                    &this->periodic_face_pairs);
    }
    else
    {
      AssertThrow(false,
                  ExcMessage("Preconditioner specified for projection step is not implemented."));
    }

    // solver
    if(param.solver_projection == SolverProjection::CG)
    {
      // setup solver data
      CGSolverData solver_data;
      solver_data.max_iter             = param.solver_data_projection.max_iter;
      solver_data.solver_tolerance_abs = param.solver_data_projection.abs_tol;
      solver_data.solver_tolerance_rel = param.solver_data_projection.rel_tol;
      // default value of use_preconditioner = false
      if(param.preconditioner_projection != PreconditionerProjection::None)
      {
        solver_data.use_preconditioner = true;
      }

      // setup solver
      projection_solver.reset(new CGSolver<PROJ_OPERATOR, PreconditionerBase<Number>, VectorType>(
        *std::dynamic_pointer_cast<PROJ_OPERATOR>(projection_operator),
        *preconditioner_projection,
        solver_data));
    }
    else if(param.solver_projection == SolverProjection::FGMRES)
    {
      // setup solver data
      FGMRESSolverData solver_data;
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
      projection_solver.reset(
        new FGMRESSolver<PROJ_OPERATOR, PreconditionerBase<Number>, VectorType>(
          *std::dynamic_pointer_cast<PROJ_OPERATOR>(projection_operator),
          *preconditioner_projection,
          solver_data));
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified projection solver not implemented."));
    }
  }
  else
  {
    AssertThrow(
      param.use_divergence_penalty == false && param.use_continuity_penalty == false,
      ExcMessage(
        "Specified combination of divergence and continuity penalty operators not implemented."));
  }
}

template<int dim, typename Number>
bool
DGNavierStokesBase<dim, Number>::unsteady_problem_has_to_be_solved() const
{
  return (this->param.solver_type == SolverType::Unsteady);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::update_projection_operator(VectorType const & velocity,
                                                            double const       time_step_size) const
{
  AssertThrow(projection_operator.get() != 0,
              ExcMessage("Projection operator is not initialized."));

  // Update projection operator, i.e., the penalty parameters that depend on the velocity field
  // and the time step size
  projection_operator->update(velocity, time_step_size);
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::rhs_add_projection_operator(VectorType & dst,
                                                             double const time) const
{
  this->projection_operator->set_time(time);
  this->projection_operator->rhs_add(dst);
}

template<int dim, typename Number>
unsigned int
DGNavierStokesBase<dim, Number>::solve_projection(VectorType &       dst,
                                                  VectorType const & src,
                                                  bool const &       update_preconditioner) const
{
  Assert(projection_solver.get() != 0, ExcMessage("Projection solver has not been initialized."));

  unsigned int n_iter = projection_solver->solve(dst, src, update_preconditioner);

  return n_iter;
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::local_interpolate_velocity_dirichlet_bc_boundary_face(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &,
  Range const & face_range) const
{
  unsigned int const dof_index  = this->get_dof_index_velocity();
  unsigned int const quad_index = this->get_quad_index_velocity_gauss_lobatto();

  FaceIntegratorU integrator(matrix_free, true, dof_index, quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    types::boundary_id const boundary_id = matrix_free.get_boundary_id(face);

    BoundaryTypeU const boundary_type =
      this->boundary_descriptor_velocity->get_boundary_type(boundary_id);

    if(boundary_type == BoundaryTypeU::Dirichlet)
    {
      integrator.reinit(face);
      integrator.read_dof_values(dst);

      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        unsigned int const local_face_number = matrix_free.get_face_info(face).interior_face_no;

        unsigned int const index = matrix_free.get_shape_info(dof_index, quad_index)
                                     .face_to_cell_index_nodal[local_face_number][q];

        Point<dim, scalar> q_points = integrator.quadrature_point(q);

        typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it =
          this->boundary_descriptor_velocity->dirichlet_bc.find(boundary_id);

        vector g = evaluate_vectorial_function(it->second, q_points, this->evaluation_time);
        integrator.submit_dof_value(g, index);
      }

      integrator.set_dof_values(dst);
    }
  }
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::local_interpolate_pressure_dirichlet_bc_boundary_face(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &,
  Range const & face_range) const
{
  unsigned int const dof_index  = this->get_dof_index_pressure();
  unsigned int const quad_index = this->get_quad_index_pressure_gauss_lobatto();

  FaceIntegratorP integrator(matrix_free, true, dof_index, quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    types::boundary_id const boundary_id = matrix_free.get_boundary_id(face);

    BoundaryTypeP const boundary_type =
      this->boundary_descriptor_pressure->get_boundary_type(boundary_id);

    if(boundary_type == BoundaryTypeP::Dirichlet)
    {
      integrator.reinit(face);
      integrator.read_dof_values(dst);

      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        unsigned int const local_face_number = matrix_free.get_face_info(face).interior_face_no;

        unsigned int const index = matrix_free.get_shape_info(dof_index, quad_index)
                                     .face_to_cell_index_nodal[local_face_number][q];

        Point<dim, scalar> q_points = integrator.quadrature_point(q);

        typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it =
          this->boundary_descriptor_pressure->dirichlet_bc.find(boundary_id);

        scalar g = evaluate_scalar_function(it->second, q_points, this->evaluation_time);
        integrator.submit_dof_value(g, index);
      }

      integrator.set_dof_values(dst);
    }
  }
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::do_postprocessing(VectorType const & velocity,
                                                   VectorType const & pressure,
                                                   double const       time,
                                                   unsigned int const time_step_number) const
{
  bool const standard = true;
  if(standard)
  {
    this->postprocessor->do_postprocessing(velocity, pressure, time, time_step_number);
  }
  else // consider velocity and pressure errors instead
  {
    VectorType velocity_error;
    this->initialize_vector_velocity(velocity_error);

    VectorType pressure_error;
    this->initialize_vector_pressure(pressure_error);

    this->prescribe_initial_conditions(velocity_error, pressure_error, time);

    velocity_error.add(-1.0, velocity);
    pressure_error.add(-1.0, pressure);

    this->postprocessor->do_postprocessing(velocity_error, // error!
                                           pressure_error, // error!
                                           time,
                                           time_step_number);
  }
}

template<int dim, typename Number>
void
DGNavierStokesBase<dim, Number>::do_postprocessing_steady_problem(VectorType const & velocity,
                                                                  VectorType const & pressure) const
{
  this->postprocessor->do_postprocessing(velocity, pressure);
}

template class DGNavierStokesBase<2, float>;
template class DGNavierStokesBase<3, float>;

template class DGNavierStokesBase<2, double>;
template class DGNavierStokesBase<3, double>;

} // namespace IncNS
