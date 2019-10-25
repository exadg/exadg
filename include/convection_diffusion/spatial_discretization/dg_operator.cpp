/*
 * dg_convection_diffusion_operation.cpp
 *
 *  Created on: Dec 4, 2018
 *      Author: fehn
 */

#include "dg_operator.h"

#include "project_velocity.h"
#include "time_integration/time_step_calculation.h"

namespace ConvDiff
{
template<int dim, typename Number>
DGOperator<dim, Number>::DGOperator(
  parallel::TriangulationBase<dim> const &            triangulation,
  InputParameters const &                         param_in,
  std::shared_ptr<PostProcessorBase<dim, Number>> postprocessor_in)
  : dealii::Subscriptor(),
    param(param_in),
    fe(param.degree),
    mapping_degree(1),
    dof_handler(triangulation),
    pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    postprocessor(postprocessor_in)
{
  if(param.mapping == MappingType::Affine)
  {
    mapping_degree = 1;
  }
  else if(param.mapping == MappingType::Isoparametric)
  {
    mapping_degree = param.degree;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented"));
  }

  mapping.reset(new MappingQGeneric<dim>(mapping_degree));

  if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
  {
    fe_velocity.reset(new FESystem<dim>(FE_DGQ<dim>(param.degree), dim));
    dof_handler_velocity.reset(new DoFHandler<dim>(triangulation));
  }
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::setup(
  PeriodicFaces const                            periodic_face_pairs_in,
  std::shared_ptr<BoundaryDescriptor<dim>> const boundary_descriptor_in,
  std::shared_ptr<FieldFunctions<dim>> const     field_functions_in)
{
  pcout << std::endl << "Setup spatial discretization (DoFs, MatrixFree) ..." << std::endl;

  periodic_face_pairs = periodic_face_pairs_in;
  boundary_descriptor = boundary_descriptor_in;
  field_functions     = field_functions_in;

  create_dofs();

  initialize_matrix_free();

  setup_postprocessor();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::create_dofs()
{
  // enumerate degrees of freedom
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();

  if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
  {
    dof_handler_velocity->distribute_dofs(*fe_velocity);
    dof_handler_velocity->distribute_mg_dofs();
  }

  unsigned int const ndofs_per_cell = Utilities::pow(param.degree + 1, dim);

  pcout << std::endl
        << "Discontinuous Galerkin finite element discretization:" << std::endl
        << std::endl;

  print_parameter(pcout, "degree of 1D polynomials", param.degree);
  print_parameter(pcout, "number of dofs per cell", ndofs_per_cell);
  print_parameter(pcout, "number of dofs (total)", dof_handler.n_dofs());
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::initialize_matrix_free()
{
  // initialize matrix_free_data
  typename MatrixFree<dim, Number>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
    MatrixFree<dim, Number>::AdditionalData::partition_partition;

  MappingFlags mapping_flags;

  if(param.problem_type == ProblemType::Unsteady)
  {
    mapping_flags = mapping_flags || MassMatrixKernel<dim, Number>::get_mapping_flags();
  }

  if(param.right_hand_side)
  {
    mapping_flags = mapping_flags || Operators::RHSKernel<dim, Number>::get_mapping_flags();
  }

  if(param.equation_type == EquationType::Convection ||
     param.equation_type == EquationType::ConvectionDiffusion)
  {
    mapping_flags = mapping_flags || Operators::ConvectiveKernel<dim, Number>::get_mapping_flags();
  }

  if(param.equation_type == EquationType::Diffusion ||
     param.equation_type == EquationType::ConvectionDiffusion)
  {
    mapping_flags = mapping_flags || Operators::DiffusiveKernel<dim, Number>::get_mapping_flags();
  }

  additional_data.mapping_update_flags                = mapping_flags.cells;
  additional_data.mapping_update_flags_inner_faces    = mapping_flags.inner_faces;
  additional_data.mapping_update_flags_boundary_faces = mapping_flags.boundary_faces;

  if(param.use_cell_based_face_loops)
  {
    auto tria = dynamic_cast<const parallel::distributed::Triangulation<dim> *>(
      &dof_handler.get_triangulation());
    Categorization::do_cell_based_loops(*tria, additional_data);
  }

  // we need two dof-handlers in case the velocity field is stored in a DoF vector
  if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
  {
    std::vector<const DoFHandler<dim> *> dof_handler_vec;
    dof_handler_vec.resize(2);
    dof_handler_vec[0] = &dof_handler;
    dof_handler_vec[1] = &(*dof_handler_velocity);

    std::vector<const AffineConstraints<double> *> constraint_vec;
    constraint_vec.resize(2);
    AffineConstraints<double> constraint_dummy;
    constraint_dummy.close();
    constraint_vec[0] = &constraint_dummy;
    constraint_vec[1] = &constraint_dummy;

    std::vector<Quadrature<1>> quadrature_vec;
    quadrature_vec.resize(1);
    quadrature_vec[0] = QGauss<1>(param.degree + 1);

    matrix_free.reinit(*mapping, dof_handler_vec, constraint_vec, quadrature_vec, additional_data);
  }
  else
  {
    AssertThrow(param.analytical_velocity_field == true, ExcMessage("Invalid parameter."));

    AffineConstraints<double> constraint_dummy;
    constraint_dummy.close();

    // quadrature formula used to perform integrals
    QGauss<1> quadrature(param.degree + 1);

    matrix_free.reinit(*mapping, dof_handler, constraint_dummy, quadrature, additional_data);
  }
}

template<int dim, typename Number>
int
DGOperator<dim, Number>::get_dof_index_velocity() const
{
  return 1;
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::setup_operators(double const       scaling_factor_mass_matrix,
                                         VectorType const * velocity)
{
  // mass matrix operator
  MassMatrixOperatorData mass_matrix_operator_data;
  mass_matrix_operator_data.dof_index            = 0;
  mass_matrix_operator_data.quad_index           = 0;
  mass_matrix_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
  mass_matrix_operator_data.implement_block_diagonal_preconditioner_matrix_free =
    param.implement_block_diagonal_preconditioner_matrix_free;

  mass_matrix_operator.reinit(matrix_free, constraint_matrix, mass_matrix_operator_data);

  // inverse mass matrix operator
  // dof_index = 0, quad_index = 0
  inverse_mass_matrix_operator.initialize(matrix_free, param.degree, 0, 0);

  // convective operator
  Operators::ConvectiveKernelData<dim> convective_kernel_data;
  convective_kernel_data.velocity_type              = param.get_type_velocity_field();
  convective_kernel_data.dof_index_velocity         = get_dof_index_velocity();
  convective_kernel_data.numerical_flux_formulation = param.numerical_flux_convective_operator;
  convective_kernel_data.velocity                   = field_functions->velocity;

  ConvectiveOperatorData<dim> convective_operator_data;
  convective_operator_data.dof_index            = 0;
  convective_operator_data.quad_index           = 0;
  convective_operator_data.bc                   = boundary_descriptor;
  convective_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
  convective_operator_data.implement_block_diagonal_preconditioner_matrix_free =
    param.implement_block_diagonal_preconditioner_matrix_free;
  convective_operator_data.kernel_data = convective_kernel_data;

  if(this->param.equation_type == EquationType::Convection ||
     this->param.equation_type == EquationType::ConvectionDiffusion)
  {
    convective_operator.reinit(matrix_free, constraint_matrix, convective_operator_data);
  }

  // diffusive operator
  Operators::DiffusiveKernelData diffusive_kernel_data;
  diffusive_kernel_data.IP_factor      = param.IP_factor;
  diffusive_kernel_data.diffusivity    = param.diffusivity;
  diffusive_kernel_data.degree         = param.degree;
  diffusive_kernel_data.degree_mapping = mapping_degree;

  DiffusiveOperatorData<dim> diffusive_operator_data;
  diffusive_operator_data.dof_index            = 0;
  diffusive_operator_data.quad_index           = 0;
  diffusive_operator_data.bc                   = boundary_descriptor;
  diffusive_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
  diffusive_operator_data.implement_block_diagonal_preconditioner_matrix_free =
    param.implement_block_diagonal_preconditioner_matrix_free;
  diffusive_operator_data.kernel_data = diffusive_kernel_data;

  if(this->param.equation_type == EquationType::Diffusion ||
     this->param.equation_type == EquationType::ConvectionDiffusion)
  {
    diffusive_operator.reinit(matrix_free, constraint_matrix, diffusive_operator_data);
  }

  // rhs operator
  RHSOperatorData<dim> rhs_operator_data;
  rhs_operator_data.dof_index     = 0;
  rhs_operator_data.quad_index    = 0;
  rhs_operator_data.kernel_data.f = field_functions->right_hand_side;
  rhs_operator.reinit(matrix_free, rhs_operator_data);

  // merged operator
  OperatorData<dim> combined_operator_data;
  combined_operator_data.dof_index            = 0;
  combined_operator_data.quad_index           = 0;
  combined_operator_data.bc                   = boundary_descriptor;
  combined_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
  combined_operator_data.implement_block_diagonal_preconditioner_matrix_free =
    param.implement_block_diagonal_preconditioner_matrix_free;
  combined_operator_data.solver_block_diagonal         = param.solver_block_diagonal;
  combined_operator_data.preconditioner_block_diagonal = param.preconditioner_block_diagonal;
  combined_operator_data.solver_data_block_diagonal    = param.solver_data_block_diagonal;

  // linear system of equations has to be solved: the problem is either steady or
  // an unsteady problem is solved with BDF time integration (semi-implicit or fully implicit
  // formulation of convective and diffusive terms)
  if(this->param.problem_type == ProblemType::Steady ||
     this->param.temporal_discretization == TemporalDiscretization::BDF)
  {
    if(this->param.problem_type == ProblemType::Unsteady)
      combined_operator_data.unsteady_problem = true;

    if((this->param.equation_type == EquationType::Convection ||
        this->param.equation_type == EquationType::ConvectionDiffusion) &&
       this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      combined_operator_data.convective_problem = true;

    if(this->param.equation_type == EquationType::Diffusion ||
       this->param.equation_type == EquationType::ConvectionDiffusion)
      combined_operator_data.diffusive_problem = true;
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::ExplRK)
  {
    // always false
    combined_operator_data.unsteady_problem = false;

    if(this->param.equation_type == EquationType::Convection ||
       this->param.equation_type == EquationType::ConvectionDiffusion)
      combined_operator_data.convective_problem = true;

    if(this->param.equation_type == EquationType::Diffusion ||
       this->param.equation_type == EquationType::ConvectionDiffusion)
      combined_operator_data.diffusive_problem = true;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  combined_operator_data.scaling_factor_mass_matrix = scaling_factor_mass_matrix;

  combined_operator_data.convective_kernel_data = convective_kernel_data;
  combined_operator_data.diffusive_kernel_data  = diffusive_kernel_data;

  combined_operator.reinit(matrix_free, constraint_matrix, combined_operator_data);

  // The velocity vector needs to be set in case the velocity field is stored in DoF Vector.
  // Otherwise, certain preconditioners requiring the velocity field during initialization can not
  // be initialized.
  if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
  {
    AssertThrow(velocity != nullptr,
                ExcMessage(
                  "In case of a numerical velocity field, a velocity vector has to be provided."));

    combined_operator.set_velocity_ptr(*velocity);
  }
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::setup_operators_and_solver(double const       scaling_factor_mass_matrix,
                                                    VectorType const * velocity)
{
  pcout << std::endl << "Setup operators and solver ..." << std::endl;

  setup_operators(scaling_factor_mass_matrix, velocity);

  if(param.linear_system_has_to_be_solved())
  {
    initialize_preconditioner();

    initialize_solver();
  }

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::initialize_preconditioner()
{
  if(param.preconditioner == Preconditioner::InverseMassMatrix)
  {
    preconditioner.reset(
      new InverseMassMatrixPreconditioner<dim, 1, Number>(matrix_free, param.degree, 0, 0));
  }
  else if(param.preconditioner == Preconditioner::PointJacobi)
  {
    typedef Operator<dim, Number> Operator;
    preconditioner.reset(new JacobiPreconditioner<Operator>(combined_operator));
  }
  else if(param.preconditioner == Preconditioner::BlockJacobi)
  {
    typedef Operator<dim, Number> Operator;
    preconditioner.reset(new BlockJacobiPreconditioner<Operator>(combined_operator));
  }
  else if(param.preconditioner == Preconditioner::Multigrid)
  {
    if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
    {
      AssertThrow(param.mg_operator_type != MultigridOperatorType::ReactionConvection &&
                    param.mg_operator_type != MultigridOperatorType::ReactionConvectionDiffusion,
                  ExcMessage(
                    "Invalid solver parameters. The convective term is treated explicitly."));
    }

    if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
    {
      AssertThrow(dof_handler_velocity.get() != 0,
                  ExcMessage("dof_handler_velocity is not initialized."));
    }

    MultigridData mg_data;
    mg_data = param.multigrid_data;

    typedef MultigridPreconditioner<dim, Number, MultigridNumber> MULTIGRID;

    preconditioner.reset(new MULTIGRID());
    std::shared_ptr<MULTIGRID> mg_preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(preconditioner);

    parallel::TriangulationBase<dim> const * tria =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(&dof_handler.get_triangulation());

    const FiniteElement<dim> & fe = dof_handler.get_fe();

    OperatorData<dim> const & data = combined_operator.get_data();

    mg_preconditioner->initialize(mg_data,
                                  tria,
                                  fe,
                                  *mapping,
                                  combined_operator,
                                  param.mg_operator_type,
                                  &data.bc->dirichlet_bc,
                                  &this->periodic_face_pairs);
  }
  else
  {
    AssertThrow(param.preconditioner == Preconditioner::None ||
                  param.preconditioner == Preconditioner::InverseMassMatrix ||
                  param.preconditioner == Preconditioner::PointJacobi ||
                  param.preconditioner == Preconditioner::BlockJacobi ||
                  param.preconditioner == Preconditioner::Multigrid,
                ExcMessage("Specified preconditioner is not implemented!"));
  }
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::initialize_solver()
{
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
    iterative_solver.reset(
      new CGSolver<Operator<dim, Number>, PreconditionerBase<Number>, VectorType>(combined_operator,
                                                                                  *preconditioner,
                                                                                  solver_data));
  }
  else if(param.solver == Solver::GMRES)
  {
    // initialize solver_data
    GMRESSolverData solver_data;
    solver_data.solver_tolerance_abs = param.solver_data.abs_tol;
    solver_data.solver_tolerance_rel = param.solver_data.rel_tol;
    solver_data.max_iter             = param.solver_data.max_iter;
    solver_data.max_n_tmp_vectors    = param.solver_data.max_krylov_size;

    if(param.preconditioner != Preconditioner::None)
      solver_data.use_preconditioner = true;

    // initialize solver
    iterative_solver.reset(
      new GMRESSolver<Operator<dim, Number>, PreconditionerBase<Number>, VectorType>(
        combined_operator, *preconditioner, solver_data));
  }
  else if(param.solver == Solver::FGMRES)
  {
    // initialize solver_data
    FGMRESSolverData solver_data;
    solver_data.solver_tolerance_abs = param.solver_data.abs_tol;
    solver_data.solver_tolerance_rel = param.solver_data.rel_tol;
    solver_data.max_iter             = param.solver_data.max_iter;
    solver_data.max_n_tmp_vectors    = param.solver_data.max_krylov_size;

    if(param.preconditioner != Preconditioner::None)
      solver_data.use_preconditioner = true;

    // initialize solver
    iterative_solver.reset(
      new FGMRESSolver<Operator<dim, Number>, PreconditionerBase<Number>, VectorType>(
        combined_operator, *preconditioner, solver_data));
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified solver is not implemented!"));
  }
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::initialize_dof_vector(VectorType & src) const
{
  matrix_free.initialize_dof_vector(src);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::initialize_dof_vector_velocity(VectorType & velocity) const
{
  matrix_free.initialize_dof_vector(velocity, get_dof_index_velocity());
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::interpolate_velocity(VectorType & velocity, double const time) const
{
  field_functions->velocity->set_time(time);

  // This is necessary if Number == float
  typedef LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

  VectorTypeDouble vector_double;
  vector_double = velocity;

  VectorTools::interpolate(*dof_handler_velocity, *(field_functions->velocity), vector_double);

  velocity = vector_double;
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::project_velocity(VectorType & velocity, double const time) const
{
  VelocityProjection<dim, Number> l2_projection;

  unsigned int const quad_index = 0;
  l2_projection.apply(matrix_free,
                      get_dof_index_velocity(),
                      quad_index,
                      param.degree,
                      field_functions->velocity,
                      time,
                      velocity);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::prescribe_initial_conditions(VectorType & src, double const time) const
{
  field_functions->initial_solution->set_time(time);

  // This is necessary if Number == float
  typedef LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

  VectorTypeDouble src_double;
  src_double = src;

  VectorTools::interpolate(dof_handler, *(field_functions->initial_solution), src_double);

  src = src_double;
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::evaluate_explicit_time_int(VectorType &       dst,
                                                    VectorType const & src,
                                                    double const       time,
                                                    VectorType const * velocity) const
{
  // evaluate each operator separately
  if(param.use_combined_operator == false)
  {
    // set dst to zero
    dst = 0.0;

    // diffusive operator
    if(param.equation_type == EquationType::Diffusion ||
       param.equation_type == EquationType::ConvectionDiffusion)
    {
      diffusive_operator.set_time(time);
      diffusive_operator.evaluate_add(dst, src);
    }

    // convective operator
    if(param.equation_type == EquationType::Convection ||
       param.equation_type == EquationType::ConvectionDiffusion)
    {
      if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
      {
        AssertThrow(velocity != nullptr, ExcMessage("velocity pointer is not initialized."));

        convective_operator.set_velocity_ptr(*velocity);
      }

      convective_operator.set_time(time);
      convective_operator.evaluate_add(dst, src);
    }

    // shift diffusive and convective term to the rhs of the equation
    dst *= -1.0;

    if(param.right_hand_side == true)
    {
      rhs_operator.evaluate_add(dst, time);
    }
  }
  else // param.use_combined_operator == true
  {
    // no need to set scaling_factor_mass_matrix because the mass matrix is not evaluated
    // in case of explicit time integration

    if(param.equation_type == EquationType::Convection ||
       param.equation_type == EquationType::ConvectionDiffusion)
    {
      if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
      {
        AssertThrow(velocity != nullptr, ExcMessage("velocity pointer is not initialized."));

        combined_operator.set_velocity_ptr(*velocity);
      }
    }

    combined_operator.set_time(time);
    combined_operator.evaluate(dst, src);

    // shift diffusive and convective term to the rhs of the equation
    dst *= -1.0;

    if(param.right_hand_side == true)
    {
      rhs_operator.evaluate_add(dst, time);
    }
  }

  // apply inverse mass matrix
  inverse_mass_matrix_operator.apply(dst, dst);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::evaluate_convective_term(VectorType &       dst,
                                                  VectorType const & src,
                                                  double const       time,
                                                  VectorType const * velocity) const
{
  if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
  {
    AssertThrow(velocity != nullptr, ExcMessage("velocity pointer is not initialized."));

    convective_operator.set_velocity_ptr(*velocity);
  }

  convective_operator.set_time(time);
  convective_operator.evaluate(dst, src);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::evaluate_oif(VectorType &       dst,
                                      VectorType const & src,
                                      double const       time,
                                      VectorType const * velocity) const
{
  if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
  {
    AssertThrow(velocity != nullptr, ExcMessage("velocity pointer is not initialized."));

    convective_operator.set_velocity_ptr(*velocity);
  }

  convective_operator.set_time(time);
  convective_operator.evaluate(dst, src);

  // shift convective term to the rhs of the equation
  dst *= -1.0;

  inverse_mass_matrix_operator.apply(dst, dst);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::rhs(VectorType & dst, double const time, VectorType const * velocity) const
{
  // evaluate each operator separately
  if(param.use_combined_operator == false)
  {
    // set dst to zero since we call functions of type ..._add()
    dst = 0;

    // diffusive operator
    if(param.equation_type == EquationType::Diffusion ||
       param.equation_type == EquationType::ConvectionDiffusion)
    {
      diffusive_operator.set_time(time);
      diffusive_operator.rhs_add(dst);
    }

    // convective operator
    if(param.linear_system_including_convective_term_has_to_be_solved())
    {
      if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
      {
        AssertThrow(velocity != nullptr, ExcMessage("velocity pointer is not initialized."));

        convective_operator.set_velocity_ptr(*velocity);
      }

      convective_operator.set_time(time);
      convective_operator.rhs_add(dst);
    }
  }
  else // param.use_combined_operator == true
  {
    // no need to set scaling_factor_mass_matrix because the mass matrix does not contribute to the
    // rhs

    if(param.linear_system_including_convective_term_has_to_be_solved())
    {
      if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
      {
        AssertThrow(velocity != nullptr, ExcMessage("velocity pointer is not initialized."));

        combined_operator.set_velocity_ptr(*velocity);
      }
    }

    combined_operator.set_time(time);
    combined_operator.rhs(dst);
  }

  // rhs operator f(t)
  if(param.right_hand_side == true)
  {
    rhs_operator.evaluate_add(dst, time);
  }
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::apply_mass_matrix(VectorType & dst, VectorType const & src) const
{
  mass_matrix_operator.apply(dst, src);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::apply_mass_matrix_add(VectorType & dst, VectorType const & src) const
{
  mass_matrix_operator.apply_add(dst, src);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::apply_convective_term(VectorType & dst, VectorType const & src) const
{
  convective_operator.apply(dst, src);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::update_convective_term(double const       time,
                                                VectorType const * velocity) const
{
  if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
  {
    AssertThrow(velocity != nullptr, ExcMessage("velocity pointer is not initialized."));

    convective_operator.set_velocity_ptr(*velocity);
  }

  convective_operator.set_time(time);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::apply_diffusive_term(VectorType & dst, VectorType const & src) const
{
  diffusive_operator.apply(dst, src);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::apply_conv_diff_operator(VectorType & dst, VectorType const & src) const
{
  combined_operator.apply(dst, src);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::update_conv_diff_operator(double const       time,
                                                   double const       scaling_factor,
                                                   VectorType const * velocity)
{
  combined_operator.set_scaling_factor_mass_matrix(scaling_factor);
  combined_operator.set_time(time);

  if(param.linear_system_including_convective_term_has_to_be_solved())
  {
    if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
    {
      AssertThrow(velocity != nullptr, ExcMessage("velocity pointer is not initialized."));

      combined_operator.set_velocity_ptr(*velocity);
    }
  }
}


template<int dim, typename Number>
unsigned int
DGOperator<dim, Number>::solve(VectorType &       sol,
                               VectorType const & rhs,
                               bool const         update_preconditioner,
                               double const       scaling_factor,
                               double const       time,
                               VectorType const * velocity)
{
  combined_operator.set_scaling_factor_mass_matrix(scaling_factor);
  combined_operator.set_time(time);

  if(param.linear_system_including_convective_term_has_to_be_solved())
  {
    if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
    {
      AssertThrow(velocity != nullptr, ExcMessage("velocity pointer is not initialized."));

      combined_operator.set_velocity_ptr(*velocity);
    }
  }

  unsigned int const iterations = iterative_solver->solve(sol, rhs, update_preconditioner);

  return iterations;
}

// use numerical velocity field
template<int dim, typename Number>
double
DGOperator<dim, Number>::calculate_time_step_cfl_numerical_velocity(
  VectorType const & velocity,
  double const       cfl,
  double const       exponent_degree) const
{
  return calculate_time_step_cfl_local<dim, Number>(matrix_free,
                                                    /*dof_index_velocity = */ 1,
                                                    /*quad_index = */ 0,
                                                    velocity,
                                                    cfl,
                                                    param.degree,
                                                    exponent_degree,
                                                    param.adaptive_time_stepping_cfl_type);
}

template<int dim, typename Number>
double
DGOperator<dim, Number>::calculate_time_step_cfl_analytical_velocity(
  double const time,
  double const cfl,
  double const exponent_degree) const
{
  return calculate_time_step_cfl_local<dim, Number>(matrix_free,
                                                    0 /*dof_index*/,
                                                    0 /*quad_index*/,
                                                    field_functions->velocity,
                                                    time,
                                                    cfl,
                                                    param.degree,
                                                    exponent_degree,
                                                    param.adaptive_time_stepping_cfl_type);
}

template<int dim, typename Number>
double
DGOperator<dim, Number>::calculate_maximum_velocity(double const time) const
{
  return calculate_max_velocity(dof_handler.get_triangulation(), field_functions->velocity, time);
}

template<int dim, typename Number>
double
DGOperator<dim, Number>::calculate_minimum_element_length() const
{
  return calculate_minimum_vertex_distance(dof_handler.get_triangulation());
}

template<int dim, typename Number>
Mapping<dim> const &
DGOperator<dim, Number>::get_mapping() const
{
  return *mapping;
}

template<int dim, typename Number>
DoFHandler<dim> const &
DGOperator<dim, Number>::get_dof_handler() const
{
  return dof_handler;
}

template<int dim, typename Number>
unsigned int
DGOperator<dim, Number>::get_polynomial_degree() const
{
  return param.degree;
}

template<int dim, typename Number>
types::global_dof_index
DGOperator<dim, Number>::get_number_of_dofs() const
{
  return dof_handler.n_dofs();
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::do_postprocessing(VectorType const & solution,
                                           double const       time,
                                           int const          time_step_number) const
{
  postprocessor->do_postprocessing(solution, time, time_step_number);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::setup_postprocessor()
{
  postprocessor->setup(dof_handler, *mapping);
}

template class DGOperator<2, float>;
template class DGOperator<2, double>;

template class DGOperator<3, float>;
template class DGOperator<3, double>;

} // namespace ConvDiff
