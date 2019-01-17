/*
 * dg_convection_diffusion_operation.cpp
 *
 *  Created on: Dec 4, 2018
 *      Author: fehn
 */

#include "dg_convection_diffusion_operation.h"

#include "time_integration/interpolate.h"
#include "time_integration/time_step_calculation.h"

namespace ConvDiff
{
template<int dim, int degree, typename Number>
DGOperation<dim, degree, Number>::DGOperation(
  parallel::distributed::Triangulation<dim> const & triangulation,
  InputParameters const &                           param_in,
  std::shared_ptr<PostProcessor<dim, degree>>       postprocessor_in)
  : dealii::Subscriptor(),
    fe(degree),
    mapping(param_in.degree_mapping),
    dof_handler(triangulation),
    param(param_in),
    pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    postprocessor(postprocessor_in)
{
  if(param.type_velocity_field == TypeVelocityField::Numerical)
  {
    fe_velocity.reset(new FESystem<dim>(FE_DGQ<dim>(degree), dim));
    dof_handler_velocity.reset(new DoFHandler<dim>(triangulation));
  }
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::setup(
  PeriodicFaces const                            periodic_face_pairs_in,
  std::shared_ptr<BoundaryDescriptor<dim>> const boundary_descriptor_in,
  std::shared_ptr<FieldFunctions<dim>> const     field_functions_in,
  std::shared_ptr<AnalyticalSolution<dim>> const analytical_solution_in)
{
  pcout << std::endl << "Setup convection-diffusion operation ..." << std::endl;

  periodic_face_pairs = periodic_face_pairs_in;
  boundary_descriptor = boundary_descriptor_in;
  field_functions     = field_functions_in;

  create_dofs();

  initialize_matrix_free();

  setup_operators();

  setup_postprocessor(analytical_solution_in);

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::create_dofs()
{
  // enumerate degrees of freedom
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();

  if(param.type_velocity_field == TypeVelocityField::Numerical)
  {
    dof_handler_velocity->distribute_dofs(*fe_velocity);
    dof_handler_velocity->distribute_mg_dofs();
  }

  constexpr int ndofs_per_cell = Utilities::pow(degree + 1, dim);

  pcout << std::endl
        << "Discontinuous Galerkin finite element discretization:" << std::endl
        << std::endl;

  print_parameter(pcout, "degree of 1D polynomials", degree);
  print_parameter(pcout, "number of dofs per cell", ndofs_per_cell);
  print_parameter(pcout, "number of dofs (total)", dof_handler.n_dofs());
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::initialize_matrix_free()
{
  // initialize matrix_free_data
  typename MatrixFree<dim, Number>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
    MatrixFree<dim, Number>::AdditionalData::partition_partition;
  additional_data.mapping_update_flags =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
     update_values);

  additional_data.mapping_update_flags_inner_faces =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
     update_values);

  additional_data.mapping_update_flags_boundary_faces =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
     update_values);

  if(param.use_cell_based_face_loops)
  {
    auto tria = dynamic_cast<const parallel::distributed::Triangulation<dim> *>(
      &dof_handler.get_triangulation());
    Categorization::do_cell_based_loops(*tria, additional_data);
  }

  if(param.type_velocity_field == TypeVelocityField::Analytical)
  {
    AffineConstraints<double> constraint_dummy;
    constraint_dummy.close();

    // quadrature formula used to perform integrals
    QGauss<1> quadrature(degree + 1);

    data.reinit(mapping, dof_handler, constraint_dummy, quadrature, additional_data);
  }
  // we need two dof-handlers in case the velocity field comes from the fluid solver.
  else if(param.type_velocity_field == TypeVelocityField::Numerical)
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
    quadrature_vec[0] = QGauss<1>(degree + 1);

    data.reinit(mapping, dof_handler_vec, constraint_vec, quadrature_vec, additional_data);
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::setup_operators()
{
  // mass matrix operator
  MassMatrixOperatorData<dim> mass_matrix_operator_data;
  mass_matrix_operator_data.dof_index            = 0;
  mass_matrix_operator_data.quad_index           = 0;
  mass_matrix_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
  mass_matrix_operator_data.implement_block_diagonal_preconditioner_matrix_free =
    param.implement_block_diagonal_preconditioner_matrix_free;
  mass_matrix_operator.reinit(data, constraint_matrix, mass_matrix_operator_data);

  // inverse mass matrix operator
  // dof_index = 0, quad_index = 0
  inverse_mass_matrix_operator.initialize(data, 0, 0);

  // convective operator
  ConvectiveOperatorData<dim> convective_operator_data;
  convective_operator_data.dof_index                  = 0;
  convective_operator_data.quad_index                 = 0;
  convective_operator_data.type_velocity_field        = param.type_velocity_field;
  convective_operator_data.dof_index_velocity         = 1;
  convective_operator_data.numerical_flux_formulation = param.numerical_flux_convective_operator;
  convective_operator_data.bc                         = boundary_descriptor;
  convective_operator_data.velocity                   = field_functions->velocity;
  convective_operator_data.use_cell_based_loops       = param.use_cell_based_face_loops;
  convective_operator_data.implement_block_diagonal_preconditioner_matrix_free =
    param.implement_block_diagonal_preconditioner_matrix_free;

  convective_operator.reinit(data, constraint_matrix, convective_operator_data);

  if(param.type_velocity_field == TypeVelocityField::Numerical)
  {
    data.initialize_dof_vector(velocity, convective_operator_data.dof_index_velocity);
  }

  // diffusive operator
  DiffusiveOperatorData<dim> diffusive_operator_data;
  diffusive_operator_data.dof_index            = 0;
  diffusive_operator_data.quad_index           = 0;
  diffusive_operator_data.IP_factor            = param.IP_factor;
  diffusive_operator_data.diffusivity          = param.diffusivity;
  diffusive_operator_data.bc                   = boundary_descriptor;
  diffusive_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
  diffusive_operator_data.implement_block_diagonal_preconditioner_matrix_free =
    param.implement_block_diagonal_preconditioner_matrix_free;
  diffusive_operator.reinit(mapping, data, constraint_matrix, diffusive_operator_data);

  // rhs operator
  RHSOperatorData<dim> rhs_operator_data;
  rhs_operator_data.dof_index  = 0;
  rhs_operator_data.quad_index = 0;
  rhs_operator_data.rhs        = field_functions->right_hand_side;
  rhs_operator.reinit(data, rhs_operator_data);

  // convection-diffusion operator (efficient implementation, only for explicit time integration,
  // includes also rhs operator)
  ConvectionDiffusionOperatorDataEfficiency<dim, Number> conv_diff_operator_data_eff;
  conv_diff_operator_data_eff.conv_data = convective_operator_data;
  conv_diff_operator_data_eff.diff_data = diffusive_operator_data;
  conv_diff_operator_data_eff.rhs_data  = rhs_operator_data;
  convection_diffusion_operator_efficiency.initialize(mapping, data, conv_diff_operator_data_eff);
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::setup_solver(double const scaling_factor_time_derivative_term_in)
{
  pcout << std::endl << "Setup solver ..." << std::endl;

  initialize_convection_diffusion_operator(scaling_factor_time_derivative_term_in);

  initialize_preconditioner();

  initialize_solver();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::initialize_convection_diffusion_operator(
  double const scaling_factor_time_derivative_term_in)
{
  // convection-diffusion operator
  ConvectionDiffusionOperatorData<dim> conv_diff_operator_data;
  conv_diff_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
  conv_diff_operator_data.implement_block_diagonal_preconditioner_matrix_free =
    param.implement_block_diagonal_preconditioner_matrix_free;
  conv_diff_operator_data.preconditioner_block_jacobi = param.preconditioner_block_diagonal;
  conv_diff_operator_data.block_jacobi_solver_data    = param.block_jacobi_solver_data;
  conv_diff_operator_data.mass_matrix_operator_data   = mass_matrix_operator.get_operator_data();
  conv_diff_operator_data.convective_operator_data    = convective_operator.get_operator_data();
  conv_diff_operator_data.diffusive_operator_data     = diffusive_operator.get_operator_data();
  conv_diff_operator_data.scaling_factor_time_derivative_term =
    scaling_factor_time_derivative_term_in;

  if(this->param.problem_type == ProblemType::Unsteady)
  {
    conv_diff_operator_data.unsteady_problem = true;
  }
  else
  {
    conv_diff_operator_data.unsteady_problem = false;
  }

  if(this->param.equation_type == EquationType::Diffusion ||
     this->param.equation_type == EquationType::ConvectionDiffusion)
  {
    conv_diff_operator_data.diffusive_problem = true;
  }
  else
  {
    conv_diff_operator_data.diffusive_problem = false;
  }

  if((this->param.equation_type == EquationType::Convection ||
      this->param.equation_type == EquationType::ConvectionDiffusion) &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
  {
    conv_diff_operator_data.convective_problem = true;
  }
  else
  {
    conv_diff_operator_data.convective_problem = false;
  }

  conv_diff_operator_data.update_mapping_update_flags();

  conv_diff_operator_data.dof_index           = 0;
  conv_diff_operator_data.dof_index_velocity  = 1;
  conv_diff_operator_data.type_velocity_field = param.type_velocity_field;
  conv_diff_operator_data.mg_operator_type    = param.mg_operator_type;

  conv_diff_operator.reinit(data,
                            constraint_matrix,
                            conv_diff_operator_data,
                            mass_matrix_operator,
                            convective_operator,
                            diffusive_operator);
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::initialize_preconditioner()
{
  if(param.preconditioner == Preconditioner::InverseMassMatrix)
  {
    preconditioner.reset(new InverseMassMatrixPreconditioner<dim, degree, Number, 1>(data, 0, 0));
  }
  else if(param.preconditioner == Preconditioner::PointJacobi)
  {
    preconditioner.reset(new JacobiPreconditioner<ConvectionDiffusionOperator<dim, degree, Number>>(
      conv_diff_operator));
  }
  else if(param.preconditioner == Preconditioner::BlockJacobi)
  {
    preconditioner.reset(
      new BlockJacobiPreconditioner<ConvectionDiffusionOperator<dim, degree, Number>>(
        conv_diff_operator));
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

    if(param.type_velocity_field == TypeVelocityField::Numerical)
    {
      AssertThrow(dof_handler_velocity.get() != 0,
                  ExcMessage("dof_handler_velocity is not initialized."));
    }

    MultigridData mg_data;
    mg_data = param.multigrid_data;

    typedef MultigridPreconditioner<dim, degree, Number, MultigridNumber> MULTIGRID;

    preconditioner.reset(new MULTIGRID());
    std::shared_ptr<MULTIGRID> mg_preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(preconditioner);


    parallel::Triangulation<dim> const * tria =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation());
    const FiniteElement<dim> & fe = dof_handler.get_fe();
    mg_preconditioner->initialize(mg_data,
                                  tria,
                                  fe,
                                  mapping,
                                  conv_diff_operator.get_operator_data(),
                                  &conv_diff_operator.get_boundary_descriptor()->dirichlet_bc,
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

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::initialize_solver()
{
  if(param.solver == Solver::PCG)
  {
    // initialize solver_data
    CGSolverData solver_data;
    solver_data.solver_tolerance_abs  = param.abs_tol;
    solver_data.solver_tolerance_rel  = param.rel_tol;
    solver_data.max_iter              = param.max_iter;
    solver_data.update_preconditioner = param.update_preconditioner;

    if(param.preconditioner != Preconditioner::None)
      solver_data.use_preconditioner = true;

    // initialize solver
    iterative_solver.reset(
      new CGSolver<ConvectionDiffusionOperator<dim, degree, Number>,
                   PreconditionerBase<Number>,
                   VectorType>(conv_diff_operator, *preconditioner, solver_data));
  }
  else if(param.solver == Solver::GMRES)
  {
    // initialize solver_data
    GMRESSolverData solver_data;
    solver_data.solver_tolerance_abs  = param.abs_tol;
    solver_data.solver_tolerance_rel  = param.rel_tol;
    solver_data.max_iter              = param.max_iter;
    solver_data.right_preconditioning = param.use_right_preconditioner;
    solver_data.max_n_tmp_vectors     = param.max_n_tmp_vectors;
    solver_data.update_preconditioner = param.update_preconditioner;

    if(param.preconditioner != Preconditioner::None)
      solver_data.use_preconditioner = true;

    // initialize solver
    iterative_solver.reset(
      new GMRESSolver<ConvectionDiffusionOperator<dim, degree, Number>,
                      PreconditionerBase<Number>,
                      VectorType>(conv_diff_operator, *preconditioner, solver_data));
  }
  else if(param.solver == Solver::FGMRES)
  {
    // initialize solver_data
    FGMRESSolverData solver_data;
    solver_data.solver_tolerance_abs  = param.abs_tol;
    solver_data.solver_tolerance_rel  = param.rel_tol;
    solver_data.max_iter              = param.max_iter;
    solver_data.max_n_tmp_vectors     = param.max_n_tmp_vectors;
    solver_data.update_preconditioner = param.update_preconditioner;

    if(param.preconditioner != Preconditioner::None)
      solver_data.use_preconditioner = true;

    // initialize solver
    iterative_solver.reset(
      new FGMRESSolver<ConvectionDiffusionOperator<dim, degree, Number>,
                       PreconditionerBase<Number>,
                       VectorType>(conv_diff_operator, *preconditioner, solver_data));
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified solver is not implemented!"));
  }
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::initialize_dof_vector(VectorType & src) const
{
  data.initialize_dof_vector(src);
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::prescribe_initial_conditions(VectorType & src,
                                                               double const evaluation_time) const
{
  field_functions->analytical_solution->set_time(evaluation_time);
  VectorTools::interpolate(dof_handler, *(field_functions->analytical_solution), src);
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::evaluate(VectorType &       dst,
                                           VectorType const & src,
                                           double const       evaluation_time) const
{
  // apply volume and surface integrals for each operator separately
  if(param.runtime_optimization == false)
  {
    // set dst to zero
    dst = 0.0;

    // diffusive operator
    if(param.equation_type == EquationType::Diffusion ||
       param.equation_type == EquationType::ConvectionDiffusion)
    {
      diffusive_operator.evaluate_add(dst, src, evaluation_time);
    }

    // convective operator
    if(param.equation_type == EquationType::Convection ||
       param.equation_type == EquationType::ConvectionDiffusion)
    {
      if(param.type_velocity_field == TypeVelocityField::Numerical)
      {
        // We first have to interpolate the velocity field so that it is evaluated at the correct
        // time.
        interpolate(velocity, evaluation_time, velocities, times);
        convective_operator.set_velocity(velocity);
      }
      convective_operator.evaluate_add(dst, src, evaluation_time);
    }

    // shift diffusive and convective term to the rhs of the equation
    dst *= -1.0;

    if(param.right_hand_side == true)
    {
      rhs_operator.evaluate_add(dst, evaluation_time);
    }
  }
  else // param.runtime_optimization == true
  {
    convection_diffusion_operator_efficiency.evaluate(dst, src, evaluation_time);
  }

  // apply inverse mass matrix
  inverse_mass_matrix_operator.apply(dst, dst);
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::evaluate_convective_term(VectorType &       dst,
                                                           VectorType const & src,
                                                           double const       evaluation_time) const
{
  convective_operator.evaluate(dst, src, evaluation_time);
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::evaluate_negative_convective_term_and_apply_inverse_mass_matrix(
  VectorType &       dst,
  VectorType const & src,
  double const       evaluation_time) const
{
  if(param.type_velocity_field == TypeVelocityField::Numerical)
  {
    // We first have to interpolate the velocity field so that it is evaluated at the correct
    // time.
    interpolate(velocity, evaluation_time, velocities, times);
    convective_operator.set_velocity(velocity);
  }
  convective_operator.evaluate(dst, src, evaluation_time);

  // shift convective term to the rhs of the equation
  dst *= -1.0;

  inverse_mass_matrix_operator.apply(dst, dst);
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::rhs(VectorType & dst, double const evaluation_time) const
{
  // set dst to zero since we call functions of type ..._add()
  dst = 0;

  // diffusive operator
  if(param.equation_type == EquationType::Diffusion ||
     param.equation_type == EquationType::ConvectionDiffusion)
  {
    diffusive_operator.rhs_add(dst, evaluation_time);
  }

  // convective operator
  if(param.equation_type == EquationType::Convection ||
     param.equation_type == EquationType::ConvectionDiffusion)
  {
    if(param.problem_type == ProblemType::Steady ||
       (param.problem_type == ProblemType::Unsteady &&
        param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit))
    {
      convective_operator.rhs_add(dst, evaluation_time);
    }
  }

  // rhs operator f(t)
  if(param.right_hand_side == true)
  {
    rhs_operator.evaluate_add(dst, evaluation_time);
  }
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::apply_mass_matrix_add(VectorType &       dst,
                                                        VectorType const & src) const
{
  mass_matrix_operator.apply_add(dst, src);
}

template<int dim, int degree, typename Number>
unsigned int
DGOperation<dim, degree, Number>::solve(VectorType &       sol,
                                        VectorType const & rhs,
                                        double const       scaling_factor,
                                        double const       time)
{
  conv_diff_operator.set_scaling_factor_time_derivative_term(scaling_factor);
  conv_diff_operator.set_evaluation_time(time);

  unsigned int const iterations = iterative_solver->solve(sol, rhs);

  return iterations;
}

// use numerical velocity field
template<int dim, int degree, typename Number>
double
DGOperation<dim, degree, Number>::calculate_time_step_cfl(double const cfl,
                                                          double const exponent_degree) const
{
  return calculate_time_step_cfl_local<dim, degree /* = degree_velocity */, Number>(
    data,
    /*dof_index_velocity = */ 1,
    /*quad_index = */ 0,
    convective_operator.get_velocity(),
    cfl,
    exponent_degree);
}

template<int dim, int degree, typename Number>
double
DGOperation<dim, degree, Number>::calculate_time_step_cfl(double const time,
                                                          double const cfl,
                                                          double const exponent_degree) const
{
  return calculate_time_step_cfl_local<dim, degree, Number>(
    data, 0 /*dof_index*/, 0 /*quad_index*/, field_functions->velocity, time, cfl, exponent_degree);
}

template<int dim, int degree, typename Number>
double
DGOperation<dim, degree, Number>::calculate_maximum_velocity(double const time) const
{
  return calculate_max_velocity(dof_handler.get_triangulation(), field_functions->velocity, time);
}

template<int dim, int degree, typename Number>
double
DGOperation<dim, degree, Number>::calculate_minimum_element_length() const
{
  return calculate_minimum_vertex_distance(dof_handler.get_triangulation());
}

template<int dim, int degree, typename Number>
MatrixFree<dim, Number> const &
DGOperation<dim, degree, Number>::get_data() const
{
  return data;
}

template<int dim, int degree, typename Number>
Mapping<dim> const &
DGOperation<dim, degree, Number>::get_mapping() const
{
  return mapping;
}

template<int dim, int degree, typename Number>
DoFHandler<dim> const &
DGOperation<dim, degree, Number>::get_dof_handler() const
{
  return dof_handler;
}

template<int dim, int degree, typename Number>
unsigned int
DGOperation<dim, degree, Number>::get_polynomial_degree() const
{
  return degree;
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::set_velocity(VectorType const & velocity) const
{
  AssertThrow(param.type_velocity_field == TypeVelocityField::Numerical,
              ExcMessage("Invalid parameter type_velocity_field."));

  convective_operator.set_velocity(velocity);
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::set_velocities_and_times(
  std::vector<VectorType const *> & velocities_in,
  std::vector<double> &             times_in) const
{
  velocities = velocities_in;
  times      = times_in;
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::do_postprocessing(VectorType const & solution,
                                                    double const       time,
                                                    int const          time_step_number) const
{
  postprocessor->do_postprocessing(solution, time, time_step_number);
}

template<int dim, int degree, typename Number>
void
DGOperation<dim, degree, Number>::setup_postprocessor(
  std::shared_ptr<AnalyticalSolution<dim>> analytical_solution)
{
  PostProcessorData pp_data;
  pp_data.output_data = param.output_data;
  pp_data.error_data  = param.error_data;

  postprocessor->setup(pp_data, dof_handler, mapping, data, analytical_solution);
}

} // namespace ConvDiff

#include "dg_convection_diffusion_operation.hpp"
