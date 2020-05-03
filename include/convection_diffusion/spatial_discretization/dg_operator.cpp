/*
 * dg_operator.cpp
 *
 *  Created on: Dec 4, 2018
 *      Author: fehn
 */

#include "dg_operator.h"

// deal.II
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>

#include "project_velocity.h"
#include "time_integration/time_step_calculation.h"

// solvers and preconditioners
#include "../../solvers_and_preconditioners/preconditioner/inverse_mass_matrix_preconditioner.h"
#include "../../solvers_and_preconditioners/preconditioner/jacobi_preconditioner.h"
#include "../../solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h"
#include "../preconditioners/multigrid_preconditioner.h"

namespace ConvDiff
{
template<int dim, typename Number>
DGOperator<dim, Number>::DGOperator(
  parallel::TriangulationBase<dim> const &       triangulation_in,
  Mapping<dim> const &                           mapping_in,
  unsigned int const                             degree_in,
  PeriodicFaces const                            periodic_face_pairs_in,
  std::shared_ptr<BoundaryDescriptor<dim>> const boundary_descriptor_in,
  std::shared_ptr<FieldFunctions<dim>> const     field_functions_in,
  InputParameters const &                        param_in,
  std::string const &                            field_in,
  MPI_Comm const &                               mpi_comm_in)
  : dealii::Subscriptor(),
    mapping(mapping_in),
    degree(degree_in),
    periodic_face_pairs(periodic_face_pairs_in),
    boundary_descriptor(boundary_descriptor_in),
    field_functions(field_functions_in),
    param(param_in),
    field(field_in),
    fe(degree_in),
    dof_handler(triangulation_in),
    mpi_comm(mpi_comm_in),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm_in) == 0)
{
  pcout << std::endl << "Construct convection-diffusion operator ..." << std::endl;

  if(needs_own_dof_handler_velocity())
  {
    fe_velocity.reset(new FESystem<dim>(FE_DGQ<dim>(degree), dim));
    dof_handler_velocity.reset(new DoFHandler<dim>(triangulation_in));
  }

  distribute_dofs();

  constraint_matrix.close();

  pcout << std::endl << "... done!" << std::endl;
}


template<int dim, typename Number>
void
DGOperator<dim, Number>::fill_matrix_free_data(MatrixFreeData<dim, Number> & matrix_free_data) const
{
  // append mapping flags
  if(param.problem_type == ProblemType::Unsteady)
  {
    matrix_free_data.append_mapping_flags(MassMatrixKernel<dim, Number>::get_mapping_flags());
  }

  if(param.right_hand_side)
  {
    matrix_free_data.append_mapping_flags(Operators::RHSKernel<dim, Number>::get_mapping_flags());
  }

  if(param.convective_problem())
  {
    matrix_free_data.append_mapping_flags(
      Operators::ConvectiveKernel<dim, Number>::get_mapping_flags());
  }

  if(param.diffusive_problem())
  {
    matrix_free_data.append_mapping_flags(
      Operators::DiffusiveKernel<dim, Number>::get_mapping_flags(true, true));
  }

  // DoFHandler, AffineConstraints
  matrix_free_data.insert_dof_handler(&dof_handler, get_dof_name());
  matrix_free_data.insert_constraint(&constraint_matrix, get_dof_name());

  if(needs_own_dof_handler_velocity())
  {
    matrix_free_data.insert_dof_handler(&(*dof_handler_velocity), get_dof_name_velocity());
    matrix_free_data.insert_constraint(&constraint_matrix, get_dof_name_velocity());
  }

  // Quadrature
  matrix_free_data.insert_quadrature(QGauss<1>(degree + 1), get_quad_name());

  if(param.use_overintegration)
  {
    matrix_free_data.insert_quadrature(QGauss<1>(degree + (degree + 2) / 2),
                                       get_quad_name_overintegration());
  }
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::setup(std::shared_ptr<MatrixFree<dim, Number>>     matrix_free_in,
                               std::shared_ptr<MatrixFreeData<dim, Number>> matrix_free_data_in,
                               std::string const & dof_index_velocity_external_in)
{
  pcout << std::endl << "Setup convection-diffusion operator ..." << std::endl;

  matrix_free      = matrix_free_in;
  matrix_free_data = matrix_free_data_in;

  dof_index_velocity_external = dof_index_velocity_external_in;

  // mass matrix operator
  MassMatrixOperatorData<dim> mass_matrix_operator_data;
  mass_matrix_operator_data.dof_index            = get_dof_index();
  mass_matrix_operator_data.quad_index           = get_quad_index();
  mass_matrix_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
  mass_matrix_operator_data.implement_block_diagonal_preconditioner_matrix_free =
    param.implement_block_diagonal_preconditioner_matrix_free;

  mass_matrix_operator.initialize(*matrix_free, constraint_matrix, mass_matrix_operator_data);

  // inverse mass matrix operator
  inverse_mass_matrix_operator.initialize(*matrix_free, get_dof_index(), get_quad_index());

  // convective operator
  unsigned int const quad_index_convective =
    param.use_overintegration ? get_quad_index_overintegration() : get_quad_index();

  Operators::ConvectiveKernelData<dim> convective_kernel_data;

  if(param.convective_problem())
  {
    convective_kernel_data.formulation                = param.formulation_convective_term;
    convective_kernel_data.velocity_type              = param.get_type_velocity_field();
    convective_kernel_data.dof_index_velocity         = get_dof_index_velocity();
    convective_kernel_data.numerical_flux_formulation = param.numerical_flux_convective_operator;
    convective_kernel_data.velocity                   = field_functions->velocity;

    convective_kernel.reset(new Operators::ConvectiveKernel<dim, Number>());
    convective_kernel->reinit(*matrix_free,
                              convective_kernel_data,
                              quad_index_convective,
                              false /* is_mg */);

    ConvectiveOperatorData<dim> convective_operator_data;
    convective_operator_data.dof_index            = get_dof_index();
    convective_operator_data.quad_index           = quad_index_convective;
    convective_operator_data.bc                   = boundary_descriptor;
    convective_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
    convective_operator_data.implement_block_diagonal_preconditioner_matrix_free =
      param.implement_block_diagonal_preconditioner_matrix_free;
    convective_operator_data.kernel_data = convective_kernel_data;

    convective_operator.initialize(*matrix_free,
                                   constraint_matrix,
                                   convective_operator_data,
                                   convective_kernel);
  }

  // diffusive operator
  Operators::DiffusiveKernelData diffusive_kernel_data;

  if(param.diffusive_problem())
  {
    diffusive_kernel_data.IP_factor   = param.IP_factor;
    diffusive_kernel_data.diffusivity = param.diffusivity;

    diffusive_kernel.reset(new Operators::DiffusiveKernel<dim, Number>());
    diffusive_kernel->reinit(*matrix_free, diffusive_kernel_data, get_dof_index());

    DiffusiveOperatorData<dim> diffusive_operator_data;
    diffusive_operator_data.dof_index            = get_dof_index();
    diffusive_operator_data.quad_index           = get_quad_index();
    diffusive_operator_data.bc                   = boundary_descriptor;
    diffusive_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
    diffusive_operator_data.implement_block_diagonal_preconditioner_matrix_free =
      param.implement_block_diagonal_preconditioner_matrix_free;
    diffusive_operator_data.kernel_data = diffusive_kernel_data;

    diffusive_operator.initialize(*matrix_free,
                                  constraint_matrix,
                                  diffusive_operator_data,
                                  diffusive_kernel);
  }

  // rhs operator
  RHSOperatorData<dim> rhs_operator_data;
  rhs_operator_data.dof_index     = get_dof_index();
  rhs_operator_data.quad_index    = get_quad_index();
  rhs_operator_data.kernel_data.f = field_functions->right_hand_side;
  rhs_operator.initialize(*matrix_free, rhs_operator_data);

  // merged operator
  if(param.temporal_discretization == TemporalDiscretization::BDF ||
     (param.temporal_discretization == TemporalDiscretization::ExplRK &&
      param.use_combined_operator == true))
  {
    OperatorData<dim> combined_operator_data;
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
    if(param.problem_type == ProblemType::Steady ||
       param.temporal_discretization == TemporalDiscretization::BDF)
    {
      if(param.problem_type == ProblemType::Unsteady)
        combined_operator_data.unsteady_problem = true;

      if(param.convective_problem() &&
         param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
        combined_operator_data.convective_problem = true;

      if(param.diffusive_problem())
        combined_operator_data.diffusive_problem = true;
    }
    else if(param.temporal_discretization == TemporalDiscretization::ExplRK)
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

    combined_operator_data.convective_kernel_data = convective_kernel_data;
    combined_operator_data.diffusive_kernel_data  = diffusive_kernel_data;

    combined_operator_data.dof_index = get_dof_index();
    combined_operator_data.quad_index =
      (param.use_overintegration && combined_operator_data.convective_problem) ?
        get_quad_index_overintegration() :
        get_quad_index();

    combined_operator.initialize(
      *matrix_free, constraint_matrix, combined_operator_data, convective_kernel, diffusive_kernel);
  }

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::distribute_dofs()
{
  // enumerate degrees of freedom
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();

  if(needs_own_dof_handler_velocity())
  {
    dof_handler_velocity->distribute_dofs(*fe_velocity);
    dof_handler_velocity->distribute_mg_dofs();
  }

  unsigned int const ndofs_per_cell = Utilities::pow(degree + 1, dim);

  pcout << std::endl
        << "Discontinuous Galerkin finite element discretization:" << std::endl
        << std::endl;

  print_parameter(pcout, "degree of 1D polynomials", degree);
  print_parameter(pcout, "number of dofs per cell", ndofs_per_cell);
  print_parameter(pcout, "number of dofs (total)", dof_handler.n_dofs());
}

template<int dim, typename Number>
std::string
DGOperator<dim, Number>::get_dof_name() const
{
  return field + dof_index_std;
}

template<int dim, typename Number>
std::string
DGOperator<dim, Number>::get_quad_name() const
{
  return field + quad_index_std;
}

template<int dim, typename Number>
std::string
DGOperator<dim, Number>::get_quad_name_overintegration() const
{
  return field + quad_index_overintegration;
}

template<int dim, typename Number>
bool
DGOperator<dim, Number>::needs_own_dof_handler_velocity() const
{
  return param.analytical_velocity_field && param.store_analytical_velocity_in_dof_vector;
}

template<int dim, typename Number>
unsigned int
DGOperator<dim, Number>::get_dof_index() const
{
  return matrix_free_data->get_dof_index(get_dof_name());
}

template<int dim, typename Number>
std::string
DGOperator<dim, Number>::get_dof_name_velocity() const
{
  if(needs_own_dof_handler_velocity())
  {
    return field + dof_index_velocity;
  }
  else // external velocity field not hosted by the convection-diffusion operator
  {
    return dof_index_velocity_external;
  }
}

template<int dim, typename Number>
unsigned int
DGOperator<dim, Number>::get_dof_index_velocity() const
{
  if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
    return matrix_free_data->get_dof_index(get_dof_name_velocity());
  else
    return numbers::invalid_unsigned_int;
}

template<int dim, typename Number>
unsigned int
DGOperator<dim, Number>::get_quad_index() const
{
  return matrix_free_data->get_quad_index(field + quad_index_std);
}

template<int dim, typename Number>
unsigned int
DGOperator<dim, Number>::get_quad_index_overintegration() const
{
  return matrix_free_data->get_quad_index(field + quad_index_overintegration);
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::setup_solver(double const       scaling_factor_mass_matrix,
                                      VectorType const * velocity)
{
  pcout << std::endl << "Setup solver ..." << std::endl;

  if(param.linear_system_has_to_be_solved())
  {
    combined_operator.set_scaling_factor_mass_matrix(scaling_factor_mass_matrix);

    // The velocity vector needs to be set in case the velocity field is stored in DoF vector.
    // Otherwise, certain preconditioners requiring the velocity field during initialization can not
    // be initialized.
    if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
    {
      AssertThrow(
        velocity != nullptr,
        ExcMessage("In case of a numerical velocity field, a velocity vector has to be provided."));

      combined_operator.set_velocity_ptr(*velocity);
    }

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
    preconditioner.reset(new InverseMassMatrixPreconditioner<dim, 1, Number>(*matrix_free,
                                                                             get_dof_index(),
                                                                             get_quad_index()));
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

    MultigridData mg_data;
    mg_data = param.multigrid_data;

    typedef MultigridPreconditioner<dim, Number, MultigridNumber> MULTIGRID;

    preconditioner.reset(new MULTIGRID(this->mpi_comm));
    std::shared_ptr<MULTIGRID> mg_preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(preconditioner);

    parallel::TriangulationBase<dim> const * tria =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(&dof_handler.get_triangulation());

    const FiniteElement<dim> & fe = dof_handler.get_fe();

    if(param.mg_operator_type == MultigridOperatorType::ReactionConvection ||
       param.mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion)
    {
      if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
      {
        unsigned int const degree_scalar = fe.degree;
        unsigned int const degree_velocity =
          matrix_free_data->get_dof_handler(get_dof_name_velocity()).get_fe().degree;
        AssertThrow(
          degree_scalar == degree_velocity,
          ExcMessage(
            "When using a multigrid preconditioner for the scalar convection-diffusion equation, "
            "the scalar field and the velocity field must have the same polynomial degree."));
      }
    }

    OperatorData<dim> const & data = combined_operator.get_data();

    mg_preconditioner->initialize(mg_data,
                                  tria,
                                  fe,
                                  mapping,
                                  combined_operator,
                                  param.mg_operator_type,
                                  param.ale_formulation,
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
        combined_operator, *preconditioner, solver_data, mpi_comm));
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
  matrix_free->initialize_dof_vector(src, get_dof_index());
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::initialize_dof_vector_velocity(VectorType & velocity) const
{
  matrix_free->initialize_dof_vector(velocity, get_dof_index_velocity());
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

  VectorTools::interpolate(get_dof_handler_velocity(), *(field_functions->velocity), vector_double);

  velocity = vector_double;
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::project_velocity(VectorType & velocity, double const time) const
{
  VelocityProjection<dim, Number> l2_projection;

  l2_projection.apply(*matrix_free,
                      get_dof_index_velocity(),
                      get_quad_index(),
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
    if(param.diffusive_problem())
    {
      diffusive_operator.set_time(time);
      diffusive_operator.evaluate_add(dst, src);
    }

    // convective operator
    if(param.convective_problem())
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

    if(param.convective_problem())
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
  // no need to set scaling_factor_mass_matrix because the mass matrix does not contribute to rhs

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
  update_conv_diff_operator(time, scaling_factor, velocity);

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
  return calculate_time_step_cfl_local<dim, Number>(*matrix_free,
                                                    get_dof_index_velocity(),
                                                    get_quad_index(),
                                                    velocity,
                                                    cfl,
                                                    degree,
                                                    exponent_degree,
                                                    param.adaptive_time_stepping_cfl_type,
                                                    mpi_comm);
}

template<int dim, typename Number>
double
DGOperator<dim, Number>::calculate_time_step_cfl_analytical_velocity(
  double const time,
  double const cfl,
  double const exponent_degree) const
{
  return calculate_time_step_cfl_local<dim, Number>(*matrix_free,
                                                    get_dof_index(),
                                                    get_quad_index(),
                                                    field_functions->velocity,
                                                    time,
                                                    cfl,
                                                    degree,
                                                    exponent_degree,
                                                    param.adaptive_time_stepping_cfl_type,
                                                    mpi_comm);
}

template<int dim, typename Number>
double
DGOperator<dim, Number>::calculate_maximum_velocity(double const time) const
{
  return calculate_max_velocity(dof_handler.get_triangulation(),
                                field_functions->velocity,
                                time,
                                mpi_comm);
}

template<int dim, typename Number>
double
DGOperator<dim, Number>::calculate_minimum_element_length() const
{
  return calculate_minimum_vertex_distance(dof_handler.get_triangulation(), mpi_comm);
}

template<int dim, typename Number>
DoFHandler<dim> const &
DGOperator<dim, Number>::get_dof_handler() const
{
  return dof_handler;
}

template<int dim, typename Number>
DoFHandler<dim> const &
DGOperator<dim, Number>::get_dof_handler_velocity() const
{
  return matrix_free_data->get_dof_handler(get_dof_name_velocity());
}

template<int dim, typename Number>
unsigned int
DGOperator<dim, Number>::get_polynomial_degree() const
{
  return degree;
}

template<int dim, typename Number>
types::global_dof_index
DGOperator<dim, Number>::get_number_of_dofs() const
{
  return dof_handler.n_dofs();
}

template<int dim, typename Number>
void
DGOperator<dim, Number>::update_after_mesh_movement()
{
  // update SIPG penalty parameter of diffusive operator which depends on the deformation
  // of elements
  if(param.diffusive_problem())
  {
    diffusive_kernel->calculate_penalty_parameter(*matrix_free, get_dof_index());
  }
}

template class DGOperator<2, float>;
template class DGOperator<2, double>;

template class DGOperator<3, float>;
template class DGOperator<3, double>;

} // namespace ConvDiff
