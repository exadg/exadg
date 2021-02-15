/*
 * dg_coupled_solver.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include <exadg/incompressible_navier_stokes/preconditioners/multigrid_preconditioner_momentum.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_coupled.h>
#include <exadg/poisson/preconditioner/multigrid_preconditioner.h>
#include <exadg/poisson/spatial_discretization/laplace_operator.h>
#include <exadg/solvers_and_preconditioners/preconditioner/inverse_mass_preconditioner.h>
#include <exadg/solvers_and_preconditioners/util/check_multigrid.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim, typename Number>
OperatorCoupled<dim, Number>::OperatorCoupled(
  parallel::TriangulationBase<dim> const & triangulation_in,
  Mapping<dim> const &                     mapping_in,
  unsigned int const                       degree_u_in,
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> const
                                                  periodic_face_pairs_in,
  std::shared_ptr<BoundaryDescriptorU<dim>> const boundary_descriptor_velocity_in,
  std::shared_ptr<BoundaryDescriptorP<dim>> const boundary_descriptor_pressure_in,
  std::shared_ptr<FieldFunctions<dim>> const      field_functions_in,
  InputParameters const &                         parameters_in,
  std::string const &                             field_in,
  MPI_Comm const &                                mpi_comm_in)
  : Base(triangulation_in,
         mapping_in,
         degree_u_in,
         periodic_face_pairs_in,
         boundary_descriptor_velocity_in,
         boundary_descriptor_pressure_in,
         field_functions_in,
         parameters_in,
         field_in,
         mpi_comm_in),
    scaling_factor_continuity(1.0)
{
}

template<int dim, typename Number>
OperatorCoupled<dim, Number>::~OperatorCoupled()
{
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::setup(std::shared_ptr<MatrixFree<dim, Number>>     matrix_free,
                                    std::shared_ptr<MatrixFreeData<dim, Number>> matrix_free_data,
                                    std::string const & dof_index_temperature)
{
  Base::setup(matrix_free, matrix_free_data, dof_index_temperature);

  this->initialize_vector_velocity(temp_vector);
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::setup_solvers(double const &     scaling_factor_time_derivative_term,
                                            VectorType const & velocity)
{
  this->pcout << std::endl << "Setup incompressible Navier-Stokes solver ..." << std::endl;

  Base::setup_solvers(scaling_factor_time_derivative_term, velocity);

  initialize_block_preconditioner();

  initialize_solver_coupled();

  if(this->param.apply_penalty_terms_in_postprocessing_step)
    this->setup_projection_solver();

  this->pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::initialize_solver_coupled()
{
  linear_operator.initialize(*this);

  // setup linear solver
  if(this->param.solver_coupled == SolverCoupled::GMRES)
  {
    GMRESSolverData solver_data;
    solver_data.max_iter             = this->param.solver_data_coupled.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_coupled.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_coupled.rel_tol;
    solver_data.max_n_tmp_vectors    = this->param.solver_data_coupled.max_krylov_size;
    solver_data.compute_eigenvalues  = false;

    if(this->param.preconditioner_coupled != PreconditionerCoupled::None)
    {
      solver_data.use_preconditioner = true;
    }

    linear_solver.reset(
      new GMRESSolver<LinearOperatorCoupled<dim, Number>, Preconditioner, BlockVectorType>(
        linear_operator, block_preconditioner, solver_data, this->mpi_comm));
  }
  else if(this->param.solver_coupled == SolverCoupled::FGMRES)
  {
    FGMRESSolverData solver_data;
    solver_data.max_iter             = this->param.solver_data_coupled.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_coupled.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_coupled.rel_tol;
    solver_data.max_n_tmp_vectors    = this->param.solver_data_coupled.max_krylov_size;

    if(this->param.preconditioner_coupled != PreconditionerCoupled::None)
    {
      solver_data.use_preconditioner = true;
    }

    linear_solver.reset(
      new FGMRESSolver<LinearOperatorCoupled<dim, Number>, Preconditioner, BlockVectorType>(
        linear_operator, block_preconditioner, solver_data));
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified solver for linearized problem is not implemented."));
  }

  // setup Newton solver
  if(this->param.nonlinear_problem_has_to_be_solved())
  {
    nonlinear_operator.initialize(*this);

    newton_solver.reset(new Newton::Solver<BlockVectorType,
                                           NonlinearOperatorCoupled<dim, Number>,
                                           LinearOperatorCoupled<dim, Number>,
                                           IterativeSolverBase<BlockVectorType>>(
      this->param.newton_solver_data_coupled, nonlinear_operator, linear_operator, *linear_solver));
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::update_divergence_penalty_operator(VectorType const & velocity)
{
  this->div_penalty_operator.update(velocity);
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::update_continuity_penalty_operator(VectorType const & velocity)
{
  this->conti_penalty_operator.update(velocity);
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::initialize_block_vector_velocity_pressure(BlockVectorType & src) const
{
  // velocity(1st block) + pressure(2nd block)
  src.reinit(2);

  this->get_matrix_free().initialize_dof_vector(src.block(0), this->get_dof_index_velocity());
  this->get_matrix_free().initialize_dof_vector(src.block(1), this->get_dof_index_pressure());

  src.collect_sizes();
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::set_scaling_factor_continuity(double const scaling_factor)
{
  scaling_factor_continuity = scaling_factor;
  this->gradient_operator.set_scaling_factor_pressure(scaling_factor);
}

template<int dim, typename Number>
unsigned int
OperatorCoupled<dim, Number>::solve_linear_stokes_problem(BlockVectorType &       dst,
                                                          BlockVectorType const & src,
                                                          bool const &   update_preconditioner,
                                                          double const & time,
                                                          double const & scaling_factor_mass)
{
  // Update linear operator
  linear_operator.update(time, scaling_factor_mass);

  return linear_solver->solve(dst, src, update_preconditioner);
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::rhs_stokes_problem(BlockVectorType & dst, double const & time) const
{
  // velocity-block
  this->gradient_operator.rhs(dst.block(0), time);
  dst.block(0) *= scaling_factor_continuity;

  this->viscous_operator.set_time(time);
  this->viscous_operator.rhs_add(dst.block(0));

  if(this->param.apply_penalty_terms_in_postprocessing_step == false)
  {
    if(this->param.use_continuity_penalty == true)
      this->conti_penalty_operator.rhs_add(dst.block(0), time);
  }

  if(this->param.right_hand_side == true)
    this->rhs_operator.evaluate_add(dst.block(0), time);

  // pressure-block
  this->divergence_operator.rhs(dst.block(1), time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  // scale by scaling_factor_continuity
  dst.block(1) *= -scaling_factor_continuity;
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::apply_linearized_problem(BlockVectorType &       dst,
                                                       BlockVectorType const & src,
                                                       double const &          time,
                                                       double const & scaling_factor_mass) const
{
  // (1,1) block of saddle point matrix
  this->momentum_operator.set_time(time);
  this->momentum_operator.set_scaling_factor_mass_operator(scaling_factor_mass);
  this->momentum_operator.vmult(dst.block(0), src.block(0));

  // Divergence and continuity penalty operators
  if(this->param.apply_penalty_terms_in_postprocessing_step == false)
  {
    if(this->param.use_divergence_penalty == true)
      this->div_penalty_operator.apply_add(dst.block(0), src.block(0));
    if(this->param.use_continuity_penalty == true)
      this->conti_penalty_operator.apply_add(dst.block(0), src.block(0));
  }

  // (1,2) block of saddle point matrix
  // gradient operator: dst = velocity, src = pressure
  this->gradient_operator.apply(temp_vector, src.block(1));
  dst.block(0).add(scaling_factor_continuity, temp_vector);

  // (2,1) block of saddle point matrix
  // divergence operator: dst = pressure, src = velocity
  this->divergence_operator.apply(dst.block(1), src.block(0));
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  // scale by scaling_factor_continuity
  dst.block(1) *= -scaling_factor_continuity;
}

template<int dim, typename Number>
std::tuple<unsigned int, unsigned int>
OperatorCoupled<dim, Number>::solve_nonlinear_problem(BlockVectorType &  dst,
                                                      VectorType const & rhs_vector,
                                                      bool const &       update_preconditioner,
                                                      double const &     time,
                                                      double const &     scaling_factor_mass)
{
  // Update nonlinear operator
  nonlinear_operator.update(rhs_vector, time, scaling_factor_mass);

  // Update linear operator
  linear_operator.update(time, scaling_factor_mass);

  // Solve nonlinear problem
  Newton::UpdateData update;
  update.do_update             = update_preconditioner;
  update.threshold_newton_iter = this->param.update_preconditioner_coupled_every_newton_iter;

  auto const iter = newton_solver->solve(dst, update);

  return iter;
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::evaluate_nonlinear_residual(BlockVectorType &       dst,
                                                          BlockVectorType const & src,
                                                          VectorType const *      rhs_vector,
                                                          double const &          time,
                                                          double const & scaling_factor_mass) const
{
  // velocity-block

  if(this->unsteady_problem_has_to_be_solved())
    this->mass_operator.apply_scale(dst.block(0), scaling_factor_mass, src.block(0));
  else
    dst.block(0) = 0.0;

  AssertThrow(this->param.convective_problem() == true, ExcMessage("Invalid parameters."));

  this->convective_operator.evaluate_nonlinear_operator_add(dst.block(0), src.block(0), time);

  if(this->param.viscous_problem())
  {
    this->viscous_operator.set_time(time);
    this->viscous_operator.evaluate_add(dst.block(0), src.block(0));
  }

  // Divergence and continuity penalty operators
  if(this->param.apply_penalty_terms_in_postprocessing_step == false)
  {
    if(this->param.use_divergence_penalty == true)
      this->div_penalty_operator.apply_add(dst.block(0), src.block(0));
    if(this->param.use_continuity_penalty == true)
      this->conti_penalty_operator.evaluate_add(dst.block(0), src.block(0), time);
  }

  // gradient operator scaled by scaling_factor_continuity
  this->gradient_operator.evaluate(temp_vector, src.block(1), time);
  dst.block(0).add(scaling_factor_continuity, temp_vector);

  // constant right-hand side vector (body force vector and sum_alphai_ui term)
  dst.block(0).add(-1.0, *rhs_vector);

  // pressure-block

  this->divergence_operator.evaluate(dst.block(1), src.block(0), time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  // scale by scaling_factor_continuity
  dst.block(1) *= -scaling_factor_continuity;
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::evaluate_nonlinear_residual_steady(BlockVectorType &       dst,
                                                                 BlockVectorType const & src,
                                                                 double const &          time) const
{
  // velocity-block

  // set dst.block(0) to zero. This is necessary since subsequent operators
  // call functions of type ..._add
  dst.block(0) = 0.0;

  if(this->param.right_hand_side == true)
  {
    this->rhs_operator.evaluate(dst.block(0), time);
    // Shift body force term to the left-hand side of the equation.
    // This works since body_force_operator is the first operator
    // that is evaluated.
    dst.block(0) *= -1.0;
  }

  if(this->param.convective_problem())
  {
    this->convective_operator.evaluate_nonlinear_operator_add(dst.block(0), src.block(0), time);
  }

  if(this->param.viscous_problem())
  {
    this->viscous_operator.set_time(time);
    this->viscous_operator.evaluate_add(dst.block(0), src.block(0));
  }

  // Divergence and continuity penalty operators
  if(this->param.apply_penalty_terms_in_postprocessing_step == false)
  {
    if(this->param.use_divergence_penalty == true)
      this->div_penalty_operator.apply_add(dst.block(0), src.block(0));
    if(this->param.use_continuity_penalty == true)
      this->conti_penalty_operator.evaluate_add(dst.block(0), src.block(0), time);
  }

  // gradient operator scaled by scaling_factor_continuity
  this->gradient_operator.evaluate(temp_vector, src.block(1), time);
  dst.block(0).add(scaling_factor_continuity, temp_vector);


  // pressure-block

  this->divergence_operator.evaluate(dst.block(1), src.block(0), time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  // scale by scaling_factor_continuity
  dst.block(1) *= -scaling_factor_continuity;
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::initialize_block_preconditioner()
{
  block_preconditioner.initialize(this);

  initialize_vectors();

  initialize_preconditioner_velocity_block();

  initialize_preconditioner_pressure_block();
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::initialize_vectors()
{
  auto type = this->param.preconditioner_coupled;

  if(type == PreconditionerCoupled::BlockTriangular)
  {
    this->initialize_vector_velocity(vec_tmp_velocity);
  }
  else if(type == PreconditionerCoupled::BlockTriangularFactorization)
  {
    this->initialize_vector_pressure(vec_tmp_pressure);
    this->initialize_vector_velocity(vec_tmp_velocity);
    this->initialize_vector_velocity(vec_tmp_velocity_2);
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::initialize_preconditioner_velocity_block()
{
  auto type = this->param.preconditioner_velocity_block;

  if(type == MomentumPreconditioner::PointJacobi)
  {
    preconditioner_momentum.reset(
      new JacobiPreconditioner<MomentumOperator<dim, Number>>(this->momentum_operator));
  }
  else if(type == MomentumPreconditioner::BlockJacobi)
  {
    preconditioner_momentum.reset(
      new BlockJacobiPreconditioner<MomentumOperator<dim, Number>>(this->momentum_operator));
  }
  else if(type == MomentumPreconditioner::InverseMassMatrix)
  {
    preconditioner_momentum.reset(
      new InverseMassPreconditioner<dim, dim, Number>(this->get_matrix_free(),
                                                      this->get_dof_index_velocity(),
                                                      this->get_quad_index_velocity_linear()));
  }
  else if(type == MomentumPreconditioner::Multigrid)
  {
    setup_multigrid_preconditioner_momentum();

    if(this->param.exact_inversion_of_velocity_block == true)
    {
      setup_iterative_solver_momentum();
    }
  }
  else
  {
    AssertThrow(type == MomentumPreconditioner::None, ExcNotImplemented());
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::setup_multigrid_preconditioner_momentum()
{
  typedef MultigridPreconditioner<dim, Number> MULTIGRID;

  preconditioner_momentum.reset(new MULTIGRID(this->mpi_comm));

  std::shared_ptr<MULTIGRID> mg_preconditioner =
    std::dynamic_pointer_cast<MULTIGRID>(preconditioner_momentum);

  auto & dof_handler = this->get_dof_handler_u();

  parallel::TriangulationBase<dim> const * tria =
    dynamic_cast<parallel::TriangulationBase<dim> const *>(&dof_handler.get_triangulation());
  FiniteElement<dim> const & fe = dof_handler.get_fe();

  mg_preconditioner->initialize(this->param.multigrid_data_velocity_block,
                                tria,
                                fe,
                                this->get_mapping(),
                                this->momentum_operator,
                                this->param.multigrid_operator_type_velocity_block,
                                this->param.ale_formulation,
                                &this->momentum_operator.get_data().bc->dirichlet_bc,
                                &this->periodic_face_pairs);
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::setup_iterative_solver_momentum()
{
  AssertThrow(preconditioner_momentum.get() != 0,
              ExcMessage("preconditioner_momentum is uninitialized"));

  // use FMGRES for "exact" solution of velocity block system
  FGMRESSolverData gmres_data;
  gmres_data.use_preconditioner   = true;
  gmres_data.max_iter             = this->param.solver_data_velocity_block.max_iter;
  gmres_data.solver_tolerance_abs = this->param.solver_data_velocity_block.abs_tol;
  gmres_data.solver_tolerance_rel = this->param.solver_data_velocity_block.rel_tol;
  gmres_data.max_n_tmp_vectors    = this->param.solver_data_velocity_block.max_krylov_size;

  solver_velocity_block.reset(
    new FGMRESSolver<MomentumOperator<dim, Number>, PreconditionerBase<Number>, VectorType>(
      this->momentum_operator, *preconditioner_momentum, gmres_data));
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::initialize_preconditioner_pressure_block()
{
  auto type = this->param.preconditioner_pressure_block;

  if(type == SchurComplementPreconditioner::InverseMassMatrix)
  {
    inverse_mass_preconditioner_schur_complement.reset(
      new InverseMassPreconditioner<dim, 1, Number>(this->get_matrix_free(),
                                                    this->get_dof_index_pressure(),
                                                    this->get_quad_index_pressure()));
  }
  else if(type == SchurComplementPreconditioner::LaplaceOperator)
  {
    setup_multigrid_preconditioner_schur_complement();

    if(this->param.exact_inversion_of_laplace_operator == true)
    {
      setup_iterative_solver_schur_complement();
    }
  }
  else if(type == SchurComplementPreconditioner::CahouetChabard)
  {
    AssertThrow(this->unsteady_problem_has_to_be_solved() == true,
                ExcMessage(
                  "Cahouet-Chabard preconditioner only makes sense for unsteady problems."));

    setup_multigrid_preconditioner_schur_complement();

    if(this->param.exact_inversion_of_laplace_operator == true)
    {
      setup_iterative_solver_schur_complement();
    }

    // inverse mass operator to also include the part of the preconditioner that is beneficial when
    // using large time steps and large viscosities.
    inverse_mass_preconditioner_schur_complement.reset(
      new InverseMassPreconditioner<dim, 1, Number>(this->get_matrix_free(),
                                                    this->get_dof_index_pressure(),
                                                    this->get_quad_index_pressure()));

    // initialize tmp vector
    this->initialize_vector_pressure(tmp_scp_pressure);
  }
  else if(type == SchurComplementPreconditioner::PressureConvectionDiffusion)
  {
    // -S^{-1} = M_p^{-1} A_p (-L)^{-1}

    // I. multigrid for negative Laplace operator (classical or compatible discretization)
    setup_multigrid_preconditioner_schur_complement();

    if(this->param.exact_inversion_of_laplace_operator == true)
    {
      setup_iterative_solver_schur_complement();
    }

    // II. pressure convection-diffusion operator
    setup_pressure_convection_diffusion_operator();

    // III. inverse pressure mass operator
    inverse_mass_preconditioner_schur_complement.reset(
      new InverseMassPreconditioner<dim, 1, Number>(this->get_matrix_free(),
                                                    this->get_dof_index_pressure(),
                                                    this->get_quad_index_pressure()));

    // initialize tmp vector
    this->initialize_vector_pressure(tmp_scp_pressure);
  }
  else
  {
    AssertThrow(type == SchurComplementPreconditioner::None, ExcNotImplemented());
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::setup_multigrid_preconditioner_schur_complement()
{
  // multigrid V-cycle for negative Laplace operator
  Poisson::LaplaceOperatorData<0, dim> laplace_operator_data;
  laplace_operator_data.dof_index             = this->get_dof_index_pressure();
  laplace_operator_data.quad_index            = this->get_quad_index_pressure();
  laplace_operator_data.operator_is_singular  = this->is_pressure_level_undefined();
  laplace_operator_data.kernel_data.IP_factor = 1.0;
  laplace_operator_data.bc                    = this->boundary_descriptor_laplace;

  MultigridData mg_data = this->param.multigrid_data_pressure_block;

  multigrid_preconditioner_schur_complement.reset(new MultigridPoisson(this->mpi_comm));

  std::shared_ptr<MultigridPoisson> mg_preconditioner =
    std::dynamic_pointer_cast<MultigridPoisson>(multigrid_preconditioner_schur_complement);

  auto & dof_handler = this->get_dof_handler_p();

  parallel::TriangulationBase<dim> const * tria =
    dynamic_cast<const parallel::TriangulationBase<dim> *>(&dof_handler.get_triangulation());
  const FiniteElement<dim> & fe = dof_handler.get_fe();

  mg_preconditioner->initialize(mg_data,
                                tria,
                                fe,
                                this->get_mapping(),
                                laplace_operator_data,
                                this->param.ale_formulation,
                                &laplace_operator_data.bc->dirichlet_bc,
                                &this->periodic_face_pairs);
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::setup_iterative_solver_schur_complement()
{
  AssertThrow(
    multigrid_preconditioner_schur_complement.get() != 0,
    ExcMessage(
      "Setup of iterative solver for Schur complement preconditioner: Multigrid preconditioner is uninitialized"));

  CGSolverData solver_data;
  solver_data.max_iter             = this->param.solver_data_pressure_block.max_iter;
  solver_data.solver_tolerance_abs = this->param.solver_data_pressure_block.abs_tol;
  solver_data.solver_tolerance_rel = this->param.solver_data_pressure_block.rel_tol;
  solver_data.use_preconditioner   = true;

  Poisson::LaplaceOperatorData<0, dim> laplace_operator_data;
  laplace_operator_data.dof_index             = this->get_dof_index_pressure();
  laplace_operator_data.quad_index            = this->get_quad_index_pressure();
  laplace_operator_data.bc                    = this->boundary_descriptor_laplace;
  laplace_operator_data.kernel_data.IP_factor = 1.0;

  laplace_operator.reset(new Poisson::LaplaceOperator<dim, Number, 1>());
  laplace_operator->initialize(this->get_matrix_free(),
                               this->get_constraint_p(),
                               laplace_operator_data);

  solver_pressure_block.reset(
    new CGSolver<Poisson::LaplaceOperator<dim, Number, 1>, PreconditionerBase<Number>, VectorType>(
      *laplace_operator, *multigrid_preconditioner_schur_complement, solver_data));
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::setup_pressure_convection_diffusion_operator()
{
  // pressure convection-diffusion operator

  // fill boundary descriptor
  std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> boundary_descriptor;
  boundary_descriptor.reset(new ConvDiff::BoundaryDescriptor<dim>());

  // For the pressure convection-diffusion operator the homogeneous operators are applied, so there
  // is no need to specify functions for boundary conditions since they will never be used.
  std::shared_ptr<Function<dim>> dummy;

  // set boundary ID's for pressure convection-diffusion operator

  // Dirichlet BC for pressure
  for(typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::const_iterator it =
        this->boundary_descriptor_pressure->dirichlet_bc.begin();
      it != this->boundary_descriptor_pressure->dirichlet_bc.end();
      ++it)
  {
    boundary_descriptor->dirichlet_bc.insert(
      std::pair<types::boundary_id, std::shared_ptr<Function<dim>>>(it->first, dummy));
  }
  // Neumann BC for pressure
  for(typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::const_iterator it =
        this->boundary_descriptor_pressure->neumann_bc.begin();
      it != this->boundary_descriptor_pressure->neumann_bc.end();
      ++it)
  {
    boundary_descriptor->neumann_bc.insert(
      std::pair<types::boundary_id, std::shared_ptr<Function<dim>>>(it->first, dummy));
  }

  // convective operator:
  // use numerical velocity field with dof index of velocity field and local Lax-Friedrichs flux to
  // mimic the upwind-like discretization of the linearized convective term in the Navier-Stokes
  // equations.
  ConvDiff::Operators::ConvectiveKernelData<dim> convective_kernel_data;
  convective_kernel_data.velocity_type      = ConvDiff::TypeVelocityField::DoFVector;
  convective_kernel_data.dof_index_velocity = this->get_dof_index_velocity();
  convective_kernel_data.numerical_flux_formulation =
    ConvDiff::NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

  // diffusive operator:
  // take interior penalty factor of diffusivity of viscous operator, but use polynomial degree of
  // pressure shape functions.
  ConvDiff::Operators::DiffusiveKernelData diffusive_kernel_data;
  diffusive_kernel_data.IP_factor = this->param.IP_factor_viscous;
  // Note: the diffusive operator is initialized with constant viscosity. In case of spatially (and
  // temporally) varying viscosities the diffusive operator has to be extended so that it can deal
  // with variable coefficients (and should be updated in case of time dependent problems before
  // applying the preconditioner).
  diffusive_kernel_data.diffusivity = this->param.viscosity;

  // combined convection-diffusion operator
  ConvDiff::CombinedOperatorData<dim> operator_data;
  operator_data.dof_index  = this->get_dof_index_pressure();
  operator_data.quad_index = this->get_quad_index_pressure();

  operator_data.bc                   = boundary_descriptor;
  operator_data.use_cell_based_loops = this->param.use_cell_based_face_loops;

  operator_data.unsteady_problem   = this->unsteady_problem_has_to_be_solved();
  operator_data.convective_problem = this->param.nonlinear_problem_has_to_be_solved();
  operator_data.diffusive_problem  = this->param.viscous_problem();

  operator_data.convective_kernel_data = convective_kernel_data;
  operator_data.diffusive_kernel_data  = diffusive_kernel_data;

  pressure_conv_diff_operator.reset(new ConvDiff::CombinedOperator<dim, Number>());
  pressure_conv_diff_operator->initialize(this->get_matrix_free(),
                                          this->get_constraint_p(),
                                          operator_data);
}

// clang-format off
/*
 * Consider the following saddle point matrix :
 *
 *       / A  B^{T} \
 *   M = |          |
 *       \ B    0   /
 *
 *  with block factorization
 *
 *       / I         0 \  / A   0 \ / I   A^{-1} B^{T} \
 *   M = |             |  |       | |                  |
 *       \ B A^{-1}  I /  \ 0   S / \ 0        I       /
 *
 *       / I         0 \  / A   B^{T} \
 *     = |             |  |           |
 *       \ B A^{-1}  I /  \ 0     S   /
 *
 *        / A  0 \  / I   A^{-1} B^{T} \
 *     =  |      |  |                  |
 *        \ B  S /  \ 0        I       /
 *
 *   with Schur complement S = -B A^{-1} B^{T}
 *
 *
 * - Block-diagonal preconditioner:
 *
 *                   / A   0 \                       / A^{-1}   0    \   / A^{-1}  0 \   / I      0    \
 *   -> P_diagonal = |       |  -> P_diagonal^{-1} = |               | = |           | * |             |
 *                   \ 0  -S /                       \   0   -S^{-1} /   \   0     I /   \ 0   -S^{-1} /
 *
 * - Block-triangular preconditioner:
 *
 *                     / A   B^{T} \                         / A^{-1}  0 \   / I  B^{T} \   / I      0    \
 *   -> P_triangular = |           |  -> P_triangular^{-1} = |           | * |          | * |             |
 *                     \ 0     S   /                         \   0     I /   \ 0   -I   /   \ 0   -S^{-1} /
 *
 * - Block-triangular factorization:
 *
 *                      / A  0 \  / I   A^{-1} B^{T} \
 *   -> P_tria-factor = |      |  |                  |
 *                      \ B  S /  \ 0        I       /
 *
 *                            / I  - A^{-1} B^{T} \   / I      0    \   / I   0 \   / A^{-1}  0 \
 *    -> P_tria-factor^{-1} = |                   | * |             | * |       | * |           |
 *                            \ 0          I      /   \ 0   -S^{-1} /   \ B  -I /   \   0     I /
 *
 *
 *  Main challenge: Development of efficient preconditioners for A and S that approximate
 *  the velocity block A and the Schur-complement block S in a spectrally equivalent way.
 *
 *
 *  Approximations of velocity block A = 1/dt M_u + C_lin(u) + nu (-L):
 *
 *   1. inverse mass preconditioner (dt small):
 *
 *     A = 1/dt M_u
 *
 *     -> A^{-1} = dt M_u^{-1}
 *
 *   2. Helmholtz operator H =  1/dt M_u + nu (-L) (neglecting the convective term):
 *
 *     -> A^{-1} = H^{-1} where H^{-1} is approximated by performing one multigrid V-cycle for the Helmholtz operator
 *
 *   3. Velocity convection-diffusion operator A = 1/dt M_u + C_lin(u) + nu (-L) including the convective term:
 *
 *      -> to approximately invert A consider one multigrid V-cycle
 *
 *  Approximations of pressure Schur-complement block S:
 *
 *   - S = - B A^{-1} B^T
 *       |
 *       |  apply method of pseudo-differential operators and neglect convective term
 *      \|/
 *       = - (- div ) * ( 1/dt * I - nu * laplace )^{-1} * grad
 *
 *   1. dt small, nu small:

 *      S = div * (1/dt I)^{-1} * grad = dt * laplace
 *
 *      -> - S^{-1} = 1/dt (-L)^{-1} (-L: negative Laplace operator)
 *
 *   2. dt large, nu large:
 *
 *      S = div * (- nu * laplace)^{-1} * grad = - 1/nu * I
 *
 *      -> - S^{-1} = nu M_p^{-1} (M_p: pressure mass operator)
 *
 *   3. Cahouet & Chabard (combines 1. and 2., robust preconditioner for whole range of time step sizes and visosities)
 *
 *      -> - S^{-1} = nu M_p^{-1} + 1/dt (-L)^{-1}
 *
 *   4. Pressure convection-diffusion preconditioner
 *
 *      -> -S^{-1} = M_p^{-1} A_p (-L)^{-1} where A_p is a convection-diffusion operator for the pressure
 */
// clang-format on

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::update_block_preconditioner()
{
  // momentum block
  preconditioner_momentum->update();

  // pressure block
  if(this->param.ale_formulation) // only if mesh is moving
  {
    auto const type = this->param.preconditioner_pressure_block;

    // inverse mass operator
    if(type == SchurComplementPreconditioner::InverseMassMatrix ||
       type == SchurComplementPreconditioner::CahouetChabard ||
       type == SchurComplementPreconditioner::PressureConvectionDiffusion)
    {
      inverse_mass_preconditioner_schur_complement->update();
    }

    // Laplace operator
    if(type == SchurComplementPreconditioner::LaplaceOperator ||
       type == SchurComplementPreconditioner::CahouetChabard ||
       type == SchurComplementPreconditioner::PressureConvectionDiffusion)
    {
      if(this->param.exact_inversion_of_laplace_operator == true)
      {
        laplace_operator->update_after_mesh_movement();
      }

      multigrid_preconditioner_schur_complement->update();
    }
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::apply_block_preconditioner(BlockVectorType &       dst,
                                                         BlockVectorType const & src) const
{
  auto type = this->param.preconditioner_coupled;

  if(type == PreconditionerCoupled::BlockDiagonal)
  {
    /*                        / A^{-1}   0    \   / A^{-1}  0 \   / I      0    \
     *   -> P_diagonal^{-1} = |               | = |           | * |             |
     *                        \   0   -S^{-1} /   \   0     I /   \ 0   -S^{-1} /
     */

    /*
     *         / I      0    \
     *  temp = |             | * src
     *         \ 0   -S^{-1} /
     */

    // apply preconditioner for pressure/Schur-complement block
    apply_preconditioner_pressure_block(dst.block(1), src.block(1));

    /*
     *        / A^{-1}  0 \
     *  dst = |           | * temp
     *        \   0     I /
     */

    // apply preconditioner for velocity/momentum block
    apply_preconditioner_velocity_block(dst.block(0), src.block(0));
  }
  else if(type == PreconditionerCoupled::BlockTriangular)
  {
    /*
     *                         / A^{-1}  0 \   / I  B^{T} \   / I      0    \
     *  -> P_triangular^{-1} = |           | * |          | * |             |
     *                         \   0     I /   \ 0   -I   /   \ 0   -S^{-1} /
     */

    /*
     *        / I      0    \
     *  dst = |             | * src
     *        \ 0   -S^{-1} /
     */

    // For the velocity block simply copy data from src to dst.
    dst.block(0) = src.block(0);
    // Apply preconditioner for pressure/Schur-complement block.
    apply_preconditioner_pressure_block(dst.block(1), src.block(1));

    /*
     *        / I  B^{T} \
     *  dst = |          | * dst
     *        \ 0   -I   /
     */

    // Apply gradient operator and add to dst vector.
    this->gradient_operator.apply(vec_tmp_velocity, dst.block(1));
    dst.block(0).add(this->scaling_factor_continuity, vec_tmp_velocity);
    dst.block(1) *= -1.0;

    /*
     *        / A^{-1}  0 \
     *  dst = |           | * dst
     *        \   0     I /
     */

    // Copy data from dst.block(0) to vec_tmp_velocity before
    // applying the preconditioner for the velocity block.
    vec_tmp_velocity = dst.block(0);
    // Apply preconditioner for velocity/momentum block.
    apply_preconditioner_velocity_block(dst.block(0), vec_tmp_velocity);
  }
  else if(type == PreconditionerCoupled::BlockTriangularFactorization)
  {
    /*
     *                          / I  - A^{-1} B^{T} \   / I      0    \   / I   0 \   / A^{-1} 0 \
     *  -> P_tria-factor^{-1} = |                   | * |             | * |       | * |          |
     *                          \ 0          I      /   \ 0   -S^{-1} /   \ B  -I /   \   0    I /
     */

    /*
     *        / A^{-1}  0 \
     *  dst = |           | * src
     *        \   0     I /
     */

    // for the pressure block simply copy data from src to dst
    dst.block(1) = src.block(1);
    // apply preconditioner for velocity/momentum block
    apply_preconditioner_velocity_block(dst.block(0), src.block(0));

    /*
     *        / I   0 \
     *  dst = |       | * dst
     *        \ B  -I /
     */

    // dst.block(1) = B*dst.block(0) - dst.block(1)
    //              = -1.0 * (dst.block(1) + (-B) * dst.block(0));
    // I. dst.block(1) += (-B) * dst.block(0);
    // Note that B represents NEGATIVE divergence operator, i.e.,
    // applying -B is equal to applying the divergence operator
    this->divergence_operator.apply(vec_tmp_pressure, dst.block(0));
    dst.block(1).add(this->scaling_factor_continuity, vec_tmp_pressure);
    // II. dst.block(1) = -dst.block(1);
    dst.block(1) *= -1.0;

    /*
     *        / I      0    \
     *  dst = |             | * dst
     *        \ 0   -S^{-1} /
     */

    // Copy data from dst.block(1) to vec_tmp_pressure before
    // applying the preconditioner for the pressure block.
    vec_tmp_pressure = dst.block(1);
    // Apply preconditioner for pressure/Schur-complement block
    apply_preconditioner_pressure_block(dst.block(1), vec_tmp_pressure);

    /*
     *        / I  - A^{-1} B^{T} \
     *  dst = |                   | * dst
     *        \ 0          I      /
     */

    // vec_tmp_velocity = B^{T} * dst(1)
    this->gradient_operator.apply(vec_tmp_velocity, dst.block(1));

    // scaling factor continuity
    vec_tmp_velocity *= this->scaling_factor_continuity;

    // vec_tmp_velocity_2 = A^{-1} * vec_tmp_velocity
    apply_preconditioner_velocity_block(vec_tmp_velocity_2, vec_tmp_velocity);

    // dst(0) = dst(0) - vec_tmp_velocity_2
    dst.block(0).add(-1.0, vec_tmp_velocity_2);
  }
  else
  {
    AssertThrow(false, ExcNotImplemented());
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::apply_preconditioner_velocity_block(VectorType &       dst,
                                                                  VectorType const & src) const
{
  auto type = this->param.preconditioner_velocity_block;

  if(type == MomentumPreconditioner::None)
  {
    dst = src;
  }
  else if(type == MomentumPreconditioner::PointJacobi ||
          type == MomentumPreconditioner::BlockJacobi)
  {
    preconditioner_momentum->vmult(dst, src);
  }
  else if(type == MomentumPreconditioner::InverseMassMatrix)
  {
    // use the inverse mass operator as an approximation to the momentum block
    preconditioner_momentum->vmult(dst, src);
    dst *= 1. / this->momentum_operator.get_scaling_factor_mass_operator();
  }
  else if(type == MomentumPreconditioner::Multigrid)
  {
    if(this->param.exact_inversion_of_velocity_block == false)
    {
      // perform one multigrid V-cylce
      preconditioner_momentum->vmult(dst, src);
    }
    else // exact_inversion_of_velocity_block == true
    {
      // check correctness of multigrid V-cycle

      // clang-format off
      /*
      typedef MultigridPreconditioner<dim, degree_u, Number, MultigridNumber> MULTIGRID;

      std::shared_ptr<MULTIGRID> preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(preconditioner_momentum);

      CheckMultigrid<dim, Number, MomentumOperator<dim, degree_u, Number>, MULTIGRID>
        check_multigrid(this->momentum_operator,preconditioner);

      check_multigrid.check();
      */
      // clang-format on

      // iteratively solve momentum equation up to given tolerance
      dst = 0.0;
      // Note that update of preconditioner is set to false here since the preconditioner has
      // already been updated in the member function update() if desired.
      unsigned int const iterations =
        solver_velocity_block->solve(dst, src, /* update_preconditioner = */ false);

      // output
      bool const print_iterations = false;
      if(print_iterations)
      {
        ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(this->mpi_comm) == 0);
        pcout << "Number of iterations for inner solver = " << iterations << std::endl;
      }
    }
  }
  else
  {
    AssertThrow(false, ExcNotImplemented());
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::apply_preconditioner_pressure_block(VectorType &       dst,
                                                                  VectorType const & src) const
{
  auto type = this->param.preconditioner_pressure_block;

  if(type == SchurComplementPreconditioner::None)
  {
    // No preconditioner for Schur-complement block
    dst = src;
  }
  else if(type == SchurComplementPreconditioner::InverseMassMatrix)
  {
    // - S^{-1} = nu M_p^{-1}
    inverse_mass_preconditioner_schur_complement->vmult(dst, src);
    dst *= this->get_viscosity();
  }
  else if(type == SchurComplementPreconditioner::LaplaceOperator)
  {
    // -S^{-1} = 1/dt  (-L)^{-1}
    apply_inverse_negative_laplace_operator(dst, src);
    dst *= this->momentum_operator.get_scaling_factor_mass_operator();
  }
  else if(type == SchurComplementPreconditioner::CahouetChabard)
  {
    // - S^{-1} = nu M_p^{-1} + 1/dt (-L)^{-1}

    // I. 1/dt (-L)^{-1}
    apply_inverse_negative_laplace_operator(dst, src);
    dst *= this->momentum_operator.get_scaling_factor_mass_operator();

    // II. M_p^{-1}, apply inverse pressure mass operator to src-vector and store the result in a
    // temporary vector
    inverse_mass_preconditioner_schur_complement->vmult(tmp_scp_pressure, src);

    // III. add temporary vector scaled by viscosity
    dst.add(this->get_viscosity(), tmp_scp_pressure);
  }
  else if(type == SchurComplementPreconditioner::PressureConvectionDiffusion)
  {
    // -S^{-1} = M_p^{-1} A_p (-L)^{-1}

    // I. inverse, negative Laplace operator (-L)^{-1}
    apply_inverse_negative_laplace_operator(tmp_scp_pressure, src);

    // II. pressure convection-diffusion operator A_p
    if(this->unsteady_problem_has_to_be_solved())
    {
      pressure_conv_diff_operator->set_scaling_factor_mass_operator(
        this->momentum_operator.get_scaling_factor_mass_operator());
    }

    if(this->param.nonlinear_problem_has_to_be_solved())
      pressure_conv_diff_operator->set_velocity_ptr(this->convective_kernel->get_velocity());

    pressure_conv_diff_operator->apply(dst, tmp_scp_pressure);

    // III. inverse pressure mass operator M_p^{-1}
    inverse_mass_preconditioner_schur_complement->vmult(dst, dst);
  }
  else
  {
    AssertThrow(false, ExcNotImplemented());
  }

  // scaling_factor_continuity: Since the Schur complement includes both the velocity divergence
  // and the pressure gradient operators as factors, we have to scale by
  // 1/(scaling_factor*scaling_factor) when applying (an approximation of) the inverse Schur
  // complement.
  double inverse_scaling_factor = 1.0 / scaling_factor_continuity;
  dst *= inverse_scaling_factor * inverse_scaling_factor;
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::apply_inverse_negative_laplace_operator(VectorType &       dst,
                                                                      VectorType const & src) const
{
  if(this->param.exact_inversion_of_laplace_operator == false)
  {
    // perform one multigrid V-cycle in order to approximately invert the negative Laplace
    // operator (classical or compatible)
    multigrid_preconditioner_schur_complement->vmult(dst, src);
  }
  else // exact_inversion_of_laplace_operator == true
  {
    // solve a linear system of equations for negative Laplace operator to given (relative)
    // tolerance using the PCG method
    VectorType const * pointer_to_src = &src;
    if(this->is_pressure_level_undefined())
    {
      VectorType vector_zero_mean;
      vector_zero_mean = src;

      if(laplace_operator->operator_is_singular())
      {
        set_zero_mean_value(vector_zero_mean);
      }

      pointer_to_src = &vector_zero_mean;
    }

    dst = 0.0;
    // Note that update of preconditioner is set to false here since the preconditioner has
    // already been updated in the function update_block_preconditioner().
    solver_pressure_block->solve(dst, *pointer_to_src, /* update_preconditioner = */ false);
  }
}


template class OperatorCoupled<2, float>;
template class OperatorCoupled<2, double>;

template class OperatorCoupled<3, float>;
template class OperatorCoupled<3, double>;

} // namespace IncNS
} // namespace ExaDG
