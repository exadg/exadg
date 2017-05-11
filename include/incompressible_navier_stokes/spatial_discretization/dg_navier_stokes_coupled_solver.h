/*
 * DGNavierStokesCoupled.h
 *
 *  Created on: Jun 27, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_COUPLED_SOLVER_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_COUPLED_SOLVER_H_

#include "../../incompressible_navier_stokes/preconditioners/preconditioner_navier_stokes.h"
#include "../../incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_base.h"
#include "../../incompressible_navier_stokes/spatial_discretization/projection_operators_and_solvers.h"
#include "solvers_and_preconditioners/newton_solver.h"


template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
class DGNavierStokesCoupled : public DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>
{
public:
  typedef DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> BASE;

  typedef DGNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> THIS;

  DGNavierStokesCoupled(parallel::distributed::Triangulation<dim> const &triangulation,
                        InputParametersNavierStokes<dim> const          &parameter)
    :
    BASE(triangulation,parameter),
    sum_alphai_ui(nullptr),
    vector_linearization(nullptr),
    evaluation_time(0.0),
    scaling_factor_time_derivative_term(1.0),
    scaling_factor_continuity(1.0)
  {

  }

  virtual ~DGNavierStokesCoupled(){};

  virtual void setup (const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> >
                                                                            periodic_face_pairs,
                      std::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_velocity,
                      std::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_pressure,
                      std::shared_ptr<FieldFunctionsNavierStokes<dim> >     field_functions);

  void setup_solvers(double const &scaling_factor_time_derivative_term = 1.0);

  void setup_velocity_conv_diff_operator(double const &scaling_factor_time_derivative_term = 1.0);

  void setup_divergence_and_continuity_penalty_operators_and_solvers();

  // initialization of vectors
  void initialize_block_vector_velocity_pressure(parallel::distributed::BlockVector<Number> &src) const
  {
    // velocity(1st block) + pressure(2nd block)
    src.reinit(2);

    this->data.initialize_dof_vector(src.block(0), this->get_dof_index_velocity());
    this->data.initialize_dof_vector(src.block(1), this->get_dof_index_pressure());

    src.collect_sizes();
  }

  void initialize_vector_for_newton_solver(parallel::distributed::BlockVector<Number> &src) const
  {
    initialize_block_vector_velocity_pressure(src);
  }

  bool nonlinear_problem_has_to_be_solved() const
  {
    return ( this->param.equation_type == EquationType::NavierStokes &&
             (this->param.problem_type == ProblemType::Steady ||
              this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit) );
  }

  bool unsteady_problem_has_to_be_solved() const
  {
    return ( this->param.problem_type == ProblemType::Unsteady );
  }

  /*
   *  This function sets the variable scaling_factor_continuity,
   *  and also the related scaling factor for the pressure gradient operator.
   */
  void set_scaling_factor_continuity(double const scaling_factor)
  {
    scaling_factor_continuity = scaling_factor;
    this->gradient_operator.set_scaling_factor_pressure(scaling_factor);
  }

  // TODO
  /*
   *  Update divergence penalty operator by recalculating the penalty parameter
   *  which depends on the current velocity field
   */
//  void update_divergence_penalty_operator (parallel::distributed::Vector<Number> const &velocity) const;

  // TODO
  /*
   *  Update continuity penalty operator by recalculating the penalty parameter
   *  which depends on the current velocity field
   */
//  void update_continuity_penalty_operator (parallel::distributed::Vector<Number> const &velocity) const;


  /*
   *  This function solves the linear Stokes problem (steady/unsteady Stokes or unsteady
   *  Navier-Stokes with explicit treatment of convective term).
   *  The parameter scaling_factor_mass_matrix_term has to be specified for unsteady problem.
   *  For steady problems this parameter is omitted.
   */
  unsigned int solve_linear_stokes_problem (parallel::distributed::BlockVector<Number>       &dst,
                                            parallel::distributed::BlockVector<Number> const &src,
                                            double const                                     &scaling_factor_mass_matrix_term = 1.0);


  /*
   *  For the linear solver, the operator of the linear(ized) problem has to
   *  implement a function called vmult().
   */
  void vmult (parallel::distributed::BlockVector<Number> &dst,
              parallel::distributed::BlockVector<Number> const &src) const;

  /*
   *  This function calculates the matrix vector product for the linear(ized) problem.
   */
  void apply_linearized_problem (parallel::distributed::BlockVector<Number> &dst,
                                 parallel::distributed::BlockVector<Number> const &src) const;

  /*
   *  This function calculates the rhs of the steady Stokes problem, or unsteady Stokes problem,
   *  or unsteady Navier-Stokes problem with explicit treatment of the convective term.
   *  The parameters 'src' and 'eval_time' have to be specified for unsteady problems.
   *  For steady problems these parameters are omitted.
   */
  void rhs_stokes_problem (parallel::distributed::BlockVector<Number>  &dst,
                           parallel::distributed::Vector<Number> const *src = nullptr,
                           double const                                &eval_time = 0.0) const;


  /*
   *  This function solves the nonlinear problem for steady problems.
   */
  void solve_nonlinear_steady_problem (parallel::distributed::BlockVector<Number>  &dst,
                                       unsigned int                                &newton_iterations,
                                       unsigned int                                &linear_iterations);

  /*
   *  This function solves the nonlinear problem for unsteady problems.
   */
  void solve_nonlinear_problem (parallel::distributed::BlockVector<Number>  &dst,
                                parallel::distributed::Vector<Number> const &sum_alphai_ui,
                                double const                                &eval_time,
                                double const                                &scaling_factor_mass_matrix_term,
                                unsigned int                                &newton_iterations,
                                unsigned int                                &linear_iterations);

  /*
   *  This function evaluates the nonlinear residual.
   */
  void evaluate_nonlinear_residual (parallel::distributed::BlockVector<Number>       &dst,
                                    parallel::distributed::BlockVector<Number> const &src);


  void set_solution_linearization(parallel::distributed::BlockVector<Number> const &solution_linearization)
  {
    velocity_conv_diff_operator.set_solution_linearization(solution_linearization.block(0));
  }

  parallel::distributed::Vector<Number> const &get_velocity_linearization() const
  {
    AssertThrow(nonlinear_problem_has_to_be_solved() == true,
        ExcMessage("Attempt to access velocity_linearization which has not been initialized."));

    return velocity_conv_diff_operator.get_solution_linearization();
  }

  void set_sum_alphai_ui(parallel::distributed::Vector<Number> const *vector = nullptr)
  {
    this->sum_alphai_ui = vector;
  }

  CompatibleLaplaceOperatorData<dim> const get_compatible_laplace_operator_data() const
  {
    CompatibleLaplaceOperatorData<dim> comp_laplace_operator_data;
    comp_laplace_operator_data.dof_index_velocity = this->get_dof_index_velocity();
    comp_laplace_operator_data.dof_index_pressure = this->get_dof_index_pressure();
    return comp_laplace_operator_data;
  }

  /*
   *  Perform projection based on divergence and continuity penalty terms in a
   *  postprocessing step after each time step.
   */
  void update_projection_operator(parallel::distributed::Vector<Number> const &velocity,
                                  double const                                time_step_size) const;

  unsigned int solve_projection (parallel::distributed::Vector<Number>       &dst,
                                 parallel::distributed::Vector<Number> const &src) const;

  void rhs_projection_add (parallel::distributed::Vector<Number> &dst,
                           double const                          time) const;

private:
  friend class BlockPreconditionerNavierStokes<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number, THIS >;

  VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> velocity_conv_diff_operator;

  // div-div-penalty and continuity penalty operator
  std::shared_ptr<DivergencePenaltyOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> > divergence_penalty_operator;
  std::shared_ptr<ContinuityPenaltyOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> > continuity_penalty_operator;

  // projection operator
  std::shared_ptr<ProjectionOperatorBase<dim> > projection_operator;

  // projection solver
  std::shared_ptr<IterativeSolverBase<parallel::distributed::Vector<Number> > > projection_solver;
  std::shared_ptr<PreconditionerBase<Number> > preconditioner_projection;

  parallel::distributed::Vector<Number> mutable temp_vector;
  parallel::distributed::Vector<Number> const *sum_alphai_ui;
  parallel::distributed::BlockVector<Number> const *vector_linearization;

  std::shared_ptr<PreconditionerNavierStokesBase<Number,THIS > > preconditioner;

  std::shared_ptr<IterativeSolverBase<parallel::distributed::BlockVector<Number> > > linear_solver;

  std::shared_ptr<NewtonSolver<parallel::distributed::BlockVector<Number>, THIS, THIS,
                               IterativeSolverBase<parallel::distributed::BlockVector<Number> > > > newton_solver;

  double evaluation_time;
  double scaling_factor_time_derivative_term;

  // scaling_factor_continuity
  double scaling_factor_continuity;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
setup (const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> >
                                                             periodic_face_pairs,
       std::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_velocity_in,
       std::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_pressure_in,
       std::shared_ptr<FieldFunctionsNavierStokes<dim> >     field_functions_in)
{
  BASE::setup(periodic_face_pairs,
              boundary_descriptor_velocity_in,
              boundary_descriptor_pressure_in,
              field_functions_in);

  this->initialize_vector_velocity(temp_vector);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
setup_solvers(double const &scaling_factor_time_derivative_term)
{
  // Setup velocity convection-diffusion operator.
  // This is done in function setup_solvers() since velocity convection-diffusion
  // operator data needs scaling_factor_time_derivative_term as input parameter.

  // Note that the velocity_conv_diff_operator has to be initialized
  // before calling the setup of the BlockPreconditioner!
  setup_velocity_conv_diff_operator(scaling_factor_time_derivative_term);

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Setup solvers ..." << std::endl;

  // setup preconditioner
  if(this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockDiagonal ||
     this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockTriangular ||
     this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockTriangularFactorization)
  {
    BlockPreconditionerData preconditioner_data;
    preconditioner_data.preconditioner_type = this->param.preconditioner_linearized_navier_stokes;
    preconditioner_data.momentum_preconditioner = this->param.momentum_preconditioner;
    preconditioner_data.exact_inversion_of_momentum_block = this->param.exact_inversion_of_momentum_block;
    preconditioner_data.multigrid_data_momentum_preconditioner = this->param.multigrid_data_momentum_preconditioner;
    preconditioner_data.rel_tol_solver_momentum_preconditioner = this->param.rel_tol_solver_momentum_preconditioner;
    preconditioner_data.max_n_tmp_vectors_solver_momentum_preconditioner = this->param.max_n_tmp_vectors_solver_momentum_preconditioner;
    preconditioner_data.schur_complement_preconditioner = this->param.schur_complement_preconditioner;
    preconditioner_data.discretization_of_laplacian = this->param.discretization_of_laplacian;
    preconditioner_data.exact_inversion_of_laplace_operator = this->param.exact_inversion_of_laplace_operator;
    preconditioner_data.multigrid_data_schur_complement_preconditioner = this->param.multigrid_data_schur_complement_preconditioner;
    preconditioner_data.rel_tol_solver_schur_complement_preconditioner = this->param.rel_tol_solver_schur_complement_preconditioner;

    preconditioner.reset(new BlockPreconditionerNavierStokes<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number, THIS>
        (this,preconditioner_data));
  }

  // setup linear solver
  if(this->param.solver_linearized_navier_stokes == SolverLinearizedNavierStokes::GMRES)
  {
    GMRESSolverData solver_data;
    solver_data.max_iter = this->param.max_iter_linear;
    solver_data.solver_tolerance_abs = this->param.abs_tol_linear;
    solver_data.solver_tolerance_rel = this->param.rel_tol_linear;
    solver_data.right_preconditioning = this->param.use_right_preconditioning;
    solver_data.update_preconditioner = this->param.update_preconditioner;
    solver_data.max_n_tmp_vectors = this->param.max_n_tmp_vectors;
    solver_data.compute_eigenvalues = false;

    if(this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockDiagonal ||
       this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockTriangular ||
       this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockTriangularFactorization)
    {
      solver_data.use_preconditioner = true;
    }

    linear_solver.reset(new GMRESSolver<THIS, PreconditionerNavierStokesBase<Number, THIS>,
                                        parallel::distributed::BlockVector<Number> >
        (*this,*preconditioner,solver_data));
  }
  else if(this->param.solver_linearized_navier_stokes == SolverLinearizedNavierStokes::FGMRES)
  {
    FGMRESSolverData solver_data;
    solver_data.max_iter = this->param.max_iter_linear;
    solver_data.solver_tolerance_abs = this->param.abs_tol_linear;
    solver_data.solver_tolerance_rel = this->param.rel_tol_linear;
    solver_data.update_preconditioner = this->param.update_preconditioner;
    solver_data.max_n_tmp_vectors = this->param.max_n_tmp_vectors;

    if(this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockDiagonal ||
       this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockTriangular ||
       this->param.preconditioner_linearized_navier_stokes == PreconditionerLinearizedNavierStokes::BlockTriangularFactorization)
    {
      solver_data.use_preconditioner = true;
    }

    linear_solver.reset(new FGMRESSolver<THIS, PreconditionerNavierStokesBase<Number, THIS>,
                                         parallel::distributed::BlockVector<Number> >
        (*this,*preconditioner,solver_data));
  }
  else
  {
    AssertThrow(this->param.solver_linearized_navier_stokes == SolverLinearizedNavierStokes::GMRES ||
                this->param.solver_linearized_navier_stokes == SolverLinearizedNavierStokes::FGMRES,
                ExcMessage("Specified solver for linearized Navier-Stokes problem not available."));
  }

  // setup Newton solver
  if(nonlinear_problem_has_to_be_solved())
  {
    newton_solver.reset(new NewtonSolver<parallel::distributed::BlockVector<Number>, THIS, THIS,
                                         IterativeSolverBase<parallel::distributed::BlockVector<Number> > >
       (this->param.newton_solver_data_coupled,*this,*this,*linear_solver));
  }

  setup_divergence_and_continuity_penalty_operators_and_solvers();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
setup_velocity_conv_diff_operator(double const &scaling_factor_time_derivative_term)
{
  VelocityConvDiffOperatorData<dim> vel_conv_diff_operator_data;

  // unsteady problem
  if(unsteady_problem_has_to_be_solved())
    vel_conv_diff_operator_data.unsteady_problem = true;
  else
    vel_conv_diff_operator_data.unsteady_problem = false;

  // convective problem
  if(nonlinear_problem_has_to_be_solved())
    vel_conv_diff_operator_data.convective_problem = true;
  else
    vel_conv_diff_operator_data.convective_problem = false;

  vel_conv_diff_operator_data.dof_index = this->get_dof_index_velocity();

  velocity_conv_diff_operator.initialize(
      this->get_data(),
      vel_conv_diff_operator_data,
      this->mass_matrix_operator,
      this->viscous_operator,
      this->convective_operator);

  velocity_conv_diff_operator.set_scaling_factor_time_derivative_term(scaling_factor_time_derivative_term);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
setup_divergence_and_continuity_penalty_operators_and_solvers()
{
  // divergence penalty operator
  if(this->param.use_divergence_penalty == true)
  {
    DivergencePenaltyOperatorData div_penalty_data;
    div_penalty_data.penalty_parameter = this->param.divergence_penalty_factor;

    divergence_penalty_operator.reset(new DivergencePenaltyOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>(
        this->data,
        this->get_dof_index_velocity(),
        this->get_quad_index_velocity_linear(),
        div_penalty_data));
  }

  // continuity penalty operator
  if(this->param.use_continuity_penalty == true)
  {
    ContinuityPenaltyOperatorData<dim> conti_penalty_data;
    conti_penalty_data.penalty_parameter = this->param.continuity_penalty_factor;
    conti_penalty_data.use_boundary_data = this->param.continuity_penalty_use_boundary_data;
    conti_penalty_data.bc = this->boundary_descriptor_velocity;

    continuity_penalty_operator.reset(new ContinuityPenaltyOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>(
        this->data,
        this->get_dof_index_velocity(),
        this->get_quad_index_velocity_linear(),
        conti_penalty_data));
  }

  // TODO: copied from DGNavierStokesProjectionMethods with only minor changes!

  // setup projection operator and projection solver

  // no penalty terms
  if(this->param.use_divergence_penalty == false &&
     this->param.use_continuity_penalty == false)
  {
    // do nothing
  }
  // divergence penalty only
  else if(this->param.use_divergence_penalty == true &&
          this->param.use_continuity_penalty == false)
  {
    // use direct solver
    if(this->param.solver_projection == SolverProjection::LU)
    {
      AssertThrow(divergence_penalty_operator.get() != 0,
          ExcMessage("Divergence penalty operator has not been initialized."));

      // projection operator
      typedef ProjectionOperatorDivergencePenaltyDirect<dim, fe_degree, fe_degree_p,
          fe_degree_xwall, xwall_quad_rule, Number> PROJ_OPERATOR;

      projection_operator.reset(new PROJ_OPERATOR(*divergence_penalty_operator));

      typedef DirectProjectionSolverDivergencePenalty<dim, fe_degree,
          fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> PROJ_SOLVER;

      projection_solver.reset(new PROJ_SOLVER(std::dynamic_pointer_cast<PROJ_OPERATOR>(projection_operator)));
    }
    // use iterative solver (PCG)
    else if(this->param.solver_projection == SolverProjection::PCG)
    {
      AssertThrow(divergence_penalty_operator.get() != 0,
          ExcMessage("Divergence penalty operator has not been initialized."));

      // projection operator
      typedef ProjectionOperatorDivergencePenaltyIterative<dim, fe_degree, fe_degree_p,
          fe_degree_xwall, xwall_quad_rule, Number> PROJ_OPERATOR;

      projection_operator.reset(new PROJ_OPERATOR(*divergence_penalty_operator));

      // solver
      ProjectionSolverData projection_solver_data;
      projection_solver_data.solver_tolerance_abs = this->param.abs_tol_projection;
      projection_solver_data.solver_tolerance_rel = this->param.rel_tol_projection;

      typedef IterativeProjectionSolverDivergencePenalty<dim, fe_degree,
          fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> PROJ_SOLVER;

      projection_solver.reset(new PROJ_SOLVER(*std::dynamic_pointer_cast<PROJ_OPERATOR>(projection_operator),
                                              projection_solver_data));
    }
    else
    {
      AssertThrow(this->param.solver_projection == SolverProjection::LU ||
                  this->param.solver_projection == SolverProjection::PCG,
          ExcMessage("Specified projection solver not implemented."));
    }
  }
  // both divergence and continuity penalty terms
  else if(this->param.use_divergence_penalty == true &&
          this->param.use_continuity_penalty == true)
  {
    AssertThrow(divergence_penalty_operator.get() != 0,
        ExcMessage("Divergence penalty operator has not been initialized."));

    AssertThrow(continuity_penalty_operator.get() != 0,
        ExcMessage("Continuity penalty operator has not been initialized."));

    // projection operator consisting of mass matrix operator,
    // divergence penalty operator, and continuity penalty operator
    typedef ProjectionOperatorDivergenceAndContinuityPenalty<dim, fe_degree,
        fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> PROJ_OPERATOR;

    projection_operator.reset(new PROJ_OPERATOR(this->mass_matrix_operator,
                                                *this->divergence_penalty_operator,
                                                *this->continuity_penalty_operator));

    // preconditioner
    if(this->param.preconditioner_projection == PreconditionerProjection::InverseMassMatrix)
    {
      preconditioner_projection.reset(new InverseMassMatrixPreconditioner<dim,fe_degree,Number>
         (this->data,
          this->get_dof_index_velocity(),
          this->get_quad_index_velocity_linear()));
    }
    else if(this->param.preconditioner_projection == PreconditionerProjection::PointJacobi)
    {
      // Note that at this point (when initializing the Jacobi preconditioner and calculating the diagonal)
      // the penalty parameter of the projection operator has not been calculated and the time step size has
      // not been set. Hence, update_preconditioner = true should be used for the Jacobi preconditioner in order
      // to use to correct diagonal for preconditioning.
      preconditioner_projection.reset(new JacobiPreconditioner<Number,PROJ_OPERATOR>
          (*std::dynamic_pointer_cast<PROJ_OPERATOR>(projection_operator)));
    }
    else if(this->param.preconditioner_projection == PreconditionerProjection::BlockJacobi)
    {
      // Note that at this point (when initializing the Jacobi preconditioner)
      // the penalty parameter of the projection operator has not been calculated and the time step size has
      // not been set. Hence, update_preconditioner = true should be used for the Jacobi preconditioner in order
      // to use to correct diagonal blocks for preconditioning.
      preconditioner_projection.reset(new BlockJacobiPreconditioner<Number,PROJ_OPERATOR>
          (*std::dynamic_pointer_cast<PROJ_OPERATOR>(projection_operator)));
    }
    else
    {
      AssertThrow(this->param.preconditioner_projection == PreconditionerProjection::None ||
                  this->param.preconditioner_projection == PreconditionerProjection::InverseMassMatrix ||
                  this->param.preconditioner_projection == PreconditionerProjection::PointJacobi ||
                  this->param.preconditioner_projection == PreconditionerProjection::BlockJacobi,
                  ExcMessage("Specified preconditioner of projection solver not implemented."));
    }

    // solver
    if(this->param.solver_projection == SolverProjection::PCG)
    {
      // setup solver data
      CGSolverData projection_solver_data;
      // use default value of max_iter
      projection_solver_data.solver_tolerance_abs = this->param.abs_tol_projection;
      projection_solver_data.solver_tolerance_rel = this->param.rel_tol_projection;
      // default value of use_preconditioner = false
      if(this->param.preconditioner_projection == PreconditionerProjection::InverseMassMatrix ||
         this->param.preconditioner_projection == PreconditionerProjection::PointJacobi ||
         this->param.preconditioner_projection == PreconditionerProjection::BlockJacobi)
      {
        projection_solver_data.use_preconditioner = true;
        projection_solver_data.update_preconditioner = this->param.update_preconditioner_projection;
      }
      else
      {
        AssertThrow(this->param.preconditioner_projection == PreconditionerProjection::None ||
                    this->param.preconditioner_projection == PreconditionerProjection::InverseMassMatrix ||
                    this->param.preconditioner_projection == PreconditionerProjection::PointJacobi ||
                    this->param.preconditioner_projection == PreconditionerProjection::BlockJacobi,
                    ExcMessage("Specified preconditioner of projection solver not implemented."));
      }

      // setup solver
      projection_solver.reset(new CGSolver<PROJ_OPERATOR,PreconditionerBase<Number>,parallel::distributed::Vector<Number> >
         (*std::dynamic_pointer_cast<PROJ_OPERATOR>(projection_operator),
          *preconditioner_projection,
          projection_solver_data));
    }
    else
    {
      AssertThrow(this->param.solver_projection == SolverProjection::PCG,
          ExcMessage("Specified projection solver not implemented."));
    }
  }
  else
  {
    AssertThrow(false,ExcMessage("Specified combination of divergence and continuity penalty operators not implemented."));
  }

}

// TODO: this function can be removed when performing the projection in a postprocessing step
//template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
//void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
//update_divergence_penalty_operator (parallel::distributed::Vector<Number> const &velocity) const
//{
//  this->divergence_penalty_operator->calculate_array_penalty_parameter(velocity);
//}

// TODO: this function can be removed when performing the projection in a postprocessing step
//template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
//void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
//update_continuity_penalty_operator (parallel::distributed::Vector<Number> const &velocity) const
//{
//  this->continuity_penalty_operator->calculate_array_penalty_parameter(velocity);
//}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
unsigned int DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
solve_linear_stokes_problem (parallel::distributed::BlockVector<Number>       &dst,
                             parallel::distributed::BlockVector<Number> const &src,
                             double const                                     &scaling_factor_mass_matrix_term)
{
  // Set scaling_factor_time_derivative_term for linear operator (velocity_conv_diff_operator).
  velocity_conv_diff_operator.set_scaling_factor_time_derivative_term(scaling_factor_mass_matrix_term);

  // Note that there is no need to set the evaluation time for the velocity_conv_diff_operator
  // because this function is only called if the convective term is not considered
  // in the velocity_conv_diff_operator (Stokes eq. or explicit treatment of convective term).

  return linear_solver->solve(dst,src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
rhs_stokes_problem (parallel::distributed::BlockVector<Number>  &dst,
                    parallel::distributed::Vector<Number> const *src,
                    double const                                &eval_time) const
{
  // velocity-block
  this->gradient_operator.rhs(dst.block(0),eval_time);
  dst.block(0) *= scaling_factor_continuity;

  this->viscous_operator.rhs_add(dst.block(0),eval_time);

  if(unsteady_problem_has_to_be_solved())
    this->mass_matrix_operator.apply_add(dst.block(0),*src);

  if(this->param.right_hand_side == true)
    this->body_force_operator.evaluate_add(dst.block(0),eval_time);

  // pressure-block
  this->divergence_operator.rhs(dst.block(1),eval_time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  // scale by scaling_factor_continuity
  dst.block(1) *= - scaling_factor_continuity;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
vmult (parallel::distributed::BlockVector<Number>       &dst,
       parallel::distributed::BlockVector<Number> const &src) const
{
  apply_linearized_problem(dst,src);
}


template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
apply_linearized_problem (parallel::distributed::BlockVector<Number>       &dst,
                          parallel::distributed::BlockVector<Number> const &src) const
{
  // (1,1) block of saddle point matrix
  velocity_conv_diff_operator.vmult(dst.block(0),src.block(0));

  // Divergence and continuity penalty operators
  // TODO this function has to be removed when performing the projection in a postprocessing step
//  if(this->param.use_divergence_penalty == true)
//    divergence_penalty_operator->apply_add(dst.block(0),src.block(0));
//  if(this->param.use_continuity_penalty == true)
//    continuity_penalty_operator->apply_add(dst.block(0),src.block(0));

  // (1,2) block of saddle point matrix
  // gradient operator: dst = velocity, src = pressure
  this->gradient_operator.apply(temp_vector,src.block(1));
  dst.block(0).add(scaling_factor_continuity,temp_vector);

  // (2,1) block of saddle point matrix
  // divergence operator: dst = pressure, src = velocity
  this->divergence_operator.apply(dst.block(1),src.block(0));
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  // scale by scaling_factor_continuity
  dst.block(1) *= - scaling_factor_continuity;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
solve_nonlinear_steady_problem (parallel::distributed::BlockVector<Number>  &dst,
                                unsigned int                                &newton_iterations,
                                unsigned int                                &linear_iterations)
{
  // solve nonlinear problem
  newton_solver->solve(dst,newton_iterations,linear_iterations);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
solve_nonlinear_problem (parallel::distributed::BlockVector<Number>  &dst,
                         parallel::distributed::Vector<Number> const &sum_alphai_ui,
                         double const                                &eval_time,
                         double const                                &scaling_factor_mass_matrix_term,
                         unsigned int                                &newton_iterations,
                         unsigned int                                &linear_iterations)
{
  // Set sum_alphai_ui (this variable is used when evaluating the nonlinear residual).
  this->sum_alphai_ui = &sum_alphai_ui;

  // Set evaluation_time for nonlinear operator (=DGNavierStokesCoupled)
  evaluation_time = eval_time;
  // Set scaling_factor_time_derivative_term for nonlinear operator (=DGNavierStokesCoupled)
  scaling_factor_time_derivative_term = scaling_factor_mass_matrix_term;

  // Set correct evaluation time for linear operator (velocity_conv_diff_operator).
  velocity_conv_diff_operator.set_evaluation_time(eval_time);
  // Set scaling_factor_time_derivative_term for linear operator (velocity_conv_diff_operator).
  velocity_conv_diff_operator.set_scaling_factor_time_derivative_term(scaling_factor_mass_matrix_term);

  // Solve nonlinear problem
  newton_solver->solve(dst,newton_iterations,linear_iterations);

  // Reset sum_alphai_ui
  this->sum_alphai_ui = nullptr;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
evaluate_nonlinear_residual (parallel::distributed::BlockVector<Number>       &dst,
                             parallel::distributed::BlockVector<Number> const &src)
{
  // velocity-block

  // set dst.block(0) to zero. This is necessary since subsequent operators
  // call functions of type ..._add
  dst.block(0) = 0.0;

  if(this->param.right_hand_side == true)
  {
    this->body_force_operator.evaluate(dst.block(0),evaluation_time);
    // Shift body force term to the left-hand side of the equation.
    // This works since body_force_operator is the first operator
    // that is evaluated.
    dst.block(0) *= -1.0;
  }

  if(unsteady_problem_has_to_be_solved())
  {
    temp_vector.equ(scaling_factor_time_derivative_term,src.block(0));
    temp_vector.add(-1.0,*sum_alphai_ui);
    this->mass_matrix_operator.apply_add(dst.block(0),temp_vector);
  }

  this->convective_operator.evaluate_add(dst.block(0),src.block(0),evaluation_time);
  this->viscous_operator.evaluate_add(dst.block(0),src.block(0),evaluation_time);

  // Divergence and continuity penalty operators
  // TODO this function has to be removed when performing the projection in a postprocessing step
//  if(this->param.use_divergence_penalty == true)
//    divergence_penalty_operator->apply_add(dst.block(0),src.block(0));
//  if(this->param.use_continuity_penalty == true)
//    continuity_penalty_operator->apply_add(dst.block(0),src.block(0));

  // gradient operator scaled by scaling_factor_continuity
  this->gradient_operator.evaluate(temp_vector,src.block(1),evaluation_time);
  dst.block(0).add(scaling_factor_continuity,temp_vector);


  // pressure-block

  this->divergence_operator.evaluate(dst.block(1),src.block(0),evaluation_time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  // scale by scaling_factor_continuity
  dst.block(1) *= - scaling_factor_continuity;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
update_projection_operator(parallel::distributed::Vector<Number> const &velocity,
                           double const                                time_step_size) const
{
  // Update projection operator, i.e., the penalty parameters that depend on
  // the current solution (velocity field).
  if(this->param.use_divergence_penalty == true)
  {
    divergence_penalty_operator->calculate_array_penalty_parameter(velocity);
  }
  if(this->param.use_continuity_penalty == true)
  {
    continuity_penalty_operator->calculate_array_penalty_parameter(velocity);
  }

  // Set the correct time step size.
  Assert(projection_operator.get() != 0, ExcMessage("Projection operator has not been initialized."));
  projection_operator->set_time_step_size(time_step_size);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
unsigned int DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
solve_projection (parallel::distributed::Vector<Number>       &dst,
                  parallel::distributed::Vector<Number> const &src) const
{
  // Solve projection equation.
  Assert(projection_solver.get() != 0, ExcMessage("Projection solver has not been initialized."));
  unsigned int n_iter = projection_solver->solve(dst,src);

  return n_iter;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesCoupled<dim,fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
rhs_projection_add (parallel::distributed::Vector<Number> &dst,
                    double const                          eval_time) const
{
  if(this->param.use_divergence_penalty == true &&
     this->param.use_continuity_penalty == true)
 {
    typedef ProjectionOperatorDivergenceAndContinuityPenalty<dim, fe_degree,
        fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> PROJ_OPERATOR;

    std::shared_ptr<PROJ_OPERATOR> proj_operator_div_and_conti_penalty
      = std::dynamic_pointer_cast<PROJ_OPERATOR>(projection_operator);
    proj_operator_div_and_conti_penalty->rhs_add(dst,eval_time);
 }
}

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_COUPLED_SOLVER_H_ */
