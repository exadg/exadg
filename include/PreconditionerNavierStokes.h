/*
 * PreconditionerNavierStokes.h
 *
 *  Created on: Jun 30, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_PRECONDITIONERNAVIERSTOKES_H_
#define INCLUDE_PRECONDITIONERNAVIERSTOKES_H_

#include "Preconditioner.h"
#include "HelmholtzOperator.h"
#include "poisson_solver.h"
#include "CompatibleLaplaceOperator.h"
#include "PressureConvectionDiffusionOperator.h"
#include "VelocityConvDiffOperator.h"
#include "IterativeSolvers.h"

// just for testing of multigrid
#include "CheckMultigrid.h"


// forward declaration
template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
class DGNavierStokesCoupled;

template<int dim> class HelmholtzOperatorData;
template<int dim> class LaplaceOperatorData;

template<typename value_type>
class PreconditionerNavierStokesBase
{
public:
  virtual ~PreconditionerNavierStokesBase(){}

  virtual void vmult(parallel::distributed::BlockVector<value_type>        &dst,
                     const parallel::distributed::BlockVector<value_type>  &src) const = 0;

  virtual void update_preconditioner() const = 0;
};

struct BlockPreconditionerData
{
  PreconditionerLinearizedNavierStokes preconditioner_type;

  // preconditioner momentum block
  MomentumPreconditioner momentum_preconditioner;
  SolverMomentumPreconditioner solver_momentum_preconditioner;
  MultigridData multigrid_data_momentum_preconditioner;
  double rel_tol_solver_momentum_preconditioner;

  // preconditioner Schur-complement block
  SchurComplementPreconditioner schur_complement_preconditioner;
  DiscretizationOfLaplacian discretization_of_laplacian;
  SolverSchurComplementPreconditioner solver_schur_complement_preconditioner;
  MultigridData multigrid_data_schur_complement_preconditioner;
  double rel_tol_solver_schur_complement_preconditioner;
};

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
class BlockPreconditionerNavierStokes : public PreconditionerNavierStokesBase<value_type>
{
public:
  typedef float Number;

  BlockPreconditionerNavierStokes(DGNavierStokesCoupled<dim, fe_degree,
                                    fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> *underlying_operator_in,
                                  BlockPreconditionerData const                        &preconditioner_data_in)
  :
    use_gmres_smoother(true) // TODO
  {
    underlying_operator = underlying_operator_in;
    preconditioner_data = preconditioner_data_in;

    /*********** initialization of temporary vector ***************/
    if(preconditioner_data.preconditioner_type == PreconditionerLinearizedNavierStokes::BlockTriangular)
    {
      underlying_operator->initialize_vector_velocity(vec_tmp_velocity);
    }
    else if(preconditioner_data.preconditioner_type == PreconditionerLinearizedNavierStokes::BlockTriangularFactorization)
    {
      underlying_operator->initialize_vector_pressure(vec_tmp_pressure);
      underlying_operator->initialize_vector_velocity(vec_tmp_velocity);
      underlying_operator->initialize_vector_velocity(vec_tmp_velocity_2);
    }
    /*********** initialization of temporary vector ***************/

    /****** setup velocity convection-diffusion operator **********/
    if(((preconditioner_data.momentum_preconditioner == MomentumPreconditioner::VelocityDiffusion ||
         preconditioner_data.momentum_preconditioner == MomentumPreconditioner::VelocityConvectionDiffusion)
          &&
         preconditioner_data.solver_momentum_preconditioner == SolverMomentumPreconditioner::GeometricMultigridGMRES)
       ||
         preconditioner_data.schur_complement_preconditioner == SchurComplementPreconditioner::Elman)
    {
      VelocityConvDiffOperatorData<dim> vel_conv_diff_operator_data;
      // unsteady problem ?
      if(underlying_operator->param.problem_type == ProblemType::Unsteady)
        vel_conv_diff_operator_data.unsteady_problem = true;
      else
        vel_conv_diff_operator_data.unsteady_problem = false;
      // convective problem ?
      if(underlying_operator->nonlinear_problem_has_to_be_solved() == true)
        vel_conv_diff_operator_data.convective_problem = true;
      else
        vel_conv_diff_operator_data.convective_problem = false;

      vel_conv_diff_operator_data.scaling_factor_time_derivative_term = underlying_operator->scaling_factor_time_derivative_term;

      vel_conv_diff_operator_data.mass_matrix_operator_data = underlying_operator->get_mass_matrix_operator_data();
      // TODO: Velocity conv diff operator is initialized with constant viscosity, in case of varying viscosities
      // the vel conv diff operator (the viscous operator of the conv diff operator) has to be updated before applying this
      // preconditioner
      vel_conv_diff_operator_data.viscous_operator_data = underlying_operator->get_viscous_operator_data();
      vel_conv_diff_operator_data.convective_operator_data = underlying_operator->get_convective_operator_data();

      vel_conv_diff_operator_data.dof_index = underlying_operator->get_dof_index_velocity();
      vel_conv_diff_operator_data.periodic_face_pairs_level0 = underlying_operator->periodic_face_pairs;

      velocity_conv_diff_operator.reset(new VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, value_type>());
      velocity_conv_diff_operator->initialize(
          underlying_operator->get_data(),
          vel_conv_diff_operator_data,
          underlying_operator->mass_matrix_operator,
          underlying_operator->viscous_operator,
          underlying_operator->convective_operator);
   }
    /****** setup velocity convection-diffusion operator **********/

    /****** preconditioner velocity/momentum block ****************/
    if(preconditioner_data.momentum_preconditioner == MomentumPreconditioner::InverseMassMatrix)
    {
      // inverse mass matrix
      inv_mass_matrix_preconditioner_momentum.reset(new InverseMassMatrixPreconditioner<dim,fe_degree,value_type>(
        underlying_operator->get_data(),
        underlying_operator->get_dof_index_velocity(),
        underlying_operator->get_quad_index_velocity_linear()));
    }
    else if(preconditioner_data.momentum_preconditioner == MomentumPreconditioner::VelocityDiffusion ||
            preconditioner_data.momentum_preconditioner == MomentumPreconditioner::VelocityConvectionDiffusion)
    {
      // multigrid preconditioner for Helmholtz operator (unsteady case) or viscous operator (steady case)
      setup_multigrid_preconditioner_momentum();

      if(preconditioner_data.solver_momentum_preconditioner == SolverMomentumPreconditioner::GeometricMultigridGMRES)
      {
        setup_iterative_solver_momentum();
      }
    }
    /****** preconditioner velocity/momentum block ****************/

    /****** preconditioner pressure/Schur-complement block ********/
    if(preconditioner_data.schur_complement_preconditioner == SchurComplementPreconditioner::InverseMassMatrix)
    {
      // inverse mass matrix
      inv_mass_matrix_preconditioner_schur_complement.reset(new InverseMassMatrixPreconditioner<dim,fe_degree_p,value_type,1>(
          underlying_operator->get_data(),
          underlying_operator->get_dof_index_pressure(),
          underlying_operator->get_quad_index_pressure()));
    }
    else if(preconditioner_data.schur_complement_preconditioner == SchurComplementPreconditioner::LaplaceOperator)
    {
      // multigrid for negative Laplace operator (classical or compatible discretization)
      setup_multigrid_preconditioner_schur_complement();

      if(preconditioner_data.solver_schur_complement_preconditioner == SolverSchurComplementPreconditioner::GeometricMultigridPCG)
      {
        // iterative solver used to invert the negative Laplace operator (classical or compatible discretization)
        setup_iterative_solver_schur_complement();
      }
    }
    else if(preconditioner_data.schur_complement_preconditioner == SchurComplementPreconditioner::CahouetChabard)
    {
      AssertThrow(underlying_operator->param.problem_type == ProblemType::Unsteady,
          ExcMessage("CahouetChabard preconditioner only makes sense for unsteady problems."));

      // multigrid for negative Laplace operator (classical or compatible discretization)
      setup_multigrid_preconditioner_schur_complement();

      if(preconditioner_data.solver_schur_complement_preconditioner == SolverSchurComplementPreconditioner::GeometricMultigridPCG)
      {
        // iterative solver used to invert the negative Laplace operator (classical or compatible discretization)
        setup_iterative_solver_schur_complement();
      }

      // inverse mass matrix to also include the part of the preconditioner that is beneficial when using large time steps
      // and large viscosities.
      inv_mass_matrix_preconditioner_schur_complement.reset(new InverseMassMatrixPreconditioner<dim,fe_degree_p,value_type,1>(
        underlying_operator->get_data(),
        underlying_operator->get_dof_index_pressure(),
        underlying_operator->get_quad_index_pressure()));

      // initialize tmp vector
      underlying_operator->initialize_vector_pressure(tmp_scp_pressure);
    }
    else if(preconditioner_data.schur_complement_preconditioner == SchurComplementPreconditioner::Elman)
    {
      // multigrid for negative Laplace operator (classical or compatible discretization)
      setup_multigrid_preconditioner_schur_complement();

      if(preconditioner_data.solver_schur_complement_preconditioner == SolverSchurComplementPreconditioner::GeometricMultigridPCG)
      {
        // iterative solver used to invert the negative Laplace operator (classical or compatible discretization)
        setup_iterative_solver_schur_complement();
      }

      if(preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Compatible)
      {
        // -S^{-1} = - (BM^{-1}B^T)^{-1} (-B M^{-1} A M^{-1} B^T) (BM^{-1}B^T)^{-1}
        // --> inverse velocity mass matrix needed for inner factor
        inv_mass_matrix_preconditioner_schur_complement.reset(new InverseMassMatrixPreconditioner<dim,fe_degree,value_type,dim>(
          underlying_operator->get_data(),
          underlying_operator->get_dof_index_velocity(),
          underlying_operator->get_quad_index_velocity_linear()));
      }

      AssertThrow(velocity_conv_diff_operator.get() != 0,
                  ExcMessage("Setup of Schur complement preconditioner: Velocity convection-diffusion operator is uninitialized"));

      // initialize tmp vectors
      underlying_operator->initialize_vector_pressure(tmp_scp_pressure);
      underlying_operator->initialize_vector_velocity(tmp_scp_velocity);
      underlying_operator->initialize_vector_velocity(tmp_scp_velocity_2);
    }
    else if(preconditioner_data.schur_complement_preconditioner == SchurComplementPreconditioner::PressureConvectionDiffusion)
    {
      // -S^{-1} = M_p^{-1} A_p (-L)^{-1}

      // I. multigrid for negative Laplace operator (classical or compatible discretization)
      setup_multigrid_preconditioner_schur_complement();

      if(preconditioner_data.solver_schur_complement_preconditioner == SolverSchurComplementPreconditioner::GeometricMultigridPCG)
      {
        setup_iterative_solver_schur_complement();
      }

      // II. pressure convection-diffusion operator
      setup_pressure_convection_diffusion_operator();

      // III. inverse pressure mass matrix
      inv_mass_matrix_preconditioner_schur_complement.reset(new InverseMassMatrixPreconditioner<dim,fe_degree_p,value_type,1>(
        underlying_operator->get_data(),
        underlying_operator->get_dof_index_pressure(),
        underlying_operator->get_quad_index_pressure()));

      // initialize tmp vector
      underlying_operator->initialize_vector_pressure(tmp_scp_pressure);
    }
    /****** preconditioner pressure/Schur-complement block ********/
  }

  void vmult(parallel::distributed::BlockVector<value_type>       &dst,
             const parallel::distributed::BlockVector<value_type> &src) const
  {
    /*
     * Saddle point matrix:
     *
     *       / A  B^{T} \
     *   M = |          |    with Schur complement S = -B A^{-1} B^{T}
     *       \ B    0   /
     *
     * Block-diagonal preconditioner:
     *
     *                   / A   0 \                       / A^{-1}   0    \   / A^{-1}  0 \   / I      0    \
     *   -> P_diagonal = |       |  -> P_diagonal^{-1} = |               | = |           | * |             |
     *                   \ 0  -S /                       \   0   -S^{-1} /   \   0     I /   \ 0   -S^{-1} /
     *
     * Block-triangular preconditioner:
     *
     *                     / A   B^{T} \                         / A^{-1}  0 \   / I  B^{T} \   / I      0    \
     *   -> P_triangular = |           |  -> P_triangular^{-1} = |           | * |          | * |             |
     *                     \ 0    -S   /                         \   0     I /   \ 0   -I   /   \ 0   -S^{-1} /
     *
     * Block-triangular factorization:
     *
     *                      / A  0 \  / I   A^{-1} B^{T} \
     *   -> P_tria-factor = |      |  |                  |
     *                      \ B  S /  \ 0        I       /
     *
     *                            / I  - A^{-1} B^{T} \   / I      0    \   / I   0 \   / A^{-1}  0 \
     *    -> P_tria-factor^{-1} = |                   | * |             | * |       | * |           |
     *                            \ 0          I      /   \ 0   -S^{-1} /   \ B  -I /   \   0     I /
     *
     * Approximations to A and S:
     *
     * STOKES EQUATIONS:
     *
     *   - A = 1/dt M_u - nu L   where M_u is the velocity mass matrix and L the Laplace operator
     *
     *   - S = - B A^{-1} B^T
     *       |
     *       |  apply method of pseudo-differential operators
     *      \|/
     *       = - (- div ) * ( 1/dt * I - nu * laplace )^{-1} * grad
     *
     *   - dt small, nu small:
     *      A = 1/dt M_u
     *        --> A^{-1} = dt M_u^{-1}
     *      S = div * (1/dt I)^{-1} * grad = dt * laplace
     *        --> - S^{-1} = 1/dt (-L)^{-1} (by peforming one multigrid V-cylce on the pressure Poisson operator)
     *
     *   - dt large, nu large:
     *      A = 1/dt M_u - nu L = H
     *        --> A^{-1} = H^{-1} (by performing one multigrid V-cycle on the Helmholtz operator)
     *      S = div * (- nu * laplace)^{-1} * grad = - 1/nu * I
     *        --> - S^{-1} = nu M_p^{-1} (M_p: pressure mass matrix)
     *
     *   - combination of both limiting cases (Cahouet Chabard approach, only relevant for unsteady case):
     *      A = 1/dt M_u - nu L = H
     *        --> A^{-1} = H^{-1} (by performing one multigrid V-cycle on the Helmholtz operator)
     *      Schur complement:
     *        --> - S^{-1} = nu M_p^{-1} + 1/dt (-L)^{-1} (by performing one multigrid V-cylce on the pressure Poisson operator
     *                                                     and subsequent application of the inverse pressure mass matrix)
     *
     *  - dt --> infinity (steady state case):
     *      A = (-nu L)
     *        --> A^{-1} = (-nu L)^{-1} (by performing one multigrid V-cycle on the viscous operator)
     *      S = div * (- nu * laplace)^{-1} * grad = - 1/nu * I
     *        --> - S^{-1} = nu M_p^{-1} (M_p: pressure mass matrix)
     *
     *  NAVIER-STOKES EQUATIONS:
     *
     *  - dt --> infinity (steady state case):
     *      A = -nu L + C(u_lin,u) ( C: linearized convective term)
     *        --> A = -nu L --> A^{-1} = (-nu L)^{-1} (by performing one multigrid V-cycle on the viscous operator)
     *      S: approach of Elman:
     *      S = - B A^{-1}B^T approx. (BB^T) (-B A B^T)^{-1} (BB^T)
     *        --> -S^{-1} = - (BB^T)^{-1} (-B A B^T) (BB^T)^{-1}: to approximate (BB^T)^{-1} perform one multgrid V-cycle on
     *            negative Laplace operator (classical discretization)
     */

    if(preconditioner_data.preconditioner_type == PreconditionerLinearizedNavierStokes::BlockDiagonal)
    {
      /*
       *         / I      0    \
       *  temp = |             | * src
       *         \ 0   -S^{-1} /
       */

      // apply preconditioner for pressure/Schur-complement block
      vmult_pressure_block(dst.block(1),src.block(1));

      /*
       *        / A^{-1}  0 \
       *  dst = |           | * temp
       *        \   0     I /
       */

      // apply preconditioner for velocity/momentum block
      vmult_velocity_block(dst.block(0),src.block(0));
    }
    else if(preconditioner_data.preconditioner_type == PreconditionerLinearizedNavierStokes::BlockTriangular)
    {
      /*
       *        / I      0    \
       *  dst = |             | * src
       *        \ 0   -S^{-1} /
       */

      // for the velocity block simply copy data from src to dst
      dst.block(0) = src.block(0);
      // apply preconditioner for pressure/Schur-complement block
      vmult_pressure_block(dst.block(1),src.block(1));

      /*
       *        / I  B^{T} \
       *  dst = |          | * dst
       *        \ 0   -I   /
       */

      underlying_operator->gradient_operator.apply_add(dst.block(0),dst.block(1));
      dst.block(1) *= -1.0;

      /*
       *        / A^{-1}  0 \
       *  dst = |           | * dst
       *        \   0     I /
       */

      // copy data from dst.block(0) to vec_tmp_velocity before applying the preconditioner for the velocity block
      vec_tmp_velocity = dst.block(0);
      // apply preconditioner for velocity/momentum block
      vmult_velocity_block(dst.block(0),vec_tmp_velocity);
    }
    else if(preconditioner_data.preconditioner_type == PreconditionerLinearizedNavierStokes::BlockTriangularFactorization)
    {
      /*
      *        / A^{-1}  0 \
      *  dst = |           | * src
      *        \   0     I /
      */

      // for the pressure block simply copy data from src to dst
      dst.block(1) = src.block(1);
      // apply preconditioner for velocity/momentum block
      vmult_velocity_block(dst.block(0),src.block(0));

      /*
      *        / I   0 \
      *  dst = |       | * dst
      *        \ B  -I /
      */

      // dst.block(1) = B*dst.block(0) - dst.block(1) = -1.0 * ( dst.block(1) + (-B) * dst.block(0) );
      // I. dst.block(1) += (-B) * dst.block(0);
      // note that B represents NEGATIVE divergence operator, i.e., applying the divergence operator means appyling -B
      underlying_operator->divergence_operator.apply_add(dst.block(1),dst.block(0));
      // II. dst.block(1) = -dst.block(1);
      dst.block(1) *= -1.0;


      /*
      *        / I      0    \
      *  dst = |             | * dst
      *        \ 0   -S^{-1} /
      */

      // copy data from dst.block(1) to vec_tmp_pressure before applying the preconditioner for the pressure block
      vec_tmp_pressure = dst.block(1);
      // apply preconditioner for pressure/Schur-complement block
      vmult_pressure_block(dst.block(1),vec_tmp_pressure);

      /*
      *        / I  - A^{-1} B^{T} \
      *  dst = |                   | * dst
      *        \ 0          I      /
      */

      // vec_tmp_velocity = B^{T} * dst(1)
      underlying_operator->gradient_operator.apply(vec_tmp_velocity,dst.block(1));

      // vec_tmp_velocity_2 = A^{-1} * vec_tmp_velocity
      vmult_velocity_block(vec_tmp_velocity_2,vec_tmp_velocity);

      // dst(0) = dst(0) - vec_tmp_velocity_2
      dst.block(0).add(-1.0,vec_tmp_velocity_2);
    }
    else
    {
      AssertThrow(preconditioner_data.preconditioner_type == PreconditionerLinearizedNavierStokes::BlockDiagonal ||
                  preconditioner_data.preconditioner_type == PreconditionerLinearizedNavierStokes::BlockTriangular ||
                  preconditioner_data.preconditioner_type == PreconditionerLinearizedNavierStokes::BlockTriangularFactorization,
                  ExcMessage("Specified preconditioner for linearized Navier-Stokes problem is not implemented."));
    }
  }

private:
  void setup_multigrid_preconditioner_momentum()
  {
    MultigridData mg_data = preconditioner_data.multigrid_data_momentum_preconditioner;

    if(preconditioner_data.momentum_preconditioner == MomentumPreconditioner::VelocityDiffusion)
    {
      //Unsteady problem -> consider Helmholtz operator
      if(underlying_operator->param.problem_type == ProblemType::Unsteady)
      {
        // Geometric multigrid V-cycle performed on Helmholtz operator
        HelmholtzOperatorData<dim> helmholtz_operator_data;

        helmholtz_operator_data.mass_matrix_operator_data = underlying_operator->get_mass_matrix_operator_data();
        // TODO: this Helmholtz operator is initialized with constant viscosity, in case of varying viscosities
        // the helmholtz operator (the viscous operator of the helmholtz operator) has to be updated before applying this
        // preconditioner
        helmholtz_operator_data.viscous_operator_data = underlying_operator->get_viscous_operator_data();

        helmholtz_operator_data.dof_index = underlying_operator->get_dof_index_velocity();
        // TODO: this Helmholtz operator is initialized with constant scaling_factor_time_derivative term,
        // in case of a varying scaling_factor_time_derivate_term (adaptive time stepping)
        // the helmholtz operator has to be updated before applying this preconditioner
        helmholtz_operator_data.scaling_factor_time_derivative_term = underlying_operator->get_scaling_factor_time_derivative_term();
        helmholtz_operator_data.periodic_face_pairs_level0 = underlying_operator->periodic_face_pairs;

        multigrid_preconditioner_momentum.reset(new MyMultigridPreconditioner<dim,value_type,
                                                      HelmholtzOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>,
                                                      HelmholtzOperatorData<dim> >());

        std_cxx11::shared_ptr<MyMultigridPreconditioner<dim,value_type,
                                HelmholtzOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>,
                                HelmholtzOperatorData<dim> > >
          mg_preconditioner = std::dynamic_pointer_cast<MyMultigridPreconditioner<dim,value_type,
                                                          HelmholtzOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>,
                                                          HelmholtzOperatorData<dim> > >(multigrid_preconditioner_momentum);

        mg_preconditioner->initialize(mg_data,
                                      underlying_operator->get_dof_handler_u(),
                                      underlying_operator->get_mapping(),
                                      helmholtz_operator_data,
                                      underlying_operator->dirichlet_boundary,
                                      underlying_operator->get_fe_parameters());
      }
      // Steady problem -> consider viscous operator
      else if(underlying_operator->param.problem_type == ProblemType::Steady)
      {
        // Geometric multigrid V-cycle performed on viscous operator
        ViscousOperatorData<dim> viscous_operator_data;
        // TODO: this viscous operator is initialized with constant viscosity, in case of varying viscosities
        // the viscous operator has to be updated before applying this preconditioner
        viscous_operator_data = underlying_operator->get_viscous_operator_data();

        multigrid_preconditioner_momentum.reset(new MyMultigridPreconditioner<dim,value_type,
                                                      ViscousOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>,
                                                      ViscousOperatorData<dim> >());

        std_cxx11::shared_ptr<MyMultigridPreconditioner<dim,value_type,
                                ViscousOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>,
                                ViscousOperatorData<dim> > >
          mg_preconditioner = std::dynamic_pointer_cast<MyMultigridPreconditioner<dim,value_type,
                                                          ViscousOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>,
                                                          ViscousOperatorData<dim> > >(multigrid_preconditioner_momentum);

        mg_preconditioner->initialize(mg_data,
                                      underlying_operator->get_dof_handler_u(),
                                      underlying_operator->get_mapping(),
                                      viscous_operator_data,
                                      underlying_operator->dirichlet_boundary,
                                      underlying_operator->get_fe_parameters());
      }
    }
    else if(preconditioner_data.momentum_preconditioner == MomentumPreconditioner::VelocityConvectionDiffusion)
    {
      // Geometric multigrid V-cycle performed on velocity conv-diff operator
      VelocityConvDiffOperatorData<dim> vel_conv_diff_operator_data;
      // unsteady problem ?
      if(underlying_operator->param.problem_type == ProblemType::Unsteady)
        vel_conv_diff_operator_data.unsteady_problem = true;
      else
        vel_conv_diff_operator_data.unsteady_problem = false;
      // convective problem ?
      if(underlying_operator->nonlinear_problem_has_to_be_solved() == true)
        vel_conv_diff_operator_data.convective_problem = true;
      else
        vel_conv_diff_operator_data.convective_problem = false;

      // TODO: Velocity conv diff operator is initialized with constant scaling_factor_time_derivative term,
      // in case of a varying scaling_factor_time_derivate_term (adaptive time stepping)
      // the velocity conv diff operator has to be updated before applying this preconditioner
      vel_conv_diff_operator_data.scaling_factor_time_derivative_term = underlying_operator->scaling_factor_time_derivative_term;

      vel_conv_diff_operator_data.mass_matrix_operator_data = underlying_operator->get_mass_matrix_operator_data();
      // TODO: Velocity conv diff operator is initialized with constant viscosity, in case of varying viscosities
      // the vel conv diff operator (the viscous operator of the conv diff operator) has to be updated before applying this
      // preconditioner
      vel_conv_diff_operator_data.viscous_operator_data = underlying_operator->get_viscous_operator_data();
      vel_conv_diff_operator_data.convective_operator_data = underlying_operator->get_convective_operator_data();

      vel_conv_diff_operator_data.dof_index = underlying_operator->get_dof_index_velocity();
      vel_conv_diff_operator_data.periodic_face_pairs_level0 = underlying_operator->periodic_face_pairs;

      if(!use_gmres_smoother)
      {
      multigrid_preconditioner_momentum.reset(new MyMultigridPreconditionerVelocityConvectionDiffusion<dim,value_type,
                                                    VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>,
                                                    VelocityConvDiffOperatorData<dim> >());

      std_cxx11::shared_ptr<MyMultigridPreconditionerVelocityConvectionDiffusion<dim,value_type,
                              VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>,
                              VelocityConvDiffOperatorData<dim> > >
        mg_preconditioner = std::dynamic_pointer_cast<MyMultigridPreconditionerVelocityConvectionDiffusion<dim,value_type,
                                                        VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>,
                                                        VelocityConvDiffOperatorData<dim> > >(multigrid_preconditioner_momentum);

      mg_preconditioner->initialize(mg_data,
                                    underlying_operator->get_dof_handler_u(),
                                    underlying_operator->get_mapping(),
                                    vel_conv_diff_operator_data,
                                    underlying_operator->dirichlet_boundary,
                                    underlying_operator->get_fe_parameters());
      }
      else
      {
        multigrid_preconditioner_momentum.reset(new MyMultigridPreconditionerGMRESSmoother<dim,value_type,
                                                      VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>,
                                                      VelocityConvDiffOperatorData<dim> >());

        std_cxx11::shared_ptr<MyMultigridPreconditionerGMRESSmoother<dim,value_type,
                                VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>,
                                VelocityConvDiffOperatorData<dim> > >
          mg_preconditioner = std::dynamic_pointer_cast<MyMultigridPreconditionerGMRESSmoother<dim,value_type,
                                                          VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>,
                                                          VelocityConvDiffOperatorData<dim> > >(multigrid_preconditioner_momentum);

        mg_preconditioner->initialize(mg_data,
                                      underlying_operator->get_dof_handler_u(),
                                      underlying_operator->get_mapping(),
                                      vel_conv_diff_operator_data,
                                      underlying_operator->dirichlet_boundary,
                                      underlying_operator->get_fe_parameters());
      }

    }
  }

  void setup_iterative_solver_momentum()
  {
    AssertThrow(velocity_conv_diff_operator.get() != 0,
                ExcMessage("Setup of iterative solver for momentum preconditioner: Velocity convection-diffusion operator is uninitialized"));

    AssertThrow(multigrid_preconditioner_momentum.get() != 0,
                ExcMessage("Setup of iterative solver for momentum preconditioner: Multigrid preconditioner is uninitialized"));

    GMRESSolverData gmres_data;
    gmres_data.use_preconditioner = true;
    gmres_data.solver_tolerance_rel = preconditioner_data.rel_tol_solver_momentum_preconditioner;
    // TODO
    gmres_data.max_n_tmp_vectors = 300;

    solver_velocity_block.reset(new GMRESSolver<VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, value_type>,
                                                PreconditionerBase<value_type>,
                                                parallel::distributed::Vector<value_type> >(
        *velocity_conv_diff_operator,
        *multigrid_preconditioner_momentum,
        gmres_data));
  }

  void setup_multigrid_preconditioner_schur_complement()
  {
    if(preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Compatible)
    {
      // compatible discretization of Laplacian
      CompatibleLaplaceOperatorData<dim> compatible_laplace_operator_data;
      compatible_laplace_operator_data.dof_index_velocity = underlying_operator->get_dof_index_velocity();
      compatible_laplace_operator_data.dof_index_pressure = underlying_operator->get_dof_index_pressure();
      compatible_laplace_operator_data.gradient_operator_data = underlying_operator->get_gradient_operator_data();
      compatible_laplace_operator_data.divergence_operator_data = underlying_operator->get_divergence_operator_data();
      compatible_laplace_operator_data.periodic_face_pairs_level0 = underlying_operator->periodic_face_pairs;

      MultigridData mg_data = preconditioner_data.multigrid_data_schur_complement_preconditioner;

//      multigrid_preconditioner_schur_complement.reset(
//          new MyMultigridPreconditioner<dim,value_type,
//                CompatibleLaplaceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, Number>,
//                CompatibleLaplaceOperatorData<dim> >
//          (mg_data,
//           underlying_operator->get_dof_handler_p(),
//           underlying_operator->get_dof_handler_u(),
//           underlying_operator->get_mapping(),
//           compatible_laplace_operator_data));

      multigrid_preconditioner_schur_complement.reset(new
          MyMultigridPreconditionerCompatibleLaplace<dim,value_type,
            CompatibleLaplaceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, Number>,
            CompatibleLaplaceOperatorData<dim> > ());

      std_cxx11::shared_ptr<MyMultigridPreconditionerCompatibleLaplace<dim,value_type,
                              CompatibleLaplaceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, Number>,
                              CompatibleLaplaceOperatorData<dim> > >
        mg_preconditioner = std::dynamic_pointer_cast<
          MyMultigridPreconditionerCompatibleLaplace<dim,value_type,
            CompatibleLaplaceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, Number>,
            CompatibleLaplaceOperatorData<dim> > >(multigrid_preconditioner_schur_complement);

      mg_preconditioner->initialize(mg_data,
                                    underlying_operator->get_dof_handler_p(),
                                    underlying_operator->get_dof_handler_u(),
                                    underlying_operator->get_mapping(),
                                    compatible_laplace_operator_data);
    }
    else if(preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Classical)
    {
      // Geometric multigrid V-cycle performed on negative Laplace operator
      LaplaceOperatorData<dim> laplace_operator_data;
      laplace_operator_data.laplace_dof_index = underlying_operator->get_dof_index_pressure();
      laplace_operator_data.laplace_quad_index = underlying_operator->get_quad_index_pressure();
      laplace_operator_data.penalty_factor = 1.0;
      laplace_operator_data.neumann_boundaries = underlying_operator->get_dirichlet_boundary();
      laplace_operator_data.dirichlet_boundaries = underlying_operator->get_neumann_boundary();
      laplace_operator_data.periodic_face_pairs_level0 = underlying_operator->periodic_face_pairs;

      MultigridData mg_data = preconditioner_data.multigrid_data_schur_complement_preconditioner;

//      multigrid_preconditioner_schur_complement.reset(new MyMultigridPreconditioner<dim,value_type,
//                                                            LaplaceOperator<dim,Number>,
//                                                            LaplaceOperatorData<dim> >
//          (mg_data,
//           underlying_operator->get_dof_handler_p(),
//           underlying_operator->get_mapping(),
//           laplace_operator_data,
//           laplace_operator_data.dirichlet_boundaries));

      multigrid_preconditioner_schur_complement.reset(new MyMultigridPreconditioner<dim,value_type,
                                                            LaplaceOperator<dim,Number>,
                                                            LaplaceOperatorData<dim> > ());

      std_cxx11::shared_ptr<MyMultigridPreconditioner<dim,value_type,
                              LaplaceOperator<dim,Number>,
                              LaplaceOperatorData<dim> > >
        mg_preconditioner = std::dynamic_pointer_cast<MyMultigridPreconditioner<dim,value_type,
                                                        LaplaceOperator<dim,Number>,
                                                        LaplaceOperatorData<dim> > >(multigrid_preconditioner_schur_complement);

      mg_preconditioner->initialize(mg_data,
                                    underlying_operator->get_dof_handler_p(),
                                    underlying_operator->get_mapping(),
                                    laplace_operator_data,
                                    laplace_operator_data.dirichlet_boundaries);
    }
    else
    {
      AssertThrow(preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Classical ||
                  preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Compatible,
                  ExcMessage("Specified discretization of Laplacian for Schur-complement preconditioner is not available."));
    }
  }

  void setup_iterative_solver_schur_complement()
  {
    AssertThrow(multigrid_preconditioner_schur_complement.get() != 0,
                ExcMessage("Setup of iterative solver for Schur complement preconditioner: Multigrid preconditioner is uninitialized"));

    CGSolverData solver_data;
    solver_data.use_preconditioner = true;
    solver_data.solver_tolerance_rel = preconditioner_data.rel_tol_solver_schur_complement_preconditioner;

    if(preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Classical)
    {
      LaplaceOperatorData<dim> laplace_operator_data;
      laplace_operator_data.laplace_dof_index = underlying_operator->get_dof_index_pressure();
      laplace_operator_data.laplace_quad_index = underlying_operator->get_quad_index_pressure();
      laplace_operator_data.penalty_factor = 1.0;
      laplace_operator_data.neumann_boundaries = underlying_operator->get_dirichlet_boundary();
      laplace_operator_data.dirichlet_boundaries = underlying_operator->get_neumann_boundary();
      laplace_operator_data.periodic_face_pairs_level0 = underlying_operator->periodic_face_pairs;
      laplace_operator_classical.reset(new LaplaceOperator<dim>());
      laplace_operator_classical->reinit(
          underlying_operator->get_data(),
          underlying_operator->get_mapping(),
          laplace_operator_data);

      solver_pressure_block.reset(new CGSolver<LaplaceOperator<dim>,PreconditionerBase<value_type>,parallel::distributed::Vector<value_type> >(
          *laplace_operator_classical,
          *multigrid_preconditioner_schur_complement,
          solver_data));
    }
    else if(preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Compatible)
    {
      CompatibleLaplaceOperatorData<dim> compatible_laplace_operator_data;
      compatible_laplace_operator_data.dof_index_velocity = underlying_operator->get_dof_index_velocity();
      compatible_laplace_operator_data.dof_index_pressure = underlying_operator->get_dof_index_pressure();
      compatible_laplace_operator_data.gradient_operator_data = underlying_operator->get_gradient_operator_data();
      compatible_laplace_operator_data.divergence_operator_data = underlying_operator->get_divergence_operator_data();
      compatible_laplace_operator_data.periodic_face_pairs_level0 = underlying_operator->periodic_face_pairs;

      laplace_operator_compatible.reset(new CompatibleLaplaceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>());
      laplace_operator_compatible->initialize(
          underlying_operator->get_data(),
          compatible_laplace_operator_data,
          underlying_operator->gradient_operator,
          underlying_operator->divergence_operator,
          *underlying_operator->inverse_mass_matrix_operator);

      solver_pressure_block.reset(new CGSolver<CompatibleLaplaceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>,PreconditionerBase<value_type>,parallel::distributed::Vector<value_type> >(
          *laplace_operator_compatible,
          *multigrid_preconditioner_schur_complement,
          solver_data));
    }
  }

  void setup_pressure_convection_diffusion_operator()
  {
    // pressure convection-diffusion operator
    // a) mass matrix operator
    ScalarConvDiffOperators::MassMatrixOperatorData mass_matrix_operator_data;
    mass_matrix_operator_data.dof_index = underlying_operator->get_dof_index_pressure();
    mass_matrix_operator_data.quad_index = underlying_operator->get_quad_index_pressure();

    std_cxx11::shared_ptr<BoundaryDescriptorConvDiff<dim> > boundary_descriptor;
    boundary_descriptor.reset(new BoundaryDescriptorConvDiff<dim>());

    // for the pressure convection-diffusion operator the homogeneous operators (convective, diffusive) are applied,
    // so there is no need to specify functions for boundary conditions since they will not be used (must not be used)
    // -> use ConstantFunction as dummy, initialized with NAN in order to detect a possible incorrect access to boundary values
    std_cxx11::shared_ptr<Function<dim> > dummy;
    dummy.reset(new ConstantFunction<dim>(NAN));

    // set boundary ID's for pressure convection-diffusion operator

    // Dirichlet BC for velocity -> Neumann BC for pressure
    for (typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::
         const_iterator it = underlying_operator->boundary_descriptor_velocity->dirichlet_bc.begin();
         it != underlying_operator->boundary_descriptor_velocity->dirichlet_bc.end(); ++it)
    {
      boundary_descriptor->neumann_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                              (it->first, dummy));
    }
    // Neumann BC for velocity -> Dirichlet BC for pressure
    for (typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::
         const_iterator it = underlying_operator->boundary_descriptor_velocity->neumann_bc.begin();
         it != underlying_operator->boundary_descriptor_velocity->neumann_bc.end(); ++it)
    {
      boundary_descriptor->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                (it->first, dummy));
    }

    // b) diffusive operator
    ScalarConvDiffOperators::DiffusiveOperatorData<dim> diffusive_operator_data;
    diffusive_operator_data.dof_index = underlying_operator->get_dof_index_pressure();
    diffusive_operator_data.quad_index = underlying_operator->get_quad_index_pressure();
    diffusive_operator_data.IP_factor = underlying_operator->param.IP_factor_viscous;
    diffusive_operator_data.bc = boundary_descriptor;
    // TODO: the pressure convection-diffusion operator is initialized with constant viscosity, in case of varying viscosities
    // the pressure convection-diffusion operator (the diffusive operator of the pressure convection-diffusion operator)
    // has to be updated before applying this preconditioner
    diffusive_operator_data.diffusivity = underlying_operator->get_viscosity();

    // c) convective operator
    ScalarConvDiffOperators::ConvectiveOperatorDataDiscontinuousVelocity<dim> convective_operator_data;
    convective_operator_data.dof_index = underlying_operator->get_dof_index_pressure();
    convective_operator_data.dof_index_velocity = underlying_operator->get_dof_index_velocity();
    convective_operator_data.quad_index = underlying_operator->get_quad_index_pressure();
    convective_operator_data.bc = boundary_descriptor;

    PressureConvectionDiffusionOperatorData<dim> pressure_convection_diffusion_operator_data;
    pressure_convection_diffusion_operator_data.mass_matrix_operator_data = mass_matrix_operator_data;
    pressure_convection_diffusion_operator_data.diffusive_operator_data = diffusive_operator_data;
    pressure_convection_diffusion_operator_data.convective_operator_data = convective_operator_data;
    if(underlying_operator->param.problem_type == ProblemType::Unsteady)
      pressure_convection_diffusion_operator_data.unsteady_problem = true;
    else
      pressure_convection_diffusion_operator_data.unsteady_problem = false;
    pressure_convection_diffusion_operator_data.convective_problem = underlying_operator->nonlinear_problem_has_to_be_solved();

    pressure_convection_diffusion_operator.reset(new PressureConvectionDiffusionOperator<dim, fe_degree_p, fe_degree, value_type>(
      underlying_operator->mapping,
      underlying_operator->get_data(),
      pressure_convection_diffusion_operator_data));

    if(underlying_operator->param.problem_type == ProblemType::Unsteady)
      pressure_convection_diffusion_operator->set_scaling_factor_time_derivative_term(&underlying_operator->scaling_factor_time_derivative_term);
  }

  void update_preconditioner() const
  {
    if(preconditioner_data.momentum_preconditioner == MomentumPreconditioner::VelocityConvectionDiffusion)
    {
      if(!use_gmres_smoother)
      {
      std_cxx11::shared_ptr<MyMultigridPreconditionerVelocityConvectionDiffusion<dim,value_type,
                              VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>,
                              VelocityConvDiffOperatorData<dim> > >
        mg_preconditioner = std::dynamic_pointer_cast<MyMultigridPreconditionerVelocityConvectionDiffusion<dim,value_type,
                                                        VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>,
                                                        VelocityConvDiffOperatorData<dim> > >(multigrid_preconditioner_momentum);

      mg_preconditioner->update_mg_matrix(*underlying_operator->get_velocity_linearization(),
                                          underlying_operator->evaluation_time);
      mg_preconditioner->update_smoother();
      mg_preconditioner->update_coarse_solver();
      }
      else
      {
        std_cxx11::shared_ptr<MyMultigridPreconditionerGMRESSmoother<dim,value_type,
                                VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>,
                                VelocityConvDiffOperatorData<dim> > >
          mg_preconditioner = std::dynamic_pointer_cast<MyMultigridPreconditionerGMRESSmoother<dim,value_type,
                                                          VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>,
                                                          VelocityConvDiffOperatorData<dim> > >(multigrid_preconditioner_momentum);

        mg_preconditioner->update_mg_matrix(*underlying_operator->get_velocity_linearization(),
                                            underlying_operator->evaluation_time);
        mg_preconditioner->update_smoother();
        mg_preconditioner->update_coarse_solver();
      }
    }
  }

  void vmult_velocity_block(parallel::distributed::Vector<value_type>       &dst,
                            const parallel::distributed::Vector<value_type> &src) const
  {
    if(preconditioner_data.momentum_preconditioner == MomentumPreconditioner::None)
    {
      dst = src;
    }
    else if(preconditioner_data.momentum_preconditioner == MomentumPreconditioner::InverseMassMatrix)
    {
      // use the inverse mass matrix as an approximation to the momentum block
      // this approach is expected to perform well for small time steps and/or small viscosities
      inv_mass_matrix_preconditioner_momentum->vmult(dst,src);
      dst *= 1./underlying_operator->get_scaling_factor_time_derivative_term();
    }
    else if(preconditioner_data.momentum_preconditioner == MomentumPreconditioner::VelocityDiffusion ||
            preconditioner_data.momentum_preconditioner == MomentumPreconditioner::VelocityConvectionDiffusion)
    {
      if(preconditioner_data.solver_momentum_preconditioner == SolverMomentumPreconditioner::GeometricMultigridVCycle)
      {
        // perform one geometric multigrid V-cylce for the Helmholtz operator or viscous operator (in case of steady-state problem)
        multigrid_preconditioner_momentum->vmult(dst,src);
      }
      else if(preconditioner_data.solver_momentum_preconditioner == SolverMomentumPreconditioner::GeometricMultigridGMRES)
      {
        velocity_conv_diff_operator->set_velocity_linearization(underlying_operator->get_velocity_linearization());
        velocity_conv_diff_operator->set_evaluation_time(underlying_operator->evaluation_time);

        // CheckMultigrid
//        std_cxx11::shared_ptr<MyMultigridPreconditionerVelocityConvectionDiffusion<dim,value_type,
//                                VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>,
//                                VelocityConvDiffOperatorData<dim> > >
//          my_preconditioner = std::dynamic_pointer_cast<MyMultigridPreconditionerVelocityConvectionDiffusion<dim,value_type,
//                                                          VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>,
//                                                          VelocityConvDiffOperatorData<dim> > >(multigrid_preconditioner_momentum);
//
//        CheckMultigrid<dim, value_type,
//                       VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, value_type>,
//                       MyMultigridPreconditionerVelocityConvectionDiffusion<dim,value_type,
//                         VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>,
//                         VelocityConvDiffOperatorData<dim> > >
//          check_multigrid(velocity_conv_diff_operator,my_preconditioner);
//
//        check_multigrid.check();
        // CheckMultigrid

        // solve velocity convection-diffusion problem using GMRES preconditioned by geometric multigrid (for Helmholtz operator
        // or viscous operator (convective operator is not considered in the multigrid preconditioner))
        dst = 0.0;
        unsigned int iterations_velocity_block = solver_velocity_block->solve(dst,src);

        //ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
        //pcout<<"Number of GMRES iterations = "<<iterations_velocity_block<<std::endl;
      }
      else
      {
        AssertThrow(preconditioner_data.solver_momentum_preconditioner == SolverMomentumPreconditioner::GeometricMultigridVCycle ||
                    preconditioner_data.solver_momentum_preconditioner == SolverMomentumPreconditioner::GeometricMultigridGMRES,
                    ExcMessage("Specified solver for velocity/momentum preconditioner is not implemented."));
      }

    }
    else
    {
      AssertThrow(preconditioner_data.momentum_preconditioner == MomentumPreconditioner::None ||
                  preconditioner_data.momentum_preconditioner == MomentumPreconditioner::InverseMassMatrix ||
                  preconditioner_data.momentum_preconditioner == MomentumPreconditioner::VelocityDiffusion ||
                  preconditioner_data.momentum_preconditioner == MomentumPreconditioner::VelocityConvectionDiffusion,
                  ExcMessage("Specified preconditioner for velocity/momentum block is not implemented."));
    }
  }

  void vmult_pressure_block(parallel::distributed::Vector<value_type>       &dst,
                            const parallel::distributed::Vector<value_type> &src) const
  {
    if(preconditioner_data.schur_complement_preconditioner == SchurComplementPreconditioner::None)
    {
      // No preconditioner for Schur-complement block
      dst = src;
    }
    else if(preconditioner_data.schur_complement_preconditioner == SchurComplementPreconditioner::InverseMassMatrix)
    {
      // - S^{-1} = nu M_p^{-1}
      inv_mass_matrix_preconditioner_schur_complement->vmult(dst,src);
      dst *=  underlying_operator->get_viscosity();
    }
    else if(preconditioner_data.schur_complement_preconditioner == SchurComplementPreconditioner::LaplaceOperator)
    {
      // -S^{-1} = 1/dt  (-L)^{-1}
      apply_inverse_negative_laplace_operator(dst,src);
      dst *=  underlying_operator->get_scaling_factor_time_derivative_term();
    }
    else if(preconditioner_data.schur_complement_preconditioner == SchurComplementPreconditioner::CahouetChabard)
    {
      // - S^{-1} = nu M_p^{-1} + 1/dt (-L)^{-1}

      // I. 1/dt (-L)^{-1}
      apply_inverse_negative_laplace_operator(dst,src);
      dst *=  underlying_operator->get_scaling_factor_time_derivative_term();

      // II. M_p^{-1}, apply inverse pressure mass matrix to src-vector and store the result in a temporary vector
      inv_mass_matrix_preconditioner_schur_complement->vmult(tmp_scp_pressure,src);

      // III. add temporary vector scaled by viscosity
      dst.add(underlying_operator->get_viscosity(),tmp_scp_pressure);
    }
    else if(preconditioner_data.schur_complement_preconditioner == SchurComplementPreconditioner::Elman)
    {
      if(preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Classical)
      {
        // -S^{-1} = - (BB^T)^{-1} (-B A B^T) (BB^T)^{-1}

        // I. (BB^T)^{-1} -> apply inverse negative Laplace operator (classical discretization), (-L)^{-1}
        apply_inverse_negative_laplace_operator(dst,src);

        // II. (-B A B^T)
        // II.a) B^T
        underlying_operator->gradient_operator.apply(tmp_scp_velocity,dst);

        // II.b) A = 1/dt * mass matrix  +  viscous term  +  linearized convective term
        velocity_conv_diff_operator->set_velocity_linearization(underlying_operator->get_velocity_linearization());
        velocity_conv_diff_operator->set_evaluation_time(underlying_operator->evaluation_time);
        velocity_conv_diff_operator->vmult(tmp_scp_velocity_2,tmp_scp_velocity);

        // II.c) -B
        underlying_operator->divergence_operator.apply(tmp_scp_pressure,tmp_scp_velocity_2);

        // III. -(BB^T)^{-1}
        // III.a) apply inverse negative Laplace operator (classical discretization), (-L)^{-1}
        apply_inverse_negative_laplace_operator(dst,tmp_scp_pressure);
        // III.b) minus sign
        dst *= -1.0;
      }
      else if(preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Compatible)
      {
        // -S^{-1} = - (BM^{-1}B^T)^{-1} (-B M^{-1} A M^{-1} B^T) (BM^{-1}B^T)^{-1}

        // I. (BM^{-1}B^T)^{-1} -> apply inverse negative Laplace operator (compatible discretization), (-L)^{-1}
        apply_inverse_negative_laplace_operator(dst,src);


        // II. (-B M^{-1} A M^{-1} B^T)
        // II.a) B^T
        underlying_operator->gradient_operator.apply(tmp_scp_velocity,dst);

        // II.b) M^{-1}
        inv_mass_matrix_preconditioner_schur_complement->vmult(tmp_scp_velocity,tmp_scp_velocity);

        // II.c) A = 1/dt * mass matrix + viscous term + linearized convective term
        velocity_conv_diff_operator->set_velocity_linearization(underlying_operator->get_velocity_linearization());
        velocity_conv_diff_operator->set_evaluation_time(underlying_operator->evaluation_time);
        velocity_conv_diff_operator->vmult(tmp_scp_velocity_2,tmp_scp_velocity);

        // II.d) M^{-1}
        inv_mass_matrix_preconditioner_schur_complement->vmult(tmp_scp_velocity_2,tmp_scp_velocity_2);

        // II.e) -B
        underlying_operator->divergence_operator.apply(tmp_scp_pressure,tmp_scp_velocity_2);


        // III. -(BM^{-1}B^T)^{-1}
        // III.a) apply inverse negative Laplace operator (compatible discretization), (-L)^{-1}
        apply_inverse_negative_laplace_operator(dst,tmp_scp_pressure);
        // III.b) minus sign
        dst *= -1.0;
      }
    }
    else if(preconditioner_data.schur_complement_preconditioner == SchurComplementPreconditioner::PressureConvectionDiffusion)
    {
      // -S^{-1} = M_p^{-1} A_p (-L)^{-1}

      // I. inverse, negative Laplace operator (-L)^{-1}
      apply_inverse_negative_laplace_operator(tmp_scp_pressure,src);

      // II. pressure convection diffusion operator A_p
      pressure_convection_diffusion_operator->apply(dst,tmp_scp_pressure,underlying_operator->get_velocity_linearization());

      // III. inverse pressure mass matrix M_p^{-1}
      inv_mass_matrix_preconditioner_schur_complement->vmult(dst,dst);
    }
    else
    {
      AssertThrow(preconditioner_data.schur_complement_preconditioner == SchurComplementPreconditioner::None ||
                  preconditioner_data.schur_complement_preconditioner == SchurComplementPreconditioner::InverseMassMatrix ||
                  preconditioner_data.schur_complement_preconditioner == SchurComplementPreconditioner::LaplaceOperator ||
                  preconditioner_data.schur_complement_preconditioner == SchurComplementPreconditioner::CahouetChabard ||
                  preconditioner_data.schur_complement_preconditioner == SchurComplementPreconditioner::Elman ||
                  preconditioner_data.schur_complement_preconditioner == SchurComplementPreconditioner::PressureConvectionDiffusion,
                  ExcMessage("Specified preconditioner for pressure/Schur complement block is not implemented."));
    }
  }

  void apply_inverse_negative_laplace_operator(parallel::distributed::Vector<value_type>       &dst,
                                               parallel::distributed::Vector<value_type> const &src) const
  {
    if(preconditioner_data.solver_schur_complement_preconditioner == SolverSchurComplementPreconditioner::GeometricMultigridVCycle)
    {
      // perform one multigrid V-cycle in order to approximately invert the negative Laplace operator (classical or compatible)
      multigrid_preconditioner_schur_complement->vmult(dst,src);
    }
    else if(preconditioner_data.solver_schur_complement_preconditioner == SolverSchurComplementPreconditioner::GeometricMultigridPCG)
    {
      // solve a linear system of equations for negative Laplace operator to given (relative) tolerance using the PCG method
      parallel::distributed::Vector<value_type> const *pointer_to_src = &src;
      if(underlying_operator->param.pure_dirichlet_bc == true)
      {
        tmp_projection_vector = src;
        if(preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Classical)
          laplace_operator_classical->apply_nullspace_projection(tmp_projection_vector);
        else if(preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Compatible)
          laplace_operator_compatible->apply_nullspace_projection(tmp_projection_vector);
        pointer_to_src = &tmp_projection_vector;
      }
      dst = 0.0;
      solver_pressure_block->solve(dst,*pointer_to_src);
    }
  }

private:
  DGNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> *underlying_operator;
  BlockPreconditionerData preconditioner_data;

  // preconditioner velocity/momentum block
  std_cxx11::shared_ptr<PreconditionerBase<value_type> > multigrid_preconditioner_momentum;
  std_cxx11::shared_ptr<PreconditionerBase<value_type> > inv_mass_matrix_preconditioner_momentum;

  std_cxx11::shared_ptr<VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, value_type> > velocity_conv_diff_operator;
  std_cxx11::shared_ptr<IterativeSolverBase<parallel::distributed::Vector<value_type> > > solver_velocity_block;

  // preconditioner pressure/Schur-complement block
  std_cxx11::shared_ptr<PreconditionerBase<value_type> > multigrid_preconditioner_schur_complement;
  std_cxx11::shared_ptr<PreconditionerBase<value_type> > inv_mass_matrix_preconditioner_schur_complement;

  std_cxx11::shared_ptr<PressureConvectionDiffusionOperator<dim, fe_degree_p, fe_degree, value_type> > pressure_convection_diffusion_operator;
  std_cxx11::shared_ptr<LaplaceOperator<dim> > laplace_operator_classical;
  std_cxx11::shared_ptr<CompatibleLaplaceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type> > laplace_operator_compatible;
  std_cxx11::shared_ptr<IterativeSolverBase<parallel::distributed::Vector<value_type> > > solver_pressure_block;

  // temporary vectors that are necessary when using preconditioners of block-triangular type
  parallel::distributed::Vector<value_type> mutable vec_tmp_pressure;
  parallel::distributed::Vector<value_type> mutable vec_tmp_velocity;
  parallel::distributed::Vector<value_type> mutable vec_tmp_velocity_2;

  // temporary vectors that are necessary when applying the Schur-complement preconditioner (scp)
  parallel::distributed::Vector<value_type> mutable tmp_scp_pressure;
  parallel::distributed::Vector<value_type> mutable tmp_scp_velocity, tmp_scp_velocity_2;

  // temporary vector that is needed if negative Laplace operator is inverted exactly
  // and if a problem with pure Dirichlet BC's is considered
  parallel::distributed::Vector<value_type> mutable tmp_projection_vector;

  // TODO
  bool const use_gmres_smoother;
};


#endif /* INCLUDE_PRECONDITIONERNAVIERSTOKES_H_ */
