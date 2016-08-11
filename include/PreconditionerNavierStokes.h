/*
 * PreconditionerNavierStokes.h
 *
 *  Created on: Jun 30, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_PRECONDITIONERNAVIERSTOKES_H_
#define INCLUDE_PRECONDITIONERNAVIERSTOKES_H_

#include "Preconditioner.h"
#include "CompatibleLaplaceOperator.h"
#include "PressureConvectionDiffusionOperator.h"
#include "VelocityConvDiffOperator.h"
#include "IterativeSolvers.h"


// forward declaration
template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
class DGNavierStokesCoupled;

template<typename value_type>
class PreconditionerNavierStokesBase
{
public:
  virtual ~PreconditionerNavierStokesBase(){}

  virtual void vmult(parallel::distributed::BlockVector<value_type>        &dst,
                     const parallel::distributed::BlockVector<value_type>  &src) const = 0;
};

struct PreconditionerDataLinearSolver
{
  PreconditionerLinearizedNavierStokes preconditioner_type;
  PreconditionerMomentum preconditioner_momentum;
  PreconditionerSchurComplement preconditioner_schur_complement;
  DiscretizationOfLaplacian discretization_of_laplacian;
};

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
class BlockPreconditionerNavierStokes : public PreconditionerNavierStokesBase<value_type>
{
public:
  typedef float Number;

  BlockPreconditionerNavierStokes(DGNavierStokesCoupled<dim, fe_degree,
                                    fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> *underlying_operator_in,
                                  PreconditionerDataLinearSolver const                 &preconditioner_data_in)
  {
    underlying_operator = underlying_operator_in;
    preconditioner_data = preconditioner_data_in;

    /****** preconditioner velocity/momentum block ****************/
    if(preconditioner_data.preconditioner_momentum == PreconditionerMomentum::InverseMassMatrix)
    {
      // inverse mass matrix
      preconditioner_momentum.reset(new InverseMassMatrixPreconditioner<dim,fe_degree,value_type>(
        underlying_operator->get_data(),
        underlying_operator->get_dof_index_velocity(),
        underlying_operator->get_quad_index_velocity_linear()));
    }
    else if(preconditioner_data.preconditioner_momentum == PreconditionerMomentum::GeometricMultigrid)
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
        helmholtz_operator_data.mass_matrix_coefficient = underlying_operator->get_scaling_factor_time_derivative_term();
        helmholtz_operator_data.periodic_face_pairs_level0 = underlying_operator->get_periodic_face_pairs();

        // currently use default parameters of MultigridData!
        MultigridData mg_data_helmholtz;

        preconditioner_momentum.reset(new MyMultigridPreconditioner<dim,value_type,HelmholtzOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>, HelmholtzOperatorData<dim> >(
            mg_data_helmholtz,
            underlying_operator->get_dof_handler_u(),
            underlying_operator->get_mapping(),
            helmholtz_operator_data,
            underlying_operator->dirichlet_boundary,
            underlying_operator->get_fe_parameters()));
      }
      // Steady problem -> consider viscous operator
      else if(underlying_operator->param.problem_type == ProblemType::Steady)
      {
        // Geometric multigrid V-cycle performed on viscous operator
        ViscousOperatorData<dim> viscous_operator_data;
        viscous_operator_data = underlying_operator->get_viscous_operator_data();
        // currently use default parameters of MultigridData!
        MultigridData mg_data;
        preconditioner_momentum.reset(new MyMultigridPreconditioner<dim,value_type,ViscousOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>, ViscousOperatorData<dim> >(
            mg_data,
            underlying_operator->get_dof_handler_u(),
            underlying_operator->get_mapping(),
            viscous_operator_data,
            underlying_operator->dirichlet_boundary,
            underlying_operator->get_fe_parameters()));
      }

      // solve a linear system of equations for the velocity block instead of applying one multigrid V-cycle
      if(true)
      {
        VelocityConvDiffOperatorData<dim> velocity_conv_diff_operator_data;
        // copy operator_data of basic operators
        velocity_conv_diff_operator_data.mass_matrix_operator_data = underlying_operator->mass_matrix_operator_data;
        velocity_conv_diff_operator_data.viscous_operator_data = underlying_operator->viscous_operator_data;
        velocity_conv_diff_operator_data.convective_operator_data = underlying_operator->convective_operator_data;
        // unsteady problem ?
        if(underlying_operator->param.problem_type == ProblemType::Unsteady)
          velocity_conv_diff_operator_data.unsteady_problem = true;
        else
          velocity_conv_diff_operator_data.unsteady_problem = false;
        // convective problem ?
        if(underlying_operator->nonlinear_problem_has_to_be_solved() == true)
          velocity_conv_diff_operator_data.convective_problem = true;
        else
          velocity_conv_diff_operator_data.convective_problem = false;

        velocity_conv_diff_operator.reset(new VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, value_type>());
        velocity_conv_diff_operator->initialize(
            underlying_operator->get_data(),
            velocity_conv_diff_operator_data,
            underlying_operator->mass_matrix_operator,
            underlying_operator->viscous_operator,
            underlying_operator->convective_operator);

        // set scaling_factor_time_derivative_term
        velocity_conv_diff_operator->set_scaling_factor_time_derivative_term(&underlying_operator->scaling_factor_time_derivative_term);

        GMRESSolverData gmres_data;
        gmres_data.use_preconditioner = true;
        gmres_data.solver_tolerance_rel = 1.e-6;

        solver_velocity_block.reset(new GMRESSolver<VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, value_type>,
                                                    PreconditionerBase<value_type>,
                                                    parallel::distributed::Vector<value_type> >(
            *velocity_conv_diff_operator,
            *preconditioner_momentum,
            gmres_data));
      }
    }
    /****** preconditioner velocity/momentum block ****************/

    /****** preconditioner pressure/Schur-complement block ********/
    if(preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::InverseMassMatrix)
    {
      // inverse mass matrix
      preconditioner_schur_complement_inv_mass_matrix.reset(new InverseMassMatrixPreconditioner<dim,fe_degree_p,value_type,1>(
          underlying_operator->get_data(),
          underlying_operator->get_dof_index_pressure(),
          underlying_operator->get_quad_index_pressure()));
    }
    else if(preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::GeometricMultigrid)
    {
      // multigrid for negative Laplace operator (classical or compatible discretization)
      setup_multigrid_schur_complement();
    }
    else if(preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::CahouetChabard)
    {
      AssertThrow(underlying_operator->param.problem_type == ProblemType::Unsteady,
          ExcMessage("CahouetChabard preconditioner only makes sense for unsteady problems."));

      // multigrid for negative Laplace operator (classical or compatible discretization)
      setup_multigrid_schur_complement();

      // inverse mass matrix to also include the part of the preconditioner that is beneficial when using large time steps
      // and large viscosities.
      preconditioner_schur_complement_inv_mass_matrix.reset(new InverseMassMatrixPreconditioner<dim,fe_degree_p,value_type,1>(
        underlying_operator->get_data(),
        underlying_operator->get_dof_index_pressure(),
        underlying_operator->get_quad_index_pressure()));

      // initialize_temp vector
      underlying_operator->initialize_vector_pressure(temp);

    }
    else if(preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::Elman)
    {
      // multigrid for negative Laplace operator (classical or compatible discretization)
      setup_multigrid_schur_complement();

      if(preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Compatible)
      {
        // -S^{-1} = - (BM^{-1}B^T)^{-1} (-B M^{-1} A M^{-1} B^T) (BM^{-1}B^T)^{-1}
        // --> inverse velocity mass matrix needed for inner factor
//        preconditioner_schur_complement_inv_mass_matrix.reset(new InverseMassMatrixPreconditioner<dim,fe_degree_p,value_type,dim>(
//                                                    underlying_operator->get_data(),
//                                                    underlying_operator->get_dof_index_velocity(),
//                                                    underlying_operator->get_quad_index_velocity_linear()));

        // -S^{-1} = - (BM^{-1}B^T)^{-1} (- M_p^{-1} B A B^T M_p^{-1}) (BM^{-1}B^T)^{-1}
        // --> inverse pressure mass matrix needed for inner factor
        preconditioner_schur_complement_inv_mass_matrix.reset(new InverseMassMatrixPreconditioner<dim,fe_degree_p,value_type,1>(
          underlying_operator->get_data(),
          underlying_operator->get_dof_index_pressure(),
          underlying_operator->get_quad_index_pressure()));
      }

      underlying_operator->initialize_vector_velocity(temp_velocity);
      underlying_operator->initialize_vector_velocity(temp_velocity_2);
    }
    else if(preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::PressureConvectionDiffusion)
    {
      // -S^{-1} = M_p^{-1} A_p (-L)^{-1}

      // I. multigrid for negative Laplace operator (classical or compatible discretization)
      setup_multigrid_schur_complement();

      if(true)
      {
        LaplaceOperatorData<dim> laplace_operator_data;
        laplace_operator_data.laplace_dof_index = underlying_operator->get_dof_index_pressure();
        laplace_operator_data.laplace_quad_index = underlying_operator->get_quad_index_pressure();
        laplace_operator_data.penalty_factor = 1.0;
        laplace_operator_data.neumann_boundaries = underlying_operator->get_dirichlet_boundary();
        laplace_operator_data.dirichlet_boundaries = underlying_operator->get_neumann_boundary();
        laplace_operator_data.periodic_face_pairs_level0 = underlying_operator->get_periodic_face_pairs();
        laplace_operator.reset(new LaplaceOperator<dim>());
        laplace_operator->reinit(
            underlying_operator->get_data(),
            underlying_operator->get_mapping(),
            laplace_operator_data);

        CGSolverData solver_data;
        solver_data.use_preconditioner = true;
        solver_pressure_block.reset(new CGSolver<LaplaceOperator<dim>,PreconditionerBase<value_type>,parallel::distributed::Vector<value_type> >(
            *laplace_operator,
            *preconditioner_schur_complement,
            solver_data));
      }

      // II. pressure convection-diffusion operator
      // II.a) mass matrix operator
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

      // II.b) diffusive operator
      ScalarConvDiffOperators::DiffusiveOperatorData<dim> diffusive_operator_data;
      diffusive_operator_data.dof_index = underlying_operator->get_dof_index_pressure();
      diffusive_operator_data.quad_index = underlying_operator->get_quad_index_pressure();
      diffusive_operator_data.IP_formulation = underlying_operator->param.IP_formulation_viscous;
      diffusive_operator_data.IP_factor = underlying_operator->param.IP_factor_viscous;
      diffusive_operator_data.bc = boundary_descriptor;
      // TODO: the pressure convection-diffusion operator is initialized with constant viscosity, in case of varying viscosities
      // the pressure convection-diffusion operator (the diffusive operator of the pressure convection-diffusion operator)
      // has to be updated before applying this preconditioner
      diffusive_operator_data.diffusivity = underlying_operator->get_viscosity();

      // II.c) convective operator
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

      // III. inverse pressure mass matrix
      preconditioner_schur_complement_inv_mass_matrix.reset(new InverseMassMatrixPreconditioner<dim,fe_degree_p,value_type,1>(
        underlying_operator->get_data(),
        underlying_operator->get_dof_index_pressure(),
        underlying_operator->get_quad_index_pressure()));

      // initialize_temp vector
      underlying_operator->initialize_vector_pressure(temp);
    }
    /****** preconditioner pressure/Schur-complement block ********/

    if(preconditioner_data.preconditioner_type == PreconditionerLinearizedNavierStokes::BlockTriangularFactorization)
    {
      underlying_operator->initialize_vector_velocity(temp_velocity);
    }
  }

  void setup_multigrid_schur_complement()
  {
    if(preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Compatible)
    {
      // compatible discretization of Laplacian
      CompatibleLaplaceOperatorData<dim> compatible_laplace_operator_data;
      compatible_laplace_operator_data.dof_index_velocity = underlying_operator->get_dof_index_velocity();
      compatible_laplace_operator_data.dof_index_pressure = underlying_operator->get_dof_index_pressure();
      compatible_laplace_operator_data.gradient_operator_data = underlying_operator->get_gradient_operator_data();
      compatible_laplace_operator_data.divergence_operator_data = underlying_operator->get_divergence_operator_data();
      compatible_laplace_operator_data.periodic_face_pairs_level0 = underlying_operator->get_periodic_face_pairs();

      // currently use default parameters of MultigridData!
      MultigridData mg_data_compatible_pressure;

      preconditioner_schur_complement.reset(new MyMultigridPreconditioner<dim,value_type,
                CompatibleLaplaceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, Number>,
                CompatibleLaplaceOperatorData<dim> >
          (mg_data_compatible_pressure,
           underlying_operator->get_dof_handler_p(),
           underlying_operator->get_dof_handler_u(),
           underlying_operator->get_mapping(),
           compatible_laplace_operator_data));
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
      laplace_operator_data.periodic_face_pairs_level0 = underlying_operator->get_periodic_face_pairs();

      // currently use default parameters of MultigridData!
      MultigridData mg_data_pressure;

      preconditioner_schur_complement.reset(new MyMultigridPreconditioner<dim,value_type,LaplaceOperator<dim,Number>, LaplaceOperatorData<dim> >
                                                (mg_data_pressure,
                                                 underlying_operator->get_dof_handler_p(),
                                                 underlying_operator->get_mapping(),
                                                 laplace_operator_data,
                                                 laplace_operator_data.dirichlet_boundaries));
    }
    else
    {
      AssertThrow(preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Classical ||
                  preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Compatible,
                  ExcMessage("Specified discretization of Laplacian for Schur-complement preconditioner is not available."));
    }
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

      dst.block(0) = src.block(0);
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

      // make sure that vmult_velocity_block(vec1,vec2) works if vec1 = vec2!
      vmult_velocity_block(dst.block(0),dst.block(0));
    }
    else if(preconditioner_data.preconditioner_type == PreconditionerLinearizedNavierStokes::BlockTriangularFactorization)
    {
      /*
      *        / A^{-1}  0 \
      *  dst = |           | * src
      *        \   0     I /
      */
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
      // apply preconditioner for pressure/Schur-complement block
      // make sure that vmult_pressure_block(vec1,vec2) works if vec1 = vec2!
      vmult_pressure_block(dst.block(1),dst.block(1));

      /*
      *        / I  - A^{-1} B^{T} \
      *  dst = |                   | * dst
      *        \ 0          I      /
      */
      // temp_velocity = B^{T} * dst(1)
      underlying_operator->gradient_operator.apply(temp_velocity,dst.block(1));

      // temp_velocity = A^{-1} * temp_velocity
      // make sure that vmult_velocity_block(vec1,vec2) works if vec1 = vec2!
      vmult_velocity_block(temp_velocity,temp_velocity);

      // dst(0) = dst(0) - temp_velocity
      dst.block(0).add(-1.0,temp_velocity);
    }
    else
    {
      AssertThrow(preconditioner_data.preconditioner_type == PreconditionerLinearizedNavierStokes::BlockDiagonal ||
                  preconditioner_data.preconditioner_type == PreconditionerLinearizedNavierStokes::BlockTriangular ||
                  preconditioner_data.preconditioner_type == PreconditionerLinearizedNavierStokes::BlockTriangularFactorization,
                  ExcMessage("Specified preconditioner for linearized Navier-Stokes problem is not implemented."));
    }
  }

  void vmult_velocity_block(parallel::distributed::Vector<value_type>       &dst,
                            const parallel::distributed::Vector<value_type> &src) const
  {
    if(preconditioner_data.preconditioner_momentum == PreconditionerMomentum::None)
    {
      dst = src;
    }
    else if(preconditioner_data.preconditioner_momentum == PreconditionerMomentum::InverseMassMatrix)
    {
      // use the inverse mass matrix as an approximation to the momentum block
      // this approach is expected to perform well for small time steps and/or small viscosities
      preconditioner_momentum->vmult(dst,src);
      dst *= 1./underlying_operator->get_scaling_factor_time_derivative_term();
    }
    else if(preconditioner_data.preconditioner_momentum == PreconditionerMomentum::GeometricMultigrid)
    {
      //TODO multigrid vs "exact" solution of velocity equation
      if(true)
      {
        // perform one multigrid V-cylce on the Helmholtz system or viscous operator (in case of steady-state problem)
        // this approach is expected to perform well for large time steps and/or large viscosities
        preconditioner_momentum->vmult(dst,src);
      }
      else
      {
        velocity_conv_diff_operator->set_velocity_linearization(underlying_operator->get_velocity_linearization());
        velocity_conv_diff_operator->set_evaluation_time(underlying_operator->time + underlying_operator->time_step);

        //TODO
        parallel::distributed::Vector<value_type> rhs(src);
        solver_velocity_block->solve(dst,rhs);
      }
    }
    else
    {
      AssertThrow(preconditioner_data.preconditioner_momentum == PreconditionerMomentum::None ||
                  preconditioner_data.preconditioner_momentum == PreconditionerMomentum::InverseMassMatrix ||
                  preconditioner_data.preconditioner_momentum == PreconditionerMomentum::GeometricMultigrid,
                  ExcMessage("Specified preconditioner for velocity/momentum block is not implemented."));
    }
  }

  void vmult_pressure_block(parallel::distributed::Vector<value_type>       &dst,
                            const parallel::distributed::Vector<value_type> &src) const
  {
    if(preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::None)
    {
      // no preconditioner for Schur complement block
      dst = src;
    }
    else if(preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::InverseMassMatrix)
    {
      // use the inverse mass matrix as an approximation to the inverse Schur complement
      // this approach is expected to perform well for large time steps and/or large viscosities
      preconditioner_schur_complement_inv_mass_matrix->vmult(dst,src);
      dst *=  underlying_operator->get_viscosity();
    }
    else if(preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::GeometricMultigrid)
    {
      // perform one multigrid V-cycle on the pressure Poisson system as an approximation to the
      // inverse Schur-complement
      // this approach is expected to perform well for small time steps and/or small viscosities
      preconditioner_schur_complement->vmult(dst,src);
      dst *=  underlying_operator->get_scaling_factor_time_derivative_term();
    }
    else if(preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::CahouetChabard)
    {
      // I. apply inverse pressure mass matrix to dst-vector and store the result in a temporary vector
      preconditioner_schur_complement_inv_mass_matrix->vmult(temp,src);

      // II. perform one multigrid V-cycle on the pressure Poisson operator and scale by "inverse time step size"
      // Note that the sequence of instructions I.,II. may not be reversed because this function may be called with
      // two identical parameters, i.e., vmult_pressure_block(vector, vector);
      // Then, the following instruction invalidates the src-vector (which is in this case equal to dst)
      preconditioner_schur_complement->vmult(dst,src);
      // N.B.: from now on, do not use src vector because src is invalid!
      dst *=  underlying_operator->get_scaling_factor_time_derivative_term();

      // III. add temporary vector scaled by viscosity
      dst.add(underlying_operator->get_viscosity(),temp);
    }
    else if(preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::Elman)
    {
      if(preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Classical)
      {
        // -S^{-1} = - (BB^T)^{-1} (-B A B^T) (BB^T)^{-1}

        // I. multigrid (BB^T)^{-1}
        preconditioner_schur_complement->vmult(dst,src);

        // II. (-B A B^T)
        // II.a) B^T
        underlying_operator->gradient_operator.apply(temp_velocity,dst);
        // II.b) A = 1/dt * mass matrix + viscous term + linearized convective term
        if(underlying_operator->param.problem_type == ProblemType::Unsteady)
        {
          underlying_operator->mass_matrix_operator.apply(temp_velocity_2,temp_velocity);
          temp_velocity_2 *= underlying_operator->get_scaling_factor_time_derivative_term();
        }
        else // ensure that temp_velocity_2 is initialized with 0 because viscous_operator calls apply_add and not apply
        {
          temp_velocity_2 = 0.0;
        }
        underlying_operator->viscous_operator.apply_add(temp_velocity_2,temp_velocity);
        if(underlying_operator->nonlinear_problem_has_to_be_solved())
        {
           underlying_operator->convective_operator.apply_linearized_add(
                                       temp_velocity_2,
                                       temp_velocity,
                                       &underlying_operator->vector_linearization->block(0),
                                       underlying_operator->time + underlying_operator->time_step);
        }
        // II.c) -B
        underlying_operator->divergence_operator.apply(dst,temp_velocity_2);

        // III. multigrid -(BB^T)^{-1}
        preconditioner_schur_complement->vmult(dst,dst);
        dst *= -1.0;
      }
      else if(preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Compatible)
      {
        // -S^{-1} = - (BM^{-1}B^T)^{-1} (-B M^{-1} A M^{-1} B^T) (BM^{-1}B^T)^{-1}

        // I. multigrid (BM^{-1}B^T)^{-1}
        preconditioner_schur_complement->vmult(dst,src);

//        // II. (-B M^{-1} A M^{-1} B^T)
//        // II.a) B^T
//        underlying_operator->gradient_operator.apply(temp_velocity,dst);
//        // II.b) M^{-1}
//        preconditioner_schur_complement_inv_mass_matrix->vmult(temp_velocity,temp_velocity);
//
//        // II.c) A = 1/dt * mass matrix + viscous term + linearized convective term
//        if(underlying_operator->param.problem_type == ProblemType::Unsteady)
//        {
//          underlying_operator->mass_matrix_operator.apply(temp_velocity_2,temp_velocity);
//          temp_velocity_2 *= underlying_operator->get_scaling_factor_time_derivative_term();
//        }
//        else // ensure that temp_velocity_2 is initialized with 0 because viscous_operator calls apply_add and not apply
//        {
//          temp_velocity_2 = 0.0;
//        }
//        underlying_operator->viscous_operator.apply_add(temp_velocity_2,temp_velocity);
//        if(underlying_operator->nonlinear_problem_has_to_be_solved())
//        {
//           underlying_operator->convective_operator.apply_linearized_add(
//                                       temp_velocity_2,
//                                       temp_velocity,
//                                       &underlying_operator->vector_linearization->block(0),
//                                       underlying_operator->time + underlying_operator->time_step);
//        }
//        // II.d) M^{-1}
//        preconditioner_schur_complement_inv_mass_matrix->vmult(temp_velocity_2,temp_velocity_2);
//        // II.e) -B
//        underlying_operator->divergence_operator.apply(dst,temp_velocity_2);

        // II. (- M_p^{-1} B A B^T M_p^{-1})
        // II.a) M_p^{-1}
        preconditioner_schur_complement_inv_mass_matrix->vmult(dst,dst);
        // II.b) B^T
        underlying_operator->gradient_operator.apply(temp_velocity,dst);
        // II.c) A = 1/dt * mass matrix + viscous term + linearized convective term
        if(underlying_operator->param.problem_type == ProblemType::Unsteady)
        {
          underlying_operator->mass_matrix_operator.apply(temp_velocity_2,temp_velocity);
          temp_velocity_2 *= underlying_operator->get_scaling_factor_time_derivative_term();
        }
        else // ensure that temp_velocity_2 is initialized with 0 because viscous_operator calls apply_add and not apply
        {
          temp_velocity_2 = 0.0;
        }
        underlying_operator->viscous_operator.apply_add(temp_velocity_2,temp_velocity);
        if(underlying_operator->nonlinear_problem_has_to_be_solved())
        {
           underlying_operator->convective_operator.apply_linearized_add(
                                       temp_velocity_2,
                                       temp_velocity,
                                       &underlying_operator->vector_linearization->block(0),
                                       underlying_operator->time + underlying_operator->time_step);
        }
        // II.d) -B
        underlying_operator->divergence_operator.apply(dst,temp_velocity_2);
        // II.e) M_p^{-1}
        preconditioner_schur_complement_inv_mass_matrix->vmult(dst,dst);

        // III. multigrid -(BM^{-1}B^T)^{-1}
        preconditioner_schur_complement->vmult(dst,dst);
        dst *= -1.0;
      }
    }
    else if(preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::PressureConvectionDiffusion)
    {
      // -S^{-1} = M_p^{-1} A_p (-L)^{-1}
      //TODO
      if(true)
      {
        // I. multigrid to approximate the inverse of the negative Laplace operator (classical or compatible), (-L)^{-1}
        preconditioner_schur_complement->vmult(temp,src);
      }
      else
      {
        //TODO
        parallel::distributed::Vector<value_type> src_null(src);
        //TODO
        if(underlying_operator->param.pure_dirichlet_bc == true)
        {
          laplace_operator->apply_nullspace_projection(src_null);
        }
        solver_pressure_block->solve(temp,src_null);
      }

      // II. pressure convection diffusion operator A_p
      pressure_convection_diffusion_operator->apply(dst,temp,underlying_operator->get_velocity_linearization());

      // III. inverse pressure mass matrix M_p^{-1}
      preconditioner_schur_complement_inv_mass_matrix->vmult(dst,dst);
    }
    else
    {
      AssertThrow(preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::None ||
                  preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::InverseMassMatrix ||
                  preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::GeometricMultigrid ||
                  preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::CahouetChabard ||
                  preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::Elman ||
                  preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::PressureConvectionDiffusion,
                  ExcMessage("Specified preconditioner for pressure/Schur complement block is not implemented."));
    }
  }

private:
  DGNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> *underlying_operator;
  PreconditionerDataLinearSolver preconditioner_data;
  std_cxx11::shared_ptr<PreconditionerBase<value_type> > preconditioner_momentum;
  std_cxx11::shared_ptr<PreconditionerBase<value_type> > preconditioner_schur_complement;
  std_cxx11::shared_ptr<PreconditionerBase<value_type> > preconditioner_schur_complement_inv_mass_matrix;
  std_cxx11::shared_ptr<PressureConvectionDiffusionOperator<dim, fe_degree_p, fe_degree, value_type> > pressure_convection_diffusion_operator;

  std_cxx11::shared_ptr<LaplaceOperator<dim> > laplace_operator;
  std_cxx11::shared_ptr<IterativeSolverBase<parallel::distributed::Vector<value_type> > > solver_pressure_block;

  std_cxx11::shared_ptr<VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, value_type> > velocity_conv_diff_operator;
  std_cxx11::shared_ptr<IterativeSolverBase<parallel::distributed::Vector<value_type> > > solver_velocity_block;

  parallel::distributed::Vector<value_type> mutable temp;
  parallel::distributed::Vector<value_type> mutable temp_velocity, temp_velocity_2;
};


#endif /* INCLUDE_PRECONDITIONERNAVIERSTOKES_H_ */
