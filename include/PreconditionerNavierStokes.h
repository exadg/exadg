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
        helmholtz_operator_data.viscous_operator_data = underlying_operator->get_viscous_operator_data();

        helmholtz_operator_data.dof_index = underlying_operator->get_dof_index_velocity();//static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity);
        helmholtz_operator_data.mass_matrix_coefficient = underlying_operator->get_scaling_factor_time_derivative_term();
        helmholtz_operator_data.periodic_face_pairs_level0 = underlying_operator->get_periodic_face_pairs();

        // currently use default parameters of MultigridData!
        MultigridData mg_data_helmholtz;

        preconditioner_momentum.reset(new MyMultigridPreconditioner<dim,value_type,HelmholtzOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>, HelmholtzOperatorData<dim> >
                                            (mg_data_helmholtz,
                                             underlying_operator->get_dof_handler_u(),
                                             underlying_operator->get_mapping(),
                                             helmholtz_operator_data,
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
        preconditioner_momentum.reset(new MyMultigridPreconditioner<dim,value_type,ViscousOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>, ViscousOperatorData<dim> >
                                            (mg_data,
                                             underlying_operator->get_dof_handler_u(),
                                             underlying_operator->get_mapping(),
                                             viscous_operator_data,
                                             underlying_operator->get_fe_parameters()));
      }
    }
    /****** preconditioner velocity/momentum block ****************/

    /****** preconditioner pressure/Schur-complement block ********/
    if(preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::InverseMassMatrix)
    {
      // inverse mass matrix
      preconditioner_schur_complement.reset(new InverseMassMatrixPreconditioner<dim,fe_degree_p,value_type,1>(
                                                  underlying_operator->get_data(),
                                                  underlying_operator->get_dof_index_pressure(),
                                                  underlying_operator->get_quad_index_pressure()));
    }
    else if(preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::GeometricMultigrid)
    {
      setup_multigrid_schur_complement();
    }
    else if(preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::CahouetChabard)
    {
      AssertThrow(underlying_operator->param.problem_type == ProblemType::Unsteady,
          ExcMessage("CahouetChabard preconditioner only makes sense for unsteady problems."));

      setup_multigrid_schur_complement();

      // inverse mass matrix to also include the part of the preconditioner that is beneficial when using large time steps
      // and large viscosities. By definition this part of the preconditioner is called preconditioner_..._cahouet_chabard.
      // This definition is, of course, arbitrary. Actually, the Cahouet-Chabard preconditioner is the preconditioner
      // that includes both parts (i.e. the combination of mass matrix and multigrid for negative Laplace operator).
      preconditioner_schur_complement_cahouet_chabard.reset(new InverseMassMatrixPreconditioner<dim,fe_degree_p,value_type,1>(
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
    bool compatible_discretization = false;

    if(compatible_discretization)
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

      preconditioner_schur_complement.reset(
          new MyMultigridPreconditioner<dim,value_type,
                CompatibleLaplaceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, Number>,
                CompatibleLaplaceOperatorData<dim> >
          (mg_data_compatible_pressure,
           underlying_operator->get_dof_handler_p(),
           underlying_operator->get_dof_handler_u(),
           underlying_operator->get_mapping(),
           compatible_laplace_operator_data));
    }
    else
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
                                                 laplace_operator_data));
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
     *   - dt --> infinity (steady state case):
     *      A = (-nu L)
     *        --> A^{-1} = (-nu L)^{-1} (by performing one multigrid V-cycle on the viscous operator)
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
      vmult_pressure_block(dst.block(1),dst.block(1));

      /*
      *        / I  - A^{-1} B^{T} \
      *  dst = |                   | * dst
      *        \ 0          I      /
      */
      // temp_velocity = B^{T} * dst(1)
      underlying_operator->gradient_operator.apply(temp_velocity,dst.block(1));

      // temp_velocity = A^{-1} * temp_velocity
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
      // perform one multigrid V-cylce on the Helmholtz system or viscous operator (in case of steady-state problem)
      // this approach is expected to perform well for large time steps and/or large viscosities
      preconditioner_momentum->vmult(dst,src);
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
      preconditioner_schur_complement->vmult(dst,src);
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
      preconditioner_schur_complement_cahouet_chabard->vmult(temp,src);

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
    else
    {
      AssertThrow(preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::None ||
                  preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::InverseMassMatrix ||
                  preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::GeometricMultigrid ||
                  preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::CahouetChabard,
                  ExcMessage("Specified preconditioner for pressure/Schur complement block is not implemented."));
    }
  }

private:
  DGNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> *underlying_operator;
  PreconditionerDataLinearSolver preconditioner_data;
  std_cxx11::shared_ptr<PreconditionerBase<value_type> > preconditioner_momentum;
  std_cxx11::shared_ptr<PreconditionerBase<value_type> > preconditioner_schur_complement;
  std_cxx11::shared_ptr<PreconditionerBase<value_type> > preconditioner_schur_complement_cahouet_chabard;
  parallel::distributed::Vector<value_type> mutable temp;
  parallel::distributed::Vector<value_type> mutable temp_velocity;
};


#endif /* INCLUDE_PRECONDITIONERNAVIERSTOKES_H_ */
