/*
 * PreconditionerNavierStokes.h
 *
 *  Created on: Jun 30, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_PRECONDITIONERNAVIERSTOKES_H_
#define INCLUDE_PRECONDITIONERNAVIERSTOKES_H_

#include "Preconditioner.h"

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

      helmholtz_operator_data.dof_index = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity);
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
     *      A = 1/dt M_u --> A^{-1} = dt M_u^{-1}
     *      S = div * (1/dt I)^{-1} * grad = dt * laplace --> - S^{-1} = 1/dt (-L)^{-1}
     *
     *   - dt large, nu large:
     *      A = nu (-L) --> A^{-1} = 1/nu (-L)^{-1}
     *      S = div * (- nu * laplace)^{-1} * grad = - 1/nu * I --> - S^{-1} = nu M_p^{-1} (M_p: pressure mass matrix)
     *
     */

    /****** (2,2) pressure/Schur-complement block ********/
    if(preconditioner_data.preconditioner_type == PreconditionerLinearizedNavierStokes::BlockDiagonal ||
       preconditioner_data.preconditioner_type == PreconditionerLinearizedNavierStokes::BlockTriangular )
    {
      // do nothing on velocity/momentum block
      dst.block(0) = src.block(0);

      if(preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::None)
      {
        // no preconditioner for Schur complement block
        dst.block(1) = src.block(1);
      }
      else if(preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::InverseMassMatrix)
      {
        // use the inverse mass matrix as an approximation to the inverse Schur complement
        // this approach is expected to perform well for large time steps and/or large viscosities
        preconditioner_schur_complement->vmult(dst.block(1),src.block(1));
        dst.block(1) *=  underlying_operator->get_viscosity();
      }
      else if(preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::GeometricMultigrid)
      {
        // perform one multigrid V-cycle on the pressure Poisson system as an approximation to the
        // inverse Schur-complement
        // this approach is expected to perform well for small time steps and/or small viscosities
        preconditioner_schur_complement->vmult(dst.block(1),src.block(1));
        dst.block(1) *=  underlying_operator->get_scaling_factor_time_derivative_term();
      }
      else
      {
        AssertThrow(preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::None ||
                    preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::InverseMassMatrix ||
                    preconditioner_data.preconditioner_schur_complement == PreconditionerSchurComplement::GeometricMultigrid,
                    ExcMessage("Specified preconditioner for velocity/momentum block is not implemented."));
      }
    }
    /****** (2,2) pressure/Schur-complement block ********/

    /********** triangular part of preconditioner ********/
    if(preconditioner_data.preconditioner_type == PreconditionerLinearizedNavierStokes::BlockTriangular )
    {
      underlying_operator->gradient_operator.apply_add(dst.block(0),dst.block(1));

      dst.block(1) *= -1.0;
    }
    /********** triangular part of preconditioner ********/

    /********** (1,1) velocity/momentum block ************/
    if(preconditioner_data.preconditioner_type == PreconditionerLinearizedNavierStokes::BlockDiagonal ||
       preconditioner_data.preconditioner_type == PreconditionerLinearizedNavierStokes::BlockTriangular )
    {
      if(preconditioner_data.preconditioner_momentum == PreconditionerMomentum::InverseMassMatrix)
      {
        // use the inverse mass matrix as an approximation to the momentum block
        // this approach is expected to perform well for small time steps and/or small viscosities
        preconditioner_momentum->vmult(dst.block(0),dst.block(0));
        dst.block(0) *= 1./underlying_operator->get_scaling_factor_time_derivative_term();
      }
      else if(preconditioner_data.preconditioner_momentum == PreconditionerMomentum::GeometricMultigrid)
      {
        // perform one multigrid V-cylce on the Helmholtz system
        // this approach is expected to perform well for large time steps and/or large viscosities
        preconditioner_momentum->vmult(dst.block(0),dst.block(0));
      }
      else
      {
        AssertThrow(preconditioner_data.preconditioner_momentum == PreconditionerMomentum::None ||
                    preconditioner_data.preconditioner_momentum == PreconditionerMomentum::InverseMassMatrix ||
                    preconditioner_data.preconditioner_momentum == PreconditionerMomentum::GeometricMultigrid,
                    ExcMessage("Specified preconditioner for pressure/Schur complement block is not implemented."));
      }
    }
    /********** (1,1) velocity/momentum block ************/
  }

private:
  DGNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> *underlying_operator;
  PreconditionerDataLinearSolver preconditioner_data;
  std_cxx11::shared_ptr<PreconditionerBase<value_type> > preconditioner_momentum;
  std_cxx11::shared_ptr<PreconditionerBase<value_type> > preconditioner_schur_complement;
};


#endif /* INCLUDE_PRECONDITIONERNAVIERSTOKES_H_ */
