/*
 * block_preconditioner.h
 *
 *  Created on: Jun 30, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_BLOCK_PRECONDITIONER_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_BLOCK_PRECONDITIONER_H_

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "../../incompressible_navier_stokes/preconditioners/compatible_laplace_operator.h"
#include "../../incompressible_navier_stokes/preconditioners/pressure_convection_diffusion_operator.h"
#include "../../solvers_and_preconditioners/preconditioner/inverse_mass_matrix_preconditioner.h"
#include "../../solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h"
#include "../../solvers_and_preconditioners/util/check_multigrid.h"

#include "../../functionalities/set_zero_mean_value.h"

#include "../../poisson/spatial_discretization/laplace_operator.h"

#include "../../poisson/preconditioner/multigrid_preconditioner.h"
#include "../spatial_discretization/momentum_operator.h"
#include "compatible_laplace_multigrid_preconditioner.h"
#include "multigrid_preconditioner.h"

namespace IncNS
{
// forward declaration
template<int dim, typename Number>
class DGNavierStokesCoupled;

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
 *   1. inverse mass matrix preconditioner (dt small):
 *
 *     A = 1/dt M_u
 *
 *     -> A^{-1} = dt M_u^{-1}
 *
 *   2. Helmholtz operator H =  1/dt M_u + nu (-L) (neglecting the convective term):
 *
 *     -> A^{-1} = H^{-1} where H^{-1} is approximated by performing one GMG V-cycle for the Helmholtz operator
 *
 *   3. Velocity convection-diffusion operator A = 1/dt M_u + C_lin(u) + nu (-L) including the convective term:
 *
 *      -> to approximately invert A consider GMG V-cycle with Chebyshev smoother for nonsymmetric problem
 *         or GMG V-cycle with GMRES (preconditioned by Jacobi method) as smoother
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
 *      -> - S^{-1} = nu M_p^{-1} (M_p: pressure mass matrix)
 *
 *   3. Cahouet & Chabard (combines 1. and 2., robust preconditioner for whole range of time step sizes and visosities)
 *
 *      -> - S^{-1} = nu M_p^{-1} + 1/dt (-L)^{-1}
 *
 *   4. Elman et al. (BFBt preconditioner, sparse approximate commutator preconditioner)
 *
 *      S = - B A^{-1}B^T is approximated by (BB^T) (-B A B^T)^{-1} (BB^T)
 *
 *      -> -S^{-1} = - (-L)^{-1} (-B A B^T) (-L)^{-1}
 *
 *      improvement: S is approximated by (BM^{-1}B^T) (-B A B^T)^{-1} (BM^{-1}B^T)
 *
 *      -> -S^{-1} = - (BM^{-1}B^T)^{-1} (-B M^{-1} A M^{-1} B^T) (BM^{-1}B^T)^{-1}
 *
 *   5. Pressure convection-diffusion preconditioner
 *
 *      -> -S^{-1} = M_p^{-1} A_p (-L)^{-1} where A_p is a convection-diffusion operator for the pressure
 */
// clang-format on

struct BlockPreconditionerData
{
  PreconditionerCoupled preconditioner_type;

  // preconditioner momentum block
  MomentumPreconditioner momentum_preconditioner;
  MultigridData          multigrid_data_momentum_preconditioner;
  bool                   exact_inversion_of_momentum_block;
  SolverData             solver_data_momentum_block;

  // preconditioner Schur-complement block
  SchurComplementPreconditioner schur_complement_preconditioner;
  DiscretizationOfLaplacian     discretization_of_laplacian;
  MultigridData                 multigrid_data_schur_complement_preconditioner;
  bool                          exact_inversion_of_laplace_operator;
  SolverData                    solver_data_schur_complement;
};

template<int dim, typename Number>
class BlockPreconditioner
{
private:
  typedef LinearAlgebra::distributed::Vector<Number>      VectorType;
  typedef LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef DGNavierStokesCoupled<dim, Number> PDEOperator;

  typedef float MultigridNumber;

public:
  BlockPreconditioner(PDEOperator *                   underlying_operator_in,
                      BlockPreconditionerData const & preconditioner_data_in);

  void
  update(PDEOperator const * /*operator*/);

  void
  vmult(BlockVectorType & dst, BlockVectorType const & src) const;

private:
  void
  initialize_vectors();

  void
  initialize_preconditioner_velocity_block();

  void
  setup_multigrid_preconditioner_momentum();

  void
  setup_iterative_solver_momentum();

  void
  initialize_preconditioner_pressure_block();

  void
  setup_multigrid_preconditioner_schur_complement();

  void
  setup_iterative_solver_schur_complement();

  void
  setup_pressure_convection_diffusion_operator();

  void
  apply_preconditioner_velocity_block(VectorType & dst, VectorType const & src) const;

  void
  apply_preconditioner_pressure_block(VectorType & dst, VectorType const & src) const;

  void
  apply_inverse_negative_laplace_operator(VectorType & dst, VectorType const & src) const;

  PDEOperator * underlying_operator;

  BlockPreconditionerData preconditioner_data;

  // preconditioner velocity/momentum block
  std::shared_ptr<PreconditionerBase<Number>> preconditioner_momentum;

  std::shared_ptr<IterativeSolverBase<VectorType>> solver_velocity_block;

  // preconditioner pressure/Schur-complement block
  std::shared_ptr<PreconditionerBase<Number>> multigrid_preconditioner_schur_complement;
  std::shared_ptr<PreconditionerBase<Number>> inv_mass_matrix_preconditioner_schur_complement;

  std::shared_ptr<PressureConvectionDiffusionOperator<dim, Number>>
    pressure_convection_diffusion_operator;

  std::shared_ptr<Poisson::LaplaceOperator<dim, Number>> laplace_operator_classical;

  std::shared_ptr<CompatibleLaplaceOperator<dim, Number>> laplace_operator_compatible;

  std::shared_ptr<IterativeSolverBase<VectorType>> solver_pressure_block;

  // temporary vectors that are necessary when using preconditioners of block-triangular type
  VectorType mutable vec_tmp_pressure;
  VectorType mutable vec_tmp_velocity, vec_tmp_velocity_2;

  // temporary vectors that are necessary when applying the Schur-complement preconditioner (scp)
  VectorType mutable tmp_scp_pressure;
  VectorType mutable tmp_scp_velocity, tmp_scp_velocity_2;

  // temporary vector that is needed if negative Laplace operator is inverted exactly
  // and if a problem with pure Dirichlet BC's is considered
  VectorType mutable tmp_projection_vector;
};

template<int dim, typename Number>
BlockPreconditioner<dim, Number>::BlockPreconditioner(
  PDEOperator *                   underlying_operator_in,
  BlockPreconditionerData const & preconditioner_data_in)
{
  underlying_operator = underlying_operator_in;
  preconditioner_data = preconditioner_data_in;

  initialize_vectors();

  initialize_preconditioner_velocity_block();

  initialize_preconditioner_pressure_block();
}

template<int dim, typename Number>
void
BlockPreconditioner<dim, Number>::update(PDEOperator const * /*operator*/)
{
  // momentum block
  preconditioner_momentum->update(&underlying_operator->momentum_operator);

  // pressure block
  if(preconditioner_data.schur_complement_preconditioner ==
     SchurComplementPreconditioner::PressureConvectionDiffusion)
  {
    if(underlying_operator->unsteady_problem_has_to_be_solved())
    {
      pressure_convection_diffusion_operator->set_scaling_factor_time_derivative_term(
        underlying_operator->momentum_operator.get_scaling_factor_time_derivative_term());
    }
  }
}

template<int dim, typename Number>
void
BlockPreconditioner<dim, Number>::vmult(BlockVectorType & dst, BlockVectorType const & src) const
{
  if(preconditioner_data.preconditioner_type == PreconditionerCoupled::BlockDiagonal)
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
  else if(preconditioner_data.preconditioner_type == PreconditionerCoupled::BlockTriangular)
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
    underlying_operator->gradient_operator.apply(vec_tmp_velocity, dst.block(1));
    dst.block(0).add(underlying_operator->scaling_factor_continuity, vec_tmp_velocity);
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
  else if(preconditioner_data.preconditioner_type ==
          PreconditionerCoupled::BlockTriangularFactorization)
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
    underlying_operator->divergence_operator.apply(vec_tmp_pressure, dst.block(0));
    dst.block(1).add(underlying_operator->scaling_factor_continuity, vec_tmp_pressure);
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
    underlying_operator->gradient_operator.apply(vec_tmp_velocity, dst.block(1));

    // scaling factor continuity
    vec_tmp_velocity *= underlying_operator->scaling_factor_continuity;

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
BlockPreconditioner<dim, Number>::apply_preconditioner_velocity_block(VectorType &       dst,
                                                                      VectorType const & src) const
{
  if(preconditioner_data.momentum_preconditioner == MomentumPreconditioner::None)
  {
    dst = src;
  }
  else if(preconditioner_data.momentum_preconditioner == MomentumPreconditioner::PointJacobi ||
          preconditioner_data.momentum_preconditioner == MomentumPreconditioner::BlockJacobi)
  {
    preconditioner_momentum->vmult(dst, src);
  }
  else if(preconditioner_data.momentum_preconditioner == MomentumPreconditioner::InverseMassMatrix)
  {
    // use the inverse mass matrix as an approximation to the momentum block
    preconditioner_momentum->vmult(dst, src);
    // clang-format off
    dst *= 1. / underlying_operator->momentum_operator.get_scaling_factor_time_derivative_term();
    // clang-format on
  }
  else if(preconditioner_data.momentum_preconditioner == MomentumPreconditioner::Multigrid)
  {
    if(preconditioner_data.exact_inversion_of_momentum_block == false)
    {
      // perform one multigrid V-cylce
      preconditioner_momentum->vmult(dst, src);
    }
    else // exact_inversion_of_momentum_block == true
    {
      // check correctness of multigrid V-cycle

      // clang-format off
      /*
      typedef MultigridPreconditioner<dim, degree_u, Number, MultigridNumber> MULTIGRID;

      std::shared_ptr<MULTIGRID> preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(preconditioner_momentum);

      CheckMultigrid<dim, Number, MomentumOperator<dim, degree_u, Number>, MULTIGRID>
        check_multigrid(underlying_operator->momentum_operator,preconditioner);

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
        ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
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
BlockPreconditioner<dim, Number>::apply_preconditioner_pressure_block(VectorType &       dst,
                                                                      VectorType const & src) const
{
  if(preconditioner_data.schur_complement_preconditioner == SchurComplementPreconditioner::None)
  {
    // No preconditioner for Schur-complement block
    dst = src;
  }
  else if(preconditioner_data.schur_complement_preconditioner ==
          SchurComplementPreconditioner::InverseMassMatrix)
  {
    // - S^{-1} = nu M_p^{-1}
    inv_mass_matrix_preconditioner_schur_complement->vmult(dst, src);
    dst *= underlying_operator->get_viscosity();
  }
  else if(preconditioner_data.schur_complement_preconditioner ==
          SchurComplementPreconditioner::LaplaceOperator)
  {
    // -S^{-1} = 1/dt  (-L)^{-1}
    apply_inverse_negative_laplace_operator(dst, src);
    dst *= underlying_operator->momentum_operator.get_scaling_factor_time_derivative_term();
  }
  else if(preconditioner_data.schur_complement_preconditioner ==
          SchurComplementPreconditioner::CahouetChabard)
  {
    // - S^{-1} = nu M_p^{-1} + 1/dt (-L)^{-1}

    // I. 1/dt (-L)^{-1}
    apply_inverse_negative_laplace_operator(dst, src);
    dst *= underlying_operator->momentum_operator.get_scaling_factor_time_derivative_term();

    // II. M_p^{-1}, apply inverse pressure mass matrix to src-vector and store the result in a
    // temporary vector
    inv_mass_matrix_preconditioner_schur_complement->vmult(tmp_scp_pressure, src);

    // III. add temporary vector scaled by viscosity
    dst.add(underlying_operator->get_viscosity(), tmp_scp_pressure);
  }
  else if(preconditioner_data.schur_complement_preconditioner ==
          SchurComplementPreconditioner::Elman)
  {
    if(preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Classical)
    {
      // -S^{-1} = - (BB^T)^{-1} (-B A B^T) (BB^T)^{-1}

      // I. (BB^T)^{-1} -> apply inverse negative Laplace operator (classical discretization),
      // (-L)^{-1}
      apply_inverse_negative_laplace_operator(dst, src);

      // II. (-B A B^T)
      // II.a) B^T
      underlying_operator->gradient_operator.apply(tmp_scp_velocity, dst);

      // II.b) A = 1/dt * mass matrix  +  viscous term  +  linearized convective term
      underlying_operator->momentum_operator.vmult(tmp_scp_velocity_2, tmp_scp_velocity);

      // II.c) -B
      underlying_operator->divergence_operator.apply(tmp_scp_pressure, tmp_scp_velocity_2);

      // III. -(BB^T)^{-1}
      // III.a) apply inverse negative Laplace operator (classical discretization), (-L)^{-1}
      apply_inverse_negative_laplace_operator(dst, tmp_scp_pressure);
      // III.b) minus sign
      dst *= -1.0;
    }
    else if(preconditioner_data.discretization_of_laplacian ==
            DiscretizationOfLaplacian::Compatible)
    {
      // -S^{-1} = - (BM^{-1}B^T)^{-1} (-B M^{-1} A M^{-1} B^T) (BM^{-1}B^T)^{-1}

      // I. (BM^{-1}B^T)^{-1} -> apply inverse negative Laplace operator (compatible
      // discretization), (-L)^{-1}
      apply_inverse_negative_laplace_operator(dst, src);


      // II. (-B M^{-1} A M^{-1} B^T)
      // II.a) B^T
      underlying_operator->gradient_operator.apply(tmp_scp_velocity, dst);

      // II.b) M^{-1}
      inv_mass_matrix_preconditioner_schur_complement->vmult(tmp_scp_velocity, tmp_scp_velocity);

      // II.c) A = 1/dt * mass matrix + viscous term + linearized convective term
      underlying_operator->momentum_operator.vmult(tmp_scp_velocity_2, tmp_scp_velocity);

      // II.d) M^{-1}
      inv_mass_matrix_preconditioner_schur_complement->vmult(tmp_scp_velocity_2,
                                                             tmp_scp_velocity_2);

      // II.e) -B
      underlying_operator->divergence_operator.apply(tmp_scp_pressure, tmp_scp_velocity_2);


      // III. -(BM^{-1}B^T)^{-1}
      // III.a) apply inverse negative Laplace operator (compatible discretization), (-L)^{-1}
      apply_inverse_negative_laplace_operator(dst, tmp_scp_pressure);
      // III.b) minus sign
      dst *= -1.0;
    }
  }
  else if(preconditioner_data.schur_complement_preconditioner ==
          SchurComplementPreconditioner::PressureConvectionDiffusion)
  {
    // -S^{-1} = M_p^{-1} A_p (-L)^{-1}

    // I. inverse, negative Laplace operator (-L)^{-1}
    apply_inverse_negative_laplace_operator(tmp_scp_pressure, src);

    // II. pressure convection diffusion operator A_p
    if(underlying_operator->nonlinear_problem_has_to_be_solved() == true)
    {
      pressure_convection_diffusion_operator->apply(
        dst, tmp_scp_pressure, underlying_operator->get_velocity_linearization());
    }
    else
    {
      VectorType dummy;
      pressure_convection_diffusion_operator->apply(dst, tmp_scp_pressure, dummy);
    }

    // III. inverse pressure mass matrix M_p^{-1}
    inv_mass_matrix_preconditioner_schur_complement->vmult(dst, dst);
  }
  else
  {
    AssertThrow(false, ExcNotImplemented());
  }

  // scaling_factor_continuity: Since the Schur complement includes both the velocity divergence
  // and the pressure gradient operators as factors, we have to scale by
  // 1/(scaling_factor*scaling_factor) when applying (an approximation of) the inverse Schur
  // complement.
  double inverse_scaling_factor = 1.0 / underlying_operator->scaling_factor_continuity;
  dst *= inverse_scaling_factor * inverse_scaling_factor;
}

template<int dim, typename Number>
void
BlockPreconditioner<dim, Number>::apply_inverse_negative_laplace_operator(
  VectorType &       dst,
  VectorType const & src) const
{
  if(preconditioner_data.exact_inversion_of_laplace_operator == false)
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
    if(underlying_operator->param.pure_dirichlet_bc == true)
    {
      tmp_projection_vector = src;

      if((preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Classical &&
          laplace_operator_classical->is_singular()) ||
         (preconditioner_data.discretization_of_laplacian ==
            DiscretizationOfLaplacian::Compatible &&
          laplace_operator_compatible->is_singular()))
      {
        set_zero_mean_value(tmp_projection_vector);
      }

      pointer_to_src = &tmp_projection_vector;
    }

    dst = 0.0;
    // Note that update of preconditioner is set to false here since the preconditioner has
    // already been updated in the member function update() if desired.
    solver_pressure_block->solve(dst, *pointer_to_src, /* update_preconditioner = */ false);
  }
}

template<int dim, typename Number>
void
BlockPreconditioner<dim, Number>::initialize_vectors()
{
  if(preconditioner_data.preconditioner_type == PreconditionerCoupled::BlockTriangular)
  {
    underlying_operator->initialize_vector_velocity(vec_tmp_velocity);
  }
  else if(preconditioner_data.preconditioner_type ==
          PreconditionerCoupled::BlockTriangularFactorization)
  {
    underlying_operator->initialize_vector_pressure(vec_tmp_pressure);
    underlying_operator->initialize_vector_velocity(vec_tmp_velocity);
    underlying_operator->initialize_vector_velocity(vec_tmp_velocity_2);
  }
}

template<int dim, typename Number>
void
BlockPreconditioner<dim, Number>::initialize_preconditioner_velocity_block()
{
  if(preconditioner_data.momentum_preconditioner == MomentumPreconditioner::PointJacobi)
  {
    // Point Jacobi preconditioner
    preconditioner_momentum.reset(new JacobiPreconditioner<MomentumOperator<dim, Number>>(
      underlying_operator->momentum_operator));
  }
  else if(preconditioner_data.momentum_preconditioner == MomentumPreconditioner::BlockJacobi)
  {
    // Block Jacobi preconditioner
    preconditioner_momentum.reset(new BlockJacobiPreconditioner<MomentumOperator<dim, Number>>(
      underlying_operator->momentum_operator));
  }
  else if(preconditioner_data.momentum_preconditioner == MomentumPreconditioner::InverseMassMatrix)
  {
    // inverse mass matrix
    preconditioner_momentum.reset(new InverseMassMatrixPreconditioner<dim, dim, Number>(
      underlying_operator->get_data(),
      underlying_operator->param.degree_u,
      underlying_operator->get_dof_index_velocity(),
      underlying_operator->get_quad_index_velocity_linear()));
  }
  else if(preconditioner_data.momentum_preconditioner == MomentumPreconditioner::Multigrid)
  {
    // multigrid preconditioner for Helmholtz operator (unsteady case) or viscous operator (steady
    // case)
    setup_multigrid_preconditioner_momentum();

    if(preconditioner_data.exact_inversion_of_momentum_block == true)
    {
      setup_iterative_solver_momentum();
    }
  }
  else
  {
    AssertThrow(preconditioner_data.momentum_preconditioner == MomentumPreconditioner::None,
                ExcNotImplemented());
  }
}

template<int dim, typename Number>
void
BlockPreconditioner<dim, Number>::setup_multigrid_preconditioner_momentum()
{
  typedef MultigridPreconditioner<dim, Number, MultigridNumber> MULTIGRID;

  preconditioner_momentum.reset(new MULTIGRID());

  std::shared_ptr<MULTIGRID> mg_preconditioner =
    std::dynamic_pointer_cast<MULTIGRID>(preconditioner_momentum);

  auto & dof_handler = underlying_operator->get_dof_handler_u();

  parallel::Triangulation<dim> const * tria =
    dynamic_cast<parallel::Triangulation<dim> const *>(&dof_handler.get_triangulation());
  FiniteElement<dim> const & fe = dof_handler.get_fe();

  mg_preconditioner->initialize(preconditioner_data.multigrid_data_momentum_preconditioner,
                                tria,
                                fe,
                                underlying_operator->get_mapping(),
                                underlying_operator->momentum_operator.get_operator_data());
}

template<int dim, typename Number>
void
BlockPreconditioner<dim, Number>::setup_iterative_solver_momentum()
{
  AssertThrow(preconditioner_momentum.get() != 0,
              ExcMessage("preconditioner_momentum is uninitialized"));

  // use FMGRES for "exact" solution of velocity block system
  FGMRESSolverData gmres_data;
  gmres_data.use_preconditioner   = true;
  gmres_data.max_iter             = preconditioner_data.solver_data_momentum_block.max_iter;
  gmres_data.solver_tolerance_abs = preconditioner_data.solver_data_momentum_block.abs_tol;
  gmres_data.solver_tolerance_rel = preconditioner_data.solver_data_momentum_block.rel_tol;
  gmres_data.max_n_tmp_vectors    = preconditioner_data.solver_data_momentum_block.max_krylov_size;

  solver_velocity_block.reset(
    new FGMRESSolver<MomentumOperator<dim, Number>, PreconditionerBase<Number>, VectorType>(
      underlying_operator->momentum_operator, *preconditioner_momentum, gmres_data));
}

template<int dim, typename Number>
void
BlockPreconditioner<dim, Number>::initialize_preconditioner_pressure_block()
{
  if(preconditioner_data.schur_complement_preconditioner ==
     SchurComplementPreconditioner::InverseMassMatrix)
  {
    // inverse mass matrix
    inv_mass_matrix_preconditioner_schur_complement.reset(
      new InverseMassMatrixPreconditioner<dim, 1, Number>(
        underlying_operator->get_data(),
        underlying_operator->param.degree_p,
        underlying_operator->get_dof_index_pressure(),
        underlying_operator->get_quad_index_pressure()));
  }
  else if(preconditioner_data.schur_complement_preconditioner ==
          SchurComplementPreconditioner::LaplaceOperator)
  {
    // multigrid for negative Laplace operator (classical or compatible discretization)
    setup_multigrid_preconditioner_schur_complement();

    if(preconditioner_data.exact_inversion_of_laplace_operator == true)
    {
      // iterative solver used to invert the negative Laplace operator (classical or compatible
      // discretization)
      setup_iterative_solver_schur_complement();
    }
  }
  else if(preconditioner_data.schur_complement_preconditioner ==
          SchurComplementPreconditioner::CahouetChabard)
  {
    AssertThrow(underlying_operator->unsteady_problem_has_to_be_solved() == true,
                ExcMessage(
                  "Cahouet-Chabard preconditioner only makes sense for unsteady problems."));

    // multigrid for negative Laplace operator (classical or compatible discretization)
    setup_multigrid_preconditioner_schur_complement();

    if(preconditioner_data.exact_inversion_of_laplace_operator == true)
    {
      // iterative solver used to invert the negative Laplace operator (classical or compatible
      // discretization)
      setup_iterative_solver_schur_complement();
    }

    // inverse mass matrix to also include the part of the preconditioner that is beneficial when
    // using large time steps and large viscosities.
    inv_mass_matrix_preconditioner_schur_complement.reset(
      new InverseMassMatrixPreconditioner<dim, 1, Number>(
        underlying_operator->get_data(),
        underlying_operator->param.degree_p,
        underlying_operator->get_dof_index_pressure(),
        underlying_operator->get_quad_index_pressure()));

    // initialize tmp vector
    underlying_operator->initialize_vector_pressure(tmp_scp_pressure);
  }
  else if(preconditioner_data.schur_complement_preconditioner ==
          SchurComplementPreconditioner::Elman)
  {
    // multigrid for negative Laplace operator (classical or compatible discretization)
    setup_multigrid_preconditioner_schur_complement();

    if(preconditioner_data.exact_inversion_of_laplace_operator == true)
    {
      // iterative solver used to invert the negative Laplace operator (classical or compatible
      // discretization)
      setup_iterative_solver_schur_complement();
    }

    if(preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Compatible)
    {
      // -S^{-1} = - (BM^{-1}B^T)^{-1} (-B M^{-1} A M^{-1} B^T) (BM^{-1}B^T)^{-1}
      // --> inverse velocity mass matrix needed for inner factor
      inv_mass_matrix_preconditioner_schur_complement.reset(
        new InverseMassMatrixPreconditioner<dim, dim, Number>(
          underlying_operator->get_data(),
          underlying_operator->param.degree_u,
          underlying_operator->get_dof_index_velocity(),
          underlying_operator->get_quad_index_velocity_linear()));
    }

    // initialize tmp vectors
    underlying_operator->initialize_vector_pressure(tmp_scp_pressure);
    underlying_operator->initialize_vector_velocity(tmp_scp_velocity);
    underlying_operator->initialize_vector_velocity(tmp_scp_velocity_2);
  }
  else if(preconditioner_data.schur_complement_preconditioner ==
          SchurComplementPreconditioner::PressureConvectionDiffusion)
  {
    // -S^{-1} = M_p^{-1} A_p (-L)^{-1}

    // I. multigrid for negative Laplace operator (classical or compatible discretization)
    setup_multigrid_preconditioner_schur_complement();

    if(preconditioner_data.exact_inversion_of_laplace_operator == true)
    {
      setup_iterative_solver_schur_complement();
    }

    // II. pressure convection-diffusion operator
    setup_pressure_convection_diffusion_operator();

    // III. inverse pressure mass matrix
    inv_mass_matrix_preconditioner_schur_complement.reset(
      new InverseMassMatrixPreconditioner<dim, 1, Number>(
        underlying_operator->get_data(),
        underlying_operator->param.degree_p,
        underlying_operator->get_dof_index_pressure(),
        underlying_operator->get_quad_index_pressure()));

    // initialize tmp vector
    underlying_operator->initialize_vector_pressure(tmp_scp_pressure);
  }
  else
  {
    AssertThrow(preconditioner_data.schur_complement_preconditioner ==
                  SchurComplementPreconditioner::None,
                ExcNotImplemented());
  }
}

template<int dim, typename Number>
void
BlockPreconditioner<dim, Number>::setup_multigrid_preconditioner_schur_complement()
{
  if(preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Compatible)
  {
    MultigridData mg_data = preconditioner_data.multigrid_data_schur_complement_preconditioner;

    typedef CompatibleLaplaceMultigridPreconditioner<dim, Number, MultigridNumber> MULTIGRID;

    multigrid_preconditioner_schur_complement.reset(new MULTIGRID());

    std::shared_ptr<MULTIGRID> mg_preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(multigrid_preconditioner_schur_complement);

    auto compatible_laplace_operator_data =
      underlying_operator->get_compatible_laplace_operator_data();

    auto & dof_handler = underlying_operator->get_dof_handler_p();

    parallel::Triangulation<dim> const * tria =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation());
    const FiniteElement<dim> & fe = dof_handler.get_fe();

    mg_preconditioner->initialize(
      mg_data, tria, fe, underlying_operator->get_mapping(), compatible_laplace_operator_data);
  }
  else if(preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Classical)
  {
    // multigrid V-cycle for negative Laplace operator
    Poisson::LaplaceOperatorData<dim> laplace_operator_data;
    laplace_operator_data.dof_index            = underlying_operator->get_dof_index_pressure();
    laplace_operator_data.quad_index           = underlying_operator->get_quad_index_pressure();
    laplace_operator_data.IP_factor            = 1.0;
    laplace_operator_data.degree               = underlying_operator->param.degree_p;
    laplace_operator_data.degree_mapping       = underlying_operator->param.degree_mapping;
    laplace_operator_data.operator_is_singular = underlying_operator->param.pure_dirichlet_bc;

    laplace_operator_data.bc = underlying_operator->boundary_descriptor_laplace;

    MultigridData mg_data = preconditioner_data.multigrid_data_schur_complement_preconditioner;

    typedef Poisson::MultigridPreconditioner<dim, Number, MultigridNumber> MULTIGRID;

    multigrid_preconditioner_schur_complement.reset(new MULTIGRID());

    std::shared_ptr<MULTIGRID> mg_preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(multigrid_preconditioner_schur_complement);

    auto & dof_handler = underlying_operator->get_dof_handler_p();

    parallel::Triangulation<dim> const * tria =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation());
    const FiniteElement<dim> & fe = dof_handler.get_fe();

    mg_preconditioner->initialize(mg_data,
                                  tria,
                                  fe,
                                  underlying_operator->get_mapping(),
                                  laplace_operator_data,
                                  &laplace_operator_data.bc->dirichlet_bc,
                                  &underlying_operator->periodic_face_pairs);
  }
  else
  {
    AssertThrow(
      preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Classical ||
        preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Compatible,
      ExcMessage(
        "Specified discretization of Laplacian for Schur-complement preconditioner is not available."));
  }
}

template<int dim, typename Number>
void
BlockPreconditioner<dim, Number>::setup_iterative_solver_schur_complement()
{
  AssertThrow(
    multigrid_preconditioner_schur_complement.get() != 0,
    ExcMessage(
      "Setup of iterative solver for Schur complement preconditioner: Multigrid preconditioner is uninitialized"));

  CGSolverData solver_data;
  solver_data.max_iter             = preconditioner_data.solver_data_schur_complement.max_iter;
  solver_data.solver_tolerance_abs = preconditioner_data.solver_data_schur_complement.abs_tol;
  solver_data.solver_tolerance_rel = preconditioner_data.solver_data_schur_complement.rel_tol;
  solver_data.use_preconditioner   = true;

  if(preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Classical)
  {
    Poisson::LaplaceOperatorData<dim> laplace_operator_data;
    laplace_operator_data.dof_index      = underlying_operator->get_dof_index_pressure();
    laplace_operator_data.quad_index     = underlying_operator->get_quad_index_pressure();
    laplace_operator_data.IP_factor      = 1.0;
    laplace_operator_data.degree         = underlying_operator->param.degree_p;
    laplace_operator_data.degree_mapping = underlying_operator->param.degree_mapping;
    laplace_operator_data.bc             = underlying_operator->boundary_descriptor_laplace;

    laplace_operator_classical.reset(new Poisson::LaplaceOperator<dim, Number>());
    laplace_operator_classical->reinit(underlying_operator->get_data(),
                                       underlying_operator->constraint_p,
                                       laplace_operator_data);

    solver_pressure_block.reset(
      new CGSolver<Poisson::LaplaceOperator<dim, Number>, PreconditionerBase<Number>, VectorType>(
        *laplace_operator_classical, *multigrid_preconditioner_schur_complement, solver_data));
  }
  else if(preconditioner_data.discretization_of_laplacian == DiscretizationOfLaplacian::Compatible)
  {
    CompatibleLaplaceOperatorData<dim> compatible_laplace_operator_data =
      underlying_operator->get_compatible_laplace_operator_data();

    laplace_operator_compatible.reset(new CompatibleLaplaceOperator<dim, Number>());

    laplace_operator_compatible->initialize(underlying_operator->get_data(),
                                            compatible_laplace_operator_data,
                                            underlying_operator->gradient_operator,
                                            underlying_operator->divergence_operator,
                                            underlying_operator->inverse_mass_velocity);

    solver_pressure_block.reset(
      new CGSolver<CompatibleLaplaceOperator<dim, Number>, PreconditionerBase<Number>, VectorType>(
        *laplace_operator_compatible, *multigrid_preconditioner_schur_complement, solver_data));
  }
}

template<int dim, typename Number>
void
BlockPreconditioner<dim, Number>::setup_pressure_convection_diffusion_operator()
{
  // pressure convection-diffusion operator
  // a) mass matrix operator
  ConvDiff::MassMatrixOperatorData<dim> mass_matrix_operator_data;
  mass_matrix_operator_data.dof_index  = underlying_operator->get_dof_index_pressure();
  mass_matrix_operator_data.quad_index = underlying_operator->get_quad_index_pressure();

  std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> boundary_descriptor;
  boundary_descriptor.reset(new ConvDiff::BoundaryDescriptor<dim>());

  // for the pressure convection-diffusion operator the homogeneous operators (convective,
  // diffusive) are applied, so there is no need to specify functions for boundary conditions
  // since they will not be used (must not be used)
  // -> use ConstantFunction as dummy, initialized with NAN in order to detect a possible
  // incorrect access to boundary values
  std::shared_ptr<Function<dim>> dummy;

  // set boundary ID's for pressure convection-diffusion operator

  // Dirichlet BC for pressure
  for(typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::const_iterator it =
        underlying_operator->boundary_descriptor_pressure->dirichlet_bc.begin();
      it != underlying_operator->boundary_descriptor_pressure->dirichlet_bc.end();
      ++it)
  {
    boundary_descriptor->dirichlet_bc.insert(
      std::pair<types::boundary_id, std::shared_ptr<Function<dim>>>(it->first, dummy));
  }
  // Neumann BC for pressure
  for(typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::const_iterator it =
        underlying_operator->boundary_descriptor_pressure->neumann_bc.begin();
      it != underlying_operator->boundary_descriptor_pressure->neumann_bc.end();
      ++it)
  {
    boundary_descriptor->neumann_bc.insert(
      std::pair<types::boundary_id, std::shared_ptr<Function<dim>>>(it->first, dummy));
  }

  // b) diffusive operator
  ConvDiff::DiffusiveOperatorData<dim> diffusive_operator_data;
  diffusive_operator_data.dof_index      = underlying_operator->get_dof_index_pressure();
  diffusive_operator_data.quad_index     = underlying_operator->get_quad_index_pressure();
  diffusive_operator_data.IP_factor      = underlying_operator->param.IP_factor_viscous;
  diffusive_operator_data.degree         = underlying_operator->param.degree_p;
  diffusive_operator_data.degree_mapping = underlying_operator->param.degree_mapping;
  diffusive_operator_data.bc             = boundary_descriptor;
  // TODO: the pressure convection-diffusion operator is initialized with constant viscosity, in
  // case of varying viscosities the pressure convection-diffusion operator (the diffusive
  // operator of the pressure convection-diffusion operator) has to be updated before applying
  // this preconditioner
  diffusive_operator_data.diffusivity = underlying_operator->get_viscosity();

  // c) convective operator
  ConvDiff::ConvectiveOperatorData<dim> convective_operator_data;
  convective_operator_data.dof_index           = underlying_operator->get_dof_index_pressure();
  convective_operator_data.dof_index_velocity  = underlying_operator->get_dof_index_velocity();
  convective_operator_data.quad_index          = underlying_operator->get_quad_index_pressure();
  convective_operator_data.type_velocity_field = ConvDiff::TypeVelocityField::Numerical;
  convective_operator_data.numerical_flux_formulation =
    ConvDiff::NumericalFluxConvectiveOperator::LaxFriedrichsFlux;
  convective_operator_data.bc = boundary_descriptor;

  PressureConvectionDiffusionOperatorData<dim> pressure_convection_diffusion_operator_data;
  pressure_convection_diffusion_operator_data.mass_matrix_operator_data = mass_matrix_operator_data;
  pressure_convection_diffusion_operator_data.diffusive_operator_data   = diffusive_operator_data;
  pressure_convection_diffusion_operator_data.convective_operator_data  = convective_operator_data;
  if(underlying_operator->unsteady_problem_has_to_be_solved())
    pressure_convection_diffusion_operator_data.unsteady_problem = true;
  else
    pressure_convection_diffusion_operator_data.unsteady_problem = false;
  pressure_convection_diffusion_operator_data.convective_problem =
    underlying_operator->nonlinear_problem_has_to_be_solved();

  pressure_convection_diffusion_operator.reset(new PressureConvectionDiffusionOperator<dim, Number>(
    underlying_operator->mapping,
    underlying_operator->get_data(),
    pressure_convection_diffusion_operator_data,
    underlying_operator->constraint_p));

  if(underlying_operator->unsteady_problem_has_to_be_solved())
    pressure_convection_diffusion_operator->set_scaling_factor_time_derivative_term(
      underlying_operator->momentum_operator.get_scaling_factor_time_derivative_term());
}

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_BLOCK_PRECONDITIONER_H_ */
