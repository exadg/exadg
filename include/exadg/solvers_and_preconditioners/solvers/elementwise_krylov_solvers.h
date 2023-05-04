/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_ELEMENTWISE_KRYLOV_SOLVERS_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_ELEMENTWISE_KRYLOV_SOLVERS_H_

// deal.II
#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/vectorization.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/solvers/solver_data.h>

namespace ExaDG
{
namespace Elementwise
{
template<typename Number, typename Number2>
bool
all_smaller(Number const a, const Number2 b)
{
  return a < b;
}

template<typename Number, typename Number2>
bool
all_smaller(dealii::VectorizedArray<Number> const a, Number2 const b)
{
  for(unsigned int i = 0; i < dealii::VectorizedArray<Number>::size(); ++i)
    if(a[i] >= b)
      return false;
  return true;
}

template<typename Number>
bool
all_true(Number const a)
{
  return (a >= 0);
}

template<typename Number>
bool
all_true(dealii::VectorizedArray<Number> const a)
{
  for(unsigned int i = 0; i < dealii::VectorizedArray<Number>::size(); ++i)
    if(a[i] < 0)
      return false;
  return true;
}

template<typename Number, typename Number2>
bool
converged(Number &     is_converged,
          Number       norm_r_abs,
          Number2      ABS_TOL,
          Number       norm_r_rel,
          Number2      REL_TOL,
          unsigned int k,
          unsigned int MAX_ITER)
{
  bool is_converged_bool = true;
  if(norm_r_abs < ABS_TOL or norm_r_rel < REL_TOL or k >= MAX_ITER)
  {
    is_converged      = 1.0;
    is_converged_bool = true;
  }
  else
  {
    is_converged      = -1.0;
    is_converged_bool = false;
  }

  return is_converged_bool;
}

template<typename Number, typename Number2>
bool
converged(dealii::VectorizedArray<Number> & is_converged,
          dealii::VectorizedArray<Number>   norm_r_abs,
          Number2                           ABS_TOL,
          dealii::VectorizedArray<Number>   norm_r_rel,
          Number2                           REL_TOL,
          unsigned int                      k,
          unsigned int                      MAX_ITER)
{
  for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
  {
    if((norm_r_abs[v] < ABS_TOL or norm_r_rel[v] < REL_TOL or k >= MAX_ITER))
      is_converged[v] = 1.0;
    else
      is_converged[v] = -1.0;
  }

  return all_true(is_converged);
}

template<typename Number>
void
adjust_division_by_zero(Number &)
{
}

template<typename Number>
void
adjust_division_by_zero(dealii::VectorizedArray<Number> & x)
{
  for(unsigned int i = 0; i < dealii::VectorizedArray<Number>::size(); ++i)
    if(x[i] < 1e-30)
      x[i] = 1;
}

template<typename value_type>
void
scale(value_type * dst, value_type const scalar, unsigned int const size)
{
  for(unsigned int i = 0; i < size; ++i)
    dst[i] *= scalar;
}

template<typename value_type, typename value_type_scalar>
void
scale(value_type * dst, value_type_scalar const scalar, unsigned int const size)
{
  for(unsigned int i = 0; i < size; ++i)
    dst[i] = scalar * dst[i];
}

template<typename value_type>
value_type
inner_product(value_type const * vector1, value_type const * vector2, unsigned int const size)
{
  value_type result = value_type();
  for(unsigned int i = 0; i < size; ++i)
    result += vector1[i] * vector2[i];

  return result;
}

template<typename value_type>
value_type
l2_norm(value_type const * vector, unsigned int const size)
{
  return std::sqrt(inner_product(vector, vector, size));
}

template<typename value_type>
void
vector_init(value_type * vector, unsigned int const size)
{
  for(unsigned int i = 0; i < size; ++i)
    vector[i] = 0.0;
}

template<typename value_type>
void
equ(value_type *       dst,
    value_type const   scalar,
    value_type const * in_vector,
    unsigned int const size)
{
  for(unsigned int i = 0; i < size; ++i)
    dst[i] = scalar * in_vector[i];
}

template<typename value_type>
void
equ(value_type *       dst,
    value_type const   scalar1,
    value_type const * in_vector1,
    value_type const   scalar2,
    value_type const * in_vector2,
    unsigned int const size)
{
  for(unsigned int i = 0; i < size; ++i)
    dst[i] = scalar1 * in_vector1[i] + scalar2 * in_vector2[i];
}

template<typename value_type>
void
add(value_type *       dst,
    value_type const   scalar,
    value_type const * in_vector,
    unsigned int const size)
{
  for(unsigned int i = 0; i < size; ++i)
    dst[i] += scalar * in_vector[i];
}

/*
 * Base class.
 */
template<typename value_type, typename Matrix, typename Preconditioner>
class SolverBase
{
public:
  virtual ~SolverBase()
  {
  }

  virtual void
  solve(Matrix const *         matrix,
        value_type *           solution,
        value_type const *     rhs,
        Preconditioner const * preconditioner) = 0;
};

/*
 * CG solver.
 */
template<typename value_type, typename Matrix, typename Preconditioner>
class SolverCG : public SolverBase<value_type, Matrix, Preconditioner>
{
public:
  SolverCG(unsigned int const unknowns, SolverData const & solver_data);

  void
  solve(Matrix const *         matrix,
        value_type *           solution,
        value_type const *     rhs,
        Preconditioner const * preconditioner);

private:
  unsigned int const                M;
  double const                      ABS_TOL;
  double const                      REL_TOL;
  unsigned int const                MAX_ITER;
  dealii::AlignedVector<value_type> storage;
  value_type *                      p, *r, *v;
};

/*
 *  Implementation of CG solver
 */
template<typename value_type, typename Matrix, typename Preconditioner>
SolverCG<value_type, Matrix, Preconditioner>::SolverCG(unsigned int const unknowns,
                                                       SolverData const & solver_data)
  : M(unknowns),
    ABS_TOL(solver_data.abs_tol),
    REL_TOL(solver_data.rel_tol),
    MAX_ITER(solver_data.max_iter)
{
  storage.resize(3 * M);
  p = storage.begin();
  r = storage.begin() + M;
  v = storage.begin() + 2 * M;
}

template<typename value_type, typename Matrix, typename Preconditioner>
void
SolverCG<value_type, Matrix, Preconditioner>::solve(Matrix const *         matrix,
                                                    value_type *           solution,
                                                    value_type const *     rhs,
                                                    Preconditioner const * preconditioner)
{
  value_type one;
  one = 1.0;

  // guess initial solution
  vector_init(solution, M);

  // apply matrix vector product: v = A*solution
  matrix->vmult(v, solution);

  // compute residual: r = rhs-A*solution
  equ(r, one, rhs, -one, v, M);
  value_type norm_r0 = l2_norm(r, M);

  // precondition
  preconditioner->vmult(p, r);

  // compute norm of residual
  value_type norm_r_abs = norm_r0;
  value_type norm_r_rel = one;

  // compute (r^{0})^T * y^{0} = (r^{0})^T * p^{0}
  value_type r_times_y = inner_product(r, p, M);

  unsigned int n_iter = 0;

  while(true)
  {
    // v = A*p
    matrix->vmult(v, p);

    // p_times_v = p^T*v
    value_type p_times_v = inner_product(p, v, M);
    adjust_division_by_zero(p_times_v);

    // alpha = (r^T*y) / (p^T*v)
    value_type alpha = (r_times_y) / (p_times_v);

    // solution <- solution + alpha*p
    add(solution, alpha, p, M);

    // r <- r - alpha*v
    add(r, -alpha, v, M);

    // calculate residual norm
    norm_r_abs = l2_norm(r, M);
    norm_r_rel = norm_r_abs / norm_r0;

    // increment iteration counter
    ++n_iter;

    // check convergence
    if(all_smaller(norm_r_abs, ABS_TOL) or all_smaller(norm_r_rel, REL_TOL) or (n_iter > MAX_ITER))
    {
      break;
    }

    // precondition: y = P^{-1} * r
    // Note: we use v instead of y to avoid
    // the storage of another variable y
    preconditioner->vmult(v, r);

    // compute r_times_y_new = r^T*y
    value_type r_times_y_new = inner_product(r, v, M);

    // beta = (r^T*y)_new / (r^T*y)
    value_type beta = r_times_y_new / r_times_y;

    // p <- y + beta*p
    equ(p, one, v, beta, p, M);

    r_times_y = r_times_y_new;
  }

  //    std::cout<<"Number of iterations = "<< n_iter << std::endl;
}


/*
 *  GMRES solver with right preconditioning and restart.
 */
template<typename value_type, typename Matrix, typename Preconditioner>
class SolverGMRES : public SolverBase<value_type, Matrix, Preconditioner>
{
public:
  SolverGMRES(unsigned int const unknowns, SolverData const & solver_data);

  void
  solve(Matrix const * A, value_type * x, value_type const * b, Preconditioner const * P);

private:
  // Matrix size MxM
  unsigned int const M;

  // absolute and relative solver tolerances
  double const ABS_TOL;
  double const REL_TOL;

  // maximum number of (overall iterations)
  unsigned int const MAX_ITER;

  // maximum number of iterations before restart
  unsigned int const MAX_KRYLOV_SIZE;

  // absolute and relative norm of residual
  value_type norm_r_abs;
  value_type norm_r_initial;
  value_type norm_r_rel;

  // store convergence status which is necessary
  // for the dealii::VectorizedArray data type where the
  // convergence status of the different elements
  // of dealii::VectorizedArray has to be tracked seperately
  value_type convergence_status;

  // accumulated iterations
  unsigned int iterations;
  // local iteration counter which
  // will be reset after restart
  unsigned int k;

  // matrices of variable size
  dealii::AlignedVector<dealii::AlignedVector<value_type>> V;
  dealii::AlignedVector<dealii::AlignedVector<value_type>> H;

  // temporary vector
  dealii::AlignedVector<value_type> temp;

  // vectors of variable size
  dealii::AlignedVector<value_type> res;
  dealii::AlignedVector<value_type> s;
  dealii::AlignedVector<value_type> c;

  // neutral element of multiplication
  // for data of type value_type
  value_type one;

  void
  clear();

  void
  do_solve(Matrix const * A, value_type * x, value_type const * b, Preconditioner const * P);

  void
  modified_gram_schmidt(dealii::AlignedVector<value_type> &                              w,
                        dealii::AlignedVector<dealii::AlignedVector<value_type>> &       H,
                        dealii::AlignedVector<dealii::AlignedVector<value_type>> const & V,
                        unsigned int const                                               dim);

  template<typename Number>
  void perform_givens_rotation_and_calculate_residual(Number);

  template<typename Number>
  void perform_givens_rotation_and_calculate_residual(dealii::VectorizedArray<Number>);

  template<typename Number>
  void print(Number, std::string);

  template<typename Number>
  void
  print(dealii::VectorizedArray<Number> y, std::string name);
};

/*
 * Implementation of GMRES solver.
 */
template<typename value_type, typename Matrix, typename Preconditioner>
SolverGMRES<value_type, Matrix, Preconditioner>::SolverGMRES(unsigned int const unknowns,
                                                             SolverData const & solver_data)
  : M(unknowns),
    ABS_TOL(solver_data.abs_tol),
    REL_TOL(solver_data.rel_tol),
    MAX_ITER(solver_data.max_iter),
    MAX_KRYLOV_SIZE(solver_data.max_krylov_size),
    iterations(0),
    k(0)
{
  norm_r_abs     = 1.0;
  norm_r_initial = 1.0;
  norm_r_rel     = 1.0;

  // negative values = false (not converged)
  convergence_status = -1.0;

  temp = dealii::AlignedVector<value_type>(M);

  one = 1.0;
}

template<typename value_type, typename Matrix, typename Preconditioner>
void
SolverGMRES<value_type, Matrix, Preconditioner>::clear()
{
  // clear all data
  V.clear();
  H.clear();
  res.clear();
  s.clear();
  c.clear();
}

template<typename value_type, typename Matrix, typename Preconditioner>
template<typename Number>
void SolverGMRES<value_type, Matrix, Preconditioner>::print(Number, std::string)
{
}

/*
 *  Print function for data of type dealii::VectorizedArray
 */
template<typename value_type, typename Matrix, typename Preconditioner>
template<typename Number>
void
SolverGMRES<value_type, Matrix, Preconditioner>::print(dealii::VectorizedArray<Number> y,
                                                       std::string                     name)
{
  for(unsigned int v = 0; v < dealii::VectorizedArray<double>::size(); ++v)
  {
    std::cout << name << "[" << v << "] = " << y[v] << std::endl;
  }
}

/*
 *  w:   new search direction
 *  H:   Hessenberg matrix
 *  V:   contains orthogonal vectors
 *  dim: number of orthogonal vectors
 *
 *  fill dim-th column of Hessenberg matrix
 *  and update vector w such that it is orthogonal
 *  the all previous vector in V
 *
 *  The operations performed here are "unproblematic"
 *  and can be performed also in the case of the
 *  dealii::VectorizedArray data type, i.e., the values written
 *  to the Hessenberg matrix in this function are
 *  ignored in case that the solver has already converged
 *  for a specific component of the vectorized array
 *  (we explicitly overwrite this column of the Hessenberg
 *  matrix when performing the Givens rotation.
 */
template<typename value_type, typename Matrix, typename Preconditioner>
void
SolverGMRES<value_type, Matrix, Preconditioner>::modified_gram_schmidt(
  dealii::AlignedVector<value_type> &                              w,
  dealii::AlignedVector<dealii::AlignedVector<value_type>> &       H,
  dealii::AlignedVector<dealii::AlignedVector<value_type>> const & V,
  unsigned int const                                               dim)
{
  for(unsigned int i = 0; i < dim; ++i)
  {
    H[dim - 1][i] = inner_product(w.begin(), V[i].begin(), M);
    add(w.begin(), -H[dim - 1][i], V[i].begin(), M);
  }

  H[dim - 1][dim] = l2_norm(w.begin(), M);
}

template<typename value_type, typename Matrix, typename Preconditioner>
template<typename Number>
void
  SolverGMRES<value_type, Matrix, Preconditioner>::perform_givens_rotation_and_calculate_residual(
    Number)
{
  // Givens rotations for Hessenberg matrix
  for(int i = 0; i <= int(k) - 1; ++i)
  {
    value_type H_i_k   = H[k][i];
    value_type H_ip1_k = H[k][i + 1];

    H[k][i]     = c[i] * H_i_k + s[i] * H_ip1_k;
    H[k][i + 1] = -s[i] * H_i_k + c[i] * H_ip1_k;
  }

  // Givens rotations for residual-vector
  value_type beta = std::sqrt(H[k][k] * H[k][k] + H[k][k + 1] * H[k][k + 1]);
  s.push_back(H[k][k + 1] / beta);
  c.push_back(H[k][k] / beta);

  H[k][k] = beta;

  value_type res_k_store = res[k];

  res[k] = c[k] * res_k_store;
  res.push_back(-s[k] * res_k_store);
}

template<typename value_type, typename Matrix, typename Preconditioner>
template<typename Number>
void
  SolverGMRES<value_type, Matrix, Preconditioner>::perform_givens_rotation_and_calculate_residual(
    dealii::VectorizedArray<Number>)
{
  dealii::VectorizedArray<Number> H_i_k   = dealii::VectorizedArray<Number>();
  dealii::VectorizedArray<Number> H_ip1_k = dealii::VectorizedArray<Number>();

  for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
  {
    if(convergence_status[v] > 0.0)
    {
      // Set a value of 1 for row_index = col_index
      // The (k,k) entry is set to 1 because we divide by H(k,k) during
      // the backward substitution.
      H[k][k][v] = 1.0;
      // set a value of 0 for all other entries of this column
      // this has to be done in order to not change the results of
      // the backward substitution.
      for(int i = 0; i <= int(k) - 1; ++i)
        H[k][i][v] = 0.0;
    }
    else
    {
      // Givens rotations for Hessenberg matrix
      for(int i = 0; i <= int(k) - 1; ++i)
      {
        H_i_k[v]   = H[k][i][v];
        H_ip1_k[v] = H[k][i + 1][v];

        H[k][i][v]     = c[i][v] * H_i_k[v] + s[i][v] * H_ip1_k[v];
        H[k][i + 1][v] = -s[i][v] * H_i_k[v] + c[i][v] * H_ip1_k[v];
      }
    }
  }

  dealii::VectorizedArray<Number> beta        = dealii::VectorizedArray<Number>();
  dealii::VectorizedArray<Number> sin         = dealii::VectorizedArray<Number>();
  dealii::VectorizedArray<Number> cos         = dealii::VectorizedArray<Number>();
  dealii::VectorizedArray<Number> res_k_store = dealii::VectorizedArray<Number>();
  res_k_store                                 = res[k];
  dealii::VectorizedArray<Number> res_k       = dealii::VectorizedArray<Number>();
  dealii::VectorizedArray<Number> res_kp1     = dealii::VectorizedArray<Number>();

  for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
  {
    if(convergence_status[v] > 0.0)
    {
      // set residual to zero
      // This has to be done in order to ensure that the
      // vector v^(k) does not contribute to the solution x,
      // see also backward substitution.
      res_k[v] = 0.0;
    }
    else
    {
      // Givens rotations for residual-vector
      beta[v]    = std::sqrt(H[k][k][v] * H[k][k][v] + H[k][k + 1][v] * H[k][k + 1][v]);
      sin[v]     = H[k][k + 1][v] / beta[v];
      cos[v]     = H[k][k][v] / beta[v];
      H[k][k][v] = beta[v];
      res_k[v]   = cos[v] * res_k_store[v];
      res_kp1[v] = -sin[v] * res_k_store[v];
    }
  }

  s.push_back(sin);
  c.push_back(cos);
  res[k] = res_k;
  res.push_back(res_kp1);
}

/*
 *  Linear system of equations: Ax = b
 *  A: matrix
 *  P: preconditioner
 *  x: solution vector
 *  b: rhs vector
 *  r: residual r = b - A*x
 */
template<typename value_type, typename Matrix, typename Preconditioner>
void
SolverGMRES<value_type, Matrix, Preconditioner>::solve(Matrix const *         A,
                                                       value_type *           x,
                                                       value_type const *     b,
                                                       Preconditioner const * P)
{
  iterations = 0;

  // restarted GMRES
  do
  {
    // reset local iteration counter
    k = 0;

    // make sure that memory allocated
    // in previous solves is released
    clear();

    // apply GMRES solver where the
    // maximum number of iterations is set
    // to the maximum size of the Krylov subspace
    do_solve(A, x, b, P);

    iterations += k;

  } while(not converged(
    convergence_status, norm_r_abs, ABS_TOL, norm_r_rel, REL_TOL, iterations, MAX_ITER));

  // output convergence info
  //    std::cout << "Number of iterations = " << iterations << std::endl;

  //    std::cout<<std::endl;
  //    print(convergence_status,"convergence status");
  //
  //    // check convergence by computing the residual using
  //    // the solution x: r = b - A*x
  //
  //    //temp2 = A*x
  //    dealii::AlignedVector<value_type> temp2 = dealii::AlignedVector<value_type>(M);
  //    A->vmult(temp2.begin(),x);
  //    // temp = b - temp2 = b - A*x = r
  //    equ(temp.begin(),one,b,-one,temp2.begin());
  //    print(l2_norm(temp.begin()),"l2-norm of residual");
}

template<typename value_type, typename Matrix, typename Preconditioner>
void
SolverGMRES<value_type, Matrix, Preconditioner>::do_solve(Matrix const *         A,
                                                          value_type *           x,
                                                          value_type const *     b,
                                                          Preconditioner const * P)
{
  // apply matrix vector product: r = A*x
  V.push_back(dealii::AlignedVector<value_type>(M));
  A->vmult(V[0].begin(), x);

  // compute residual r = b - A*x and its norm
  equ(V[0].begin(), one, b, -one, V[0].begin(), M);
  res.push_back(l2_norm(V[0].begin(), M));

  // reset initial residual only in the first iteration
  // but not for the restarted iterations and
  // make sure that initial residual is not zero
  if(iterations == k)
  {
    norm_r_initial = res[0];
    adjust_division_by_zero(norm_r_initial);
  }

  // compute absolute and relative residuals
  norm_r_abs = res[0];
  norm_r_rel = res[0] / norm_r_initial;

  while(
    not converged(convergence_status, norm_r_abs, ABS_TOL, norm_r_rel, REL_TOL, k, MAX_KRYLOV_SIZE))
  {
    // normalize vector v^(k)
    if(k == 0)
    {
      adjust_division_by_zero(res[0]);
      scale(V[0].begin(), one / res[0], M);
    }
    else
    {
      int k_last = k - 1;

      adjust_division_by_zero(H[k_last][k_last + 1]);
      scale(V[k_last + 1].begin(), one / H[k_last][k_last + 1], M);
    }

    // calculate new search direction by performing
    // matrix-vector product: V[k+1] = A*V[k]
    V.push_back(dealii::AlignedVector<value_type>(M));

    // apply preconditioner
    P->vmult(temp.begin(), V[k].begin());
    // apply matrix-vector product
    A->vmult(V[k + 1].begin(), temp.begin());

    // resize H
    H.push_back(dealii::AlignedVector<value_type>(k + 2));

    // perform modified Gram-Schmidt orthogonalization
    modified_gram_schmidt(V[k + 1], H, V, k + 1);

    value_type dummy = value_type();
    perform_givens_rotation_and_calculate_residual(dummy);

    // calculate residual norm
    norm_r_abs = std::abs(res[k + 1]);
    norm_r_rel = norm_r_abs / norm_r_initial;

    // increment local iteration counter
    ++k;
  }

  // calculate solution
  dealii::AlignedVector<value_type> y(k);
  dealii::AlignedVector<value_type> delta = dealii::AlignedVector<value_type>(M);

  /*
   *  calculate solution as linear combination of
   *  Krylov subspace vectors multiplied by coefficients y:
   *
   *   x = x_0 + M^{-1}*V*y
   *
   *  obtain coefficients y by backward substitution:
   *
   *  H * y = res, where ...
   *   ... H       is an uppper triangular matrix of dimension k x k, and
   *   ... y, res  are vectors of dimension k
   *
   *  standard: e.g. double/float type
   *  _                    _   _     _     _         _
   *  | H_00 H_01 H_02 H_03  | |  y_0  |   |  res_0    |
   *  |   0  H_11 H_12 H_13  | |  y_1  |   |  res_1    | -> ...
   *  |   0    0  H_22 H_23  | |  y_2  | = |  res_2    | -> y_2 = (res_2-H_23*y_3)/H_22
   *  |   0    0    0  H_33  | |  y_3  |   |  res_3    | -> y_3 = res_3/H_33
   *  |_  0    0    0    0  _| |_  *  _|   |_ res_k+1 _|
   *
   *  dealii::VectorizedArray: assume that the solver converged after 2 iterations
   *                   for some components of the dealii::VectorizedArray
   *                   but after 4 iterations for the other components
   *   _                    _   _     _     _         _
   *  | H_00 H_01   0    0   | |  y_0  |   |  res_0    | -> y_0 = (res_0-H_01*y_0)/H_00
   *  |   0  H_11   0    0   | |  y_1  |   |  res_1    | -> y_1 = res_1/H_11
   *  |   0    0    1    0   | |  y_2  | = |    0      | -> y_2 = 0 -> no contribution of v^(2)
   *  |   0    0    0    1   | |  y_3  |   |    0      | -> y_3 = 0 -> no contribution of v^(3)
   *  |_  0    0    0    0  _| |_  *  _|   |_ res_k+1 _|
   *
   */
  for(int i = int(k) - 1; i >= 0; --i)
  {
    value_type sum = value_type();
    for(unsigned int j = i + 1; j <= k - 1; ++j)
    {
      sum += H[j][i] * y[j];
    }
    adjust_division_by_zero(H[i][i]);
    y[i] = one / H[i][i] * (res[i] - sum);

    add(delta.begin(), y[i], V[i].begin(), M);
  }

  // apply preconditioner and/or add to solution vector
  P->vmult(temp.begin(), delta.begin());
  add(x, one, temp.begin(), M);
}

} // namespace Elementwise
} // namespace ExaDG

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_ELEMENTWISE_KRYLOV_SOLVERS_H_ */
