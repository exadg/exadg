/*
 * MGCoarseGridSolvers.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_MGCOARSEGRIDSOLVERS_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_MGCOARSEGRIDSOLVERS_H_

#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/multigrid/mg_base.h>

#include "preconditioner_base.h"
#include "solvers_and_preconditioners/jacobi_preconditioner.h"

template<typename Operator>
class MGCoarsePCG : public MGCoarseGridBase<parallel::distributed::Vector<typename Operator::value_type> >
{
public:
  MGCoarsePCG(const Operator &matrix,
              const bool     use_preconditioner_in)
    :
    coarse_matrix (matrix),
    use_preconditioner(use_preconditioner_in)
  {
    if (use_preconditioner)
    {
      preconditioner.reset(new JacobiPreconditioner<typename Operator::value_type,Operator>(coarse_matrix));
      std_cxx11::shared_ptr<JacobiPreconditioner<typename Operator::value_type,Operator> > precon =
          std::dynamic_pointer_cast<JacobiPreconditioner<typename Operator::value_type,Operator> > (preconditioner);
      AssertDimension(precon->get_size_of_diagonal(), coarse_matrix.m());
    }
  }

  virtual ~MGCoarsePCG()
  {}

  void update_preconditioner(const Operator &underlying_operator)
  {
    if(use_preconditioner)
    {
      preconditioner->update(&underlying_operator);
    }
  }

  virtual void operator() (const unsigned int                                                 ,
                           parallel::distributed::Vector<typename Operator::value_type>       &dst,
                           const parallel::distributed::Vector<typename Operator::value_type> &src) const
  {
    const double abs_tol = 1.e-20;
    const double rel_tol = 1.e-3; //1.e-4;
    ReductionControl solver_control (1e4, abs_tol, rel_tol);

    SolverCG<parallel::distributed::Vector<typename Operator::value_type> >
      solver_coarse (solver_control, solver_memory);
    typename VectorMemory<parallel::distributed::Vector<typename Operator::value_type> >::Pointer r(solver_memory);
    *r = src;
    coarse_matrix.apply_nullspace_projection(*r);
    if (use_preconditioner)
      solver_coarse.solve (coarse_matrix, dst, *r, *preconditioner);
    else
      solver_coarse.solve (coarse_matrix, dst, *r, PreconditionIdentity());

//    std::cout << "Iterations coarse grid solver = " << solver_control.last_step() << std::endl;
  }

private:
  const Operator &coarse_matrix;
  std_cxx11::shared_ptr<PreconditionerBase<typename Operator::value_type> > preconditioner;
  mutable GrowingVectorMemory<parallel::distributed::Vector<typename Operator::value_type> > solver_memory;
  const bool use_preconditioner;
};


template<typename Operator>
class MGCoarseGMRES : public MGCoarseGridBase<parallel::distributed::Vector<typename Operator::value_type> >
{
public:
  MGCoarseGMRES(const Operator &matrix,
                const bool     use_preconditioner_in)
    :
    coarse_matrix (matrix),
    use_preconditioner(use_preconditioner_in)
  {
    if (use_preconditioner)
    {
      preconditioner.reset(new JacobiPreconditioner<typename Operator::value_type,Operator>(coarse_matrix));
      std_cxx11::shared_ptr<JacobiPreconditioner<typename Operator::value_type,Operator> > precon =
          std::dynamic_pointer_cast<JacobiPreconditioner<typename Operator::value_type,Operator> >(preconditioner);
      AssertDimension(precon->get_size_of_diagonal(), coarse_matrix.m());
    }
  }

  virtual ~MGCoarseGMRES()
  {}

  void update_preconditioner(const Operator &underlying_operator)
  {
    if (use_preconditioner)
    {
      preconditioner->update(&underlying_operator);
    }
  }

  virtual void operator() (const unsigned int                                                 ,
                           parallel::distributed::Vector<typename Operator::value_type>       &dst,
                           const parallel::distributed::Vector<typename Operator::value_type> &src) const
  {
    const double abs_tol = 1.e-20;
    const double rel_tol = 1.e-3; //1.e-4;
    ReductionControl solver_control (1e4, abs_tol, rel_tol);

    typename SolverGMRES<parallel::distributed::Vector<typename Operator::value_type> >::AdditionalData additional_data;
    additional_data.max_n_tmp_vectors = 100;
    additional_data.right_preconditioning = true;

    SolverGMRES<parallel::distributed::Vector<typename Operator::value_type> >
      solver_coarse (solver_control, solver_memory, additional_data);

    typename VectorMemory<parallel::distributed::Vector<typename Operator::value_type> >::Pointer r(solver_memory);
    *r = src;
    coarse_matrix.apply_nullspace_projection(*r);

    if (use_preconditioner)
      solver_coarse.solve (coarse_matrix, dst, *r, *preconditioner);
    else
      solver_coarse.solve (coarse_matrix, dst, *r, PreconditionIdentity());

//    std::cout << "Iterations coarse grid solver = " << solver_control.last_step() << std::endl;
  }

private:
  const Operator &coarse_matrix;
  std_cxx11::shared_ptr<PreconditionerBase<typename Operator::value_type> > preconditioner;
  mutable GrowingVectorMemory<parallel::distributed::Vector<typename Operator::value_type> > solver_memory;
  const bool use_preconditioner;
};


template<typename Vector, typename InverseOperator>
class MGCoarseInverseOperator : public MGCoarseGridBase<Vector>
{
public:
  MGCoarseInverseOperator(std_cxx11::shared_ptr<InverseOperator const> inverse_coarse_grid_operator)
    : inverse_operator(inverse_coarse_grid_operator)
  {}

  virtual ~MGCoarseInverseOperator()
  {}

  virtual void operator() (const unsigned int level,
                           Vector             &dst,
                           const Vector       &src) const
  {
    AssertThrow(inverse_operator.get() != 0, ExcMessage("InverseOperator of multigrid coarse grid solver is uninitialized!"));
    AssertThrow(level == 0, ExcNotImplemented());

    inverse_operator->vmult(dst, src);
  }

  std_cxx11::shared_ptr<InverseOperator const> inverse_operator;
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MGCOARSEGRIDSOLVERS_H_ */
