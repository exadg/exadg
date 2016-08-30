/*
 * Preconditioner.h
 *
 *  Created on: May 17, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_PRECONDITIONER_H_
#define INCLUDE_PRECONDITIONER_H_

template<typename value_type>
class PreconditionerBase
{
public:
  virtual ~PreconditionerBase(){}

  virtual void vmult(parallel::distributed::Vector<value_type>        &dst,
                     const parallel::distributed::Vector<value_type>  &src) const = 0;
};

#include "InverseMassMatrix.h"

template<int dim, int fe_degree, typename value_type, int n_components=dim>
class InverseMassMatrixPreconditioner : public PreconditionerBase<value_type>
{
public:
  InverseMassMatrixPreconditioner(MatrixFree<dim,value_type> const &mf_data,
                                  unsigned int const               dof_index,
                                  unsigned int const               quad_index)
  {
    inverse_mass_matrix_operator.initialize(mf_data,dof_index,quad_index);
  }

  void vmult (parallel::distributed::Vector<value_type>        &dst,
              const parallel::distributed::Vector<value_type>  &src) const
  {
    inverse_mass_matrix_operator.apply_inverse_mass_matrix(dst,src);
  }

private:
  InverseMassMatrixOperator<dim,fe_degree,value_type,n_components> inverse_mass_matrix_operator;
};

template<int dim, int fe_degree, typename value_type, int n_components=dim>
class InverseMassMatrixPreconditionerPtr : public PreconditionerBase<value_type>
{
public:
  InverseMassMatrixPreconditionerPtr(std_cxx11::shared_ptr<InverseMassMatrixOperator<dim,fe_degree,value_type,n_components> > inv_mass_operator)
    :
    inverse_mass_matrix_operator(inv_mass_operator)
  {}

  void vmult (parallel::distributed::Vector<value_type>        &dst,
              const parallel::distributed::Vector<value_type>  &src) const
  {
    inverse_mass_matrix_operator->apply_inverse_mass_matrix(dst,src);
  }

private:
  std_cxx11::shared_ptr<InverseMassMatrixOperator<dim,fe_degree,value_type,n_components> > inverse_mass_matrix_operator;
};


template<typename value_type, typename Operator>
class JacobiPreconditioner : public PreconditionerBase<value_type>
{
public:
  JacobiPreconditioner(Operator const &underlying_operator)
  {
    underlying_operator.initialize_dof_vector(inverse_diagonal);

    underlying_operator.calculate_inverse_diagonal(inverse_diagonal);
  }

  void vmult (parallel::distributed::Vector<value_type>        &dst,
              const parallel::distributed::Vector<value_type>  &src) const
  {
    if (!PointerComparison::equal(&dst, &src))
      dst = src;
    dst.scale(inverse_diagonal);
  }

  void recalculate_diagonal(Operator const &underlying_operator)
  {
    underlying_operator.calculate_inverse_diagonal(inverse_diagonal);
  }

  unsigned int get_size_of_diagonal()
  {
    return inverse_diagonal.size();
  }

private:
  parallel::distributed::Vector<value_type> inverse_diagonal;
};

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/base/function_lib.h>

// Specialized matrix-free implementation that overloads the copy_to_mg
// function for proper initialization of the vectors in matrix-vector
// products.
template <int dim, typename Operator>
class MGTransferMF : public MGTransferMatrixFree<dim, typename Operator::value_type>
{
public:
  MGTransferMF()
    :
    underlying_operator (0)
  {}

  void set_operator(const MGLevelObject<Operator> &operator_in)
  {
    underlying_operator = &operator_in;
  }

  /**
   * Overload copy_to_mg from MGTransferMatrixFree
   */
  template <class InVector, int spacedim>
  void
  copy_to_mg (const DoFHandler<dim,spacedim>                                               &mg_dof,
              MGLevelObject<parallel::distributed::Vector<typename Operator::value_type> > &dst,
              const InVector                                                               &src) const
  {
    AssertThrow(underlying_operator != 0, ExcNotInitialized());

    for (unsigned int level=dst.min_level();level<=dst.max_level(); ++level)
      (*underlying_operator)[level].initialize_dof_vector(dst[level]);

    MGLevelGlobalTransfer<parallel::distributed::Vector<typename Operator::value_type> >::copy_to_mg(mg_dof, dst, src);
  }

private:
  const MGLevelObject<Operator> *underlying_operator;
};

template<typename Operator>
class MGCoarseIterative : public MGCoarseGridBase<parallel::distributed::Vector<typename Operator::value_type> >
{
public:
  MGCoarseIterative(const Operator &matrix,
                    const bool     use_jacobi)
    :
    coarse_matrix (matrix),
    use_jacobi_preconditioner(use_jacobi)
  {
    if (use_jacobi_preconditioner)
    {
      preconditioner.reset(new JacobiPreconditioner<typename Operator::value_type,Operator>(coarse_matrix));
      std_cxx11::shared_ptr<JacobiPreconditioner<typename Operator::value_type,Operator> > precon =
          std::dynamic_pointer_cast<JacobiPreconditioner<typename Operator::value_type,Operator> > (preconditioner);
      AssertDimension(precon->get_size_of_diagonal(), coarse_matrix.m());
    }
  }

  virtual ~MGCoarseIterative()
  {}

  virtual void operator() (const unsigned int                                                 ,
                           parallel::distributed::Vector<typename Operator::value_type>       &dst,
                           const parallel::distributed::Vector<typename Operator::value_type> &src) const
  {
    ReductionControl solver_control (1e4, 1e-50, 1e-4);
    //IterationNumberControl solver_control (10, 1e-15);

    SolverCG<parallel::distributed::Vector<typename Operator::value_type> >
      solver_coarse (solver_control, solver_memory);
    typename VectorMemory<parallel::distributed::Vector<typename Operator::value_type> >::Pointer r(solver_memory);
    *r = src;
    coarse_matrix.apply_nullspace_projection(*r);
    if (use_jacobi_preconditioner)
      solver_coarse.solve (coarse_matrix, dst, *r, *preconditioner);
    else
      solver_coarse.solve (coarse_matrix, dst, *r, PreconditionIdentity());

    // std::cout<<"Iterations coarse grid solver = "<<solver_control.last_step()<<std::endl;
  }

private:
  const Operator &coarse_matrix;
  std_cxx11::shared_ptr<PreconditionerBase<typename Operator::value_type> > preconditioner;
  mutable GrowingVectorMemory<parallel::distributed::Vector<typename Operator::value_type> > solver_memory;
  const bool use_jacobi_preconditioner;
};



template<typename VECTOR, typename PreconditionType>
class MGCoarseFromSmoother : public MGCoarseGridBase<VECTOR>
{
public:
  MGCoarseFromSmoother(const PreconditionType &mg_smoother,
                       const bool             is_empty)
    : smoother(mg_smoother),
      is_empty(is_empty)
  {}

  virtual ~MGCoarseFromSmoother()
  {}

  virtual void operator() (const unsigned int level,
                           VECTOR             &dst,
                           const VECTOR       &src) const
  {
    if (is_empty)
      return;
    AssertThrow(level == 0, ExcNotImplemented());
    smoother.vmult(dst, src);
  }

  const PreconditionType &smoother;
  const bool is_empty;
};

namespace
{
  // manually compute eigenvalues for the coarsest level for proper setup of the Chebyshev iteration
  template <typename Operator>
  std::pair<double,double>
  compute_eigenvalues(const Operator &op,
                      const parallel::distributed::Vector<typename Operator::value_type> &inverse_diagonal)
  {
    typedef typename Operator::value_type value_type;
    parallel::distributed::Vector<value_type> left, right;
    left.reinit(inverse_diagonal);
    right.reinit(inverse_diagonal, true);
    // NB: initialize rand in order to obtain "reproducible" results !!!
    srand(1);
    for (unsigned int i=0; i<right.local_size(); ++i)
      right.local_element(i) = (double)rand()/RAND_MAX;
    op.apply_nullspace_projection(right);

    SolverControl control(10000, right.l2_norm()*1e-5);
    internal::PreconditionChebyshev::EigenvalueTracker eigenvalue_tracker;
    SolverCG<parallel::distributed::Vector<value_type> > solver (control);
    solver.connect_eigenvalues_slot(std_cxx11::bind(&internal::PreconditionChebyshev::EigenvalueTracker::slot,
                                                    &eigenvalue_tracker,
                                                    std_cxx11::_1));
    JacobiPreconditioner<value_type, Operator> preconditioner(op);
    try
    {
      solver.solve(op, left, right, preconditioner);
    }
    catch (SolverControl::NoConvergence &)
    {
    }

    std::pair<double,double> eigenvalues;
    if (eigenvalue_tracker.values.empty())
      eigenvalues.first = eigenvalues.second = 1;
    else
    {
      eigenvalues.first = eigenvalue_tracker.values.front();
      eigenvalues.second = eigenvalue_tracker.values.back();
    }
    return eigenvalues;
  }
}

// re-implement the multigrid preconditioner in order to have more direct
// control over its individual components and avoid inner products and other
// expensive stuff
template <int dim, typename VectorType, typename MatrixType, typename TransferType, typename PreconditionerType>
class MultigridPreconditioner
{
public:
  MultigridPreconditioner(const DoFHandler<dim>                   &dof_handler,
                          const MGLevelObject<MatrixType>         &matrix,
                          const MGCoarseGridBase<VectorType>      &coarse,
                          const TransferType                      &transfer,
                          const MGLevelObject<PreconditionerType> &smooth,
                          const unsigned int                      n_cycles = 1)
    :
    dof_handler(&dof_handler),
    minlevel(0),
    maxlevel(dof_handler.get_triangulation().n_global_levels()-1),
    defect(minlevel, maxlevel),
    solution(minlevel, maxlevel),
    t(minlevel, maxlevel),
    defect2(minlevel, maxlevel),
    matrix(&matrix, typeid(*this).name()),
    coarse(&coarse, typeid(*this).name()),
    transfer(&transfer, typeid(*this).name()),
    smooth(&smooth, typeid(*this).name()),
    n_cycles (n_cycles)
  {
    AssertThrow(n_cycles == 1, ExcNotImplemented());
    for (unsigned int level = minlevel; level <= maxlevel; ++level)
    {
      matrix[level].initialize_dof_vector(solution[level]);
      defect[level] = solution[level];
      t[level] = solution[level];
      if (n_cycles > 1)
        defect2[level] = solution[level];
    }
  }

  template<class OtherVectorType>
  void vmult (OtherVectorType       &dst,
              const OtherVectorType &src) const
  {
    transfer->copy_to_mg(*dof_handler,
                         defect,
                         src);
    v_cycle(maxlevel);
    transfer->copy_from_mg(*dof_handler,
                           dst,
                           solution);
  }


private:
  /**
   * A pointer to the DoFHandler object
   */
  const SmartPointer<const DoFHandler<dim> > dof_handler;

  /**
   * Lowest level of cells.
   */
  unsigned int minlevel;

  /**
   * Highest level of cells.
   */
  unsigned int maxlevel;

  /**
   * Input vector for the cycle. Contains the defect of the outer method
   * projected to the multilevel vectors.
   */
  mutable MGLevelObject<VectorType> defect;

  /**
   * The solution update after the multigrid step.
   */
  mutable MGLevelObject<VectorType> solution;

  /**
   * Auxiliary vector.
   */
  mutable MGLevelObject<VectorType> t;

  /**
   * Auxiliary vector if more than 1 cycle is needed
   */
  mutable MGLevelObject<VectorType> defect2;

  /**
   * The matrix for each level.
   */
  SmartPointer<const MGLevelObject<MatrixType> > matrix;

  /**
   * The matrix for each level.
   */
  SmartPointer<const MGCoarseGridBase<VectorType> > coarse;

  /**
   * Object for grid tranfer.
   */
  SmartPointer<const TransferType> transfer;

  /**
   * The smoothing object.
   */
  SmartPointer<const MGLevelObject<PreconditionerType> > smooth;

  const unsigned int n_cycles;

  /**
   * Implements the v-cycle
   */
  void v_cycle(const unsigned int level) const
  {
    if (level==minlevel)
    {
      (*coarse)(level, solution[level], defect[level]);
      return;
    }

    (*smooth)[level].vmult(solution[level], defect[level]);
    (*matrix)[level].vmult_interface_down(t[level], solution[level]);
    t[level].sadd(-1.0, 1.0, defect[level]);

    // transfer to next level
    transfer->restrict_and_add(level, defect[level-1], t[level]);

    v_cycle(level-1);

    transfer->prolongate(level, t[level], solution[level-1]);
    solution[level] += t[level];
    // smooth on the negative part of the residual
    defect[level] *= -1.0;
    (*matrix)[level].vmult_add_interface_up(defect[level], solution[level]);
    (*smooth)[level].vmult(t[level], defect [level]);
    solution[level] -= t[level];
  }
};

#include "MultigridInputParameters.h"

#include "FE_Parameters.h"

template<int dim, typename value_type, typename Operator, typename OperatorData>
class MyMultigridPreconditioner : public PreconditionerBase<value_type>
{
public:
  MyMultigridPreconditioner(const MultigridData                &mg_data_in,
                            const DoFHandler<dim>              &dof_handler,
                            const Mapping<dim>                 &mapping,
                            const OperatorData                 &operator_data_in,
                            std::set<types::boundary_id> const &dirichlet_boundaries,
                            FEParameters<dim> const            &fe_param = FEParameters<dim>())
  {
    this->mg_data = mg_data_in;

    const parallel::Triangulation<dim> *tria =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation());

    // needed for continuous elements
    mg_constrained_dofs.clear();
    ZeroFunction<dim> zero_function(dof_handler.get_fe().n_components());
    typename FunctionMap<dim>::type dirichlet_boundary;
    for (std::set<types::boundary_id>::const_iterator it = dirichlet_boundaries.begin();
         it != dirichlet_boundaries.end(); ++it)
      dirichlet_boundary[*it] = &zero_function;
    mg_constrained_dofs.initialize(dof_handler, dirichlet_boundary);
    // needed for continuous elements

    mg_matrices.resize(0, tria->n_global_levels()-1);
    mg_smoother.resize(0, tria->n_global_levels()-1);

    for (unsigned int level = 0; level<tria->n_global_levels(); ++level)
    {
      mg_matrices[level].reinit(dof_handler, mapping, operator_data_in, mg_constrained_dofs, level,fe_param);

      typename SMOOTHER::AdditionalData smoother_data;

      mg_matrices[level].initialize_dof_vector(smoother_data.matrix_diagonal_inverse);
      mg_matrices[level].calculate_inverse_diagonal(smoother_data.matrix_diagonal_inverse);
      /*
      std::pair<double,double> eigenvalues = compute_eigenvalues(mg_matrices[level], smoother_data.matrix_diagonal_inverse);
      std::cout<<"Max EW = "<< eigenvalues.second <<" : Min EW = "<<eigenvalues.first<<std::endl;
      */
      if (level > 0)
      {
        smoother_data.smoothing_range = mg_data.smoother_smoothing_range;
        smoother_data.degree = mg_data.smoother_poly_degree;
        smoother_data.eig_cg_n_iterations = 20;
      }
      else
      {
        smoother_data.smoothing_range = 0.;
        if (mg_data.coarse_solver != MultigridCoarseGridSolver::coarse_chebyshev_smoother)
        {
          smoother_data.eig_cg_n_iterations = 0;
        }
        else // use Chebyshev smoother of high degree to solve the coarse grid problem approximately
        {
          std::pair<double,double> eigenvalues = compute_eigenvalues(mg_matrices[0], smoother_data.matrix_diagonal_inverse);
          smoother_data.max_eigenvalue = 1.1 * eigenvalues.second;
          smoother_data.smoothing_range = eigenvalues.second/eigenvalues.first*1.1;
          double sigma = (1.-std::sqrt(1./smoother_data.smoothing_range))/(1.+std::sqrt(1./smoother_data.smoothing_range));
          const double eps = 1e-3;
          smoother_data.degree = std::log(1./eps+std::sqrt(1./eps/eps-1))/std::log(1./sigma);
          smoother_data.eig_cg_n_iterations = 0;
        }
      }
      mg_smoother[level].initialize(mg_matrices[level], smoother_data);
    }

    switch (mg_data.coarse_solver)
    {
      case MultigridCoarseGridSolver::coarse_chebyshev_smoother:
      {
        mg_coarse.reset(new MGCoarseFromSmoother<parallel::distributed::Vector<typename Operator::value_type>, SMOOTHER>(mg_smoother[0], false));
        break;
      }
      case MultigridCoarseGridSolver::coarse_iterative_nopreconditioner:
      {
        mg_coarse.reset(new MGCoarseIterative<Operator>(mg_matrices[0],false)); // false: no Jacobi preconditioner
        break;
      }
      case MultigridCoarseGridSolver::coarse_iterative_jacobi:
      {
        mg_coarse.reset(new MGCoarseIterative<Operator>(mg_matrices[0],true)); // true: use Jacobi preconditioner
        break;
      }
      default:
        AssertThrow(false, ExcMessage("Unknown coarse-grid solver given"));
    }

    mg_transfer.set_operator(mg_matrices);
    mg_transfer.initialize_constraints(mg_constrained_dofs);
    mg_transfer.add_periodicity(operator_data_in.periodic_face_pairs_level0);
    mg_transfer.build(dof_handler);
    mg_transfer.set_restriction_type(false);

    multigrid_preconditioner.reset(new MultigridPreconditioner<dim, VECTOR_TYPE, Operator, MG_TRANSFER, SMOOTHER>
                                   (dof_handler, mg_matrices, *mg_coarse, mg_transfer, mg_smoother));
  }

  // multigrid preconditioner for compatible discretization of Laplace operator B * M^{-1} * B^{T} = (-div) * M^{-1} * grad
  MyMultigridPreconditioner(const MultigridData                &mg_data_in,
                            const DoFHandler<dim>              &dof_handler,
                            const DoFHandler<dim>              &dof_handler_additional,
                            const Mapping<dim>                 &mapping,
                            const OperatorData                 &operator_data_in,
                            FEParameters<dim> const            &fe_param = FEParameters<dim>())
  {
    this->mg_data = mg_data_in;

    const parallel::Triangulation<dim> *tria =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation());

    // TODO
    // only needed for continuous elements
    mg_constrained_dofs.clear();
    ZeroFunction<dim> zero_function(dof_handler.get_fe().n_components());
    typename FunctionMap<dim>::type dirichlet_boundary;
//    for (std::set<types::boundary_id>::const_iterator it = dirichlet_boundaries.begin();
//         it != dirichlet_boundaries.end(); ++it)
//      dirichlet_boundary[*it] = &zero_function;
    mg_constrained_dofs.initialize(dof_handler, dirichlet_boundary);
    // only needed for continuous elements

    mg_matrices.resize(0, tria->n_global_levels()-1);
    mg_smoother.resize(0, tria->n_global_levels()-1);

    for (unsigned int level = 0; level<tria->n_global_levels(); ++level)
    {
      mg_matrices[level].reinit(dof_handler,
                                dof_handler_additional,
                                mapping,
                                operator_data_in,
                                mg_constrained_dofs /*function reinit of compatible laplace operator does not use mg_constrained_dofs*/,
                                level,fe_param);

      typename SMOOTHER::AdditionalData smoother_data;

      mg_matrices[level].initialize_dof_vector(smoother_data.matrix_diagonal_inverse);
      mg_matrices[level].calculate_inverse_diagonal(smoother_data.matrix_diagonal_inverse);
      /*
      std::pair<double,double> eigenvalues = compute_eigenvalues(mg_matrices[level], smoother_data.matrix_diagonal_inverse);
      std::cout<<"Max EW = "<< eigenvalues.second <<" : Min EW = "<<eigenvalues.first<<std::endl;
      */
      if (level > 0)
      {
        smoother_data.smoothing_range = mg_data.smoother_smoothing_range;
        smoother_data.degree = mg_data.smoother_poly_degree;
        smoother_data.eig_cg_n_iterations = 20;
      }
      else
      {
        smoother_data.smoothing_range = 0.;
        if (mg_data.coarse_solver != MultigridCoarseGridSolver::coarse_chebyshev_smoother)
        {
          smoother_data.eig_cg_n_iterations = 0;
        }
        else // use Chebyshev smoother of high degree to solve the coarse grid problem approximately
        {
          std::pair<double,double> eigenvalues = compute_eigenvalues(mg_matrices[0], smoother_data.matrix_diagonal_inverse);
          smoother_data.max_eigenvalue = 1.1 * eigenvalues.second;
          smoother_data.smoothing_range = eigenvalues.second/eigenvalues.first*1.1;
          double sigma = (1.-std::sqrt(1./smoother_data.smoothing_range))/(1.+std::sqrt(1./smoother_data.smoothing_range));
          const double eps = 1e-3;
          smoother_data.degree = std::log(1./eps+std::sqrt(1./eps/eps-1))/std::log(1./sigma);
          smoother_data.eig_cg_n_iterations = 0;
        }
      }
      mg_smoother[level].initialize(mg_matrices[level], smoother_data);
    }

    switch (mg_data.coarse_solver)
    {
      case MultigridCoarseGridSolver::coarse_chebyshev_smoother:
      {
        mg_coarse.reset(new MGCoarseFromSmoother<parallel::distributed::Vector<typename Operator::value_type>, SMOOTHER>(mg_smoother[0], false));
        break;
      }
      case MultigridCoarseGridSolver::coarse_iterative_nopreconditioner:
      {
        mg_coarse.reset(new MGCoarseIterative<Operator>(mg_matrices[0],false)); // false: no Jacobi preconditioner
        break;
      }
      case MultigridCoarseGridSolver::coarse_iterative_jacobi:
      {
        mg_coarse.reset(new MGCoarseIterative<Operator>(mg_matrices[0],true)); // true: use Jacobi preconditioner
        break;
      }
      default:
        AssertThrow(false, ExcMessage("Unknown coarse-grid solver given"));
    }

    mg_transfer.set_operator(mg_matrices);
    mg_transfer.initialize_constraints(mg_constrained_dofs);
    mg_transfer.add_periodicity(operator_data_in.periodic_face_pairs_level0);
    mg_transfer.build(dof_handler);
    mg_transfer.set_restriction_type(false);

    multigrid_preconditioner.reset(new MultigridPreconditioner<dim, VECTOR_TYPE, Operator, MG_TRANSFER, SMOOTHER>
                                   (dof_handler, mg_matrices, *mg_coarse, mg_transfer, mg_smoother));
  }

  void vmult (parallel::distributed::Vector<value_type>        &dst,
              const parallel::distributed::Vector<value_type>  &src) const
  {
    multigrid_preconditioner->vmult(dst,src);
  }

private:
  MultigridData mg_data;
  MGConstrainedDoFs mg_constrained_dofs;

  MGLevelObject<Operator> mg_matrices;
  typedef MGTransferMF<dim,Operator> MG_TRANSFER;
  MG_TRANSFER mg_transfer;

  typedef parallel::distributed::Vector<typename Operator::value_type> VECTOR_TYPE;
  typedef PreconditionChebyshev<Operator,VECTOR_TYPE> SMOOTHER;
  MGLevelObject<SMOOTHER> mg_smoother;

  std_cxx11::shared_ptr<MGCoarseGridBase<VECTOR_TYPE> > mg_coarse;

  std_cxx11::shared_ptr<MultigridPreconditioner<dim, VECTOR_TYPE, Operator, MG_TRANSFER, SMOOTHER> > multigrid_preconditioner;

};

#endif /* INCLUDE_PRECONDITIONER_H_ */
