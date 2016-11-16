/*
 * Preconditioner.h
 *
 *  Created on: May 17, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_PRECONDITIONER_H_
#define INCLUDE_PRECONDITIONER_H_


#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/base/function_lib.h>

#include "InverseMassMatrix.h"
#include "MatrixOperatorBase.h"

template<typename value_type>
class PreconditionerBase
{
public:
  virtual ~PreconditionerBase(){}

  virtual void vmult(parallel::distributed::Vector<value_type>        &dst,
                     const parallel::distributed::Vector<value_type>  &src) const = 0;

  virtual void update(MatrixOperatorBase const *matrix_operator) = 0;
};

template<int dim, int fe_degree, typename value_type,  int n_components=dim>
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
    inverse_mass_matrix_operator.apply(dst,src);
  }

  void update(MatrixOperatorBase const * /*matrix_operator*/) {} // do nothing

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

  void update(MatrixOperatorBase const * /*matrix_operator*/) {} // do nothing

private:
  std_cxx11::shared_ptr<InverseMassMatrixOperator<dim,fe_degree,value_type,n_components> > inverse_mass_matrix_operator;
};


template<typename value_type, typename UnderlyingOperator>
class JacobiPreconditioner : public PreconditionerBase<value_type>
{
public:
  JacobiPreconditioner(UnderlyingOperator const &underlying_operator)
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

  void update(MatrixOperatorBase const * matrix_operator)
  {
    UnderlyingOperator const *underlying_operator = dynamic_cast<UnderlyingOperator const *>(matrix_operator);
    if(underlying_operator)
      underlying_operator->calculate_inverse_diagonal(inverse_diagonal);
    else
      AssertThrow(false,ExcMessage("Jacobi preconditioner: UnderlyingOperator and MatrixOperator are not compatible!"));
  }

  unsigned int get_size_of_diagonal()
  {
    return inverse_diagonal.size();
  }

private:
  parallel::distributed::Vector<value_type> inverse_diagonal;
};

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

  void update_preconditioner(const Operator &matrix)
  {
    if(use_preconditioner)
    {
      std_cxx11::shared_ptr<JacobiPreconditioner<typename Operator::value_type,Operator> > precon =
          std::dynamic_pointer_cast<JacobiPreconditioner<typename Operator::value_type,Operator> > (preconditioner);

      precon->recalculate_diagonal(coarse_matrix);
    }
  }

  virtual void operator() (const unsigned int                                                 ,
                           parallel::distributed::Vector<typename Operator::value_type>       &dst,
                           const parallel::distributed::Vector<typename Operator::value_type> &src) const
  {
    const double abs_tol = 1.e-20;
    const double rel_tol = 1.e-6; //1.e-4;
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

#include <deal.II/lac/solver_gmres.h>

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
          std::dynamic_pointer_cast<JacobiPreconditioner<typename Operator::value_type,Operator> > (preconditioner);
      AssertDimension(precon->get_size_of_diagonal(), coarse_matrix.m());
    }
  }

  virtual ~MGCoarseGMRES()
  {}

  void update_preconditioner(const Operator &underlying_operator)
  {
    if (use_preconditioner)
    {
//      std_cxx11::shared_ptr<JacobiPreconditioner<typename Operator::value_type,Operator> > precon =
//          std::dynamic_pointer_cast<JacobiPreconditioner<typename Operator::value_type,Operator> > (preconditioner);

      preconditioner->update(&underlying_operator);
    }
  }

  virtual void operator() (const unsigned int                                                 ,
                           parallel::distributed::Vector<typename Operator::value_type>       &dst,
                           const parallel::distributed::Vector<typename Operator::value_type> &src) const
  {
    const double abs_tol = 1.e-20;
    const double rel_tol = 1.e-5; //1.e-4;
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

  template<typename Number>
  struct EigenvalueTracker
  {
  public:
    void slot(const std::vector<Number> &eigenvalues)
    {
      values = eigenvalues;
    }

    std::vector<Number> values;
  };


  // manually compute eigenvalues for the coarsest level for proper setup of the Chebyshev iteration
  template <typename Operator>
  std::pair<std::complex<double>,std::complex<double> >
  compute_eigenvalues_gmres(const Operator &op,
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
    EigenvalueTracker<std::complex<double> > eigenvalue_tracker;
    SolverGMRES<parallel::distributed::Vector<value_type> > solver (control);
    solver.connect_eigenvalues_slot(std_cxx11::bind(&EigenvalueTracker<std::complex<double> >::slot,
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

    std::pair<std::complex<double>,std::complex<double> > eigenvalues;
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

//#include "FE_Parameters.h"

template<int dim, typename value_type, typename Operator>
class MyMultigridPreconditionerBase : public PreconditionerBase<value_type>
{
public:
  MyMultigridPreconditionerBase()
    :
    n_global_levels(0)
  {}

  virtual ~MyMultigridPreconditionerBase(){};

  virtual void update(MatrixOperatorBase const * /*matrix_operator*/) = 0;

  virtual void resize_level_objects()
  {
    mg_matrices.resize(0, n_global_levels -1);
    mg_smoother.resize(0, n_global_levels -1);
  }

  virtual void initialize_smoother(unsigned int level)
  {
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
    else // coarse grid
    {
      if (mg_data.coarse_solver == MultigridCoarseGridSolver::ChebyshevSmoother)
      {
        // use Chebyshev smoother of high degree to solve the coarse grid problem approximately
        std::pair<double,double> eigenvalues = compute_eigenvalues(mg_matrices[0], smoother_data.matrix_diagonal_inverse);
        smoother_data.max_eigenvalue = 1.1 * eigenvalues.second;
        smoother_data.smoothing_range = eigenvalues.second/eigenvalues.first*1.1;
        double sigma = (1.-std::sqrt(1./smoother_data.smoothing_range))/(1.+std::sqrt(1./smoother_data.smoothing_range));
        const double eps = 1e-3;
        smoother_data.degree = std::log(1./eps+std::sqrt(1./eps/eps-1))/std::log(1./sigma);
      }
      smoother_data.eig_cg_n_iterations = 0;
    }

    mg_smoother[level].initialize(mg_matrices[level], smoother_data);
  }

  virtual void initialize_coarse_solver()
  {
    switch (mg_data.coarse_solver)
    {
      case MultigridCoarseGridSolver::ChebyshevSmoother:
      {
        mg_coarse.reset(new MGCoarseFromSmoother<parallel::distributed::Vector<typename Operator::value_type>, SMOOTHER>(mg_smoother[0], false));
        break;
      }
      case MultigridCoarseGridSolver::PCG_NoPreconditioner:
      {
        mg_coarse.reset(new MGCoarsePCG<Operator>(mg_matrices[0],false)); // false: no Jacobi preconditioner
        break;
      }
      case MultigridCoarseGridSolver::PCG_Jacobi:
      {
        mg_coarse.reset(new MGCoarsePCG<Operator>(mg_matrices[0],true)); // true: use Jacobi preconditioner
        break;
      }
      case MultigridCoarseGridSolver::GMRES_NoPreconditioner:
      {
        mg_coarse.reset(new MGCoarseGMRES<Operator>(mg_matrices[0],false)); // false: no Jacobi preconditioner
        break;
      }
      case MultigridCoarseGridSolver::GMRES_Jacobi:
      {
        mg_coarse.reset(new MGCoarseGMRES<Operator>(mg_matrices[0],true)); // true: use Jacobi preconditioner
        break;
      }
      default:
        AssertThrow(false, ExcMessage("Unknown coarse-grid solver given"));
    }
  }

  virtual void initialize_mg_transfer(const DoFHandler<dim> &dof_handler,
                                      const std::vector<GridTools::PeriodicFacePair<typename
                                        Triangulation<dim>::cell_iterator> > &periodic_face_pairs_level0)
  {
    mg_transfer.set_operator(mg_matrices);
    mg_transfer.initialize_constraints(mg_constrained_dofs);
    mg_transfer.add_periodicity(periodic_face_pairs_level0);
    mg_transfer.build(dof_handler);
  }

  void vmult (parallel::distributed::Vector<value_type>        &dst,
              const parallel::distributed::Vector<value_type>  &src) const
  {
    multigrid_preconditioner->vmult(dst,src);
  }

  virtual void apply_smoother_on_fine_level(parallel::distributed::Vector<typename Operator::value_type>        &dst,
                                            const parallel::distributed::Vector<typename Operator::value_type>  &src) const
  {
    this->mg_smoother[this->mg_smoother.max_level()].vmult(dst,src);
  }

protected:
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

  unsigned int n_global_levels;
};

template<int dim, typename value_type, typename Operator, typename OperatorData>
class MyMultigridPreconditioner : public MyMultigridPreconditionerBase<dim,value_type,Operator>
{
public:
  MyMultigridPreconditioner(){}

  virtual ~MyMultigridPreconditioner(){}

  virtual void update(MatrixOperatorBase const * /*matrix_operator*/){}

  void initialize(const MultigridData                             &mg_data_in,
                  const DoFHandler<dim>                           &dof_handler,
                  const Mapping<dim>                              &mapping,
                  const OperatorData                              &operator_data_in,
                  std::map<types::boundary_id,
                    std_cxx11::shared_ptr<Function<dim> > > const &dirichlet_bc)
  {
    this->mg_data = mg_data_in;

    const parallel::Triangulation<dim> *tria =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation());

    // needed for continuous elements
    this->mg_constrained_dofs.clear();
    std::set<types::boundary_id> dirichlet_boundary;
    for(typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::const_iterator
        it = dirichlet_bc.begin(); it != dirichlet_bc.end(); ++it)
      dirichlet_boundary.insert(it->first);
    this->mg_constrained_dofs.initialize(dof_handler);
    this->mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary);
    // needed for continuous elements

    this->n_global_levels = tria->n_global_levels();
    this->resize_level_objects();

    for (unsigned int level = 0; level<this->n_global_levels; ++level)
    {
      initialize_mg_matrix(level, dof_handler, mapping, operator_data_in);

      this->initialize_smoother(level);
    }

    this->initialize_coarse_solver();

    this->initialize_mg_transfer(dof_handler, operator_data_in.periodic_face_pairs_level0);

    typedef MGTransferMF<dim,Operator> MG_TRANSFER;
    typedef parallel::distributed::Vector<typename Operator::value_type> VECTOR_TYPE;
    typedef PreconditionChebyshev<Operator,VECTOR_TYPE> SMOOTHER;

    this->multigrid_preconditioner.reset(new MultigridPreconditioner<dim, VECTOR_TYPE, Operator, MG_TRANSFER, SMOOTHER>
                                   (dof_handler, this->mg_matrices, *(this->mg_coarse), this->mg_transfer, this->mg_smoother));
  }

  virtual void initialize_mg_matrix(unsigned int            level,
                                    const DoFHandler<dim>   &dof_handler,
                                    const Mapping<dim>      &mapping,
                                    const OperatorData      &operator_data_in)
  {
    this->mg_matrices[level].reinit(dof_handler, mapping, operator_data_in, this->mg_constrained_dofs, level);
  }
};

/*
 *  Multigrid preconditioner for velocity convection-diffusion operator of
 *  the incompressible Navier-Stokes equations
 */
template<int dim, typename value_type, typename Operator, typename OperatorData, typename UnderlyingOperator>
class MyMultigridPreconditionerVelocityConvectionDiffusion : public MyMultigridPreconditionerBase<dim,value_type,Operator>
{
public:
  MyMultigridPreconditionerVelocityConvectionDiffusion(){}

  virtual ~MyMultigridPreconditionerVelocityConvectionDiffusion(){};

  void initialize(const MultigridData                             &mg_data_in,
                  const DoFHandler<dim>                           &dof_handler,
                  const Mapping<dim>                              &mapping,
                  const OperatorData                              &operator_data_in,
                  std::map<types::boundary_id,
                    std_cxx11::shared_ptr<Function<dim> > > const &dirichlet_bc)
  {
    this->mg_data = mg_data_in;

    const parallel::Triangulation<dim> *tria =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation());

    // needed for continuous elements
    this->mg_constrained_dofs.clear();
    std::set<types::boundary_id> dirichlet_boundary;
    for(typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::const_iterator
        it = dirichlet_bc.begin(); it != dirichlet_bc.end(); ++it)
      dirichlet_boundary.insert(it->first);
    this->mg_constrained_dofs.initialize(dof_handler);
    this->mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary);
    // needed for continuous elements

    this->n_global_levels = tria->n_global_levels();
    this->resize_level_objects();

    for (unsigned int level = 0; level<this->n_global_levels; ++level)
    {
      initialize_mg_matrix(level, dof_handler, mapping, operator_data_in);

      this->initialize_smoother(level);
    }

    this->initialize_coarse_solver();

    this->initialize_mg_transfer(dof_handler, operator_data_in.periodic_face_pairs_level0);

    typedef MGTransferMF<dim,Operator> MG_TRANSFER;
    typedef parallel::distributed::Vector<typename Operator::value_type> VECTOR_TYPE;
    typedef PreconditionChebyshev<Operator,VECTOR_TYPE> SMOOTHER;

    this->multigrid_preconditioner.reset(new MultigridPreconditioner<dim, VECTOR_TYPE, Operator, MG_TRANSFER, SMOOTHER>
                                   (dof_handler, this->mg_matrices, *(this->mg_coarse), this->mg_transfer, this->mg_smoother));
  }

  virtual void initialize_mg_matrix(unsigned int            level,
                                    const DoFHandler<dim>   &dof_handler,
                                    const Mapping<dim>      &mapping,
                                    const OperatorData      &operator_data_in)
  {
    // initialize mg_matrix for given level
    this->mg_matrices[level].reinit(dof_handler, mapping, operator_data_in, level);

    // set evaluation time
    this->mg_matrices[level].set_evaluation_time(0.0);
  }

  /*
   *  This function updates the multigrid preconditioner.
   *
   */
  virtual void update(MatrixOperatorBase const * matrix_operator)
  {
    UnderlyingOperator const
      *underlying_operator = dynamic_cast<UnderlyingOperator const *>(matrix_operator);

    if(underlying_operator)
    {
      parallel::distributed::Vector<value_type> const & vector_linearization = underlying_operator->get_solution_linearization();

      // convert value_type --> Operator::value_type, i.e., double --> float
      parallel::distributed::Vector<typename Operator::value_type> vector_multigrid_type;
      vector_multigrid_type = vector_linearization;

      update_mg_matrices(vector_multigrid_type,
                         underlying_operator->get_evaluation_time());
      update_smoothers();
      update_coarse_solver();
    }
    else
    {
      AssertThrow(false,ExcMessage("Multigrid preconditioner: UnderlyingOperator and MatrixOperator are not compatible!"));
    }
  }

  /*
   *  This function updates mg_matrices
   *  To do this, two functions are called:
   *   - set_vector_linearization
   *   - set_evaluation_time
   */
  void update_mg_matrices(parallel::distributed::Vector<typename Operator::value_type> const &vector_linearization,
                          double const                                                       &evaluation_time)
  {
    set_vector_linearization(vector_linearization);
    set_evaluation_time(evaluation_time);
  }

  /*
   *  This function updates vector_linearization.
   *  In order to update mg_matrices[level] this function has to be called.
   */
  void set_vector_linearization(parallel::distributed::Vector<typename Operator::value_type> const &vector_linearization)
  {
    for (int level = this->n_global_levels-1; level>=0; --level)
    {
      if(level == (int)this->n_global_levels-1) // finest level
      {
        this->mg_matrices[level].set_solution_linearization(vector_linearization);
      }
      else // all coarser levels
      {
        // restrict vector_linearization from fine to coarse level
        parallel::distributed::Vector<typename Operator::value_type> & vector_fine_level = this->mg_matrices[level+1].get_solution_linearization();
        parallel::distributed::Vector<typename Operator::value_type> & vector_coarse_level = this->mg_matrices[level].get_solution_linearization();
        // set vector_coarse_level to zero since ..._add is called
        vector_coarse_level = 0.0;
        this->mg_transfer.restrict_and_add(level+1,vector_coarse_level,vector_fine_level);
      }
    }
  }

  /*
   *  This function updates the evaluation time.
   *  In order to update mg_matrices[level] this function has to be called.
   *  (This is due to the fact that the linearized convective term does not
   *  only depend on the linearized velocity field but also on Dirichlet boundary
   *  data which itself depends on the current time.)
   */
  void set_evaluation_time(double const &evaluation_time)
  {
    for (int level = this->n_global_levels-1; level>=0; --level)
    {
      this->mg_matrices[level].set_evaluation_time(evaluation_time);
    }
  }

  /*
   *  This function updates the smoother for all levels of the multigrid
   *  algorithm.
   *  The prerequisite to call this function is that mg_matrices[level] have
   *  been updated.
   */
  void update_smoothers()
  {
    for (unsigned int level = 0; level<this->n_global_levels; ++level)
    {
      this->initialize_smoother(level);
    }
  }

  /*
   *  This function updates the (preconditioner of the) coarse grid solver.
   *  The prerequisite to call this function is that mg_matrices[0] has
   *  been updated.
   */
  void update_coarse_solver()
  {
    if(this->mg_data.coarse_solver == MultigridCoarseGridSolver::GMRES_Jacobi)
    {
      std_cxx11::shared_ptr<MGCoarseGMRES<Operator> >
        coarse_solver = std::dynamic_pointer_cast<MGCoarseGMRES<Operator> >(this->mg_coarse);

      coarse_solver->update_preconditioner(this->mg_matrices[0]);
    }
  }

  /*
   *  This function initializes the Chebyshev smoother
   *  by performing a few GMRES iterations.
   */
  virtual void initialize_smoother(unsigned int level)
  {
    typename SMOOTHER::AdditionalData smoother_data;

    this->mg_matrices[level].initialize_dof_vector(smoother_data.matrix_diagonal_inverse);
    this->mg_matrices[level].calculate_inverse_diagonal(smoother_data.matrix_diagonal_inverse);

    const double factor = 1.1;

    if (level > 0)
    {
      std::pair<std::complex<double>,std::complex<double> > eigenvalues = compute_eigenvalues_gmres(this->mg_matrices[level], smoother_data.matrix_diagonal_inverse);
      smoother_data.max_eigenvalue = factor * std::abs(eigenvalues.second);
      smoother_data.smoothing_range = this->mg_data.smoother_smoothing_range;
      smoother_data.degree = this->mg_data.smoother_poly_degree;
      smoother_data.eig_cg_n_iterations = 0;
    }
    else // coarse grid
    {
      if (this->mg_data.coarse_solver == MultigridCoarseGridSolver::ChebyshevSmoother)
      {
        // use Chebyshev smoother of high degree to solve the coarse grid problem approximately
        std::pair<std::complex<double>,std::complex<double> > eigenvalues = compute_eigenvalues_gmres(this->mg_matrices[level], smoother_data.matrix_diagonal_inverse);
        smoother_data.max_eigenvalue = factor * std::abs(eigenvalues.second);
        smoother_data.smoothing_range = factor * std::abs(eigenvalues.second)/std::abs(eigenvalues.first);
        double sigma = (1.-std::sqrt(1./smoother_data.smoothing_range))/(1.+std::sqrt(1./smoother_data.smoothing_range));
        const double eps = 1e-3;
        smoother_data.degree = std::log(1./eps+std::sqrt(1./eps/eps-1))/std::log(1./sigma);
      }
      smoother_data.eig_cg_n_iterations = 0;
    }

    this->mg_smoother[level].initialize(this->mg_matrices[level], smoother_data);
  }

private:

  typedef parallel::distributed::Vector<typename Operator::value_type> VECTOR_TYPE;
  typedef PreconditionChebyshev<Operator,VECTOR_TYPE> SMOOTHER;
};



#include "GMRESSmoother.h"

template<int dim, typename value_type, typename Operator, typename OperatorData,typename UnderlyingOperator>
class MyMultigridPreconditionerGMRESSmoother : public PreconditionerBase<value_type>
{
public:
  MyMultigridPreconditionerGMRESSmoother()
    :
    n_global_levels(0)
  {}

  virtual ~MyMultigridPreconditionerGMRESSmoother(){};

  void initialize(const MultigridData                             &mg_data_in,
                  const DoFHandler<dim>                           &dof_handler,
                  const Mapping<dim>                              &mapping,
                  const OperatorData                              &operator_data_in,
                  std::map<types::boundary_id,
                    std_cxx11::shared_ptr<Function<dim> > > const &dirichlet_bc)
  {
    this->mg_data = mg_data_in;

    const parallel::Triangulation<dim> *tria =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation());

    // needed for continuous elements
    this->mg_constrained_dofs.clear();
    std::set<types::boundary_id> dirichlet_boundary;
    for(typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::const_iterator
        it = dirichlet_bc.begin(); it != dirichlet_bc.end(); ++it)
      dirichlet_boundary.insert(it->first);
    this->mg_constrained_dofs.initialize(dof_handler);
    this->mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary);
    // needed for continuous elements

    this->n_global_levels = tria->n_global_levels();
    this->resize_level_objects();

    for (unsigned int level = 0; level<this->n_global_levels; ++level)
    {
      initialize_mg_matrix(level, dof_handler, mapping, operator_data_in);

      this->initialize_smoother(level);
    }

    this->initialize_coarse_solver();

    this->initialize_mg_transfer(dof_handler, operator_data_in.periodic_face_pairs_level0);

    this->multigrid_preconditioner.reset(new MultigridPreconditioner<dim, VECTOR_TYPE, Operator, MG_TRANSFER, SMOOTHER>
        (dof_handler, this->mg_matrices, *(this->mg_coarse), this->mg_transfer, this->mg_smoother));
  }

  void resize_level_objects()
  {
    this->mg_matrices.resize(0, this->n_global_levels-1);
    this->mg_smoother.resize(0, this->n_global_levels-1);
  }

  void initialize_smoother(unsigned int level)
  {
    mg_smoother[level].initialize(mg_matrices[level]);
  }

  void initialize_mg_matrix(unsigned int            level,
                            const DoFHandler<dim>   &dof_handler,
                            const Mapping<dim>      &mapping,
                            const OperatorData      &operator_data_in)
  {
    // initialize mg_matrix for given level
    this->mg_matrices[level].reinit(dof_handler, mapping, operator_data_in, level);

    // set evaluation time
    this->mg_matrices[level].set_evaluation_time(0.0);
  }

  /*
   *  This function updates the multigrid preconditioner.
   *
   */
  virtual void update(MatrixOperatorBase const * matrix_operator)
  {
    UnderlyingOperator const
      *underlying_operator = dynamic_cast<UnderlyingOperator const *>(matrix_operator);
    if(underlying_operator)
    {
      parallel::distributed::Vector<value_type> const & vector_linearization = underlying_operator->get_solution_linearization();

      // convert value_type --> Operator::value_type, i.e., double --> float
      parallel::distributed::Vector<typename Operator::value_type> vector_multigrid_type;
      vector_multigrid_type = vector_linearization;

      update_mg_matrices(vector_multigrid_type,
                         underlying_operator->get_evaluation_time());
      update_smoothers();
      update_coarse_solver();
    }
    else
      AssertThrow(false,ExcMessage("Multigrid preconditioner: UnderlyingOperator and MatrixOperator are not compatible!"));
  }

  /*
   *  This function updates mg_matrices
   *  To do this, two functions are called:
   *   - set_vector_linearization
   *   - set_evaluation_time
   */
  void update_mg_matrices(parallel::distributed::Vector<typename Operator::value_type> const &vector_linearization,
                          double const                                    &evaluation_time)
  {
    set_vector_linearization(vector_linearization);
    set_evaluation_time(evaluation_time);
  }

  /*
   *  This function updates vector_linearization.
   *  In order to update mg_matrices[level], this function has to be called.
   */
  void set_vector_linearization(parallel::distributed::Vector<typename Operator::value_type> const &vector_linearization)
  {
    for (int level = this->n_global_levels-1; level>=0; --level)
    {
      if(level == (int)this->n_global_levels-1) // finest level
      {
        this->mg_matrices[level].set_solution_linearization(vector_linearization);
      }
      else // all coarser levels
      {
        // restrict vector_linearization from fine to coarse level
        parallel::distributed::Vector<typename Operator::value_type> & vector_fine_level = this->mg_matrices[level+1].get_solution_linearization();
        parallel::distributed::Vector<typename Operator::value_type> & vector_coarse_level = this->mg_matrices[level].get_solution_linearization();
        // set vector_coarse_level to zero since ..._add is called
        vector_coarse_level = 0.0;
        this->mg_transfer.restrict_and_add(level+1,vector_coarse_level,vector_fine_level);
      }
    }
  }

  /*
   *  This function updates the evaluation time.
   *  In order to update mg_matrices[level] this function has to be called.
   *  (This is due to the fact that the linearized convective term does not
   *  only depend on the linearized velocity field but also on Dirichlet boundary
   *  data which itself depends on the current time.)
   */
  void set_evaluation_time(double const &evaluation_time)
  {
    for (int level = this->n_global_levels-1; level>=0; --level)
    {
      this->mg_matrices[level].set_evaluation_time(evaluation_time);
    }
  }

  void initialize_coarse_solver()
  {
    switch (mg_data.coarse_solver)
    {
      case MultigridCoarseGridSolver::ChebyshevSmoother:
      {
        mg_coarse.reset(new MGCoarseFromSmoother<parallel::distributed::Vector<typename Operator::value_type>, SMOOTHER>(mg_smoother[0], false));
        break;
      }
      case MultigridCoarseGridSolver::PCG_NoPreconditioner:
      {
        mg_coarse.reset(new MGCoarsePCG<Operator>(mg_matrices[0],false)); // false: no Jacobi preconditioner
        break;
      }
      case MultigridCoarseGridSolver::PCG_Jacobi:
      {
        mg_coarse.reset(new MGCoarsePCG<Operator>(mg_matrices[0],true)); // true: use Jacobi preconditioner
        break;
      }
      case MultigridCoarseGridSolver::GMRES_NoPreconditioner:
      {
        mg_coarse.reset(new MGCoarseGMRES<Operator>(mg_matrices[0],false)); // false: no Jacobi preconditioner
        break;
      }
      case MultigridCoarseGridSolver::GMRES_Jacobi:
      {
        mg_coarse.reset(new MGCoarseGMRES<Operator>(mg_matrices[0],true)); // true: use Jacobi preconditioner
        break;
      }
      default:
        AssertThrow(false, ExcMessage("Unknown coarse-grid solver given"));
    }
  }

  /*
   *  This function updates the (preconditioner of the) coarse grid solver.
   *  The prerequisite to call this function is that mg_matrices[0] has
   *  been updated.
   */
  void update_coarse_solver()
  {
    if(this->mg_data.coarse_solver == MultigridCoarseGridSolver::GMRES_Jacobi)
    {
      std_cxx11::shared_ptr<MGCoarseGMRES<Operator> >
        coarse_solver = std::dynamic_pointer_cast<MGCoarseGMRES<Operator> >(this->mg_coarse);

      coarse_solver->update_preconditioner(this->mg_matrices[0]);
    }
  }

  /*
   *  This function updates the smoother for all levels of the multigrid
   *  algorithm.
   *  The prerequisite to call this function is that mg_matrices[level] have
   *  been updated.
   */
  void update_smoothers()
  {
    for (unsigned int level = 0; level<this->n_global_levels; ++level)
    {
      mg_smoother[level].update();
    }
  }

  virtual void initialize_mg_transfer(const DoFHandler<dim> &dof_handler,
                                      const std::vector<GridTools::PeriodicFacePair<typename
                                        Triangulation<dim>::cell_iterator> > &periodic_face_pairs_level0)
  {
    mg_transfer.set_operator(mg_matrices);
    mg_transfer.initialize_constraints(mg_constrained_dofs);
    mg_transfer.add_periodicity(periodic_face_pairs_level0);
    mg_transfer.build(dof_handler);
  }

  void vmult (parallel::distributed::Vector<value_type>        &dst,
              const parallel::distributed::Vector<value_type>  &src) const
  {
    multigrid_preconditioner->vmult(dst,src);
  }

  void apply_smoother_on_fine_level(parallel::distributed::Vector<typename Operator::value_type>        &dst,
                                    const parallel::distributed::Vector<typename Operator::value_type>  &src) const
  {
    this->mg_smoother[this->mg_smoother.max_level()].vmult(dst,src);
  }

protected:
  MultigridData mg_data;
  MGConstrainedDoFs mg_constrained_dofs;

  MGLevelObject<Operator> mg_matrices;
  typedef MGTransferMF<dim,Operator> MG_TRANSFER;
  MG_TRANSFER mg_transfer;

  typedef parallel::distributed::Vector<typename Operator::value_type> VECTOR_TYPE;
  typedef GMRESSmoother<Operator,VECTOR_TYPE> SMOOTHER;
  MGLevelObject<SMOOTHER> mg_smoother;

  std_cxx11::shared_ptr<MGCoarseGridBase<VECTOR_TYPE> > mg_coarse;

  std_cxx11::shared_ptr<MultigridPreconditioner<dim, VECTOR_TYPE, Operator, MG_TRANSFER, SMOOTHER> > multigrid_preconditioner;

  unsigned int n_global_levels;
};


/*
 *   Multigrid preconditioner for compatible discretization
 *   of Laplace operator occuring in the Schur-complement of
 *   the incompressible (Navier-)Stokes equations
 *   Operator: B * M^{-1} * B^{T} = (-div) * M^{-1} * grad
 *   where M^{-1} is the inverse velocity mass matrix
 */
template<int dim, typename value_type, typename Operator, typename OperatorData>
class MyMultigridPreconditionerCompatibleLaplace : public MyMultigridPreconditionerBase<dim,value_type,Operator>
{
public:
  MyMultigridPreconditionerCompatibleLaplace(){}

  void update(MatrixOperatorBase const * /*matrix_operator*/){}

  void initialize(const MultigridData      &mg_data_in,
                  const DoFHandler<dim>    &dof_handler,
                  const DoFHandler<dim>    &dof_handler_additional,
                  const Mapping<dim>       &mapping,
                  const OperatorData       &operator_data_in)
  {
    this->mg_data = mg_data_in;

    const parallel::Triangulation<dim> *tria =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation());

    // only needed for continuous elements
    this->mg_constrained_dofs.clear();
    std::set<types::boundary_id> dirichlet_boundary;
//    for(typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::const_iterator
//        it = dirichlet_bc.begin(); it != dirichlet_bc.end(); ++it)
//      dirichlet_boundary.insert(it->first);
    this->mg_constrained_dofs.initialize(dof_handler);
    this->mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary);
    // only needed for continuous elements

    this->n_global_levels = tria->n_global_levels();
    this->resize_level_objects();

    for (unsigned int level = 0; level<tria->n_global_levels(); ++level)
    {
      this->mg_matrices[level].reinit(dof_handler,
                                      dof_handler_additional,
                                      mapping,
                                      operator_data_in,
                                      this->mg_constrained_dofs,
                                      level);

      this->initialize_smoother(level);
    }

    this->initialize_coarse_solver();

    this->initialize_mg_transfer(dof_handler, operator_data_in.periodic_face_pairs_level0);

    typedef MGTransferMF<dim,Operator> MG_TRANSFER;
    typedef parallel::distributed::Vector<typename Operator::value_type> VECTOR_TYPE;
    typedef PreconditionChebyshev<Operator,VECTOR_TYPE> SMOOTHER;

    this->multigrid_preconditioner.reset(new MultigridPreconditioner<dim, VECTOR_TYPE, Operator, MG_TRANSFER, SMOOTHER>
                                   (dof_handler, this->mg_matrices, *(this->mg_coarse), this->mg_transfer, this->mg_smoother));
  }
};

#endif /* INCLUDE_PRECONDITIONER_H_ */
