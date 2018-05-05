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

#include "./preconditioner_base.h"

#include "../solvers_and_preconditioners/jacobi_preconditioner.h"
#include "../solvers_and_preconditioners/block_jacobi_preconditioner.h"


#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_precondition.h>

enum class PreconditionerCoarseGridSolver
{
  None,
  PointJacobi,
  BlockJacobi
};

template<typename Operator>
class MGCoarsePCG : public MGCoarseGridBase<parallel::distributed::Vector<typename Operator::value_type> >
{
public:
  struct AdditionalData
  {
    /**
     * Constructor.
     */
    AdditionalData()
     :
     preconditioner(PreconditionerCoarseGridSolver::None)
    {}

    // preconditioner
    PreconditionerCoarseGridSolver preconditioner;
  };

  MGCoarsePCG(Operator const       &matrix,
              AdditionalData const &additional_data)
    :
    coarse_matrix (matrix),
    use_preconditioner(false)
  {
    if (additional_data.preconditioner == PreconditionerCoarseGridSolver::PointJacobi)
    {
      use_preconditioner = true;

      preconditioner.reset(new JacobiPreconditioner<typename Operator::value_type,Operator>(coarse_matrix));
      std::shared_ptr<JacobiPreconditioner<typename Operator::value_type,Operator> > precon =
          std::dynamic_pointer_cast<JacobiPreconditioner<typename Operator::value_type,Operator> > (preconditioner);
      AssertDimension(precon->get_size_of_diagonal(), coarse_matrix.m());
    }
    else if(additional_data.preconditioner == PreconditionerCoarseGridSolver::BlockJacobi)
    {
      use_preconditioner = true;

      preconditioner.reset(new BlockJacobiPreconditioner<typename Operator::value_type,Operator>(coarse_matrix));
    }
    else
    {
      AssertThrow(additional_data.preconditioner == PreconditionerCoarseGridSolver::None ||
                  additional_data.preconditioner == PreconditionerCoarseGridSolver::PointJacobi ||
                  additional_data.preconditioner == PreconditionerCoarseGridSolver::BlockJacobi,
                  ExcMessage("Specified preconditioner for PCG coarse grid solver not implemented."));
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
  std::shared_ptr<PreconditionerBase<typename Operator::value_type> > preconditioner;
  mutable GrowingVectorMemory<parallel::distributed::Vector<typename Operator::value_type> > solver_memory;
  bool use_preconditioner;
};


template<typename Operator>
class MGCoarseGMRES : public MGCoarseGridBase<parallel::distributed::Vector<typename Operator::value_type> >
{
public:
  struct AdditionalData
  {
    /**
     * Constructor.
     */
    AdditionalData()
     :
     preconditioner(PreconditionerCoarseGridSolver::None)
    {}

    // preconditioner
    PreconditionerCoarseGridSolver preconditioner;
  };

  MGCoarseGMRES(Operator const       &matrix,
                AdditionalData const &additional_data)
    :
    coarse_matrix (matrix),
    use_preconditioner(false)
  {
    if (additional_data.preconditioner == PreconditionerCoarseGridSolver::PointJacobi)
    {
      use_preconditioner = true;

      preconditioner.reset(new JacobiPreconditioner<typename Operator::value_type,Operator>(coarse_matrix));
      std::shared_ptr<JacobiPreconditioner<typename Operator::value_type,Operator> > precon =
          std::dynamic_pointer_cast<JacobiPreconditioner<typename Operator::value_type,Operator> > (preconditioner);
      AssertDimension(precon->get_size_of_diagonal(), coarse_matrix.m());
    }
    else if(additional_data.preconditioner == PreconditionerCoarseGridSolver::BlockJacobi)
    {
      use_preconditioner = true;

      preconditioner.reset(new BlockJacobiPreconditioner<typename Operator::value_type,Operator>(coarse_matrix));
    }
    else
    {
      AssertThrow(additional_data.preconditioner == PreconditionerCoarseGridSolver::None ||
                  additional_data.preconditioner == PreconditionerCoarseGridSolver::PointJacobi ||
                  additional_data.preconditioner == PreconditionerCoarseGridSolver::BlockJacobi,
                  ExcMessage("Specified preconditioner for PCG coarse grid solver not implemented."));
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
    additional_data.max_n_tmp_vectors = 200;
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

    if(solver_control.last_step() > 2*additional_data.max_n_tmp_vectors)
      std::cout << "Number of iterations of GMRES coarse grid solver significantly larger than max_n_tmp_vectors." << std::endl;

//    std::cout << "Iterations coarse grid solver = " << solver_control.last_step() << std::endl;
  }

private:
  const Operator &coarse_matrix;
  std::shared_ptr<PreconditionerBase<typename Operator::value_type> > preconditioner;
  mutable GrowingVectorMemory<parallel::distributed::Vector<typename Operator::value_type> > solver_memory;
  bool use_preconditioner;
};


template<typename Vector, typename InverseOperator>
class MGCoarseInverseOperator : public MGCoarseGridBase<Vector>
{
public:
  MGCoarseInverseOperator(std::shared_ptr<InverseOperator const> inverse_coarse_grid_operator)
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

  std::shared_ptr<InverseOperator const> inverse_operator;
};


template <int dim, class T, class Number>
class MeshWorkerWrapper : public MeshWorker::LocalIntegrator<dim,dim,Number>{
    
    public:
        
        MeshWorkerWrapper(const T& t) : t(t){ }
    
    inline DEAL_II_ALWAYS_INLINE 
    void cell(MeshWorker::DoFInfo<dim,dim,Number> &dinfo,
          typename MeshWorker::IntegrationInfo<dim> &info) const {
        t.cell(dinfo, info);
    }

    inline DEAL_II_ALWAYS_INLINE 
    void boundary(MeshWorker::DoFInfo<dim,dim,Number> &dinfo,
                  typename MeshWorker::IntegrationInfo<dim> &info) const{
        t.boundary(dinfo, info);
    }

    inline DEAL_II_ALWAYS_INLINE 
    void face(MeshWorker::DoFInfo<dim,dim,Number> &dinfo1,
              MeshWorker::DoFInfo<dim,dim,Number> &dinfo2,
              typename MeshWorker::IntegrationInfo<dim> &info1,
              typename MeshWorker::IntegrationInfo<dim> &info2) const{ 
        t.face(dinfo1, dinfo2, info1, info2);
    }
    
    const T& t;
};

template<int DIM, typename Operator>
class MGCoarseML : public MGCoarseGridBase<parallel::distributed::Vector<typename Operator::value_type> >
{
public:
    
  typedef TrilinosWrappers::SparseMatrix MatrixType;

  /**
   * Constructor
   */
  MGCoarseML(Operator const &matrix , bool setup = true) :  coarse_matrix (matrix){
      if(setup)
        this->reinit();
  }

  /**
   * Deconstructor
   */
  virtual ~MGCoarseML() { }
  
  /**
   * Setup system matrix and AMG
   */
  void reinit(){
      
    // extract relevant data structures
    const DoFHandler<DIM>& dof_handler = coarse_matrix.get_data().get_dof_handler();
    
    // create temporal classes
    FE_DGQ<DIM> fe(dof_handler.get_fe().degree);
    MappingQGeneric<DIM> mapping(dof_handler.get_fe().degree);
      
    // initialize the system matrix ...
    // ... create a sparsity pattern
    TrilinosWrappers::SparsityPattern dsp(dof_handler.locally_owned_mg_dofs(0), MPI_COMM_WORLD);
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    dsp.compress();
    system_matrix.reinit(dsp);
    
    // ... assemble system matrix with the help of MeshWorker and Assembler
    MeshWorker::IntegrationInfoBox<DIM> info_box;
    const unsigned int n_gauss_points = dof_handler.get_fe().degree+1;
    info_box.initialize_gauss_quadrature(
            n_gauss_points, n_gauss_points, n_gauss_points);
    info_box.initialize_update_flags();
    UpdateFlags update_flags = 
            update_quadrature_points | update_values | update_gradients;
    info_box.add_update_flags(update_flags, true, true, true, true);
    info_box.initialize(fe, mapping);
    MeshWorker::DoFInfo<DIM,DIM,double> dof_info(dof_handler);

    MeshWorker::Assembler::MatrixSimple<MatrixType> assembler;
    assembler.initialize(system_matrix);

    MeshWorker::LoopControl lc;
    lc.faces_to_ghost = MeshWorker::LoopControl::both;

    MeshWorker::integration_loop<DIM, DIM> (
        dof_handler.begin_mg(0), dof_handler.end_mg(0),
        dof_info, info_box, MeshWorkerWrapper<DIM, Operator, double>(coarse_matrix), assembler,lc);
    
    // intialize Trilinos' AMG
    pamg.initialize(system_matrix);
    
  }

  /**
   *  Solve Ax = b with Trilinos's AMG (default settings)
   * 
   * @param level not used
   * @param dst solution vector x
   * @param src right hand side b
   */
  virtual void operator() (const unsigned int                                                  /*level*/,
                           parallel::distributed::Vector<typename Operator::value_type>       & dst,
                           const parallel::distributed::Vector<typename Operator::value_type> & src) const{
      
      // create temporal vectors of type TrilinosScalar
      parallel::distributed::Vector<TrilinosWrappers::SparseMatrix::value_type> dst_; 
      dst_.reinit(dst, true);
      parallel::distributed::Vector<TrilinosWrappers::SparseMatrix::value_type> src_; 
      src_.reinit(src, true);
      
      // convert Operator::value_type to TrilinosScalar
      src_.copy_locally_owned_data_from(src);
      
      // use Trilinos to perform AMG
      pamg.vmult(dst_,src_);
      
      // convert TrilinosScalar to Operator::value_type
      dst.copy_locally_owned_data_from(dst_);
      
  }

  /**
   * Perform sparse matrix vector multiplication (SpMV): u = Av
   * 
   * @param dst input vector v
   * @param src output vector u
   */
  virtual void vmult(parallel::distributed::Vector<typename Operator::value_type> & dst,
               const parallel::distributed::Vector<typename Operator::value_type> & src) const{
      
      // create temporal vectors of type TrilinosScalar
      parallel::distributed::Vector<TrilinosWrappers::SparseMatrix::value_type> dst_; 
      dst_.reinit(dst, true);
      parallel::distributed::Vector<TrilinosWrappers::SparseMatrix::value_type> src_; 
      src_.reinit(src, true);
      
      // convert Operator::value_type to TrilinosScalar
      src_.copy_locally_owned_data_from(src);
      
      // use Trilinos for SpMV
      system_matrix.vmult(dst_,src_);
      
      // convert TrilinosScalar to Operator::value_type
      dst.copy_locally_owned_data_from(dst_);
      
  }

private:
    // reference to operator
    const Operator &coarse_matrix;
    // distributed sparse system matrix
    MatrixType system_matrix;
    // AMG preconditioner
    TrilinosWrappers::PreconditionAMG pamg;
};

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MGCOARSEGRIDSOLVERS_H_ */
