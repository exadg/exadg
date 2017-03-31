/*
 * MultigridPreconditionerWrapperBase.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_ADAPTER_BASE_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_ADAPTER_BASE_H_

#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>

#include "smoother_base.h"
#include "solvers_and_preconditioners/chebyshev_smoother.h"
#include "solvers_and_preconditioners/gmres_smoother.h"
#include "solvers_and_preconditioners/cg_smoother.h"
#include "solvers_and_preconditioners/jacobi_smoother.h"
#include "solvers_and_preconditioners/mg_coarse_grid_solvers.h"
#include "solvers_and_preconditioners/multigrid_input_parameters.h"
#include "solvers_and_preconditioners/multigrid_preconditioner.h"

namespace
{
  // manually compute eigenvalues for the coarsest level for proper setup of the Chebyshev iteration
  template <typename Operator>
  std::pair<double,double>
  compute_eigenvalues(const Operator &op,
                      const parallel::distributed::Vector<typename Operator::value_type> &inverse_diagonal,
                      const unsigned int eig_n_iter = 10000)
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

    SolverControl control(eig_n_iter, right.l2_norm()*1e-5);
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
                            const parallel::distributed::Vector<typename Operator::value_type> &inverse_diagonal,
                            const unsigned int eig_n_iter = 10000)
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

//    SolverControl control(10000, right.l2_norm()*1e-5);
    ReductionControl control (eig_n_iter, right.l2_norm()*1.0e-5, 1.0e-5);

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

  void vmult (parallel::distributed::Vector<value_type>        &dst,
              const parallel::distributed::Vector<value_type>  &src) const
  {
    multigrid_preconditioner->vmult(dst,src);
  }

  virtual void apply_smoother_on_fine_level(parallel::distributed::Vector<typename Operator::value_type>        &dst,
                                            const parallel::distributed::Vector<typename Operator::value_type>  &src) const
  {
    this->mg_smoother[this->mg_smoother.max_level()]->vmult(dst,src);
  }

protected:
  /*
   *  This function initializes the multigrid smoothers on all levels with level >= 1
   */
  virtual void initialize_smoothers()
  {
    // resize
    this->mg_smoother.resize(0, this->n_global_levels-1);

    // initialize mg_matrices for all levels >= 1
    for (unsigned int level = 1; level<this->n_global_levels; ++level)
    {
      this->initialize_smoother(level);
    }
  }

  virtual void initialize_smoother(unsigned int level)
  {
    AssertThrow(level > 0, ExcMessage("Multigrid level is invalid when initializing multigrid smoother!"));

    switch (mg_data.smoother)
    {
      case MultigridSmoother::Chebyshev:
      {
        mg_smoother[level].reset(new ChebyshevSmoother<Operator,VECTOR_TYPE>());
        initialize_chebyshev_smoother(level);
        break;
      }
      case MultigridSmoother::ChebyshevNonsymmetricOperator:
      {
        mg_smoother[level].reset(new ChebyshevSmoother<Operator,VECTOR_TYPE>());
        initialize_chebyshev_smoother_nonsymmetric_operator(level);
        break;
      }
      case MultigridSmoother::GMRES:
      {
        typedef GMRESSmoother<dim,Operator,VECTOR_TYPE> GMRES_SMOOTHER;
        mg_smoother[level].reset(new GMRES_SMOOTHER());

        typename GMRES_SMOOTHER::AdditionalData smoother_data;
        smoother_data.preconditioner = mg_data.gmres_smoother_data.preconditioner;
        smoother_data.number_of_iterations = mg_data.gmres_smoother_data.number_of_iterations;

        std_cxx11::shared_ptr<GMRES_SMOOTHER> smoother = std::dynamic_pointer_cast<GMRES_SMOOTHER>(mg_smoother[level]);
        smoother->initialize(mg_matrices[level],smoother_data);
        break;
      }
      case MultigridSmoother::CG:
      {
        typedef CGSmoother<dim,Operator,VECTOR_TYPE> CG_SMOOTHER;
        mg_smoother[level].reset(new CG_SMOOTHER());

        typename CG_SMOOTHER::AdditionalData smoother_data;
        smoother_data.preconditioner = mg_data.cg_smoother_data.preconditioner;
        smoother_data.number_of_iterations = mg_data.cg_smoother_data.number_of_iterations;

        std_cxx11::shared_ptr<CG_SMOOTHER> smoother = std::dynamic_pointer_cast<CG_SMOOTHER>(mg_smoother[level]);
        smoother->initialize(mg_matrices[level],smoother_data);
        break;
      }
      case MultigridSmoother::Jacobi:
      {
        typedef JacobiSmoother<dim,Operator,VECTOR_TYPE> JACOBI_SMOOTHER;
        mg_smoother[level].reset(new JACOBI_SMOOTHER());

        typename JACOBI_SMOOTHER::AdditionalData smoother_data;
        smoother_data.preconditioner = mg_data.jacobi_smoother_data.preconditioner;
        smoother_data.number_of_smoothing_steps = mg_data.jacobi_smoother_data.number_of_smoothing_steps;
        smoother_data.damping_factor = mg_data.jacobi_smoother_data.damping_factor;

        std_cxx11::shared_ptr<JACOBI_SMOOTHER> smoother = std::dynamic_pointer_cast<JACOBI_SMOOTHER>(mg_smoother[level]);
        smoother->initialize(mg_matrices[level],smoother_data);
        break;
      }
      default:
      {
        AssertThrow(false, ExcMessage("Specified MultigridSmoother not implemented!"));
      }
    }
  }

  virtual void update_smoother(unsigned int level)
  {
    AssertThrow(level > 0, ExcMessage("Multigrid level is invalid when initializing multigrid smoother!"));

    switch (mg_data.smoother)
    {
      case MultigridSmoother::Chebyshev:
      {
        initialize_chebyshev_smoother(level);
        break;
      }
      case MultigridSmoother::ChebyshevNonsymmetricOperator:
      {
        initialize_chebyshev_smoother_nonsymmetric_operator(level);
        break;
      }
      case MultigridSmoother::GMRES:
      {
        typedef GMRESSmoother<dim,Operator,VECTOR_TYPE> GMRES_SMOOTHER;
        std_cxx11::shared_ptr<GMRES_SMOOTHER> smoother = std::dynamic_pointer_cast<GMRES_SMOOTHER>(mg_smoother[level]);
        smoother->update();
        break;
      }
      case MultigridSmoother::CG:
      {
        typedef CGSmoother<dim,Operator,VECTOR_TYPE> CG_SMOOTHER;
        std_cxx11::shared_ptr<CG_SMOOTHER> smoother = std::dynamic_pointer_cast<CG_SMOOTHER>(mg_smoother[level]);
        smoother->update();
        break;
      }
      case MultigridSmoother::Jacobi:
      {
        typedef JacobiSmoother<dim,Operator,VECTOR_TYPE> JACOBI_SMOOTHER;
        std_cxx11::shared_ptr<JACOBI_SMOOTHER> smoother = std::dynamic_pointer_cast<JACOBI_SMOOTHER>(mg_smoother[level]);
        smoother->update();
        break;
      }
      default:
      {
        AssertThrow(false, ExcMessage("Specified MultigridSmoother not implemented!"));
      }
    }
  }

  virtual void initialize_coarse_solver()
  {
    switch (mg_data.coarse_solver)
    {
      case MultigridCoarseGridSolver::Chebyshev:
      {
        mg_smoother[0].reset(new ChebyshevSmoother<Operator,VECTOR_TYPE>());
        initialize_chebyshev_smoother_coarse_grid();

        mg_coarse.reset(new MGCoarseInverseOperator<parallel::distributed::Vector<typename Operator::value_type>, SMOOTHER>(mg_smoother[0]));
        break;
      }
      case MultigridCoarseGridSolver::ChebyshevNonsymmetricOperator:
      {
        mg_smoother[0].reset(new ChebyshevSmoother<Operator,VECTOR_TYPE>());
        initialize_chebyshev_smoother_nonsymmetric_operator_coarse_grid();

        mg_coarse.reset(new MGCoarseInverseOperator<parallel::distributed::Vector<typename Operator::value_type>, SMOOTHER>(mg_smoother[0]));
        break;
      }
      case MultigridCoarseGridSolver::PCG_NoPreconditioner:
      {
        typename MGCoarsePCG<Operator>::AdditionalData additional_data;
        additional_data.preconditioner = PreconditionerCoarseGridSolver::None;

        mg_coarse.reset(new MGCoarsePCG<Operator>(mg_matrices[0],additional_data));
        break;
      }
      case MultigridCoarseGridSolver::PCG_PointJacobi:
      {
        typename MGCoarsePCG<Operator>::AdditionalData additional_data;
        additional_data.preconditioner = PreconditionerCoarseGridSolver::PointJacobi;

        mg_coarse.reset(new MGCoarsePCG<Operator>(mg_matrices[0],additional_data));
        break;
      }
      case MultigridCoarseGridSolver::PCG_BlockJacobi:
      {
        typename MGCoarsePCG<Operator>::AdditionalData additional_data;
        additional_data.preconditioner = PreconditionerCoarseGridSolver::BlockJacobi;

        mg_coarse.reset(new MGCoarsePCG<Operator>(mg_matrices[0],additional_data));
        break;
      }
      case MultigridCoarseGridSolver::GMRES_NoPreconditioner:
      {
        typename MGCoarseGMRES<Operator>::AdditionalData additional_data;
        additional_data.preconditioner = PreconditionerCoarseGridSolver::None;

        mg_coarse.reset(new MGCoarseGMRES<Operator>(mg_matrices[0],additional_data));
        break;
      }
      case MultigridCoarseGridSolver::GMRES_PointJacobi:
      {
        typename MGCoarseGMRES<Operator>::AdditionalData additional_data;
        additional_data.preconditioner = PreconditionerCoarseGridSolver::PointJacobi;

        mg_coarse.reset(new MGCoarseGMRES<Operator>(mg_matrices[0],additional_data));
        break;
      }
      case MultigridCoarseGridSolver::GMRES_BlockJacobi:
      {
        typename MGCoarseGMRES<Operator>::AdditionalData additional_data;
        additional_data.preconditioner = PreconditionerCoarseGridSolver::BlockJacobi;

        mg_coarse.reset(new MGCoarseGMRES<Operator>(mg_matrices[0],additional_data));
        break;
      }
      default:
      {
        AssertThrow(false, ExcMessage("Unknown coarse-grid solver specified."));
      }
    }
  }

  /*
   *  This function updates the (preconditioner of the) coarse grid solver.
   *  The prerequisite to call this function is that mg_matrices[0] has
   *  been updated.
   */
  virtual void update_coarse_solver()
  {
    switch (mg_data.coarse_solver)
    {
      case MultigridCoarseGridSolver::Chebyshev:
      {
        initialize_chebyshev_smoother_coarse_grid();
        break;
      }
      case MultigridCoarseGridSolver::ChebyshevNonsymmetricOperator:
      {
        initialize_chebyshev_smoother_nonsymmetric_operator_coarse_grid();
        break;
      }
      case MultigridCoarseGridSolver::PCG_NoPreconditioner:
      {
        // do nothing
        break;
      }
      case MultigridCoarseGridSolver::PCG_PointJacobi:
      {
        std_cxx11::shared_ptr<MGCoarsePCG<Operator> >
          coarse_solver = std::dynamic_pointer_cast<MGCoarsePCG<Operator> >(mg_coarse);
        coarse_solver->update_preconditioner(this->mg_matrices[0]);

        break;
      }
      case MultigridCoarseGridSolver::PCG_BlockJacobi:
      {
        std_cxx11::shared_ptr<MGCoarsePCG<Operator> >
          coarse_solver = std::dynamic_pointer_cast<MGCoarsePCG<Operator> >(mg_coarse);
        coarse_solver->update_preconditioner(this->mg_matrices[0]);

        break;
      }
      case MultigridCoarseGridSolver::GMRES_NoPreconditioner:
      {
        // do nothing
        break;
      }
      case MultigridCoarseGridSolver::GMRES_PointJacobi:
      {
        std_cxx11::shared_ptr<MGCoarseGMRES<Operator> >
          coarse_solver = std::dynamic_pointer_cast<MGCoarseGMRES<Operator> >(mg_coarse);
        coarse_solver->update_preconditioner(this->mg_matrices[0]);
        break;
      }
      case MultigridCoarseGridSolver::GMRES_BlockJacobi:
      {
        std_cxx11::shared_ptr<MGCoarseGMRES<Operator> >
          coarse_solver = std::dynamic_pointer_cast<MGCoarseGMRES<Operator> >(mg_coarse);
        coarse_solver->update_preconditioner(this->mg_matrices[0]);
        break;
      }
      default:
      {
        AssertThrow(false, ExcMessage("Unknown coarse-grid solver given"));
      }
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

  virtual void initialize_multigrid_preconditioner(DoFHandler<dim> const &dof_handler)
  {
    this->multigrid_preconditioner.reset(new MultigridPreconditioner<dim, VECTOR_TYPE, Operator, MG_TRANSFER, SMOOTHER>
        (dof_handler, this->mg_matrices, *(this->mg_coarse), this->mg_transfer, this->mg_smoother));
  }

  MultigridData mg_data;
  MGConstrainedDoFs mg_constrained_dofs;

  MGLevelObject<Operator> mg_matrices;
  typedef MGTransferMF<dim,Operator> MG_TRANSFER;
  MG_TRANSFER mg_transfer;

  typedef parallel::distributed::Vector<typename Operator::value_type> VECTOR_TYPE;
  typedef SmootherBase<VECTOR_TYPE> SMOOTHER;
  MGLevelObject<std_cxx11::shared_ptr<SMOOTHER> > mg_smoother;

  std_cxx11::shared_ptr<MGCoarseGridBase<VECTOR_TYPE> > mg_coarse;

  std_cxx11::shared_ptr<MultigridPreconditioner<dim, VECTOR_TYPE, Operator, MG_TRANSFER, SMOOTHER> > multigrid_preconditioner;

  unsigned int n_global_levels;

private:
  void initialize_chebyshev_smoother(unsigned int level)
  {
    typedef ChebyshevSmoother<Operator,VECTOR_TYPE> CHEBYSHEV_SMOOTHER;
    typename CHEBYSHEV_SMOOTHER::AdditionalData smoother_data;

    mg_matrices[level].initialize_dof_vector(smoother_data.matrix_diagonal_inverse);
    mg_matrices[level].calculate_inverse_diagonal(smoother_data.matrix_diagonal_inverse);

    /*
    std::pair<double,double> eigenvalues = compute_eigenvalues(mg_matrices[level], smoother_data.matrix_diagonal_inverse);
    std::cout<<"Max EW = "<< eigenvalues.second <<" : Min EW = "<<eigenvalues.first<<std::endl;
    */

    smoother_data.smoothing_range = mg_data.chebyshev_smoother_data.smoother_smoothing_range;
    smoother_data.degree = mg_data.chebyshev_smoother_data.smoother_poly_degree;
    smoother_data.eig_cg_n_iterations = mg_data.chebyshev_smoother_data.eig_cg_n_iterations;

    std_cxx11::shared_ptr<CHEBYSHEV_SMOOTHER> smoother = std::dynamic_pointer_cast<CHEBYSHEV_SMOOTHER>(mg_smoother[level]);
    smoother->initialize(mg_matrices[level], smoother_data);
  }

  void initialize_chebyshev_smoother_coarse_grid()
  {
    // use Chebyshev smoother of high degree to solve the coarse grid problem approximately
    typedef ChebyshevSmoother<Operator,VECTOR_TYPE> CHEBYSHEV_SMOOTHER;
    typename CHEBYSHEV_SMOOTHER::AdditionalData smoother_data;

    mg_matrices[0].initialize_dof_vector(smoother_data.matrix_diagonal_inverse);
    mg_matrices[0].calculate_inverse_diagonal(smoother_data.matrix_diagonal_inverse);

    std::pair<double,double> eigenvalues = compute_eigenvalues(mg_matrices[0], smoother_data.matrix_diagonal_inverse);
    smoother_data.max_eigenvalue = 1.1 * eigenvalues.second;
    smoother_data.smoothing_range = eigenvalues.second/eigenvalues.first*1.1;
    double sigma = (1.-std::sqrt(1./smoother_data.smoothing_range))/(1.+std::sqrt(1./smoother_data.smoothing_range));
    const double eps = 1e-3;
    smoother_data.degree = std::log(1./eps+std::sqrt(1./eps/eps-1))/std::log(1./sigma);
    smoother_data.eig_cg_n_iterations = 0;

    std_cxx11::shared_ptr<CHEBYSHEV_SMOOTHER> smoother = std::dynamic_pointer_cast<CHEBYSHEV_SMOOTHER>(mg_smoother[0]);
    smoother->initialize(mg_matrices[0], smoother_data);
  }

  void initialize_chebyshev_smoother_nonsymmetric_operator(unsigned int level)
  {
    typedef ChebyshevSmoother<Operator,VECTOR_TYPE> CHEBYSHEV_SMOOTHER;
    typename CHEBYSHEV_SMOOTHER::AdditionalData smoother_data;

    this->mg_matrices[level].initialize_dof_vector(smoother_data.matrix_diagonal_inverse);
    this->mg_matrices[level].calculate_inverse_diagonal(smoother_data.matrix_diagonal_inverse);

    /*
    std::pair<double,double> eigenvalues = compute_eigenvalues_gmres(mg_matrices[level], smoother_data.matrix_diagonal_inverse);
    std::cout<<"Max EW = "<< eigenvalues.second <<" : Min EW = "<<eigenvalues.first<<std::endl;
    */

    // use gmres to calculate eigenvalues for nonsymmetric problem
    const unsigned int eig_n_iter = 20;
    std::pair<std::complex<double>,std::complex<double> > eigenvalues =
        compute_eigenvalues_gmres(mg_matrices[level], smoother_data.matrix_diagonal_inverse,eig_n_iter);
    const double factor = 1.1;
    smoother_data.max_eigenvalue = factor * std::abs(eigenvalues.second);
    smoother_data.smoothing_range = mg_data.chebyshev_smoother_data.smoother_smoothing_range;
    smoother_data.degree = mg_data.chebyshev_smoother_data.smoother_poly_degree;
    smoother_data.eig_cg_n_iterations = 0;

    std_cxx11::shared_ptr<CHEBYSHEV_SMOOTHER> smoother = std::dynamic_pointer_cast<CHEBYSHEV_SMOOTHER>(mg_smoother[level]);
    smoother->initialize(mg_matrices[level], smoother_data);
  }

  void initialize_chebyshev_smoother_nonsymmetric_operator_coarse_grid()
  {
    // use Chebyshev smoother of high degree to solve the coarse grid problem approximately
    typedef ChebyshevSmoother<Operator,VECTOR_TYPE> CHEBYSHEV_SMOOTHER;
    typename CHEBYSHEV_SMOOTHER::AdditionalData smoother_data;

    mg_matrices[0].initialize_dof_vector(smoother_data.matrix_diagonal_inverse);
    mg_matrices[0].calculate_inverse_diagonal(smoother_data.matrix_diagonal_inverse);

    const double factor = 1.1;
    std::pair<std::complex<double>,std::complex<double> > eigenvalues = compute_eigenvalues_gmres(mg_matrices[0], smoother_data.matrix_diagonal_inverse);
    smoother_data.max_eigenvalue = factor * std::abs(eigenvalues.second);
    smoother_data.smoothing_range = factor * std::abs(eigenvalues.second)/std::abs(eigenvalues.first);
    double sigma = (1.-std::sqrt(1./smoother_data.smoothing_range))/(1.+std::sqrt(1./smoother_data.smoothing_range));
    const double eps = 1e-3;
    smoother_data.degree = std::log(1./eps+std::sqrt(1./eps/eps-1))/std::log(1./sigma);
    smoother_data.eig_cg_n_iterations = 0;

    std_cxx11::shared_ptr<CHEBYSHEV_SMOOTHER> smoother = std::dynamic_pointer_cast<CHEBYSHEV_SMOOTHER>(mg_smoother[0]);
    smoother->initialize(this->mg_matrices[0], smoother_data);
  }
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_ADAPTER_BASE_H_ */
