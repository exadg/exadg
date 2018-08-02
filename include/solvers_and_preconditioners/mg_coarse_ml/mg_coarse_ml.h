#ifndef MG_COARSE_ML
#define MG_COARSE_ML

#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/multigrid/mg_base.h>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include "../preconditioner/preconditioner_base.h"

#include "mg_coarse_ml_cg.h"
#include "mg_coarse_ml_dg.h"

struct MGCoarseMLData
{
    MGCoarseMLData() : 
        use_pcg(true), 
        pcg_max_iterations(10000),
        pcg_abs_residuum(1e-20),
        pcg_rel_residuum(1e-2),
        pcg_failure_criterion(100.0),
        use_cg(true)
    {
       amg_data.smoother_sweeps = 1;
       amg_data.n_cycles        = 1;
       amg_data.smoother_type   = "ILU";
    };
    bool   use_pcg;
    int    pcg_max_iterations;
    double pcg_abs_residuum;
    double pcg_rel_residuum;
    double pcg_failure_criterion;
    bool   use_cg;
    
    TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
};


template <typename Operator, typename Number = typename Operator::value_type>
class MGCoarseML
    : public MGCoarseGridBase<
          parallel::distributed::Vector<typename Operator::value_type>>,
      public PreconditionerBase<Number> {
public:
  typedef typename Operator::value_type MultigridNumber;
  typedef TrilinosWrappers::SparseMatrix MatrixType;

  static const int DIM = Operator::DIM;

  /**
   * Constructor
   */
  MGCoarseML(Operator const &matrix, 
             Operator const &matrix_q, 
             bool setup = false,
             int level = -1,
             MGCoarseMLData data = MGCoarseMLData());

  /**
   * Deconstructor
   */
  virtual ~MGCoarseML();

  /**
   * Setup system matrix and AMG
   */
  void reinit(int level = 0, MGCoarseMLData data = MGCoarseMLData());

  virtual void update(MatrixOperatorBase const * /*matrix_operator*/);

  /**
   *  Solve Ax = b with Trilinos's AMG (default settings)
   *
   * @param level not used
   * @param dst solution vector x
   * @param src right hand side b
   */
  virtual void
  operator()(const unsigned int /*level*/,
             parallel::distributed::Vector<MultigridNumber> &dst,
             const parallel::distributed::Vector<MultigridNumber> &src) const;

  void vmult(parallel::distributed::Vector<Number> &dst,
             const parallel::distributed::Vector<Number> &src) const;

private:
  // reference to operator
  const Operator &coarse_matrix;
  const Operator &coarse_matrix_q;
  // distributed sparse system matrix
  MatrixType system_matrix;
  // AMG preconditioner
  TrilinosWrappers::PreconditionAMG pamg;

  std::shared_ptr<MGCoarseMLWrapper<DIM, MultigridNumber>> wrapper;
  
  MGCoarseMLData additional_data;
};

#endif