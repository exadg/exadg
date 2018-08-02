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

#ifndef DEAL_II_WITH_TRILINOS
namespace dealii{
namespace TrilinosWrappers{
namespace PreconditionAMG{

// copy of interface from deal.II (lac/trilinos_precondition.h)
struct AdditionalData{
  AdditionalData(const bool         /*elliptic*/              = true,
                 const bool         /*higher_order_elements*/ = false,
                 const unsigned int /*n_cycles*/              = 1,
                 const bool         /*w_cyle*/                = false,
                 const double       /*aggregation_threshold*/ = 1e-4,
                 const std::vector<std::vector<bool>> &/*constant_modes*/ =
                   std::vector<std::vector<bool>>(0),
                 const unsigned int /*smoother_sweeps*/  = 2,
                 const unsigned int /*smoother_overlap*/ = 0,
                 const bool         /*output_details*/   = false,
                 const char *       /*smoother_type*/    = "Chebyshev",
                 const char *       /*coarse_type*/      = "Amesos-KLU"){
  }
  bool elliptic;
  bool higher_order_elements;
  unsigned int n_cycles;
  bool w_cycle;
  double aggregation_threshold;
  std::vector<std::vector<bool>> constant_modes;
  unsigned int smoother_sweeps;
  unsigned int smoother_overlap;
  bool output_details;
  const char *smoother_type;
  const char *coarse_type;
};
       
}
}
}
#endif

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

#ifdef DEAL_II_WITH_TRILINOS

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

#else

template <typename Operator, typename Number = typename Operator::value_type>
class MGCoarseML
    : public MGCoarseGridBase<
          parallel::distributed::Vector<typename Operator::value_type>>,
      public PreconditionerBase<Number> {
public:
  typedef typename Operator::value_type MultigridNumber;
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

};

#endif

#endif