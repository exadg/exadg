#ifndef MG_COARSE_ML
#define MG_COARSE_ML

#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/multigrid/mg_base.h>


#include "../preconditioner/preconditioner_base.h"
#include "../transfer/dg_to_cg_transfer.h"

#include "../../functionalities/print_functions.h"

#ifndef DEAL_II_WITH_TRILINOS
namespace dealii
{
namespace TrilinosWrappers
{
namespace PreconditionAMG
{
// copy of interface from deal.II (lac/trilinos_precondition.h)
struct AdditionalData
{
  AdditionalData(
    const bool /*elliptic*/                                   = true,
    const bool /*higher_order_elements*/                      = false,
    const unsigned int /*n_cycles*/                           = 1,
    const bool /*w_cyle*/                                     = false,
    const double /*aggregation_threshold*/                    = 1e-4,
    const std::vector<std::vector<bool>> & /*constant_modes*/ = std::vector<std::vector<bool>>(0),
    const unsigned int /*smoother_sweeps*/                    = 2,
    const unsigned int /*smoother_overlap*/                   = 0,
    const bool /*output_details*/                             = false,
    const char * /*smoother_type*/                            = "Chebyshev",
    const char * /*coarse_type*/                              = "Amesos-KLU")
  {
  }
  bool                           elliptic;
  bool                           higher_order_elements;
  unsigned int                   n_cycles;
  bool                           w_cycle;
  double                         aggregation_threshold;
  std::vector<std::vector<bool>> constant_modes;
  unsigned int                   smoother_sweeps;
  unsigned int                   smoother_overlap;
  bool                           output_details;
  const char *                   smoother_type;
  const char *                   coarse_type;
};

} // namespace PreconditionAMG
} // namespace TrilinosWrappers
} // namespace dealii
#endif

struct MGCoarseMLData
{
  MGCoarseMLData()
    : use_conjugate_gradient_solver(true),
      max_iter(10000),
      solver_tolerance_abs(1e-20),
      solver_tolerance_rel(1e-2),
      pcg_failure_criterion(100.0),
      transfer_to_continuous_galerkin(true)
  {
    amg_data.smoother_sweeps = 1;
    amg_data.n_cycles        = 1;
    amg_data.smoother_type   = "ILU";
  };
  
  void print(ConditionalOStream &pcout)
  {
    print_parameter(pcout,"  Accelerate with conjugate gradient solver (PCG)",use_conjugate_gradient_solver);
    if(use_conjugate_gradient_solver){
      print_parameter(pcout,"    PCG max. iterations",max_iter);
      print_parameter(pcout,"    PCG abs. tolerance",solver_tolerance_abs);
      print_parameter(pcout,"    PCG rel. tolerance",solver_tolerance_rel);
      print_parameter(pcout,"    PCG failure criterion",pcg_failure_criterion);
    }
    print_parameter(pcout,"  Perform transfer to continuous Galerkin",transfer_to_continuous_galerkin);
  }
  
  bool   use_conjugate_gradient_solver;
  int    max_iter;
  double solver_tolerance_abs;
  double solver_tolerance_rel;
  double pcg_failure_criterion;
  bool   transfer_to_continuous_galerkin;

  TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
};

#ifdef DEAL_II_WITH_TRILINOS

template<typename Operator, typename Number = typename Operator::value_type>
class MGCoarseML : public MGCoarseGridBase<parallel::distributed::Vector<typename Operator::value_type>>,
                   public PreconditionerBase<Number>
{
public:
  typedef typename Operator::value_type  MultigridNumber;
  typedef TrilinosWrappers::SparseMatrix MatrixType;

  static const int DIM = Operator::DIM;

  /**
   * Constructor
   */
  MGCoarseML(Operator const & matrix,
             Operator const & matrix_q,
             bool             setup = false,
             int              level = -1,
             MGCoarseMLData   data  = MGCoarseMLData());

  /**
   * Deconstructor
   */
  virtual ~MGCoarseML();

  /**
   * Setup system matrix and AMG
   */
  void
  reinit(int level = 0, MGCoarseMLData data = MGCoarseMLData());

  virtual void
  update(MatrixOperatorBase const * /*matrix_operator*/);

  /**
   *  Solve Ax = b with Trilinos's AMG (default settings)
   *
   * @param level not used
   * @param dst solution vector x
   * @param src right hand side b
   */
  virtual void
  operator()(const unsigned int /*level*/,
             parallel::distributed::Vector<MultigridNumber> &       dst,
             const parallel::distributed::Vector<MultigridNumber> & src) const;

  void
  vmult(parallel::distributed::Vector<Number> & dst, const parallel::distributed::Vector<Number> & src) const;

private:
  // reference to matrix-free operators
  const Operator & operator_dg;
  const Operator & operator_cg;
  
  std::shared_ptr<CGToDGTransfer<Operator::DIM, MultigridNumber>> transfer;
  // distributed sparse system matrix
  MatrixType system_matrix;
  // AMG preconditioner
  TrilinosWrappers::PreconditionAMG pamg;

  MGCoarseMLData additional_data;
};

#else

template<typename Operator, typename Number = typename Operator::value_type>
class MGCoarseML : public MGCoarseGridBase<parallel::distributed::Vector<typename Operator::value_type>>,
                   public PreconditionerBase<Number>
{
public:
  typedef typename Operator::value_type MultigridNumber;
  static const int                      DIM = Operator::DIM;

  /**
   * Constructor
   */
  MGCoarseML(Operator const & matrix,
             Operator const & matrix_q,
             bool             setup = false,
             int              level = -1,
             MGCoarseMLData   data  = MGCoarseMLData());

  /**
   * Deconstructor
   */
  virtual ~MGCoarseML();

  /**
   * Setup system matrix and AMG
   */
  void
  reinit(int level = 0, MGCoarseMLData data = MGCoarseMLData());

  virtual void
  update(MatrixOperatorBase const * /*matrix_operator*/);

  /**
   *  Solve Ax = b with Trilinos's AMG (default settings)
   *
   * @param level not used
   * @param dst solution vector x
   * @param src right hand side b
   */
  virtual void
  operator()(const unsigned int /*level*/,
             parallel::distributed::Vector<MultigridNumber> &       dst,
             const parallel::distributed::Vector<MultigridNumber> & src) const;

  void
  vmult(parallel::distributed::Vector<Number> & dst, const parallel::distributed::Vector<Number> & src) const;
};

#endif

#endif