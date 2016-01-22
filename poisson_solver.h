
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

using namespace dealii;


template <int dim>
struct PoissonSolverData
{
  PoissonSolverData ()
    :
    pressure_dof_index(0),
    pressure_quad_index(0),
    penalty_factor(1.),
    solver_tolerance(1e-5),
    smoother_poly_degree(5),
    smoother_smoothing_range(20),
    coarse_solver(coarse_chebyshev_smoother)
  {}

  // If an external MatrixFree object is given which can contain other
  // components than the variable for which the Poisson equation should be
  // solved, this selects the correct DoFHandler component
  unsigned int pressure_dof_index;

  // If an external MatrixFree object is given which can contain other
  // quadrature formulas than the quadrature formula which should be used by
  // the Poisson solver, this selects the correct quadrature index
  unsigned int pressure_quad_index;

  // The penalty parameter for the symmetric interior penalty method is
  // computed as penalty_factor * (fe_degree+1)^2 /
  // characteristic_element_length. This variable gives the scaling factor
  double penalty_factor;

  // Specifies the boundary ids with Dirichlet boundary conditions
  std::set<types::boundary_id> dirichlet_boundaries;

  // Specifies the boundary ids with Neumann boundary conditions
  std::set<types::boundary_id> neumann_boundaries;

  // If periodic boundaries are present, this variable collects faces on the
  // two sides of the domain
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs;

  // Sets the tolerance for the linear solver
  double solver_tolerance;

  // Sets the polynomial degree of the Chebyshev smoother (Chebyshev
  // accelerated Jacobi smoother)
  double smoother_poly_degree;

  // Sets the smoothing range of the Chebyshev smoother
  double smoother_smoothing_range;

  // Sets the coarse grid solver
  enum CoarseSolveSelector { coarse_chebyshev_smoother , coarse_iterative_noprec,
                             coarse_iterative_jacobi } coarse_solver;
};



// Generic implementation of Laplace operator for both continuous elements
// (FE_Q) and discontinuous elements (FE_DGQ).
//
// TODO: Continuous elements are work in progress...
template <int dim, typename Number=double>
class LaplaceOperator : public Subscriptor
{
public:
  typedef Number value_type;

  LaplaceOperator ();

  void clear();

  // Initialization with given MatrixFree object. In case of continuous FE_Q
  // elements, it is expected that hanging node constraints are present in
  // mf_data.
  void reinit(const MatrixFree<dim,Number>       &mf_data,
              const Mapping<dim>                 &mapping,
              const PoissonSolverData<dim>       &solver_data);

  void reinit (const DoFHandler<dim> &dof_handler,
               const Mapping<dim> &mapping,
               const PoissonSolverData<dim> &solver_data,
               const unsigned int level = numbers::invalid_unsigned_int);

  // Ensures that the boundary conditions make sense and computes the array
  // penalty parameter
  void check_boundary_conditions(const Mapping<dim> &mapping);

  void vmult(parallel::distributed::Vector<Number> &dst,
             const parallel::distributed::Vector<Number> &src) const;

  void Tvmult(parallel::distributed::Vector<Number> &dst,
              const parallel::distributed::Vector<Number> &src) const;

  void Tvmult_add(parallel::distributed::Vector<Number> &dst,
                  const parallel::distributed::Vector<Number> &src) const;

  void vmult_add(parallel::distributed::Vector<Number> &dst,
                 const parallel::distributed::Vector<Number> &src) const;

  void apply_nullspace_projection(parallel::distributed::Vector<Number> &vec) const;

  types::global_dof_index m() const;

  types::global_dof_index n() const;

  Number el (const unsigned int,  const unsigned int) const;

  void
  initialize_dof_vector(parallel::distributed::Vector<Number> &vector) const;

  void
  compute_inverse_diagonal (parallel::distributed::Vector<Number> &inverse_diagonal_entries);

  const AlignedVector<VectorizedArray<Number> > &
  get_array_penalty_parameter() const
  {
    return array_penalty_parameter;
  }

  Number get_penalty_factor() const
  {
    return solver_data.penalty_factor * (fe_degree + 1.0) * (fe_degree + 1.0);
  }

  const PoissonSolverData<dim> &
  get_solver_data() const
  {
    return solver_data;
  }

private:

  template <int degree>
  void
  local_apply (const MatrixFree<dim,Number>                &data,
               parallel::distributed::Vector<Number>       &dst,
               const parallel::distributed::Vector<Number> &src,
               const std::pair<unsigned int,unsigned int>  &cell_range) const;

  template <int degree>
  void
  local_apply_face (const MatrixFree<dim,Number>                &data,
                    parallel::distributed::Vector<Number>       &dst,
                    const parallel::distributed::Vector<Number> &src,
                    const std::pair<unsigned int,unsigned int>  &face_range) const;

  template <int degree>
  void
  local_apply_boundary (const MatrixFree<dim,Number>                &data,
                        parallel::distributed::Vector<Number>       &dst,
                        const parallel::distributed::Vector<Number> &src,
                        const std::pair<unsigned int,unsigned int>  &face_range) const;

  template <int degree>
  void
  local_diagonal_cell (const MatrixFree<dim,Number>                &data,
                       parallel::distributed::Vector<Number>       &dst,
                       const unsigned int  &,
                       const std::pair<unsigned int,unsigned int>  &cell_range) const;

  template <int degree>
  void
  local_diagonal_face (const MatrixFree<dim,Number>                &data,
                       parallel::distributed::Vector<Number>       &dst,
                       const unsigned int  &,
                       const std::pair<unsigned int,unsigned int>  &face_range) const;

  template <int degree>
  void
  local_diagonal_boundary (const MatrixFree<dim,Number>                &data,
                           parallel::distributed::Vector<Number>       &dst,
                           const unsigned int  &,
                           const std::pair<unsigned int,unsigned int>  &face_range) const;

  const MatrixFree<dim,Number> *data;
  MatrixFree<dim,Number> own_matrix_free_storage;
  PoissonSolverData<dim> solver_data;
  unsigned int fe_degree;
  bool pure_neumann_problem;
  AlignedVector<VectorizedArray<Number> > array_penalty_parameter;
  mutable parallel::distributed::Vector<Number> tmp_projection_vector;
};



template <int dim, typename Operator>
class MGTransferMF : public MGTransferMatrixFree<dim, typename Operator::value_type>
{
public:
  MGTransferMF()
    :
    laplace_operator (0)
  {}

  void set_laplace_operator(const MGLevelObject<Operator> &laplace)
  {
    laplace_operator = &laplace;
  }

  /**
   * Overload copy_to_mg from MGTransferMatrixFree
   */
  template <class InVector, int spacedim>
  void
  copy_to_mg (const DoFHandler<dim,spacedim> &mg_dof,
              MGLevelObject<parallel::distributed::Vector<typename Operator::value_type> > &dst,
              const InVector &src) const
  {
    AssertThrow(laplace_operator != 0, ExcNotInitialized());
    for (unsigned int level=dst.min_level();
         level<=dst.max_level(); ++level)
      (*laplace_operator)[level].initialize_dof_vector(dst[level]);
    MGLevelGlobalTransfer<parallel::distributed::Vector<typename Operator::value_type> >::copy_to_mg(mg_dof, dst, src);
  }

private:
  const MGLevelObject<Operator> *laplace_operator;
};



template <int dim>
class PoissonSolver
{
public:
  typedef float Number;

  PoissonSolver() {}

  void initialize (const Mapping<dim> &mapping,
                   const MatrixFree<dim,double> &matrix_free,
                   const PoissonSolverData<dim> &solver_data);

  unsigned int solve (parallel::distributed::Vector<double> &dst,
                      const parallel::distributed::Vector<double> &src) const;

  const LaplaceOperator<dim,double> &
  get_matrix() const
  {
    return global_matrix;
  }

private:
  MPI_Comm mpi_communicator;

  LaplaceOperator<dim,double> global_matrix;

  typedef LaplaceOperator<dim,Number> LevelMatrixType;

  MGLevelObject<LevelMatrixType> mg_matrices;
  MGTransferMF<dim,LevelMatrixType> mg_transfer;

  typedef PreconditionChebyshev<LevelMatrixType,parallel::distributed::Vector<Number> > SMOOTHER;
  MGSmootherPrecondition<LevelMatrixType, SMOOTHER, parallel::distributed::Vector<Number> >
  mg_smoother;

  std_cxx11::shared_ptr<MGCoarseGridBase<parallel::distributed::Vector<Number> > > mg_coarse;

  std_cxx11::shared_ptr<mg::Matrix<parallel::distributed::Vector<Number> > > mg_matrix;

  std_cxx11::shared_ptr<Multigrid<parallel::distributed::Vector<Number> > > mg;

  std_cxx11::shared_ptr<PreconditionMG<dim, parallel::distributed::Vector<Number>,
                                       MGTransferMF<dim,LevelMatrixType> > > preconditioner;
};
