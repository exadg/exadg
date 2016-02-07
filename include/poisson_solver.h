
#ifndef __indexa_poisson_solver_h
#define __indexa_poisson_solver_h

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
    poisson_dof_index(0),
    poisson_quad_index(0),
    penalty_factor(1.),
    solver_tolerance(1e-5),
    smoother_poly_degree(5),
    smoother_smoothing_range(20),
    coarse_solver(coarse_chebyshev_smoother)
  {}

  // If an external MatrixFree object is given which can contain other
  // components than the variable for which the Poisson equation should be
  // solved, this selects the correct DoFHandler component
  unsigned int poisson_dof_index;

  // If an external MatrixFree object is given which can contain other
  // quadrature formulas than the quadrature formula which should be used by
  // the Poisson solver, this selects the correct quadrature index
  unsigned int poisson_quad_index;

  // The penalty parameter for the symmetric interior penalty method is
  // computed as penalty_factor * (fe_degree+1)^2 /
  // characteristic_element_length. This variable gives the scaling factor
  double penalty_factor;

  // Specifies the boundary ids with Dirichlet boundary conditions
  std::set<types::boundary_id> dirichlet_boundaries;

  // Specifies the boundary ids with Neumann boundary conditions
  std::set<types::boundary_id> neumann_boundaries;

  // If periodic boundaries are present, this variable collects matching faces
  // on the two sides of the domain
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs_level0;

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

  // Initialization given a DoFHandler object. This internally creates a
  // MatrixFree object. Note that the integration routines and loop bounds
  // from MatrixFree cannot be combined with evaluators from another
  // MatrixFree object.
  void reinit (const DoFHandler<dim> &dof_handler,
               const Mapping<dim> &mapping,
               const PoissonSolverData<dim> &solver_data,
               const MGConstrainedDoFs &mg_constrained_dofs,
               const unsigned int level = numbers::invalid_unsigned_int);

  // Checks whether the boundary conditions are consistent, i.e., no overlap
  // between the Dirichlet, Neumann, and periodic parts. The return value of
  // this function indicates whether a pure Neumann problem is detected (and
  // additional measures for making the linear system non-singular are
  // necessary).
  static bool verify_boundary_conditions(const DoFHandler<dim>        &dof_handler,
                                         const PoissonSolverData<dim> &solver_data);

  // Performs a matrix-vector multiplication
  void vmult(parallel::distributed::Vector<Number> &dst,
             const parallel::distributed::Vector<Number> &src) const;

  // Performs a transpose matrix-vector multiplication. Since the Poisson
  // operator is symmetric, this simply redirects to the vmult call.
  void Tvmult(parallel::distributed::Vector<Number> &dst,
              const parallel::distributed::Vector<Number> &src) const;

  // Performs a transpose matrix-vector multiplication, adding the result in
  // the previous content of dst. Since the Poisson operator is symmetric,
  // this simply redirects to the vmult_add call.
  void Tvmult_add(parallel::distributed::Vector<Number> &dst,
                  const parallel::distributed::Vector<Number> &src) const;

  // Performs a matrix-vector multiplication, adding the result in
  // the previous content of dst
  void vmult_add(parallel::distributed::Vector<Number> &dst,
                 const parallel::distributed::Vector<Number> &src) const;

  // Performs the part of the matrix-vector multiplication on refinement edges
  // that distributes the residual to the refinement edge (used in the
  // restriction phase)
  void vmult_interface_down(parallel::distributed::Vector<Number> &dst,
                            const parallel::distributed::Vector<Number> &src) const;

  // Performs the part of the matrix-vector multiplication on refinement edges
  // that takes an input from the refinement edge to the interior (used in the
  // prolongation phase)
  void vmult_interface_up(parallel::distributed::Vector<Number> &dst,
                          const parallel::distributed::Vector<Number> &src) const;

  // For a pure Neumann problem, this call subtracts the mean value of 'vec'
  // from all entries, ensuring that all operations with this matrix lie in
  // the subspace of zero mean
  void apply_nullspace_projection(parallel::distributed::Vector<Number> &vec) const;

  // Returns the number of global rows of this matrix
  types::global_dof_index m() const;

  // Returns the number of global columns of this matrix, the same as m().
  types::global_dof_index n() const;

  // Function to provide access to an element of this operator. Since this is
  // a matrix-free implementation, no access is implemented. (Diagonal
  // elements can be filled by compute_inverse_diagonal).
  Number el (const unsigned int,  const unsigned int) const;

  // Initializes a vector with the correct parallel layout suitable for
  // multiplication in vmult() and friends. This includes setting the local
  // size and an appropriate ghost layer as necessary by the specific access
  // pattern.
  void
  initialize_dof_vector(parallel::distributed::Vector<Number> &vector) const;

  // Compute the inverse diagonal entries of this operator. This method is
  // rather expensive as the current implementation computes everything that
  // would be needed for a sparse matrix, but only keeping the diagonal. The
  // vector needs not be correctly set at entry as it will be sized
  // appropriately by initialize_dof_vector internally.
  void
  compute_inverse_diagonal (parallel::distributed::Vector<Number> &inverse_diagonal_entries);

  // Returns a reference to the ratio between the element surface and the
  // element volume for the symmetric interior penalty method (only available
  // in the DG case).
  const AlignedVector<VectorizedArray<Number> > &
  get_array_penalty_parameter() const
  {
    return array_penalty_parameter;
  }

  // Returns the current factor by which array_penalty_parameter() is
  // multiplied in the definition of the interior penalty parameter through
  // get_array_penalty_parameter()[cell] * get_penalty_factor().
  Number get_penalty_factor() const
  {
    return solver_data.penalty_factor * (fe_degree + 1.0) * (fe_degree + 1.0);
  }

  // Returns a reference to the data in use.
  const PoissonSolverData<dim> &
  get_solver_data() const
  {
    return solver_data;
  }

private:

  // Computes the array penalty parameter for later use of the symmetric
  // interior penalty method. Called in reinit().
  void compute_array_penalty_parameter(const Mapping<dim> &mapping);

  // Runs the loop over all cells and faces for use in matrix-vector
  // multiplication, adding the result in the previous content of dst
  void run_vmult_loop(parallel::distributed::Vector<Number> &dst,
                      const parallel::distributed::Vector<Number> &src) const;

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
  bool apply_mean_value_constraint;
  AlignedVector<VectorizedArray<Number> > array_penalty_parameter;
  mutable parallel::distributed::Vector<Number> tmp_projection_vector;

  std::vector<unsigned int> edge_constrained_indices;
  mutable std::vector<std::pair<Number,Number> > edge_constrained_values;
  bool have_interface_matrices;
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



template <typename LAPLACEOPERATOR>
class MGInterfaceMatrix : public Subscriptor
{
public:
  void initialize (const LAPLACEOPERATOR &laplace)
  {
    this->laplace = &laplace;
  }

  void vmult (parallel::distributed::Vector<typename LAPLACEOPERATOR::value_type> &dst,
              const parallel::distributed::Vector<typename LAPLACEOPERATOR::value_type> &src) const
  {
    laplace->vmult_interface_down(dst, src);
  }

  void Tvmult (parallel::distributed::Vector<typename LAPLACEOPERATOR::value_type> &dst,
               const parallel::distributed::Vector<typename LAPLACEOPERATOR::value_type> &src) const
  {
    laplace->vmult_interface_up(dst, src);
  }

private:
  SmartPointer<const LAPLACEOPERATOR> laplace;
};



// Implementation of a Poisson solver class that wraps around the matrix-free
// implementation in LaplaceOperator.
template <int dim>
class PoissonSolver
{
public:
  typedef float Number;

  // Constructor. Does nothing.
  PoissonSolver() {}

  // Initialize the LaplaceOperator implementing the matrix-vector product and
  // the multigird preconditioner object given a mapping object and a
  // matrix_free object defining the active cells. Adaptive meshes with
  // hanging nodes are allowed but currently only the case of continuous
  // elements is implemented. (The DG case is not very difficult.)
  void initialize (const Mapping<dim> &mapping,
                   const MatrixFree<dim,double> &matrix_free,
                   const PoissonSolverData<dim> &solver_data);

  // Solves a linear system. The prerequisite for this function is that
  // initialize() has been called.
  unsigned int solve (parallel::distributed::Vector<double> &dst,
                      const parallel::distributed::Vector<double> &src) const;

  // Returns a reference to the underlying matrix object. The object only
  // makes sense after a call to initialize().
  const LaplaceOperator<dim,double> &
  get_matrix() const
  {
    return global_matrix;
  }

private:
  MPI_Comm mpi_communicator;

  LaplaceOperator<dim,double> global_matrix;

  typedef LaplaceOperator<dim,Number> LevelMatrixType;

  MGConstrainedDoFs mg_constrained_dofs;
  MGLevelObject<LevelMatrixType> mg_matrices;
  MGLevelObject<MGInterfaceMatrix<LevelMatrixType> > mg_interface_matrices;
  MGTransferMF<dim,LevelMatrixType> mg_transfer;

  typedef PreconditionChebyshev<LevelMatrixType,parallel::distributed::Vector<Number> > SMOOTHER;
  MGSmootherPrecondition<LevelMatrixType, SMOOTHER, parallel::distributed::Vector<Number> >
  mg_smoother;

  std_cxx11::shared_ptr<MGCoarseGridBase<parallel::distributed::Vector<Number> > > mg_coarse;

  std_cxx11::shared_ptr<mg::Matrix<parallel::distributed::Vector<Number> > > mg_matrix;
  std_cxx11::shared_ptr<mg::Matrix<parallel::distributed::Vector<Number> > > mg_interface;

  std_cxx11::shared_ptr<Multigrid<parallel::distributed::Vector<Number> > > mg;

  std_cxx11::shared_ptr<PreconditionMG<dim, parallel::distributed::Vector<Number>,
                                       MGTransferMF<dim,LevelMatrixType> > > preconditioner;
};


#endif // ifndef __indexa_poisson_solver_h
