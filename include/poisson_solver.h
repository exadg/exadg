
#ifndef __indexa_poisson_solver_h
#define __indexa_poisson_solver_h

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/lac/parallel_vector.h>

using namespace dealii;

#include "MatrixOperatorBase.h"

#include "BoundaryDescriptorLaplace.h"

template<int dim>
struct LaplaceOperatorData
{
  LaplaceOperatorData ()
    :
    laplace_dof_index(0),
    laplace_quad_index(0),
    penalty_factor(1.)
  {}

  // If an external MatrixFree object is given which can contain other
  // components than the variable for which the Poisson equation should be
  // solved, this selects the correct DoFHandler component
  unsigned int laplace_dof_index;

  // If an external MatrixFree object is given which can contain other
  // quadrature formulas than the quadrature formula which should be used by
  // the Poisson solver, this selects the correct quadrature index
  unsigned int laplace_quad_index;

  // The penalty parameter for the symmetric interior penalty method is
  // computed as penalty_factor * (fe_degree+1)^2 /
  // characteristic_element_length. This variable gives the scaling factor
  double penalty_factor;

//  // Specifies the boundary ids with Dirichlet boundary conditions
//  std::set<types::boundary_id> dirichlet_boundaries;
//
//  // Specifies the boundary ids with Neumann boundary conditions
//  std::set<types::boundary_id> neumann_boundaries;

  // boundary descriptor:
  std_cxx11::shared_ptr<BoundaryDescriptorLaplace<dim> >  bc;

  // If periodic boundaries are present, this variable collects matching faces
  // on the two sides of the domain
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs_level0;
};

// Generic implementation of Laplace operator for both continuous elements
// (FE_Q) and discontinuous elements (FE_DGQ).
template <int dim, typename Number=double>
class LaplaceOperator : public MatrixOperatorBase
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
              const LaplaceOperatorData<dim>     &operator_data);

  // Initialization given a DoFHandler object. This internally creates a
  // MatrixFree object. Note that the integration routines and loop bounds
  // from MatrixFree cannot be combined with evaluators from another
  // MatrixFree object.
  void reinit (const DoFHandler<dim>          &dof_handler,
               const Mapping<dim>             &mapping,
               const LaplaceOperatorData<dim> &operator_data,
               const MGConstrainedDoFs        &mg_constrained_dofs,
               const unsigned int             level = numbers::invalid_unsigned_int);

  // Checks whether the boundary conditions are consistent, i.e., no overlap
  // between the Dirichlet, Neumann, and periodic parts. The return value of
  // this function indicates whether a pure Neumann problem is detected (and
  // additional measures for making the linear system non-singular are
  // necessary).
  static bool verify_boundary_conditions(const DoFHandler<dim>          &dof_handler,
                                         const LaplaceOperatorData<dim> &operator_data);

  // Performs a matrix-vector multiplication
  void vmult(parallel::distributed::Vector<Number>       &dst,
             const parallel::distributed::Vector<Number> &src) const;

  // Performs a transpose matrix-vector multiplication. Since the Poisson
  // operator is symmetric, this simply redirects to the vmult call.
  void Tvmult(parallel::distributed::Vector<Number>       &dst,
              const parallel::distributed::Vector<Number> &src) const;

  // Performs a transpose matrix-vector multiplication, adding the result in
  // the previous content of dst. Since the Poisson operator is symmetric,
  // this simply redirects to the vmult_add call.
  void Tvmult_add(parallel::distributed::Vector<Number>       &dst,
                  const parallel::distributed::Vector<Number> &src) const;

  // Performs a matrix-vector multiplication, adding the result in
  // the previous content of dst
  void vmult_add(parallel::distributed::Vector<Number>       &dst,
                 const parallel::distributed::Vector<Number> &src) const;

  // Performs the matrix-vector multiplication including the refinement edges
  // that distributes the residual to the refinement edge (used in the
  // restriction phase)
  void vmult_interface_down(parallel::distributed::Vector<Number>       &dst,
                            const parallel::distributed::Vector<Number> &src) const;

  // Performs the matrix-vector multiplication including the refinement edges
  // that takes an input from the refinement edge to the interior (used in the
  // prolongation phase)
  void vmult_add_interface_up(parallel::distributed::Vector<Number>       &dst,
                              const parallel::distributed::Vector<Number> &src) const;

  // Evaluates inhomogeneous parts of boundary face integrals occuring on
  // the right-hand side of the linear system of equations
  void rhs(parallel::distributed::Vector<Number> &dst) const;

  void rhs_add(parallel::distributed::Vector<Number> &dst) const;

  // For a pure Neumann problem, this call subtracts the mean value of 'vec'
  // from all entries, ensuring that all operations with this matrix lie in
  // the subspace of zero mean
  void apply_nullspace_projection(parallel::distributed::Vector<Number> &vec) const;

  // This allows to disable the mean value constraint on the matrix even
  // though a zero mode has been detected. Handle this with care.
  void disable_mean_value_constraint();

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
  calculate_inverse_diagonal (parallel::distributed::Vector<Number> &inverse_diagonal_entries) const;

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
    return operator_data.penalty_factor * (fe_degree + 1.0) * (fe_degree + 1.0);
  }

  const LaplaceOperatorData<dim> &
  get_operator_data() const
  {
    return operator_data;
  }

  bool is_empty_locally() const
  {
    return data->n_macro_cells() == 0;
  }

  const MatrixFree<dim,Number> & get_data() const
  {
    return *data;
  }

private:

  // Computes the array penalty parameter for later use of the symmetric
  // interior penalty method. Called in reinit().
  void compute_array_penalty_parameter(const Mapping<dim> &mapping);

  // Runs the loop over all cells and faces for use in matrix-vector
  // multiplication, adding the result in the previous content of dst
  void run_vmult_loop(parallel::distributed::Vector<Number>       &dst,
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

  // Runs the loop over all cells and interior faces (does nothing)
  // and boundary faces (to evaluate inhomgeneous boundary conditions)
  void run_rhs_loop(parallel::distributed::Vector<Number>       &dst) const;

  template <int degree>
  void
  local_rhs (const MatrixFree<dim,Number>                &,
             parallel::distributed::Vector<Number>       &,
             const parallel::distributed::Vector<Number> &,
             const std::pair<unsigned int,unsigned int>  &) const;

  template <int degree>
  void
  local_rhs_face (const MatrixFree<dim,Number>                &,
                  parallel::distributed::Vector<Number>       &,
                  const parallel::distributed::Vector<Number> &,
                  const std::pair<unsigned int,unsigned int>  &) const;

  template <int degree>
  void
  local_rhs_boundary (const MatrixFree<dim,Number>                &data,
                      parallel::distributed::Vector<Number>       &dst,
                      const parallel::distributed::Vector<Number> &src,
                      const std::pair<unsigned int,unsigned int>  &face_range) const;

  template <int degree>
  void
  local_diagonal_cell (const MatrixFree<dim,Number>                &data,
                       parallel::distributed::Vector<Number>       &dst,
                       const unsigned int                          &,
                       const std::pair<unsigned int,unsigned int>  &cell_range) const;

  template <int degree>
  void
  local_diagonal_face (const MatrixFree<dim,Number>                &data,
                       parallel::distributed::Vector<Number>       &dst,
                       const unsigned int                          &,
                       const std::pair<unsigned int,unsigned int>  &face_range) const;

  template <int degree>
  void
  local_diagonal_boundary (const MatrixFree<dim,Number>                &data,
                           parallel::distributed::Vector<Number>       &dst,
                           const unsigned int                          &,
                           const std::pair<unsigned int,unsigned int>  &face_range) const;

  const MatrixFree<dim,Number> *data;
  MatrixFree<dim,Number> own_matrix_free_storage;

  LaplaceOperatorData<dim> operator_data;

  unsigned int fe_degree;
  bool needs_mean_value_constraint;
  bool apply_mean_value_constraint_in_matvec;
  AlignedVector<VectorizedArray<Number> > array_penalty_parameter;
  mutable parallel::distributed::Vector<Number> tmp_projection_vector;

  std::vector<unsigned int> edge_constrained_indices;
  mutable std::vector<std::pair<Number,Number> > edge_constrained_values;
};

#include "PoissonSolverInputParameters.h"
#include "MultigridInputParameters.h"

struct PoissonSolverData
{
  PoissonSolverData ()
    :
    max_iter(1e4),
    solver_tolerance_rel(1e-5),
    solver_tolerance_abs(1e-12),
    use_preconditioner(false)
  {}

  // maximum number of iterations
  unsigned int max_iter;
  // Sets the tolerance for the linear solver
  double solver_tolerance_rel;
  // Sets the tolerance for the linear solver
  double solver_tolerance_abs;

  bool use_preconditioner;
};

#include "Preconditioner.h"

// Implementation of a Poisson solver class that wraps around the matrix-free
// implementation in LaplaceOperator.
template <int dim, typename value_type=double>
class PoissonSolver
{
public:
  typedef float Number;

  // Constructor. Does nothing.
  PoissonSolver()
    :
    global_matrix(nullptr),
    preconditioner(nullptr)
  {}

  // Initialize the LaplaceOperator implementing the matrix-vector product and
  // the multigird preconditioner object given a mapping object and a
  // matrix_free object defining the active cells. Adaptive meshes with
  // hanging nodes are allowed but currently only the case of continuous
  // elements is implemented. (The DG case is not very difficult.)
  void initialize (const LaplaceOperator<dim,value_type>  &laplace_operator,
                   const PreconditionerBase<value_type>   &preconditioner_in,
                   const MatrixFree<dim,value_type>       &matrix_free,
                   const PoissonSolverData                &solver_data);

  // Solves a linear system. The prerequisite for this function is that
  // initialize() has been called.
  unsigned int solve (parallel::distributed::Vector<value_type>       &dst,
                      const parallel::distributed::Vector<value_type> &src) const;

private:
  MPI_Comm mpi_communicator;

  LaplaceOperator<dim,value_type> const *global_matrix;
  PreconditionerBase<value_type> const *preconditioner;

  PoissonSolverData solver_data;
};


#endif // ifndef __indexa_poisson_solver_h
