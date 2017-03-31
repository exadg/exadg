
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

#include "operators/matrix_operator_base.h"

#include "../poisson/boundary_descriptor_laplace.h"

template<int dim>
struct LaplaceOperatorData
{
  LaplaceOperatorData ()
    :
    laplace_dof_index(0),
    laplace_quad_index(0),
    penalty_factor(1.),
    needs_mean_value_constraint(false)
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

  // pure Neumann BC's - set needs_mean_value_constraint to true in order
  // to solve a transformed system of equations based on Krylov projection
  bool needs_mean_value_constraint;

  // boundary descriptor:
  std_cxx11::shared_ptr<BoundaryDescriptorLaplace<dim> >  bc;

  // If periodic boundaries are present, this variable collects matching faces
  // on the two sides of the domain
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs_level0;
};

// Generic implementation of Laplace operator for both continuous elements
// (FE_Q) and discontinuous elements (FE_DGQ).
template <int dim, int degree, typename Number=double>
class LaplaceOperator : public MatrixOperatorBase
{
public:
  typedef Number value_type;

  /*
   * Constructor.
   */
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

  /*
   *  Apply block Jacobi preconditioner
   */
  void apply_block_jacobi (parallel::distributed::Vector<Number>       &dst,
                           parallel::distributed::Vector<Number> const &src) const;

  /*
   *  Update block Jacobi preconditioner
   */
  void update_block_jacobi () const;

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
    return operator_data.penalty_factor * (degree + 1.0) * (degree + 1.0);
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

  template<typename FEEvaluation>
  inline void do_cell_integral(FEEvaluation &fe_eval) const
  {
    fe_eval.evaluate (false,true,false);

    for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
    {
      fe_eval.submit_gradient (fe_eval.get_gradient(q), q);
    }

    fe_eval.integrate (false,true);
  }

  void local_apply (const MatrixFree<dim,Number>                &data,
                    parallel::distributed::Vector<Number>       &dst,
                    const parallel::distributed::Vector<Number> &src,
                    const std::pair<unsigned int,unsigned int>  &cell_range) const;

  void local_apply_face (const MatrixFree<dim,Number>                &data,
                         parallel::distributed::Vector<Number>       &dst,
                         const parallel::distributed::Vector<Number> &src,
                         const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_apply_boundary (const MatrixFree<dim,Number>                &data,
                             parallel::distributed::Vector<Number>       &dst,
                             const parallel::distributed::Vector<Number> &src,
                             const std::pair<unsigned int,unsigned int>  &face_range) const;

  // Runs the loop over all cells and interior faces (does nothing)
  // and boundary faces (to evaluate inhomgeneous boundary conditions)
  void run_rhs_loop(parallel::distributed::Vector<Number>       &dst) const;

  void local_rhs (const MatrixFree<dim,Number>                &,
                  parallel::distributed::Vector<Number>       &,
                  const parallel::distributed::Vector<Number> &,
                  const std::pair<unsigned int,unsigned int>  &) const;

  void local_rhs_face (const MatrixFree<dim,Number>                &,
                       parallel::distributed::Vector<Number>       &,
                       const parallel::distributed::Vector<Number> &,
                       const std::pair<unsigned int,unsigned int>  &) const;

  void local_rhs_boundary (const MatrixFree<dim,Number>                &data,
                           parallel::distributed::Vector<Number>       &dst,
                           const parallel::distributed::Vector<Number> &src,
                           const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_diagonal_cell (const MatrixFree<dim,Number>                &data,
                            parallel::distributed::Vector<Number>       &dst,
                            const unsigned int                          &,
                            const std::pair<unsigned int,unsigned int>  &cell_range) const;

  void local_diagonal_face (const MatrixFree<dim,Number>                &data,
                            parallel::distributed::Vector<Number>       &dst,
                            const unsigned int                          &,
                            const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_diagonal_boundary (const MatrixFree<dim,Number>                &data,
                                parallel::distributed::Vector<Number>       &dst,
                                const unsigned int                          &,
                                const std::pair<unsigned int,unsigned int>  &face_range) const;

  /*
   *  This function calculates the block Jacobi matrices.
   */
  void add_block_jacobi_matrices(std::vector<LAPACKFullMatrix<Number> > &matrices) const;

  void cell_loop_calculate_block_jacobi_matrices (const MatrixFree<dim,Number>                 &data,
                                                  std::vector<LAPACKFullMatrix<Number> >       &matrices,
                                                  const parallel::distributed::Vector<Number>  &,
                                                  const std::pair<unsigned int,unsigned int>   &cell_range) const;

  void face_loop_calculate_block_jacobi_matrices (const MatrixFree<dim,Number>                &data,
                                                  std::vector<LAPACKFullMatrix<Number> >      &matrices,
                                                  const parallel::distributed::Vector<Number> &,
                                                  const std::pair<unsigned int,unsigned int>  &face_range) const;

  void boundary_face_loop_calculate_block_jacobi_matrices (const MatrixFree<dim,Number>                &data,
                                                           std::vector<LAPACKFullMatrix<Number> >      &matrices,
                                                           const parallel::distributed::Vector<Number> &,
                                                           const std::pair<unsigned int,unsigned int>  &face_range) const;

  /*
   *  This function loops over all cells and applies the inverse block Jacobi matrices elementwise.
   */
  void cell_loop_apply_inverse_block_jacobi_matrices (const MatrixFree<dim,Number>                 &data,
                                                      parallel::distributed::Vector<Number>        &dst,
                                                      const parallel::distributed::Vector<Number>  &src,
                                                      const std::pair<unsigned int,unsigned int>   &cell_range) const;

  const MatrixFree<dim,Number> *data;
  MatrixFree<dim,Number> own_matrix_free_storage;

  LaplaceOperatorData<dim> operator_data;

  bool needs_mean_value_constraint;
  bool apply_mean_value_constraint_in_matvec;
  AlignedVector<VectorizedArray<Number> > array_penalty_parameter;
  mutable parallel::distributed::Vector<Number> tmp_projection_vector;

  std::vector<unsigned int> edge_constrained_indices;
  mutable std::vector<std::pair<Number,Number> > edge_constrained_values;

  mutable std::vector<LAPACKFullMatrix<Number> > matrices;
  mutable bool block_jacobi_matrices_have_been_initialized;
};


#include <deal.II/base/function_lib.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/distributed/tria_base.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/numerics/vector_tools.h>

#include "solvers_and_preconditioners/block_jacobi_matrices.h"


template <int dim, int degree, typename Number>
LaplaceOperator<dim,degree,Number>::LaplaceOperator ()
  :
  data (0),
  needs_mean_value_constraint (false),
  apply_mean_value_constraint_in_matvec (false),
  block_jacobi_matrices_have_been_initialized(false)
{}


template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::clear()
{
  operator_data = LaplaceOperatorData<dim>();
  data = 0;
  needs_mean_value_constraint = false;
  apply_mean_value_constraint_in_matvec = false;
  own_matrix_free_storage.clear();
  tmp_projection_vector.reinit(0);
}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::reinit (const MatrixFree<dim,Number>       &mf_data,
                                                 const Mapping<dim>                 &mapping,
                                                 const LaplaceOperatorData<dim>     &operator_data)
{
  this->data = &mf_data;
  this->operator_data = operator_data;
//  this->fe_degree = mf_data.get_dof_handler(operator_data.laplace_dof_index).get_fe().degree;
  AssertThrow (Utilities::fixed_power<dim>((unsigned int)degree+1) ==
               mf_data.get_n_q_points(operator_data.laplace_quad_index),
               ExcMessage("Expected fe_degree+1 quadrature points"));

  compute_array_penalty_parameter(mapping);

  // TODO
//  // Check whether the Poisson matrix is singular when applied to a vector
//  // consisting of only ones (except for constrained entries)
//  parallel::distributed::Vector<Number> in_vec, out_vec;
//  initialize_dof_vector(in_vec);
//  initialize_dof_vector(out_vec);
//  in_vec = 1;
//  const std::vector<unsigned int> &constrained_entries =
//    mf_data.get_constrained_dofs(operator_data.laplace_dof_index);
//  for (unsigned int i=0; i<constrained_entries.size(); ++i)
//    in_vec.local_element(constrained_entries[i]) = 0;
//  vmult_add(out_vec, in_vec);
//  const double linfty_norm = out_vec.linfty_norm();
//
//  // since we cannot know the magnitude of the entries at this point (the
//  // diagonal entries would be a guideline but they are not available here),
//  // we instead multiply by a random vector
//  for (unsigned int i=0; i<in_vec.local_size(); ++i)
//    in_vec.local_element(i) = (double)rand()/RAND_MAX;
//  vmult(out_vec, in_vec);
//  const double linfty_norm_compare = out_vec.linfty_norm();
//
//  // use mean value constraint if the infty norm with the one vector is very small
//  needs_mean_value_constraint =
//    linfty_norm / linfty_norm_compare < std::pow(std::numeric_limits<Number>::epsilon(), 2./3.);

  // TODO
  // alternative approach: specify whether the system of equations is singular by using an input parameter
  needs_mean_value_constraint = operator_data.needs_mean_value_constraint;

  apply_mean_value_constraint_in_matvec = needs_mean_value_constraint;
}



namespace
{
  template <int dim>
  void add_periodicity_constraints(const unsigned int level,
                                   const unsigned int target_level,
                                   const typename DoFHandler<dim>::face_iterator face1,
                                   const typename DoFHandler<dim>::face_iterator face2,
                                   ConstraintMatrix &constraints)
  {
    if (level == 0)
      {
        const unsigned int dofs_per_face = face1->get_fe(0).dofs_per_face;
        std::vector<types::global_dof_index> dofs_1(dofs_per_face);
        std::vector<types::global_dof_index> dofs_2(dofs_per_face);

        face1->get_mg_dof_indices(target_level, dofs_1, 0);
        face2->get_mg_dof_indices(target_level, dofs_2, 0);

        for (unsigned int i=0; i<dofs_per_face; ++i)
          if (constraints.can_store_line(dofs_2[i]) &&
              constraints.can_store_line(dofs_1[i]) &&
              !constraints.is_constrained(dofs_2[i]))
            {
              constraints.add_line(dofs_2[i]);
              constraints.add_entry(dofs_2[i], dofs_1[i], 1.);
            }
      }
    else if (face1->has_children() && face2->has_children())
      {
        for (unsigned int c=0; c<face1->n_children(); ++c)
          add_periodicity_constraints<dim>(level-1, target_level, face1->child(c),
                                           face2->child(c), constraints);
      }
  }
}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::reinit (const DoFHandler<dim>           &dof_handler,
                                                 const Mapping<dim>              &mapping,
                                                 const LaplaceOperatorData<dim>  &operator_data,
                                                 const MGConstrainedDoFs         &mg_constrained_dofs,
                                                 const unsigned int              level)
{
  clear();
  this->operator_data = operator_data;

  const QGauss<1> quad(dof_handler.get_fe().degree+1);
  typename MatrixFree<dim,Number>::AdditionalData addit_data;
  addit_data.tasks_parallel_scheme = MatrixFree<dim,Number>::AdditionalData::none;
  if (dof_handler.get_fe().dofs_per_vertex == 0)
    addit_data.build_face_info = true;
  addit_data.level_mg_handler = level;
  addit_data.mpi_communicator =
    dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation()) ?
    (dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation()))->get_communicator() : MPI_COMM_SELF;
  addit_data.periodic_face_pairs_level_0 = operator_data.periodic_face_pairs_level0;

  ConstraintMatrix constraints;
  const bool is_feq = dof_handler.get_fe().dofs_per_vertex > 0;

  // For continuous elements, add the constraints due to hanging nodes and
  // boundary conditions
  if (is_feq && level == numbers::invalid_unsigned_int)
  {
    ZeroFunction<dim> zero_function(dof_handler.get_fe().n_components());
    typename FunctionMap<dim>::type dirichlet_boundary;
    for (typename std::map<types::boundary_id, std_cxx11::shared_ptr<Function<dim> > >::const_iterator it =
           operator_data.bc->dirichlet.begin();
         it != operator_data.bc->dirichlet.end(); ++it)
      dirichlet_boundary[it->first] = &zero_function;

    IndexSet relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
    constraints.reinit(relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    // add periodicity constraints
    std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator> > periodic_faces;
    for (typename std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> >::const_iterator
           it = operator_data.periodic_face_pairs_level0.begin();
         it != operator_data.periodic_face_pairs_level0.end(); ++it)
    {
      GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator> periodic;
      for (unsigned int i=0; i<2; ++i)
      {
        periodic.cell[i] = typename DoFHandler<dim>::cell_iterator
          (&dof_handler.get_triangulation(),
           it->cell[i]->level(), it->cell[i]->index(), &dof_handler);
        periodic.face_idx[i] = it->face_idx[i];
      }
      periodic.orientation = it->orientation;
      periodic.matrix = it->matrix;
      periodic_faces.push_back(periodic);
    }
    DoFTools::make_periodicity_constraints<DoFHandler<dim> > (periodic_faces,
                                                              constraints);

    VectorTools::interpolate_boundary_values(dof_handler, dirichlet_boundary,
                                             constraints);
  }
  else if (is_feq)
  {
    IndexSet relevant_dofs;
    DoFTools::extract_locally_relevant_level_dofs(dof_handler, level,
                                                  relevant_dofs);
    constraints.reinit(relevant_dofs);

    // add periodicity constraints
    for (typename std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> >::const_iterator
           it = operator_data.periodic_face_pairs_level0.begin();
         it != operator_data.periodic_face_pairs_level0.end(); ++it)
    {
      typename DoFHandler<dim>::cell_iterator
        cell1(&dof_handler.get_triangulation(), 0, it->cell[1]->index(), &dof_handler),
        cell0(&dof_handler.get_triangulation(), 0, it->cell[0]->index(), &dof_handler);
      add_periodicity_constraints<dim>(level, level,
                                       cell1->face(it->face_idx[1]),
                                       cell0->face(it->face_idx[0]),
                                       constraints);
    }

    constraints.add_lines(mg_constrained_dofs.get_boundary_indices(level));

    std::vector<types::global_dof_index> interface_indices;
    mg_constrained_dofs.get_refinement_edge_indices(level).fill_index_vector(interface_indices);
    edge_constrained_indices.clear();
    edge_constrained_indices.reserve(interface_indices.size());
    edge_constrained_values.resize(interface_indices.size());
    const IndexSet &locally_owned = dof_handler.locally_owned_mg_dofs(level);
    for (unsigned int i=0; i<interface_indices.size(); ++i)
      if (locally_owned.is_element(interface_indices[i]))
        edge_constrained_indices.push_back(locally_owned.index_within_set(interface_indices[i]));
  }

  // constraint zeroth DoF in continuous case (the mean value constraint will
  // be applied in the DG case). In case we have interface matrices, there are
  // Dirichlet constraints on parts of the boundary and no such transformation
  // is required.
  if (verify_boundary_conditions(dof_handler, operator_data)
      && is_feq && Utilities::MPI::sum(edge_constrained_indices.size(),addit_data.mpi_communicator)==0
      && constraints.can_store_line(0))
  {
    // if dof 0 is constrained, it must be a periodic dof, so we take the
    // value on the other side
    types::global_dof_index line_index = 0;
    while (true)
    {
      const std::vector<std::pair<types::global_dof_index,double> >* lines =
        constraints.get_constraint_entries(line_index);
      if (lines == 0)
      {
        constraints.add_line(line_index);
        // add the constraint back to the MGConstrainedDoFs field. This
        // is potentially dangerous but we know what we are doing... ;-)
        if (level != numbers::invalid_unsigned_int)
          const_cast<IndexSet &>(mg_constrained_dofs.get_boundary_indices(level))
            .add_index(line_index);
        break;
      }
      else
      {
        Assert(lines->size() == 1 && std::abs((*lines)[0].second-1.)<1e-15,
               ExcMessage("Periodic index expected, bailing out"));
        line_index = (*lines)[0].first;
      }
    }
  }

  constraints.close();

  LaplaceOperatorData<dim> my_operator_data = operator_data;
  my_operator_data.laplace_dof_index = 0;
  my_operator_data.laplace_quad_index = 0;

  own_matrix_free_storage.reinit(mapping, dof_handler, constraints, quad,
                                 addit_data);

  reinit(own_matrix_free_storage, mapping, my_operator_data);

  // we do not need the mean value constraint for smoothers on the
  // multigrid levels, so we can disable it
  disable_mean_value_constraint();
}



template <int dim, int degree, typename Number>
bool LaplaceOperator<dim,degree,Number>::
verify_boundary_conditions(DoFHandler<dim> const          &dof_handler,
                           LaplaceOperatorData<dim> const &operator_data)
{
  // Check that the Dirichlet and Neumann boundary conditions do not overlap
  std::set<types::boundary_id> periodic_boundary_ids;
  for (unsigned int i=0; i<operator_data.periodic_face_pairs_level0.size(); ++i)
    {
      AssertThrow(operator_data.periodic_face_pairs_level0[i].cell[0]->level() == 0,
                  ExcMessage("Received periodic cell pairs on non-zero level"));
      periodic_boundary_ids.insert(operator_data.periodic_face_pairs_level0[i].cell[0]->face(operator_data.periodic_face_pairs_level0[i].face_idx[0])->boundary_id());
      periodic_boundary_ids.insert(operator_data.periodic_face_pairs_level0[i].cell[1]->face(operator_data.periodic_face_pairs_level0[i].face_idx[1])->boundary_id());
    }

  bool pure_neumann_problem = true;
  const Triangulation<dim> &tria = dof_handler.get_triangulation();
  for (typename Triangulation<dim>::cell_iterator cell = tria.begin();
       cell != tria.end(); ++cell)
    for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      if (cell->at_boundary(f))
        {
          types::boundary_id bid = cell->face(f)->boundary_id();
          if (operator_data.bc->dirichlet.find(bid) !=
              operator_data.bc->dirichlet.end())
            {
              AssertThrow(operator_data.bc->neumann.find(bid) ==
                          operator_data.bc->neumann.end(),
                          ExcMessage("Boundary id " + Utilities::to_string((int)bid) +
                                     " wants to set both Dirichlet and Neumann " +
                                     "boundary conditions, which is impossible!"));
              AssertThrow(periodic_boundary_ids.find(bid) ==
                          periodic_boundary_ids.end(),
                          ExcMessage("Boundary id " + Utilities::to_string((int)bid) +
                                     " wants to set both Dirichlet and periodic " +
                                     "boundary conditions, which is impossible!"));
              pure_neumann_problem = false;
              continue;
            }
          if (operator_data.bc->neumann.find(bid) !=
              operator_data.bc->neumann.end())
            {
              AssertThrow(periodic_boundary_ids.find(bid) ==
                          periodic_boundary_ids.end(),
                          ExcMessage("Boundary id " + Utilities::to_string((int)bid) +
                                     " wants to set both Neumann and periodic " +
                                     "boundary conditions, which is impossible!"));
              continue;
            }
          AssertThrow(periodic_boundary_ids.find(bid) != periodic_boundary_ids.end(),
                      ExcMessage("Boundary id " + Utilities::to_string((int)bid) +
                                 " does neither set Dirichlet, Neumann, nor periodic " +
                                 "boundary conditions! Bailing out."));
        }

  // Check for consistency of 'pure_neumann_problem' over all participating
  // processors
  int my_neumann = pure_neumann_problem;
  MPI_Comm mpi_communicator =
    dynamic_cast<const parallel::Triangulation<dim> *>(&tria) ?
    (dynamic_cast<const parallel::Triangulation<dim> *>(&tria))->get_communicator() :
    MPI_COMM_SELF;
  const int max_pure_neumann = Utilities::MPI::max(my_neumann,
                                                   mpi_communicator);
  int min_pure_neumann = Utilities::MPI::min(my_neumann, mpi_communicator);
  AssertThrow(max_pure_neumann == min_pure_neumann,
              ExcMessage("Neumann/Dirichlet assignment over processors does not match."));

  return pure_neumann_problem;
}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::compute_array_penalty_parameter(const Mapping<dim> &mapping)
{
  std::set<types::boundary_id> periodic_boundary_ids;
  for (unsigned int i=0; i<operator_data.periodic_face_pairs_level0.size(); ++i)
    {
      AssertThrow(operator_data.periodic_face_pairs_level0[i].cell[0]->level() == 0,
                  ExcMessage("Received periodic cell pairs on non-zero level"));
      periodic_boundary_ids.insert(operator_data.periodic_face_pairs_level0[i].cell[0]->face(operator_data.periodic_face_pairs_level0[i].face_idx[0])->boundary_id());
      periodic_boundary_ids.insert(operator_data.periodic_face_pairs_level0[i].cell[1]->face(operator_data.periodic_face_pairs_level0[i].face_idx[1])->boundary_id());
    }

  // Compute penalty parameter for each cell
  array_penalty_parameter.resize(data->n_macro_cells()+data->n_macro_ghost_cells());
  QGauss<dim> quadrature(degree+1);
  FEValues<dim> fe_values(mapping,
                          data->get_dof_handler(operator_data.laplace_dof_index).get_fe(),
                          quadrature, update_JxW_values);
  QGauss<dim-1> face_quadrature(degree+1);
  FEFaceValues<dim> fe_face_values(mapping, data->get_dof_handler(operator_data.laplace_dof_index).get_fe(), face_quadrature, update_JxW_values);

  for (unsigned int i=0; i<data->n_macro_cells()+data->n_macro_ghost_cells(); ++i)
  {
    for (unsigned int v=0; v<data->n_components_filled(i); ++v)
    {
      typename DoFHandler<dim>::cell_iterator cell = data->get_cell_iterator(i,v,operator_data.laplace_dof_index);
      fe_values.reinit(cell);
      double volume = 0;
      for (unsigned int q=0; q<quadrature.size(); ++q)
        volume += fe_values.JxW(q);
      double surface_area = 0;
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      {
        fe_face_values.reinit(cell, f);
        const double factor = (cell->at_boundary(f) &&
                               periodic_boundary_ids.find(cell->face(f)->boundary_id()) ==
                               periodic_boundary_ids.end()) ? 1. : 0.5;
        for (unsigned int q=0; q<face_quadrature.size(); ++q)
          surface_area += fe_face_values.JxW(q) * factor;
      }
      array_penalty_parameter[i][v] = surface_area / volume;
    }
  }
}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::disable_mean_value_constraint()
{
  this->apply_mean_value_constraint_in_matvec = false;
}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::vmult(parallel::distributed::Vector<Number>       &dst,
                                               parallel::distributed::Vector<Number> const &src) const
{
  dst = 0;
  vmult_add(dst, src);
}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::Tvmult(parallel::distributed::Vector<Number>       &dst,
                                                parallel::distributed::Vector<Number> const &src) const
{
  dst = 0;
  vmult_add(dst, src);
}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::Tvmult_add(parallel::distributed::Vector<Number>       &dst,
                                                    parallel::distributed::Vector<Number> const &src) const
{
  vmult_add(dst, src);
}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::vmult_add(parallel::distributed::Vector<Number>       &dst,
                                                   parallel::distributed::Vector<Number> const &src) const
{
  const parallel::distributed::Vector<Number> *actual_src = &src;
  if(apply_mean_value_constraint_in_matvec)
  {
    tmp_projection_vector = src;
    apply_nullspace_projection(tmp_projection_vector);
    actual_src = &tmp_projection_vector;
  }

  // For continuous elements: set zero Dirichlet values on the input vector
  // (and remember the src and dst values because we need to reset them at the
  // end). Note that we should only have edge constrained indices for non-pure
  // Neumann problems.
  for (unsigned int i=0; i<edge_constrained_indices.size(); ++i)
  {
    Assert(!apply_mean_value_constraint_in_matvec, ExcInternalError());

    edge_constrained_values[i] =
      std::pair<Number,Number>(src.local_element(edge_constrained_indices[i]),
                               dst.local_element(edge_constrained_indices[i]));
    const_cast<parallel::distributed::Vector<Number>&>(src).local_element(edge_constrained_indices[i]) = 0.;
  }

  run_vmult_loop(dst, *actual_src);

  // reset edge constrained values, multiply by unit matrix and add into
  // destination
  for (unsigned int i=0; i<edge_constrained_indices.size(); ++i)
  {
    const_cast<parallel::distributed::Vector<Number>&>(src).local_element(edge_constrained_indices[i]) = edge_constrained_values[i].first;
    dst.local_element(edge_constrained_indices[i]) = edge_constrained_values[i].second
      + edge_constrained_values[i].first;
  }

  if (apply_mean_value_constraint_in_matvec)
  {
    apply_nullspace_projection(dst);
  }
}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::run_vmult_loop(parallel::distributed::Vector<Number>       &dst,
                                                        parallel::distributed::Vector<Number> const &src) const
{
  Assert(src.partitioners_are_globally_compatible(*data->get_dof_info(operator_data.laplace_dof_index).vector_partitioner), ExcInternalError());
  Assert(dst.partitioners_are_globally_compatible(*data->get_dof_info(operator_data.laplace_dof_index).vector_partitioner), ExcInternalError());

  data->loop (&LaplaceOperator<dim, degree, Number>::local_apply,
              &LaplaceOperator<dim, degree, Number>::local_apply_face,
              &LaplaceOperator<dim, degree, Number>::local_apply_boundary,
              this, dst, src);

  // Apply Dirichlet boundary conditions in the continuous case by simulating
  // a one in the diagonal (note that the ConstraintMatrix passed to the
  // MatrixFree object takes care of Dirichlet conditions on outer
  // (non-refinement edge) boundaries)
  const std::vector<unsigned int> &
    constrained_dofs = data->get_constrained_dofs(operator_data.laplace_dof_index);
  for (unsigned int i=0; i<constrained_dofs.size(); ++i)
    dst.local_element(constrained_dofs[i]) += src.local_element(constrained_dofs[i]);
}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::vmult_interface_down(parallel::distributed::Vector<Number>       &dst,
                                                              parallel::distributed::Vector<Number> const &src) const
{
  dst = 0;

  // set zero edge constrained values on the input vector (and remember the
  // src and dst values because we need to reset them at the end)
  for (unsigned int i=0; i<edge_constrained_indices.size(); ++i)
  {
    const double src_val = src.local_element(edge_constrained_indices[i]);
    const_cast<parallel::distributed::Vector<Number>&>(src).local_element(edge_constrained_indices[i]) = 0.;
    edge_constrained_values[i] = std::pair<Number,Number>(src_val, 0.);
  }

  run_vmult_loop(dst, src);

  // reset the input vector
  for (unsigned int i=0; i<edge_constrained_indices.size(); ++i)
    const_cast<parallel::distributed::Vector<Number>&>(src).local_element(edge_constrained_indices[i]) = edge_constrained_values[i].first;
}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::vmult_add_interface_up(parallel::distributed::Vector<Number>       &dst,
                                                                parallel::distributed::Vector<Number> const &src) const
{
  for (unsigned int i=0; i<edge_constrained_indices.size(); ++i)
  {
    edge_constrained_values[i] =
      std::pair<Number,Number>(src.local_element(edge_constrained_indices[i]),
                               dst.local_element(edge_constrained_indices[i]));
  }
  run_vmult_loop (dst, src);

  // when transferring back to the finer grid, we need to simulate the
  // diagonal part of the matrix that discards the entries we computed for the
  // edges
  for (unsigned int i=0; i<edge_constrained_indices.size(); ++i)
  {
    dst.local_element(edge_constrained_indices[i]) = edge_constrained_values[i].first
      + edge_constrained_values[i].second;
  }
}
template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::rhs(parallel::distributed::Vector<Number> &dst) const
{
  dst = 0;
  rhs_add(dst);
}

template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::rhs_add(parallel::distributed::Vector<Number> &dst) const
{
  run_rhs_loop(dst);
}


template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::run_rhs_loop(parallel::distributed::Vector<Number> &dst) const
{
  Assert(dst.partitioners_are_globally_compatible(*data->get_dof_info(operator_data.laplace_dof_index).vector_partitioner), ExcInternalError());

  parallel::distributed::Vector<Number> src;

  data->loop (&LaplaceOperator<dim, degree, Number>::local_rhs,
              &LaplaceOperator<dim, degree, Number>::local_rhs_face,
              &LaplaceOperator<dim, degree, Number>::local_rhs_boundary,
              this, dst, src);
}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::apply_nullspace_projection(parallel::distributed::Vector<Number> &vec) const
{
  if (needs_mean_value_constraint)
  {
    const Number mean_val = vec.mean_value();
    vec.add(-mean_val);
  }
}



template <int dim, int degree, typename Number>
types::global_dof_index LaplaceOperator<dim,degree,Number>::m() const
{
  return data->get_vector_partitioner(operator_data.laplace_dof_index)->size();
}



template <int dim, int degree, typename Number>
types::global_dof_index LaplaceOperator<dim,degree,Number>::n() const
{
  return data->get_vector_partitioner(operator_data.laplace_dof_index)->size();
}



template <int dim, int degree, typename Number>
Number LaplaceOperator<dim,degree,Number>::el (const unsigned int,  const unsigned int) const
{
  AssertThrow(false, ExcMessage("Matrix-free does not allow for entry access"));
  return Number();
}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::initialize_dof_vector(parallel::distributed::Vector<Number> &vector) const
{
  if (!vector.partitioners_are_compatible(*data->get_dof_info(operator_data.laplace_dof_index).vector_partitioner))
    data->initialize_dof_vector(vector, operator_data.laplace_dof_index);
}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::calculate_inverse_diagonal (parallel::distributed::Vector<Number> &inverse_diagonal_entries) const
{
  data->initialize_dof_vector(inverse_diagonal_entries, operator_data.laplace_dof_index);
  unsigned int dummy;

  data->loop (&LaplaceOperator<dim, degree, Number>::local_diagonal_cell,
              &LaplaceOperator<dim, degree, Number>::local_diagonal_face,
              &LaplaceOperator<dim, degree, Number>::local_diagonal_boundary,
              this, inverse_diagonal_entries, dummy);

  if(apply_mean_value_constraint_in_matvec)
  {
    parallel::distributed::Vector<Number> vec1;
    vec1.reinit(inverse_diagonal_entries, true);
    for(unsigned int i=0;i<vec1.local_size();++i)
      vec1.local_element(i) = 1.;
    parallel::distributed::Vector<Number> d;
    d.reinit(inverse_diagonal_entries, true);
    vmult(d,vec1);
    double length = vec1*vec1;
    double factor = vec1*d;
    inverse_diagonal_entries.add(-2./length,d,factor/pow(length,2.),vec1);
  }

  const std::vector<unsigned int> &
    constrained_dofs = data->get_constrained_dofs();
  for (unsigned int i=0; i<constrained_dofs.size(); ++i)
    inverse_diagonal_entries.local_element(constrained_dofs[i]) = 1.;
  for (unsigned int i=0; i<edge_constrained_indices.size(); ++i)
  {
    inverse_diagonal_entries.local_element(edge_constrained_indices[i]) = 1.;
  }

  for (unsigned int i=0; i<inverse_diagonal_entries.local_size(); ++i)
  {
    if (std::abs(inverse_diagonal_entries.local_element(i)) > 1e-10)
      inverse_diagonal_entries.local_element(i) = 1./inverse_diagonal_entries.local_element(i);
    else
      inverse_diagonal_entries.local_element(i) = 1.;
  }
}

template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
apply_block_jacobi (parallel::distributed::Vector<Number>       &dst,
                    parallel::distributed::Vector<Number> const &src) const
{
  // apply_inverse_matrices
  data->cell_loop(&LaplaceOperator<dim,degree,Number>::cell_loop_apply_inverse_block_jacobi_matrices, this, dst, src);
}

template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
cell_loop_apply_inverse_block_jacobi_matrices (const MatrixFree<dim,Number>                 &data,
                                               parallel::distributed::Vector<Number>        &dst,
                                               const parallel::distributed::Vector<Number>  &src,
                                               const std::pair<unsigned int,unsigned int>   &cell_range) const
{
  // apply inverse block matrices
  FEEvaluation<dim,degree,degree+1,1,Number> fe_eval(data,
                                                     operator_data.laplace_dof_index,
                                                     operator_data.laplace_quad_index);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    fe_eval.reinit(cell);
    fe_eval.read_dof_values(src);

    for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
    {
      // fill source vector
      Vector<Number> src_vector(fe_eval.dofs_per_cell);
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
        src_vector(j) = fe_eval.begin_dof_values()[j][v];

      // apply inverse matrix
      matrices[cell*VectorizedArray<Number>::n_array_elements+v].apply_lu_factorization(src_vector,false);

      // write solution to dst-vector
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j][v] = src_vector(j);
    }

    fe_eval.set_dof_values (dst);
  }
}

template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
update_block_jacobi () const
{
  if(block_jacobi_matrices_have_been_initialized == false)
  {
    matrices.resize(data->n_macro_cells()*VectorizedArray<Number>::n_array_elements,
      LAPACKFullMatrix<Number>(data->get_shape_info().dofs_per_cell, data->get_shape_info().dofs_per_cell));

    block_jacobi_matrices_have_been_initialized = true;
  }

  initialize_block_jacobi_matrices_with_zero(matrices);

  add_block_jacobi_matrices(matrices);

  calculate_lu_factorization_block_jacobi(matrices);
}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
local_apply (const MatrixFree<dim,Number>                &data,
             parallel::distributed::Vector<Number>       &dst,
             const parallel::distributed::Vector<Number> &src,
             const std::pair<unsigned int,unsigned int>  &cell_range) const
{
  FEEvaluation<dim,degree,degree+1,1,Number> phi (data,
                                                  operator_data.laplace_dof_index,
                                                  operator_data.laplace_quad_index);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    phi.reinit (cell);
    phi.read_dof_values(src);

    do_cell_integral(phi);

    phi.distribute_local_to_global (dst);
  }
}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
local_apply_face (const MatrixFree<dim,Number>                &data,
                  parallel::distributed::Vector<Number>       &dst,
                  const parallel::distributed::Vector<Number> &src,
                  const std::pair<unsigned int,unsigned int>  &face_range) const
{
  // Nothing to do for continuous elements
  if (data.get_dof_handler(operator_data.laplace_dof_index).get_fe().dofs_per_vertex > 0)
    return;

  FEFaceEvaluation<dim,degree,degree+1,1,Number> fe_eval(data,true,
                                                         operator_data.laplace_dof_index,
                                                         operator_data.laplace_quad_index);
  FEFaceEvaluation<dim,degree,degree+1,1,Number> fe_eval_neighbor(data,false,
                                                                  operator_data.laplace_dof_index,
                                                                  operator_data.laplace_quad_index);

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval.reinit (face);
    fe_eval_neighbor.reinit (face);

    fe_eval.read_dof_values(src);
    fe_eval.evaluate(true,true);
    fe_eval_neighbor.read_dof_values(src);
    fe_eval_neighbor.evaluate(true,true);

    VectorizedArray<Number> sigmaF =
      std::max(fe_eval.read_cell_data(array_penalty_parameter),
               fe_eval_neighbor.read_cell_data(array_penalty_parameter)) *
      get_penalty_factor();

    for(unsigned int q=0;q<fe_eval.n_q_points;++q)
    {
      VectorizedArray<Number> valueM = fe_eval.get_value(q);
      VectorizedArray<Number> valueP = fe_eval_neighbor.get_value(q);

      VectorizedArray<Number> jump_value = valueM - valueP;
      VectorizedArray<Number> average_gradient =
        ( fe_eval.get_normal_gradient(q) + fe_eval_neighbor.get_normal_gradient(q) ) * 0.5;
      average_gradient = average_gradient - jump_value * sigmaF;

      fe_eval.submit_normal_gradient(-0.5*jump_value,q);
      fe_eval_neighbor.submit_normal_gradient(-0.5*jump_value,q);
      fe_eval.submit_value(-average_gradient,q);
      fe_eval_neighbor.submit_value(average_gradient,q);
    }
    fe_eval.integrate(true,true);
    fe_eval.distribute_local_to_global(dst);
    fe_eval_neighbor.integrate(true,true);
    fe_eval_neighbor.distribute_local_to_global(dst);
  }
}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
local_apply_boundary (const MatrixFree<dim,Number>                &data,
                      parallel::distributed::Vector<Number>       &dst,
                      const parallel::distributed::Vector<Number> &src,
                      const std::pair<unsigned int,unsigned int>  &face_range) const
{
  // Nothing to do for continuous elements
  if (data.get_dof_handler(operator_data.laplace_dof_index).get_fe().dofs_per_vertex > 0)
    return;

  FEFaceEvaluation<dim,degree,degree+1,1,Number> fe_eval(data, true,
                                                         operator_data.laplace_dof_index,
                                                         operator_data.laplace_quad_index);
  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval.reinit (face);
    fe_eval.read_dof_values(src);
    fe_eval.evaluate(true,true);
    VectorizedArray<Number> sigmaF =
      fe_eval.read_cell_data(array_penalty_parameter) *
      get_penalty_factor();

    typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
    types::boundary_id boundary_id = data.get_boundary_indicator(face);

    for(unsigned int q=0;q<fe_eval.n_q_points;++q)
    {
      it = operator_data.bc->dirichlet.find(boundary_id);
      if(it != operator_data.bc->dirichlet.end())
      {
        //set value to zero, i.e. u+ = - u- , grad+ = grad-
        VectorizedArray<Number> valueM = fe_eval.get_value(q);

        VectorizedArray<Number> jump_value = 2.0*valueM;
        VectorizedArray<Number> average_gradient = fe_eval.get_normal_gradient(q);
        average_gradient = average_gradient - jump_value * sigmaF;

        fe_eval.submit_normal_gradient(-0.5*jump_value,q);
        fe_eval.submit_value(-average_gradient,q);
      }
      it = operator_data.bc->neumann.find(boundary_id);
      if (it != operator_data.bc->neumann.end())
      {
        //set gradient in normal direction to zero, i.e. u+ = u-, grad+ = -grad-
        VectorizedArray<Number> jump_value = make_vectorized_array<Number>(0.0);
        VectorizedArray<Number> average_gradient = make_vectorized_array<Number>(0.0);
        average_gradient = average_gradient - jump_value * sigmaF;

        fe_eval.submit_normal_gradient(-0.5*jump_value,q);
        fe_eval.submit_value(-average_gradient,q);
      }
    }

    fe_eval.integrate(true,true);
    fe_eval.distribute_local_to_global(dst);
  }
}


template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
local_rhs (const MatrixFree<dim,Number>                &,
           parallel::distributed::Vector<Number>       &,
           const parallel::distributed::Vector<Number> &,
           const std::pair<unsigned int,unsigned int>  &) const
{}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
local_rhs_face (const MatrixFree<dim,Number>                &,
                parallel::distributed::Vector<Number>       &,
                const parallel::distributed::Vector<Number> &,
                const std::pair<unsigned int,unsigned int>  &) const
{}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
local_rhs_boundary (const MatrixFree<dim,Number>                &data,
                    parallel::distributed::Vector<Number>       &dst,
                    const parallel::distributed::Vector<Number> &,
                    const std::pair<unsigned int,unsigned int>  &face_range) const
{
  // Nothing to do for continuous elements
  if (data.get_dof_handler(operator_data.laplace_dof_index).get_fe().dofs_per_vertex > 0)
    return;

  FEFaceEvaluation<dim,degree,degree+1,1,Number> fe_eval(data, true,
                                                         operator_data.laplace_dof_index,
                                                         operator_data.laplace_quad_index);

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval.reinit (face);

    VectorizedArray<Number> sigmaF =
      fe_eval.read_cell_data(array_penalty_parameter) *
      get_penalty_factor();

    typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
    types::boundary_id boundary_id = data.get_boundary_indicator(face);

    for(unsigned int q=0;q<fe_eval.n_q_points;++q)
    {
      it = operator_data.bc->dirichlet.find(boundary_id);
      if(it != operator_data.bc->dirichlet.end())
      {
        // u+ = 2g , grad+ = 0 (inhomogeneous parts)
        VectorizedArray<Number> g = make_vectorized_array<Number>(0.0);

        Point<dim,VectorizedArray<Number> > q_points = fe_eval.quadrature_point(q);
        Number array [VectorizedArray<Number>::n_array_elements];
        for (unsigned int n=0; n<VectorizedArray<Number>::n_array_elements; ++n)
        {
          Point<dim> q_point;
          for (unsigned int d=0; d<dim; ++d)
            q_point[d] = q_points[d][n];
          array[n] = it->second->value(q_point);
        }
        g.load(&array[0]);

        VectorizedArray<Number> jump_value = - 2.0*g;
        VectorizedArray<Number> average_gradient = make_vectorized_array<Number>(0.0);
        average_gradient = average_gradient - jump_value * sigmaF;

        fe_eval.submit_normal_gradient(-0.5*(-jump_value),q); // (-jump_value) since this term is shifted to the rhs of the equations
        fe_eval.submit_value(average_gradient,q); // (+average_gradient) since this term is shifted to the rhs of the equations
      }

      it = operator_data.bc->neumann.find(boundary_id);
      if (it != operator_data.bc->neumann.end())
      {
        // u+ = 0, grad+ = 2h (inhomogeneous parts)
        VectorizedArray<Number> jump_value = make_vectorized_array<Number>(0.0);
        VectorizedArray<Number> average_gradient = make_vectorized_array<Number>(0.0);

        Point<dim,VectorizedArray<Number> > q_points = fe_eval.quadrature_point(q);
        Number array [VectorizedArray<Number>::n_array_elements];
        for (unsigned int n=0; n<VectorizedArray<Number>::n_array_elements; ++n)
        {
          Point<dim> q_point;
          for (unsigned int d=0; d<dim; ++d)
            q_point[d] = q_points[d][n];
          array[n] = it->second->value(q_point);
        }
        average_gradient.load(&array[0]);

        average_gradient = average_gradient - jump_value * sigmaF;

        fe_eval.submit_normal_gradient(-0.5*(-jump_value),q); // (-jump_value) since this term is shifted to the rhs of the equations
        fe_eval.submit_value(average_gradient,q); // (+average_gradient) since this term is shifted to the rhs of the equations
      }
    }

    fe_eval.integrate(true,true);
    fe_eval.distribute_local_to_global(dst);
  }
}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
local_diagonal_cell (const MatrixFree<dim,Number>                &data,
                     parallel::distributed::Vector<Number>       &dst,
                     const unsigned int                          &,
                     const std::pair<unsigned int,unsigned int>  &cell_range) const
{
  FEEvaluation<dim,degree,degree+1,1,Number> phi (data,
                                                  operator_data.laplace_dof_index,
                                                  operator_data.laplace_quad_index);

  VectorizedArray<Number> local_diagonal_vector[phi.tensor_dofs_per_cell];
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    phi.reinit (cell);

    for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
    {
      for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
        phi.begin_dof_values()[j] = VectorizedArray<Number>();
      phi.begin_dof_values()[i] = 1.;

      do_cell_integral(phi);

      local_diagonal_vector[i] = phi.begin_dof_values()[i];
    }
    for (unsigned int i=0; i<phi.tensor_dofs_per_cell; ++i)
      phi.begin_dof_values()[i] = local_diagonal_vector[i];
    phi.distribute_local_to_global (dst);
  }
}

//template <int dim, typename Number>
//template <int degree>
//void LaplaceOperator<dim,Number>::
//local_diagonal_cell (const MatrixFree<dim,Number>                &data,
//                     parallel::distributed::Vector<Number>       &dst,
//                     const unsigned int                          &,
//                     const std::pair<unsigned int,unsigned int>  &cell_range) const
//{
//  FEEvaluation<dim,degree,degree+1,1,Number> phi (data,
//                                                  operator_data.laplace_dof_index,
//                                                  operator_data.laplace_quad_index);
//
//  std::vector<FullMatrix<Number>> matrices(data.n_macro_cells()*VectorizedArray<Number>::n_array_elements,
//      FullMatrix<Number>(phi.dofs_per_cell, phi.dofs_per_cell));
//  VectorizedArray<Number> local_diagonal_vector[phi.tensor_dofs_per_cell];
//  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
//    {
//      phi.reinit (cell);
//
//      for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
//        {
//          for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
//            phi.begin_dof_values()[j] = VectorizedArray<Number>();
//          phi.begin_dof_values()[i] = 1.;
//          phi.evaluate (false,true,false);
//          for (unsigned int q=0; q<phi.n_q_points; ++q)
//            phi.submit_gradient (phi.get_gradient(q), q);
//          phi.integrate (false,true);
//          for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
//            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
//              matrices[cell*VectorizedArray<Number>::n_array_elements+v](j,i) = phi.begin_dof_values()[j][v];
//          local_diagonal_vector[i] = phi.begin_dof_values()[i];
//        }
//      for (unsigned int i=0; i<phi.tensor_dofs_per_cell; ++i)
//        phi.begin_dof_values()[i] = local_diagonal_vector[i];
//      phi.distribute_local_to_global (dst);
//    }
//}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
local_diagonal_face (const MatrixFree<dim,Number>                &data,
                     parallel::distributed::Vector<Number>       &dst,
                     const unsigned int                          &,
                     const std::pair<unsigned int,unsigned int>  &face_range) const
{
  // Nothing to do for continuous elements
  if (data.get_dof_handler(operator_data.laplace_dof_index).get_fe().dofs_per_vertex > 0)
    return;

  FEFaceEvaluation<dim,degree,degree+1,1,Number> phi(data,true,
                                                     operator_data.laplace_dof_index,
                                                     operator_data.laplace_quad_index);
  FEFaceEvaluation<dim,degree,degree+1,1,Number> phi_outer(data,false,
                                                           operator_data.laplace_dof_index,
                                                           operator_data.laplace_quad_index);

  VectorizedArray<Number> local_diagonal_vector[phi.tensor_dofs_per_cell];
  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    phi.reinit (face);
    phi_outer.reinit (face);

    VectorizedArray<Number> sigmaF =
      std::max(phi.read_cell_data(array_penalty_parameter),
               phi_outer.read_cell_data(array_penalty_parameter)) *
      get_penalty_factor();

    // element-
    for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
      phi_outer.begin_dof_values()[j] = VectorizedArray<Number>();
    phi_outer.evaluate(true, true);
    for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
    {
      for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
        phi.begin_dof_values()[j] = VectorizedArray<Number>();
      phi.begin_dof_values()[i] = 1.;
      phi.evaluate(true,true);

      for(unsigned int q=0;q<phi.n_q_points;++q)
      {
        VectorizedArray<Number> valueM = phi.get_value(q);
        VectorizedArray<Number> valueP = phi_outer.get_value(q);

        VectorizedArray<Number> jump_value = valueM - valueP;
        VectorizedArray<Number> average_gradient =
          ( phi.get_normal_gradient(q) + phi_outer.get_normal_gradient(q) ) * 0.5;
        average_gradient = average_gradient - jump_value * sigmaF;

        phi.submit_normal_gradient(-0.5*jump_value,q);
        phi.submit_value(-average_gradient,q);
      }
      phi.integrate(true,true);
      local_diagonal_vector[i] = phi.begin_dof_values()[i];
    }
    for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
      phi.begin_dof_values()[i] = local_diagonal_vector[i];
    phi.distribute_local_to_global(dst);

    // neighbor (element+)
    for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
      phi.begin_dof_values()[j] = VectorizedArray<Number>();
    phi.evaluate(true, true);
    for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
    {
      for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
        phi_outer.begin_dof_values()[j] = VectorizedArray<Number>();
      phi_outer.begin_dof_values()[i] = 1.;
      phi_outer.evaluate(true,true);

      for(unsigned int q=0;q<phi.n_q_points;++q)
      {
        VectorizedArray<Number> valueM = phi.get_value(q);
        VectorizedArray<Number> valueP = phi_outer.get_value(q);

        VectorizedArray<Number> jump_value = valueM - valueP;
        VectorizedArray<Number> average_gradient =
          ( phi.get_normal_gradient(q) + phi_outer.get_normal_gradient(q) ) * 0.5;
        average_gradient = average_gradient - jump_value * sigmaF;

        phi_outer.submit_normal_gradient(-0.5*jump_value,q);
        phi_outer.submit_value(average_gradient,q);
      }
      phi_outer.integrate(true,true);
      local_diagonal_vector[i] = phi_outer.begin_dof_values()[i];
    }
    for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
      phi_outer.begin_dof_values()[i] = local_diagonal_vector[i];
    phi_outer.distribute_local_to_global(dst);
  }
}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
local_diagonal_boundary (const MatrixFree<dim,Number>                &data,
                         parallel::distributed::Vector<Number>       &dst,
                         const unsigned int                          &,
                         const std::pair<unsigned int,unsigned int>  &face_range) const
{
  // Nothing to do for continuous elements
  if (data.get_dof_handler(operator_data.laplace_dof_index).get_fe().dofs_per_vertex > 0)
    return;

  FEFaceEvaluation<dim,degree,degree+1,1,Number> phi (data, true,
                                                      operator_data.laplace_dof_index,
                                                      operator_data.laplace_quad_index);

  VectorizedArray<Number> local_diagonal_vector[phi.tensor_dofs_per_cell];
  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    phi.reinit (face);

    VectorizedArray<Number> sigmaF =
      phi.read_cell_data(array_penalty_parameter) *
      get_penalty_factor();

    typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
    types::boundary_id boundary_id = data.get_boundary_indicator(face);

    for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
    {
      for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
        phi.begin_dof_values()[j] = VectorizedArray<Number>();
      phi.begin_dof_values()[i] = 1.;
      phi.evaluate(true,true);

      for(unsigned int q=0;q<phi.n_q_points;++q)
      {
        it = operator_data.bc->dirichlet.find(boundary_id);
        if(it != operator_data.bc->dirichlet.end())
        {
          //set value to zero, i.e. u+ = - u- , grad+ = grad-
          VectorizedArray<Number> valueM = phi.get_value(q);

          VectorizedArray<Number> jump_value = 2.0*valueM;
          VectorizedArray<Number> average_gradient = phi.get_normal_gradient(q);
          average_gradient = average_gradient - jump_value * sigmaF;

          phi.submit_normal_gradient(-0.5*jump_value,q);
          phi.submit_value(-average_gradient,q);
        }

        it = operator_data.bc->neumann.find(boundary_id);
        if (it != operator_data.bc->neumann.end())
        {
          //set solution gradient in normal direction to zero, i.e. u+ = u-, grad+ = -grad-
          VectorizedArray<Number> jump_value = make_vectorized_array<Number>(0.0);
          VectorizedArray<Number> average_gradient = make_vectorized_array<Number>(0.0);
          average_gradient = average_gradient - jump_value * sigmaF;

          phi.submit_normal_gradient(-0.5*jump_value,q);
          phi.submit_value(-average_gradient,q);
        }
      }

      phi.integrate(true,true);
      local_diagonal_vector[i] = phi.begin_dof_values()[i];
    }
    for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
      phi.begin_dof_values()[i] = local_diagonal_vector[i];
    phi.distribute_local_to_global(dst);
  }
}


template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
add_block_jacobi_matrices(std::vector<LAPACKFullMatrix<Number> > &matrices) const
{
  parallel::distributed::Vector<Number>  src;

  data->loop(&LaplaceOperator<dim,degree,Number>::cell_loop_calculate_block_jacobi_matrices,
             &LaplaceOperator<dim,degree,Number>::face_loop_calculate_block_jacobi_matrices,
             &LaplaceOperator<dim,degree,Number>::boundary_face_loop_calculate_block_jacobi_matrices, this, matrices, src);
}

template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
cell_loop_calculate_block_jacobi_matrices (const MatrixFree<dim,Number>                 &data,
                                           std::vector<LAPACKFullMatrix<Number> >       &matrices,
                                           const parallel::distributed::Vector<Number>  &,
                                           const std::pair<unsigned int,unsigned int>   &cell_range) const
{
  FEEvaluation<dim,degree,degree+1,1,Number> phi (data,
                                                  operator_data.laplace_dof_index,
                                                  operator_data.laplace_quad_index);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    phi.reinit(cell);

    for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
    {
      for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
        phi.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
      phi.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

      do_cell_integral(phi);

      for(unsigned int i=0; i<phi.dofs_per_cell; ++i)
        for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
          matrices[cell*VectorizedArray<Number>::n_array_elements+v](i,j) += phi.begin_dof_values()[i][v];
    }
  }
}

template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
face_loop_calculate_block_jacobi_matrices (const MatrixFree<dim,Number>                &data,
                                           std::vector<LAPACKFullMatrix<Number> >      &matrices,
                                           const parallel::distributed::Vector<Number> &,
                                           const std::pair<unsigned int,unsigned int>  &face_range) const
{
  FEFaceEvaluation<dim,degree,degree+1,1,Number> phi(data,true,
                                                     operator_data.laplace_dof_index,
                                                     operator_data.laplace_quad_index);
  FEFaceEvaluation<dim,degree,degree+1,1,Number> phi_outer(data,false,
                                                           operator_data.laplace_dof_index,
                                                           operator_data.laplace_quad_index);

  // Perform face intergrals for element e⁻.
  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    phi.reinit (face);
    phi_outer.reinit (face);

    VectorizedArray<Number> sigmaF =
      std::max(phi.read_cell_data(array_penalty_parameter),
               phi_outer.read_cell_data(array_penalty_parameter)) *
      get_penalty_factor();

    for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
    {
      // set dof value j of element- to 1 and all other dof values of element- to zero
      for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
        phi.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
      phi.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

      phi.evaluate(true,true);

      for(unsigned int q=0;q<phi.n_q_points;++q)
      {
        VectorizedArray<Number> jump_value = phi.get_value(q);
        VectorizedArray<Number> average_gradient = 0.5 * phi.get_normal_gradient(q);
        average_gradient = average_gradient - jump_value * sigmaF;

        phi.submit_normal_gradient(-0.5*jump_value,q);
        phi.submit_value(-average_gradient,q);
      }

      phi.integrate(true,true);

      for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
      {
        const unsigned int cell_number = data.faces[face].left_cell[v];
        if (cell_number != numbers::invalid_unsigned_int)
          for(unsigned int i=0; i<phi.dofs_per_cell; ++i)
            matrices[cell_number](i,j) += phi.begin_dof_values()[i][v];
      }
    }
  }



  // TODO: This has to be removed as soon as the new infrastructure is used that
  // allows to perform face integrals over all faces of the current element.
  // Perform face intergrals for element e⁺.
  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    phi.reinit (face);
    phi_outer.reinit (face);

    VectorizedArray<Number> sigmaF =
      std::max(phi.read_cell_data(array_penalty_parameter),
               phi_outer.read_cell_data(array_penalty_parameter)) *
      get_penalty_factor();

    for (unsigned int j=0; j<phi_outer.dofs_per_cell; ++j)
    {
      // set dof value j of element+ to 1 and all other dof values of element+ to zero
      for (unsigned int i=0; i<phi_outer.dofs_per_cell; ++i)
        phi_outer.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
      phi_outer.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

      phi_outer.evaluate(true,true);

      for(unsigned int q=0;q<phi.n_q_points;++q)
      {
        VectorizedArray<Number> jump_value = phi_outer.get_value(q);
        // minus sign to get the correct normal vector n⁺ = -n⁻
        VectorizedArray<Number> average_gradient = -0.5 * phi_outer.get_normal_gradient(q);
        average_gradient = average_gradient - jump_value * sigmaF;

        // plus sign since n⁺ = -n⁻
        phi_outer.submit_normal_gradient(0.5*jump_value,q);
        phi_outer.submit_value(-average_gradient,q);
      }
      phi_outer.integrate(true,true);

      for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
      {
        const unsigned int cell_number = data.faces[face].right_cell[v];
        if (cell_number != numbers::invalid_unsigned_int)
          for(unsigned int i=0; i<phi_outer.dofs_per_cell; ++i)
            matrices[cell_number](i,j) += phi_outer.begin_dof_values()[i][v];
      }
    }
  }
}

// TODO: This function has to be removed as soon as the new infrastructure is used that
// allows to perform face integrals over all faces of the current element.
template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
boundary_face_loop_calculate_block_jacobi_matrices (const MatrixFree<dim,Number>                &data,
                                                    std::vector<LAPACKFullMatrix<Number> >      &matrices,
                                                    const parallel::distributed::Vector<Number> &,
                                                    const std::pair<unsigned int,unsigned int>  &face_range) const
{
  FEFaceEvaluation<dim,degree,degree+1,1,Number> phi (data, true,
                                                      operator_data.laplace_dof_index,
                                                      operator_data.laplace_quad_index);

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    phi.reinit (face);

    VectorizedArray<Number> sigmaF =
      phi.read_cell_data(array_penalty_parameter) *
      get_penalty_factor();

    typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
    types::boundary_id boundary_id = data.get_boundary_indicator(face);

    for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
    {
      // set dof value j of element- to 1 and all other dof values of element- to zero
      for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
        phi.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
      phi.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

      phi.evaluate(true,true);

      for(unsigned int q=0;q<phi.n_q_points;++q)
      {
        it = operator_data.bc->dirichlet.find(boundary_id);
        if(it != operator_data.bc->dirichlet.end())
        {
          //set value to zero, i.e. u+ = - u- , grad+ = grad-
          VectorizedArray<Number> valueM = phi.get_value(q);

          VectorizedArray<Number> jump_value = 2.0*valueM;
          VectorizedArray<Number> average_gradient = phi.get_normal_gradient(q);
          average_gradient = average_gradient - jump_value * sigmaF;

          phi.submit_normal_gradient(-0.5*jump_value,q);
          phi.submit_value(-average_gradient,q);
        }

        it = operator_data.bc->neumann.find(boundary_id);
        if (it != operator_data.bc->neumann.end())
        {
          //set solution gradient in normal direction to zero, i.e. u+ = u-, grad+ = -grad-
          VectorizedArray<Number> jump_value = make_vectorized_array<Number>(0.0);
          VectorizedArray<Number> average_gradient = make_vectorized_array<Number>(0.0);
          average_gradient = average_gradient - jump_value * sigmaF;

          phi.submit_normal_gradient(-0.5*jump_value,q);
          phi.submit_value(-average_gradient,q);
        }
      }

      phi.integrate(true,true);

      for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
      {
        const unsigned int cell_number = data.faces[face].left_cell[v];
        if (cell_number != numbers::invalid_unsigned_int)
          for(unsigned int i=0; i<phi.dofs_per_cell; ++i)
            matrices[cell_number](i,j) += phi.begin_dof_values()[i][v];
      }
    }
  }
}

#endif // ifndef __indexa_poisson_solver_h
