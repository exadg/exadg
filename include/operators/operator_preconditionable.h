#ifndef OPERATOR_PRECONDITIONABLE_H
#define OPERATOR_PRECONDITIONABLE_H

#include <deal.II/matrix_free/matrix_free.h>

#include "operator.h"

template<int dim, typename Number>
class PreconditionableOperator : virtual public LinearOperatorBaseNew<Number>
{
public:
  typedef LinearOperatorBaseNew<Number> Parent;

  typedef typename Parent::value_type value_type;
  static const int                    DIM = dim;
  typedef typename Parent::VectorType VectorType;


  virtual void
  reinit_void(MatrixFree<dim, Number> const &   matrix_free,
              AffineConstraints<double> const & constraint_matrix,
              void *                            operator_data_in) const = 0;

  virtual void
  reinit_multigrid(
    const DoFHandler<dim> &   dof_handler,
    const Mapping<dim> &      mapping,
    void *                    operator_data,
    const MGConstrainedDoFs & mg_constrained_dofs,
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                       periodic_face_pairs,
    const unsigned int level = numbers::invalid_unsigned_int) = 0;

  virtual void
  reinit_multigrid_add_dof_handler(
    const DoFHandler<dim> &   dof_handler,
    const Mapping<dim> &      mapping,
    void *                    operator_data,
    const MGConstrainedDoFs & mg_constrained_dofs,
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                            periodic_face_pairs,
    const unsigned int      level,
    const DoFHandler<dim> * additional_dof_handler) = 0;

  virtual void
  vmult(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  vmult_add(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  vmult_interface_down(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  vmult_add_interface_up(VectorType & dst, VectorType const & src) const = 0;


  virtual void
  calculate_inverse_diagonal(VectorType & inverse_diagonal_entries) const = 0;

  virtual void
  apply_inverse_block_diagonal(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  update_block_diagonal_preconditioner() const = 0;

  virtual AffineConstraints<double> const &
  get_constraint_matrix() const = 0;

  virtual const MatrixFree<dim, Number> &
  get_data() const = 0;

  virtual unsigned int
  get_dof_index() const = 0;

  virtual PreconditionableOperator<dim, Number> *
  get_new(unsigned int deg) const = 0;

  virtual bool
  is_singular() const = 0;

#ifdef DEAL_II_WITH_TRILINOS
  virtual void
  init_system_matrix(TrilinosWrappers::SparseMatrix & system_matrix) const = 0;

  virtual void
  calculate_system_matrix(TrilinosWrappers::SparseMatrix & system_matrix) const = 0;
#endif
};

#endif