#ifndef MULTIGRID_OPERATOR_BASE
#define MULTIGRID_OPERATOR_BASE

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>

#include "operator_preconditionable.h"

using namespace dealii;

template<int dim, typename Number = double>
class PreconditionableOperatorDummy : public PreconditionableOperator<dim, Number>
{
public:
  typedef Number value_type;

  static const int DIM = dim;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;



  virtual void
  apply(VectorType & dst, VectorType const & src) const
  {
    (void)dst;
    (void)src;

    AssertThrow(false, ExcMessage("MultigridOperatorBase::apply should be overwritten!"));
  }

  virtual void
  apply_add(VectorType & dst, VectorType const & src, Number const time) const
  {
    (void)dst;
    (void)src;
    (void)time;
    AssertThrow(false, ExcMessage("MultigridOperatorBase::apply_add should be overwritten!"));
  }

  virtual void
  apply_add(VectorType & dst, VectorType const & src) const
  {
    (void)dst;
    (void)src;
    AssertThrow(false, ExcMessage("MultigridOperatorBase::apply_add should be overwritten!"));
  }

  virtual void
  rhs(VectorType & dst) const
  {
    (void)dst;
    AssertThrow(false, ExcMessage("MultigridOperatorBase::rhs should be overwritten!"));
  }

  virtual void
  rhs(VectorType & dst, Number const time) const
  {
    (void)dst;
    (void)time;
    AssertThrow(false, ExcMessage("MultigridOperatorBase::rhs should be overwritten!"));
  }

  virtual void
  rhs_add(VectorType & dst) const
  {
    (void)dst;
    AssertThrow(false, ExcMessage("MultigridOperatorBase::rhs_add should be overwritten!"));
  }

  virtual void
  rhs_add(VectorType & dst, Number const time) const
  {
    (void)dst;
    (void)time;
    AssertThrow(false, ExcMessage("MultigridOperatorBase::rhs_add should be overwritten!"));
  }

  virtual void
  evaluate(VectorType & dst, VectorType const & src, Number const time) const
  {
    (void)dst;
    (void)src;
    (void)time;
    AssertThrow(false, ExcMessage("MultigridOperatorBase::evaluate should be overwritten!"));
  }

  virtual void
  evaluate_add(VectorType & dst, VectorType const & src, Number const time) const
  {
    (void)dst;
    (void)src;
    (void)time;
    AssertThrow(false, ExcMessage("MultigridOperatorBase::evaluate_add should be overwritten!"));
  }


  void
  reinit_preconditionable_operator_data(
    MatrixFree<dim, Number> const & /*matrix_free*/,
    AffineConstraints<double> const & /*constraint_matrix*/,
    PreconditionableOperatorData<dim> const & /*operator_data_in*/) const
  {
    AssertThrow(
      false,
      ExcMessage(
        "MultigridOperatorBase::reinit_preconditionable_operator_data should be overwritten!"));
  }

  virtual void
  vmult(VectorType & /*dst*/, VectorType const & /*src*/) const
  {
    AssertThrow(false, ExcMessage("MultigridOperatorBase::vmult should be overwritten!"));
  }

  virtual void
  vmult_add(VectorType & /*dst*/, VectorType const & /*src*/) const
  {
    AssertThrow(false, ExcMessage("MultigridOperatorBase::vmult_add should be overwritten!"));
  }

  void
  vmult_interface_down(VectorType & dst, VectorType const & src) const
  {
    vmult(dst, src);
  }

  void
  vmult_add_interface_up(VectorType & dst, VectorType const & src) const
  {
    vmult_add(dst, src);
  }

  types::global_dof_index
  m() const
  {
    return n();
  }

  types::global_dof_index
  n() const
  {
    MatrixFree<dim, Number> const & data      = get_data();
    unsigned int                    dof_index = get_dof_index();

    return data.get_vector_partitioner(dof_index)->size();
  }

  Number
  el(const unsigned int, const unsigned int) const
  {
    AssertThrow(false, ExcMessage("Matrix-free does not allow for entry access"));
    return Number();
  }

  bool
  is_empty_locally() const
  {
    MatrixFree<dim, Number> const & data = get_data();
    return (data.n_macro_cells() == 0);
  }

  void
  initialize_dof_vector(VectorType & vector) const
  {
    MatrixFree<dim, Number> const & data      = get_data();
    unsigned int                    dof_index = get_dof_index();

    data.initialize_dof_vector(vector, dof_index);
  }

  virtual void
  calculate_inverse_diagonal(VectorType & /*inverse_diagonal_entries*/) const
  {
    AssertThrow(false,
                ExcMessage(
                  "MultigridOperatorBase::calculate_inverse_diagonal should be overwritten!"));
  }

  virtual void
  apply_inverse_block_diagonal(VectorType & /*dst*/, VectorType const & /*src*/) const
  {
    AssertThrow(false,
                ExcMessage(
                  "MultigridOperatorBase::apply_inverse_block_diagonal should be overwritten!"));
  }

  virtual void
  update_block_diagonal_preconditioner() const
  {
    AssertThrow(
      false,
      ExcMessage(
        "MultigridOperatorBase::update_block_diagonal_preconditioner should be overwritten!"));
  }

  virtual AffineConstraints<double> const &
  get_constraint_matrix() const
  {
    AssertThrow(false,
                ExcMessage("MultigridOperatorBase::get_constraint_matrix should be overwritten!"));
    return *(new AffineConstraints<double>());
  }

  virtual const MatrixFree<dim, Number> &
  get_data() const
  {
    AssertThrow(false, ExcMessage("MultigridOperatorBase::get_data should be overwritten!"));
    return *(new MatrixFree<dim, Number>);
  }

  virtual unsigned int
  get_dof_index() const
  {
    AssertThrow(false, ExcMessage("MultigridOperatorBase::get_dof_index should be overwritten!"));
    return 0;
  }

  virtual PreconditionableOperator<dim, Number> *
  get_new(unsigned int /*deg*/) const
  {
    AssertThrow(false, ExcMessage("MultigridOperatorBase::get_new should be overwritten!"));
    return nullptr;
  }

  virtual bool
  is_singular() const
  {
    // per default the operator is not singular
    // if an operator can be singular, this method has to be overwritten
    return false;
  }

#ifdef DEAL_II_WITH_TRILINOS
  virtual void
  init_system_matrix(TrilinosWrappers::SparseMatrix & /*system_matrix*/) const
  {
    AssertThrow(false,
                ExcMessage("MultigridOperatorBase::init_system_matrix should be overwritten!"));
  }

  virtual void
  calculate_system_matrix(TrilinosWrappers::SparseMatrix & /*system_matrix*/) const
  {
    AssertThrow(
      false, ExcMessage("MultigridOperatorBase::calculate_system_matrix should be overwritten!"));
  }
#endif
};

#endif
