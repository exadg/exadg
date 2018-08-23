#ifndef MULTIGRID_OPERATOR_BASE
#define MULTIGRID_OPERATOR_BASE

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>

#include "matrix_operator_base.h"

using namespace dealii;

template<int dim, typename Number = double>
class MultigridOperatorBase : public MatrixOperatorBase
{
public:
  typedef Number   value_type;
  static const int DIM = dim;

  virtual void
  clear()
  {
    AssertThrow(false, ExcMessage("MultigridOperatorBase::clear should be overwritten!"));
  }

  virtual void
  vmult(parallel::distributed::Vector<Number> & /*dst*/,
        const parallel::distributed::Vector<Number> & /*src*/) const
  {
    AssertThrow(false, ExcMessage("MultigridOperatorBase::vmult should be overwritten!"));
  }

  virtual void
  vmult_add(parallel::distributed::Vector<Number> & /*dst*/,
            const parallel::distributed::Vector<Number> & /*src*/) const
  {
    AssertThrow(false, ExcMessage("MultigridOperatorBase::vmult_add should be overwritten!"));
  }

  virtual void
  vmult_interface_down(parallel::distributed::Vector<Number> & /*dst*/,
                       const parallel::distributed::Vector<Number> & /*src*/) const
  {
    AssertThrow(false, ExcMessage("MultigridOperatorBase::vmult_interface_down should be overwritten!"));
  }

  virtual void
  vmult_add_interface_up(parallel::distributed::Vector<Number> & /*dst*/,
                         const parallel::distributed::Vector<Number> & /*src*/) const
  {
    AssertThrow(false, ExcMessage("MultigridOperatorBase::vmult_add_interface_up should be overwritten!"));
  }
  
  virtual types::global_dof_index
  m() const
  {
    AssertThrow(false, ExcMessage("MultigridOperatorBase::m should be overwritten!"));
    return 0;
  }

  virtual types::global_dof_index
  n() const
  {
    AssertThrow(false, ExcMessage("MultigridOperatorBase::n should be overwritten!"));
    return 0;
  }

  virtual Number
  el(const unsigned int, const unsigned int) const
  {
    AssertThrow(false, ExcMessage("MultigridOperatorBase::el should be overwritten!"));
    return 0;
  }

  virtual void
  initialize_dof_vector(parallel::distributed::Vector<Number> & /*vector*/) const
  {
    AssertThrow(false, ExcMessage("MultigridOperatorBase::initialize_dof_vector should be overwritten!"));
  }

  virtual void
  calculate_inverse_diagonal(parallel::distributed::Vector<Number> & /*inverse_diagonal_entries*/) const
  {
    AssertThrow(false, ExcMessage("MultigridOperatorBase::calculate_inverse_diagonal should be overwritten!"));
  }

  virtual void
  apply_block_jacobi(parallel::distributed::Vector<Number> & /*dst*/,
                     parallel::distributed::Vector<Number> const & /*src*/) const
  {
    AssertThrow(false, ExcMessage("MultigridOperatorBase::apply_block_jacobi should be overwritten!"));
  }

  virtual void
  update_block_jacobi() const
  {
    AssertThrow(false, ExcMessage("MultigridOperatorBase::update_block_jacobi should be overwritten!"));
  }

  virtual bool
  is_empty_locally() const
  {
    AssertThrow(false, ExcMessage("MultigridOperatorBase::is_empty_locally should be overwritten!"));
    return 0;
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

  virtual MultigridOperatorBase<dim, Number> *
  get_new(unsigned int /*deg*/) const
  {
    AssertThrow(false, ExcMessage("MultigridOperatorBase::get_new should be overwritten!"));
    return nullptr;
  }
  
  virtual bool
  is_singular() const
  {
    // per default the operator is not singular
    // if an operator can be singular, you have to overwrite this method
    return false;
  }

#ifdef DEAL_II_WITH_TRILINOS
  virtual void
  init_system_matrix(TrilinosWrappers::SparseMatrix & /*system_matrix*/) const
  {
    AssertThrow(false, ExcMessage("MultigridOperatorBase::init_system_matrix should be overwritten!"));
  }

  virtual void
  calculate_system_matrix(TrilinosWrappers::SparseMatrix & /*system_matrix*/) const
  {
    AssertThrow(false, ExcMessage("MultigridOperatorBase::calculate_system_matrix should be overwritten!"));
  }
#endif

  virtual void
  reinit(const DoFHandler<dim> & /*dof_handler*/,
         const Mapping<dim> & /*mapping*/,
         void * /*operator_data*/,
         const MGConstrainedDoFs & /*mg_constrained_dofs*/,
         const unsigned int /*level*/ = numbers::invalid_unsigned_int)
  {
    AssertThrow(false, ExcMessage("MultigridOperatorBase::reinit should be overwritten!"));
  }
};

#endif