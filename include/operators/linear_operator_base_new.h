#ifndef OPERATOR_H
#define OPERATOR_H


#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>

#include "linear_operator_base.h"

using namespace dealii;

template<typename Number>
class LinearOperatorBaseNew : public LinearOperatorBase
{
public:
  typedef Number                                     value_type;
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  virtual void
  apply(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  apply_add(VectorType & dst, VectorType const & src, Number const time) const = 0;

  virtual void
  apply_add(VectorType & dst, VectorType const & src) const = 0;

  virtual void
  rhs(VectorType & dst) const = 0;

  virtual void
  rhs(VectorType & dst, Number const time) const = 0;

  virtual void
  rhs_add(VectorType & dst) const = 0;

  virtual void
  rhs_add(VectorType & dst, Number const time) const = 0;

  virtual void
  evaluate(VectorType & dst, VectorType const & src, Number const time) const = 0;

  virtual void
  evaluate_add(VectorType & dst, VectorType const & src, Number const time) const = 0;

  virtual types::global_dof_index
  m() const = 0;

  virtual types::global_dof_index
  n() const = 0;

  virtual Number
  el(const unsigned int, const unsigned int) const = 0;

  virtual bool
  is_empty_locally() const = 0;

  virtual void
  initialize_dof_vector(VectorType & vector) const = 0;
};

#endif
