/*
 * InverseMassMatrixPreconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERSEMASSMATRIXPRECONDITIONER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERSEMASSMATRIXPRECONDITIONER_H_

#include <deal.II/lac/parallel_vector.h>

#include "../preconditioner/preconditioner_base.h"
#include "operators/inverse_mass_matrix.h"
#include "operators/matrix_operator_base.h"

template<int dim, int fe_degree, typename value_type, int n_components = dim>
class InverseMassMatrixPreconditioner : public PreconditionerBase<value_type>
{
public:
  InverseMassMatrixPreconditioner(MatrixFree<dim, value_type> const & mf_data,
                                  unsigned int const                  dof_index,
                                  unsigned int const                  quad_index)
  {
    inverse_mass_matrix_operator.initialize(mf_data, dof_index, quad_index);
  }

  void
  vmult(parallel::distributed::Vector<value_type> &       dst,
        const parallel::distributed::Vector<value_type> & src) const
  {
    inverse_mass_matrix_operator.apply(dst, src);
  }

  void
  update(MatrixOperatorBase const * /*matrix_operator*/)
  {
  } // do nothing

private:
  InverseMassMatrixOperator<dim, fe_degree, value_type, n_components> inverse_mass_matrix_operator;
};

template<int dim, int fe_degree, typename value_type, int n_components = dim>
class InverseMassMatrixPreconditionerPtr : public PreconditionerBase<value_type>
{
public:
  InverseMassMatrixPreconditionerPtr(
    std::shared_ptr<InverseMassMatrixOperator<dim, fe_degree, value_type, n_components>>
      inv_mass_operator)
    : inverse_mass_matrix_operator(inv_mass_operator)
  {
  }

  void
  vmult(parallel::distributed::Vector<value_type> &       dst,
        const parallel::distributed::Vector<value_type> & src) const
  {
    inverse_mass_matrix_operator->apply_inverse_mass_matrix(dst, src);
  }

  void
  update(MatrixOperatorBase const * /*matrix_operator*/)
  {
  } // do nothing

private:
  std::shared_ptr<InverseMassMatrixOperator<dim, fe_degree, value_type, n_components>>
    inverse_mass_matrix_operator;
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERSEMASSMATRIXPRECONDITIONER_H_ */
