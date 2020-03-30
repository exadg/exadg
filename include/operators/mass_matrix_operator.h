/*
 * mass_matrix_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MASS_MATRIX_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MASS_MATRIX_OPERATOR_H_

#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "mass_matrix_kernel.h"
#include "operator_base.h"

using namespace dealii;

// required for OperatorBase interface but is never used for the mass matrix operator
template<int dim>
struct BoundaryDescriptorDummy
{
  // Dirichlet: prescribe all components of the velocity
  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> dirichlet_bc;
};

template<int dim>
struct MassMatrixOperatorData : public OperatorBaseData
{
  MassMatrixOperatorData() : OperatorBaseData(0 /* dof_index */, 0 /* quad_index */)
  {
  }

  // required by OperatorBase interface
  std::shared_ptr<BoundaryDescriptorDummy<dim>> bc;
};

template<int dim, int n_components, typename Number>
class MassMatrixOperator
  : public OperatorBase<dim, Number, MassMatrixOperatorData<dim>, n_components>
{
public:
  typedef OperatorBase<dim, Number, MassMatrixOperatorData<dim>, n_components> Base;

  typedef typename Base::VectorType     VectorType;
  typedef typename Base::IntegratorCell IntegratorCell;

  MassMatrixOperator();

  void
  reinit(MatrixFree<dim, Number> const &     matrix_free,
         AffineConstraints<double> const &   constraint_matrix,
         MassMatrixOperatorData<dim> const & data);

  void
  set_scaling_factor(Number const & number);

  void
  apply_scale(VectorType & dst, Number const & factor, VectorType const & src) const;

  void
  apply_scale_add(VectorType & dst, Number const & factor, VectorType const & src) const;

private:
  void
  do_cell_integral(IntegratorCell & integrator) const;

  MassMatrixKernel<dim, Number> kernel;

  mutable double scaling_factor;
};



#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MASS_MATRIX_OPERATOR_H_ \
        */
