/*
 * elasticity_operator_base.h
 *
 *  Created on: 16.04.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_ELASTICITY_OPERATOR_BASE_H_
#define INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_ELASTICITY_OPERATOR_BASE_H_

#include "../../../operators/operator_base.h"

#include "../../material/material_handler.h"

#include "../../user_interface/boundary_descriptor.h"
#include "../../user_interface/material_descriptor.h"

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

template<int dim>
struct OperatorData : public OperatorBaseData
{
  OperatorData()
    : OperatorBaseData(),
      pull_back_traction(false),
      unsteady(false),
      density(1.0),
      n_q_points_1d(2),
      quad_index_gauss_lobatto(0)
  {
  }

  std::shared_ptr<BoundaryDescriptor<dim>> bc;
  std::shared_ptr<MaterialDescriptor>      material_descriptor;

  // This parameter is only relevant for nonlinear operator
  // with large deformations. When set to true, the traction t
  // is pulled back to the reference configuration, t_0 = da/dA t.
  bool pull_back_traction;

  // activates mass matrix operator in operator evaluation for unsteady problems
  bool unsteady;

  // density
  double density;

  // for a material law with spatially varying coefficients, the number of 1D
  // quadrature points is needed
  unsigned int n_q_points_1d;

  // for Dirichlet mortar boundary conditions, another quadrature rule
  // is needed to set the constrained DoFs.
  unsigned int quad_index_gauss_lobatto;
};

template<int dim, typename Number>
class ElasticityOperatorBase : public OperatorBase<dim, Number, dim>
{
public:
  typedef Number value_type;

protected:
  typedef OperatorBase<dim, Number, dim> Base;
  typedef typename Base::VectorType      VectorType;
  typedef typename Base::IntegratorFace  IntegratorFace;

public:
  ElasticityOperatorBase();

  virtual ~ElasticityOperatorBase()
  {
  }

  IntegratorFlags
  get_integrator_flags(bool const unsteady) const;

  static MappingFlags
  get_mapping_flags();

  virtual void
  initialize(MatrixFree<dim, Number> const &   matrix_free,
             AffineConstraints<double> const & constraint_matrix,
             OperatorData<dim> const &         data);

  OperatorData<dim> const &
  get_data() const;

  void
  set_scaling_factor_mass(double const factor) const;

  void
  set_constrained_values(VectorType & dst, double const time) const override;

protected:
  virtual void
  reinit_cell(unsigned int const cell) const;

  OperatorData<dim> operator_data;

  mutable MaterialHandler<dim, Number> material_handler;

  mutable double scaling_factor_mass;
};

} // namespace Structure
} // namespace ExaDG


#endif /* INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_ELASTICITY_OPERATOR_BASE_H_ */
