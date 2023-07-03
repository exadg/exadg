/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_ELASTICITY_OPERATOR_BASE_H_
#define INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_ELASTICITY_OPERATOR_BASE_H_

#include <exadg/operators/operator_base.h>
#include <exadg/structure/material/material_handler.h>
#include <exadg/structure/user_interface/boundary_descriptor.h>
#include <exadg/structure/user_interface/material_descriptor.h>

namespace ExaDG
{
namespace Structure
{
template<int dim>
struct OperatorData : public OperatorBaseData
{
  OperatorData()
    : OperatorBaseData(),
      pull_back_traction(false),
      unsteady(false),
      density(1.0),
      quad_index_gauss_lobatto(0)
  {
  }

  std::shared_ptr<BoundaryDescriptor<dim> const> bc;
  std::shared_ptr<MaterialDescriptor const>      material_descriptor;

  // This parameter is only relevant for nonlinear operator
  // with large deformations. When set to true, the traction t
  // is pulled back to the reference configuration, t_0 = da/dA t.
  bool pull_back_traction;

  // activates mass operator in operator evaluation for unsteady problems
  bool unsteady;

  // density
  double density;

  // for DirichletCached boundary conditions, another quadrature rule
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
  typedef typename Base::IntegratorCell  IntegratorCell;
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
  initialize(dealii::MatrixFree<dim, Number> const &   matrix_free,
             dealii::AffineConstraints<Number> const & affine_constraints,
             OperatorData<dim> const &                 data);

  OperatorData<dim> const &
  get_data() const;

  void
  set_scaling_factor_mass_operator(double const scaling_factor) const;

  double
  get_scaling_factor_mass_operator() const;

  void
  set_inhomogeneous_boundary_values(VectorType & dst) const final;

protected:
  void
  reinit_cell_derived(IntegratorCell & integrator, unsigned int const cell) const override;

  OperatorData<dim> operator_data;

  mutable MaterialHandler<dim, Number> material_handler;

  mutable double scaling_factor_mass;
};

} // namespace Structure
} // namespace ExaDG


#endif /* INCLUDE_EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_ELASTICITY_OPERATOR_BASE_H_ */
