/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef INCLUDE_OPERATORS_BOUNDARY_MASS_OPERATOR_H_
#define INCLUDE_OPERATORS_BOUNDARY_MASS_OPERATOR_H_

#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/boundary_mass_kernel.h>
#include <exadg/operators/operator_base.h>

namespace ExaDG
{
template<int dim, typename Number>
struct BoundaryMassOperatorData : public OperatorBaseData
{
  BoundaryMassOperatorData() : OperatorBaseData()
  {
  }

  std::map<dealii::types::boundary_id, std::pair<bool, Number>> ids_normal_coefficients;
};

template<int dim, typename Number, int n_components>
class BoundaryMassOperator : public OperatorBase<dim, Number, n_components>
{
public:
  typedef OperatorBase<dim, Number, n_components>         Base;
  typedef BoundaryMassOperator<dim, Number, n_components> This;

  typedef std::pair<unsigned int, unsigned int> Range;
  typedef typename Base::VectorType             VectorType;

  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  BoundaryMassOperator();

  virtual ~BoundaryMassOperator()
  {
  }

  bool
  non_empty() const;

  IntegratorFlags
  get_integrator_flags() const;

  static MappingFlags
  get_mapping_flags();

  virtual void
  initialize(dealii::MatrixFree<dim, Number> const &       matrix_free,
             dealii::AffineConstraints<Number> const &     affine_constraints,
             BoundaryMassOperatorData<dim, Number> const & data);

  void
  set_scaling_factor(Number const & factor) const;

  void
  set_ids_normal_coefficients(std::map<dealii::types::boundary_id, std::pair<bool, Number>> const &
                                ids_normal_coefficients_in) const;

  void
  evaluate_add(VectorType & dst, VectorType const & src) const final;

private:
  void
  cell_loop_empty(dealii::MatrixFree<dim, Number> const & matrix_free,
                  VectorType &                            dst,
                  VectorType const &                      src,
                  Range const &                           range) const;

  void
  face_loop_empty(dealii::MatrixFree<dim, Number> const & matrix_free,
                  VectorType &                            dst,
                  VectorType const &                      src,
                  Range const &                           range) const;

  void
  boundary_face_loop_full_operator(dealii::MatrixFree<dim, Number> const & matrix_free,
                                   VectorType &                            dst,
                                   VectorType const &                      src,
                                   Range const &                           range) const;

  void
  do_boundary_segment_integral(IntegratorFace & integrator_m,
                               Number const &   scaled_coefficient,
                               bool const       normal_projection) const;

  BoundaryMassKernel<dim, Number> kernel;

  mutable double                                                        scaling_factor;
  mutable std::map<dealii::types::boundary_id, std::pair<bool, Number>> ids_normal_coefficients;
};

} // namespace ExaDG

#endif /* INCLUDE_OPERATORS_BOUNDARY_MASS_OPERATOR_H_ */
