/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by the ExaDG authors
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

#ifndef EXADG_OPERATORS_BOUNDARY_MASS_OPERATOR_H_
#define EXADG_OPERATORS_BOUNDARY_MASS_OPERATOR_H_

#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/boundary_mass_kernel.h>
#include <exadg/operators/operator_base.h>

namespace ExaDG
{
template<typename Number>
struct BoundaryMassParameters
{
  Number scaling_factor_per_id;
  bool   use_normal_projection_per_id;
};

template<int dim, typename Number>
struct BoundaryMassOperatorData : public OperatorBaseData
{
  BoundaryMassOperatorData() : OperatorBaseData()
  {
  }

  std::map<dealii::types::boundary_id, BoundaryMassParameters<Number>> ids_to_parameters;
};

/*
 * The boundary mass operator adds boundary integrals of the form
 *
 * (v_h, c_i * u_h)_{Gamma_i}
 *
 * or
 *
 * (v_h, n * c_i * (u_h * n))_{Gamma_i}
 *
 * for all boundary parts Gamma_i with boundary ID i, where `factor_i` is the product of
 * `scaling_factor_per_id` and `global_scaling_factor`, and is hence a scaling factor associated to
 * boundary Gamma_i with ID i.
 */

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
  set_global_scaling_factor(Number const & factor) const;

  void
  set_ids_to_parameters(std::map<dealii::types::boundary_id, BoundaryMassParameters<Number>> const &
                          ids_to_parameters_in) const;

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
  do_boundary_mass_integral(IntegratorFace & integrator_m,
                            Number const &   scaled_coefficient,
                            bool const       normal_projection) const;

  BoundaryMassKernel<dim, Number> kernel;

  mutable Number global_scaling_factor;
  mutable std::map<dealii::types::boundary_id, BoundaryMassParameters<Number>> ids_to_parameters;
};

} // namespace ExaDG

#endif /* EXADG_OPERATORS_BOUNDARY_MASS_OPERATOR_H_ */
