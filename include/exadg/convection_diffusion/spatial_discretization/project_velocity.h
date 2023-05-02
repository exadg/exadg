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

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_PROJECT_VELOCITY_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_PROJECT_VELOCITY_H_

#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/inverse_mass_operator.h>

namespace ExaDG
{
template<int dim, typename Number>
class VelocityProjection
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef CellIntegrator<dim, dim, Number> IntegratorCell;

  typedef std::pair<unsigned int, unsigned int> Range;

public:
  /*
   * (v_h, u_h)_Omega^e = (v_h, f)_Omega^e -> M * U = RHS -> U = M^{-1} * RHS
   */
  void
  apply(dealii::MatrixFree<dim, Number> const &      matrix_free,
        InverseMassOperatorData const                inverse_mass_operator_data,
        std::shared_ptr<dealii::Function<dim>> const function,
        double const &                               time,
        VectorType &                                 vector)
  {
    this->dof_index  = inverse_mass_operator_data.dof_index;
    this->quad_index = inverse_mass_operator_data.quad_index;
    this->function   = function;
    this->time       = time;

    // calculate RHS
    VectorType src;
    matrix_free.cell_loop(&VelocityProjection<dim, Number>::cell_loop, this, vector, src);

    // apply M^{-1}
    InverseMassOperator<dim, dim, Number> inverse_mass;
    inverse_mass.initialize(matrix_free, inverse_mass_operator_data);
    inverse_mass.apply(vector, vector);
  }

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst,
            VectorType const &                      src,
            Range const &                           cell_range) const
  {
    (void)src;

    IntegratorCell integrator(matrix_free, dof_index, quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);

      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        integrator.submit_value(
          FunctionEvaluator<1, dim, Number>::value(function, integrator.quadrature_point(q), time),
          q);
      }

      integrator.integrate(dealii::EvaluationFlags::values);

      integrator.distribute_local_to_global(dst);
    }
  }

  unsigned int                           dof_index;
  unsigned int                           quad_index;
  std::shared_ptr<dealii::Function<dim>> function;
  double                                 time;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_PROJECT_VELOCITY_H_ */
