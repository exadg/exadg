/*
 * project_velocity.h
 *
 *  Created on: Jul 1, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_PROJECT_VELOCITY_H_
#define INCLUDE_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_PROJECT_VELOCITY_H_

#include "../../matrix_free/integrators.h"

#include "../../functions_and_boundary_conditions/evaluate_functions.h"
#include "../../operators/inverse_mass_matrix.h"

namespace ExaDG
{
using namespace dealii;

template<int dim, typename Number>
class VelocityProjection
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef CellIntegrator<dim, dim, Number> IntegratorCell;

  typedef std::pair<unsigned int, unsigned int> Range;

public:
  /*
   * (v_h, u_h)_Omega^e = (v_h, f)_Omega^e -> M * U = RHS -> U = M^{-1} * RHS
   */
  void
  apply(MatrixFree<dim, Number> const &      matrix_free,
        unsigned int const                   dof_index,
        unsigned int const                   quad_index,
        std::shared_ptr<Function<dim>> const function,
        double const &                       time,
        VectorType &                         vector)
  {
    this->dof_index  = dof_index;
    this->quad_index = quad_index;
    this->function   = function;
    this->time       = time;

    // calculate RHS
    VectorType src;
    matrix_free.cell_loop(&VelocityProjection<dim, Number>::cell_loop, this, vector, src);

    // apply M^{-1}
    InverseMassMatrixOperator<dim, dim, Number> inverse_mass;
    inverse_mass.initialize(matrix_free, dof_index, quad_index);
    inverse_mass.apply(vector, vector);
  }

private:
  void
  cell_loop(MatrixFree<dim, Number> const & matrix_free,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const
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

      integrator.integrate(true, false);

      integrator.distribute_local_to_global(dst);
    }
  }

  unsigned int                   dof_index;
  unsigned int                   quad_index;
  std::shared_ptr<Function<dim>> function;
  double                         time;
};

} // namespace ExaDG

#endif /* INCLUDE_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_PROJECT_VELOCITY_H_ */
