/*
 * divergence_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_DIVERGENCE_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_DIVERGENCE_OPERATOR_H_


#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "../../user_interface/input_parameters.h"
#include "weak_boundary_conditions.h"

using namespace dealii;

namespace IncNS
{
template<int dim>
struct DivergenceOperatorData
{
  DivergenceOperatorData()
    : dof_index_velocity(0),
      dof_index_pressure(1),
      quad_index(0),
      integration_by_parts(true),
      use_boundary_data(true)
  {
  }

  unsigned int dof_index_velocity;
  unsigned int dof_index_pressure;

  unsigned int quad_index;

  bool integration_by_parts;
  bool use_boundary_data;

  std::shared_ptr<BoundaryDescriptorU<dim>> bc;
};

template<int dim, typename Number>
class DivergenceOperator
{
public:
  typedef DivergenceOperator<dim, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorU;
  typedef CellIntegrator<dim, 1, Number>   CellIntegratorP;

  typedef FaceIntegrator<dim, dim, Number> FaceIntegratorU;
  typedef FaceIntegrator<dim, 1, Number>   FaceIntegratorP;


  DivergenceOperator() : matrix_free(nullptr), eval_time(0.0)
  {
  }

  void
  initialize(MatrixFree<dim, Number> const &     matrix_free_in,
             DivergenceOperatorData<dim> const & operator_data_in)
  {
    this->matrix_free   = &matrix_free_in;
    this->operator_data = operator_data_in;
  }

  void
  apply(VectorType & dst, VectorType const & src) const
  {
    matrix_free->loop(&This::cell_loop,
                      &This::face_loop,
                      &This::boundary_face_loop_hom_operator,
                      this,
                      dst,
                      src,
                      true /*zero_dst_vector = true*/);
  }

  void
  apply_add(VectorType & dst, VectorType const & src) const
  {
    matrix_free->loop(&This::cell_loop,
                      &This::face_loop,
                      &This::boundary_face_loop_hom_operator,
                      this,
                      dst,
                      src,
                      false /*zero_dst_vector = false*/);
  }

  void
  rhs(VectorType & dst, Number const evaluation_time) const
  {
    dst = 0.0;
    rhs_add(dst, evaluation_time);
  }

  void
  rhs_add(VectorType & dst, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    VectorType tmp;
    tmp.reinit(dst, false /* init with 0 */);

    matrix_free->loop(&This::cell_loop_inhom_operator,
                      &This::face_loop_inhom_operator,
                      &This::boundary_face_loop_inhom_operator,
                      this,
                      tmp,
                      tmp,
                      false /*zero_dst_vector = false*/);

    // multiply by -1.0 since the boundary face integrals have to be shifted to the right hand side
    dst.add(-1.0, tmp);
  }

  void
  evaluate(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    matrix_free->loop(&This::cell_loop,
                      &This::face_loop,
                      &This::boundary_face_loop_full_operator,
                      this,
                      dst,
                      src,
                      true /*zero_dst_vector = true*/);
  }

  void
  evaluate_add(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    matrix_free->loop(&This::cell_loop,
                      &This::face_loop,
                      &This::boundary_face_loop_full_operator,
                      this,
                      dst,
                      src,
                      false /*zero_dst_vector = false*/);
  }

private:
  template<typename CellIntegratorP, typename CellIntegratorU>
  void
  do_cell_integral_weak(CellIntegratorP & pressure, CellIntegratorU & velocity) const
  {
    for(unsigned int q = 0; q < velocity.n_q_points; ++q)
    {
      // minus sign due to integration by parts
      pressure.submit_gradient(-velocity.get_value(q), q);
    }
  }

  template<typename CellIntegratorP, typename CellIntegratorU>
  void
  do_cell_integral_strong(CellIntegratorP & pressure, CellIntegratorU & velocity) const
  {
    for(unsigned int q = 0; q < velocity.n_q_points; ++q)
    {
      pressure.submit_value(velocity.get_divergence(q), q);
    }
  }

  template<typename FaceIntegratorP, typename FaceIntegratorU>
  void
  do_face_integral(FaceIntegratorU & velocity_m,
                   FaceIntegratorU & velocity_p,
                   FaceIntegratorP & pressure_m,
                   FaceIntegratorP & pressure_p) const
  {
    for(unsigned int q = 0; q < velocity_m.n_q_points; ++q)
    {
      vector value_m = velocity_m.get_value(q);
      vector value_p = velocity_p.get_value(q);

      vector flux = calculate_flux(value_m, value_p);

      scalar flux_times_normal = flux * velocity_m.get_normal_vector(q);

      pressure_m.submit_value(flux_times_normal, q);
      // minus sign since n⁺ = - n⁻
      pressure_p.submit_value(-flux_times_normal, q);
    }
  }

  template<typename FaceIntegratorP, typename FaceIntegratorU>
  void
  do_boundary_integral(FaceIntegratorU &          velocity,
                       FaceIntegratorP &          pressure,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id) const
  {
    BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < pressure.n_q_points; ++q)
    {
      vector flux;

      if(operator_data.use_boundary_data == true)
      {
        vector value_m = calculate_interior_value(q, velocity, operator_type);
        vector value_p = calculate_exterior_value(value_m,
                                                  q,
                                                  velocity,
                                                  operator_type,
                                                  boundary_type,
                                                  boundary_id,
                                                  operator_data.bc,
                                                  this->eval_time);

        flux = calculate_flux(value_m, value_p);
      }
      else // use_boundary_data == false
      {
        vector value_m = velocity.get_value(q);

        flux = calculate_flux(value_m, value_m /* value_p = value_m */);
      }

      scalar flux_times_normal = flux * velocity.get_normal_vector(q);
      pressure.submit_value(flux_times_normal, q);
    }
  }

  /*
   *  This function implements the central flux as numerical flux function.
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_flux(vector const & value_m, vector const & value_p) const
  {
    return 0.5 * (value_m + value_p);
  }

  void
  cell_loop(MatrixFree<dim, Number> const & matrix_free,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const
  {
    CellIntegratorU velocity(matrix_free,
                             operator_data.dof_index_velocity,
                             operator_data.quad_index);
    CellIntegratorP pressure(matrix_free,
                             operator_data.dof_index_pressure,
                             operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      pressure.reinit(cell);

      velocity.reinit(cell);

      if(operator_data.integration_by_parts == true)
      {
        velocity.gather_evaluate(src, true, false, false);

        do_cell_integral_weak(pressure, velocity);

        pressure.integrate_scatter(false, true, dst);
      }
      else // integration_by_parts == false
      {
        velocity.gather_evaluate(src, false, true, false);

        do_cell_integral_strong(pressure, velocity);

        pressure.integrate_scatter(true, false, dst);
      }
    }
  }

  void
  face_loop(MatrixFree<dim, Number> const & matrix_free,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   face_range) const
  {
    if(operator_data.integration_by_parts == true)
    {
      FaceIntegratorU velocity_m(matrix_free,
                                 true,
                                 operator_data.dof_index_velocity,
                                 operator_data.quad_index);
      FaceIntegratorU velocity_p(matrix_free,
                                 false,
                                 operator_data.dof_index_velocity,
                                 operator_data.quad_index);

      FaceIntegratorP pressure_m(matrix_free,
                                 true,
                                 operator_data.dof_index_pressure,
                                 operator_data.quad_index);
      FaceIntegratorP pressure_p(matrix_free,
                                 false,
                                 operator_data.dof_index_pressure,
                                 operator_data.quad_index);

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        pressure_m.reinit(face);
        pressure_p.reinit(face);

        velocity_m.reinit(face);
        velocity_p.reinit(face);

        velocity_m.gather_evaluate(src, true, false);
        velocity_p.gather_evaluate(src, true, false);

        do_face_integral(velocity_m, velocity_p, pressure_m, pressure_p);

        pressure_m.integrate_scatter(true, false, dst);
        pressure_p.integrate_scatter(true, false, dst);
      }
    }
  }

  void
  boundary_face_loop_hom_operator(MatrixFree<dim, Number> const & matrix_free,
                                  VectorType &                    dst,
                                  VectorType const &              src,
                                  Range const &                   face_range) const
  {
    if(operator_data.integration_by_parts == true)
    {
      FaceIntegratorU velocity(matrix_free,
                               true,
                               operator_data.dof_index_velocity,
                               operator_data.quad_index);

      FaceIntegratorP pressure(matrix_free,
                               true,
                               operator_data.dof_index_pressure,
                               operator_data.quad_index);

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        pressure.reinit(face);
        velocity.reinit(face);

        velocity.gather_evaluate(src, true, false);

        do_boundary_integral(velocity,
                             pressure,
                             OperatorType::homogeneous,
                             matrix_free.get_boundary_id(face));

        pressure.integrate_scatter(true, false, dst);
      }
    }
  }

  void
  boundary_face_loop_full_operator(MatrixFree<dim, Number> const & matrix_free,
                                   VectorType &                    dst,
                                   VectorType const &              src,
                                   Range const &                   face_range) const
  {
    if(operator_data.integration_by_parts == true)
    {
      FaceIntegratorU velocity(matrix_free,
                               true,
                               operator_data.dof_index_velocity,
                               operator_data.quad_index);

      FaceIntegratorP pressure(matrix_free,
                               true,
                               operator_data.dof_index_pressure,
                               operator_data.quad_index);

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        pressure.reinit(face);
        velocity.reinit(face);

        velocity.gather_evaluate(src, true, false);

        do_boundary_integral(velocity,
                             pressure,
                             OperatorType::full,
                             matrix_free.get_boundary_id(face));

        pressure.integrate_scatter(true, false, dst);
      }
    }
  }

  void
  cell_loop_inhom_operator(MatrixFree<dim, Number> const &,
                           VectorType &,
                           VectorType const &,
                           Range const &) const
  {
  }

  void
  face_loop_inhom_operator(MatrixFree<dim, Number> const &,
                           VectorType &,
                           VectorType const &,
                           Range const &) const
  {
  }

  void
  boundary_face_loop_inhom_operator(MatrixFree<dim, Number> const & matrix_free,
                                    VectorType &                    dst,
                                    VectorType const &,
                                    std::pair<unsigned int, unsigned int> const & face_range) const
  {
    if(operator_data.integration_by_parts == true)
    {
      FaceIntegratorU velocity(matrix_free,
                               true,
                               operator_data.dof_index_velocity,
                               operator_data.quad_index);

      FaceIntegratorP pressure(matrix_free,
                               true,
                               operator_data.dof_index_pressure,
                               operator_data.quad_index);

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        pressure.reinit(face);
        velocity.reinit(face);

        do_boundary_integral(velocity,
                             pressure,
                             OperatorType::inhomogeneous,
                             matrix_free.get_boundary_id(face));

        pressure.integrate_scatter(true, false, dst);
      }
    }
  }

  MatrixFree<dim, Number> const * matrix_free;

  DivergenceOperatorData<dim> operator_data;

  mutable Number eval_time;
};

} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_DIVERGENCE_OPERATOR_H_ \
        */
