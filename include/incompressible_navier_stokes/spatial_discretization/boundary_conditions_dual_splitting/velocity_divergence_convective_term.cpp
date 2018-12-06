/*
 * velocity_divergence_convective_term.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include "velocity_divergence_convective_term.h"

namespace IncNS
{
template<int dim, int degree_u, int degree_p, typename Number>
VelocityDivergenceConvectiveTerm<dim, degree_u, degree_p, Number>::
  VelocityDivergenceConvectiveTerm()
  : data(nullptr)
{
}

template<int dim, int degree_u, int degree_p, typename Number>
void
VelocityDivergenceConvectiveTerm<dim, degree_u, degree_p, Number>::initialize(
  MatrixFree<dim, Number> const &             data,
  VelocityDivergenceConvectiveTermData<dim> & operator_data)
{
  this->data          = &data;
  this->operator_data = operator_data;
}

template<int dim, int degree_u, int degree_p, typename Number>
void
VelocityDivergenceConvectiveTerm<dim, degree_u, degree_p, Number>::calculate(
  VectorType &       dst,
  VectorType const & src) const
{
  this->data->loop(&This::cell_loop, &This::face_loop, &This::boundary_face_loop, this, dst, src);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
VelocityDivergenceConvectiveTerm<dim, degree_u, degree_p, Number>::cell_loop(
  MatrixFree<dim, Number> const &,
  VectorType &,
  VectorType const &,
  Range const &) const
{
}

template<int dim, int degree_u, int degree_p, typename Number>
void
VelocityDivergenceConvectiveTerm<dim, degree_u, degree_p, Number>::face_loop(
  MatrixFree<dim, Number> const &,
  VectorType &,
  VectorType const &,
  Range const &) const
{
}

template<int dim, int degree_u, int degree_p, typename Number>
void
VelocityDivergenceConvectiveTerm<dim, degree_u, degree_p, Number>::boundary_face_loop(
  MatrixFree<dim, Number> const & data,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   face_range) const
{
  FEFaceEvalVelocity fe_eval_velocity(data,
                                      true,
                                      operator_data.dof_index_velocity,
                                      operator_data.quad_index);
  FEFaceEvalPressure fe_eval_pressure(data,
                                      true,
                                      operator_data.dof_index_pressure,
                                      operator_data.quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    fe_eval_velocity.reinit(face);
    fe_eval_velocity.gather_evaluate(src, true, true);

    fe_eval_pressure.reinit(face);

    BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(data.get_boundary_id(face));

    for(unsigned int q = 0; q < fe_eval_pressure.n_q_points; ++q)
    {
      if(boundary_type == BoundaryTypeU::Dirichlet)
      {
        vector normal = fe_eval_pressure.get_normal_vector(q);

        vector u      = fe_eval_velocity.get_value(q);
        tensor grad_u = fe_eval_velocity.get_gradient(q);
        scalar div_u  = fe_eval_velocity.get_divergence(q);

        scalar flux_times_normal = (grad_u * u + div_u * u) * normal;

        fe_eval_pressure.submit_value(flux_times_normal, q);
      }
      else if(boundary_type == BoundaryTypeU::Neumann || boundary_type == BoundaryTypeU::Symmetry)
      {
        // Do nothing on Neumann and Symmetry boundaries.
        // Remark: on symmetry boundaries we prescribe g_u * n = 0, and also g_{u_hat}*n = 0 in
        // case of the dual splitting scheme. This is in contrast to Dirichlet boundaries where we
        // prescribe a consistent boundary condition for g_{u_hat} derived from the convective
        // step of the dual splitting scheme which differs from the DBC g_u. Applying this
        // consistent DBC to symmetry boundaries and using g_u*n=0 as well as exploiting symmetry,
        // we obtain g_{u_hat}*n=0 on symmetry boundaries. Hence, there are no inhomogeneous
        // contributions for g_{u_hat}*n.
        scalar zero = make_vectorized_array<Number>(0.0);
        fe_eval_pressure.submit_value(zero, q);
      }
      else
      {
        AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    fe_eval_pressure.integrate_scatter(true, false, dst);
  }
}

} // namespace IncNS

#include "velocity_divergence_convective_term.hpp"
