/*
 * pressure_neumann_bc_convective_term.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include "pressure_neumann_bc_convective_term.h"

namespace IncNS
{
template<int dim, int degree_u, int degree_p, typename Number>
PressureNeumannBCConvectiveTerm<dim, degree_u, degree_p, Number>::PressureNeumannBCConvectiveTerm()
  : data(nullptr)
{
}

template<int dim, int degree_u, int degree_p, typename Number>
void
PressureNeumannBCConvectiveTerm<dim, degree_u, degree_p, Number>::initialize(
  MatrixFree<dim, Number> const &            data,
  PressureNeumannBCConvectiveTermData<dim> & operator_data)
{
  this->data          = &data;
  this->operator_data = operator_data;
}

template<int dim, int degree_u, int degree_p, typename Number>
void
PressureNeumannBCConvectiveTerm<dim, degree_u, degree_p, Number>::calculate(
  VectorType &       dst,
  VectorType const & src) const
{
  this->data->loop(&This::cell_loop, &This::face_loop, &This::boundary_face_loop, this, dst, src);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
PressureNeumannBCConvectiveTerm<dim, degree_u, degree_p, Number>::cell_loop(
  MatrixFree<dim, Number> const &,
  VectorType &,
  VectorType const &,
  Range const &) const
{
}

template<int dim, int degree_u, int degree_p, typename Number>
void
PressureNeumannBCConvectiveTerm<dim, degree_u, degree_p, Number>::face_loop(
  MatrixFree<dim, Number> const &,
  VectorType &,
  VectorType const &,
  Range const &) const
{
}

template<int dim, int degree_u, int degree_p, typename Number>
void
PressureNeumannBCConvectiveTerm<dim, degree_u, degree_p, Number>::boundary_face_loop(
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

    BoundaryTypeP boundary_type = operator_data.bc->get_boundary_type(data.get_boundary_id(face));

    for(unsigned int q = 0; q < fe_eval_pressure.n_q_points; ++q)
    {
      if(boundary_type == BoundaryTypeP::Neumann)
      {
        scalar h = make_vectorized_array<Number>(0.0);

        vector normal = fe_eval_pressure.get_normal_vector(q);

        vector u      = fe_eval_velocity.get_value(q);
        tensor grad_u = fe_eval_velocity.get_gradient(q);
        scalar div_u  = fe_eval_velocity.get_divergence(q);

        h = -normal * (grad_u * u + div_u * u);

        fe_eval_pressure.submit_value(h, q);
      }
      else if(boundary_type == BoundaryTypeP::Dirichlet)
      {
        fe_eval_pressure.submit_value(make_vectorized_array<Number>(0.0), q);
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

#include "pressure_neumann_bc_convective_term.hpp"
