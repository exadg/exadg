/*
 * pressure_neumann_bc_viscous_term.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include "pressure_neumann_bc_viscous_term.h"

namespace IncNS
{
template<int dim, int degree_u, int degree_p, typename Number>
PressureNeumannBCViscousTerm<dim, degree_u, degree_p, Number>::PressureNeumannBCViscousTerm()
  : data(nullptr), viscous_operator(nullptr)
{
}

template<int dim, int degree_u, int degree_p, typename Number>
void
PressureNeumannBCViscousTerm<dim, degree_u, degree_p, Number>::initialize(
  MatrixFree<dim, Number> const &                data,
  PressureNeumannBCViscousTermData<dim> &        operator_data,
  ViscousOperator<dim, degree_u, Number> const & viscous_operator)
{
  this->data             = &data;
  this->operator_data    = operator_data;
  this->viscous_operator = &viscous_operator;
}

template<int dim, int degree_u, int degree_p, typename Number>
void
PressureNeumannBCViscousTerm<dim, degree_u, degree_p, Number>::calculate(
  VectorType &       dst,
  VectorType const & src) const
{
  this->data->loop(&This::cell_loop, &This::face_loop, &This::boundary_face_loop, this, dst, src);
}

template<int dim, int degree_u, int degree_p, typename Number>
void
PressureNeumannBCViscousTerm<dim, degree_u, degree_p, Number>::cell_loop(
  MatrixFree<dim, Number> const &,
  VectorType &,
  VectorType const &,
  Range const &) const
{
}

template<int dim, int degree_u, int degree_p, typename Number>
void
PressureNeumannBCViscousTerm<dim, degree_u, degree_p, Number>::face_loop(
  MatrixFree<dim, Number> const &,
  VectorType &,
  VectorType const &,
  Range const &) const
{
}

template<int dim, int degree_u, int degree_p, typename Number>
void
PressureNeumannBCViscousTerm<dim, degree_u, degree_p, Number>::boundary_face_loop(
  MatrixFree<dim, Number> const & data,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   face_range) const
{
  FEFaceEvalVelocity fe_eval_omega(data,
                                   true,
                                   operator_data.dof_index_velocity,
                                   operator_data.quad_index);

  FEFaceEvalPressure fe_eval_pressure(data,
                                      true,
                                      operator_data.dof_index_pressure,
                                      operator_data.quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    fe_eval_pressure.reinit(face);

    fe_eval_omega.reinit(face);
    fe_eval_omega.gather_evaluate(src, false, true);

    BoundaryTypeP boundary_type = operator_data.bc->get_boundary_type(data.get_boundary_id(face));

    for(unsigned int q = 0; q < fe_eval_pressure.n_q_points; ++q)
    {
      scalar viscosity = viscous_operator->get_viscosity(face, q);

      if(boundary_type == BoundaryTypeP::Neumann)
      {
        scalar h = make_vectorized_array<Number>(0.0);

        vector normal = fe_eval_pressure.get_normal_vector(q);

        vector curl_omega = CurlCompute<dim, FEFaceEvalVelocity>::compute(fe_eval_omega, q);

        h = -normal * (viscosity * curl_omega);

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

#include "pressure_neumann_bc_viscous_term.hpp"
