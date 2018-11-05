/*
 * dual_splitting_boundary_conditions.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DUAL_SPLITTING_BOUNDARY_CONDITIONS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DUAL_SPLITTING_BOUNDARY_CONDITIONS_H_


#include <deal.II/matrix_free/fe_evaluation.h>

#include "../../incompressible_navier_stokes/spatial_discretization/curl_compute.h"
#include "../../incompressible_navier_stokes/user_interface/boundary_descriptor.h"

namespace IncNS
{
template<int dim>
class PressureNeumannBCConvectiveTermData
{
public:
  PressureNeumannBCConvectiveTermData()
    : dof_index_velocity(0), dof_index_pressure(0), quad_index(0)
  {
  }

  unsigned int dof_index_velocity;
  unsigned int dof_index_pressure;

  unsigned int quad_index;

  std::shared_ptr<BoundaryDescriptorP<dim>> bc;
};

template<int dim, int degree_u, int degree_p, typename Number>
class PressureNeumannBCConvectiveTerm
{
public:
  typedef PressureNeumannBCConvectiveTerm<dim, degree_u, degree_p, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;
  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  typedef std::pair<unsigned int, unsigned int> Range;

  static const unsigned int n_q_points_overint = degree_u + (degree_u + 2) / 2;

  typedef FEFaceEvaluation<dim, degree_u, n_q_points_overint, dim, Number> FEFaceEvalVelocity;
  typedef FEFaceEvaluation<dim, degree_p, n_q_points_overint, 1, Number>   FEFaceEvalPressure;

  PressureNeumannBCConvectiveTerm() : data(nullptr)
  {
  }

  void
  initialize(MatrixFree<dim, Number> const &            data,
             PressureNeumannBCConvectiveTermData<dim> & operator_data)
  {
    this->data          = &data;
    this->operator_data = operator_data;
  }

  void
  calculate(VectorType & dst, VectorType const & src) const
  {
    this->data->loop(&This::cell_loop, &This::face_loop, &This::boundary_face_loop, this, dst, src);
  }

private:
  void
  cell_loop(MatrixFree<dim, Number> const &, VectorType &, VectorType const &, Range const &) const
  {
  }

  void
  face_loop(MatrixFree<dim, Number> const &, VectorType &, VectorType const &, Range const &) const
  {
  }

  void
  boundary_face_loop(MatrixFree<dim, Number> const & data,
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

  MatrixFree<dim, Number> const *          data;
  PressureNeumannBCConvectiveTermData<dim> operator_data;
};

template<int dim>
class PressureNeumannBCViscousTermData
{
public:
  PressureNeumannBCViscousTermData() : dof_index_velocity(0), dof_index_pressure(0), quad_index(0)
  {
  }

  unsigned int dof_index_velocity;
  unsigned int dof_index_pressure;

  unsigned int quad_index;

  std::shared_ptr<BoundaryDescriptorP<dim>> bc;
};

template<int dim, int degree_u, int degree_p, typename Number>
class PressureNeumannBCViscousTerm
{
public:
  typedef PressureNeumannBCViscousTerm<dim, degree_u, degree_p, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef FEFaceEvaluation<dim, degree_u, degree_u + 1, dim, Number> FEFaceEvalVelocity;
  typedef FEFaceEvaluation<dim, degree_p, degree_u + 1, 1, Number>   FEFaceEvalPressure;

  PressureNeumannBCViscousTerm() : data(nullptr), viscous_operator(nullptr)
  {
  }

  void
  initialize(MatrixFree<dim, Number> const &                data,
             PressureNeumannBCViscousTermData<dim> &        operator_data,
             ViscousOperator<dim, degree_u, Number> const & viscous_operator)
  {
    this->data             = &data;
    this->operator_data    = operator_data;
    this->viscous_operator = &viscous_operator;
  }

  void
  calculate(VectorType & dst, VectorType const & src) const
  {
    this->data->loop(&This::cell_loop, &This::face_loop, &This::boundary_face_loop, this, dst, src);
  }

private:
  void
  cell_loop(MatrixFree<dim, Number> const &, VectorType &, VectorType const &, Range const &) const
  {
  }

  void
  face_loop(MatrixFree<dim, Number> const &, VectorType &, VectorType const &, Range const &) const
  {
  }

  void
  boundary_face_loop(MatrixFree<dim, Number> const & data,
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

  MatrixFree<dim, Number> const *                data;
  PressureNeumannBCViscousTermData<dim>          operator_data;
  ViscousOperator<dim, degree_u, Number> const * viscous_operator;
};

template<int dim>
class VelocityDivergenceConvectiveTermData
{
public:
  VelocityDivergenceConvectiveTermData()
    : dof_index_velocity(0), dof_index_pressure(0), quad_index(0)
  {
  }

  unsigned int dof_index_velocity;
  unsigned int dof_index_pressure;

  unsigned int quad_index;

  std::shared_ptr<BoundaryDescriptorU<dim>> bc;
};

template<int dim, int degree_u, int degree_p, typename Number>
class VelocityDivergenceConvectiveTerm
{
public:
  typedef VelocityDivergenceConvectiveTerm<dim, degree_u, degree_p, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;
  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  typedef std::pair<unsigned int, unsigned int> Range;

  static const unsigned int n_q_points_overint = degree_u + (degree_u + 2) / 2;

  typedef FEFaceEvaluation<dim, degree_u, n_q_points_overint, dim, Number> FEFaceEvalVelocity;
  typedef FEFaceEvaluation<dim, degree_p, n_q_points_overint, 1, Number>   FEFaceEvalPressure;

  VelocityDivergenceConvectiveTerm() : data(nullptr)
  {
  }

  void
  initialize(MatrixFree<dim, Number> const &             data,
             VelocityDivergenceConvectiveTermData<dim> & operator_data)
  {
    this->data          = &data;
    this->operator_data = operator_data;
  }

  void
  calculate(VectorType & dst, VectorType const & src) const
  {
    this->data->loop(&This::cell_loop, &This::face_loop, &This::boundary_face_loop, this, dst, src);
  }

private:
  void
  cell_loop(MatrixFree<dim, Number> const &, VectorType &, VectorType const &, Range const &) const
  {
  }

  void
  face_loop(MatrixFree<dim, Number> const &, VectorType &, VectorType const &, Range const &) const
  {
  }

  void
  boundary_face_loop(MatrixFree<dim, Number> const & data,
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

  MatrixFree<dim, Number> const * data;

  VelocityDivergenceConvectiveTermData<dim> operator_data;
};

} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DUAL_SPLITTING_BOUNDARY_CONDITIONS_H_ \
        */
