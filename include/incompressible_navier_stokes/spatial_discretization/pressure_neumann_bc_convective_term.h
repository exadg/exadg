/*
 * pressure_neumann_bc_convective_term.h
 *
 *  Created on: Nov 14, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PRESSURE_NEUMANN_BC_CONVECTIVE_TERM_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PRESSURE_NEUMANN_BC_CONVECTIVE_TERM_H_

#include <deal.II/matrix_free/fe_evaluation.h>

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


} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PRESSURE_NEUMANN_BC_CONVECTIVE_TERM_H_ \
        */
