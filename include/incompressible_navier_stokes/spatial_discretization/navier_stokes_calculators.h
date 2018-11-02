/*
 * navier_stokes_calculators.h
 *
 *  Created on: Oct 28, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_NAVIER_STOKES_CALCULATORS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_NAVIER_STOKES_CALCULATORS_H_

#include <deal.II/matrix_free/fe_evaluation.h>

namespace IncNS
{
template<int dim, int degree, typename Number>
class VorticityCalculator
{
public:
  typedef VorticityCalculator<dim, degree, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  static const unsigned int number_vorticity_components = (dim == 2) ? 1 : dim;

  typedef FEEvaluation<dim, degree, degree + 1, dim, Number> FEEval;

  VorticityCalculator() : data(nullptr), dof_index(0), quad_index(0){};

  void
  initialize(MatrixFree<dim, Number> const & data_in,
             unsigned int const              dof_index_in,
             unsigned int const              quad_index_in)
  {
    this->data = &data_in;
    dof_index  = dof_index_in;
    quad_index = quad_index_in;
  }

  void
  compute_vorticity(VectorType & dst, VectorType const & src) const
  {
    dst = 0;

    data->cell_loop(&This::local_compute_vorticity, this, dst, src);
  }

private:
  void
  local_compute_vorticity(MatrixFree<dim, Number> const & data,
                          VectorType &                    dst,
                          VectorType const &              src,
                          Range const &                   cell_range) const
  {
    FEEval fe_eval_velocity(data, dof_index, quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit(cell);
      fe_eval_velocity.gather_evaluate(src, false, true, false);

      for(unsigned int q = 0; q < fe_eval_velocity.n_q_points; ++q)
      {
        // omega is a scalar quantity in 2D and a vector with dim components in 3D
        Tensor<1, number_vorticity_components, VectorizedArray<Number>> omega =
          fe_eval_velocity.get_curl(q);

        // omega_vector is a vector with dim components
        // for dim=3: omega_vector[i] = omega[i], i=1,...,dim
        // for dim=2: omega_vector[0] = omega,
        //            omega_vector[1] = 0
        vector omega_vector;
        for(unsigned int d = 0; d < number_vorticity_components; ++d)
          omega_vector[d] = omega[d];

        fe_eval_velocity.submit_value(omega_vector, q);
      }

      fe_eval_velocity.integrate_scatter(true, false, dst);
    }
  }

  MatrixFree<dim, Number> const * data;

  unsigned int dof_index;
  unsigned int quad_index;
};

template<int dim, int degree, typename Number>
class DivergenceCalculator
{
public:
  typedef DivergenceCalculator<dim, degree, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number> scalar;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef FEEvaluation<dim, degree, degree + 1, dim, Number> FEEval;
  typedef FEEvaluation<dim, degree, degree + 1, 1, Number>   FEEvalScalar;

  DivergenceCalculator() : data(nullptr), dof_index_u(0), dof_index_u_scalar(0), quad_index(0){};

  void
  initialize(MatrixFree<dim, Number> const & data_in,
             unsigned int const              dof_index_u_in,
             unsigned int const              dof_index_u_scalar_in,
             unsigned int const              quad_index_in)
  {
    this->data         = &data_in;
    dof_index_u        = dof_index_u_in;
    dof_index_u_scalar = dof_index_u_scalar_in;
    quad_index         = quad_index_in;
  }

  void
  compute_divergence(VectorType & dst, VectorType const & src) const
  {
    dst = 0;

    data->cell_loop(&This::local_compute_divergence, this, dst, src);
  }

private:
  void
  local_compute_divergence(MatrixFree<dim, Number> const & data,
                           VectorType &                    dst,
                           VectorType const &              src,
                           Range const &                   cell_range) const
  {
    FEEval       fe_eval_velocity(data, dof_index_u, quad_index);
    FEEvalScalar fe_eval_velocity_scalar(data, dof_index_u_scalar, quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit(cell);
      fe_eval_velocity.gather_evaluate(src, false, true);

      fe_eval_velocity_scalar.reinit(cell);

      for(unsigned int q = 0; q < fe_eval_velocity_scalar.n_q_points; q++)
      {
        scalar div = fe_eval_velocity.get_divergence(q);
        fe_eval_velocity_scalar.submit_value(div, q);
      }

      fe_eval_velocity_scalar.integrate_scatter(true, false, dst);
    }
  }

  MatrixFree<dim, Number> const * data;

  unsigned int dof_index_u;
  unsigned int dof_index_u_scalar;
  unsigned int quad_index;
};

template<int dim, int degree, typename Number>
class VelocityMagnitudeCalculator
{
public:
  typedef VelocityMagnitudeCalculator<dim, degree, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number> scalar;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef FEEvaluation<dim, degree, degree + 1, dim, Number> FEEval;
  typedef FEEvaluation<dim, degree, degree + 1, 1, Number>   FEEvalScalar;

  VelocityMagnitudeCalculator()
    : data(nullptr), dof_index_u(0), dof_index_u_scalar(0), quad_index(0){};

  void
  initialize(MatrixFree<dim, Number> const & data_in,
             unsigned int const              dof_index_u_in,
             unsigned int const              dof_index_u_scalar_in,
             unsigned int const              quad_index_in)
  {
    this->data         = &data_in;
    dof_index_u        = dof_index_u_in;
    dof_index_u_scalar = dof_index_u_scalar_in;
    quad_index         = quad_index_in;
  }

  void
  compute(VectorType & dst, VectorType const & src) const
  {
    dst = 0;

    data->cell_loop(&This::local_compute, this, dst, src);
  }

private:
  void
  local_compute(MatrixFree<dim, Number> const & data,
                VectorType &                    dst,
                VectorType const &              src,
                Range const &                   cell_range) const
  {
    FEEval       fe_eval_velocity(data, dof_index_u, quad_index);
    FEEvalScalar fe_eval_velocity_scalar(data, dof_index_u_scalar, quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit(cell);
      fe_eval_velocity.gather_evaluate(src, true, false);

      fe_eval_velocity_scalar.reinit(cell);

      for(unsigned int q = 0; q < fe_eval_velocity_scalar.n_q_points; q++)
      {
        scalar magnitude = fe_eval_velocity.get_value(q).norm();
        fe_eval_velocity_scalar.submit_value(magnitude, q);
      }
      fe_eval_velocity_scalar.integrate_scatter(true, false, dst);
    }
  }

  MatrixFree<dim, Number> const * data;

  unsigned int dof_index_u;
  unsigned int dof_index_u_scalar;
  unsigned int quad_index;
};

template<int dim, int degree, typename Number>
class QCriterionCalculator
{
public:
  typedef QCriterionCalculator<dim, degree, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef FEEvaluation<dim, degree, degree + 1, dim, Number> FEEval;
  typedef FEEvaluation<dim, degree, degree + 1, 1, Number>   FEEvalScalar;

  QCriterionCalculator() : data(nullptr), dof_index_u(0), dof_index_u_scalar(0), quad_index(0){};

  void
  initialize(MatrixFree<dim, Number> const & data_in,
             unsigned int const              dof_index_u_in,
             unsigned int const              dof_index_u_scalar_in,
             unsigned int const              quad_index_in)
  {
    this->data         = &data_in;
    dof_index_u        = dof_index_u_in;
    dof_index_u_scalar = dof_index_u_scalar_in;
    quad_index         = quad_index_in;
  }

  void
  compute(VectorType & dst, VectorType const & src) const
  {
    dst = 0;

    data->cell_loop(&This::local_compute, this, dst, src);
  }

private:
  void
  local_compute(MatrixFree<dim, Number> const & data,
                VectorType &                    dst,
                VectorType const &              src,
                Range const &                   cell_range) const
  {
    FEEval       fe_eval_velocity(data, dof_index_u, quad_index);
    FEEvalScalar fe_eval_velocity_scalar(data, dof_index_u_scalar, quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit(cell);
      fe_eval_velocity.gather_evaluate(src, false, true);

      fe_eval_velocity_scalar.reinit(cell);

      for(unsigned int q = 0; q < fe_eval_velocity_scalar.n_q_points; q++)
      {
        tensor gradu = fe_eval_velocity.get_gradient(q);
        tensor Om, S;
        for(unsigned int i = 0; i < dim; i++)
        {
          for(unsigned int j = 0; j < dim; j++)
          {
            Om[i][j] = 0.5 * (gradu[i][j] - gradu[j][i]);
            S[i][j]  = 0.5 * (gradu[i][j] + gradu[j][i]);
          }
        }

        scalar const Q = 0.5 * (Om.norm_square() - S.norm_square());
        fe_eval_velocity_scalar.submit_value(Q, q);
      }
      fe_eval_velocity_scalar.integrate_scatter(true, false, dst);
    }
  }

  MatrixFree<dim, Number> const * data;

  unsigned int dof_index_u;
  unsigned int dof_index_u_scalar;
  unsigned int quad_index;
};

/*
 *  This function calculates the right-hand side of the Laplace equation that is solved in order to
 * obtain the streamfunction psi
 *
 *    - laplace(psi) = omega  (where omega is the vorticity).
 *
 *  Note that this function can only be used for the two-dimensional case (dim==2).
 */
template<int dim, int degree, typename Number>
class StreamfunctionCalculatorRHSOperator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef StreamfunctionCalculatorRHSOperator<dim, degree, Number> This;

  typedef FEEvaluation<dim, degree, degree + 1, dim, Number> FEEval;
  typedef FEEvaluation<dim, degree, degree + 1, 1, Number>   FEEvalScalar;

  StreamfunctionCalculatorRHSOperator()
    : data(nullptr), dof_index_u(0), dof_index_u_scalar(0), quad_index(0)
  {
    AssertThrow(dim == 2, ExcMessage("Calculation of streamfunction can only be used for dim==2."));
  }

  void
  initialize(MatrixFree<dim, Number> const & data_in,
             unsigned int const              dof_index_u_in,
             unsigned int const              dof_index_u_scalar_in,
             unsigned int const              quad_index_in)
  {
    this->data         = &data_in;
    dof_index_u        = dof_index_u_in;
    dof_index_u_scalar = dof_index_u_scalar_in;
    quad_index         = quad_index_in;
  }

  void
  apply(VectorType & dst, VectorType const & src) const
  {
    dst = 0;

    data->cell_loop(&This::local_apply, this, dst, src);
  }

private:
  void
  local_apply(MatrixFree<dim, Number> const & data,
              VectorType &                    dst,
              VectorType const &              src,
              Range const &                   cell_range) const
  {
    FEEval       fe_eval_velocity(data, dof_index_u, quad_index);
    FEEvalScalar fe_eval_velocity_scalar(data, dof_index_u_scalar, quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit(cell);
      fe_eval_velocity.gather_evaluate(src, true, false);

      fe_eval_velocity_scalar.reinit(cell);

      for(unsigned int q = 0; q < fe_eval_velocity_scalar.n_q_points; q++)
      {
        // we exploit that the (scalar) vorticity is stored in the first component of the vector
        // in case of 2D problems
        fe_eval_velocity_scalar.submit_value(fe_eval_velocity.get_value(q)[0], q);
      }
      fe_eval_velocity_scalar.integrate_scatter(true, false, dst);
    }
  }

  MatrixFree<dim, Number> const * data;

  unsigned int dof_index_u;
  unsigned int dof_index_u_scalar;
  unsigned int quad_index;
};


} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_NAVIER_STOKES_CALCULATORS_H_ \
        */
