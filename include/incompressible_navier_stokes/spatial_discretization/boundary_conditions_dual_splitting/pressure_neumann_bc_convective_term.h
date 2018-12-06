/*
 * pressure_neumann_bc_convective_term.h
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_BOUNDARY_CONDITIONS_DUAL_SPLITTING_PRESSURE_NEUMANN_BC_CONVECTIVE_TERM_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_BOUNDARY_CONDITIONS_DUAL_SPLITTING_PRESSURE_NEUMANN_BC_CONVECTIVE_TERM_H_

#include <deal.II/matrix_free/fe_evaluation.h>

#include "../../user_interface/boundary_descriptor.h"

using namespace dealii;

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
private:
  typedef PressureNeumannBCConvectiveTerm<dim, degree_u, degree_p, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;
  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  typedef std::pair<unsigned int, unsigned int> Range;

  static const unsigned int n_q_points_overint = degree_u + (degree_u + 2) / 2;

  typedef FEFaceEvaluation<dim, degree_u, n_q_points_overint, dim, Number> FEFaceEvalVelocity;
  typedef FEFaceEvaluation<dim, degree_p, n_q_points_overint, 1, Number>   FEFaceEvalPressure;

public:
  PressureNeumannBCConvectiveTerm();

  void
  initialize(MatrixFree<dim, Number> const &            data,
             PressureNeumannBCConvectiveTermData<dim> & operator_data);

  void
  calculate(VectorType & dst, VectorType const & src) const;

private:
  void
  cell_loop(MatrixFree<dim, Number> const &, VectorType &, VectorType const &, Range const &) const;

  void
  face_loop(MatrixFree<dim, Number> const &, VectorType &, VectorType const &, Range const &) const;

  void
  boundary_face_loop(MatrixFree<dim, Number> const & data,
                     VectorType &                    dst,
                     VectorType const &              src,
                     Range const &                   face_range) const;

  MatrixFree<dim, Number> const *          data;
  PressureNeumannBCConvectiveTermData<dim> operator_data;
};

} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_BOUNDARY_CONDITIONS_DUAL_SPLITTING_PRESSURE_NEUMANN_BC_CONVECTIVE_TERM_H_ \
        */
