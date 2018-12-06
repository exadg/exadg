/*
 * pressure_neumann_bc_viscous_term.h
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_BOUNDARY_CONDITIONS_DUAL_SPLITTING_PRESSURE_NEUMANN_BC_VISCOUS_TERM_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_BOUNDARY_CONDITIONS_DUAL_SPLITTING_PRESSURE_NEUMANN_BC_VISCOUS_TERM_H_

#include <deal.II/matrix_free/fe_evaluation.h>

#include "../../user_interface/boundary_descriptor.h"
#include "../curl_compute.h"
#include "../operators/viscous_operator.h"

using namespace dealii;

namespace IncNS
{
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
private:
  typedef PressureNeumannBCViscousTerm<dim, degree_u, degree_p, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef FEFaceEvaluation<dim, degree_u, degree_u + 1, dim, Number> FEFaceEvalVelocity;
  typedef FEFaceEvaluation<dim, degree_p, degree_u + 1, 1, Number>   FEFaceEvalPressure;

public:
  PressureNeumannBCViscousTerm();

  void
  initialize(MatrixFree<dim, Number> const &                data,
             PressureNeumannBCViscousTermData<dim> &        operator_data,
             ViscousOperator<dim, degree_u, Number> const & viscous_operator);

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

  MatrixFree<dim, Number> const *                data;
  PressureNeumannBCViscousTermData<dim>          operator_data;
  ViscousOperator<dim, degree_u, Number> const * viscous_operator;
};

} // namespace IncNS



#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_BOUNDARY_CONDITIONS_DUAL_SPLITTING_PRESSURE_NEUMANN_BC_VISCOUS_TERM_H_ \
        */
