/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_pressure_correction.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
OperatorPressureCorrection<dim, Number>::OperatorPressureCorrection(
  std::shared_ptr<Grid<dim> const>                      grid_in,
  std::shared_ptr<dealii::Mapping<dim> const>           mapping_in,
  std::shared_ptr<MultigridMappings<dim, Number>> const multigrid_mappings_in,
  std::shared_ptr<BoundaryDescriptor<dim> const>        boundary_descriptor_in,
  std::shared_ptr<FieldFunctions<dim> const>            field_functions_in,
  Parameters const &                                    parameters_in,
  std::string const &                                   field_in,
  MPI_Comm const &                                      mpi_comm_in)
  : ProjectionBase(grid_in,
                   mapping_in,
                   multigrid_mappings_in,
                   boundary_descriptor_in,
                   field_functions_in,
                   parameters_in,
                   field_in,
                   mpi_comm_in)
{
}

template<int dim, typename Number>
OperatorPressureCorrection<dim, Number>::~OperatorPressureCorrection()
{
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::setup_derived()
{
  ProjectionBase::setup_derived();

  setup_inverse_mass_operator_pressure();
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::update_after_grid_motion(bool const update_matrix_free)
{
  ProjectionBase::update_after_grid_motion(update_matrix_free);

  // The inverse mass operator might contain matrix-based components, in which cases it needs to be
  // updated after the grid has been deformed.
  inverse_mass_pressure.update();
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::setup_inverse_mass_operator_pressure()
{
  // inverse mass operator pressure (needed for pressure update in case of rotational
  // formulation)
  InverseMassOperatorData inverse_mass_operator_data_pressure;
  inverse_mass_operator_data_pressure.dof_index  = this->get_dof_index_pressure();
  inverse_mass_operator_data_pressure.quad_index = this->get_quad_index_pressure();
  inverse_mass_operator_data_pressure.parameters = this->param.inverse_mass_operator;

  inverse_mass_pressure.initialize(this->get_matrix_free(), inverse_mass_operator_data_pressure);
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::evaluate_nonlinear_residual_steady(
  VectorType &       dst_u,
  VectorType &       dst_p,
  VectorType const & src_u,
  VectorType const & src_p,
  double const &     time) const
{
  // update implicitly coupled viscosity
  if(this->param.nonlinear_viscous_problem())
  {
    this->update_viscosity(src_u);
  }

  // velocity-block

  // set dst_u to zero. This is necessary since subsequent operators
  // call functions of type ..._add
  dst_u = 0.0;

  if(this->param.right_hand_side == true)
  {
    this->rhs_operator.evaluate(dst_u, time);
    // Shift body force term to the left-hand side of the equation.
    // This works since body_force_operator is the first operator
    // that is evaluated.
    dst_u *= -1.0;
  }

  if(this->param.implicit_convective_problem())
  {
    this->convective_operator.evaluate_nonlinear_operator_add(dst_u, src_u, time);
  }

  if(this->param.viscous_problem())
  {
    this->viscous_operator.set_time(time);
    this->viscous_operator.evaluate_add(dst_u, src_u);
  }

  // gradient operator scaled by scaling_factor_continuity
  this->gradient_operator.evaluate_add(dst_u, src_p, time);

  // pressure-block

  this->divergence_operator.evaluate(dst_p, src_u, time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  dst_p *= -1.0;
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::apply_momentum_operator(VectorType &       dst,
                                                                 VectorType const & src)
{
  this->momentum_operator.apply(dst, src);
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::rhs_pressure_gradient_term_dirichlet_bc_from_dof_vector(
  VectorType &       dst,
  VectorType const & pressure) const
{
  this->gradient_operator.rhs_bc_from_dof_vector(dst, pressure);
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::
  evaluate_pressure_gradient_term_dirichlet_bc_from_dof_vector(VectorType &       dst,
                                                               VectorType const & src,
                                                               VectorType const & pressure) const
{
  this->gradient_operator.evaluate_bc_from_dof_vector(dst, src, pressure);
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::apply_inverse_pressure_mass_operator(
  VectorType &       dst,
  VectorType const & src) const
{
  inverse_mass_pressure.apply(dst, src);
}

template<int dim, typename Number>
unsigned int
OperatorPressureCorrection<dim, Number>::solve_pressure(VectorType &       dst,
                                                        VectorType const & src,
                                                        bool const update_preconditioner) const
{
  return ProjectionBase::do_solve_pressure(dst, src, update_preconditioner);
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::rhs_ppe_laplace_add(VectorType &   dst,
                                                             double const & time) const
{
  ProjectionBase::do_rhs_ppe_laplace_add(dst, time);
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::rhs_ppe_laplace_add_dirichlet_bc_from_dof_vector(
  VectorType &       dst,
  VectorType const & src) const
{
  this->laplace_operator.rhs_add_dirichlet_bc_from_dof_vector(dst, src);
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::interpolate_pressure_dirichlet_bc(
  VectorType &   dst,
  double const & time) const
{
  this->evaluation_time = time;

  dst = 0.0;

  VectorType src_dummy;
  this->get_matrix_free().loop(&This::cell_loop_empty,
                               &This::face_loop_empty,
                               &This::local_interpolate_pressure_dirichlet_bc_boundary_face,
                               this,
                               dst,
                               src_dummy);
}

template<int dim, typename Number>
void
OperatorPressureCorrection<dim, Number>::local_interpolate_pressure_dirichlet_bc_boundary_face(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &,
  Range const & face_range) const
{
  unsigned int const dof_index = this->get_dof_index_pressure();
  unsigned int const quad_index = this->get_quad_index_pressure_nodal_points();

  FaceIntegratorP integrator(matrix_free, true, dof_index, quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    dealii::types::boundary_id const boundary_id = matrix_free.get_boundary_id(face);

    BoundaryTypeP const boundary_type =
      this->boundary_descriptor->pressure->get_boundary_type(boundary_id);

    if(boundary_type == BoundaryTypeP::Dirichlet)
    {
      integrator.reinit(face);
      integrator.read_dof_values(dst);

      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        unsigned int const local_face_number = matrix_free.get_face_info(face).interior_face_no;

        unsigned int const index = matrix_free.get_shape_info(dof_index, quad_index)
                                     .face_to_cell_index_nodal[local_face_number][q];

        auto bc       = this->boundary_descriptor->pressure->dirichlet_bc.find(boundary_id)->second;
        auto q_points = integrator.quadrature_point(q);

        scalar g = FunctionEvaluator<0, dim, Number>::value(*bc, q_points, this->evaluation_time);
        integrator.submit_dof_value(g, index);
      }

      integrator.set_dof_values(dst);
    }
    else
    {
      AssertThrow(boundary_type == BoundaryTypeP::Neumann,
                  dealii::ExcMessage("BoundaryTypeP not implemented."));
    }
  }
}

template class OperatorPressureCorrection<2, float>;
template class OperatorPressureCorrection<2, double>;

template class OperatorPressureCorrection<3, float>;
template class OperatorPressureCorrection<3, double>;

} // namespace IncNS
} // namespace ExaDG
