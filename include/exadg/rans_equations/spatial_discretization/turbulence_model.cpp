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

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <exadg/rans_equations/spatial_discretization/turbulence_model.h>
#include <exadg/operators/quadrature.h>
#include <cmath>
#include <memory>
#include "exadg/rans_equations/user_interface/enum_types.h"
#include "exadg/rans_equations/user_interface/viscosity_model_data.h"

namespace ExaDG
{
namespace RANS
{
template<int dim, typename Number>
TurbulenceModel<dim, Number>::TurbulenceModel()
{
}

template<int dim, typename Number>
TurbulenceModel<dim, Number>::~TurbulenceModel()
{
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &                matrix_free_in,
  TurbulenceModelData const &                            turbulence_model_data_in)
{
  Base::initialize(matrix_free_in, dof_index_eddy_viscosity, this->quad_index);

  turbulence_model_data = turbulence_model_data_in;

  turbulence_model_data.check();

  viscosity_coefficients.initialize(matrix_free_in, this->quad_index, true, true);
  viscosity_coefficients.set_coefficients(diffusivity);
  eddy_viscosity_coefficients.initialize(matrix_free_in, this->quad_index, true, true);
  eddy_viscosity_coefficients.set_coefficients(1.0);

  this->matrix_free->initialize_dof_vector(eddy_viscosity, dof_index_eddy_viscosity);

  /*model_coefficients = this->turbulence_data_base->get_all_coefficients();*/
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::set_viscosity(VectorType const & solution) 
{
  this->set_constant_coefficient(this->diffusivity);

  this->add_viscosity(solution);
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::add_viscosity(VectorType const & solution) 
{
  VectorType dummy;

  this->matrix_free->loop(&This::cell_loop_set_coefficients,
                          &This::face_loop_set_coefficients,
                          &This::boundary_face_loop_set_coefficients,
                          this,
                          dummy,
                          this->eddy_viscosity);
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::cell_loop_set_coefficients(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &,
  VectorType const & src,
  Range const &      cell_range) 
{
  IntegratorCell integrator(matrix_free,
                            dof_index_eddy_viscosity,
                            this->quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);
    integrator.read_dof_values(src);
    integrator.evaluate(dealii::EvaluationFlags::values);

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // get the current viscosity
      scalar current_viscosity = this->diffusivity;
      scalar viscosity = integrator.get_value(q);

      eddy_viscosity_coefficients.set_coefficient_cell(cell, q, viscosity);
      if (scalar_type==ScalarType::TurbulentKineticEnergy) {
        viscosity = current_viscosity + (viscosity / model_coefficients[0]);
      }
      else if (scalar_type==ScalarType::TKEDissipationRate) {
        viscosity = current_viscosity + (viscosity / model_coefficients[4]);
      }
      // set the coefficients
      viscosity_coefficients.set_coefficient_cell(cell, q, viscosity);
    }
  }
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::face_loop_set_coefficients(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &,
  VectorType const & src,
  Range const &      face_range) 
{
  IntegratorFace integrator_m(matrix_free,
                              true,
                              dof_index_eddy_viscosity,
                              this->quad_index);
  IntegratorFace integrator_p(matrix_free,
                              false,
                              dof_index_eddy_viscosity,
                              this->quad_index);
  // loop over all interior faces
  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator_m.reinit(face);
    integrator_p.reinit(face);

    integrator_m.read_dof_values(src);
    integrator_p.read_dof_values(src);

    // we only need the gradient
    integrator_m.evaluate(dealii::EvaluationFlags::values);
    integrator_p.evaluate(dealii::EvaluationFlags::values);

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      // calculate velocity gradient for both elements adjacent to the current face
      scalar viscosity          = integrator_m.get_value(q);
      scalar viscosity_neighbor = integrator_p.get_value(q);

      // get the coefficients
      scalar current_viscosity          = this->diffusivity;
      scalar current_viscosity_neighbor = this->diffusivity;

      eddy_viscosity_coefficients.set_coefficient_face(face, q, viscosity);
      eddy_viscosity_coefficients.set_coefficient_face_neighbor(face, q, viscosity_neighbor);

      if (scalar_type==ScalarType::TurbulentKineticEnergy) {
        viscosity = current_viscosity + (viscosity / model_coefficients[0]);
        viscosity_neighbor = current_viscosity_neighbor + (viscosity_neighbor / model_coefficients[0]);
      }
      else if (scalar_type==ScalarType::TKEDissipationRate) {
        viscosity = current_viscosity + (viscosity / model_coefficients[4]);
        viscosity_neighbor = current_viscosity_neighbor + (viscosity_neighbor / model_coefficients[4]);
      }
    }
  }
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::boundary_face_loop_set_coefficients(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &,
  VectorType const & src,
  Range const &      face_range) 
{
  IntegratorFace integrator(matrix_free,
                            true,
                            dof_index_eddy_viscosity,
                            this->quad_index);
  IntegratorFace integrator_secondary(matrix_free,
                                      true,
                                      dof_index_eddy_viscosity,
                                      this->quad_index);

  // loop over all boundary faces
  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator.reinit(face);
    integrator.read_dof_values(src);

    // we only need the gradient
    integrator.evaluate(dealii::EvaluationFlags::values);

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // calculate velocity gradient
      scalar viscosity = integrator.get_value(q);

      // get the coefficients
      scalar current_viscosity = this->diffusivity;

      eddy_viscosity_coefficients.set_coefficient_face(face, q, viscosity);

      if (scalar_type==ScalarType::TurbulentKineticEnergy) {
        viscosity = current_viscosity + (viscosity / model_coefficients[0]);
      }
      else if (scalar_type==ScalarType::TKEDissipationRate) {
        viscosity = current_viscosity + (viscosity / model_coefficients[4]);
      }
    }
  }
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::update_eddy_viscosity(VectorType const & viscosity_in)
{
  this->eddy_viscosity = viscosity_in;
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::get_eddy_viscosity(VectorType & dst)
{
  dst.equ(1.0, this->eddy_viscosity);
}

template class TurbulenceModel<2, float>;
template class TurbulenceModel<2, double>;
template class TurbulenceModel<3, float>;
template class TurbulenceModel<3, double>;

} // namespace RANSEqns
} // namespace ExaDG
