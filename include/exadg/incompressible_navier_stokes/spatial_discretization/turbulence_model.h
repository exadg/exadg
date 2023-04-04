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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_TURBULENCE_MODEL_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_TURBULENCE_MODEL_H_

// ExaDG
#include <exadg/incompressible_navier_stokes/spatial_discretization/viscosity_model_base.h>

namespace ExaDG
{
namespace IncNS
{
/*
 *  Turbulence model.
 */
template<int dim, typename Number>
class TurbulenceModel : public ViscosityModelBase<dim, Number>
{
private:
  typedef TurbulenceModel<dim, Number>    This;
  typedef ViscosityModelBase<dim, Number> Base;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> tensor;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorU;
  typedef FaceIntegrator<dim, dim, Number> FaceIntegratorU;

public:
  /*
   *  Constructor.
   */
  TurbulenceModel();

  /*
   * Destructor.
   */
  virtual ~TurbulenceModel();

  /*
   * Initialization function.
   */
  void
  initialize(dealii::MatrixFree<dim, Number> const &                matrix_free_in,
             dealii::Mapping<dim> const &                           mapping_in,
             std::shared_ptr<Operators::ViscousKernel<dim, Number>> viscous_kernel_in,
             TurbulenceModelData const &                            turbulence_model_data_in,
             unsigned int                                           dof_index_velocity_in,
             unsigned int                                           quad_index_velocity_linear_in,
             unsigned int                                           degree_velocity_in);

  /*
   *  Function for *setting* the viscosity to viscosity_newtonian_limit.
   */
  void
  set_viscosity(VectorType const & velocity) const;

  /*
   *  Function for *adding to* the viscosity taking the currently stored viscosity as a basis.
   */
  void
  add_viscosity(VectorType const & velocity) const;

  /*
   *  This function calculates the filter width for each cell.
   */
  void
  calculate_filter_width(dealii::Mapping<dim> const & mapping);

private:
  void
  cell_loop_set_coefficients(dealii::MatrixFree<dim, Number> const & data,
                             VectorType &,
                             VectorType const & src,
                             Range const &      cell_range) const;

  void
  face_loop_set_coefficients(dealii::MatrixFree<dim, Number> const & data,
                             VectorType &,
                             VectorType const & src,
                             Range const &      face_range) const;

  void
  boundary_face_loop_set_coefficients(dealii::MatrixFree<dim, Number> const & data,
                                      VectorType &,
                                      VectorType const & src,
                                      Range const &      face_range) const;

  /*
   *  This function adds the turbulent eddy-viscosity to the laminar viscosity
   *  by using one of the implemented models.
   */
  void
  add_turbulent_viscosity(scalar &       viscosity,
                          scalar const & filter_width,
                          tensor const & velocity_gradient,
                          double const & model_constant) const;

  /*
   *  Smagorinsky model (1963):
   *
   *    nu_SGS = (C * filter_width)^{2} * sqrt(2 * S:S)
   *
   *    where S is the symmetric part of the velocity gradient
   *
   *      S = 1/2 * (grad(u) + grad(u)^T) and S:S = S_ij * S_ij
   *
   *    and the model constant is
   *
   *      C = 0.165 (Nicoud et al. (2011))
   *      C = 0.18  (Toda et al. (2010))
   */
  void
  smagorinsky_model(scalar const & filter_width,
                    tensor const & velocity_gradient,
                    double const & C,
                    scalar &       viscosity) const;

  /*
   *  Vreman model (2004): Note that we only consider the isotropic variant of the Vreman model:
   *
   *    nu_SGS = (C * filter_width)^{2} * D
   *
   *  where the differential operator D is defined as
   *
   *    D = sqrt(B_gamma / ||grad(u)||^{2})
   *
   *  with
   *
   *    ||grad(u)||^{2} = grad(u) : grad(u) and grad(u) = d(u_i)/d(x_j) ,
   *
   *    gamma = grad(u) * grad(u)^T ,
   *
   *  and
   *
   *    B_gamma = gamma_11 * gamma_22 - gamma_12^{2}
   *             +gamma_11 * gamma_33 - gamma_13^{2}
   *             +gamma_22 * gamma_33 - gamma_23^{2}
   *
   *  Note that if ||grad(u)||^{2} = 0, nu_SGS is consistently defined as zero.
   *
   */
  void
  vreman_model(scalar const & filter_width,
               tensor const & velocity_gradient,
               double const & C,
               scalar &       viscosity) const;

  /*
   *  WALE (wall-adapting local eddy-viscosity) model (Nicoud & Ducros 1999):
   *
   *    nu_SGS = (C * filter_width)^{2} * D ,
   *
   *  where the differential operator D is defined as
   *
   *    D = (S^{d}:S^{d})^{3/2} / ( (S:S)^{5/2} + (S^{d}:S^{d})^{5/4} )
   *
   *    where S is the symmetric part of the velocity gradient
   *
   *      S = 1/2 * (grad(u) + grad(u)^T) and S:S = S_ij * S_ij
   *
   *    and S^{d} the traceless symmetric part of the square of the velocity
   *    gradient tensor
   *
   *      S^{d} = 1/2 * (g^{2} + (g^{2})^T) - 1/3 * trace(g^{2}) * I
   *
   *    with the square of the velocity gradient tensor
   *
   *      g^{2} = grad(u) * grad(u)
   *
   *    and the identity tensor I.
   *
   */
  void
  wale_model(scalar const & filter_width,
             tensor const & velocity_gradient,
             double const & C,
             scalar &       viscosity) const;

  /*
   *  Sigma model (Toda et al. 2010, Nicoud et al. 2011):
   *
   *    nu_SGS = (C * filter_width)^{2} * D
   *
   *    where the differential operator D is defined as
   *
   *      D = s3 * (s1 - s2) * (s2 - s3) / s1^{2}
   *
   *    where s1 >= s2 >= s3 >= 0 are the singular values of
   *    the velocity gradient tensor g = grad(u).
   *
   *    The model constant is
   *
   *      C = 1.35 (Nicoud et al. (2011)) ,
   *      C = 1.5  (Toda et al. (2010)) .
   */
  void
  sigma_model(scalar const & filter_width,
              tensor const & velocity_gradient,
              double const & C,
              scalar &       viscosity) const;

  unsigned int                  degree_velocity;
  TurbulenceModelData           turbulence_model_data;
  dealii::AlignedVector<scalar> filter_width_vector;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_TURBULENCE_MODEL_H_ */
