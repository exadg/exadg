/*
 * turbulence_model.h
 *
 *  Created on: Apr 4, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_TURBULENCE_MODEL_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_TURBULENCE_MODEL_H_

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "../user_interface/input_parameters.h"
#include "operators/viscous_operator.h"

using namespace dealii;

namespace IncNS
{
/*
 *  Turbulence model data.
 */
struct TurbulenceModelData
{
  TurbulenceModelData()
    : turbulence_model(TurbulenceEddyViscosityModel::Undefined),
      constant(1.0),
      kinematic_viscosity(1.0),
      dof_index(0),
      quad_index(0),
      degree(1)
  {
  }

  TurbulenceEddyViscosityModel turbulence_model;
  double                       constant;

  // constant kinematic viscosity (physical viscosity)
  double kinematic_viscosity;

  // required for matrix-free loops
  unsigned int dof_index;
  unsigned int quad_index;

  // required for calculation of filter width
  unsigned int degree;
};


/*
 *  Algebraic subgrid-scale turbulence models for LES of incompressible flows.
 */
template<int dim, typename Number>
class TurbulenceModel
{
private:
  typedef TurbulenceModel<dim, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorU;
  typedef FaceIntegrator<dim, dim, Number> FaceIntegratorU;

public:
  /*
   *  Constructor.
   */
  TurbulenceModel();

  /*
   * Initialization function.
   */
  void
  initialize(MatrixFree<dim, Number> const &                        matrix_free_in,
             Mapping<dim> const &                                   mapping_in,
             std::shared_ptr<Operators::ViscousKernel<dim, Number>> viscous_kernel_in,
             TurbulenceModelData const &                            data_in);

  /*
   *  This function calculates the turbulent viscosity for a given velocity field.
   */
  void
  calculate_turbulent_viscosity(VectorType const & velocity) const;

  /*
   *  This function calculates the filter width for each cell.
   */
  void
  calculate_filter_width(Mapping<dim> const & mapping);

private:
  void
  cell_loop_set_coefficients(MatrixFree<dim, Number> const & data,
                             VectorType &,
                             VectorType const & src,
                             Range const &      cell_range) const;

  void
  face_loop_set_coefficients(MatrixFree<dim, Number> const & data,
                             VectorType &,
                             VectorType const & src,
                             Range const &      face_range) const;

  void
  boundary_face_loop_set_coefficients(MatrixFree<dim, Number> const & data,
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


  TurbulenceModelData turb_model_data;

  MatrixFree<dim, Number> const * matrix_free;

  std::shared_ptr<Operators::ViscousKernel<dim, Number>> viscous_kernel;

  AlignedVector<scalar> filter_width_vector;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_TURBULENCE_MODEL_H_ */
