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

#ifndef INCLUDE_EXADG_RANS_EQUATIONS_SPATIAL_DISCRETIZATION_TURBULENCE_MODEL_H_
#define INCLUDE_EXADG_RANS_EQUATIONS_SPATIAL_DISCRETIZATION_TURBULENCE_MODEL_H_

// ExaDG
#include <exadg/rans_equations/spatial_discretization/viscosity_model_base.h>
#include <exadg/operators/variable_coefficients.h>
#include <iostream>
#include <memory>
#include "exadg/rans_equations/user_interface/enum_types.h"
#include "exadg/rans_equations/user_interface/viscosity_model_data.h"

namespace ExaDG
{
namespace RANS
{
struct TurbulenceDataBase
{
  TurbulenceDataBase() : sigma_k(1.0)
  {}
  virtual ~TurbulenceDataBase() {}
  double sigma_k;

  virtual std::vector<double> get_all_coefficients() const = 0;
};
struct PrandtlMixingLengthData : public TurbulenceDataBase
{
  PrandtlMixingLengthData() : C_D(0.07),
    turbulent_length_scale(1.0)
  {}

  double C_D;
  double turbulent_length_scale;

  virtual std::vector<double> get_all_coefficients() const override
  {
    return {sigma_k, C_D, turbulent_length_scale};
  }
};
struct StandardKEpsilonData : public TurbulenceDataBase
{
  StandardKEpsilonData() : C_epsilon_1(1.44),
                  C_epsilon_2(1.92),
                  C_mu(0.09),
                  sigma_epsilon(1.3)
  {}

  double C_epsilon_1;
  double C_epsilon_2;
  double C_mu;
  double sigma_epsilon;

  virtual std::vector<double> get_all_coefficients() const override
  {
    return {sigma_k, C_epsilon_1, C_epsilon_2, C_mu, sigma_epsilon};
  }

};

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

  typedef CellIntegrator<dim, 1, Number> IntegratorCell;
  typedef FaceIntegrator<dim, 1, Number> IntegratorFace;

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
             TurbulenceModelData const &                            turbulence_model_data_in,
             unsigned int const                                     dof_index_in,
             unsigned int const                                     quad_index_in);

  /**
   * Function for *setting* the viscosity taking the viscosity stored in the viscous_kernel's data
   * as a basis.
   */
  void
  set_viscosity(VectorType const & solution) final;

  /**
   * Function for *adding to* the viscosity taking the currently stored viscosity as a basis.
   */
  void
  add_viscosity(VectorType const & solution) final;

  void
  set_turbulent_kinetic_energy(VectorType const & tke_in);

  void
  set_tke_dissipation_rate(VectorType const & epsilon_in);

  void
  get_eddy_viscosity(VectorType & dst);

private:
  void
  cell_loop_set_coefficients(dealii::MatrixFree<dim, Number> const & data,
                             VectorType &,
                             VectorType const & src,
                             Range const &      cell_range);

  void
  face_loop_set_coefficients(dealii::MatrixFree<dim, Number> const & data,
                             VectorType &,
                             VectorType const & src,
                             Range const &      face_range);

  void
  boundary_face_loop_set_coefficients(dealii::MatrixFree<dim, Number> const & data,
                                      VectorType &,
                                      VectorType const & src,
                                      Range const &      face_range);

public:
  /**
   *  This function adds the turbulent eddy-viscosity to the laminar viscosity
   *  by using one of the implemented models.
   */
  void
  add_one_equation_turbulent_viscosity(scalar &       viscosity,
                          scalar const & solution) const;

  void
  add_two_equation_turbulent_viscosity(scalar &       viscosity,
                          scalar const & tke,
                          scalar const & epsilon) const;

  void
  prandtl_mixing_length_model(scalar const & sol,
                     scalar & viscosity) const;

  void
  k_epsilon_model(scalar const & tke,
                  scalar const & epsilon,
                  scalar & viscosity) const;

  void
  set_coefficient(VariableCoefficients<dealii::VectorizedArray<Number>> src)
  {
    viscosity_coefficients = src;
  }

  void
  set_constant_coefficient(Number const & constant_coefficient)
  {
    viscosity_coefficients.set_coefficients(constant_coefficient);
    eddy_viscosity_coefficients.set_coefficients(1.0);
  }
  scalar
  get_coefficient_cell(unsigned int const cell, unsigned int const q)
  {
    return viscosity_coefficients.get_coefficient_cell(cell, q);
  }

  void
  set_coefficient_cell(unsigned int const cell, unsigned int const q, scalar const & value)
  {
    viscosity_coefficients.set_coefficient_cell(cell, q, value);
  }

  scalar
  get_coefficient_face(unsigned int const face, unsigned int const q)
  {
    return viscosity_coefficients.get_coefficient_face(face, q);
  }

  void
  set_coefficient_face(unsigned int const face, unsigned int const q, scalar const & value)
  {
    viscosity_coefficients.set_coefficient_face(face, q, value);
  }

  scalar
  get_coefficient_face_neighbor(unsigned int const face, unsigned int const q)
  {
    return viscosity_coefficients.get_coefficient_face_neighbor(face, q);
  }

  void
  set_coefficient_face_neighbor(unsigned int const face, unsigned int const q, scalar const & value)
  {
    viscosity_coefficients.set_coefficient_face_neighbor(face, q, value);
  }
/*
 *  returns the value of viscosity for the cell as a scalar 
 */
  inline DEAL_II_ALWAYS_INLINE //
  scalar
  get_viscosity_cell(unsigned int const cell, unsigned int const q, VaryingViscosityType viscosity_type) const
  {
    scalar viscosity;
    if(viscosity_type == VaryingViscosityType::CombinedViscosity)
    {
      viscosity = viscosity_coefficients.get_coefficient_cell(cell, q);
    }
    else if (viscosity_type == VaryingViscosityType::EddyViscosity) {
      viscosity = eddy_viscosity_coefficients.get_coefficient_cell(cell, q);
    }
    else {
      std::cerr << " Implementation only available for VaryingViscosity::CombinedViscosity and VaryingViscosity::EddyViscosity" << std::endl;
    }
    return viscosity;
  }

  /*
   *  This function returns the viscosity for interior faces.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    get_viscosity_interior_face(unsigned int const face, unsigned int const q, VaryingViscosityType viscosity_type) const
  {
    scalar viscosity = calculate_average_viscosity(face, q, viscosity_type);

    return viscosity;
  }

  /*
   *  This function returns the viscosity for boundary faces.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    get_viscosity_boundary_face(unsigned int const face, unsigned int const q, VaryingViscosityType viscosity_type) const
  {
    scalar viscosity;

    if(viscosity_type == VaryingViscosityType::CombinedViscosity)
    {
      viscosity = viscosity_coefficients.get_coefficient_face(face, q);
    }
    else if (viscosity_type == VaryingViscosityType::EddyViscosity) {
      viscosity = eddy_viscosity_coefficients.get_coefficient_face(face, q);
    }
    else {
      std::cerr << " Implementation only available for VaryingViscosity::CombinedViscosity and VaryingViscosity::EddyViscosity" << std::endl;
    }

    return viscosity;
  }

  /*
   *  This function calculates the average viscosity for interior faces.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_average_viscosity(unsigned int const face, unsigned int const q, VaryingViscosityType viscosity_type) const
  {
    scalar average_viscosity = dealii::make_vectorized_array<Number>(0.0);

    scalar coefficient_face;
    scalar coefficient_face_neighbor;

    if(viscosity_type==VaryingViscosityType::CombinedViscosity)
    {
      coefficient_face = viscosity_coefficients.get_coefficient_face(face, q);
      coefficient_face_neighbor =
        viscosity_coefficients.get_coefficient_face_neighbor(face, q);
    }
    else if (viscosity_type==VaryingViscosityType::EddyViscosity) {
      coefficient_face = eddy_viscosity_coefficients.get_coefficient_face(face, q);
      coefficient_face_neighbor = eddy_viscosity_coefficients.get_coefficient_face_neighbor(face, q);
    }
    else {
      std::cerr << " Implementation only available for VaryingViscosity::CombinedViscosity and VaryingViscosity::EddyViscosity" << std::endl;
    }

    // harmonic mean (harmonic weighting according to Schott and Rasthofer et al. (2015))
    // average_viscosity = 2.0 * coefficient_face * coefficient_face_neighbor /
    //                     (coefficient_face + coefficient_face_neighbor);

    // arithmetic mean
    average_viscosity = 0.5 * (coefficient_face + coefficient_face_neighbor);

    // maximum value
    // average_viscosity = std::max(coefficient_face, coefficient_face_neighbor);

    return average_viscosity;
  }

  std::shared_ptr<TurbulenceDataBase>
  create_turbulence_data();

  TurbulenceModelData           turbulence_model_data;
  VariableCoefficients<dealii::VectorizedArray<Number>> viscosity_coefficients;
  VariableCoefficients<dealii::VectorizedArray<Number>> eddy_viscosity_coefficients;

  VectorType const * turbulent_kinetic_energy;
  VectorType const * tke_dissipation_rate;
  VectorType eddy_viscosity;

public:
  double diffusivity;
  ScalarType scalar_type;

  std::shared_ptr<TurbulenceDataBase> turbulence_data_base = create_turbulence_data();
  std::vector<double> model_coefficients;
};

} // namespace RANSEqns
} // namespace ExaDG

#endif /* INCLUDE_EXADG_RANS_EQUATIONS_SPATIAL_DISCRETIZATION_TURBULENCE_MODEL_H_ */
