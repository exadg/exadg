/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef INCLUDE_EXADG_RANS_EQUATIONS_USER_INTERFACE_VISCOSITY_MODEL_DATA_H_
#define INCLUDE_EXADG_RANS_EQUATIONS_USER_INTERFACE_VISCOSITY_MODEL_DATA_H_

// ExaDG
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace RANS
{
/**
 *  RANS based turbulence models
 *
 * Standard k-epsilon model
 */
enum class TurbulenceEddyViscosityModel
{
  Undefined,
  StandardKEpsilon
};

enum class VaryingViscosityType
{
  CombinedViscosity,
  EddyViscosity
};

/*
 * Numerical limiters to ensure physical results (non-negative viscosity).
 */
enum class PositivityPreservingLimiter
{
  Undefined,
  Clipper,
  LogarithmicTransportVariable
};

struct TurbulenceDataBase
{
  TurbulenceDataBase() : sigma_k(1.0)
  {
  }
  virtual ~TurbulenceDataBase()
  {
  }
  double sigma_k;

  virtual void
  set_all_coefficients(std::vector<double> const & coefficients) = 0;
  virtual std::vector<double>
  get_all_coefficients() const = 0;

  virtual void
  print_coefficients(dealii::ConditionalOStream const & pcout) const = 0;
};

/*
 * @brief Parameters for the Standard k-epsilon model.
 * @details Default values follow the standard model constants.
 * Reference: https://www.cfd-online.com/Wiki/Standard_k-epsilon_model
 * * The constants are:
 * \f$ C_{\mu} = 0.09 \f$, \f$ C_{\epsilon 1} = 1.44 \f$, \f$ C_{\epsilon 2} = 1.92 \f$, 
 * \f$ \sigma_k = 1.0 \f$, \f$ \sigma_{\epsilon} = 1.3 \f$.
 */
struct StandardKEpsilonData : public TurbulenceDataBase
{
  StandardKEpsilonData() : C_epsilon_1(1.44), C_epsilon_2(1.92), C_mu(0.09), sigma_epsilon(1.3)
  {
  }

  double C_epsilon_1;
  double C_epsilon_2;
  double C_mu;
  double sigma_epsilon;

  virtual void
  set_all_coefficients(std::vector<double> const & coefficients) override
  {
    sigma_k       = coefficients[0];
    C_epsilon_1   = coefficients[1];
    C_epsilon_2   = coefficients[2];
    C_mu          = coefficients[3];
    sigma_epsilon = coefficients[4];
  }

  virtual std::vector<double>
  get_all_coefficients() const override
  {
    return {sigma_k, C_epsilon_1, C_epsilon_2, C_mu, sigma_epsilon};
  }

  virtual void
  print_coefficients(dealii::ConditionalOStream const & pcout) const override
  {
    print_parameter(pcout, "sigma_k", sigma_k);
    print_parameter(pcout, "C_epsilon_1", C_epsilon_1);
    print_parameter(pcout, "C_epsilon_2", C_epsilon_2);
    print_parameter(pcout, "C_mu", C_mu);
    print_parameter(pcout, "sigma_epsilon", sigma_epsilon);
  }
};

struct TurbulenceModelData
{
  TurbulenceModelData()
  {
  }

  TurbulenceEddyViscosityModel turbulence_model{TurbulenceEddyViscosityModel::Undefined};
  bool                         is_active{false};
  bool                         production_term{false};
  bool                         dissipation_term{false};
  PositivityPreservingLimiter positivity_preserving_limiter{PositivityPreservingLimiter::Undefined};
  std::shared_ptr<TurbulenceDataBase> turbulence_data_base;

  void
  initialize_and_set_turbulence_coefficients(std::vector<double> const & coefficients)
  {
    if(is_active)
    {
      turbulence_data_base = create_turbulence_data();
      turbulence_data_base->set_all_coefficients(coefficients);
    }
  }

  void
  check()
  {
    AssertThrow(is_active, dealii::ExcMessage("Turbulence model is inactive."));
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    pcout << std::endl << "Turbulence:" << std::endl;

    print_parameter(pcout, "Use turbulence model", is_active);

    if(is_active)
    {
      AssertThrow(turbulence_data_base != nullptr,
                  dealii::ExcMessage("Turbulence data base not initialized."));
      print_parameter(pcout, "Turbulence model", turbulence_model);
      print_parameter(pcout, "Positivity preserving limiter", positivity_preserving_limiter);
      print_parameter(pcout, "Use production term", production_term);
      print_parameter(pcout, "Use dissipation term", dissipation_term);
      turbulence_data_base->print_coefficients(pcout);
    }
  }

  std::shared_ptr<TurbulenceDataBase>
  create_turbulence_data()
  {
    switch(turbulence_model)
    {
      case TurbulenceEddyViscosityModel::StandardKEpsilon:
        return std::make_shared<StandardKEpsilonData>();
        break;
    }
  }
};


} // namespace RANS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_RANS_EQUATIONS_USER_INTERFACE_VISCOSITY_MODEL_DATA_H_ */
