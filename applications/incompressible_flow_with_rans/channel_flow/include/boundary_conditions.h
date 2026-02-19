
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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CHANNEL_FLOW_INCLUDE_BOUNDARY_CONDITIONS_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CHANNEL_FLOW_INCLUDE_BOUNDARY_CONDITIONS_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <cmath>
#include "exadg/rans_equations/spatial_discretization/turbulence_model.h"
#include "exadg/rans_equations/user_interface/viscosity_model_data.h"
namespace ExaDG
{
namespace NSRans
{
enum class InletVelocityProfileType
{
  Undefined,
  Uniform,
  Parabolic,
  PowerLaw
};

template<int dim>
class InletVelocityProfile : public dealii::Function<dim>
{
public:
  InletVelocityProfile(double const U_bulk,
                       double const half_width,
                       InletVelocityProfileType const velocity_profile_type)
  : dealii::Function<dim>(dim, 0.0),
  U_bulk(U_bulk),
  half_width(half_width),
  velocity_profile_type(velocity_profile_type)
  {}

  double
  value(dealii::Point<dim> const &p, unsigned int const component = 0) const final
  {
    double result = 0.0;
    if (component==0) {
      if (velocity_profile_type==InletVelocityProfileType::Parabolic) {
        double H = 2.0 * half_width;
        result = 4.0 * U_bulk * (p[1] / H) * ( 1 - (p[1]/H));
      } else if (velocity_profile_type==InletVelocityProfileType::Uniform) {
        result = U_bulk;
      } else if (velocity_profile_type==InletVelocityProfileType::PowerLaw) {
        double m,n,U_max,H,y_H;
        H = 2.0 * half_width;
        y_H = p[1] / H;
        m = 2.0;
        n = 7.0;
        /*U_max = 8.0 / 7.0 * U_bulk;*/
        result = std::pow(y_H - std::pow(y_H, m),(1.0/n)) * U_bulk;
      } else {
        AssertThrow(false, dealii::ExcMessage("InletVelocityProfileType is not specified"));
      }
    }
    else {
      result = 0.0;
    }
    return result;
  }
private:
  double U_bulk;
  double half_width;
  InletVelocityProfileType velocity_profile_type;
};

template<int dim>
class InletTKE : public dealii::Function<dim>
{
public:
  InletTKE(double const U_bulk,
           double const half_width,
           double const turb_intensity,
           bool const positivity_limiter,
           InletVelocityProfileType const velocity_profile_type)
    : dealii::Function<dim>(1 /*n_components*/, 0.0),
    U_bulk(U_bulk),
    half_width(half_width),
    turb_intensity(turb_intensity),
    positivity_limiter(positivity_limiter),
    velocity_profile_type(velocity_profile_type)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const /*component*/) const final
  {
    // tke = 3/2 (U I)^2
    // U -> time averaged velocity
    // I -> turbulent intensity
    // I = \frac{u^{'}}{U}
    // Assume u^{'} = 5% of U => I = 0.05
    // \kappa = ln(tke)
    double velocity;
      if (velocity_profile_type==InletVelocityProfileType::Parabolic) {
        velocity = U_bulk * (1 - std::pow(p[1]/(half_width), 2.0));
      } else if (velocity_profile_type==InletVelocityProfileType::Uniform) {
        velocity = U_bulk;
      } else if (velocity_profile_type==InletVelocityProfileType::PowerLaw) {
        double m,n,U_max;
        m = 2.0;
        n = 7.0;
        /*U_max = 8.0 / 7.0 * U_bulk;*/
        velocity = std::pow(1 - std::pow(p[1]/half_width, m),(1.0/n)) * U_bulk;
      } else {
        AssertThrow(false, dealii::ExcMessage("InletVelocityProfileType is not specified"));
      }
    double result = 1.0;
    if (positivity_limiter) {
      result = std::log(1.5 * std::pow(velocity * turb_intensity, 2.0));
    } else {
      result = 1.5 * std::pow(velocity * turb_intensity, 2.0);
    }
    return result;
  }

private:
  double U_bulk;
  double half_width;
  double turb_intensity;
  bool positivity_limiter;
  InletVelocityProfileType velocity_profile_type;
};


template<int dim>
class InletTKEDissipationRate : public dealii::Function<dim>
{
public:
  InletTKEDissipationRate(double const U_bulk,
           double const half_width,
           double const turb_intensity,
           bool const positivity_limiter,
           InletVelocityProfileType const velocity_profile_type)
    : dealii::Function<dim>(1 /*n_components*/, 0.0),
    U_bulk(U_bulk),
    half_width(half_width),
    turb_intensity(turb_intensity),
    positivity_limiter(positivity_limiter),
    velocity_profile_type(velocity_profile_type)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const /*component*/) const final
  {
    // tke = 3/2 (U I)^2
    // U -> time averaged velocity
    // I -> turbulent intensity
    // I = \frac{u^{'}}{U}
    // Assume u^{'} = 5% of U => I = 0.05
    // \kappa = ln(tke)

    double velocity;
      if (velocity_profile_type==InletVelocityProfileType::Parabolic) {
        velocity = U_bulk * (1 - std::pow(p[1]/(half_width), 2.0));
      } else if (velocity_profile_type==InletVelocityProfileType::Uniform) {
        velocity = U_bulk;
      } else if (velocity_profile_type==InletVelocityProfileType::PowerLaw) {
        double m,n,U_max;
        m = 2.0;
        n = 7.0;
        /*U_max = 8.0 / 7.0 * U_bulk;*/
        velocity = std::pow(1 - std::pow(p[1]/half_width, m),(1.0/n)) * U_bulk;
      } else {
        AssertThrow(false, dealii::ExcMessage("InletVelocityProfileType is not specified"));
      }

    double result = 1.0;
    double tke = 1.5 * std::pow(velocity * turb_intensity, 2.0);
    double mixing_length = 2 * half_width * 0.07;
    double C_mu = 0.09;
    result = std::pow(C_mu, 3/4) * std::pow(tke, 3/2) / mixing_length;
    if (positivity_limiter) {
      result = std::log(result);
    }
    return result;
  }

private:
  double U_bulk;
  double half_width;
  double turb_intensity;
  bool positivity_limiter;
  InletVelocityProfileType velocity_profile_type;
};

} // namespace IncRANS
} // namespace ExaDG

#endif
