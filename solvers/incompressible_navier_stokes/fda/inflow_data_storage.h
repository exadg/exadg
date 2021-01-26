/*
 * inflow_data_storage.h
 *
 *  Created on: 30.03.2020
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FDA_INFLOW_DATA_STORAGE_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FDA_INFLOW_DATA_STORAGE_H_

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim>
struct InflowDataStorage
{
  InflowDataStorage(unsigned int const n_points,
                    double const       R,
                    double const       max_velocity,
                    bool const         use_random_perturbations,
                    double const       factor_random_perturbations)
    : n_points_r(n_points),
      n_points_phi(n_points),
      R(R),
      max_velocity(max_velocity),
      use_random_perturbations(use_random_perturbations),
      factor_random_perturbations(factor_random_perturbations)
  {
    r_values.resize(n_points_r);
    phi_values.resize(n_points_phi);
    velocity_values.resize(n_points_r * n_points_phi);

    initialize_r_and_phi_values();
    initialize_velocity_values();
  }

  // initialize vectors
  void
  initialize_r_and_phi_values()
  {
    AssertThrow(n_points_r >= 2, ExcMessage("Variable n_points_r is invalid"));
    AssertThrow(n_points_phi >= 2, ExcMessage("Variable n_points_phi is invalid"));

    // 0 <= radius <= R_OUTER
    for(unsigned int i = 0; i < n_points_r; ++i)
      r_values[i] = double(i) / double(n_points_r - 1) * R;

    // - pi <= phi <= pi
    for(unsigned int i = 0; i < n_points_phi; ++i)
      phi_values[i] = -numbers::PI + double(i) / double(n_points_phi - 1) * 2.0 * numbers::PI;
  }

  void
  initialize_velocity_values()
  {
    AssertThrow(n_points_r >= 2, ExcMessage("Variable n_points_r is invalid"));
    AssertThrow(n_points_phi >= 2, ExcMessage("Variable n_points_phi is invalid"));

    for(unsigned int iy = 0; iy < n_points_r; ++iy)
    {
      for(unsigned int iz = 0; iz < n_points_phi; ++iz)
      {
        Tensor<1, dim, double> velocity;
        // flow in z-direction
        velocity[2] = max_velocity * (1.0 - std::pow(r_values[iy] / R, 2.0));

        if(use_random_perturbations == true)
        {
          // Add random perturbation
          double perturbation =
            factor_random_perturbations * velocity[2] * ((double)rand() / RAND_MAX - 0.5) / 0.5;
          velocity[2] += perturbation;
        }

        velocity_values[iy * n_points_phi + iz] = velocity;
      }
    }
  }

  void
  add_random_perturbations()
  {
    AssertThrow(n_points_r >= 2, ExcMessage("Variable n_points_r is invalid"));
    AssertThrow(n_points_phi >= 2, ExcMessage("Variable n_points_phi is invalid"));

    for(unsigned int iy = 0; iy < n_points_r; ++iy)
    {
      for(unsigned int iz = 0; iz < n_points_phi; ++iz)
      {
        // Add random perturbation
        double perturbation = factor_random_perturbations * ((double)rand() / RAND_MAX - 0.5) / 0.5;

        velocity_values[iy * n_points_phi + iz] *= (1.0 + perturbation);
      }
    }
  }

  unsigned int n_points_r;
  unsigned int n_points_phi;

  double const R;
  double const max_velocity;
  bool const   use_random_perturbations;
  double const factor_random_perturbations;

  std::vector<double>                 r_values;
  std::vector<double>                 phi_values;
  std::vector<Tensor<1, dim, double>> velocity_values;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FDA_INFLOW_DATA_STORAGE_H_ */
