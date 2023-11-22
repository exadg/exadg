/*
 * matrix_free_integrals_and_kernels.cpp
 *
 *  Created on: 22.11.2023
 *      Author: fehnnikl
 */

#include <exadg/matrix_free/integrals_and_kernels.h>

void
my_cell_integral()
{
  unsigned int const velocity = 0;
  unsigned int const pressure = velocity + 1;

  std::vector<ConfigurationData> config_data;
  config_data.resize(2);
  config_data[velocity].needs_value     = true;
  config_data[velocity].needs_gradient  = true;
  config_data[velocity].submit_value    = true;
  config_data[velocity].submit_gradient = true;
  config_data[pressure].needs_gradient  = true;
  config_data[pressure].submit_gradient = true;

  auto const navier_stokes_kernel = [&](KernelData & data) {
    double const dt_inv = 10.0, nu = 0.001;

    data[velocity].value_flux    = dt_inv * data[velocity].value + data[pressure].gradient;
    data[velocity].gradient_flux = nu * data[velocity].gradient;
    data[pressure].gradient_flux = data[velocity].value;
  };

  generic_cell_integral(config_data, navier_stokes_kernel);
}

int
main()
{
  my_cell_integral();
}
