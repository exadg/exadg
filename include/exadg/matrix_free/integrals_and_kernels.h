/*
 * integrals_and_kernels.h
 *
 *  Created on: 22.11.2023
 *      Author: fehnnikl
 */

#ifndef INCLUDE_EXADG_MATRIX_FREE_INTEGRALS_AND_KERNELS_H_
#define INCLUDE_EXADG_MATRIX_FREE_INTEGRALS_AND_KERNELS_H_

#include <functional>
#include <vector>

struct PointData
{
  double value;
  double gradient;
  double value_flux;
  double gradient_flux;
};

class KernelData
{
public:
  PointData &
  operator[](unsigned int const field)
  {
    return data[field];
  }

private:
  std::vector<PointData> data;
};

class ConfigurationData
{
public:
  bool needs_value{false};
  bool needs_gradient{false};
  bool submit_value{false};
  bool submit_gradient{false};
};

class FEEvaluation
{
public:
  void
  reinit(unsigned int const cell)
  {
    (void)cell;
  }

  double
  get_value(unsigned int const q) const
  {
    return values[q];
  }

  double
  get_gradient(unsigned int const q) const
  {
    return gradients[q];
  }

  void
  submit_value(double value, unsigned int const q)
  {
    values[q] = value;
  }

  void
  submit_gradient(double gradient, unsigned int const q)
  {
    gradients[q] = gradient;
  }

private:
  std::vector<double> values;
  std::vector<double> gradients;
};

class Integrators
{
public:
  Integrators(std::vector<ConfigurationData> const & config_data)
  {
    integrators.resize(config_data.size());
  }

  void
  reinit(unsigned int const cell)
  {
    for(unsigned int i = 0; i < integrators.size(); ++i)
    {
      integrators[i].reinit(cell);
    }
  }

  void
  get(KernelData & data, unsigned int const q)
  {
    for(unsigned int i = 0; i < integrators.size(); ++i)
    {
      if(config_data[i].needs_value)
        data[i].value = integrators[i].get_value(q);
      if(config_data[i].needs_gradient)
        data[i].gradient = integrators[i].get_gradient(q);
    }
  }

  void
  submit(KernelData & data, unsigned int const q)
  {
    for(unsigned int i = 0; i < integrators.size(); ++i)
    {
      if(config_data[i].submit_value)
        integrators[i].submit_value(data[i].value, q);
      if(config_data[i].submit_gradient)
        integrators[i].submit_gradient(data[i].gradient, q);
    }
  }

private:
  std::vector<ConfigurationData> config_data;
  std::vector<FEEvaluation>      integrators;
};

void
generic_cell_integral(std::vector<ConfigurationData> const &    config_data,
                      std::function<void(KernelData &)> const & kernel)
{
  Integrators integrators{config_data};

  unsigned int const n_cells = 10;
  for(unsigned int cell = 0; cell < n_cells; ++cell)
  {
    integrators.reinit(cell);

    // TODO: evaluate (generic functionality)

    unsigned int const n_q_points = 10;
    for(unsigned int q = 0; q < n_q_points; ++q)
    {
      KernelData data;

      integrators.get(data, q);

      kernel(data);

      integrators.submit(data, q);
    }

    // TODO: integrate (generic functionality)
  }
}


#endif /* INCLUDE_EXADG_MATRIX_FREE_INTEGRALS_AND_KERNELS_H_ */
