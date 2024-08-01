/*
 * matrix_free_integrals_and_kernels.cpp
 *
 *  Created on: 22.11.2023
 *      Author: fehnnikl
 */

#include <functional>
#include <vector>

struct FieldData
{
  // think of dealii::Tensor here
  typedef double Tensor;

  Tensor value;
  Tensor gradient;
  Tensor submit_value;
  Tensor submit_gradient;
};

class QuadraturePointData
{
public:
  FieldData &
  operator[](unsigned int const field)
  {
    return data[field];
  }

private:
  std::vector<FieldData> data;
};

class ConfigurationData
{
public:
  int  dof_index{0};
  int  quad_index{0};
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

  void
  evaluate(std::vector<double> const & src, bool needs_value, bool needs_gradient)
  {
    (void)src;
    (void)needs_value;
    (void)needs_gradient;
  }

  void
  integrate(std::vector<double> & dst, bool submit_value, bool submit_gradient)
  {
    (void)dst;
    (void)submit_value;
    (void)submit_gradient;
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
  get(QuadraturePointData & data, unsigned int const q)
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
  submit(QuadraturePointData & data, unsigned int const q)
  {
    for(unsigned int i = 0; i < integrators.size(); ++i)
    {
      if(config_data[i].submit_value)
        integrators[i].submit_value(data[i].submit_value, q);
      if(config_data[i].submit_gradient)
        integrators[i].submit_gradient(data[i].submit_gradient, q);
    }
  }

  void
  evaluate(std::vector<double> const & src)
  {
    (void)src;
  }

  void
  integrate(std::vector<double> & dst)
  {
    (void)dst;
  }

private:
  std::vector<ConfigurationData> config_data;
  std::vector<FEEvaluation>      integrators;
};

void
cell_integral_navier_stokes_cpu_backend(std::vector<double> & dst, std::vector<double> const & src)
{
  FEEvaluation fe_eval_velocity, fe_eval_pressure;

  std::vector<double> velocity_src = src; //.block(0);
  std::vector<double> pressure_src = src; //.block(1);

  std::vector<double> & velocity_dst = dst; //.block(0);
  std::vector<double> & pressure_dst = dst; //.block(1);

  unsigned int const n_cells = 10;
  for(unsigned int cell = 0; cell < n_cells; ++cell)
  {
    fe_eval_velocity.reinit(cell);
    fe_eval_pressure.reinit(cell);

    fe_eval_velocity.evaluate(velocity_src, true, true);
    fe_eval_pressure.evaluate(pressure_src, false, true);

    unsigned int const n_q_points = 10;
    for(unsigned int q = 0; q < n_q_points; ++q)
    {
      double u      = fe_eval_velocity.get_value(q);
      double u_grad = fe_eval_velocity.get_gradient(q);
      double p_grad = fe_eval_pressure.get_gradient(q);

      double const dt_inv = 10.0, nu = 0.001;

      double submit_value_u    = dt_inv * u + p_grad;
      double submit_gradient_u = nu * u_grad;
      double submit_gradient_p = u;

      fe_eval_velocity.submit_value(submit_value_u, q);
      fe_eval_velocity.submit_gradient(submit_gradient_u, q);
      fe_eval_pressure.submit_gradient(submit_gradient_p, q);
    }

    fe_eval_velocity.integrate(velocity_dst, true, true);
    fe_eval_pressure.integrate(pressure_dst, false, true);
  }
}

void
generic_cell_integral_cpu_backend(std::vector<double> &                              dst,
                                  std::vector<double> const &                        src,
                                  std::vector<ConfigurationData> const &             config_data,
                                  std::function<void(QuadraturePointData &)> const & kernel)
{
  Integrators integrators = Integrators(config_data);

  unsigned int const n_cells = 10;
  for(unsigned int cell = 0; cell < n_cells; ++cell)
  {
    integrators.reinit(cell);

    integrators.evaluate(src);

    unsigned int const n_q_points = 10;
    for(unsigned int q = 0; q < n_q_points; ++q)
    {
      QuadraturePointData data;

      integrators.get(data, q);

      kernel(data);

      integrators.submit(data, q);
    }

    integrators.integrate(dst);
  }
}

void
portable_cell_integral(std::vector<double> & dst, std::vector<double> const & src)
{
  unsigned int const velocity = 0;
  unsigned int const pressure = velocity + 1;

  std::vector<ConfigurationData> config_data;
  config_data.resize(2);
  config_data[velocity].dof_index       = 0;
  config_data[velocity].quad_index      = 0;
  config_data[velocity].needs_value     = true;
  config_data[velocity].needs_gradient  = true;
  config_data[velocity].submit_value    = true;
  config_data[velocity].submit_gradient = true;
  config_data[pressure].dof_index       = 1;
  config_data[pressure].quad_index      = 0;
  config_data[pressure].needs_gradient  = true;
  config_data[pressure].submit_gradient = true;

  auto const navier_stokes_kernel = [&](QuadraturePointData & data) {
    double const dt_inv = 10.0, nu = 0.001;

    data[velocity].submit_value    = dt_inv * data[velocity].value + data[pressure].gradient;
    data[velocity].submit_gradient = nu * data[velocity].gradient;
    data[pressure].submit_gradient = data[velocity].value;
  };

  generic_cell_integral_cpu_backend(dst, src, config_data, navier_stokes_kernel);
}

int
main()
{
  std::vector<double> dst, src;

  // old
  cell_integral_navier_stokes_cpu_backend(dst, src);

  // new
  portable_cell_integral(dst, src);
}
