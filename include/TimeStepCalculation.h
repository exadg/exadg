/*
 * TimeStepCalculation.h
 *
 *  Created on: May 30, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_TIMESTEPCALCULATION_H_
#define INCLUDE_TIMESTEPCALCULATION_H_


double calculate_const_time_step(double const dt,
                                 unsigned int const n_refine_time)
{
  double time_step = dt/std::pow(2.,n_refine_time);

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "User specified time step size:" << std::endl << std::endl
              << "  time step size:" << std::scientific << std::setprecision(4) << std::setw(14) << std::right << time_step << std::endl;

  return time_step;
}

template<int dim, int fe_degree>
double calculate_const_time_step_cfl(Triangulation<dim> const &triangulation,
                                     double const cfl,
                                     double const max_velocity,
                                     double const start_time,
                                     double const end_time,
                                     double const exponent_fe_degree = 2.0)
{
  typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(), endc = triangulation.end();

  double diameter = 0.0, min_cell_diameter = std::numeric_limits<double>::max();
  for (; cell!=endc; ++cell)
  {
    if (cell->is_locally_owned())
    {
      diameter = cell->minimum_vertex_distance();
      if (diameter < min_cell_diameter)
        min_cell_diameter = diameter;
    }
  }
  const double global_min_cell_diameter = -Utilities::MPI::max(-min_cell_diameter, MPI_COMM_WORLD);

  // cfl/p^{exponent_fe_degree} = || U || * time_step / h
  double time_step = cfl/pow(fe_degree,exponent_fe_degree) * global_min_cell_diameter / max_velocity;

  // decrease time_step in order to exactly hit END_TIME
  time_step = (end_time-start_time)/(1+int((end_time-start_time)/time_step));

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "Calculation of time step size according to CFL condition:" << std::endl << std::endl
              << "  h_min:         " << std::scientific << std::setprecision(4) << std::setw(14) << std::right << global_min_cell_diameter << std::endl
              << "  u_max:         " << std::scientific << std::setprecision(4) << std::setw(14) << std::right << max_velocity << std::endl
              << "  CFL:           " << std::scientific << std::setprecision(4) << std::setw(14) << std::right << cfl  << std::endl
              << "  time step size:" << std::scientific << std::setprecision(4) << std::setw(14) << std::right << time_step << std::endl;

  return time_step;
}

template<int dim, int fe_degree, typename value_type>
double calculate_adaptive_time_step_cfl(MatrixFree<dim,value_type> const                &data,
                                        parallel::distributed::Vector<value_type> const &velocity,
                                        double const                                    cfl,
                                        double const                                    last_time_step,
                                        bool const                                      use_limiter = true,
                                        double const                                    exponent_fe_degree = 1.5)
{
  FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval(data,
      static_cast <typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity),
      static_cast<typename std::underlying_type_t<QuadratureSelector> >(QuadratureSelector::velocity));

  value_type new_time_step = std::numeric_limits<value_type>::max();
  value_type cfl_p = cfl/pow(fe_degree,exponent_fe_degree);

  // loop over cells of processor
  for (unsigned int cell=0; cell<data.n_macro_cells(); ++cell)
  {
    VectorizedArray<value_type> delta_t_cell =
        make_vectorized_array<value_type>(std::numeric_limits<value_type>::max());
    Tensor<2,dim,VectorizedArray<value_type> > invJ;
    Tensor<1,dim,VectorizedArray<value_type> > u_x;
    Tensor<1,dim,VectorizedArray<value_type> > ut_xi;

    fe_eval.reinit(cell);
    fe_eval.read_dof_values(velocity);
    fe_eval.evaluate(true,false);

    // loop over quadrature points
    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      u_x = fe_eval.get_value(q);
      invJ = fe_eval.inverse_jacobian(q);
      invJ = transpose(invJ);
      ut_xi = invJ*u_x;

      delta_t_cell = std::min(delta_t_cell,cfl_p/ut_xi.norm());
    }

    // loop over vectorized array
    value_type dt = std::numeric_limits<value_type>::max();
    for (unsigned int v = 0; v < VectorizedArray<value_type>::n_array_elements; ++v)
    {
      dt = std::min(dt,delta_t_cell[v]);
    }

    new_time_step = std::min(new_time_step,dt);
  }

  // find minimum over all processors
  new_time_step = Utilities::MPI::min(new_time_step, MPI_COMM_WORLD);

  // limit the maximum increase/decrease of the time step size
  if(use_limiter == true)
  {
    double fac = 1.2; //TODO
    if (new_time_step >= fac*last_time_step)
    {
      new_time_step = fac*last_time_step;
    }

    else if (new_time_step <= last_time_step/fac)
    {
      new_time_step = last_time_step/fac;
    }

  }

  return new_time_step;
}


#endif /* INCLUDE_TIMESTEPCALCULATION_H_ */
