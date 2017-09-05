/*
 * perturbation_energy_orr_sommerfeld.h
 *
 *  Created on: Sep 1, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_PERTURBATION_ENERGY_ORR_SOMMERFELD_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_PERTURBATION_ENERGY_ORR_SOMMERFELD_H_

template<int dim, int fe_degree, typename Number>
class PerturbationEnergyCalculator
{
public:
  PerturbationEnergyCalculator()
    :
    clear_files(true),
    counter(0),
    initial_perturbation_energy_has_been_calculated(false),
    initial_perturbation_energy(1.0),
    start_time(0.0), //TODO
    initial_perturbation_energy_2(1.0), //TODO
    matrix_free_data(nullptr)
  {}

  void setup(MatrixFree<dim,Number> const &matrix_free_data_in,
             DofQuadIndexData const       &dof_quad_index_data_in,
             PerturbationEnergyData const &data_in)
  {
    matrix_free_data = &matrix_free_data_in;
    dof_quad_index_data = dof_quad_index_data_in;
    energy_data = data_in;
  }

  void evaluate(parallel::distributed::Vector<Number> const &velocity,
                double const                                &time,
                int const                                   &time_step_number)
  {
    if(energy_data.calculate == true)
    {
      if(time_step_number >= 0) // unsteady problem
        calculate_unsteady(velocity,time,time_step_number);
      else // steady problem (time_step_number = -1)
        calculate_steady(velocity);
    }
  }

private:
  bool clear_files;
  unsigned int counter;
  bool initial_perturbation_energy_has_been_calculated;
  Number initial_perturbation_energy;

  //TODO
  Number start_time;
  Number initial_perturbation_energy_2;

  MatrixFree<dim,Number> const * matrix_free_data;
  DofQuadIndexData dof_quad_index_data;
  PerturbationEnergyData energy_data;

  void calculate_unsteady(parallel::distributed::Vector<Number> const &velocity,
                          double const                                time,
                          unsigned int const                          time_step_number)
  {
    if((time_step_number-1)%energy_data.calculate_every_time_steps == 0)
    {
      Number perturbation_energy = 0.0;

      integrate(*matrix_free_data, velocity, perturbation_energy);

      if(!initial_perturbation_energy_has_been_calculated)
      {
        initial_perturbation_energy = perturbation_energy;
        initial_perturbation_energy_has_been_calculated = true;
      }

      // TODO
      // use pertubation energy after the first time step as reference value
      if(time_step_number == 2)
      {
        start_time = time;
        initial_perturbation_energy_2 = perturbation_energy;
      }

      // write output file
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      {
        unsigned int l = matrix_free_data->get_dof_handler(dof_quad_index_data.dof_index_velocity).get_triangulation().n_levels()-1;
        std::ostringstream filename;
        filename << energy_data.filename_prefix + "_l" + Utilities::int_to_string(l);

        std::ofstream f;
        if(clear_files == true)
        {
          f.open(filename.str().c_str(),std::ios::trunc);
          f << "Perturbation energy: E = (1,(u-u_base)^2)_Omega" << std::endl
            << "Error:               e = |exp(2*omega_i*t) - E(t)/E(0)|" << std::endl
            << "Error2:              e2 = |exp(2*omega_i*(t-t_start)) - E(t)/E(t_start)|" << std::endl; //TODO

          f << std::endl
//            << "  Time           energy         error" << std::endl
            << "  Time           energy         error          error2" << std::endl; //TODO

          clear_files = false;
        }
        else
        {
          f.open(filename.str().c_str(),std::ios::app);
        }

        Number const rel = perturbation_energy/initial_perturbation_energy;
        Number const error = std::abs(std::exp<Number>(2*energy_data.omega_i*time) - rel);

        //TODO
        Number const rel2 = perturbation_energy/initial_perturbation_energy_2;
        Number const error2 = std::abs(std::exp<Number>(2*energy_data.omega_i*(time-start_time)) - rel2);

        f << std::scientific << std::setprecision(7)
          << std::setw(15) << time
          << std::setw(15) << perturbation_energy
          << std::setw(15) << error
          << std::setw(15) << error2 //TODO
          << std::endl;
      }
    }
  }

  void calculate_steady(parallel::distributed::Vector<Number> const &velocity)
  {
    AssertThrow(false, ExcMessage("Calculation of perturbation energy for "
        "Orr-Sommerfeld problem only makes sense for unsteady problems."));
  }

  /*
   *  This function calculates the perturbation energy
   *
   *  Perturbation energy: E = (1,u*u)_Omega
   */
  void integrate(MatrixFree<dim,Number> const                &matrix_free_data,
                 parallel::distributed::Vector<Number> const &velocity,
                 Number                                      &energy)
  {
    std::vector<Number> dst(1,0.0);
    matrix_free_data.cell_loop (&PerturbationEnergyCalculator<dim,fe_degree,Number>::local_compute, this, dst, velocity);

    // sum over all MPI processes
    energy = Utilities::MPI::sum (dst.at(0), MPI_COMM_WORLD);
  }

  void local_compute(const MatrixFree<dim,Number>                &data,
                     std::vector<Number>                         &dst,
                     const parallel::distributed::Vector<Number> &src,
                     const std::pair<unsigned int,unsigned int>  &cell_range)
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,Number> fe_eval(data,
                                                           dof_quad_index_data.dof_index_velocity,
                                                           dof_quad_index_data.quad_index_velocity);

    AlignedVector<VectorizedArray<Number> > JxW_values(fe_eval.n_q_points);

    // Loop over all elements
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,false);
      fe_eval.fill_JxW_values(JxW_values);

      VectorizedArray<Number> energy_vec = make_vectorized_array<Number>(0.);
      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        Tensor<1,dim,VectorizedArray<Number> > velocity = fe_eval.get_value(q);
        Point<dim,VectorizedArray<Number> > q_points = fe_eval.quadrature_point(q);
        VectorizedArray<Number> y = q_points[1]/energy_data.h;
        Tensor<1,dim,VectorizedArray<Number> > velocity_base;
        velocity_base[0] = energy_data.U_max * (1.0 - y*y);
        energy_vec += JxW_values[q]*(velocity-velocity_base)*(velocity-velocity_base);
      }

      // sum over entries of VectorizedArray, but only over those
      // that are "active"
      for(unsigned int v=0; v<data.n_components_filled(cell); ++v)
      {
        dst.at(0) += energy_vec[v];
      }
    }
  }
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_PERTURBATION_ENERGY_ORR_SOMMERFELD_H_ */
