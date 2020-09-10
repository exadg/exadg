/*
 * perturbation_energy_orr_sommerfeld.h
 *
 *  Created on: Sep 1, 2017
 *      Author: fehn
 */

#ifndef SOLVERS_INCOMPRESSIBLE_NAVIER_STOKES_APPLICATIONS_ORR_SOMMERFELD_PERTURBATION_ENERGY_H_
#define SOLVERS_INCOMPRESSIBLE_NAVIER_STOKES_APPLICATIONS_ORR_SOMMERFELD_PERTURBATION_ENERGY_H_

// C/C++
#include <fstream>

// ExaDG
#include <exadg/matrix_free/integrators.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

struct PerturbationEnergyData
{
  PerturbationEnergyData()
    : calculate(false),
      calculate_every_time_steps(std::numeric_limits<unsigned int>::max()),
      filename_prefix("orr_sommerfeld"),
      omega_i(0.0),
      h(1.0),
      U_max(1.0)
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    if(calculate == true)
    {
      pcout << "  Calculate perturbation energy:" << std::endl;
      print_parameter(pcout, "Calculate perturbation energy", calculate);
      print_parameter(pcout, "Calculate every time steps", calculate_every_time_steps);
      print_parameter(pcout, "Filename output", filename_prefix);
      print_parameter(pcout, "Amplification omega_i", omega_i);
      print_parameter(pcout, "Channel height h", h);
      print_parameter(pcout, "Maximum velocity U_max", U_max);
    }
  }

  bool         calculate;
  unsigned int calculate_every_time_steps;
  std::string  filename_prefix;
  double       omega_i;
  double       h;
  double       U_max;
};

/*
 * Calculation of perturbation energy for Orr-Sommerfeld problem
 */
template<int dim, typename Number>
class PerturbationEnergyCalculator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef PerturbationEnergyCalculator<dim, Number> This;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  PerturbationEnergyCalculator(MPI_Comm const & comm)
    : mpi_comm(comm),
      clear_files(true),
      initial_perturbation_energy_has_been_calculated(false),
      initial_perturbation_energy(1.0),
      matrix_free_data(nullptr),
      dof_index(0),
      quad_index(0)
  {
  }

  void
  setup(MatrixFree<dim, Number> const & matrix_free_data_in,
        unsigned int const              dof_index_in,
        unsigned int const              quad_index_in,
        PerturbationEnergyData const &  data_in)
  {
    matrix_free_data = &matrix_free_data_in;
    dof_index        = dof_index_in;
    quad_index       = quad_index_in;
    energy_data      = data_in;
  }

  void
  evaluate(VectorType const & velocity, double const & time, int const & time_step_number)
  {
    if(energy_data.calculate == true)
    {
      if(time_step_number >= 0) // unsteady problem
      {
        do_evaluate(velocity, time, time_step_number);
      }
      else // steady problem (time_step_number = -1)
      {
        AssertThrow(false,
                    ExcMessage("Calculation of perturbation energy for "
                               "Orr-Sommerfeld problem only makes sense for unsteady problems."));
      }
    }
  }

private:
  void
  do_evaluate(VectorType const & velocity, double const time, unsigned int const time_step_number)
  {
    if((time_step_number - 1) % energy_data.calculate_every_time_steps == 0)
    {
      Number perturbation_energy = 0.0;

      integrate(*matrix_free_data, velocity, perturbation_energy);

      if(!initial_perturbation_energy_has_been_calculated)
      {
        initial_perturbation_energy = perturbation_energy;

        initial_perturbation_energy_has_been_calculated = true;
      }

      // write output file
      if(Utilities::MPI::this_mpi_process(mpi_comm) == 0)
      {
        // clang-format off
        unsigned int l = matrix_free_data->get_dof_handler(dof_index)
                           .get_triangulation().n_global_levels() - 1;
        // clang-format on

        std::ostringstream filename;
        filename << energy_data.filename_prefix + "_l" + Utilities::int_to_string(l);

        std::ofstream f;
        if(clear_files == true)
        {
          f.open(filename.str().c_str(), std::ios::trunc);
          f << "Perturbation energy: E = (1,(u-u_base)^2)_Omega" << std::endl
            << "Error:               e = |exp(2*omega_i*t) - E(t)/E(0)|" << std::endl;

          f << std::endl << "  Time                energy              error" << std::endl;

          clear_files = false;
        }
        else
        {
          f.open(filename.str().c_str(), std::ios::app);
        }

        Number const rel   = perturbation_energy / initial_perturbation_energy;
        Number const error = std::abs(std::exp(2 * energy_data.omega_i * time) - rel);

        unsigned int const precision = 12;
        f << std::scientific << std::setprecision(precision) << std::setw(precision + 8) << time
          << std::setw(precision + 8) << perturbation_energy << std::setw(precision + 8) << error
          << std::endl;
      }
    }
  }

  /*
   *  This function calculates the perturbation energy
   *
   *  Perturbation energy: E = (1,u*u)_Omega
   */
  void
  integrate(MatrixFree<dim, Number> const & matrix_free_data,
            VectorType const &              velocity,
            Number &                        energy)
  {
    std::vector<Number> dst(1, 0.0);
    matrix_free_data.cell_loop(&This::local_compute, this, dst, velocity);

    // sum over all MPI processes
    energy = Utilities::MPI::sum(dst.at(0), mpi_comm);
  }

  void
  local_compute(MatrixFree<dim, Number> const &               data,
                std::vector<Number> &                         dst,
                VectorType const &                            src,
                std::pair<unsigned int, unsigned int> const & cell_range)
  {
    CellIntegrator<dim, dim, Number> fe_eval(data, dof_index, quad_index);

    // Loop over all elements
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true, false);

      VectorizedArray<Number> energy_vec = make_vectorized_array<Number>(0.);
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        vector velocity = fe_eval.get_value(q);

        Point<dim, scalar> q_points = fe_eval.quadrature_point(q);

        scalar y = q_points[1] / energy_data.h;

        vector velocity_base;
        velocity_base[0] = energy_data.U_max * (1.0 - y * y);
        energy_vec += fe_eval.JxW(q) * (velocity - velocity_base) * (velocity - velocity_base);
      }

      // sum over entries of VectorizedArray, but only over those
      // that are "active"
      for(unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
      {
        dst.at(0) += energy_vec[v];
      }
    }
  }

  MPI_Comm const & mpi_comm;

  bool   clear_files;
  bool   initial_perturbation_energy_has_been_calculated;
  Number initial_perturbation_energy;

  MatrixFree<dim, Number> const * matrix_free_data;
  unsigned int                    dof_index, quad_index;
  PerturbationEnergyData          energy_data;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* SOLVERS_INCOMPRESSIBLE_NAVIER_STOKES_APPLICATIONS_ORR_SOMMERFELD_PERTURBATION_ENERGY_H_ \
        */
