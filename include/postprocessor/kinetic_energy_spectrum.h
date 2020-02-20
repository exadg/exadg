/*
 * energy_spectrum_calculation.h
 *
 *  Created on: Feb 7, 2018
 *      Author: fehn/muench
 */

#ifndef INCLUDE_POSTPROCESSOR_KINETIC_ENERGY_SPECTRUM_H_
#define INCLUDE_POSTPROCESSOR_KINETIC_ENERGY_SPECTRUM_H_

// deal.II
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/la_parallel_vector.h>
#include "deal.II/matrix_free/matrix_free.h"

#include "../functionalities/print_functions.h"

// kinetic energy spectrum data

struct KineticEnergySpectrumData
{
  KineticEnergySpectrumData()
    : calculate(false),
      write_raw_data_to_files(false),
      do_fftw(true),
      start_time(0.0),
      calculate_every_time_steps(-1),
      calculate_every_time_interval(-1.0),
      filename("energy_spectrum"),
      clear_file(true),
      degree(0),
      evaluation_points_per_cell(0),
      exploit_symmetry(false),
      n_cells_1d_coarse_grid(1),
      refine_level(0),
      length_symmetric_domain(numbers::PI)

  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    if(calculate == true)
    {
      pcout << std::endl << "  Calculate kinetic energy spectrum:" << std::endl;
      print_parameter(pcout, "Write raw data to files", write_raw_data_to_files);
      print_parameter(pcout, "Do FFTW", do_fftw);
      print_parameter(pcout, "Start time", start_time);
      if(calculate_every_time_steps >= 0)
        print_parameter(pcout, "Calculate every timesteps", calculate_every_time_steps);
      if(calculate_every_time_interval >= 0.0)
        print_parameter(pcout, "Calculate every time interval", calculate_every_time_interval);
      print_parameter(pcout, "Filename", filename);
      print_parameter(pcout, "Clear file", clear_file);

      print_parameter(pcout, "Evaluation points per cell", evaluation_points_per_cell);

      print_parameter(pcout, "Exploit symmetry", exploit_symmetry);
      if(exploit_symmetry)
      {
        print_parameter(pcout, "n_cells_1d_coarse_grid", n_cells_1d_coarse_grid);
        print_parameter(pcout, "refine_level", refine_level);
        print_parameter(pcout, "length_symmetric_domain", length_symmetric_domain);
      }
    }
  }

  bool   calculate;
  bool   write_raw_data_to_files;
  bool   do_fftw;
  double start_time;
  int    calculate_every_time_steps;
  double calculate_every_time_interval;

  // these parameters are only relevant if do_fftw = true
  std::string filename;
  bool        clear_file;

  unsigned int degree;
  unsigned int evaluation_points_per_cell;

  // exploit symmetry for Navier-Stokes simulation and mirror dof-vector
  // according to Taylor-Green symmetries for evaluation of energy spectrum.
  bool         exploit_symmetry;
  unsigned int n_cells_1d_coarse_grid;
  unsigned int refine_level;
  double       length_symmetric_domain;
};

template<int dim, typename Number>
class KineticEnergySpectrumCalculator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  KineticEnergySpectrumCalculator(MPI_Comm const & mpi_comm);

  void
  setup(MatrixFree<dim, Number> const &   matrix_free_data_in,
        DoFHandler<dim> const &           dof_handler_in,
        KineticEnergySpectrumData const & data_in);

  void
  evaluate(VectorType const & velocity, double const & time, int const & time_step_number);

private:
  bool
  needs_to_be_evaluated(double const time, unsigned int const time_step_number);

  void
  do_evaluate(VectorType const & velocity, double const time);

  MPI_Comm const & mpi_comm;

  bool                      clear_files;
  KineticEnergySpectrumData data;
  unsigned int              counter;
  bool                      reset_counter;
  const unsigned int        precision = 12;

  SmartPointer<DoFHandler<dim> const> dof_handler;

  std::shared_ptr<VectorType>                       velocity_full;
  std::shared_ptr<parallel::TriangulationBase<dim>> tria_full;
  std::shared_ptr<FESystem<dim>>                    fe_full;
  std::shared_ptr<DoFHandler<dim>>                  dof_handler_full;
};


#endif /* INCLUDE_POSTPROCESSOR_KINETIC_ENERGY_SPECTRUM_H_ */
