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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_DIVERGENCE_AND_MASS_ERROR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_DIVERGENCE_AND_MASS_ERROR_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/matrix_free/integrators.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

struct MassConservationData
{
  MassConservationData()
    : calculate(false),
      start_time(std::numeric_limits<double>::max()),
      sample_every_time_steps(std::numeric_limits<unsigned int>::max()),
      directory("output/"),
      filename("mass"),
      reference_length_scale(1.0)
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    if(calculate == true)
    {
      pcout << "  Analysis of divergence and mass error:" << std::endl;
      print_parameter(pcout, "Start time", start_time);
      print_parameter(pcout, "Sample every timesteps", sample_every_time_steps);
      print_parameter(pcout, "Directory", directory);
      print_parameter(pcout, "Filename", filename);
    }
  }

  bool         calculate;
  double       start_time;
  unsigned int sample_every_time_steps;
  std::string  directory;
  std::string  filename;
  double       reference_length_scale;
};

template<int dim, typename Number>
class DivergenceAndMassErrorCalculator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorU;
  typedef FaceIntegrator<dim, dim, Number> FaceIntegratorU;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef DivergenceAndMassErrorCalculator<dim, Number> This;

  DivergenceAndMassErrorCalculator(MPI_Comm const & comm);

  void
  setup(MatrixFree<dim, Number> const & matrix_free_in,
        unsigned int const              dof_index_in,
        unsigned int const              quad_index_in,
        MassConservationData const &    data_in);

  void
  evaluate(VectorType const & velocity, double const & time, int const & time_step_number);

private:
  /*
   *  This function calculates the divergence error and the error of mass flux
   *  over interior element faces.
   *
   *  Divergence error: L * (1,|divu|)_Omega, L is a reference length scale
   *  Reference value for divergence error: (1,|| u ||)_Omega
   *
   *  and
   *
   *  Mass error: (1,|(um - up)*n|)_dOmegaI
   *  Reference value for mass error: (1,|0.5(um + up)*n|)_dOmegaI
   */
  void
  do_evaluate(MatrixFree<dim, Number> const & matrix_free,
              VectorType const &              velocity,
              Number &                        div_error,
              Number &                        div_error_reference,
              Number &                        mass_error,
              Number &                        mass_error_reference);

  void
  local_compute_div(MatrixFree<dim, Number> const &               data,
                    std::vector<Number> &                         dst,
                    VectorType const &                            source,
                    const std::pair<unsigned int, unsigned int> & cell_range);

  void
  local_compute_div_face(MatrixFree<dim, Number> const &               data,
                         std::vector<Number> &                         dst,
                         VectorType const &                            source,
                         const std::pair<unsigned int, unsigned int> & face_range);

  // not needed
  void
  local_compute_div_boundary_face(MatrixFree<dim, Number> const &,
                                  std::vector<Number> &,
                                  VectorType const &,
                                  const std::pair<unsigned int, unsigned int> &);

  void
  analyze_div_and_mass_error_unsteady(VectorType const & velocity,
                                      double const       time,
                                      unsigned int const time_step_number);

  void
  analyze_div_and_mass_error_steady(VectorType const & velocity);

  MPI_Comm const mpi_comm;

  bool   clear_files_mass_error;
  int    number_of_samples;
  Number divergence_sample;
  Number mass_sample;

  MatrixFree<dim, Number> const * matrix_free;
  unsigned int                    dof_index, quad_index;
  MassConservationData            data;
};


} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_DIVERGENCE_AND_MASS_ERROR_H_ */
