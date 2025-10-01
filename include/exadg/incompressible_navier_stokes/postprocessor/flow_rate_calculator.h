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

#ifndef EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_FLOW_RATE_CALCULATOR_H_
#define EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_FLOW_RATE_CALCULATOR_H_

// ExaDG
#include <exadg/matrix_free/integrators.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim>
struct FlowRateCalculatorData
{
  FlowRateCalculatorData() : calculate(false), write_to_file(false), filename("flow_rate")
  {
  }

  void
  print(dealii::ConditionalOStream & pcout)
  {
    if(calculate == true)
    {
      pcout << "  Flow rate calculator:" << std::endl;

      print_parameter(pcout, "Calculate flow rate", calculate);
      print_parameter(pcout, "Write results to file", write_to_file);
      if(write_to_file == true)
      {
        print_parameter(pcout, "Directory", directory);
        print_parameter(pcout, "Filename", filename);
      }
    }
  }

  // calculate?
  bool calculate;

  // write results to file?
  bool write_to_file;

  // directory and filename
  std::string directory;
  std::string filename;
};


/*
 * This class calculates a vector of flow rates where the different entries of the vector correspond
 * to different boundary IDs, i.e., one outflow boundary may only consist of faces with the same
 * boundary ID. This class is intended to be used in cases where the number of outflow boundaries is
 * very large with a typical use case being the human lung (where the number of outflow boundaries
 * is 2^{N-1} with N being the number of airway generations). The reason behind is that calculating
 * the flow rate requires global communication since different processors may share the same outflow
 * boundary and the implementation in this class only requires one communication for all outflow
 * boundaries.
 *
 * Note: For a small number of outflow boundaries (for example 1-10), the more modular class
 * MeanVelocityCalculator should be used rather than this specialized implementation.
 */
template<int dim, typename Number>
class FlowRateCalculator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorU;
  typedef FaceIntegrator<dim, dim, Number> FaceIntegratorU;

  typedef dealii::VectorizedArray<Number> scalar;

  FlowRateCalculator(dealii::MatrixFree<dim, Number> const & matrix_free_in,
                     unsigned int const                      dof_index_in,
                     unsigned int const                      quad_index_in,
                     FlowRateCalculatorData<dim> const &     data_in,
                     MPI_Comm const &                        mpi_comm_in);

  Number
  calculate_flow_rates(VectorType const &                             velocity,
                       double const &                                 time,
                       std::map<dealii::types::boundary_id, Number> & flow_rates);


private:
  void
  write_output(Number const & value, double const & time, std::string const & name);

  void
  do_calculate_flow_rates(VectorType const &                             velocity,
                          std::map<dealii::types::boundary_id, Number> & flow_rates);

  FlowRateCalculatorData<dim> const &     data;
  dealii::MatrixFree<dim, Number> const & matrix_free;
  unsigned int                            dof_index, quad_index;
  bool                                    clear_files;

  MPI_Comm const mpi_comm;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_FLOW_RATE_CALCULATOR_H_ */
