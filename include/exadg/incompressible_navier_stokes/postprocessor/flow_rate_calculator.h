/*
 * flow_rate_calculator.h
 *
 *  Created on: May 18, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_FLOW_RATE_CALCULATOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_FLOW_RATE_CALCULATOR_H_

#include <exadg/matrix_free/integrators.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim>
struct FlowRateCalculatorData
{
  FlowRateCalculatorData() : calculate(false), write_to_file(false), filename_prefix("flow_rate")
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    if(calculate == true)
    {
      pcout << "  Flow rate calculator:" << std::endl;

      print_parameter(pcout, "Calculate flow rate", calculate);
      print_parameter(pcout, "Write results to file", write_to_file);
      if(write_to_file == true)
        print_parameter(pcout, "Filename", filename_prefix);
    }
  }

  // calculate?
  bool calculate;

  // write results to file?
  bool write_to_file;

  // filename
  std::string filename_prefix;
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
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorU;
  typedef FaceIntegrator<dim, dim, Number> FaceIntegratorU;

  typedef VectorizedArray<Number> scalar;

  FlowRateCalculator(MatrixFree<dim, Number> const &     matrix_free_in,
                     unsigned int const                  dof_index_in,
                     unsigned int const                  quad_index_in,
                     FlowRateCalculatorData<dim> const & data_in,
                     MPI_Comm const &                    mpi_comm_in);

  Number
  calculate_flow_rates(VectorType const &                     velocity,
                       double const &                         time,
                       std::map<types::boundary_id, Number> & flow_rates);


private:
  void
  write_output(Number const & value, double const & time, std::string const & name);

  void
  do_calculate_flow_rates(VectorType const &                     velocity,
                          std::map<types::boundary_id, Number> & flow_rates);

  FlowRateCalculatorData<dim> const & data;
  MatrixFree<dim, Number> const &     matrix_free;
  unsigned int                        dof_index, quad_index;
  bool                                clear_files;

  MPI_Comm const & mpi_comm;
};

} // namespace IncNS
} // namespace ExaDG


#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_FLOW_RATE_CALCULATOR_H_ */
