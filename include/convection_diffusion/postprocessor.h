/*
 * PostProcessorConvDiff.h
 *
 *  Created on: Oct 12, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_H_
#define INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_H_

#include <deal.II/lac/parallel_vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <sstream>

#include "../convection_diffusion/analytical_solution.h"
#include "../include/postprocessor/output_data.h"
#include "../include/postprocessor/error_calculation_data.h"
#include "../include/postprocessor/calculate_l2_error.h"

namespace ConvDiff
{

struct PostProcessorData
{
  PostProcessorData(){}

  OutputData output_data;
  ErrorCalculationData error_data;
};

template<int dim, int fe_degree>
class PostProcessor
{
public:
  PostProcessor()
    :
    matrix_free_data(nullptr),
    output_counter(0),
    error_counter(0)
  {}

  void setup(PostProcessorData const                                  &postprocessor_data,
             DoFHandler<dim> const                                    &dof_handler_in,
             Mapping<dim> const                                       &mapping_in,
             MatrixFree<dim,double> const                             &matrix_free_data_in,
             std_cxx11::shared_ptr<AnalyticalSolutionConvDiff<dim> >  solution)
  {
    pp_data = postprocessor_data;
    dof_handler = &dof_handler_in;
    mapping = &mapping_in;
    matrix_free_data = &matrix_free_data_in;
    analytical_solution = solution;
  }

  // unsteady problems
  void do_postprocessing(parallel::distributed::Vector<double> const &solution,
                         double const                                time);

  // steady problems
  void do_postprocessing(parallel::distributed::Vector<double> const &solution);

private:
  void calculate_error(parallel::distributed::Vector<double> const &solution,
                       double const                                time) const;

  void write_output(parallel::distributed::Vector<double> const &solution) const;

  SmartPointer< DoFHandler<dim> const > dof_handler;
  SmartPointer< Mapping<dim> const > mapping;
  MatrixFree<dim,double> const * matrix_free_data;

  PostProcessorData pp_data;
  std_cxx11::shared_ptr<AnalyticalSolutionConvDiff<dim> > analytical_solution;

  unsigned int output_counter;
  unsigned int error_counter;
};

// unsteady problems
template<int dim, int fe_degree>
void PostProcessor<dim, fe_degree>::
do_postprocessing(parallel::distributed::Vector<double> const &solution_vector,
                  double const                                time)
{
  const double EPSILON = 1.0e-10; // small number which is much smaller than the time step size

  // calculate error
  if( (pp_data.error_data.analytical_solution_available == true) &&
      (time > (pp_data.error_data.error_calc_start_time + error_counter*pp_data.error_data.error_calc_interval_time - EPSILON)) )
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << std::endl << "Calculate error at time t = " << std::scientific << std::setprecision(4) << time << ":"
          << std::endl << std::endl;

    calculate_error(solution_vector,time);
    ++error_counter;
  }

  // write output
  if( pp_data.output_data.write_output == true &&
      (time > (pp_data.output_data.output_start_time + output_counter*pp_data.output_data.output_interval_time - EPSILON)) )
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << std::endl << "OUTPUT << Write data at time t = " << std::scientific << std::setprecision(4) << time
          << std::endl;

    write_output(solution_vector);
    ++output_counter;
  }
}

// steady problems
template<int dim, int fe_degree>
void PostProcessor<dim, fe_degree>::
do_postprocessing(parallel::distributed::Vector<double> const &solution_vector)
{
  // calculate error
  if(pp_data.error_data.analytical_solution_available == true)
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << std::endl << "Calculate error for " << (error_counter == 0 ? "initial" : "solution") << " data"
          << std::endl << std::endl;

    calculate_error(solution_vector,1.0);
    ++error_counter;
  }

  // write output
  if(pp_data.output_data.write_output == true )
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << std::endl << "OUTPUT << Write " << (output_counter == 0 ? "initial" : "solution") << " data"
          << std::endl;

    write_output(solution_vector);
    ++output_counter;
  }
}

template<int dim, int fe_degree>
void PostProcessor<dim, fe_degree>::
write_output(parallel::distributed::Vector<double> const &solution_vector) const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler (*dof_handler);
  data_out.add_data_vector (solution_vector, "solution");
  data_out.build_patches (pp_data.output_data.number_of_patches);

  const std::string filename = "output_conv_diff/" + pp_data.output_data.output_prefix + "_" + Utilities::int_to_string (output_counter, 3);

  std::ofstream output_data ((filename + ".vtu").c_str());
  data_out.write_vtu (output_data);
}

template<int dim, int fe_degree>
void PostProcessor<dim, fe_degree>::
calculate_error(parallel::distributed::Vector<double> const &solution_vector,
                double const                                time) const
{
  analytical_solution->solution->set_time(time);
  double error = 1.0;
  bool relative = true;

  calculate_L2_error<dim>(*dof_handler,
                          *mapping,
                          solution_vector,
                          analytical_solution->solution,
                          error,
                          relative,
                          3);

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << ((relative == true) ? "  Relative " : "  ABSOLUTE ") << "error (L2-norm): "
        << std::scientific << std::setprecision(5) << error << std::endl;
}

}


#endif /* INCLUDE_CONVECTION_DIFFUSION_POSTPROCESSOR_H_ */
