/*
 * PostProcessorConvDiff.h
 *
 *  Created on: Oct 12, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_POSTPROCESSORCONVDIFF_H_
#define INCLUDE_POSTPROCESSORCONVDIFF_H_

#include "../include/OutputData.h"
#include "../include/ErrorCalculationData.h"
#include "../include/AnalyticalSolutionConvDiff.h"

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

  void do_postprocessing(parallel::distributed::Vector<double> const &solution,
                         double const                                time);

private:
  void calculate_error(parallel::distributed::Vector<double> const &solution,
                       double const                                time) const;

  void write_output(parallel::distributed::Vector<double> const &solution,
                    double const                                time) const;

  SmartPointer< DoFHandler<dim> const > dof_handler;
  SmartPointer< Mapping<dim> const > mapping;
  MatrixFree<dim,double> const * matrix_free_data;

  PostProcessorData pp_data;
  std_cxx11::shared_ptr<AnalyticalSolutionConvDiff<dim> > analytical_solution;

  unsigned int output_counter;
  unsigned int error_counter;
};

template<int dim, int fe_degree>
void PostProcessor<dim, fe_degree>::
do_postprocessing(parallel::distributed::Vector<double> const &solution_vector,
                  double const                                time)
{
  const double EPSILON = 1.0e-10; // small number which is much smaller than the time step size
  if( (pp_data.error_data.analytical_solution_available == true) &&
      (time > (pp_data.error_data.error_calc_start_time + error_counter*pp_data.error_data.error_calc_interval_time - EPSILON)) )
  {
    calculate_error(solution_vector,time);
    ++error_counter;
  }
  if( pp_data.output_data.write_output == true &&
      (time > (pp_data.output_data.output_start_time + output_counter*pp_data.output_data.output_interval_time - EPSILON)) )
  {
    write_output(solution_vector,time);
    ++output_counter;
  }
}

template<int dim, int fe_degree>
void PostProcessor<dim, fe_degree>::
write_output(parallel::distributed::Vector<double> const &solution_vector,
             double const                                time) const
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "OUTPUT << Write data at time t = " << std::scientific << std::setprecision(4) << time << std::endl;

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
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Calculate error at time t = " << std::scientific << std::setprecision(4) << time << ":" << std::endl;

  Vector<double> error_norm_per_cell (dof_handler->get_triangulation().n_active_cells());
  Vector<double> solution_norm_per_cell (dof_handler->get_triangulation().n_active_cells());

  analytical_solution->solution->set_time(time);
  VectorTools::integrate_difference (*mapping,
                                     *dof_handler,
                                     solution_vector,
                                     *(analytical_solution->solution),
                                     error_norm_per_cell,
                                     QGauss<dim>(dof_handler->get_fe().degree+2),
                                     VectorTools::L2_norm);

  parallel::distributed::Vector<double> dummy;
  dummy.reinit(solution_vector);
  VectorTools::integrate_difference (*mapping,
                                     *dof_handler,
                                     dummy,
                                     *(analytical_solution->solution),
                                     solution_norm_per_cell,
                                     QGauss<dim>(dof_handler->get_fe().degree+2),
                                     VectorTools::L2_norm);

  double error_norm = std::sqrt(Utilities::MPI::sum (error_norm_per_cell.norm_sqr(), MPI_COMM_WORLD));
  double solution_norm = std::sqrt(Utilities::MPI::sum (solution_norm_per_cell.norm_sqr(), MPI_COMM_WORLD));

  if(solution_norm > 1.e-12)
  {
    pcout << std::endl << "Relative error (L2-norm): "
          << std::setprecision(5) << std::setw(10) << error_norm/solution_norm
          << std::endl;
  }
  else
  {
    pcout << std::endl << "ABSOLUTE error (L2-norm): "
          << std::setprecision(5) << std::setw(10) << error_norm
          << std::endl;
  }
}

}


#endif /* INCLUDE_POSTPROCESSORCONVDIFF_H_ */
