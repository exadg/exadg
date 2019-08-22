/*
 * flow_rate_calculator.cpp
 *
 *  Created on: May 18, 2019
 *      Author: fehn
 */

#include "flow_rate_calculator.h"

namespace IncNS
{
template<int dim, typename Number>
FlowRateCalculator<dim, Number>::FlowRateCalculator(MatrixFree<dim, Number> const & matrix_free_in,
                                                    const DoFHandler<dim> & dof_handler_velocity_in,
                                                    unsigned int const      dof_index_in,
                                                    unsigned int const      quad_index_in,
                                                    FlowRateCalculatorData<dim> const & data_in)
  : data(data_in),
    matrix_free(matrix_free_in),
    dof_index(dof_index_in),
    quad_index(quad_index_in),
    clear_files(true),
    communicator(dynamic_cast<const parallel::TriangulationBase<dim> *>(
                   &dof_handler_velocity_in.get_triangulation()) ?
                   (dynamic_cast<const parallel::TriangulationBase<dim> *>(
                      &dof_handler_velocity_in.get_triangulation())
                      ->get_communicator()) :
                   MPI_COMM_SELF)
{
}

template<int dim, typename Number>
Number
FlowRateCalculator<dim, Number>::calculate_flow_rates(
  VectorType const &                     velocity,
  double const &                         time,
  std::map<types::boundary_id, Number> & flow_rates)
{
  if(data.calculate == true)
  {
    do_calculate_flow_rates(velocity, flow_rates);

    // initialize with zero since we accumulate into this variable
    Number flow_rate = 0.0;
    for(auto it = flow_rates.begin(); it != flow_rates.end(); ++it)
    {
      flow_rate += it->second;
    }

    write_output(flow_rate, time, "Flow rate [m^3/s]");

    return flow_rate;
  }
  else
  {
    return -1.0;
  }
}

template<int dim, typename Number>
void
FlowRateCalculator<dim, Number>::write_output(Number const &      value,
                                              double const &      time,
                                              std::string const & name)
{
  // write output file
  if(data.write_to_file == true && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::ostringstream filename;
    filename << data.filename_prefix;

    std::ofstream f;
    if(clear_files == true)
    {
      f.open(filename.str().c_str(), std::ios::trunc);
      f << std::endl << "  Time                " + name << std::endl;

      clear_files = false;
    }
    else
    {
      f.open(filename.str().c_str(), std::ios::app);
    }

    unsigned int precision = 12;
    f << std::scientific << std::setprecision(precision) << std::setw(precision + 8) << time
      << std::setw(precision + 8) << value << std::endl;
  }
}

template<int dim, typename Number>
void
FlowRateCalculator<dim, Number>::do_calculate_flow_rates(
  VectorType const &                     velocity,
  std::map<types::boundary_id, Number> & flow_rates)
{
  // zero flow rates since we sum into these variables
  for(auto iterator = flow_rates.begin(); iterator != flow_rates.end(); ++iterator)
  {
    iterator->second = 0.0;
  }

  FaceIntegratorU integrator(matrix_free, true, dof_index, quad_index);

  for(unsigned int face = matrix_free.n_inner_face_batches();
      face < (matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches());
      face++)
  {
    typename std::map<types::boundary_id, Number>::iterator it;
    types::boundary_id boundary_id = matrix_free.get_boundary_id(face);

    it = flow_rates.find(boundary_id);
    if(it != flow_rates.end())
    {
      integrator.reinit(face);
      integrator.read_dof_values(velocity);
      integrator.evaluate(true, false);

      scalar flow_rate_face = make_vectorized_array<Number>(0.0);

      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        flow_rate_face +=
          integrator.JxW(q) * integrator.get_value(q) * integrator.get_normal_vector(q);
      }

      // sum over all entries of VectorizedArray
      for(unsigned int n = 0; n < matrix_free.n_active_entries_per_face_batch(face); ++n)
        flow_rates.at(boundary_id) += flow_rate_face[n];
    }
  }

  std::vector<double> flow_rates_vector(flow_rates.size());
  auto                iterator = flow_rates.begin();
  for(unsigned int counter = 0; counter < flow_rates.size(); ++counter)
  {
    flow_rates_vector[counter] = (iterator++)->second;
  }

  Utilities::MPI::sum(ArrayView<const double>(&(*flow_rates_vector.begin()),
                                              flow_rates_vector.size()),
                      communicator,
                      ArrayView<double>(&(*flow_rates_vector.begin()), flow_rates_vector.size()));

  iterator = flow_rates.begin();
  for(unsigned int counter = 0; counter < flow_rates.size(); ++counter)
  {
    (iterator++)->second = flow_rates_vector[counter];
  }
}

template class FlowRateCalculator<2, float>;
template class FlowRateCalculator<2, double>;

template class FlowRateCalculator<3, float>;
template class FlowRateCalculator<3, double>;

} // namespace IncNS
