/*
 * mean_velocity_calculator.h
 *
 *  Created on: Nov 20, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_MEAN_VELOCITY_CALCULATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_MEAN_VELOCITY_CALCULATOR_H_

#include <deal.II/matrix_free/fe_evaluation_notemplate.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <fstream>
#include <sstream>

#include "postprocessor_base.h"

namespace IncNS
{
template<int dim>
struct MeanVelocityCalculatorData
{
  MeanVelocityCalculatorData()
    : calculate(false),
      write_to_file(false),
      direction(Tensor<1, dim, double>()),
      filename_prefix("mean_velocity")
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    if(calculate == true)
    {
      pcout << "  Mean velocity/flow rate calculator:" << std::endl;

      print_parameter(pcout, "Calculate mean velocity/flow rate", calculate);
      print_parameter(pcout, "Write results to file", write_to_file);
      if(write_to_file == true)
        print_parameter(pcout, "Filename", filename_prefix);
    }
  }

  // calculate mean velocity?
  bool calculate;

  // Set containing boundary ID's of the surface area
  // for which we want to calculate the mean velocity.
  // This parameter is only relevant for area-based computation.
  std::set<types::boundary_id> boundary_IDs;

  // write results to file?
  bool write_to_file;

  // Direction in which we want to compute the flow rate
  // This parameter is only relevant for volume-based computation.
  Tensor<1, dim, double> direction;

  // filename
  std::string filename_prefix;
};

template<int dim, typename Number>
class MeanVelocityCalculator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorU;
  typedef FaceIntegrator<dim, dim, Number> FaceIntegratorU;

  typedef MeanVelocityCalculator<dim, Number> This;

  MeanVelocityCalculator(MatrixFree<dim, Number> const &         matrix_free_in,
                         DofQuadIndexData const &                dof_quad_index_data_in,
                         MeanVelocityCalculatorData<dim> const & data_in)
    : data(data_in),
      matrix_free(matrix_free_in),
      dof_quad_index_data(dof_quad_index_data_in),
      area_has_been_initialized(false),
      volume_has_been_initialized(false),
      area(0.0),
      volume(0.0),
      clear_files(true)
  {
  }

  Number
  calculate_mean_velocity_area(VectorType const & velocity, double const & time)
  {
    if(data.calculate == true)
    {
      if(area_has_been_initialized == false)
      {
        this->area = calculate_area();

        area_has_been_initialized = true;
      }

      Number flow_rate = do_calculate_flow_rate_area(velocity);

      AssertThrow(area_has_been_initialized == true, ExcMessage("Area has not been initialized."));
      AssertThrow(this->area != 0.0, ExcMessage("Area has not been initialized."));
      Number mean_velocity = flow_rate / this->area;

      write_output(mean_velocity, time, "Mean velocity [m/s]");

      return mean_velocity;
    }
    else
    {
      return -1.0;
    }
  }

  /*
   * Calculate mean velocity (only makes sense if the domain has a constant cross-section area in
   * streamwise direction
   */
  Number
  calculate_mean_velocity_volume(VectorType const & velocity, double const & time)
  {
    if(data.calculate == true)
    {
      if(volume_has_been_initialized == false)
      {
        this->volume = calculate_volume();

        volume_has_been_initialized = true;
      }

      Number mean_velocity = do_calculate_mean_velocity_volume(velocity);

      write_output(mean_velocity, time, "Mean velocity [m/s]");

      return mean_velocity;
    }
    else
    {
      return -1.0;
    }
  }

  /*
   * Calculate flow rate in m^3/s, for example for problems with non-constant cross-section area. To
   * obtain the flow rate, the length of the domain in streamwise direction has to be specified.
   */
  Number
  calculate_flow_rate_volume(VectorType const & velocity,
                             double const &     time,
                             double const &     length)
  {
    if(data.calculate == true)
    {
      Number flow_rate = 1.0 / length * do_calculate_flow_rate_volume(velocity);

      write_output(flow_rate, time, "Flow rate [m^3/s]");

      return flow_rate;
    }
    else
    {
      return -1.0;
    }
  }

  Number
  calculate_flow_rate_area(VectorType const & velocity, double const & time)
  {
    if(data.calculate == true)
    {
      Number flow_rate = do_calculate_flow_rate_area(velocity);

      write_output(flow_rate, time, "Flow rate [m^3/s]");

      return flow_rate;
    }
    else
    {
      return -1.0;
    }
  }


private:
  void
  write_output(Number const & value, double const & time, std::string const & name)
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

  Number
  calculate_area() const
  {
    FaceIntegratorU integrator(matrix_free,
                               true,
                               dof_quad_index_data.dof_index_velocity,
                               dof_quad_index_data.quad_index_velocity);

    AlignedVector<VectorizedArray<Number>> JxW_values(integrator.n_q_points);

    Number area = 0.0;

    for(unsigned int face = matrix_free.n_inner_face_batches();
        face < (matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches());
        face++)
    {
      typename std::set<types::boundary_id>::iterator it;
      types::boundary_id boundary_id = matrix_free.get_boundary_id(face);

      it = data.boundary_IDs.find(boundary_id);
      if(it != data.boundary_IDs.end())
      {
        integrator.reinit(face);
        integrator.fill_JxW_values(JxW_values);

        VectorizedArray<Number> area_local = make_vectorized_array<Number>(0.0);

        for(unsigned int q = 0; q < integrator.n_q_points; ++q)
        {
          area_local += JxW_values[q];
        }

        // sum over all entries of VectorizedArray
        for(unsigned int n = 0; n < matrix_free.n_active_entries_per_face_batch(face); ++n)
          area += area_local[n];
      }
    }

    area = Utilities::MPI::sum(area, MPI_COMM_WORLD);

    return area;
  }

  Number
  calculate_volume()
  {
    std::vector<Number> dst(1, 0.0);

    VectorType src_dummy;
    matrix_free.cell_loop(&This::local_calculate_volume, this, dst, src_dummy);

    // sum over all MPI processes
    Number volume = 1.0;
    volume        = Utilities::MPI::sum(dst.at(0), MPI_COMM_WORLD);

    return volume;
  }

  void
  local_calculate_volume(MatrixFree<dim, Number> const & data,
                         std::vector<Number> &           dst,
                         VectorType const &,
                         std::pair<unsigned int, unsigned int> const & cell_range)
  {
    CellIntegratorU integrator(data,
                               dof_quad_index_data.dof_index_velocity,
                               dof_quad_index_data.quad_index_velocity);

    AlignedVector<VectorizedArray<Number>> JxW_values(integrator.n_q_points);

    Number volume = 0.;

    // Loop over all elements
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);
      integrator.fill_JxW_values(JxW_values);

      VectorizedArray<Number> volume_vec = make_vectorized_array<Number>(0.);

      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        volume_vec += JxW_values[q];
      }

      // sum over entries of VectorizedArray, but only over those that are "active"
      for(unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
      {
        volume += volume_vec[v];
      }
    }

    dst.at(0) += volume;
  }

  Number
  do_calculate_flow_rate_area(VectorType const & velocity)
  {
    FaceIntegratorU integrator(matrix_free,
                               true,
                               dof_quad_index_data.dof_index_velocity,
                               dof_quad_index_data.quad_index_velocity);

    AlignedVector<VectorizedArray<Number>> JxW_values(integrator.n_q_points);

    // initialize with zero since we accumulate into this variable
    Number flow_rate = 0.0;

    for(unsigned int face = matrix_free.n_inner_face_batches();
        face < (matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches());
        face++)
    {
      typename std::set<types::boundary_id>::iterator it;
      types::boundary_id boundary_id = matrix_free.get_boundary_id(face);

      it = data.boundary_IDs.find(boundary_id);
      if(it != data.boundary_IDs.end())
      {
        integrator.reinit(face);
        integrator.read_dof_values(velocity);
        integrator.evaluate(true, false);
        integrator.fill_JxW_values(JxW_values);

        VectorizedArray<Number> flow_rate_face = make_vectorized_array<Number>(0.0);

        for(unsigned int q = 0; q < integrator.n_q_points; ++q)
        {
          flow_rate_face +=
            JxW_values[q] * integrator.get_value(q) * integrator.get_normal_vector(q);
        }

        // sum over all entries of VectorizedArray
        for(unsigned int n = 0; n < matrix_free.n_active_entries_per_face_batch(face); ++n)
          flow_rate += flow_rate_face[n];
      }
    }

    flow_rate = Utilities::MPI::sum(flow_rate, MPI_COMM_WORLD);

    return flow_rate;
  }

  Number
  do_calculate_mean_velocity_volume(VectorType const & velocity)
  {
    std::vector<Number> dst(1, 0.0);
    matrix_free.cell_loop(&This::local_calculate_flow_rate_volume, this, dst, velocity);

    // sum over all MPI processes
    Number mean_velocity = Utilities::MPI::sum(dst.at(0), MPI_COMM_WORLD);

    AssertThrow(volume_has_been_initialized == true,
                ExcMessage("Volume has not been initialized."));
    AssertThrow(this->volume != 0.0, ExcMessage("Volume has not been initialized."));

    mean_velocity /= this->volume;

    return mean_velocity;
  }

  Number
  do_calculate_flow_rate_volume(VectorType const & velocity)
  {
    std::vector<Number> dst(1, 0.0);
    matrix_free.cell_loop(&This::local_calculate_flow_rate_volume, this, dst, velocity);

    // sum over all MPI processes
    Number flow_rate_times_length = Utilities::MPI::sum(dst.at(0), MPI_COMM_WORLD);

    return flow_rate_times_length;
  }

  void
  local_calculate_flow_rate_volume(MatrixFree<dim, Number> const &               data,
                                   std::vector<Number> &                         dst,
                                   VectorType const &                            src,
                                   std::pair<unsigned int, unsigned int> const & cell_range)
  {
    CellIntegratorU integrator(data,
                               dof_quad_index_data.dof_index_velocity,
                               dof_quad_index_data.quad_index_velocity);

    AlignedVector<VectorizedArray<Number>> JxW_values(integrator.n_q_points);

    Number flow_rate = 0.;

    // Loop over all elements
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);
      integrator.read_dof_values(src);
      integrator.evaluate(true, false);
      integrator.fill_JxW_values(JxW_values);

      VectorizedArray<Number> flow_rate_vec = make_vectorized_array<Number>(0.);

      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        VectorizedArray<Number> velocity_direction = this->data.direction * integrator.get_value(q);

        flow_rate_vec += velocity_direction * JxW_values[q];
      }

      // sum over entries of VectorizedArray, but only over those
      // that are "active"
      for(unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
      {
        flow_rate += flow_rate_vec[v];
      }
    }

    dst.at(0) += flow_rate;
  }

  MeanVelocityCalculatorData<dim> const & data;
  MatrixFree<dim, Number> const &         matrix_free;
  DofQuadIndexData                        dof_quad_index_data;
  bool                                    area_has_been_initialized, volume_has_been_initialized;
  double                                  area, volume;
  bool                                    clear_files;
};

/*
 * In contrast to the above class, a vector of flow rates is calculated where the different entries
 * of the vector correspond to different boundary IDs, i.e., one outflow boundary may only consist
 * of faces with the same boundary ID
 */
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

  // Set containing boundary ID's of the surface area
  // for which we want to calculate the mean velocity.
  std::set<types::boundary_id> boundary_IDs;

  // write results to file?
  bool write_to_file;

  // filename
  std::string filename_prefix;
};

template<int dim, typename Number>
class FlowRateCalculator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorU;
  typedef FaceIntegrator<dim, dim, Number> FaceIntegratorU;

  FlowRateCalculator(MatrixFree<dim, Number> const &     matrix_free_in,
                     const DoFHandler<dim> &             dof_handler_velocity_in,
                     DofQuadIndexData const &            dof_quad_index_data_in,
                     FlowRateCalculatorData<dim> const & data_in)
    : data(data_in),
      matrix_free(matrix_free_in),
      dof_quad_index_data(dof_quad_index_data_in),
      clear_files(true),
      communicator(dynamic_cast<const parallel::Triangulation<dim> *>(
                     &dof_handler_velocity_in.get_triangulation()) ?
                     (dynamic_cast<const parallel::Triangulation<dim> *>(
                        &dof_handler_velocity_in.get_triangulation())
                        ->get_communicator()) :
                     MPI_COMM_SELF)
  {
  }

  Number
  calculate_flow_rates(VectorType const &                     velocity,
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


private:
  void
  write_output(Number const & value, double const & time, std::string const & name)
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

  void
  do_calculate_flow_rates(VectorType const &                     velocity,
                          std::map<types::boundary_id, Number> & flow_rates)
  {
    // zero flow rates since we sum into these variables
    for(auto iterator = flow_rates.begin(); iterator != flow_rates.end(); ++iterator)
    {
      iterator->second = 0.0;
    }

    FaceIntegratorU integrator(matrix_free,
                               true,
                               dof_quad_index_data.dof_index_velocity,
                               dof_quad_index_data.quad_index_velocity);

    AlignedVector<VectorizedArray<Number>> JxW_values(integrator.n_q_points);

    for(unsigned int face = matrix_free.n_inner_face_batches();
        face < (matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches());
        face++)
    {
      typename std::set<types::boundary_id>::iterator it;
      types::boundary_id boundary_id = matrix_free.get_boundary_id(face);

      it = data.boundary_IDs.find(boundary_id);
      if(it != data.boundary_IDs.end())
      {
        integrator.reinit(face);
        integrator.read_dof_values(velocity);
        integrator.evaluate(true, false);
        integrator.fill_JxW_values(JxW_values);

        VectorizedArray<Number> flow_rate_face = make_vectorized_array<Number>(0.0);

        for(unsigned int q = 0; q < integrator.n_q_points; ++q)
        {
          flow_rate_face +=
            JxW_values[q] * integrator.get_value(q) * integrator.get_normal_vector(q);
        }

        // sum over all entries of VectorizedArray
        for(unsigned int n = 0; n < matrix_free.n_active_entries_per_face_batch(face); ++n)
          flow_rates.at(boundary_id) += flow_rate_face[n];
      }
    }

    std::vector<Number> flow_rates_vector(flow_rates.size());
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

  FlowRateCalculatorData<dim> const & data;
  MatrixFree<dim, Number> const &     matrix_free;
  DofQuadIndexData                    dof_quad_index_data;
  bool                                clear_files;

  MPI_Comm communicator;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_MEAN_VELOCITY_CALCULATOR_H_ */
