/*
 * mean_velocity_calculator.h
 *
 *  Created on: Nov 20, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_MEAN_VELOCITY_CALCULATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_MEAN_VELOCITY_CALCULATOR_H_

#include <deal.II/matrix_free/fe_evaluation.h>
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

template<int dim, int fe_degree, typename Number>
class MeanVelocityCalculator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  MeanVelocityCalculator(MatrixFree<dim, Number> const &         matrix_free_data_in,
                         DofQuadIndexData const &                dof_quad_index_data_in,
                         MeanVelocityCalculatorData<dim> const & data_in)
    : data(data_in),
      matrix_free_data(matrix_free_data_in),
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

      Number flow_rate = do_calculate_flow_rate(velocity);

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
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, dim, Number> fe_eval_velocity(
      matrix_free_data,
      true,
      dof_quad_index_data.dof_index_velocity,
      dof_quad_index_data.quad_index_velocity);

    AlignedVector<VectorizedArray<Number>> JxW_values(fe_eval_velocity.n_q_points);

    Number area = 0.0;

    for(unsigned int face = matrix_free_data.n_macro_inner_faces();
        face < (matrix_free_data.n_macro_inner_faces() + matrix_free_data.n_macro_boundary_faces());
        face++)
    {
      fe_eval_velocity.reinit(face);
      fe_eval_velocity.fill_JxW_values(JxW_values);

      typename std::set<types::boundary_id>::iterator it;
      types::boundary_id boundary_id = matrix_free_data.get_boundary_id(face);

      it = data.boundary_IDs.find(boundary_id);
      if(it != data.boundary_IDs.end())
      {
        VectorizedArray<Number> area_local = make_vectorized_array<Number>(0.0);

        for(unsigned int q = 0; q < fe_eval_velocity.n_q_points; ++q)
        {
          area_local += JxW_values[q];
        }

        // sum over all entries of VectorizedArray
        for(unsigned int n = 0; n < matrix_free_data.n_active_entries_per_face_batch(face); ++n)
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
    matrix_free_data.cell_loop(
      &MeanVelocityCalculator<dim, fe_degree, Number>::local_calculate_volume,
      this,
      dst,
      src_dummy);

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
    FEEvaluation<dim, fe_degree, fe_degree + 1, dim, Number> fe_eval(
      data, dof_quad_index_data.dof_index_velocity, dof_quad_index_data.quad_index_velocity);

    AlignedVector<VectorizedArray<Number>> JxW_values(fe_eval.n_q_points);

    Number volume = 0.;

    // Loop over all elements
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.fill_JxW_values(JxW_values);

      VectorizedArray<Number> volume_vec = make_vectorized_array<Number>(0.);

      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
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
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, dim, Number> fe_eval_velocity(
      matrix_free_data,
      true,
      dof_quad_index_data.dof_index_velocity,
      dof_quad_index_data.quad_index_velocity);

    AlignedVector<VectorizedArray<Number>> JxW_values(fe_eval_velocity.n_q_points);

    // initialize with zero since we accumulate into this variable
    Number flow_rate = 0.0;

    for(unsigned int face = matrix_free_data.n_macro_inner_faces();
        face < (matrix_free_data.n_macro_inner_faces() + matrix_free_data.n_macro_boundary_faces());
        face++)
    {
      fe_eval_velocity.reinit(face);
      fe_eval_velocity.read_dof_values(velocity);
      fe_eval_velocity.evaluate(true, false);
      fe_eval_velocity.fill_JxW_values(JxW_values);

      typename std::set<types::boundary_id>::iterator it;
      types::boundary_id boundary_id = matrix_free_data.get_boundary_id(face);

      it = data.boundary_IDs.find(boundary_id);
      if(it != data.boundary_IDs.end())
      {
        VectorizedArray<Number> flow_rate_face = make_vectorized_array<Number>(0.0);

        for(unsigned int q = 0; q < fe_eval_velocity.n_q_points; ++q)
        {
          flow_rate_face +=
            JxW_values[q] * fe_eval_velocity.get_value(q) * fe_eval_velocity.get_normal_vector(q);
        }

        // sum over all entries of VectorizedArray
        for(unsigned int n = 0; n < matrix_free_data.n_active_entries_per_face_batch(face); ++n)
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
    matrix_free_data.cell_loop(
      &MeanVelocityCalculator<dim, fe_degree, Number>::local_calculate_flow_rate_volume,
      this,
      dst,
      velocity);

    // sum over all MPI processes
    Number mean_velocity = Utilities::MPI::sum(dst.at(0), MPI_COMM_WORLD);

    AssertThrow(volume_has_been_initialized == true,
                ExcMessage("Volume has not been initialized."));
    AssertThrow(this->volume != 0.0, ExcMessage("Volume has not been initialized."));

    mean_velocity /= this->volume;

    return mean_velocity;
  }

  void
  local_calculate_flow_rate_volume(MatrixFree<dim, Number> const &               data,
                                   std::vector<Number> &                         dst,
                                   VectorType const &                            src,
                                   std::pair<unsigned int, unsigned int> const & cell_range)
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, dim, Number> fe_eval(
      data, dof_quad_index_data.dof_index_velocity, dof_quad_index_data.quad_index_velocity);

    AlignedVector<VectorizedArray<Number>> JxW_values(fe_eval.n_q_points);

    Number flow_rate = 0.;

    // Loop over all elements
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true, false);
      fe_eval.fill_JxW_values(JxW_values);

      VectorizedArray<Number> flow_rate_vec = make_vectorized_array<Number>(0.);

      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        VectorizedArray<Number> velocity_direction = this->data.direction * fe_eval.get_value(q);

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
  MatrixFree<dim, Number> const &         matrix_free_data;
  DofQuadIndexData                        dof_quad_index_data;
  bool                                    area_has_been_initialized, volume_has_been_initialized;
  double                                  area, volume;
  bool                                    clear_files;
};


} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_MEAN_VELOCITY_CALCULATOR_H_ */
