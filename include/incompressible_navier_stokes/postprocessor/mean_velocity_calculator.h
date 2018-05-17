/*
 * mean_velocity_calculator.h
 *
 *  Created on: Nov 20, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_MEAN_VELOCITY_CALCULATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_MEAN_VELOCITY_CALCULATOR_H_

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

#include <fstream>
#include <sstream>

#include "postprocessor_base.h"

template<int dim>
struct MeanVelocityCalculatorData
{
  MeanVelocityCalculatorData()
  :
  calculate(false),
  write_to_file(false),
  filename_prefix("mean_velocity")
 {}

 void print(ConditionalOStream &pcout)
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

  // set containing boundary ID's of the surface area
  // for which we want to calculate the mean velocity
  std::set<types::boundary_id> boundary_IDs;

  // write results to file?
  bool write_to_file;

  // filename
  std::string filename_prefix;
};

template<int dim, int fe_degree, typename Number>
class MeanVelocityCalculator
{
public:
  MeanVelocityCalculator(MatrixFree<dim,Number> const          &matrix_free_data_in,
                         DofQuadIndexData const                &dof_quad_index_data_in,
                         MeanVelocityCalculatorData<dim> const &data_in)
    :
    data(data_in),
    matrix_free_data(matrix_free_data_in),
    dof_quad_index_data(dof_quad_index_data_in),
    area_has_been_initialized(false),
    area(0.0),
    clear_files(true)
  {}

  Number calculate_mean_velocity(parallel::distributed::Vector<Number> const &velocity,
                                 double const                                &time)
  {
    if(data.calculate == true)
    {
      if(area_has_been_initialized == false)
      {
        this->area = calculate_area(matrix_free_data,
                                    dof_quad_index_data.dof_index_velocity,
                                    dof_quad_index_data.quad_index_velocity);

        area_has_been_initialized = true;
      }

      Number flow_rate = do_calculate_flow_rate(matrix_free_data,
                                                dof_quad_index_data.dof_index_velocity,
                                                dof_quad_index_data.quad_index_velocity,
                                                velocity);

      AssertThrow(area_has_been_initialized == true, ExcMessage("Area has not been initialized."));
      AssertThrow(this->area != 0.0, ExcMessage("Area has not been initialized."));
      Number mean_velocity = flow_rate / this->area;

//      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
//        std::cout << "Mean velocity = " << -mean_velocity << " [m^3/s]" << std::endl;

      // write output file
      if(data.write_to_file == true && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      {
        std::ostringstream filename;
        filename << data.filename_prefix;

        std::ofstream f;
        if(clear_files == true)
        {
          f.open(filename.str().c_str(),std::ios::trunc);
          f << std::endl << "  Time                Mean velocity [m/s]" << std::endl;

          clear_files = false;
        }
        else
        {
          f.open(filename.str().c_str(),std::ios::app);
        }

        unsigned int precision = 12;
        f << std::scientific << std::setprecision(precision)
          << std::setw(precision+8) << time
          << std::setw(precision+8) << mean_velocity << std::endl;
      }

      return mean_velocity;
    }
    else
    {
      return -1.0;
    }
  }

  Number calculate_flow_rate(parallel::distributed::Vector<Number> const &velocity,
                             double const                                &time)
  {
    if(data.calculate == true)
    {
      Number flow_rate = do_calculate_flow_rate(matrix_free_data,
                                                dof_quad_index_data.dof_index_velocity,
                                                dof_quad_index_data.quad_index_velocity,
                                                velocity);

//      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
//        std::cout << "Flow rate = " << -flow_rate << " [m^3/s]" << std::endl;

      // write output file
      if(data.write_to_file == true && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      {
        std::ostringstream filename;
        filename << data.filename_prefix;

        std::ofstream f;
        if(clear_files == true)
        {
          f.open(filename.str().c_str(),std::ios::trunc);
          f << std::endl << "  Time                Flow rate [m^3/s]" << std::endl;

          clear_files = false;
        }
        else
        {
          f.open(filename.str().c_str(),std::ios::app);
        }

        unsigned int precision = 12;
        f << std::scientific << std::setprecision(precision)
          << std::setw(precision+8) << time
          << std::setw(precision+8) << flow_rate << std::endl;
      }

      return flow_rate;
    }
    else
    {
      return -1.0;
    }
  }

  Number calculate_area(MatrixFree<dim,Number> const &matrix_free_data,
                        unsigned int const           &dof_index_velocity,
                        unsigned int const           &quad_index_velocity)
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,Number>
      fe_eval_velocity(matrix_free_data,true,dof_index_velocity,quad_index_velocity);

    AlignedVector<VectorizedArray<Number> > JxW_values(fe_eval_velocity.n_q_points);

    Number area = 0.0;

    for(unsigned int face=matrix_free_data.n_macro_inner_faces();
        face<(matrix_free_data.n_macro_inner_faces()+matrix_free_data.n_macro_boundary_faces());
        face++)
    {
      fe_eval_velocity.reinit(face);
      fe_eval_velocity.fill_JxW_values(JxW_values);

      typename std::set<types::boundary_id>::iterator it;
      types::boundary_id boundary_id = matrix_free_data.get_boundary_id(face);

      it = data.boundary_IDs.find(boundary_id);
      if (it != data.boundary_IDs.end())
      {
        VectorizedArray<Number> area_local = make_vectorized_array<Number>(0.0);

        for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
        {
          area_local += JxW_values[q];
        }

        // sum over all entries of VectorizedArray
        for(unsigned int n=0; n<matrix_free_data.n_active_entries_per_face_batch(face); ++n)
          area += area_local[n];
      }
    }

    area = Utilities::MPI::sum(area,MPI_COMM_WORLD);

    return area;
  }

  Number do_calculate_flow_rate(MatrixFree<dim,Number> const                &matrix_free_data,
                                unsigned int const                          &dof_index_velocity,
                                unsigned int const                          &quad_index_velocity,
                                parallel::distributed::Vector<Number> const &velocity)
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,Number>
      fe_eval_velocity(matrix_free_data,true,dof_index_velocity,quad_index_velocity);

    AlignedVector<VectorizedArray<Number> > JxW_values(fe_eval_velocity.n_q_points);

    // initialize with zero since we accumulate into this variable
    Number flow_rate = 0.0;

    for(unsigned int face=matrix_free_data.n_macro_inner_faces();
        face<(matrix_free_data.n_macro_inner_faces()+matrix_free_data.n_macro_boundary_faces());
        face++)
    {
      fe_eval_velocity.reinit(face);
      fe_eval_velocity.read_dof_values(velocity);
      fe_eval_velocity.evaluate(true,false);
      fe_eval_velocity.fill_JxW_values(JxW_values);

      typename std::set<types::boundary_id>::iterator it;
      types::boundary_id boundary_id = matrix_free_data.get_boundary_id(face);

      it = data.boundary_IDs.find(boundary_id);
      if (it != data.boundary_IDs.end())
      {
        VectorizedArray<Number> flow_rate_face = make_vectorized_array<Number>(0.0);

        for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
        {
          flow_rate_face += JxW_values[q]*fe_eval_velocity.get_value(q)*fe_eval_velocity.get_normal_vector(q);
        }

        // sum over all entries of VectorizedArray
        for(unsigned int n=0; n<matrix_free_data.n_active_entries_per_face_batch(face); ++n)
          flow_rate += flow_rate_face[n];
      }
    }

    flow_rate = Utilities::MPI::sum(flow_rate,MPI_COMM_WORLD);

    return flow_rate;
  }

private:
  MeanVelocityCalculatorData<dim> const &data;
  MatrixFree<dim,Number> const &matrix_free_data;
  DofQuadIndexData dof_quad_index_data;
  bool area_has_been_initialized;
  double area;
  bool clear_files;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_MEAN_VELOCITY_CALCULATOR_H_ */
