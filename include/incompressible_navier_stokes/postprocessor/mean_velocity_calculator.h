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
  calculate_statistics(false),
  sample_start_time(0.0),
  sample_end_time(1.0),
  sample_every_timesteps(1),
  filename_prefix("indexa"),
  normal_vector(Tensor<1,dim,double>()),
  area(1.0)
 {}

 void print(ConditionalOStream &pcout)
 {
   if(calculate_statistics == true)
   {
     pcout << "  Mean centerline velocity:" << std::endl;
     print_parameter(pcout,"Calculate statistics",calculate_statistics);
     print_parameter(pcout,"Sample start time",sample_start_time);
     print_parameter(pcout,"Sample end time",sample_end_time);
     print_parameter(pcout,"Sample every timesteps",sample_every_timesteps);
     print_parameter(pcout,"Filename prefix",filename_prefix);

     for(unsigned int d=0; d<dim; ++d)
     {
       print_parameter(pcout,"Normal vector[" + Utilities::int_to_string(d) + "]",normal_vector[d]);
     }
   }
 }
  // calculate statistics?
  bool calculate_statistics;

  // start time for sampling
  double sample_start_time;

  // end time for sampling
  double sample_end_time;

  // perform sampling every ... timesteps
  unsigned int sample_every_timesteps;

  std::string filename_prefix;

  // set containing boundary ID's of the surface area for which we want to calculate the mean velocity
  std::set<types::boundary_id> boundary_IDs;

  Tensor<1,dim,double> normal_vector;

  // we need the area to calculate the flow rate: velocity = volume_flux / area
  double area;
};

template<int dim, int fe_degree, typename Number>
class MeanVelocityCalculator
{
public:
  MeanVelocityCalculator(const MatrixFree<dim,Number>          &matrix_free_data_in,
                         const DofQuadIndexData                &dof_quad_index_data_in,
                         const MeanVelocityCalculatorData<dim> &data_in)
    :
    data(data_in),
    matrix_free_data(matrix_free_data_in),
    dof_quad_index_data(dof_quad_index_data_in),
    mean_centerline_velocity(0.0),
    mean_centerline_velocity_sum(0.0),
    number_of_samples(0)
  {}

  void evaluate(const parallel::distributed::Vector<Number> &velocity)
  {

    Number mean_velocity;

    do_evaluate(matrix_free_data,
                dof_quad_index_data.dof_index_velocity,
                dof_quad_index_data.quad_index_velocity,
                velocity,
                mean_velocity);

    ++number_of_samples;

    mean_centerline_velocity_sum += mean_velocity;
    mean_centerline_velocity = mean_centerline_velocity_sum / (Number)number_of_samples;
  }

  void write_output (const std::string output_prefix)
  {
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    {
      std::ofstream f;
      f.open((output_prefix + "_mean_centerline_velocity.flow_statistics").c_str(),std::ios::trunc);
      f << "U_C = " << mean_centerline_velocity << std::endl;
    }
  }

  void do_evaluate(const MatrixFree<dim,Number>                &matrix_free_data,
                   unsigned int const                          &dof_index_velocity,
                   unsigned int const                          &quad_index_velocity,
                   const parallel::distributed::Vector<Number> &velocity,
                   Number                                      &mean_velocity)
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,Number>
      fe_eval_velocity(matrix_free_data,true,dof_index_velocity,quad_index_velocity);

    for(unsigned int face=matrix_free_data.n_macro_inner_faces();
        face<(matrix_free_data.n_macro_inner_faces()+matrix_free_data.n_macro_boundary_faces());
        face++)
    {
      fe_eval_velocity.reinit(face);
      fe_eval_velocity.read_dof_values(velocity);
      fe_eval_velocity.evaluate(true,false);

      typename std::set<types::boundary_id>::iterator it;
      types::boundary_id boundary_id = matrix_free_data.get_boundary_id(face);

      it = data.boundary_IDs.find(boundary_id);
      if (it != data.boundary_IDs.end())
      {
        for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
        {
          fe_eval_velocity.submit_value(fe_eval_velocity.get_value(q), q);
        }
        VectorizedArray<Number> mean_velocity_local = data.normal_vector * fe_eval_velocity.integrate_value();

        // sum over all entries of VectorizedArray
        for(unsigned int n=0; n<matrix_free_data.n_active_entries_per_face_batch(face); ++n)
          mean_velocity += mean_velocity_local[n];
      }
    }
    mean_velocity = Utilities::MPI::sum(mean_velocity,MPI_COMM_WORLD);
    mean_velocity = mean_velocity / data.area;
  }

private:
  MeanVelocityCalculatorData<dim> const &data;
  MatrixFree<dim,Number> const  &matrix_free_data;
  DofQuadIndexData              dof_quad_index_data;
  Number                        mean_centerline_velocity;
  Number                        mean_centerline_velocity_sum;
  unsigned int                  number_of_samples;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_MEAN_VELOCITY_CALCULATOR_H_ */
