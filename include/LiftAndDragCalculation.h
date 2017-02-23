/*
 * LiftAndDragCalculation.h
 *
 *  Created on: Oct 14, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_LIFTANDDRAGCALCULATION_H_
#define INCLUDE_LIFTANDDRAGCALCULATION_H_


#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

#include <fstream>
#include <sstream>


template<int dim, int fe_degree_u, int fe_degree_p>
void calculate_lift_and_drag_force(MatrixFree<dim,double> const                &matrix_free_data,
                                   unsigned int const                          &dof_index_velocity,
                                   unsigned int const                          &quad_index_velocity,
                                   unsigned int const                          &dof_index_pressure,
                                   std::set<types::boundary_id> const          &boundary_IDs,
                                   parallel::distributed::Vector<double> const &velocity,
                                   parallel::distributed::Vector<double> const &pressure,
                                   double const                                &viscosity,
                                   Tensor<1,dim,double>                        &Force)
{
  FEFaceEvaluation<dim,fe_degree_u,fe_degree_u+1,dim,double> fe_eval_velocity
      (matrix_free_data,true,dof_index_velocity,quad_index_velocity);
  FEFaceEvaluation<dim,fe_degree_p,fe_degree_u+1,1,double> fe_eval_pressure
      (matrix_free_data,true,dof_index_pressure,quad_index_velocity);

  for(unsigned int d=0;d<dim;++d)
    Force[d] = 0.0;

  for(unsigned int face=matrix_free_data.n_macro_inner_faces();
      face<(matrix_free_data.n_macro_inner_faces()+matrix_free_data.n_macro_boundary_faces());
      face++)
  {
    fe_eval_velocity.reinit (face);
    fe_eval_velocity.read_dof_values(velocity);
    fe_eval_velocity.evaluate(false,true);

    fe_eval_pressure.reinit (face);
    fe_eval_pressure.read_dof_values(pressure);
    fe_eval_pressure.evaluate(true,false);

    typename std::set<types::boundary_id>::iterator it;
    types::boundary_id boundary_id = matrix_free_data.get_boundary_indicator(face);

    it = boundary_IDs.find(boundary_id);
    if (it != boundary_IDs.end())
    {
      for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
      {
        VectorizedArray<double> pressure = fe_eval_pressure.get_value(q);
        Tensor<1,dim,VectorizedArray<double> > normal = fe_eval_velocity.get_normal_vector(q);
        Tensor<2,dim,VectorizedArray<double> > velocity_gradient = fe_eval_velocity.get_gradient(q);
        fe_eval_velocity.submit_value(pressure*normal -  make_vectorized_array<double>(viscosity)*
                                        (velocity_gradient+transpose(velocity_gradient))*normal,q);
      }
      Tensor<1,dim,VectorizedArray<double> > Force_local = fe_eval_velocity.integrate_value();

      // sum over all entries of VectorizedArray
      for (unsigned int d=0; d<dim; ++d)
        for (unsigned int n=0; n<VectorizedArray<double>::n_array_elements; ++n)
          Force[d] += Force_local[d][n];
    }
  }
  Force = Utilities::MPI::sum(Force,MPI_COMM_WORLD);
}

template<int dim, int fe_degree_u, int fe_degree_p>
class LiftAndDragCalculator
{
public:
  LiftAndDragCalculator()
    :
    clear_files_lift_and_drag(true),
    matrix_free_data(nullptr)
  {}

  void setup(DoFHandler<dim> const  &dof_handler_velocity_in,
             MatrixFree<dim> const  &matrix_free_data_in,
             DofQuadIndexData const &dof_quad_index_data_in,
             LiftAndDragData const  &lift_and_drag_data_in)
  {
    dof_handler_velocity = &dof_handler_velocity_in;
    matrix_free_data = &matrix_free_data_in;
    dof_quad_index_data = dof_quad_index_data_in;
    lift_and_drag_data = lift_and_drag_data_in;
  }

  void evaluate(parallel::distributed::Vector<double> const &velocity,
                parallel::distributed::Vector<double> const &pressure,
                double const                                &time) const
   {
     if(lift_and_drag_data.calculate_lift_and_drag == true)
     {
       Tensor<1,dim,double> Force;

       calculate_lift_and_drag_force<dim,fe_degree_u,fe_degree_p>(
           *matrix_free_data,
           dof_quad_index_data.dof_index_velocity,
           dof_quad_index_data.quad_index_velocity,
           dof_quad_index_data.dof_index_pressure,
           lift_and_drag_data.boundary_IDs,
           velocity,
           pressure,
           lift_and_drag_data.viscosity,
           Force);

       // compute lift and drag coefficients (c = (F/rho)/(1/2 U² A)
       const double reference_value = lift_and_drag_data.reference_value;
       Force /= reference_value;

       if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
       {
         std::string filename_drag, filename_lift;
         filename_drag = "output/FPC/"
             + lift_and_drag_data.filename_prefix_drag
             + "_refine_" + Utilities::int_to_string(dof_handler_velocity->get_triangulation().n_levels()-1)
             + "_fe_degree_" + Utilities::int_to_string(fe_degree_u) + "-" + Utilities::int_to_string(fe_degree_p)
             + "_drag.txt";
         filename_lift = "output/FPC/"
             + lift_and_drag_data.filename_prefix_lift
             + "_refine_" + Utilities::int_to_string(dof_handler_velocity->get_triangulation().n_levels()-1)
             + "_fe_degree_" + Utilities::int_to_string(fe_degree_u) + "-" + Utilities::int_to_string(fe_degree_p)
             + "_lift.txt";

         std::ofstream f_drag, f_lift;
         if(clear_files_lift_and_drag)
         {
           f_drag.open(filename_drag.c_str(),std::ios::trunc);
           f_lift.open(filename_lift.c_str(),std::ios::trunc);
           clear_files_lift_and_drag = false;
         }
         else
         {
           f_drag.open(filename_drag.c_str(),std::ios::app);
           f_lift.open(filename_lift.c_str(),std::ios::app);
         }

         unsigned int precision = 12;

         f_drag << std::scientific << std::setprecision(precision)
                << time << "\t" << Force[0] << std::endl;
         f_drag.close();

         f_lift << std::scientific << std::setprecision(precision)
                << time << "\t" << Force[1] << std::endl;
         f_lift.close();
       }
     }
   }

private:
  mutable bool clear_files_lift_and_drag;

  SmartPointer< DoFHandler<dim> const > dof_handler_velocity;
  MatrixFree<dim,double> const * matrix_free_data;
  DofQuadIndexData dof_quad_index_data;
  LiftAndDragData lift_and_drag_data;

};


#endif /* INCLUDE_LIFTANDDRAGCALCULATION_H_ */
