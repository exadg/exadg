/*
 * stress_calculation.h
 *
 *  Created on: Aug 30, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_H_


#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>
#include "../../postprocessor/evaluate_solution_in_given_point.h"


#include <fstream>
#include <sstream>

#include "../postprocessor/line_plot_data.h"


template<int dim, int fe_degree_u, int fe_degree_p, typename Number>
class LinePlotCalculator
{
public:
  LinePlotCalculator()
    :
    clear_files(true)
  {}

  void setup(DoFHandler<dim> const             &dof_handler_velocity_in,
             DoFHandler<dim> const             &dof_handler_pressure_in,
             Mapping<dim> const                &mapping_in,
             LinePlotData<dim> const           &line_plot_data_in)
  {
    dof_handler_velocity = &dof_handler_velocity_in;
    dof_handler_pressure = &dof_handler_pressure_in;
    mapping = &mapping_in;
    data = line_plot_data_in;
  }

  void evaluate(parallel::distributed::Vector<Number> const &velocity,
                parallel::distributed::Vector<Number> const &pressure) const
  {
    if(data.write_output == true)
    {
      // precision
      unsigned int const precision = data.precision;

      // loop over all lines
      for(typename std::vector<Line<dim> >::const_iterator
          line = data.lines.begin(); line != data.lines.end(); ++line)
      {
        // store all points along current line in a vector
        unsigned int n_points = line->n_points;
        std::vector<Point<dim> > points(n_points);
        for(unsigned int i=0; i<n_points; ++i)
          points[i] = line->begin + double(i)/double(n_points-1)*(line->end - line->begin);

        // filename prefix for current line
        std::string filename_prefix = data.filename_prefix
            + "l" + Utilities::int_to_string(dof_handler_velocity->get_triangulation().n_levels()-1)
            + "_ku" + Utilities::int_to_string(fe_degree_u) + "_kp" + Utilities::int_to_string(fe_degree_p)
            + "_" + line->name;

        // write output for all specified quantities
        for(std::vector<Quantity>::const_iterator
            quantity = line->quantities.begin(); quantity != line->quantities.end(); ++quantity)
        {
          if(quantity->type == QuantityType::Velocity)
          {
            std::vector<Tensor<1,dim,Number> > solution_vector(n_points);

            // calculate velocity for all points along line
            for (unsigned int i=0; i<n_points; ++i)
            {
              Tensor<1,dim,Number> u;
              evaluate_vectorial_quantity_in_point(*dof_handler_velocity,*mapping,velocity,points[i],u);
              solution_vector[i] = u;
            }

            // write output to file
            if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
            {
              std::string filename = filename_prefix + "_velocity" + ".txt";

              std::ofstream f;
              if(clear_files)
              {
                f.open(filename.c_str(),std::ios::trunc);
//                clear_files = false; // TODO
              }
              else
              {
                f.open(filename.c_str(),std::ios::app);
              }

              // headline
              for(unsigned int d=0; d<dim; ++d)
                f << std::setw(precision+8) << std::left << "x_" + Utilities::int_to_string(d+1);
              for(unsigned int d=0; d<dim; ++d)
                f << std::setw(precision+8) << std::left << "u_" + Utilities::int_to_string(d+1);
              f << std::endl;

              // loop over all points
              for (unsigned int i=0; i<n_points; ++i)
              {
                f << std::scientific << std::setprecision(precision);

                // write data
                for(unsigned int d=0; d<dim; ++d)
                  f << std::setw(precision+8) << std::left << points[i][d];
                for(unsigned int d=0; d<dim; ++d)
                  f << std::setw(precision+8) << std::left << solution_vector[i][d];
                f << std::endl;
              }
              f.close();
            }
          }
          else if(quantity->type == QuantityType::Pressure)
          {
            std::vector<Number> solution_vector(n_points);

            // calculate pressure for all points along line
            for (unsigned int i = 0; i<n_points; ++i)
            {
              Number p;
              evaluate_scalar_quantity_in_point(*dof_handler_pressure,*mapping,pressure,points[i],p);
              solution_vector[i] = p;
            }

            // write output to file
            if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
            {
              std::string filename = filename_prefix + "_pressure" + ".txt";

              std::ofstream f;
              if(clear_files)
              {
                f.open(filename.c_str(),std::ios::trunc);
//                clear_files = false; // TODO
              }
              else
              {
                f.open(filename.c_str(),std::ios::app);
              }

              // headline
              for(unsigned int d=0; d<dim; ++d)
                f << std::setw(precision+8) << std::left << "x_" + Utilities::int_to_string(d+1);
              f << std::setw(precision+8) << std::left << "p";
              f << std::endl;

              // loop over all points
              for (unsigned int i=0; i<n_points; ++i)
              {
                f << std::scientific << std::setprecision(precision);

                // write data
                for(unsigned int d=0; d<dim; ++d)
                  f << std::setw(precision+8) << std::left << points[i][d];
                f << std::setw(precision+8) << std::left << solution_vector[i];
                f << std::endl;
              }
              f.close();
            }
          }
        } // loop over quantities
      } // loop over lines
    } // write_output == true
  }

private:
  mutable bool clear_files;

  SmartPointer<DoFHandler<dim> const> dof_handler_velocity;
  SmartPointer<DoFHandler<dim> const> dof_handler_pressure;
  SmartPointer<Mapping<dim> const> mapping;

  LinePlotData<dim> data;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_H_ */
