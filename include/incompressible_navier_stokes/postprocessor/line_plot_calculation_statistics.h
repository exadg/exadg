/*
 * line_plot_calculation_statistics.h
 *
 *  Created on: Nov 17, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_STATISTICS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_STATISTICS_H_

#include "line_plot_data.h"

/*
 * This function calculates statistics along lines over time
 * and one spatial, homogeneous direction (averaging_direction = {0,1,2}), e.g.,
 * in the x-direction with a line in the y-z plane.
 *
 * NOTE: This function just works for geometries whose cells are aligned with the coordinate axis.
 */

//TODO Adapt code to geometries whose elements are not aligned with the coordinate axis.

template <int dim>
class LineStatisticsCalculator
{
public:
  LineStatisticsCalculator(const DoFHandler<dim> &dof_handler_velocity_in,
                           const DoFHandler<dim> &dof_handler_pressure_in,
                           const Mapping<dim>    &mapping_in)
    :
    clear_files(true),
    dof_handler_velocity(dof_handler_velocity_in),
    dof_handler_pressure(dof_handler_pressure_in),
    mapping (mapping_in),
    communicator (dynamic_cast<const parallel::Triangulation<dim>*>(&dof_handler_velocity.get_triangulation()) ?
                 (dynamic_cast<const parallel::Triangulation<dim>*>(&dof_handler_velocity.get_triangulation())
                  ->get_communicator()) : MPI_COMM_SELF),
    number_of_samples(0)
  {}

  void setup(LinePlotData<dim>  &line_statistics_data_in)
  {
    data = line_statistics_data_in;
    AssertThrow(dim==3,ExcMessage("Not implemented."));

    velocity_global.resize(data.lines.size());
    pressure_global.resize(data.lines.size());
    wall_shear_global.resize(data.lines.size());
    reynolds_global.resize(data.lines.size());
    global_points.resize(data.lines.size());

    //Iterator for lines
    unsigned int line_iterator = 0;
    for(typename std::vector<Line<dim> >::iterator line = data.lines.begin();
        line != data.lines.end(); ++line, ++line_iterator)
    {
      //Resize global variables for # of points on line
      velocity_global[line_iterator].resize(line->n_points);
      pressure_global[line_iterator].resize(line->n_points);
      wall_shear_global[line_iterator].resize(line->n_points);
      reynolds_global[line_iterator].resize(line->n_points);

      for (typename std::vector<Quantity*>::iterator quantity = line->quantities.begin();
           quantity != line->quantities.end(); ++quantity)
      {
        QuantityStatistics* stats_ptr = dynamic_cast<QuantityStatistics* > (*quantity);

        unsigned int const direction = stats_ptr->averaging_direction;

        AssertThrow(direction == 0 || direction == 1 || direction == 2,
            ExcMessage("Take the average either in x, y or z - direction"));
      }

      // use equidistant points along line
      for(unsigned int i = 0; i < line->n_points; ++i)
      {
        Point<dim> point = line->begin + double(i)/double(line->n_points-1)*(line->end - line->begin);
        global_points[line_iterator].push_back(point);
      }
    }
  }

  void evaluate(const parallel::distributed::Vector<double> &velocity,
                const parallel::distributed::Vector<double> &pressure)
  {
    std::vector<const parallel::distributed::Vector<double> *> velocity_vec;
    std::vector<const parallel::distributed::Vector<double> *> pressure_vec;
    velocity_vec.push_back(&velocity);
    pressure_vec.push_back(&pressure);

    do_evaluate(velocity_vec, pressure_vec);
  }

  void print_headline(std::ofstream      &f,
                      const unsigned int number_of_samples)
  {
    f << "number of samples: N = "  << number_of_samples << std::endl;
  }

  void write_output(const std::string &output_prefix)
  {
    if(Utilities::MPI::this_mpi_process(communicator)== 0 && data.write_output == true)
    {
      unsigned int const precision = data.precision;

      // Iterator for lines
      unsigned int line_iterator = 0;
      for(typename std::vector<Line<dim> >::iterator line = data.lines.begin();
          line != data.lines.end(); ++line, ++line_iterator)
      {
        std::string filename_prefix = output_prefix
                                      + "l" + Utilities::int_to_string(dof_handler_velocity.get_triangulation().n_global_levels()-1)
                                      + "_" + line->name + ".txt";

        for (typename std::vector<Quantity*>::iterator quantity = line->quantities.begin();
             quantity != line->quantities.end(); ++quantity)
        {
          if((*quantity)->type == QuantityType::Velocity)
          {
            std::string filename = filename_prefix + "_velocity" + ".txt";
            std::ofstream f;
            if(clear_files)
            {
              f.open(filename.c_str(),std::ios::trunc);
            }
            else
            {
              f.open(filename.c_str(),std::ios::app);
            }

            print_headline(f,number_of_samples);

            for(unsigned int d=0; d<dim; ++d)
              f << std::setw(precision+8) << std::left << "x_" + Utilities::int_to_string(d+1);
            for(unsigned int d=0; d<dim; ++d)
              f << std::setw(precision+8) << std::left << "u_" + Utilities::int_to_string(d+1);

            f << std::endl;

            // loop over all points
            for (unsigned int p=0; p<line->n_points; ++p)
            {
              f << std::scientific << std::setprecision(precision);

              // write data
              for(unsigned int d=0; d<dim; ++d)
                f << std::setw(precision+8) << std::left << global_points[line_iterator][p][d];

              // write velocity and average over time
              for(unsigned int d=0; d<dim; ++d)
                f << std::setw(precision+8) << std::left << velocity_global[line_iterator][p][d]/number_of_samples;

              f << std::endl;
            }
            f.close();
          }

          if((*quantity) -> type == QuantityType::ReynoldsStresses)
          {
            std::string filename = filename_prefix + "_reynoldsstresses" + ".txt";
            std::ofstream f;
            if(clear_files)
            {
              f.open(filename.c_str(),std::ios::trunc);
            }
            else
            {
              f.open(filename.c_str(),std::ios::app);
            }

            print_headline(f,number_of_samples);

            for(unsigned int d=0; d<dim; ++d)
              f << std::setw(precision+8) << std::left << "x_" + Utilities::int_to_string(d+1);

            for(unsigned int i=0; i<dim; ++i)
            {
              for(unsigned int j = 0; j<dim; ++j)
              {
                f << std::setw(precision+8) << std::left
                  << "u_" + Utilities::int_to_string(i+1) + "u_" + Utilities::int_to_string(j+1);
              }
            }
            f << std::endl;

            // loop over all points
            for (unsigned int p=0; p<line->n_points; ++p)
            {
              f << std::scientific << std::setprecision(precision);

              for(unsigned int d=0; d<dim; ++d)
                f << std::setw(precision+8) << std::left << global_points[line_iterator][p][d];

              for(unsigned int i=0; i<dim; ++i)
              {
                for(unsigned int j=0; j<dim; ++j)
                {
                  // equation <u_i' u_j'> = <u_i*u_j> - <u_i> * <u_j>
                  f << std::setw(precision+8) << std::left
                    << reynolds_global[line_iterator][p][i][j]/number_of_samples
                    - (velocity_global[line_iterator][p][i]*velocity_global[line_iterator][p][j])/(number_of_samples*number_of_samples);
                }
              }

              f << std::endl;
            }
            f.close();
          }

          if((*quantity) -> type == QuantityType::SkinFriction)
          {
            QuantityStatisticsSkinFriction<dim>* averaging_quantity =
                dynamic_cast<QuantityStatisticsSkinFriction<dim>* > (*quantity);

            std::string filename = filename_prefix + "_skinfriction" + ".txt";
            std::ofstream f;
            if(clear_files)
            {
              f.open(filename.c_str(),std::ios::trunc);
            }
            else
            {
              f.open(filename.c_str(),std::ios::app);
            }

            print_headline(f,number_of_samples);

            for(unsigned int d=0; d<dim; ++d)
              f << std::setw(precision+8) << std::left << "x_" + Utilities::int_to_string(d+1);

            f << std::setw(precision+8) << std::left << "C_f" << std::endl;

            // loop over all points
            for (unsigned int p=0; p<line->n_points; ++p)
            {
              f << std::scientific << std::setprecision(precision);

              // write data
              for(unsigned int d=0; d<dim; ++d)
                f << std::setw(precision+8) << std::left << global_points[line_iterator][p][d];

              double const ref_velocity_square = averaging_quantity->reference_velocity *
                                                 averaging_quantity->reference_velocity;

              // C_f = tau_w / (1/2 rho U^2)
              double const viscosity = averaging_quantity->viscosity;
              f << std::setw(precision+8) << std::left
                << 2.0*viscosity*wall_shear_global[line_iterator][p]/ref_velocity_square/number_of_samples;

              f << std::endl;
            }
            f.close();
          }

          if((*quantity) -> type == QuantityType::Pressure ||
             (*quantity) -> type == QuantityType::PressureCoefficient)
          {
            std::string filename = filename_prefix + "_pressure" + ".txt";
            std::ofstream f;

            if(clear_files)
            {
              f.open(filename.c_str(),std::ios::trunc);
            }
            else
            {
              f.open(filename.c_str(),std::ios::app);
            }

            print_headline(f,number_of_samples);

            for(unsigned int d=0; d<dim; ++d)
              f << std::setw(precision+8) << std::left << "x_" + Utilities::int_to_string(d+1);

            f << std::setw(precision+8) << std::left << "p";

            if((*quantity) -> type == QuantityType::PressureCoefficient)
              f << std::setw(precision+8) << std::left << "C_p";

            f << std::endl;

            for (unsigned int p=0; p<line->n_points; ++p)
            {
              f << std::scientific << std::setprecision(precision);

              for(unsigned int d=0; d<dim; ++d)
                f << std::setw(precision+8) << std::left << global_points[line_iterator][p][d];

              f << std::setw(precision+8) << std::left << pressure_global[line_iterator][p]/number_of_samples;

              if((*quantity)->type == QuantityType::PressureCoefficient)
              {
                QuantityStatisticsPressureCoefficient<dim>* averaging_quantity =
                    dynamic_cast<QuantityStatisticsPressureCoefficient<dim>* > (*quantity);

                double const ref_velocity_square = averaging_quantity->reference_velocity *
                                                   averaging_quantity->reference_velocity;

                // equation C_p = (p - p_0)/(1/2 rho U^2)
                f << std::left << 2.0*(pressure_global[line_iterator][p] - averaging_quantity->reference_pressure)/ref_velocity_square/number_of_samples;
              }
              f << std::endl;
            }
            f.close();
          }
        }
      }
    }
  }

private:
  void do_evaluate(const std::vector<const parallel::distributed::Vector<double> *> &velocity,
                   const std::vector<const parallel::distributed::Vector<double> *> &pressure)
  {
    // increment number of samples
    number_of_samples++;

    //Iterator for lines
    unsigned int line_iterator = 0;
    for(typename std::vector<Line<dim> >::iterator line = data.lines.begin();
        line != data.lines.end(); ++line, ++line_iterator)
    {
      for (typename std::vector<Quantity*>::iterator quantity = line->quantities.begin();
          quantity != line->quantities.end(); ++quantity)
      {
        //evaluate quantities that involve velocity
        if((*quantity)->type == QuantityType::Velocity ||
           (*quantity)->type == QuantityType::SkinFriction ||
           (*quantity)->type == QuantityType::ReynoldsStresses)
        {
          do_evaluate_velocity(velocity, *line, *quantity, line_iterator);
        }
        // ... or pressure
        if((*quantity)->type == QuantityType::Pressure ||
           (*quantity)->type == QuantityType::PressureCoefficient)
        {
          do_evaluate_pressure(pressure, *line, *quantity, line_iterator);
        }
      }
    }
  }

  void do_evaluate_velocity(const std::vector<const parallel::distributed::Vector<double> *>  &velocity,
                            const Line<dim>                                                   &line,
                            Quantity*                                                         quantity_base,
                            const unsigned int                                                line_iterator)
  {
    QuantityStatistics* quantity = dynamic_cast<QuantityStatistics* > (quantity_base);

    //Local variables for specific line
    std::vector<double> length_local(line.n_points);
    std::vector<Tensor<1, dim, double> > velocity_local(line.n_points);
    std::vector<double> wall_shear_local(line.n_points);
    std::vector<Tensor<2, dim, double> > reynolds_local(line.n_points);

    for(typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_velocity.begin_active();
        cell != dof_handler_velocity.end(); ++cell)
    {
      if(cell->is_locally_owned())
      {
        for(unsigned int p = 0; p < line.n_points; ++p)
        {
          // First, we move the line to the position of the current cell (vertex 0) in
          // averaging direction and check whether this new point is inside the current cell
          Point<dim> stats_point = global_points[line_iterator][p];
          unsigned int averaging_direction = quantity->averaging_direction;
          stats_point[averaging_direction] = cell->vertex(0)[averaging_direction];

          // If the new point lies in the current cell, we have to take the current cell into account
          if(cell->point_inside(stats_point))
          {
            const unsigned int fe_degree_velocity = dof_handler_velocity.get_fe().degree;

            // use quadrature for averaging in homogeneous direction
            QGauss<1> gauss_1d(fe_degree_velocity+1);
            std::vector<Point<dim> > points(gauss_1d.size()); // 1D points
            std::vector<double> weights(gauss_1d.size()); // 1D weights

            //Find points and weights for Gauss quadrature
            find_points_and_weights(cell, stats_point, points, weights, averaging_direction, gauss_1d);

            FEValues<dim,dim> fe_values(mapping,
                                        dof_handler_velocity.get_fe().base_element(0),
                                        Quadrature<dim>(points, weights),
                                        update_values | update_jacobians |
                                        update_quadrature_points |update_gradients);

            fe_values.reinit(typename Triangulation<dim>::active_cell_iterator(cell));

            const unsigned int scalar_dofs_per_cell = dof_handler_velocity.get_fe().base_element(0).dofs_per_cell;
            std::vector<Tensor<1,dim> > velocity_vector(scalar_dofs_per_cell);
            std::vector<types::global_dof_index> dof_indices(dof_handler_velocity.get_fe().dofs_per_cell);
            cell->get_dof_indices(dof_indices);

            // resort velocity dofs depending on the data structure used for the global
            // velocity vector
            if (dof_handler_velocity.get_fe().element_multiplicity(0) >= dim)
            {
              for (unsigned int j=0; j<dof_indices.size(); ++j)
              {
                const std::pair<unsigned int,unsigned int> comp =
                  dof_handler_velocity.get_fe().system_to_component_index(j);
                if (comp.first < dim)
                  velocity_vector[comp.second][comp.first] = (*velocity[0])(dof_indices[j]);
              }
            }
            else // scalar FE where we have several vectors referring to the same DoFHandler
            {
              AssertDimension(dof_handler_velocity.get_fe().element_multiplicity(0), 1);
              for (unsigned int j=0; j<scalar_dofs_per_cell; ++j)
                for (unsigned int d=0; d<dim; ++d)
                  velocity_vector[j][d] = (*velocity[d])(dof_indices[j]);
            }

            std::vector<std::vector<Tensor<1,dim, double> > > velocity_gradient(
                fe_values.n_quadrature_points, std::vector<Tensor<1,dim, double> >(dim));

            if(quantity->type == QuantityType::SkinFriction)
            {
              AssertThrow(dof_handler_velocity.get_fe().element_multiplicity(0) >= dim, ExcMessage("Not implemented."));
              FEValues<dim> fe_values_gradients(mapping,
                                                dof_handler_velocity.get_fe(),
                                                Quadrature<dim>(points, weights),
                                                update_values | update_jacobians |
                                                update_quadrature_points |update_gradients);

              fe_values_gradients.reinit(cell);
              // evaluate velocity gradient in all quadrature points
              fe_values_gradients.get_function_gradients(*velocity[0],velocity_gradient);
            }

            // perform averaging in homogeneous direction
            for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q)
            {
              double det = std::abs(fe_values.jacobian(q)[averaging_direction][averaging_direction]);
              double JxW = det * fe_values.get_quadrature().weight(q);

              // calculate integrals in homogeneous direction
              length_local[p] += JxW;

              Tensor<1,dim> velocity;

              if(quantity->type == QuantityType::Velocity ||
                 quantity->type == QuantityType::ReynoldsStresses)
              {
                // evaluate velocity solution in current quadrature points
                for (unsigned int j=0; j<velocity_vector.size(); ++j)
                  velocity += fe_values.shape_value(j,q) * velocity_vector[j];
              }

              if(quantity->type == QuantityType::Velocity)
              {
                for(unsigned int i=0; i<dim; ++i)
                  velocity_local[p][i] += velocity[i] * JxW;
              }

              if(quantity->type == QuantityType::ReynoldsStresses)
              {
                for(unsigned int i=0; i<dim; ++i)
                  for(unsigned int j=0; j<dim; ++j)
                    reynolds_local[p][i][j] += velocity[i] * velocity[j] * JxW;
              }

              if(quantity->type == QuantityType::SkinFriction)
              {
                QuantityStatisticsSkinFriction<dim>* quantity_skin_friction =
                    dynamic_cast<QuantityStatisticsSkinFriction<dim>* > (quantity);

                Tensor<1, dim, double> normal = quantity_skin_friction->normal_vector;
                Tensor<1, dim, double> tangent = quantity_skin_friction->tangent_vector;

                for (unsigned int i=0; i<dim; ++i)
                  for (unsigned int j=0; j<dim; ++j)
                    wall_shear_local[p] += tangent[i] * velocity_gradient[q][i][j] * normal[j] * JxW;
              }
            }
          }
        }
      }
    }

    Utilities::MPI::sum(length_local, communicator, length_local);

    // Cells are distributed over processors, therefore we need
    // to sum the contributions of every single processor.
    if(quantity->type == QuantityType::Velocity)
    {
      Utilities::MPI::sum(ArrayView<const double>(&velocity_local[0][0],dim*velocity_local.size()),
                          communicator,
                          ArrayView<double>(&velocity_local[0][0],dim*velocity_local.size()));

      for(unsigned int p=0; p<line.n_points; ++p)
      {
        for(unsigned int d=0; d<dim; ++d)
        {
          velocity_global[line_iterator][p][d] += velocity_local[p][d]/length_local[p];
        }
      }
    }

    if(quantity->type == QuantityType::ReynoldsStresses)
    {
      Utilities::MPI::sum(ArrayView<const double>(&reynolds_local[0][0][0],dim*dim*reynolds_local.size()),
                          communicator,
                          ArrayView<double>(&reynolds_local[0][0][0],dim*dim*reynolds_local.size()));

      for(unsigned int p=0; p<line.n_points; ++p)
      {
        for(unsigned int i=0; i<dim; ++i)
        {
          for(unsigned int j=0; j<dim; ++j)
          {
            reynolds_global[line_iterator][p][i][j] += reynolds_local[p][i][j]/length_local[p];
          }
        }
      }
    }

    if(quantity->type == QuantityType::SkinFriction)
    {
      Utilities::MPI::sum(wall_shear_local, communicator, wall_shear_local);

      for(unsigned int p = 0; p < line.n_points; ++p)
      {
        wall_shear_global[line_iterator][p] += wall_shear_local[p]/length_local[p];
      }
    }
  }

  void do_evaluate_pressure(const std::vector<const parallel::distributed::Vector<double> *>  &pressure,
                            const Line<dim>                                                   &line,
                            Quantity*                                                         quantity_base,
                            const unsigned int                                                line_iterator)
  {
    QuantityStatistics* quantity = dynamic_cast<QuantityStatistics* > (quantity_base);

    unsigned int number_of_points = line.n_points;

    // If we evaluate the pressure coefficient we need to evaluate an additional point
    // which is the last element in the vector
    if(quantity->type == QuantityType::PressureCoefficient)
    {
      number_of_points++;
    }

    //Local variables for specific line
    std::vector<double> length_local(number_of_points);
    std::vector<double> pressure_local(number_of_points);

    for(typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_pressure.begin_active();
        cell != dof_handler_pressure.end(); ++cell)
    {
      if(cell->is_locally_owned())
      {
        for(unsigned int p=0; p<number_of_points; ++p)
        {
          // First, we move the line to the position of the current cell (vertex 0) in
          // averaging direction and check whether this new point is inside the current cell
          Point<dim> stats_point;
          unsigned int averaging_direction = quantity->averaging_direction;

          //Evaluate reference point
          if(quantity->type == QuantityType::PressureCoefficient)
          {
            QuantityStatisticsPressureCoefficient<dim>* averaging_quantity =
                dynamic_cast<QuantityStatisticsPressureCoefficient<dim>* > (quantity);
            if(p < number_of_points-1)
              stats_point = global_points[line_iterator][p];
            else
              stats_point = averaging_quantity->reference_point;
          }
          else
            stats_point = global_points[line_iterator][p];

          stats_point[averaging_direction] = cell->vertex(0)[averaging_direction];

          //If the new point lies in the given cell we take the cell into account
          if(cell->point_inside(stats_point))
          {
            const unsigned int fe_degree_pressure = dof_handler_pressure.get_fe().degree;

            // use quadrature for averaging in homogeneous direction
            QGauss<1> gauss_1d(fe_degree_pressure+1);
            std::vector<Point<dim> > points(gauss_1d.size()); // 1D points
            std::vector<double> weights(gauss_1d.size()); // 1D weights

            //Find points and weights for Gauss quadrature
            find_points_and_weights(cell, stats_point, points, weights, averaging_direction, gauss_1d);

            FEValues<dim,dim> fe_values(mapping,
                                        dof_handler_pressure.get_fe().base_element(0),
                                        Quadrature<dim>(points, weights),
                                        update_values | update_jacobians |
                                        update_quadrature_points);
            fe_values.reinit(typename Triangulation<dim>::active_cell_iterator(cell));

            const unsigned int scalar_dofs_per_cell = dof_handler_pressure.get_fe().base_element(0).dofs_per_cell;
            std::vector<double> pressure_vector(scalar_dofs_per_cell);
            std::vector<types::global_dof_index> dof_indices(dof_handler_pressure.get_fe().dofs_per_cell);
            cell->get_dof_indices(dof_indices);

            for (unsigned int j=0; j<scalar_dofs_per_cell; ++j)
              pressure_vector[j] = (*pressure[0])(dof_indices[j]);

            for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q)
            {
              double pressure = 0;

              for (unsigned int j=0; j<pressure_vector.size(); ++j)
                pressure += fe_values.shape_value(j,q) * pressure_vector[j];

              double det = std::abs(fe_values.jacobian(q)[averaging_direction][averaging_direction]);
              double JxW = det * fe_values.get_quadrature().weight(q);

              length_local[p] += JxW;
              pressure_local[p] += pressure * JxW;
            }
          }
        }
      }
    }

    Utilities::MPI::sum(length_local, communicator, length_local);

    if(quantity->type == QuantityType::Pressure ||
       quantity->type == QuantityType::PressureCoefficient)
    {
      Utilities::MPI::sum(pressure_local, communicator, pressure_local);

      for(unsigned int p=0; p<line.n_points; ++p)
        pressure_global[line_iterator][p] += pressure_local[p]/length_local[p];
    }

    if(quantity->type == QuantityType::PressureCoefficient)
    {
      QuantityStatisticsPressureCoefficient<dim>* averaging_quantity =
          dynamic_cast<QuantityStatisticsPressureCoefficient<dim>* > (quantity);

      // the reference point is the last element of the vector with point along the line
      averaging_quantity->reference_pressure += pressure_local[line.n_points]/length_local[line.n_points];
    }
  }

  void find_points_and_weights (typename DoFHandler<dim>::active_cell_iterator const  cell,
                                Point<dim>                                            global_point,
                                std::vector<Point<dim> >                              &points,
                                std::vector<double>                                   &weights,
                                unsigned int                                          averaging_direction,
                                QGauss<1>                                             gauss_1d)
  {
    Point<dim> point_in_ref_coord = mapping.transform_real_to_unit_cell(cell, global_point);
    for(unsigned int q=0; q<gauss_1d.size(); ++q)
    {
      for(unsigned int d=0; d<dim; ++d)
      {
        if(d == averaging_direction)
          points[q][d] = gauss_1d.point(q)[0];
        else
          points[q][d] = point_in_ref_coord[d];
      }
      weights[q] = gauss_1d.weight(q);
    }
  }

  mutable bool clear_files;

  DoFHandler<dim> const &dof_handler_velocity;
  DoFHandler<dim> const &dof_handler_pressure;
  Mapping<dim>    const &mapping;
  MPI_Comm communicator;

  LinePlotData<dim> data;

  //Global points
  std::vector<std::vector<Point<dim> > > global_points;

  unsigned int number_of_samples;

  //Velocity quantities
  std::vector<std::vector<Tensor<1, dim, double> > > velocity_global;

  //Skin Friction quantities
  std::vector<std::vector<double> > wall_shear_global;

  //Reynolds Stress quantities
  std::vector<std::vector<Tensor<2, dim, double> > > reynolds_global;

  //Pressure quantities
  std::vector<std::vector<double> > pressure_global;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_STATISTICS_H_ */
