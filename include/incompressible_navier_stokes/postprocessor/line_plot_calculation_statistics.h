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
  typedef typename std::vector<std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> > > TYPE;

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
    number_of_samples(0),
    averaging_direction(2)
  {}

  void setup(LinePlotData<dim> const &line_statistics_data_in)
  {
    // use a tolerance to check whether a point is inside the unit cell
    double const tolerance = 1.e-12;

    data = line_statistics_data_in;
    AssertThrow(dim==3,ExcMessage("Not implemented."));

    velocity_global.resize(data.lines.size());
    pressure_global.resize(data.lines.size());
    reference_pressure_global.resize(data.lines.size());
    wall_shear_global.resize(data.lines.size());
    reynolds_global.resize(data.lines.size());
    global_points.resize(data.lines.size());
    cells_and_ref_points_velocity.resize(data.lines.size());
    cells_and_ref_points_pressure.resize(data.lines.size());
    ref_pressure_cells_and_ref_points.resize(data.lines.size());

    //Iterator for lines
    unsigned int line_iterator = 0;
    for(typename std::vector<Line<dim> >::iterator line = data.lines.begin();
        line != data.lines.end(); ++line, ++line_iterator)
    {
      cells_and_ref_points_velocity[line_iterator].resize(line->n_points);
      cells_and_ref_points_pressure[line_iterator].resize(line->n_points);
    }

    // initialize homogeneous direction: use the first line and the first quantity since all
    // quantities are averaged over the same homogeneous direction for all lines
    AssertThrow(data.lines.size()>0, ExcMessage("Empty data"));
    AssertThrow(data.lines[0].quantities.size()>0, ExcMessage("Empty data"));
    const QuantityStatistics* quantity = dynamic_cast<const QuantityStatistics* > (data.lines[0].quantities[0]);
    averaging_direction = quantity->averaging_direction;

    AssertThrow(averaging_direction == 0 || averaging_direction == 1 || averaging_direction == 2,
                ExcMessage("Take the average either in x, y or z - direction"));

    //Iterator for lines
    line_iterator = 0;
    for(typename std::vector<Line<dim> >::iterator line = data.lines.begin();
        line != data.lines.end(); ++line, ++line_iterator)
    {
      //Resize global variables for # of points on line
      velocity_global[line_iterator].resize(line->n_points);
      pressure_global[line_iterator].resize(line->n_points);
      wall_shear_global[line_iterator].resize(line->n_points);
      reynolds_global[line_iterator].resize(line->n_points);

      // make sure that all lines/quantities really use the same averaging direction
      for (typename std::vector<Quantity*>::iterator quantity = line->quantities.begin();
           quantity != line->quantities.end(); ++quantity)
      {
        QuantityStatistics* stats_ptr = dynamic_cast<QuantityStatistics* > (*quantity);

        unsigned int const direction = stats_ptr->averaging_direction;

        AssertThrow(direction == averaging_direction,
            ExcMessage("Averaging directions for different lines/quantities do not match."));
      }

      // initialize global_points: use equidistant points along line
      for(unsigned int i = 0; i < line->n_points; ++i)
      {
        Point<dim> point = line->begin + double(i)/double(line->n_points-1)*(line->end - line->begin);
        global_points[line_iterator].push_back(point);
      }
    }

    // Save all cells and corresponding points on unit cell
    // that are relevant for a given point along the line.
    // For velocity quantities:
    for(typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_velocity.begin_active();
        cell != dof_handler_velocity.end(); ++cell)
    {
      if(cell->is_locally_owned())
      {
        line_iterator = 0;
        for(typename std::vector<Line<dim> >::iterator line = data.lines.begin();
            line != data.lines.end(); ++line, ++line_iterator)
        {
          bool velocity_has_to_be_evaluated = false;
          for (typename std::vector<Quantity*>::iterator quantity = line->quantities.begin();
              quantity != line->quantities.end(); ++quantity)
          {
            if((*quantity)->type == QuantityType::Velocity ||
               (*quantity)->type == QuantityType::SkinFriction ||
               (*quantity)->type == QuantityType::ReynoldsStresses)
            {
              velocity_has_to_be_evaluated = true;
            }
          }

          if(velocity_has_to_be_evaluated == true)
          {
            // cells and reference points for all points along a line
            for(unsigned int p = 0; p < line->n_points; ++p)
            {
              // First, we move the line to the position of the current cell (vertex 0) in
              // averaging direction and check whether this new point is inside the current cell
              Point<dim> translated_point = global_points[line_iterator][p];
              translated_point[averaging_direction] = cell->vertex(0)[averaging_direction];

              // If the new point lies in the current cell, we have to take the current cell into account
              const Point<dim> p_unit = cell->real_to_unit_cell_affine_approximation(translated_point);
              if(GeometryInfo<dim>::is_inside_unit_cell(p_unit,tolerance))
              {
                cells_and_ref_points_velocity[line_iterator][p].push_back(
                    std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >(cell,p_unit));
              }

//              Point<dim> p_unit = Point<dim>();
//              try
//              {
//                p_unit = mapping.transform_real_to_unit_cell(cell, translated_point);
//              }
//              catch(...)
//              {
//                // A point that does not lie on the reference cell.
//                p_unit[0] = 2.0;
//              }
//              if(GeometryInfo<dim>::is_inside_unit_cell(p_unit,1.e-12))
//              {
//                cells_and_ref_points_velocity[line_iterator][p].push_back(
//                    std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >(cell,p_unit));
//              }
            }
          }
        }
      }
    }

    // Save all cells and corresponding points on unit cell
    // that are relevant for a given point along the line.
    // We have to do the same for the pressure because the
    // DoFHandlers for velocity and pressureare different.
    for(typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_pressure.begin_active();
        cell != dof_handler_pressure.end(); ++cell)
    {
      if(cell->is_locally_owned())
      {
        line_iterator = 0;
        for(typename std::vector<Line<dim> >::iterator line = data.lines.begin();
            line != data.lines.end(); ++line, ++line_iterator)
        {
          // cells and reference points for reference pressure (only one point for each line)
          for (typename std::vector<Quantity*>::iterator quantity = line->quantities.begin();
              quantity != line->quantities.end(); ++quantity)
          {
            //evaluate quantities that involve velocity
            if((*quantity)->type == QuantityType::Pressure)
            {
              // cells and reference points for all points along a line
              for(unsigned int p = 0; p < line->n_points; ++p)
              {
                // First, we move the line to the position of the current cell (vertex 0) in
                // averaging direction and check whether this new point is inside the current cell
                Point<dim> translated_point = global_points[line_iterator][p];
                translated_point[averaging_direction] = cell->vertex(0)[averaging_direction];

                // If the new point lies in the current cell, we have to take the current cell into account
                const Point<dim> p_unit = cell->real_to_unit_cell_affine_approximation(translated_point);
                if(GeometryInfo<dim>::is_inside_unit_cell(p_unit,tolerance))
                {
                  cells_and_ref_points_pressure[line_iterator][p].push_back(
                      std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >(cell,p_unit));
                }

//                Point<dim> p_unit = Point<dim>();
//                try
//                {
//                  p_unit = mapping.transform_real_to_unit_cell(cell, translated_point);
//                }
//                catch(...)
//                {
//                  // A point that does not lie on the reference cell.
//                  p_unit[0] = 2.0;
//                }
//                if(GeometryInfo<dim>::is_inside_unit_cell(p_unit,1.e-12))
//                {
//                  cells_and_ref_points_pressure[line_iterator][p].push_back(
//                    std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >(cell,p_unit));
//                }
              }
            }
          }

          // cells and reference points for reference pressure (only one point for each line)
          for (typename std::vector<Quantity*>::iterator quantity = line->quantities.begin();
              quantity != line->quantities.end(); ++quantity)
          {
            //evaluate quantities that involve velocity
            if((*quantity)->type == QuantityType::PressureCoefficient)
            {
              QuantityStatisticsPressureCoefficient<dim>* quantity_ref_pressure =
                  dynamic_cast<QuantityStatisticsPressureCoefficient<dim>* > (*quantity);

              // First, we move the line to the position of the current cell (vertex 0) in
              // averaging direction and check whether this new point is inside the current cell
              Point<dim> translated_point = quantity_ref_pressure->reference_point;
              translated_point[averaging_direction] = cell->vertex(0)[averaging_direction];

              // If the new point lies in the current cell, we have to take the current cell into account
              const Point<dim> p_unit = cell->real_to_unit_cell_affine_approximation(translated_point);
              if(GeometryInfo<dim>::is_inside_unit_cell(p_unit,tolerance))
              {
                ref_pressure_cells_and_ref_points[line_iterator].push_back(
                    std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >(cell,p_unit));
              }

//              Point<dim> p_unit = Point<dim>();
//              try
//              {
//                p_unit = mapping.transform_real_to_unit_cell(cell, translated_point);
//              }
//              catch(...)
//              {
//                // A point that does not lie on the reference cell.
//                p_unit[0] = 2.0;
//              }
//              if(GeometryInfo<dim>::is_inside_unit_cell(p_unit,1.e-12))
//              {
//                ref_pressure_cells_and_ref_points[line_iterator].push_back(
//                    std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >(cell,p_unit));
//              }
            }
          }
        }
      }
    }
  }

  void evaluate(const parallel::distributed::Vector<double> &velocity,
                const parallel::distributed::Vector<double> &pressure)
  {
    do_evaluate(velocity, pressure);
  }

  void print_headline(std::ofstream      &f,
                      const unsigned int number_of_samples)
  {
    f << "number of samples: N = "  << number_of_samples << std::endl;
  }

  void write_output(const std::string &output_prefix)
  {
    do_write_output(output_prefix);
  }

private:
  void do_evaluate(const parallel::distributed::Vector<double> &velocity,
                   const parallel::distributed::Vector<double> &pressure)
  {
    // increment number of samples
    number_of_samples++;

    //Iterator for lines
    unsigned int line_iterator = 0;
    for(typename std::vector<Line<dim> >::iterator line = data.lines.begin();
        line != data.lines.end(); ++line, ++line_iterator)
    {
      bool evaluate_velocity = false;
      for (typename std::vector<Quantity*>::iterator quantity = line->quantities.begin();
          quantity != line->quantities.end(); ++quantity)
      {
        //evaluate quantities that involve velocity
        if((*quantity)->type == QuantityType::Velocity ||
           (*quantity)->type == QuantityType::SkinFriction ||
           (*quantity)->type == QuantityType::ReynoldsStresses)
        {
          evaluate_velocity = true;
        }
      }
      if(evaluate_velocity == true)
        do_evaluate_velocity(velocity, *line, line_iterator);

      bool evaluate_pressure = false;
      for (typename std::vector<Quantity*>::iterator quantity = line->quantities.begin();
          quantity != line->quantities.end(); ++quantity)
      {
        //evaluate quantities that involve velocity
        if((*quantity)->type == QuantityType::Pressure ||
           (*quantity)->type == QuantityType::PressureCoefficient)
        {
          evaluate_pressure = true;
        }
      }
      if(evaluate_pressure == true)
        do_evaluate_pressure(pressure, *line, line_iterator);
    }
  }

  void do_evaluate_velocity(parallel::distributed::Vector<double> const &velocity,
                            Line<dim> const                             &line,
                            unsigned int const                          line_iterator)
  {
    //Local variables for specific line
    std::vector<double> length_local(line.n_points);
    std::vector<Tensor<1, dim, double> > velocity_local(line.n_points);
    std::vector<double> wall_shear_local(line.n_points);
    std::vector<Tensor<2, dim, double> > reynolds_local(line.n_points);

    const unsigned int scalar_dofs_per_cell = dof_handler_velocity.get_fe().base_element(0).dofs_per_cell;
    std::vector<Tensor<1,dim> > velocity_vector(scalar_dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices(dof_handler_velocity.get_fe().dofs_per_cell);

    for(unsigned int p = 0; p < line.n_points; ++p)
    {
      typedef typename std::vector<std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> > > TYPE;
      for(typename TYPE::const_iterator cell_and_ref_point = cells_and_ref_points_velocity[line_iterator][p].begin();
          cell_and_ref_point != cells_and_ref_points_velocity[line_iterator][p].end(); ++cell_and_ref_point)
      {
        const unsigned int fe_degree_velocity = dof_handler_velocity.get_fe().degree;

        // use quadrature for averaging in homogeneous direction
        QGauss<1> gauss_1d(fe_degree_velocity+1);
        std::vector<Point<dim> > points(gauss_1d.size()); // 1D points
        std::vector<double> weights(gauss_1d.size()); // 1D weights

        typename DoFHandler<dim>::active_cell_iterator const cell = cell_and_ref_point->first;
        Point<dim> const p_unit = cell_and_ref_point->second;

        //Find points and weights for Gauss quadrature
        find_points_and_weights(p_unit, points, weights, averaging_direction, gauss_1d);

        FEValues<dim,dim> fe_values(mapping,
                                    dof_handler_velocity.get_fe().base_element(0),
                                    Quadrature<dim>(points, weights),
                                    update_values | update_jacobians |
                                    update_quadrature_points |update_gradients);

        fe_values.reinit(typename Triangulation<dim>::active_cell_iterator(cell));

        cell->get_dof_indices(dof_indices);

        // resort velocity dofs
        for (unsigned int j=0; j<dof_indices.size(); ++j)
        {
          const std::pair<unsigned int,unsigned int> comp =
            dof_handler_velocity.get_fe().system_to_component_index(j);
          if (comp.first < dim)
            velocity_vector[comp.second][comp.first] = velocity(dof_indices[j]);
        }

        // perform averaging in homogeneous direction
        for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q)
        {
          double det = std::abs(fe_values.jacobian(q)[averaging_direction][averaging_direction]);
          double JxW = det * fe_values.get_quadrature().weight(q);

          // calculate integrals in homogeneous direction
          length_local[p] += JxW;

          Tensor<1,dim> velocity;

          for (typename std::vector<Quantity*>::const_iterator quantity = line.quantities.begin();
              quantity != line.quantities.end(); ++quantity)
          {
            if((*quantity)->type == QuantityType::Velocity ||
               (*quantity)->type == QuantityType::ReynoldsStresses)
            {
              // evaluate velocity solution in current quadrature points
              for (unsigned int j=0; j<velocity_vector.size(); ++j)
                velocity += fe_values.shape_value(j,q) * velocity_vector[j];
            }

            if((*quantity)->type == QuantityType::Velocity)
            {
              for(unsigned int i=0; i<dim; ++i)
                velocity_local[p][i] += velocity[i] * JxW;
            }

            if((*quantity)->type == QuantityType::ReynoldsStresses)
            {
              for(unsigned int i=0; i<dim; ++i)
                for(unsigned int j=0; j<dim; ++j)
                  reynolds_local[p][i][j] += velocity[i] * velocity[j] * JxW;
            }

            if((*quantity)->type == QuantityType::SkinFriction)
            {
              Tensor<2,dim> velocity_gradient;
              for (unsigned int j=0; j<velocity_vector.size(); ++j)
                velocity_gradient += outer_product(velocity_vector[j], fe_values.shape_grad(j,q));

              const QuantityStatisticsSkinFriction<dim>* quantity_skin_friction =
                  dynamic_cast<const QuantityStatisticsSkinFriction<dim>* > (*quantity);

              Tensor<1, dim, double> normal = quantity_skin_friction->normal_vector;
              Tensor<1, dim, double> tangent = quantity_skin_friction->tangent_vector;

              for (unsigned int i=0; i<dim; ++i)
                for (unsigned int j=0; j<dim; ++j)
                  wall_shear_local[p] += tangent[i] * velocity_gradient[i][j] * normal[j] * JxW;
            }
          }
        }
      }
    }

    Utilities::MPI::sum(length_local, communicator, length_local);

    for (typename std::vector<Quantity*>::const_iterator quantity = line.quantities.begin();
        quantity != line.quantities.end(); ++quantity)
    {
      // Cells are distributed over processors, therefore we need
      // to sum the contributions of every single processor.
      if((*quantity)->type == QuantityType::Velocity)
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

      if((*quantity)->type == QuantityType::ReynoldsStresses)
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

      if((*quantity)->type == QuantityType::SkinFriction)
      {
        Utilities::MPI::sum(wall_shear_local, communicator, wall_shear_local);

        for(unsigned int p = 0; p < line.n_points; ++p)
        {
          wall_shear_global[line_iterator][p] += wall_shear_local[p]/length_local[p];
        }
      }
    }
  }

  void do_evaluate_pressure(parallel::distributed::Vector<double> const &pressure,
                            Line<dim> const                             &line,
                            unsigned int const                          line_iterator)
  {
    for (typename std::vector<Quantity*>::const_iterator quantity = line.quantities.begin();
        quantity != line.quantities.end(); ++quantity)
    {
      if((*quantity)->type == QuantityType::Pressure)
      {
        //Local variables for specific line
        std::vector<double> length_local(line.n_points);
        std::vector<double> pressure_local(line.n_points);

        for(unsigned int p = 0; p < line.n_points; ++p)
        {
          TYPE vector_cells_and_ref_points = cells_and_ref_points_pressure[line_iterator][p];

          average_pressure_for_given_point(pressure,
                                           vector_cells_and_ref_points,
                                           length_local[p],
                                           pressure_local[p]);
        }

        // MPI communication
        Utilities::MPI::sum(length_local, communicator, length_local);
        Utilities::MPI::sum(pressure_local, communicator, pressure_local);

        // averaging in space (over homogeneous direction)
        for(unsigned int p=0; p<line.n_points; ++p)
          pressure_global[line_iterator][p] += pressure_local[p]/length_local[p];
      }

      if((*quantity)->type == QuantityType::PressureCoefficient)
      {
        double length_local = 0.0;
        double pressure_local = 0.0;

        TYPE vector_cells_and_ref_points = ref_pressure_cells_and_ref_points[line_iterator];

        average_pressure_for_given_point(pressure,
                                         vector_cells_and_ref_points,
                                         length_local,
                                         pressure_local);

        // MPI communication
        length_local = Utilities::MPI::sum(length_local, communicator);
        pressure_local = Utilities::MPI::sum(pressure_local, communicator);

        // averaging in space (over homogeneous direction)
        reference_pressure_global[line_iterator] += pressure_local/length_local;
      }
    }
  }

  void average_pressure_for_given_point(parallel::distributed::Vector<double> const &pressure,
                                        TYPE const                                  &vector_cells_and_ref_points,
                                        double                                      &length_local,
                                        double                                      &pressure_local)
  {
    const unsigned int scalar_dofs_per_cell = dof_handler_pressure.get_fe().base_element(0).dofs_per_cell;
    std::vector<double> pressure_vector(scalar_dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices(dof_handler_pressure.get_fe().dofs_per_cell);

    for(typename TYPE::const_iterator cell_and_ref_point = vector_cells_and_ref_points.begin();
        cell_and_ref_point != vector_cells_and_ref_points.end(); ++cell_and_ref_point)
    {
      const unsigned int fe_degree_pressure = dof_handler_pressure.get_fe().degree;

      // use quadrature for averaging in homogeneous direction
      QGauss<1> gauss_1d(fe_degree_pressure+1);
      std::vector<Point<dim> > points(gauss_1d.size()); // 1D points
      std::vector<double> weights(gauss_1d.size()); // 1D weights

      typename DoFHandler<dim>::active_cell_iterator const cell = cell_and_ref_point->first;
      Point<dim> const p_unit = cell_and_ref_point->second;

      //Find points and weights for Gauss quadrature
      find_points_and_weights(p_unit, points, weights, averaging_direction, gauss_1d);

      FEValues<dim,dim> fe_values(mapping,
                                  dof_handler_pressure.get_fe().base_element(0),
                                  Quadrature<dim>(points, weights),
                                  update_values | update_jacobians |
                                  update_quadrature_points);

      fe_values.reinit(typename Triangulation<dim>::active_cell_iterator(cell));

      cell->get_dof_indices(dof_indices);

      for (unsigned int j=0; j<scalar_dofs_per_cell; ++j)
        pressure_vector[j] = pressure(dof_indices[j]);

      for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q)
      {
        double p = 0;

        for (unsigned int j=0; j<pressure_vector.size(); ++j)
          p += fe_values.shape_value(j,q) * pressure_vector[j];

        double det = std::abs(fe_values.jacobian(q)[averaging_direction][averaging_direction]);
        double JxW = det * fe_values.get_quadrature().weight(q);

        length_local += JxW;
        pressure_local += p * JxW;
      }
    }
  }

  void find_points_and_weights (Point<dim> const         &point_in_ref_coord,
                                std::vector<Point<dim> > &points,
                                std::vector<double>      &weights,
                                unsigned int const       averaging_direction,
                                QGauss<1> const          &gauss_1d)
  {
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

  void do_write_output(const std::string &output_prefix)
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
                f << std::left << 2.0*(pressure_global[line_iterator][p] - reference_pressure_global[line_iterator])/ref_velocity_square/number_of_samples;
              }
              f << std::endl;
            }
            f.close();
          }
        }
      }
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

  // For all lines: for all ppints along the line: list of all relevant cells and points in ref coordinates
  std::vector<std::vector<std::vector<std::pair<typename DoFHandler<dim>::active_cell_iterator,Point<dim> > > > > cells_and_ref_points_velocity;

  // For all lines: for all ppints along the line: list of all relevant cells and points in ref coordinates
  std::vector<std::vector<std::vector<std::pair<typename DoFHandler<dim>::active_cell_iterator,Point<dim> > > > > cells_and_ref_points_pressure;

  // For all lines: for pressure reference point: list of all relevant cells and points in ref coordinates
  std::vector<std::vector<std::pair<typename DoFHandler<dim>::active_cell_iterator,Point<dim> > > > ref_pressure_cells_and_ref_points;

  // number of samples for averaging in time
  unsigned int number_of_samples;

  // homogeneous direction for averaging in space
  unsigned int averaging_direction;

  //Velocity quantities
  std::vector<std::vector<Tensor<1, dim, double> > > velocity_global;

  //Skin Friction quantities
  std::vector<std::vector<double> > wall_shear_global;

  //Reynolds Stress quantities
  std::vector<std::vector<Tensor<2, dim, double> > > reynolds_global;

  //Pressure quantities
  std::vector<std::vector<double> > pressure_global;
  std::vector<double> reference_pressure_global;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_STATISTICS_H_ */
