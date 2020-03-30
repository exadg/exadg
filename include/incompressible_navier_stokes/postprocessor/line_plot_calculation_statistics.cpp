/*
 * line_plot_calculation_statistics.cpp
 *
 *  Created on: May 18, 2019
 *      Author: fehn
 */

#include "line_plot_calculation_statistics.h"

#include "../../functionalities/evaluate_solution_in_given_point.h"

template<int dim, typename Number>
LinePlotCalculatorStatistics<dim, Number>::LinePlotCalculatorStatistics(
  DoFHandler<dim> const & dof_handler_velocity_in,
  DoFHandler<dim> const & dof_handler_pressure_in,
  Mapping<dim> const &    mapping_in,
  MPI_Comm const &        mpi_comm)
  : clear_files(true),
    dof_handler_velocity(dof_handler_velocity_in),
    dof_handler_pressure(dof_handler_pressure_in),
    mapping(mapping_in),
    communicator(mpi_comm),
    cell_data_has_been_initialized(false),
    number_of_samples(0),
    write_final_output(false)
{
}

template<int dim, typename Number>
void
LinePlotCalculatorStatistics<dim, Number>::setup(
  LinePlotDataStatistics<dim> const & line_plot_data_in)
{
  // initialize data
  data = line_plot_data_in;

  if(data.statistics_data.calculate_statistics == true)
  {
    AssertThrow(data.line_data.lines.size() > 0, ExcMessage("Empty data"));

    // allocate data structures
    velocity_global.resize(data.line_data.lines.size());
    pressure_global.resize(data.line_data.lines.size());
    global_points.resize(data.line_data.lines.size());
    cells_global_velocity.resize(data.line_data.lines.size());
    cells_global_pressure.resize(data.line_data.lines.size());

    unsigned int line_iterator = 0;
    for(typename std::vector<std::shared_ptr<Line<dim>>>::iterator line =
          data.line_data.lines.begin();
        line != data.line_data.lines.end();
        ++line, ++line_iterator)
    {
      // Resize global variables for number of points on line
      velocity_global[line_iterator].resize((*line)->n_points);
      pressure_global[line_iterator].resize((*line)->n_points);

      // initialize global_points: use/assume equidistant points along line
      for(unsigned int i = 0; i < (*line)->n_points; ++i)
      {
        Point<dim> point = (*line)->begin + double(i) / double((*line)->n_points - 1) *
                                              ((*line)->end - (*line)->begin);
        global_points[line_iterator].push_back(point);
      }

      cells_global_velocity[line_iterator].resize((*line)->n_points);
      cells_global_pressure[line_iterator].resize((*line)->n_points);
    }
  }
}

template<int dim, typename Number>
void
LinePlotCalculatorStatistics<dim, Number>::evaluate(VectorType const &   velocity,
                                                    VectorType const &   pressure,
                                                    double const &       time,
                                                    unsigned int const & time_step_number)
{
  if(data.statistics_data.calculate_statistics == true)
  {
    // EPSILON: small number which is much smaller than the time step size
    const double EPSILON = 1.0e-10;

    if((time > data.statistics_data.sample_start_time - EPSILON) &&
       (time < data.statistics_data.sample_end_time + EPSILON) &&
       (time_step_number % data.statistics_data.sample_every_timesteps == 0))
    {
      // evaluate statistics
      do_evaluate(velocity, pressure);

      // write intermediate output
      if(time_step_number % data.statistics_data.write_output_every_timesteps == 0)
      {
        do_write_output();
      }
    }

    // write final output
    if((time > data.statistics_data.sample_end_time - EPSILON) && write_final_output)
    {
      do_write_output();
      write_final_output = false;
    }
  }
}

template<int dim, typename Number>
void
LinePlotCalculatorStatistics<dim, Number>::initialize_cell_data(VectorType const & velocity,
                                                                VectorType const & pressure)
{
  // Save data related to all adjacent cells for a given point along the line.
  unsigned int line_iterator = 0;
  for(typename std::vector<std::shared_ptr<Line<dim>>>::iterator line =
        data.line_data.lines.begin();
      line != data.line_data.lines.end();
      ++line, ++line_iterator)
  {
    // make sure that line type is correct
    std::shared_ptr<LineCircumferentialAveraging<dim>> line_circ =
      std::dynamic_pointer_cast<LineCircumferentialAveraging<dim>>(*line);

    AssertThrow(line_circ.get() != 0,
                ExcMessage("Invalid line type, expected LineCircumferentialAveraging<dim>"));

    // find out which quantities have to be evaluated
    bool velocity_has_to_be_evaluated = false;
    bool pressure_has_to_be_evaluated = false;

    for(typename std::vector<std::shared_ptr<Quantity>>::iterator quantity =
          (*line)->quantities.begin();
        quantity != (*line)->quantities.end();
        ++quantity)
    {
      if((*quantity)->type == QuantityType::Velocity ||
         (*quantity)->type == QuantityType::SkinFriction ||
         (*quantity)->type == QuantityType::ReynoldsStresses)
      {
        velocity_has_to_be_evaluated = true;
      }

      if((*quantity)->type == QuantityType::Pressure)
      {
        pressure_has_to_be_evaluated = true;
      }
    }

    // determine two unit vectors defining circumferential plane
    Tensor<1, dim, double> normal_vector;
    Tensor<1, dim, double> unit_vector_1, unit_vector_2;
    if(line_circ->average_circumferential == true)
    {
      normal_vector = line_circ->normal_vector;

      // We assume that line->begin is the center of the circle for circumferential averaging.

      // Calculate two unit vectors in the plane that is normal to the normal_vector.
      unit_vector_1       = (*line)->end - (*line)->begin;
      double const norm_1 = unit_vector_1.norm();
      AssertThrow(norm_1 > 1.e-12, ExcMessage("Invalid begin and end points found."));

      unit_vector_1 /= norm_1;

      AssertThrow(dim == 3, ExcMessage("Not implemented."));

      unit_vector_2       = cross_product_3d(normal_vector, unit_vector_1);
      double const norm_2 = unit_vector_2.norm();

      AssertThrow(norm_2 > 1.e-12, ExcMessage("Invalid begin and end points found."));

      unit_vector_2 /= norm_2;
    }

    // for all points along a line
    for(unsigned int p = 0; p < (*line)->n_points; ++p)
    {
      Point<dim> point = global_points[line_iterator][p];

      // In case no averaging in circumferential direction is performed, just insert point
      // "point".
      std::vector<Point<dim>> points;
      points.push_back(point);

      // If averaging in circumferential direction is used, we insert additional points along
      // the circle for points p>=1. The first point p=0 lies in the center of the circle
      // (point(p=0) == line.begin).
      if(p >= 1 && line_circ->average_circumferential == true)
      {
        // begin with 1 since the first point has already been inserted.
        for(unsigned int i = 1; i < line_circ->n_points_circumferential; ++i)
        {
          double cos =
            std::cos((double(i) / line_circ->n_points_circumferential) * 2.0 * numbers::PI);
          double sin =
            std::sin((double(i) / line_circ->n_points_circumferential) * 2.0 * numbers::PI);
          double radius = (point - (*line)->begin).norm();

          Point<dim> new_point;
          for(unsigned int d = 0; d < dim; ++d)
          {
            new_point[d] = ((*line)->begin)[d] + cos * radius * unit_vector_1[d] +
                           sin * radius * unit_vector_2[d];
          }

          points.push_back(new_point);
        }
      }

      for(typename std::vector<Point<dim>>::iterator point_it = points.begin();
          point_it != points.end();
          ++point_it)
      {
        if(velocity_has_to_be_evaluated == true)
        {
          // find adjacent cells and store data required later for evaluating the solution.
          std::vector<std::pair<unsigned int, std::vector<Number>>>
            dof_index_first_dof_and_shape_values;

          get_dof_index_and_shape_values(dof_handler_velocity,
                                         mapping,
                                         velocity,
                                         *point_it,
                                         dof_index_first_dof_and_shape_values);

          for(typename std::vector<std::pair<unsigned int, std::vector<Number>>>::iterator iter =
                dof_index_first_dof_and_shape_values.begin();
              iter != dof_index_first_dof_and_shape_values.end();
              ++iter)
          {
            cells_global_velocity[line_iterator][p].push_back(*iter);
          }
        }

        if(pressure_has_to_be_evaluated == true)
        {
          // find adjacent cells and store data required later for evaluating the solution.
          std::vector<std::pair<unsigned int, std::vector<Number>>>
            dof_index_first_dof_and_shape_values;

          get_dof_index_and_shape_values(dof_handler_pressure,
                                         mapping,
                                         pressure,
                                         *point_it,
                                         dof_index_first_dof_and_shape_values);

          for(typename std::vector<std::pair<unsigned int, std::vector<Number>>>::iterator iter =
                dof_index_first_dof_and_shape_values.begin();
              iter != dof_index_first_dof_and_shape_values.end();
              ++iter)
          {
            cells_global_pressure[line_iterator][p].push_back(*iter);
          }
        }
      }
    }
  }
}

template<int dim, typename Number>
void
LinePlotCalculatorStatistics<dim, Number>::do_evaluate(VectorType const & velocity,
                                                       VectorType const & pressure)
{
  // increment number of samples
  number_of_samples++;

  // Make sure that all data has been initialized before evaluating the solution.
  if(cell_data_has_been_initialized == false)
  {
    initialize_cell_data(velocity, pressure);

    cell_data_has_been_initialized = true;
  }

  // Iterator for lines
  unsigned int line_iterator = 0;
  for(typename std::vector<std::shared_ptr<Line<dim>>>::iterator line =
        data.line_data.lines.begin();
      line != data.line_data.lines.end();
      ++line, ++line_iterator)
  {
    bool evaluate_velocity = false;
    for(typename std::vector<std::shared_ptr<Quantity>>::iterator quantity =
          (*line)->quantities.begin();
        quantity != (*line)->quantities.end();
        ++quantity)
    {
      // evaluate quantities that involve velocity
      if((*quantity)->type == QuantityType::Velocity ||
         (*quantity)->type == QuantityType::SkinFriction ||
         (*quantity)->type == QuantityType::ReynoldsStresses)
      {
        evaluate_velocity = true;
      }
    }
    if(evaluate_velocity == true)
    {
      do_evaluate_velocity(velocity, *(*line), line_iterator);
    }

    bool evaluate_pressure = false;
    for(typename std::vector<std::shared_ptr<Quantity>>::iterator quantity =
          (*line)->quantities.begin();
        quantity != (*line)->quantities.end();
        ++quantity)
    {
      // evaluate quantities that involve velocity
      if((*quantity)->type == QuantityType::Pressure ||
         (*quantity)->type == QuantityType::PressureCoefficient)
      {
        evaluate_pressure = true;
      }
    }
    if(evaluate_pressure == true)
    {
      do_evaluate_pressure(pressure, *(*line), line_iterator);
    }
  }
}

template<int dim, typename Number>
void
LinePlotCalculatorStatistics<dim, Number>::do_evaluate_velocity(VectorType const & velocity,
                                                                Line<dim> const &  line,
                                                                unsigned int const line_iterator)
{
  // Local variables for the current line:

  // for all points along the line: velocity vector
  std::vector<Tensor<1, dim, Number>> velocity_vector_local(line.n_points);
  // for all points along the line: counter
  std::vector<unsigned int> counter_vector_local(line.n_points);

  for(unsigned int p = 0; p < line.n_points; ++p)
  {
    std::vector<std::pair<unsigned int, std::vector<Number>>> & adjacent_cells(
      cells_global_velocity[line_iterator][p]);

    // loop over all adjacent, locally owned cells for the current point
    for(auto iter = adjacent_cells.begin(); iter != adjacent_cells.end(); ++iter)
    {
      // increment counter (because this is a locally owned cell)
      counter_vector_local[p] += 1;

      for(typename std::vector<std::shared_ptr<Quantity>>::const_iterator quantity =
            line.quantities.begin();
          quantity != line.quantities.end();
          ++quantity)
      {
        if((*quantity)->type == QuantityType::Velocity)
        {
          // interpolate solution using the precomputed shape values and the global dof index
          Tensor<1, dim, Number> velocity_value = Interpolator<1, dim, Number>::value(
            dof_handler_velocity, velocity, iter->first, iter->second);

          // add result to array with velocity values
          velocity_vector_local[p] += velocity_value;
        }
        else
        {
          AssertThrow(false, ExcMessage("Not implemented."));
        }
      }
    }
  }

  // Cells are distributed over processors, therefore we need
  // to sum the contributions of every single processor.
  Utilities::MPI::sum(counter_vector_local, communicator, counter_vector_local);

  // Perform MPI communcation as well as averaging for all quantities of the current line.
  for(typename std::vector<std::shared_ptr<Quantity>>::const_iterator quantity =
        line.quantities.begin();
      quantity != line.quantities.end();
      ++quantity)
  {
    if((*quantity)->type == QuantityType::Velocity)
    {
      Utilities::MPI::sum(
        ArrayView<const Number>(&velocity_vector_local[0][0], dim * velocity_vector_local.size()),
        communicator,
        ArrayView<Number>(&velocity_vector_local[0][0], dim * velocity_vector_local.size()));

      // Accumulate instantaneous values into global vector.
      // When writing the output files, we calculate the time-averaged values
      // by dividing the global (accumulated) values by the number of samples.
      for(unsigned int p = 0; p < line.n_points; ++p)
      {
        // Take average value over all adjacent cells for a given point.
        for(unsigned int d = 0; d < dim; ++d)
        {
          if(counter_vector_local[p] > 0)
          {
            velocity_global[line_iterator][p][d] +=
              velocity_vector_local[p][d] / counter_vector_local[p];
          }
        }
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }
}

template<int dim, typename Number>
void
LinePlotCalculatorStatistics<dim, Number>::do_evaluate_pressure(VectorType const & pressure,
                                                                Line<dim> const &  line,
                                                                unsigned int const line_iterator)
{
  // Local variables for the current line:

  // for all points along the line: pressure value
  std::vector<Number> pressure_vector_local(line.n_points);
  // for all points along the line: counter
  std::vector<unsigned int> counter_vector_local(line.n_points);

  for(unsigned int p = 0; p < line.n_points; ++p)
  {
    std::vector<std::pair<unsigned int, std::vector<Number>>> & adjacent_cells(
      cells_global_pressure[line_iterator][p]);

    // loop over all adjacent, locally owned cells for the current point
    for(auto iter = adjacent_cells.begin(); iter != adjacent_cells.end(); ++iter)
    {
      // increment counter (because this is a locally owned cell)
      counter_vector_local[p] += 1;

      for(typename std::vector<std::shared_ptr<Quantity>>::const_iterator quantity =
            line.quantities.begin();
          quantity != line.quantities.end();
          ++quantity)
      {
        if((*quantity)->type == QuantityType::Pressure)
        {
          // interpolate solution using the precomputed shape values and the global dof index
          Number pressure_value = Interpolator<0, dim, Number>::value(dof_handler_pressure,
                                                                      pressure,
                                                                      iter->first,
                                                                      iter->second);

          // add result to array with pressure values
          pressure_vector_local[p] += pressure_value;
        }
        else
        {
          AssertThrow(false, ExcMessage("Not implemented."));
        }
      }
    }
  }

  // Cells are distributed over processors, therefore we need
  // to sum the contributions of every single processor.
  Utilities::MPI::sum(counter_vector_local, communicator, counter_vector_local);

  // Perform MPI communcation as well as averaging for all quantities of the current line.
  for(typename std::vector<std::shared_ptr<Quantity>>::const_iterator quantity =
        line.quantities.begin();
      quantity != line.quantities.end();
      ++quantity)
  {
    if((*quantity)->type == QuantityType::Pressure)
    {
      Utilities::MPI::sum(pressure_vector_local, communicator, pressure_vector_local);

      // Accumulate instantaneous values into global vector.
      // When writing the output files, we calculate the time-averaged values
      // by dividing the global (accumulated) values by the number of samples.
      for(unsigned int p = 0; p < line.n_points; ++p)
      {
        // Take average value over all adjacent cells for a given point.
        if(counter_vector_local[p] > 0)
        {
          pressure_global[line_iterator][p] += pressure_vector_local[p] / counter_vector_local[p];
        }
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }
}

template<int dim, typename Number>
void
LinePlotCalculatorStatistics<dim, Number>::do_write_output() const
{
  if(Utilities::MPI::this_mpi_process(communicator) == 0 &&
     data.statistics_data.calculate_statistics == true)
  {
    unsigned int const precision = data.line_data.precision;

    // Iterator for lines
    unsigned int line_iterator = 0;
    for(typename std::vector<std::shared_ptr<Line<dim>>>::const_iterator line =
          data.line_data.lines.begin();
        line != data.line_data.lines.end();
        ++line, ++line_iterator)
    {
      std::string filename_prefix = data.line_data.directory + (*line)->name;

      for(typename std::vector<std::shared_ptr<Quantity>>::const_iterator quantity =
            (*line)->quantities.begin();
          quantity != (*line)->quantities.end();
          ++quantity)
      {
        // Velocity quantities ...
        if((*quantity)->type == QuantityType::Velocity)
        {
          std::string   filename = filename_prefix + "_velocity" + ".txt";
          std::ofstream f;
          if(clear_files)
          {
            f.open(filename.c_str(), std::ios::trunc);
          }
          else
          {
            f.open(filename.c_str(), std::ios::app);
          }

          print_headline(f, number_of_samples);

          for(unsigned int d = 0; d < dim; ++d)
            f << std::setw(precision + 8) << std::left << "x_" + Utilities::int_to_string(d + 1);
          for(unsigned int d = 0; d < dim; ++d)
            f << std::setw(precision + 8) << std::left << "u_" + Utilities::int_to_string(d + 1);

          f << std::endl;

          // loop over all points
          for(unsigned int p = 0; p < (*line)->n_points; ++p)
          {
            f << std::scientific << std::setprecision(precision);

            // write data
            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left << global_points[line_iterator][p][d];

            // write velocity and average over time
            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left
                << velocity_global[line_iterator][p][d] / number_of_samples;

            f << std::endl;
          }
          f.close();
        }

        if((*quantity)->type == QuantityType::ReynoldsStresses)
        {
          AssertThrow(false, ExcMessage("Not implemented."));
        }

        if((*quantity)->type == QuantityType::SkinFriction)
        {
          AssertThrow(false, ExcMessage("Not implemented."));
        }

        // ... and pressure quantities.
        if((*quantity)->type == QuantityType::Pressure)
        {
          std::string   filename = filename_prefix + "_pressure" + ".txt";
          std::ofstream f;

          if(clear_files)
          {
            f.open(filename.c_str(), std::ios::trunc);
          }
          else
          {
            f.open(filename.c_str(), std::ios::app);
          }

          print_headline(f, number_of_samples);

          for(unsigned int d = 0; d < dim; ++d)
            f << std::setw(precision + 8) << std::left << "x_" + Utilities::int_to_string(d + 1);

          f << std::setw(precision + 8) << std::left << "p";

          f << std::endl;

          for(unsigned int p = 0; p < (*line)->n_points; ++p)
          {
            f << std::scientific << std::setprecision(precision);

            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left << global_points[line_iterator][p][d];

            f << std::setw(precision + 8) << std::left
              << pressure_global[line_iterator][p] / number_of_samples;

            f << std::endl;
          }
          f.close();
        }

        if((*quantity)->type == QuantityType::PressureCoefficient)
        {
          AssertThrow(false, ExcMessage("Not implemented."));
        }
      }
    }
  }
}

template class LinePlotCalculatorStatistics<2, float>;
template class LinePlotCalculatorStatistics<3, float>;

template class LinePlotCalculatorStatistics<2, double>;
template class LinePlotCalculatorStatistics<3, double>;
