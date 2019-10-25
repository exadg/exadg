/*
 * line_plot_calculation_statistics.cpp
 *
 *  Created on: May 18, 2019
 *      Author: fehn
 */

#include "line_plot_calculation_statistics.h"

template<int dim>
LinePlotCalculatorStatistics<dim>::LinePlotCalculatorStatistics(
  const DoFHandler<dim> & dof_handler_velocity_in,
  const DoFHandler<dim> & dof_handler_pressure_in,
  const Mapping<dim> &    mapping_in)
  : clear_files(true),
    dof_handler_velocity(dof_handler_velocity_in),
    dof_handler_pressure(dof_handler_pressure_in),
    mapping(mapping_in),
    communicator(dynamic_cast<const parallel::TriangulationBase<dim> *>(
                   &dof_handler_velocity.get_triangulation()) ?
                   (dynamic_cast<const parallel::TriangulationBase<dim> *>(
                      &dof_handler_velocity.get_triangulation())
                      ->get_communicator()) :
                   MPI_COMM_SELF),
    cell_data_has_been_initialized(false),
    number_of_samples(0),
    write_final_output(false)
{
}

template<int dim>
void
LinePlotCalculatorStatistics<dim>::setup(LinePlotData<dim> const & line_plot_data_in)
{
  // initialize data
  data = line_plot_data_in;

  if(data.write_output)
  {
    AssertThrow(data.lines.size() > 0, ExcMessage("Empty data"));

    // allocate data structures
    velocity_global.resize(data.lines.size());
    pressure_global.resize(data.lines.size());
    global_points.resize(data.lines.size());
    cells_global_velocity.resize(data.lines.size());
    cells_global_pressure.resize(data.lines.size());

    unsigned int line_iterator = 0;
    for(typename std::vector<Line<dim>>::iterator line = data.lines.begin();
        line != data.lines.end();
        ++line, ++line_iterator)
    {
      // Resize global variables for number of points on line
      velocity_global[line_iterator].resize(line->n_points);
      pressure_global[line_iterator].resize(line->n_points);

      // initialize global_points: use/assume equidistant points along line
      for(unsigned int i = 0; i < line->n_points; ++i)
      {
        Point<dim> point =
          line->begin + double(i) / double(line->n_points - 1) * (line->end - line->begin);
        global_points[line_iterator].push_back(point);
      }

      cells_global_velocity[line_iterator].resize(line->n_points);
      cells_global_pressure[line_iterator].resize(line->n_points);
    }
  }
}

template<int dim>
void
LinePlotCalculatorStatistics<dim>::evaluate(VectorType const &   velocity,
                                            VectorType const &   pressure,
                                            double const &       time,
                                            unsigned int const & time_step_number)
{
  // EPSILON: small number which is much smaller than the time step size
  const double EPSILON = 1.0e-10;
  if(data.statistics_data.calculate_statistics == true)
  {
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

template<int dim>
void
LinePlotCalculatorStatistics<dim>::do_evaluate(VectorType const & velocity,
                                               VectorType const & pressure)
{
  // increment number of samples
  number_of_samples++;

  // Make sure that all data has been initialized before evaluating the solution.
  if(cell_data_has_been_initialized == false)
  {
    // Save data related to all adjacent cells for a given point along the line.
    unsigned int line_iterator = 0;
    for(typename std::vector<Line<dim>>::iterator line = data.lines.begin();
        line != data.lines.end();
        ++line, ++line_iterator)
    {
      bool velocity_has_to_be_evaluated = false;
      bool pressure_has_to_be_evaluated = false;

      for(typename std::vector<Quantity *>::iterator quantity = line->quantities.begin();
          quantity != line->quantities.end();
          ++quantity)
      {
        // make sure that the averaging type is correct
        const QuantityStatistics<dim> * quantity_statistics =
          dynamic_cast<const QuantityStatistics<dim> *>(*quantity);

        AssertThrow(
          quantity_statistics->average_homogeneous_direction == false,
          ExcMessage(
            "Averaging type for QuantityStatistics is not compatible with this type of line plot "
            "calculation. Either select no averaging in space or averaging in circumferential direction."));

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

      // take the first quantity to get the normal vector. All quantities for the
      // current line must have the same averaging type
      const QuantityStatistics<dim> * quantity =
        dynamic_cast<const QuantityStatistics<dim> *>(line->quantities[0]);

      Tensor<1, dim, double> normal_vector;
      Tensor<1, dim, double> unit_vector_1, unit_vector_2;

      if(quantity->average_circumferential == true)
      {
        normal_vector = quantity->normal_vector;

        // We assume that line->begin is the center of the circle for circumferential averaging.

        // Calculate two unit vectors in the plane that is normal to the normal_vector.
        unit_vector_1       = line->end - line->begin;
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
      for(unsigned int p = 0; p < line->n_points; ++p)
      {
        Point<dim> point = global_points[line_iterator][p];

        // In case no averaging in circumferential direction is performed, just insert point
        // "point".
        std::vector<Point<dim>> points;
        points.push_back(point);

        // If averaging in circumferential direction is used, we insert additional points along
        // the circle for points p>=1. The first point p=0 lies in the center of the circle
        // (point(p=0) == line.begin).
        if(p >= 1 && quantity->average_circumferential == true)
        {
          // begin with 1 since the first point has already been inserted.
          for(unsigned int i = 1; i < quantity->n_points_circumferential; ++i)
          {
            double cos =
              std::cos((double(i) / quantity->n_points_circumferential) * 2.0 * numbers::PI);
            double sin =
              std::sin((double(i) / quantity->n_points_circumferential) * 2.0 * numbers::PI);
            double radius = (point - line->begin).norm();

            Point<dim> new_point;
            for(unsigned int d = 0; d < dim; ++d)
            {
              new_point[d] = (line->begin)[d] + cos * radius * unit_vector_1[d] +
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
            std::vector<std::pair<unsigned int, std::vector<double>>>
              dof_index_first_dof_and_shape_values_velocity;

            get_global_dof_index_and_shape_values(dof_handler_velocity,
                                                  mapping,
                                                  velocity,
                                                  *point_it,
                                                  dof_index_first_dof_and_shape_values_velocity);

            for(typename std::vector<std::pair<unsigned int, std::vector<double>>>::iterator iter =
                  dof_index_first_dof_and_shape_values_velocity.begin();
                iter != dof_index_first_dof_and_shape_values_velocity.end();
                ++iter)
            {
              cells_global_velocity[line_iterator][p].push_back(*iter);
            }
          }

          if(pressure_has_to_be_evaluated == true)
          {
            // find adjacent cells and store data required later for evaluating the solution.
            std::vector<std::pair<unsigned int, std::vector<double>>>
              dof_index_first_dof_and_shape_values_pressure;

            get_global_dof_index_and_shape_values(dof_handler_pressure,
                                                  mapping,
                                                  pressure,
                                                  *point_it,
                                                  dof_index_first_dof_and_shape_values_pressure);

            for(typename std::vector<std::pair<unsigned int, std::vector<double>>>::iterator iter =
                  dof_index_first_dof_and_shape_values_pressure.begin();
                iter != dof_index_first_dof_and_shape_values_pressure.end();
                ++iter)
            {
              cells_global_pressure[line_iterator][p].push_back(*iter);
            }
          }
        }
      }
    }

    cell_data_has_been_initialized = true;
  }

  // Iterator for lines
  unsigned int line_iterator = 0;
  for(typename std::vector<Line<dim>>::iterator line = data.lines.begin(); line != data.lines.end();
      ++line, ++line_iterator)
  {
    bool evaluate_velocity = false;
    for(typename std::vector<Quantity *>::iterator quantity = line->quantities.begin();
        quantity != line->quantities.end();
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
      do_evaluate_velocity(velocity, *line, line_iterator);
    }

    bool evaluate_pressure = false;
    for(typename std::vector<Quantity *>::iterator quantity = line->quantities.begin();
        quantity != line->quantities.end();
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
      do_evaluate_pressure(pressure, *line, line_iterator);
    }
  }
}

template<int dim>
void
LinePlotCalculatorStatistics<dim>::do_evaluate_velocity(VectorType const & velocity,
                                                        Line<dim> const &  line,
                                                        unsigned int const line_iterator)
{
  // Local variables for the current line:

  // for all points along the line: velocity vector
  std::vector<Tensor<1, dim, double>> velocity_vector_local(line.n_points);
  // for all points along the line: counter
  std::vector<unsigned int> counter_vector_local(line.n_points);

  for(unsigned int p = 0; p < line.n_points; ++p)
  {
    std::vector<std::pair<unsigned int, std::vector<double>>> & adjacent_cells(
      cells_global_velocity[line_iterator][p]);

    // loop over all adjacent, locally owned cells for the current point
    for(typename std::vector<std::pair<unsigned int, std::vector<double>>>::iterator iter =
          adjacent_cells.begin();
        iter != adjacent_cells.end();
        ++iter)
    {
      // increment counter (because this is a locally owned cell)
      counter_vector_local[p] += 1;

      for(typename std::vector<Quantity *>::const_iterator quantity = line.quantities.begin();
          quantity != line.quantities.end();
          ++quantity)
      {
        if((*quantity)->type == QuantityType::Velocity)
        {
          // interpolate solution using the precomputed shape values and the global dof index
          Tensor<1, dim, double> velocity_value;
          interpolate_value_vectorial_quantity(
            dof_handler_velocity, velocity, iter->first, iter->second, velocity_value);

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
  for(typename std::vector<Quantity *>::const_iterator quantity = line.quantities.begin();
      quantity != line.quantities.end();
      ++quantity)
  {
    if((*quantity)->type == QuantityType::Velocity)
    {
      Utilities::MPI::sum(
        ArrayView<const double>(&velocity_vector_local[0][0], dim * velocity_vector_local.size()),
        communicator,
        ArrayView<double>(&velocity_vector_local[0][0], dim * velocity_vector_local.size()));

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

template<int dim>
void
LinePlotCalculatorStatistics<dim>::do_evaluate_pressure(VectorType const & pressure,
                                                        Line<dim> const &  line,
                                                        unsigned int const line_iterator)
{
  // Local variables for the current line:

  // for all points along the line: pressure value
  std::vector<double> pressure_vector_local(line.n_points);
  // for all points along the line: counter
  std::vector<unsigned int> counter_vector_local(line.n_points);

  for(unsigned int p = 0; p < line.n_points; ++p)
  {
    std::vector<std::pair<unsigned int, std::vector<double>>> & adjacent_cells(
      cells_global_pressure[line_iterator][p]);

    // loop over all adjacent, locally owned cells for the current point
    for(typename std::vector<std::pair<unsigned int, std::vector<double>>>::iterator iter =
          adjacent_cells.begin();
        iter != adjacent_cells.end();
        ++iter)
    {
      // increment counter (because this is a locally owned cell)
      counter_vector_local[p] += 1;

      for(typename std::vector<Quantity *>::const_iterator quantity = line.quantities.begin();
          quantity != line.quantities.end();
          ++quantity)
      {
        if((*quantity)->type == QuantityType::Pressure)
        {
          // interpolate solution using the precomputed shape values and the global dof index
          double pressure_value = 0.0;
          interpolate_value_scalar_quantity(
            dof_handler_pressure, pressure, iter->first, iter->second, pressure_value);

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
  for(typename std::vector<Quantity *>::const_iterator quantity = line.quantities.begin();
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

template<int dim>
void
LinePlotCalculatorStatistics<dim>::do_write_output() const
{
  if(Utilities::MPI::this_mpi_process(communicator) == 0 && data.write_output == true)
  {
    unsigned int const precision = data.precision;

    // Iterator for lines
    unsigned int line_iterator = 0;
    for(typename std::vector<Line<dim>>::const_iterator line = data.lines.begin();
        line != data.lines.end();
        ++line, ++line_iterator)
    {
      std::string filename_prefix = data.filename_prefix + line->name;

      for(typename std::vector<Quantity *>::const_iterator quantity = line->quantities.begin();
          quantity != line->quantities.end();
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
          for(unsigned int p = 0; p < line->n_points; ++p)
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

          for(unsigned int p = 0; p < line->n_points; ++p)
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

template class LinePlotCalculatorStatistics<2>;
template class LinePlotCalculatorStatistics<3>;

template<int dim>
LinePlotCalculatorStatisticsHomogeneousDirection<dim>::
  LinePlotCalculatorStatisticsHomogeneousDirection(const DoFHandler<dim> & dof_handler_velocity_in,
                                                   const DoFHandler<dim> & dof_handler_pressure_in,
                                                   const Mapping<dim> &    mapping_in)
  : clear_files(true),
    dof_handler_velocity(dof_handler_velocity_in),
    dof_handler_pressure(dof_handler_pressure_in),
    mapping(mapping_in),
    communicator(dynamic_cast<const parallel::TriangulationBase<dim> *>(
                   &dof_handler_velocity.get_triangulation()) ?
                   (dynamic_cast<const parallel::TriangulationBase<dim> *>(
                      &dof_handler_velocity.get_triangulation())
                      ->get_communicator()) :
                   MPI_COMM_SELF),
    number_of_samples(0),
    averaging_direction(2),
    write_final_output(false)
{
}

template<int dim>
void
LinePlotCalculatorStatisticsHomogeneousDirection<dim>::setup(
  LinePlotData<dim> const & line_statistics_data_in)
{
  AssertThrow(dim == 3, ExcMessage("Not implemented."));

  // use a tolerance to check whether a point is inside the unit cell
  double const tolerance = 1.e-12;

  data = line_statistics_data_in;

  velocity_global.resize(data.lines.size());
  pressure_global.resize(data.lines.size());
  reference_pressure_global.resize(data.lines.size());
  wall_shear_global.resize(data.lines.size());
  reynolds_global.resize(data.lines.size());
  global_points.resize(data.lines.size());
  cells_and_ref_points_velocity.resize(data.lines.size());
  cells_and_ref_points_pressure.resize(data.lines.size());
  ref_pressure_cells_and_ref_points.resize(data.lines.size());

  // Iterator for lines
  unsigned int line_iterator = 0;
  for(typename std::vector<Line<dim>>::iterator line = data.lines.begin(); line != data.lines.end();
      ++line, ++line_iterator)
  {
    cells_and_ref_points_velocity[line_iterator].resize(line->n_points);
    cells_and_ref_points_pressure[line_iterator].resize(line->n_points);
  }

  // initialize homogeneous direction: use the first line and the first quantity since all
  // quantities are averaged over the same homogeneous direction for all lines
  AssertThrow(data.lines.size() > 0, ExcMessage("Empty data"));
  AssertThrow(data.lines[0].quantities.size() > 0, ExcMessage("Empty data"));
  const QuantityStatistics<dim> * quantity =
    dynamic_cast<const QuantityStatistics<dim> *>(data.lines[0].quantities[0]);
  averaging_direction = quantity->averaging_direction;

  AssertThrow(averaging_direction == 0 || averaging_direction == 1 || averaging_direction == 2,
              ExcMessage("Take the average either in x, y or z - direction"));

  // Iterator for lines
  line_iterator = 0;
  for(typename std::vector<Line<dim>>::iterator line = data.lines.begin(); line != data.lines.end();
      ++line, ++line_iterator)
  {
    // Resize global variables for # of points on line
    velocity_global[line_iterator].resize(line->n_points);
    pressure_global[line_iterator].resize(line->n_points);
    wall_shear_global[line_iterator].resize(line->n_points);
    reynolds_global[line_iterator].resize(line->n_points);

    // make sure that all lines/quantities really use the same averaging direction
    for(typename std::vector<Quantity *>::iterator quantity = line->quantities.begin();
        quantity != line->quantities.end();
        ++quantity)
    {
      QuantityStatistics<dim> * stats_ptr = dynamic_cast<QuantityStatistics<dim> *>(*quantity);

      // make sure that the averaging type is correct
      AssertThrow(
        stats_ptr->average_homogeneous_direction == true,
        ExcMessage("Averaging type for QuantityStatistics is not compatible with this type of line "
                   "plot calculation, where averaging is performed in the homogeneous direction."));

      unsigned int const direction = stats_ptr->averaging_direction;

      AssertThrow(direction == averaging_direction,
                  ExcMessage("Averaging directions for different lines/quantities do not match."));
    }

    // initialize global_points: use equidistant points along line
    for(unsigned int i = 0; i < line->n_points; ++i)
    {
      Point<dim> point =
        line->begin + double(i) / double(line->n_points - 1) * (line->end - line->begin);
      global_points[line_iterator].push_back(point);
    }
  }

  // Save all cells and corresponding points on unit cell
  // that are relevant for a given point along the line.
  // For velocity quantities:
  for(typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_velocity.begin_active();
      cell != dof_handler_velocity.end();
      ++cell)
  {
    if(cell->is_locally_owned())
    {
      line_iterator = 0;
      for(typename std::vector<Line<dim>>::iterator line = data.lines.begin();
          line != data.lines.end();
          ++line, ++line_iterator)
      {
        bool velocity_has_to_be_evaluated = false;
        for(typename std::vector<Quantity *>::iterator quantity = line->quantities.begin();
            quantity != line->quantities.end();
            ++quantity)
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
            Point<dim> translated_point           = global_points[line_iterator][p];
            translated_point[averaging_direction] = cell->vertex(0)[averaging_direction];

            // If the new point lies in the current cell, we have to take the current cell into
            // account
            const Point<dim> p_unit =
              cell->real_to_unit_cell_affine_approximation(translated_point);

            if(GeometryInfo<dim>::is_inside_unit_cell(p_unit, tolerance))
            {
              cells_and_ref_points_velocity[line_iterator][p].push_back(
                std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>>(cell,
                                                                                      p_unit));
            }

            //              Point<dim> p_unit = Point<dim>();
            //              try
            //              {
            //                p_unit = mapping.transform_real_to_unit_cell(cell,
            //                translated_point);
            //              }
            //              catch(...)
            //              {
            //                // A point that does not lie on the reference cell.
            //                p_unit[0] = 2.0;
            //              }
            //              if(GeometryInfo<dim>::is_inside_unit_cell(p_unit,1.e-12))
            //              {
            //                cells_and_ref_points_velocity[line_iterator][p].push_back(
            //                    std::pair<typename DoFHandler<dim>::active_cell_iterator,
            //                    Point<dim> >(cell,p_unit));
            //              }
          }
        }
      }
    }
  }

  // Save all cells and corresponding points on unit cell that are relevant for a given point
  // along the line. We have to do the same for the pressure because the DoFHandlers for velocity
  // and pressure are different.
  for(typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_pressure.begin_active();
      cell != dof_handler_pressure.end();
      ++cell)
  {
    if(cell->is_locally_owned())
    {
      line_iterator = 0;
      for(typename std::vector<Line<dim>>::iterator line = data.lines.begin();
          line != data.lines.end();
          ++line, ++line_iterator)
      {
        for(typename std::vector<Quantity *>::iterator quantity = line->quantities.begin();
            quantity != line->quantities.end();
            ++quantity)
        {
          // evaluate quantities that involve pressure
          if((*quantity)->type == QuantityType::Pressure)
          {
            // cells and reference points for all points along a line
            for(unsigned int p = 0; p < line->n_points; ++p)
            {
              // First, we move the line to the position of the current cell (vertex 0) in
              // averaging direction and check whether this new point is inside the current cell
              Point<dim> translated_point           = global_points[line_iterator][p];
              translated_point[averaging_direction] = cell->vertex(0)[averaging_direction];

              // If the new point lies in the current cell, we have to take the current cell into
              // account
              const Point<dim> p_unit =
                cell->real_to_unit_cell_affine_approximation(translated_point);
              if(GeometryInfo<dim>::is_inside_unit_cell(p_unit, tolerance))
              {
                cells_and_ref_points_pressure[line_iterator][p].push_back(
                  std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>>(cell,
                                                                                        p_unit));
              }

              //                Point<dim> p_unit = Point<dim>();
              //                try
              //                {
              //                  p_unit = mapping.transform_real_to_unit_cell(cell,
              //                  translated_point);
              //                }
              //                catch(...)
              //                {
              //                  // A point that does not lie on the reference cell.
              //                  p_unit[0] = 2.0;
              //                }
              //                if(GeometryInfo<dim>::is_inside_unit_cell(p_unit,1.e-12))
              //                {
              //                  cells_and_ref_points_pressure[line_iterator][p].push_back(
              //                    std::pair<typename DoFHandler<dim>::active_cell_iterator,
              //                    Point<dim> >(cell,p_unit));
              //                }
            }
          }
        }

        // cells and reference points for reference pressure (only one point for each line)
        for(typename std::vector<Quantity *>::iterator quantity = line->quantities.begin();
            quantity != line->quantities.end();
            ++quantity)
        {
          // evaluate quantities that involve pressure
          if((*quantity)->type == QuantityType::PressureCoefficient)
          {
            QuantityStatisticsPressureCoefficient<dim> * quantity_ref_pressure =
              dynamic_cast<QuantityStatisticsPressureCoefficient<dim> *>(*quantity);

            // First, we move the line to the position of the current cell (vertex 0) in
            // averaging direction and check whether this new point is inside the current cell
            Point<dim> translated_point           = quantity_ref_pressure->reference_point;
            translated_point[averaging_direction] = cell->vertex(0)[averaging_direction];

            // If the new point lies in the current cell, we have to take the current cell into
            // account
            const Point<dim> p_unit =
              cell->real_to_unit_cell_affine_approximation(translated_point);
            if(GeometryInfo<dim>::is_inside_unit_cell(p_unit, tolerance))
            {
              ref_pressure_cells_and_ref_points[line_iterator].push_back(
                std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>>(cell,
                                                                                      p_unit));
            }

            //              Point<dim> p_unit = Point<dim>();
            //              try
            //              {
            //                p_unit = mapping.transform_real_to_unit_cell(cell,
            //                translated_point);
            //              }
            //              catch(...)
            //              {
            //                // A point that does not lie on the reference cell.
            //                p_unit[0] = 2.0;
            //              }
            //              if(GeometryInfo<dim>::is_inside_unit_cell(p_unit,1.e-12))
            //              {
            //                ref_pressure_cells_and_ref_points[line_iterator].push_back(
            //                    std::pair<typename DoFHandler<dim>::active_cell_iterator,
            //                    Point<dim> >(cell,p_unit));
            //              }
          }
        }
      }
    }
  }
}

template<int dim>
void
LinePlotCalculatorStatisticsHomogeneousDirection<dim>::evaluate(
  VectorType const &   velocity,
  VectorType const &   pressure,
  double const &       time,
  unsigned int const & time_step_number)
{
  // EPSILON: small number which is much smaller than the time step size
  const double EPSILON = 1.0e-10;
  if(data.statistics_data.calculate_statistics == true)
  {
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

template<int dim>
void
LinePlotCalculatorStatisticsHomogeneousDirection<dim>::do_evaluate(VectorType const & velocity,
                                                                   VectorType const & pressure)
{
  // increment number of samples
  number_of_samples++;

  // Iterator for lines
  unsigned int line_iterator = 0;
  for(typename std::vector<Line<dim>>::iterator line = data.lines.begin(); line != data.lines.end();
      ++line, ++line_iterator)
  {
    bool evaluate_velocity = false;
    for(typename std::vector<Quantity *>::iterator quantity = line->quantities.begin();
        quantity != line->quantities.end();
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
      do_evaluate_velocity(velocity, *line, line_iterator);

    bool evaluate_pressure = false;
    for(typename std::vector<Quantity *>::iterator quantity = line->quantities.begin();
        quantity != line->quantities.end();
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
      do_evaluate_pressure(pressure, *line, line_iterator);
  }
}

template<int dim>
void
LinePlotCalculatorStatisticsHomogeneousDirection<dim>::do_evaluate_velocity(
  VectorType const & velocity,
  Line<dim> const &  line,
  unsigned int const line_iterator)
{
  // Local variables for specific line
  std::vector<double>                 length_local(line.n_points);
  std::vector<Tensor<1, dim, double>> velocity_local(line.n_points);
  std::vector<double>                 wall_shear_local(line.n_points);
  std::vector<Tensor<2, dim, double>> reynolds_local(line.n_points);

  const unsigned int scalar_dofs_per_cell =
    dof_handler_velocity.get_fe().base_element(0).dofs_per_cell;

  std::vector<Tensor<1, dim>> velocity_vector(scalar_dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dof_handler_velocity.get_fe().dofs_per_cell);

  for(unsigned int p = 0; p < line.n_points; ++p)
  {
    for(typename TYPE::const_iterator cell_and_ref_point =
          cells_and_ref_points_velocity[line_iterator][p].begin();
        cell_and_ref_point != cells_and_ref_points_velocity[line_iterator][p].end();
        ++cell_and_ref_point)
    {
      const unsigned int fe_degree_velocity = dof_handler_velocity.get_fe().degree;

      // use quadrature for averaging in homogeneous direction
      QGauss<1>               gauss_1d(fe_degree_velocity + 1);
      std::vector<Point<dim>> points(gauss_1d.size());  // 1D points
      std::vector<double>     weights(gauss_1d.size()); // 1D weights

      typename DoFHandler<dim>::active_cell_iterator const cell = cell_and_ref_point->first;

      Point<dim> const p_unit = cell_and_ref_point->second;

      // Find points and weights for Gauss quadrature
      find_points_and_weights(p_unit, points, weights, averaging_direction, gauss_1d);

      FEValues<dim, dim> fe_values(mapping,
                                   dof_handler_velocity.get_fe().base_element(0),
                                   Quadrature<dim>(points, weights),
                                   update_values | update_jacobians | update_quadrature_points |
                                     update_gradients);

      fe_values.reinit(typename Triangulation<dim>::active_cell_iterator(cell));

      cell->get_dof_indices(dof_indices);

      // resort velocity dofs
      for(unsigned int j = 0; j < dof_indices.size(); ++j)
      {
        const std::pair<unsigned int, unsigned int> comp =
          dof_handler_velocity.get_fe().system_to_component_index(j);
        if(comp.first < dim)
          velocity_vector[comp.second][comp.first] = velocity(dof_indices[j]);
      }

      // perform averaging in homogeneous direction
      for(unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
      {
        double det = std::abs(fe_values.jacobian(q)[averaging_direction][averaging_direction]);
        double JxW = det * fe_values.get_quadrature().weight(q);

        // calculate integrals in homogeneous direction
        length_local[p] += JxW;

        for(typename std::vector<Quantity *>::const_iterator quantity = line.quantities.begin();
            quantity != line.quantities.end();
            ++quantity)
        {
          Tensor<1, dim> velocity;

          if((*quantity)->type == QuantityType::Velocity ||
             (*quantity)->type == QuantityType::ReynoldsStresses)
          {
            // evaluate velocity solution in current quadrature points
            for(unsigned int j = 0; j < velocity_vector.size(); ++j)
              velocity += fe_values.shape_value(j, q) * velocity_vector[j];
          }

          if((*quantity)->type == QuantityType::Velocity)
          {
            for(unsigned int i = 0; i < dim; ++i)
              velocity_local[p][i] += velocity[i] * JxW;
          }
          else if((*quantity)->type == QuantityType::ReynoldsStresses)
          {
            for(unsigned int i = 0; i < dim; ++i)
            {
              for(unsigned int j = 0; j < dim; ++j)
              {
                reynolds_local[p][i][j] += velocity[i] * velocity[j] * JxW;
              }
            }
          }
          else if((*quantity)->type == QuantityType::SkinFriction)
          {
            Tensor<2, dim> velocity_gradient;
            for(unsigned int j = 0; j < velocity_vector.size(); ++j)
              velocity_gradient += outer_product(velocity_vector[j], fe_values.shape_grad(j, q));

            const QuantityStatisticsSkinFriction<dim> * quantity_skin_friction =
              dynamic_cast<const QuantityStatisticsSkinFriction<dim> *>(*quantity);

            Tensor<1, dim, double> normal  = quantity_skin_friction->normal_vector;
            Tensor<1, dim, double> tangent = quantity_skin_friction->tangent_vector;

            for(unsigned int i = 0; i < dim; ++i)
              for(unsigned int j = 0; j < dim; ++j)
                wall_shear_local[p] += tangent[i] * velocity_gradient[i][j] * normal[j] * JxW;
          }
        }
      }
    }
  }

  Utilities::MPI::sum(length_local, communicator, length_local);

  for(typename std::vector<Quantity *>::const_iterator quantity = line.quantities.begin();
      quantity != line.quantities.end();
      ++quantity)
  {
    // Cells are distributed over processors, therefore we need
    // to sum the contributions of every single processor.
    if((*quantity)->type == QuantityType::Velocity)
    {
      Utilities::MPI::sum(ArrayView<const double>(&velocity_local[0][0],
                                                  dim * velocity_local.size()),
                          communicator,
                          ArrayView<double>(&velocity_local[0][0], dim * velocity_local.size()));

      for(unsigned int p = 0; p < line.n_points; ++p)
      {
        for(unsigned int d = 0; d < dim; ++d)
        {
          velocity_global[line_iterator][p][d] += velocity_local[p][d] / length_local[p];
        }
      }
    }
    else if((*quantity)->type == QuantityType::ReynoldsStresses)
    {
      Utilities::MPI::sum(
        ArrayView<const double>(&reynolds_local[0][0][0], dim * dim * reynolds_local.size()),
        communicator,
        ArrayView<double>(&reynolds_local[0][0][0], dim * dim * reynolds_local.size()));

      for(unsigned int p = 0; p < line.n_points; ++p)
      {
        for(unsigned int i = 0; i < dim; ++i)
        {
          for(unsigned int j = 0; j < dim; ++j)
          {
            reynolds_global[line_iterator][p][i][j] += reynolds_local[p][i][j] / length_local[p];
          }
        }
      }
    }
    else if((*quantity)->type == QuantityType::SkinFriction)
    {
      Utilities::MPI::sum(wall_shear_local, communicator, wall_shear_local);

      for(unsigned int p = 0; p < line.n_points; ++p)
      {
        wall_shear_global[line_iterator][p] += wall_shear_local[p] / length_local[p];
      }
    }
  }
}


template<int dim>
void
LinePlotCalculatorStatisticsHomogeneousDirection<dim>::do_evaluate_pressure(
  VectorType const & pressure,
  Line<dim> const &  line,
  unsigned int const line_iterator)
{
  for(typename std::vector<Quantity *>::const_iterator quantity = line.quantities.begin();
      quantity != line.quantities.end();
      ++quantity)
  {
    if((*quantity)->type == QuantityType::Pressure)
    {
      // Local variables for specific line
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
      for(unsigned int p = 0; p < line.n_points; ++p)
        pressure_global[line_iterator][p] += pressure_local[p] / length_local[p];
    }

    if((*quantity)->type == QuantityType::PressureCoefficient)
    {
      double length_local   = 0.0;
      double pressure_local = 0.0;

      TYPE vector_cells_and_ref_points = ref_pressure_cells_and_ref_points[line_iterator];

      average_pressure_for_given_point(pressure,
                                       vector_cells_and_ref_points,
                                       length_local,
                                       pressure_local);

      // MPI communication
      length_local   = Utilities::MPI::sum(length_local, communicator);
      pressure_local = Utilities::MPI::sum(pressure_local, communicator);

      // averaging in space (over homogeneous direction)
      reference_pressure_global[line_iterator] += pressure_local / length_local;
    }
  }
}

template<int dim>
void
LinePlotCalculatorStatisticsHomogeneousDirection<dim>::average_pressure_for_given_point(
  VectorType const & pressure,
  TYPE const &       vector_cells_and_ref_points,
  double &           length_local,
  double &           pressure_local)
{
  const unsigned int scalar_dofs_per_cell =
    dof_handler_pressure.get_fe().base_element(0).dofs_per_cell;
  std::vector<double>                  pressure_vector(scalar_dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dof_handler_pressure.get_fe().dofs_per_cell);

  for(typename TYPE::const_iterator cell_and_ref_point = vector_cells_and_ref_points.begin();
      cell_and_ref_point != vector_cells_and_ref_points.end();
      ++cell_and_ref_point)
  {
    const unsigned int fe_degree_pressure = dof_handler_pressure.get_fe().degree;

    // use quadrature for averaging in homogeneous direction
    QGauss<1>               gauss_1d(fe_degree_pressure + 1);
    std::vector<Point<dim>> points(gauss_1d.size());  // 1D points
    std::vector<double>     weights(gauss_1d.size()); // 1D weights

    typename DoFHandler<dim>::active_cell_iterator const cell = cell_and_ref_point->first;

    Point<dim> const p_unit = cell_and_ref_point->second;

    // Find points and weights for Gauss quadrature
    find_points_and_weights(p_unit, points, weights, averaging_direction, gauss_1d);

    FEValues<dim, dim> fe_values(mapping,
                                 dof_handler_pressure.get_fe().base_element(0),
                                 Quadrature<dim>(points, weights),
                                 update_values | update_jacobians | update_quadrature_points);

    fe_values.reinit(typename Triangulation<dim>::active_cell_iterator(cell));

    cell->get_dof_indices(dof_indices);

    for(unsigned int j = 0; j < scalar_dofs_per_cell; ++j)
      pressure_vector[j] = pressure(dof_indices[j]);

    for(unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
    {
      double p = 0;

      for(unsigned int j = 0; j < pressure_vector.size(); ++j)
        p += fe_values.shape_value(j, q) * pressure_vector[j];

      double det = std::abs(fe_values.jacobian(q)[averaging_direction][averaging_direction]);
      double JxW = det * fe_values.get_quadrature().weight(q);

      length_local += JxW;
      pressure_local += p * JxW;
    }
  }
}

template<int dim>
void
LinePlotCalculatorStatisticsHomogeneousDirection<dim>::find_points_and_weights(
  Point<dim> const &        point_in_ref_coord,
  std::vector<Point<dim>> & points,
  std::vector<double> &     weights,
  unsigned int const        averaging_direction,
  QGauss<1> const &         gauss_1d)
{
  for(unsigned int q = 0; q < gauss_1d.size(); ++q)
  {
    for(unsigned int d = 0; d < dim; ++d)
    {
      if(d == averaging_direction)
        points[q][d] = gauss_1d.point(q)[0];
      else
        points[q][d] = point_in_ref_coord[d];
    }
    weights[q] = gauss_1d.weight(q);
  }
}

template<int dim>
void
LinePlotCalculatorStatisticsHomogeneousDirection<dim>::do_write_output() const
{
  if(Utilities::MPI::this_mpi_process(communicator) == 0 && data.write_output == true)
  {
    unsigned int const precision = data.precision;

    // Iterator for lines
    unsigned int line_iterator = 0;
    for(typename std::vector<Line<dim>>::const_iterator line = data.lines.begin();
        line != data.lines.end();
        ++line, ++line_iterator)
    {
      std::string filename_prefix = data.filename_prefix + line->name;

      for(typename std::vector<Quantity *>::const_iterator quantity = line->quantities.begin();
          quantity != line->quantities.end();
          ++quantity)
      {
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
          for(unsigned int p = 0; p < line->n_points; ++p)
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
          std::string   filename = filename_prefix + "_reynoldsstresses" + ".txt";
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

          for(unsigned int i = 0; i < dim; ++i)
          {
            for(unsigned int j = 0; j < dim; ++j)
            {
              f << std::setw(precision + 8) << std::left
                << "u_" + Utilities::int_to_string(i + 1) + "u_" + Utilities::int_to_string(j + 1);
            }
          }
          f << std::endl;

          // loop over all points
          for(unsigned int p = 0; p < line->n_points; ++p)
          {
            f << std::scientific << std::setprecision(precision);

            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left << global_points[line_iterator][p][d];

            for(unsigned int i = 0; i < dim; ++i)
            {
              for(unsigned int j = 0; j < dim; ++j)
              {
                // equation <u_i' u_j'> = <u_i*u_j> - <u_i> * <u_j>
                f << std::setw(precision + 8) << std::left
                  << reynolds_global[line_iterator][p][i][j] / number_of_samples -
                       (velocity_global[line_iterator][p][i] / number_of_samples) *
                         (velocity_global[line_iterator][p][j] / number_of_samples);
              }
            }

            f << std::endl;
          }
          f.close();
        }

        if((*quantity)->type == QuantityType::SkinFriction)
        {
          QuantityStatisticsSkinFriction<dim> * averaging_quantity =
            dynamic_cast<QuantityStatisticsSkinFriction<dim> *>(*quantity);

          std::string   filename = filename_prefix + "_wall_shear_stress" + ".txt";
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

          f << std::setw(precision + 8) << std::left << "tau_w" << std::endl;

          // loop over all points
          for(unsigned int p = 0; p < line->n_points; ++p)
          {
            f << std::scientific << std::setprecision(precision);

            // write data
            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left << global_points[line_iterator][p][d];

            // tau_w -> C_f = tau_w / (1/2 rho u²)
            double const viscosity = averaging_quantity->viscosity;
            f << std::setw(precision + 8) << std::left
              << viscosity * wall_shear_global[line_iterator][p] / number_of_samples;

            f << std::endl;
          }
          f.close();
        }

        if((*quantity)->type == QuantityType::Pressure ||
           (*quantity)->type == QuantityType::PressureCoefficient)
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

          if((*quantity)->type == QuantityType::PressureCoefficient)
            f << std::setw(precision + 8) << std::left << "p-p_ref";

          f << std::endl;

          for(unsigned int p = 0; p < line->n_points; ++p)
          {
            f << std::scientific << std::setprecision(precision);

            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left << global_points[line_iterator][p][d];

            f << std::setw(precision + 8) << std::left
              << pressure_global[line_iterator][p] / number_of_samples;

            if((*quantity)->type == QuantityType::PressureCoefficient)
            {
              // p - p_ref -> C_p = (p - p_ref) / (1/2 rho u²)
              f << std::left
                << (pressure_global[line_iterator][p] - reference_pressure_global[line_iterator]) /
                     number_of_samples;
            }
            f << std::endl;
          }
          f.close();
        }
      }
    }
  }
}

template class LinePlotCalculatorStatisticsHomogeneousDirection<2>;
template class LinePlotCalculatorStatisticsHomogeneousDirection<3>;
