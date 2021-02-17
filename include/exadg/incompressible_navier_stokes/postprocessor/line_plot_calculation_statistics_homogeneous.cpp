/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// deal.II
#include <deal.II/fe/fe_values.h>

// ExaDG
#include <exadg/incompressible_navier_stokes/postprocessor/line_plot_calculation_statistics_homogeneous.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim, typename Number>
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::LinePlotCalculatorStatisticsHomogeneous(
  DoFHandler<dim> const & dof_handler_velocity_in,
  DoFHandler<dim> const & dof_handler_pressure_in,
  Mapping<dim> const &    mapping_in,
  MPI_Comm const &        mpi_comm_in)
  : clear_files(true),
    dof_handler_velocity(dof_handler_velocity_in),
    dof_handler_pressure(dof_handler_pressure_in),
    mapping(mapping_in),
    communicator(mpi_comm_in),
    number_of_samples(0),
    averaging_direction(2),
    write_final_output(false)
{
}

template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::setup(
  LinePlotDataStatistics<dim> const & data_in)
{
  data = data_in;

  if(data.statistics_data.calculate_statistics == true)
  {
    AssertThrow(dim == 3, ExcMessage("Not implemented."));

    AssertThrow(data.line_data.lines.size() > 0, ExcMessage("Empty data"));

    global_points.resize(data.line_data.lines.size());
    cells_and_ref_points_velocity.resize(data.line_data.lines.size());
    cells_and_ref_points_pressure.resize(data.line_data.lines.size());
    cells_and_ref_points_ref_pressure.resize(data.line_data.lines.size());

    velocity_global.resize(data.line_data.lines.size());
    wall_shear_global.resize(data.line_data.lines.size());
    reynolds_global.resize(data.line_data.lines.size());
    pressure_global.resize(data.line_data.lines.size());
    reference_pressure_global.resize(data.line_data.lines.size());

    // make sure that line type is correct
    std::shared_ptr<LineHomogeneousAveraging<dim>> line_hom =
      std::dynamic_pointer_cast<LineHomogeneousAveraging<dim>>(data.line_data.lines[0]);
    AssertThrow(line_hom.get() != 0,
                ExcMessage("Invalid line type, expected LineHomogeneousAveraging<dim>"));
    averaging_direction = line_hom->averaging_direction;

    AssertThrow(averaging_direction == 0 || averaging_direction == 1 || averaging_direction == 2,
                ExcMessage("Take the average either in x, y or z-direction"));

    unsigned int line_iterator = 0;
    for(typename std::vector<std::shared_ptr<Line<dim>>>::iterator line =
          data.line_data.lines.begin();
        line != data.line_data.lines.end();
        ++line, ++line_iterator)
    {
      // make sure that line type is correct
      std::shared_ptr<LineHomogeneousAveraging<dim>> line_hom =
        std::dynamic_pointer_cast<LineHomogeneousAveraging<dim>>(*line);

      AssertThrow(line_hom.get() != 0,
                  ExcMessage("Invalid line type, expected LineHomogeneousAveraging<dim>"));

      AssertThrow(averaging_direction == line_hom->averaging_direction,
                  ExcMessage("All lines must use the same averaging direction."));

      // Resize global variables for # of points on line
      velocity_global[line_iterator].resize((*line)->n_points);
      pressure_global[line_iterator].resize((*line)->n_points);
      wall_shear_global[line_iterator].resize((*line)->n_points);
      reynolds_global[line_iterator].resize((*line)->n_points);
      cells_and_ref_points_velocity[line_iterator].resize((*line)->n_points);
      cells_and_ref_points_pressure[line_iterator].resize((*line)->n_points);

      // initialize global_points: use equidistant points along line
      for(unsigned int i = 0; i < (*line)->n_points; ++i)
      {
        Point<dim> point = (*line)->begin + double(i) / double((*line)->n_points - 1) *
                                              ((*line)->end - (*line)->begin);
        global_points[line_iterator].push_back(point);
      }
    }

    // Save all cells and corresponding points on unit cell
    // that are relevant for a given point along the line.

    // use a tolerance to check whether a point is inside the unit cell
    double const tolerance = 1.e-12;

    // For velocity quantities:
    for(typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_velocity.begin_active();
        cell != dof_handler_velocity.end();
        ++cell)
    {
      if(cell->is_locally_owned())
      {
        line_iterator = 0;
        for(typename std::vector<std::shared_ptr<Line<dim>>>::iterator line =
              data.line_data.lines.begin();
            line != data.line_data.lines.end();
            ++line, ++line_iterator)
        {
          AssertThrow((*line)->quantities.size() > 0,
                      ExcMessage("No quantities specified for line."));

          bool velocity_has_to_be_evaluated = false;
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
          }

          if(velocity_has_to_be_evaluated == true)
          {
            // cells and reference points for all points along a line
            for(unsigned int p = 0; p < (*line)->n_points; ++p)
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
        for(typename std::vector<std::shared_ptr<Line<dim>>>::iterator line =
              data.line_data.lines.begin();
            line != data.line_data.lines.end();
            ++line, ++line_iterator)
        {
          for(typename std::vector<std::shared_ptr<Quantity>>::iterator quantity =
                (*line)->quantities.begin();
              quantity != (*line)->quantities.end();
              ++quantity)
          {
            AssertThrow((*line)->quantities.size() > 0,
                        ExcMessage("No quantities specified for line."));

            // evaluate quantities that involve pressure
            if((*quantity)->type == QuantityType::Pressure)
            {
              // cells and reference points for all points along a line
              for(unsigned int p = 0; p < (*line)->n_points; ++p)
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
          for(typename std::vector<std::shared_ptr<Quantity>>::iterator quantity =
                (*line)->quantities.begin();
              quantity != (*line)->quantities.end();
              ++quantity)
          {
            AssertThrow((*line)->quantities.size() > 0,
                        ExcMessage("No quantities specified for line."));

            // evaluate quantities that involve pressure
            if((*quantity)->type == QuantityType::PressureCoefficient)
            {
              std::shared_ptr<QuantityPressureCoefficient<dim>> quantity_ref_pressure =
                std::dynamic_pointer_cast<QuantityPressureCoefficient<dim>>(*quantity);

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
                cells_and_ref_points_ref_pressure[line_iterator].push_back(
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
}

template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::evaluate(
  VectorType const &   velocity,
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
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::print_headline(
  std::ofstream &    f,
  unsigned int const number_of_samples) const
{
  f << "number of samples: N = " << number_of_samples << std::endl;
}

template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::do_evaluate(VectorType const & velocity,
                                                                  VectorType const & pressure)
{
  // increment number of samples
  number_of_samples++;

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
      do_evaluate_velocity(velocity, *(*line), line_iterator);

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
      do_evaluate_pressure(pressure, *(*line), line_iterator);
  }
}

template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::do_evaluate_velocity(
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

        for(typename std::vector<std::shared_ptr<Quantity>>::const_iterator quantity =
              line.quantities.begin();
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

            std::shared_ptr<QuantitySkinFriction<dim>> quantity_skin_friction =
              std::dynamic_pointer_cast<QuantitySkinFriction<dim>>(*quantity);

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

  for(typename std::vector<std::shared_ptr<Quantity>>::const_iterator quantity =
        line.quantities.begin();
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


template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::do_evaluate_pressure(
  VectorType const & pressure,
  Line<dim> const &  line,
  unsigned int const line_iterator)
{
  for(typename std::vector<std::shared_ptr<Quantity>>::const_iterator quantity =
        line.quantities.begin();
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

      TYPE vector_cells_and_ref_points = cells_and_ref_points_ref_pressure[line_iterator];

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

template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::average_pressure_for_given_point(
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

template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::find_points_and_weights(
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

template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::do_write_output() const
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
          for(unsigned int p = 0; p < (*line)->n_points; ++p)
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
          std::shared_ptr<QuantitySkinFriction<dim>> averaging_quantity =
            std::dynamic_pointer_cast<QuantitySkinFriction<dim>>(*quantity);

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
          for(unsigned int p = 0; p < (*line)->n_points; ++p)
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

          for(unsigned int p = 0; p < (*line)->n_points; ++p)
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

template class LinePlotCalculatorStatisticsHomogeneous<2, float>;
template class LinePlotCalculatorStatisticsHomogeneous<3, float>;

template class LinePlotCalculatorStatisticsHomogeneous<2, double>;
template class LinePlotCalculatorStatisticsHomogeneous<3, double>;

} // namespace IncNS
} // namespace ExaDG
