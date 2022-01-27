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

// C/C++
#include <fstream>

// deal.II
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/tria_base.h>
#include <deal.II/fe/fe_values.h>

// ExaDG
#include <exadg/postprocessor/statistics_manager.h>
#include <exadg/utilities/create_directories.h>

//#define OUTPUT_DEBUG_INFO

namespace ExaDG
{
template<int dim, typename Number>
StatisticsManager<dim, Number>::StatisticsManager(
  dealii::DoFHandler<dim> const & dof_handler_velocity,
  dealii::Mapping<dim> const &    mapping_in)
  : n_points_y_per_cell(0),
    dof_handler(dof_handler_velocity),
    mapping(mapping_in),
    mpi_comm(dof_handler_velocity.get_communicator()),
    number_of_samples(0),
    write_final_output(true),
    data(TurbulentChannelData())
{
}


template<int dim, typename Number>
void
StatisticsManager<dim, Number>::setup(const std::function<double(double const &)> & grid_transform,
                                      TurbulentChannelData const &                  data_in)
{
  data = data_in;

  if(data.calculate)
  {
    // note: this code only works on structured meshes where the faces in
    // y-direction are faces 2 and 3

    /*
     *           face 3
     *   __________________________
     *  y      |       |
     *         |_______|
     * /|\     |       |
     *  |      |_______| n_cells_y_dir = 3
     *  |      |       |
     *   ______|_______|___________
     *
     *           face 2
     */

    // find the number of refinements in the mesh, first the number of coarse
    // cells in y-direction and then the number of refinements.
    unsigned int n_cells_y_dir = 1;

    typename dealii::Triangulation<dim>::cell_iterator cell =
      dof_handler.get_triangulation().begin(0);
    while(cell != dof_handler.get_triangulation().end(0) && !cell->at_boundary(2))
    {
      ++cell;
    }
    while(!cell->at_boundary(3))
    {
      ++n_cells_y_dir;
      cell = cell->neighbor(3);
    }

    unsigned int const fe_degree = dof_handler.get_fe().degree;

    n_points_y_per_cell = n_points_y_per_cell_linear * fe_degree;

    AssertThrow(n_points_y_per_cell >= 2,
                dealii::ExcMessage("Number of points in y-direction per cell is invalid."));

    n_cells_y_dir *=
      dealii::Utilities::pow(2, dof_handler.get_triangulation().n_global_levels() - 1);

    unsigned int const n_points_y_glob = n_cells_y_dir * (n_points_y_per_cell - 1) + 1;

    // velocity vector with 3-components
    vel_glob.resize(3);
    for(unsigned int i = 0; i < 3; i++)
      vel_glob[i].resize(n_points_y_glob); // vector for all y-coordinates

    // velocity vector with 3-components
    velsq_glob.resize(3);
    for(unsigned int i = 0; i < 3; i++)
      velsq_glob[i].resize(n_points_y_glob); // vector for all y-coordinates

    // u*v (scalar quantity)
    veluv_glob.resize(n_points_y_glob); // vector for all y-coordinates

    // initialize number of samples
    number_of_samples = 0;

    // calculate y-coordinates in physical space where we want to perform the sampling (averaging)
    y_glob.reserve(n_points_y_glob);

    // loop over all cells in y-direction
    if(data.cells_are_stretched == true)
    {
      for(unsigned int cell = 0; cell < n_cells_y_dir; cell++)
      {
        // determine lower and upper y-coordinates of current cell in ref space [0,1]
        double pointlower = 1. / (double)n_cells_y_dir * (double)cell;
        double pointupper = 1. / (double)n_cells_y_dir * (double)(cell + 1);

        // loop over all y-coordinates inside the current cell
        for(unsigned int plane = 0; plane < n_points_y_per_cell - 1; plane++)
        {
          // reference space: use a linear distribution inside each cell [0,1]
          double coord_ref =
            pointlower + (pointupper - pointlower) / (n_points_y_per_cell - 1) * plane;

          // transform ref coordinate [0,1] to physical space
          double y_coord = grid_transform(coord_ref);

          y_glob.push_back(y_coord);
        }

        // push back last missing coordinate at upper cell/wall
        if(cell == n_cells_y_dir - 1)
        {
          double y_coord = grid_transform(pointupper);
          y_glob.push_back(y_coord);
        }
      }

#ifdef OUTPUT_DEBUG_INFO
      std::cout << std::endl << "Intermediate vector with y-coordinates:" << std::endl;
      for(unsigned int i = 0; i < y_glob.size(); ++i)
        std::cout << "y_glob[" << i << "]=" << y_glob[i] << std::endl;
      std::vector<double> y_temp;
      y_temp = y_glob;
#endif

      // y_glob contains y-coordinates using the exact mapping

      // However, when calculating the statistics we use the polynomial mapping of degree
      // 'fe_degree' which leads to slightly different values as compared to the exact mapping.
      // -> overwrite values in y_glob with values resulting from polynomial mapping

      // use 2d quadrature to integrate over x-z-planes
      unsigned int const      fe_degree = dof_handler.get_fe().degree;
      dealii::QGauss<dim - 1> gauss_2d(fe_degree + 1);

      std::vector<double> y_processor;
      y_processor.resize(n_points_y_glob, std::numeric_limits<double>::lowest());

      // vector of dealii::FEValues for all x-z-planes of a cell
      std::vector<std::shared_ptr<dealii::FEValues<dim, dim>>> fe_values(n_points_y_per_cell);

      for(unsigned int i = 0; i < n_points_y_per_cell; ++i)
      {
        std::vector<dealii::Point<dim>> points(gauss_2d.size());
        std::vector<double>             weights(gauss_2d.size());
        for(unsigned int j = 0; j < gauss_2d.size(); ++j)
        {
          points[j][0] = gauss_2d.point(j)[0];
          if(dim == 3)
            points[j][2] = gauss_2d.point(j)[1];
          points[j][1] = (double)i / (n_points_y_per_cell - 1);
          weights[j]   = gauss_2d.weight(j);
        }
        fe_values[i].reset(new dealii::FEValues<dim>(mapping,
                                                     dof_handler.get_fe().base_element(0),
                                                     dealii::Quadrature<dim>(points, weights),
                                                     dealii::update_values |
                                                       dealii::update_jacobians |
                                                       dealii::update_quadrature_points));
      }

      // loop over all cells
      for(typename dealii::DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
          cell != dof_handler.end();
          ++cell)
      {
        if(cell->is_locally_owned())
        {
          // loop over all y-coordinates of current cell
          unsigned int idx = 0;
          for(unsigned int i = 0; i < n_points_y_per_cell; ++i)
          {
            fe_values[i]->reinit(typename dealii::Triangulation<dim>::active_cell_iterator(cell));

            // Transform cell index 'i' to global index 'idx' of y_glob-vector

            // find index within the y-values: first do a binary search to find
            // the next larger value of y in the list...
            double const y = fe_values[i]->quadrature_point(0)[1];

            // identify index for first point located on the boundary of the cell because for this
            // point the mapping can not cause any trouble. For interior points, the deviations
            // introduced may the mapping can be so strong that the identification of the index is
            // no longer unique.
            if(i == 0)
            {
              idx =
                std::distance(y_glob.begin(), std::lower_bound(y_glob.begin(), y_glob.end(), y));

              // make sure that the index does not exceed the array bounds in case of round-off
              // errors
              if(idx == y_glob.size())
                idx--;

              // reduce index by 1 in case that the previous point is closer to y than
              // the next point
              if(idx > 0 && std::abs(y_glob[idx - 1] - y) < std::abs(y_glob[idx] - y))
                idx--;
            }
            else // simply increment index for subsequent points of a cell
            {
              ++idx;
            }

            y_processor[idx] = y;
          }
        }
      }

      dealii::Utilities::MPI::max(y_processor, mpi_comm, y_glob);

#ifdef OUTPUT_DEBUG_INFO
      // print final vector
      std::cout << std::endl << "Final vector with y-coordinates:" << std::endl;
      for(unsigned int i = 0; i < y_glob.size(); ++i)
        std::cout << "y_glob[" << i << "] = " << y_glob[i] << std::endl;

      // compare intermediate and final vector
      for(unsigned int i = 0; i < y_glob.size(); ++i)
        std::cout << "y_temp[" << i << "] = " << y_temp[i] << " "
                  << "y_glob[" << i << "] = " << y_glob[i] << std::endl;
#endif
    }
    else // data.cells_are_stretched == false
    {
      // use equidistant distribution of points within each cell
      for(unsigned int cell = 0; cell < n_cells_y_dir; cell++)
      {
        // determine lower and upper y-coordinates of current cell in physical space
        double pointlower = 1. / (double)n_cells_y_dir * (double)cell;
        double pointupper = 1. / (double)n_cells_y_dir * (double)(cell + 1);
        double ylower     = grid_transform(pointlower);
        double yupper     = grid_transform(pointupper);

        // loop over all y-coordinates inside the current cell
        for(unsigned int plane = 0; plane < n_points_y_per_cell - 1; plane++)
        {
          // use a linear distribution inside each cell
          double coord = ylower + (yupper - ylower) / (n_points_y_per_cell - 1) * plane;
          y_glob.push_back(coord);
        }

        // push back last missing coordinate at upper cell/wall
        if(cell == n_cells_y_dir - 1)
        {
          y_glob.push_back(yupper);
        }
      }
    }

    AssertThrow(y_glob.size() == n_points_y_glob, dealii::ExcInternalError());

    create_directories(data.directory, mpi_comm);
  }
}

template<int dim, typename Number>
void
StatisticsManager<dim, Number>::evaluate(VectorType const &   velocity,
                                         double const &       time,
                                         unsigned int const & time_step_number)
{
  if(data.calculate)
  {
    std::string filename = data.directory + data.filename;

    // EPSILON: small number which is much smaller than the time step size
    double const EPSILON = 1.0e-10;
    if((time > data.sample_start_time - EPSILON) && (time < data.sample_end_time + EPSILON) &&
       (time_step_number % data.sample_every_timesteps == 0))
    {
      // evaluate statistics
      this->evaluate(velocity);

      // write intermediate output
      if(time_step_number % (data.sample_every_timesteps * 100) == 0)
      {
        this->write_output(filename, data.viscosity, data.density);
      }
    }

    // write final output
    if((time > data.sample_end_time - EPSILON) && write_final_output)
    {
      this->write_output(filename, data.viscosity, data.density);

      write_final_output = false;
    }
  }
}


template<int dim, typename Number>
void
StatisticsManager<dim, Number>::evaluate(VectorType const & velocity)
{
  std::vector<VectorType const *> vecs;
  vecs.push_back(&velocity);
  do_evaluate(vecs);
}



template<int dim, typename Number>
void
StatisticsManager<dim, Number>::evaluate(std::vector<VectorType> const & velocity)
{
  std::vector<VectorType const *> vecs;
  for(unsigned int i = 0; i < velocity.size(); ++i)
    vecs.push_back(&velocity[i]);
  do_evaluate(vecs);
}

template<int dim, typename Number>
void
StatisticsManager<dim, Number>::write_output(const std::string filename,
                                             double const      dynamic_viscosity,
                                             double const      density)
{
  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    // tau_w = mu * d<u>/dy = mu * (<u>(y2)-<u>(y1))/(y2-y1), where mu = rho * nu
    double tau_w = dynamic_viscosity * ((vel_glob[0].at(1) - vel_glob[0].at(0)) /
                                        (double)number_of_samples / (y_glob.at(1) - y_glob.at(0)));

    // Re_tau = u_tau * delta / nu = sqrt(tau_w/rho) * delta / (mu/rho), where delta = 1
    double Re_tau = sqrt(tau_w / density) / (dynamic_viscosity / density);

    std::ofstream f;
    f.open((filename + ".flow_statistics").c_str(), std::ios::trunc);

    // clang-format off
    f << std::scientific << std::setprecision(7)
      << "Statistics of turbulent channel flow" << std::endl << std::endl
      << "number of samples:             N = " << number_of_samples << std::endl
      << "friction Reynolds number: Re_tau = " << Re_tau << std::endl
      << "wall shear stress:         tau_w = " << tau_w << std::endl << std::endl;

    f << "  y              u              v              w            "
      << "  rms(u')        rms(v')        rms(w')        u'v'         " << std::endl;
    // clang-format on

    for(unsigned int idx = 0; idx < y_glob.size(); idx++)
    {
      // clang-format off

      // y-values
      f << std::scientific << std::setprecision(7) << std::setw(15) << y_glob.at(idx);

      // mean velocity <u_i>, i=1,...,d
      f << std::setw(15) << vel_glob[0].at(idx) / (double)number_of_samples  /* <u_1> */
        << std::setw(15) << vel_glob[1].at(idx) / (double)number_of_samples  /* <u_2> */
        << std::setw(15) << vel_glob[2].at(idx) / (double)number_of_samples; /* <u_3> */

      // rms values: sqrt( <u_i'²> ) = sqrt( <u_i²> - <u_i>² ) where <u_i> = 0 for i=2,3
      double mean_u1 = vel_glob[0].at(idx) / (double)number_of_samples;
      f << std::setw(15) << std::sqrt(std::abs((velsq_glob[0].at(idx) / (double)(number_of_samples)-mean_u1 * mean_u1))) /* rms(u_1) */
        << std::setw(15) << sqrt(velsq_glob[1].at(idx) / (double)(number_of_samples))                                    /* rms(u_2) */
        << std::setw(15) << sqrt(velsq_glob[2].at(idx) / (double)(number_of_samples));                                   /* rms(u_3) */

      // <u'v'> = <u*v>
      f << std::setw(15) << (veluv_glob.at(idx)) / (double)(number_of_samples) << std::endl;

      // clang-format on
    }

    f.close();
  }
}

template<int dim, typename Number>
void
StatisticsManager<dim, Number>::reset()
{
  for(unsigned int i = 0; i < dim; i++)
    std::fill(vel_glob[i].begin(), vel_glob[i].end(), 0.);

  for(unsigned int i = 0; i < dim; i++)
    std::fill(velsq_glob[i].begin(), velsq_glob[i].end(), 0.);

  std::fill(veluv_glob.begin(), veluv_glob.end(), 0.);

  number_of_samples = 0;
}


/*
 *  This function calculates the following statistical quantities of the flow ...
 *
 *   - Mean velocity:  <u>
 *   - rms values of velocity: sqrt(<u'²>)
 *   - and Reynolds shear stress: <u'v'>
 *
 *  Averaging is performed by ...
 *
 *   - averaging over homogeneous directions (=averaging over x-z-planes)
 *   - and subsequently averaging the x-z-plane-averaged quantities over time samples
 *
 *  Therefore, we have to compute the following quantities: <u>, <u²>, <u*v>, since ...
 *
 *   - <u'²> = <(u-<u>)²> = <u² - 2*u<u> + <u>²> = <u²> - 2*<u>² + <u>² = <u²> - <u>²
 *   - <u'v'> = <(u-<u>)*(v-<v>)> = <u*v> - <u*<v>> - <<u>*v> + <u><v> = <u*v> - <u><v>
 *            = <u*v> since <v> = 0
 */
template<int dim, typename Number>
void
StatisticsManager<dim, Number>::do_evaluate(const std::vector<VectorType const *> & velocity)
{
  // Use local vectors xxx_loc in order to average/integrate over all
  // locally owned cells of current processor.
  std::vector<double> area_loc(vel_glob[0].size());

  std::vector<std::vector<double>> vel_loc(dim);
  for(unsigned int i = 0; i < dim; i++)
    vel_loc[i].resize(vel_glob[0].size());

  std::vector<std::vector<double>> velsq_loc(dim);
  for(unsigned int i = 0; i < dim; i++)
    velsq_loc[i].resize(vel_glob[0].size());

  std::vector<double> veluv_loc(vel_glob[0].size());

  // use 2d quadrature to integrate over x-z-planes
  unsigned int const      fe_degree = dof_handler.get_fe().degree;
  dealii::QGauss<dim - 1> gauss_2d(fe_degree + 1);

  // vector of dealii::FEValues for all x-z-planes of a cell
  std::vector<std::shared_ptr<dealii::FEValues<dim, dim>>> fe_values(n_points_y_per_cell);

  // TODO
  //  dealii::MappingQGeneric<dim> mapping(fe_degree);
  for(unsigned int i = 0; i < n_points_y_per_cell; ++i)
  {
    std::vector<dealii::Point<dim>> points(gauss_2d.size());
    std::vector<double>             weights(gauss_2d.size());
    for(unsigned int j = 0; j < gauss_2d.size(); ++j)
    {
      points[j][0] = gauss_2d.point(j)[0];
      if(dim == 3)
        points[j][2] = gauss_2d.point(j)[1];
      points[j][1] = (double)i / (n_points_y_per_cell - 1);
      weights[j]   = gauss_2d.weight(j);
    }

    fe_values[i].reset(new dealii::FEValues<dim>(mapping,
                                                 dof_handler.get_fe().base_element(0),
                                                 dealii::Quadrature<dim>(points, weights),
                                                 dealii::update_values | dealii::update_jacobians |
                                                   dealii::update_quadrature_points));
  }

  unsigned int const scalar_dofs_per_cell = dof_handler.get_fe().base_element(0).dofs_per_cell;
  // TODO this variable is not used
  //  std::vector<double> vel_values(fe_values[0]->n_quadrature_points);
  std::vector<dealii::Tensor<1, dim>>          velocity_vector(scalar_dofs_per_cell);
  std::vector<dealii::types::global_dof_index> dof_indices(dof_handler.get_fe().dofs_per_cell);

  // loop over all cells and perform averaging/integration for all locally owned cells
  for(typename dealii::DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
      cell != dof_handler.end();
      ++cell)
  {
    if(cell->is_locally_owned())
    {
      cell->get_dof_indices(dof_indices);

      // vector-valued FE where all components are explicitly listed in the dealii::DoFHandler
      if(dof_handler.get_fe().element_multiplicity(0) >= dim)
      {
        for(unsigned int j = 0; j < dof_indices.size(); ++j)
        {
          const std::pair<unsigned int, unsigned int> comp =
            dof_handler.get_fe().system_to_component_index(j);
          if(comp.first < dim)
            velocity_vector[comp.second][comp.first] = (*velocity[0])(dof_indices[j]);
        }
      }
      else // scalar FE where we have several vectors referring to the same dealii::DoFHandler
      {
        AssertDimension(dof_handler.get_fe().element_multiplicity(0), 1);
        for(unsigned int j = 0; j < scalar_dofs_per_cell; ++j)
          for(unsigned int d = 0; d < dim; ++d)
            velocity_vector[j][d] = (*velocity[d])(dof_indices[j]);
      }

      // loop over all x-z-planes of current cell
      for(unsigned int i = 0; i < n_points_y_per_cell; ++i)
      {
        fe_values[i]->reinit(typename dealii::Triangulation<dim>::active_cell_iterator(cell));

        std::vector<double> vel(dim, 0.);
        std::vector<double> velsq(dim, 0.);
        double              area = 0, veluv = 0;

        // perform integral over current x-z-plane of current cell
        for(unsigned int q = 0; q < fe_values[i]->n_quadrature_points; ++q)
        {
          // interpolate velocity to the quadrature point
          dealii::Tensor<1, dim> velocity;
          for(unsigned int j = 0; j < velocity_vector.size(); ++j)
            velocity += fe_values[i]->shape_value(j, q) * velocity_vector[j];

          double det = 0.;
          if(dim == 3)
          {
            dealii::Tensor<2, 2> reduced_jacobian;
            reduced_jacobian[0][0] = fe_values[i]->jacobian(q)[0][0];
            reduced_jacobian[0][1] = fe_values[i]->jacobian(q)[0][2];
            reduced_jacobian[1][0] = fe_values[i]->jacobian(q)[2][0];
            reduced_jacobian[1][1] = fe_values[i]->jacobian(q)[2][2];
            det                    = determinant(reduced_jacobian);
          }
          else
          {
            det = std::abs(fe_values[i]->jacobian(q)[0][0]);
          }

          double area_ele = det * fe_values[i]->get_quadrature().weight(q);
          area += area_ele;

          for(unsigned int i = 0; i < dim; i++)
            vel[i] += velocity[i] * area_ele;

          for(unsigned int i = 0; i < dim; i++)
            velsq[i] += velocity[i] * velocity[i] * area_ele;

          veluv += velocity[0] * velocity[1] * area_ele;
        }

        // Tranform cell index 'i' to global index 'idx' of y_glob-vector

        // find index within the y-values: first do a binary search to find
        // the next larger value of y in the list...
        double const y = fe_values[i]->quadrature_point(0)[1];
        // std::lower_bound: returns iterator to first element that is >= y.
        // Note that the vector y_glob has to be sorted. As a result, the
        // index might be too large.
        unsigned int idx =
          std::distance(y_glob.begin(), std::lower_bound(y_glob.begin(), y_glob.end(), y));

        // make sure that the index does not exceed the array bounds in case of round-off errors
        if(idx == y_glob.size())
          idx--;

        // reduce index by 1 in case that the previous point is closer to y than
        // the next point
        if(idx > 0 && std::abs(y_glob[idx - 1] - y) < std::abs(y_glob[idx] - y))
          idx--;

        AssertThrow(std::abs(y_glob[idx] - y) < 1e-13,
                    dealii::ExcMessage("Could not locate " + std::to_string(y) +
                                       " among pre-evaluated points. Closest point is " +
                                       std::to_string(y_glob[idx]) + " at distance " +
                                       std::to_string(std::abs(y_glob[idx] - y)) +
                                       ". Check transform() function given to constructor."));

        // Add results of cellwise integral to xxx_loc vectors since we want
        // to average/integrate over all locally owned cells.
        for(unsigned int i = 0; i < dim; i++)
          vel_loc[i].at(idx) += vel[i];

        for(unsigned int i = 0; i < dim; i++)
          velsq_loc[i].at(idx) += velsq[i];

        veluv_loc.at(idx) += veluv;
        area_loc.at(idx) += area;
      }
    }
  }

  // accumulate data over all processors overwriting
  // the processor-local data in xxx_loc since we want
  // to average/integrate over the global x-z-plane.
  for(unsigned int i = 0; i < dim; i++)
    dealii::Utilities::MPI::sum(vel_loc[i], mpi_comm, vel_loc[i]);

  for(unsigned int i = 0; i < dim; i++)
    dealii::Utilities::MPI::sum(velsq_loc[i], mpi_comm, velsq_loc[i]);

  dealii::Utilities::MPI::sum(veluv_loc, mpi_comm, veluv_loc);
  dealii::Utilities::MPI::sum(area_loc, mpi_comm, area_loc);

  // Add values averaged over global x-z-planes
  // (=MPI::sum(xxx_loc)/MPI::sum(area_loc)) to xxx_glob vectors.
  // Averaging over time-samples is performed when writing the output.
  for(unsigned int idx = 0; idx < y_glob.size(); idx++)
  {
    for(unsigned int i = 0; i < dim; i++)
      vel_glob[i].at(idx) += vel_loc[i][idx] / area_loc[idx];

    for(unsigned int i = 0; i < dim; i++)
      velsq_glob[i].at(idx) += velsq_loc[i][idx] / area_loc[idx];

    veluv_glob.at(idx) += veluv_loc[idx] / area_loc[idx];
  }

  // increment number of samples
  number_of_samples++;
}


template class StatisticsManager<2, float>;
template class StatisticsManager<3, float>;

template class StatisticsManager<2, double>;
template class StatisticsManager<3, double>;

} // namespace ExaDG
