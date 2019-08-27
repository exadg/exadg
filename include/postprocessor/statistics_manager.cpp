
#include "statistics_manager.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/tria_base.h>
#include <deal.II/fe/fe_values.h>

template<int dim>
StatisticsManager<dim>::StatisticsManager(const DoFHandler<dim> & dof_handler_velocity,
                                          const Mapping<dim> &    mapping_in)
  : n_points_y_per_cell(0),
    dof_handler(dof_handler_velocity),
    mapping(mapping_in),
    communicator(dynamic_cast<const parallel::TriangulationBase<dim> *>(
                   &dof_handler_velocity.get_triangulation()) ?
                   (dynamic_cast<const parallel::TriangulationBase<dim> *>(
                      &dof_handler_velocity.get_triangulation())
                      ->get_communicator()) :
                   MPI_COMM_SELF),
    number_of_samples(0),
    write_final_output(true),
    turb_channel_data(TurbulentChannelData())
{
}


template<int dim>
void
StatisticsManager<dim>::setup(const std::function<double(double const &)> & grid_transform,
                              TurbulentChannelData const &                  turb_channel_data_in)
{
  turb_channel_data = turb_channel_data_in;

  if(turb_channel_data.calculate_statistics == true)
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

    typename Triangulation<dim>::cell_iterator cell = dof_handler.get_triangulation().begin(0);
    while(cell != dof_handler.get_triangulation().end(0) && !cell->at_boundary(2))
    {
      ++cell;
    }
    while(!cell->at_boundary(3))
    {
      ++n_cells_y_dir;
      cell = cell->neighbor(3);
    }

    const unsigned int fe_degree = dof_handler.get_fe().degree;

    n_points_y_per_cell = n_points_y_per_cell_linear * fe_degree;

    AssertThrow(n_points_y_per_cell >= 2,
                ExcMessage("Number of points in y-direction per cell is invalid."));

    n_cells_y_dir *= std::pow(2, dof_handler.get_triangulation().n_global_levels() - 1);

    const unsigned int n_points_y_glob = n_cells_y_dir * (n_points_y_per_cell - 1) + 1;

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

    // calculate y-coordinates in physical space where we want to peform the sampling (averaging)
    y_glob.reserve(n_points_y_glob);

    // loop over all cells in y-direction
    if(turb_channel_data.cells_are_stretched == true)
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

      //    std::cout<<std::endl<<"Intermediate vector with y-coordinates:"<<std::endl;
      //    for(unsigned int i=0; i<y_glob.size();++i)
      //      std::cout<<"y_glob["<<i<<"]="<<y_glob[i]<<std::endl;
      //    std::vector<double> y_temp;
      //    y_temp = y_glob;

      // y_glob contains y-coordinates using the exact mapping

      // However, when calculating the statistics we use the polynomial mapping of degree
      // 'fe_degree' which leads to slightly different values as compared to the exact mapping.
      // -> overwrite values in y_glob with values resulting from polynomial mapping

      // use 2d quadrature to integrate over x-z-planes
      const unsigned int fe_degree = dof_handler.get_fe().degree;
      QGauss<dim - 1>    gauss_2d(fe_degree + 1);

      std::vector<double> y_processor;
      y_processor.resize(n_points_y_glob, std::numeric_limits<double>::lowest());

      // vector of FEValues for all x-z-planes of a cell
      std::vector<std::shared_ptr<FEValues<dim, dim>>> fe_values(n_points_y_per_cell);

      for(unsigned int i = 0; i < n_points_y_per_cell; ++i)
      {
        std::vector<Point<dim>> points(gauss_2d.size());
        std::vector<double>     weights(gauss_2d.size());
        for(unsigned int j = 0; j < gauss_2d.size(); ++j)
        {
          points[j][0] = gauss_2d.point(j)[0];
          if(dim == 3)
            points[j][2] = gauss_2d.point(j)[1];
          points[j][1] = (double)i / (n_points_y_per_cell - 1);
          weights[j]   = gauss_2d.weight(j);
        }
        fe_values[i].reset(
          new FEValues<dim>(mapping,
                            dof_handler.get_fe().base_element(0),
                            Quadrature<dim>(points, weights),
                            update_values | update_jacobians | update_quadrature_points));
      }

      // loop over all cells
      for(typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
          cell != dof_handler.end();
          ++cell)
      {
        if(cell->is_locally_owned())
        {
          // loop over all y-coordinates of current cell
          for(unsigned int i = 0; i < n_points_y_per_cell; ++i)
          {
            fe_values[i]->reinit(typename Triangulation<dim>::active_cell_iterator(cell));

            // Transform cell index 'i' to global index 'idx' of y_glob-vector

            // find index within the y-values: first do a binary search to find
            // the next larger value of y in the list...
            const double y = fe_values[i]->quadrature_point(0)[1];
            // std::lower_bound: returns iterator to first element that is >= y
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

            y_processor[idx] = y;
          }
        }
      }

      Utilities::MPI::max(y_processor, communicator, y_glob);

      //    // print final vector
      //    std::cout<<std::endl<<"Final vector with y-coordinates:"<<std::endl;
      //    for(unsigned int i=0; i<y_glob.size();++i)
      //      std::cout<<"y_glob["<<i<<"]="<<y_glob[i]<<std::endl;
      //
      //    // compare intermediate and final vector
      //    for(unsigned int i=0; i<y_glob.size();++i)
      //      std::cout<<"y_temp["<<i<<"]="<<y_temp[i]<<"
      //      "<<"y_glob["<<i<<"]="<<y_glob[i]<<std::endl;
    }
    else
    {
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

    AssertThrow(y_glob.size() == n_points_y_glob, ExcInternalError());
  }
}

template<int dim>
void
StatisticsManager<dim>::evaluate(const VectorType &   velocity,
                                 double const &       time,
                                 unsigned int const & time_step_number)
{
  if(turb_channel_data.calculate_statistics == true)
  {
    // EPSILON: small number which is much smaller than the time step size
    const double EPSILON = 1.0e-10;
    if((time > turb_channel_data.sample_start_time - EPSILON) &&
       (time < turb_channel_data.sample_end_time + EPSILON) &&
       (time_step_number % turb_channel_data.sample_every_timesteps == 0))
    {
      // evaluate statistics
      this->evaluate(velocity);

      // write intermediate output
      if(time_step_number % (turb_channel_data.sample_every_timesteps * 100) == 0)
      {
        this->write_output(turb_channel_data.filename_prefix,
                           turb_channel_data.viscosity,
                           turb_channel_data.density);
      }
    }

    // write final output
    if((time > turb_channel_data.sample_end_time - EPSILON) && write_final_output)
    {
      this->write_output(turb_channel_data.filename_prefix,
                         turb_channel_data.viscosity,
                         turb_channel_data.density);
      write_final_output = false;
    }
  }
}


template<int dim>
void
StatisticsManager<dim>::evaluate(const VectorType & velocity)
{
  std::vector<const VectorType *> vecs;
  vecs.push_back(&velocity);
  do_evaluate(vecs);
}



template<int dim>
void
StatisticsManager<dim>::evaluate(const std::vector<VectorType> & velocity)
{
  std::vector<const VectorType *> vecs;
  for(unsigned int i = 0; i < velocity.size(); ++i)
    vecs.push_back(&velocity[i]);
  do_evaluate(vecs);
}

template<int dim>
void
StatisticsManager<dim>::write_output(const std::string output_prefix,
                                     const double      dynamic_viscosity,
                                     const double      density)
{
  if(Utilities::MPI::this_mpi_process(communicator) == 0)
  {
    // tau_w = mu * d<u>/dy = mu * (<u>(y2)-<u>(y1))/(y2-y1), where mu = rho * nu
    double tau_w = dynamic_viscosity * ((vel_glob[0].at(1) - vel_glob[0].at(0)) /
                                        (double)number_of_samples / (y_glob.at(1) - y_glob.at(0)));

    // Re_tau = u_tau * delta / nu = sqrt(tau_w/rho) * delta / (mu/rho), where delta = 1
    double Re_tau = sqrt(tau_w / density) / (dynamic_viscosity / density);

    std::ofstream f;
    f.open((output_prefix + ".flow_statistics").c_str(), std::ios::trunc);

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



template<int dim>
void
StatisticsManager<dim>::reset()
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
template<int dim>
void
StatisticsManager<dim>::do_evaluate(const std::vector<const VectorType *> & velocity)
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
  const unsigned int fe_degree = dof_handler.get_fe().degree;
  QGauss<dim - 1>    gauss_2d(fe_degree + 1);

  // vector of FEValues for all x-z-planes of a cell
  std::vector<std::shared_ptr<FEValues<dim, dim>>> fe_values(n_points_y_per_cell);

  // TODO
  //  MappingQGeneric<dim> mapping(fe_degree);
  for(unsigned int i = 0; i < n_points_y_per_cell; ++i)
  {
    std::vector<Point<dim>> points(gauss_2d.size());
    std::vector<double>     weights(gauss_2d.size());
    for(unsigned int j = 0; j < gauss_2d.size(); ++j)
    {
      points[j][0] = gauss_2d.point(j)[0];
      if(dim == 3)
        points[j][2] = gauss_2d.point(j)[1];
      points[j][1] = (double)i / (n_points_y_per_cell - 1);
      weights[j]   = gauss_2d.weight(j);
    }

    fe_values[i].reset(
      new FEValues<dim>(mapping,
                        dof_handler.get_fe().base_element(0),
                        Quadrature<dim>(points, weights),
                        update_values | update_jacobians | update_quadrature_points));
  }

  const unsigned int scalar_dofs_per_cell = dof_handler.get_fe().base_element(0).dofs_per_cell;
  // TODO this variable is not used
  //  std::vector<double> vel_values(fe_values[0]->n_quadrature_points);
  std::vector<Tensor<1, dim>>          velocity_vector(scalar_dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dof_handler.get_fe().dofs_per_cell);

  // loop over all cells and perform averaging/integration for all locally owned cells
  for(typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
      cell != dof_handler.end();
      ++cell)
  {
    if(cell->is_locally_owned())
    {
      cell->get_dof_indices(dof_indices);

      // vector-valued FE where all components are explicitly listed in the DoFHandler
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
      else // scalar FE where we have several vectors referring to the same DoFHandler
      {
        AssertDimension(dof_handler.get_fe().element_multiplicity(0), 1);
        for(unsigned int j = 0; j < scalar_dofs_per_cell; ++j)
          for(unsigned int d = 0; d < dim; ++d)
            velocity_vector[j][d] = (*velocity[d])(dof_indices[j]);
      }

      // loop over all x-z-planes of current cell
      for(unsigned int i = 0; i < n_points_y_per_cell; ++i)
      {
        fe_values[i]->reinit(typename Triangulation<dim>::active_cell_iterator(cell));

        std::vector<double> vel(dim, 0.);
        std::vector<double> velsq(dim, 0.);
        double              area = 0, veluv = 0;

        // perform integral over current x-z-plane of current cell
        for(unsigned int q = 0; q < fe_values[i]->n_quadrature_points; ++q)
        {
          // interpolate velocity to the quadrature point
          Tensor<1, dim> velocity;
          for(unsigned int j = 0; j < velocity_vector.size(); ++j)
            velocity += fe_values[i]->shape_value(j, q) * velocity_vector[j];

          double det = 0.;
          if(dim == 3)
          {
            Tensor<2, 2> reduced_jacobian;
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
        const double y = fe_values[i]->quadrature_point(0)[1];
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
                    ExcMessage("Could not locate " + std::to_string(y) +
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
    Utilities::MPI::sum(vel_loc[i], communicator, vel_loc[i]);

  for(unsigned int i = 0; i < dim; i++)
    Utilities::MPI::sum(velsq_loc[i], communicator, velsq_loc[i]);

  Utilities::MPI::sum(veluv_loc, communicator, veluv_loc);
  Utilities::MPI::sum(area_loc, communicator, area_loc);

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


template class StatisticsManager<2>;
template class StatisticsManager<3>;
