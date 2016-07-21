
#include <deal.II/fe/fe_values.h>
#include <deal.II/distributed/tria_base.h>
#include <statistics_manager.h>
#include <SpaldingsLaw.h>

template <int dim>
StatisticsManager<dim>::StatisticsManager(const DoFHandler<dim> &dof_handler_velocity)
  :
  dof_handler (dof_handler_velocity),
  communicator (dynamic_cast<const parallel::Triangulation<dim>*>(&dof_handler_velocity.get_triangulation()) ?
                (dynamic_cast<const parallel::Triangulation<dim>*>(&dof_handler_velocity.get_triangulation())
                 ->get_communicator()) :
                MPI_COMM_SELF)
{
}


template <int dim>
void StatisticsManager<dim>::setup(const std_cxx11::function<Point<dim>(const Point<dim> &)> &grid_transform)
{
  AssertThrow(dim == 3, ExcNotImplemented()); //TODO: implement for 2D problems

  // note: this code only works on structured meshes where the faces in
  // y-direction are faces 2 and 3

  // find the number of refinements in the mesh, first the number of coarse
  // cells in y-direction and then the number of refinements.
  unsigned int n_cells_y_dir = 1;
  typename Triangulation<dim>::cell_iterator cell = dof_handler.get_triangulation().begin(0);
  while (cell != dof_handler.get_triangulation().end(0) && !cell->at_boundary(2))
    ++cell;
  while (!cell->at_boundary(3))
    {
      ++n_cells_y_dir;
      cell = cell->neighbor(3);
    }

  n_cells_y_dir *= std::pow(2, dof_handler.get_triangulation().n_global_levels()-1);

  const unsigned int n_points_y_glob =  n_cells_y_dir*(n_points_y-1)+1;

  velx_glob.resize(n_points_y_glob);
  vely_glob.resize(n_points_y_glob);
  velz_glob.resize(n_points_y_glob);
  velxsq_glob.resize(n_points_y_glob);
  velysq_glob.resize(n_points_y_glob);
  velzsq_glob.resize(n_points_y_glob);
  veluv_glob.resize(n_points_y_glob);
  numchsamp = 0;

  y_glob.reserve(n_points_y_glob);
  for (unsigned int ele = 0; ele < n_cells_y_dir;ele++)
    {
      double elelower = 1./(double)n_cells_y_dir*(double)ele;
      double eleupper = 1./(double)n_cells_y_dir*(double)(ele+1);
      Point<dim> pointlower;
      pointlower[1]=elelower;
      Point<dim> pointupper;
      pointupper[1]=eleupper;
      double ylower = grid_transform(pointlower)[1];
      double yupper = grid_transform(pointupper)[1];
      for (unsigned int plane = 0; plane<n_points_y-1;plane++)
        {
          double coord = ylower + (yupper-ylower)/(n_points_y-1)*plane;
          y_glob.push_back(coord);
        }
    }
  //push back last missing coordinate at upper wall
  Point<dim> upper;
  upper[1] = 1.;
  y_glob.push_back(grid_transform(upper)[1]);
  AssertThrow(y_glob.size() == n_points_y_glob, ExcInternalError());
}



template <int dim>
void
StatisticsManager<dim>::evaluate(const parallel::distributed::Vector<double> &velocity)
{
  std::vector<const parallel::distributed::Vector<double> *> vecs;
  vecs.push_back(&velocity);
  do_evaluate(vecs);
}



template <int dim>
void
StatisticsManager<dim>::evaluate(const std::vector<parallel::distributed::Vector<double> > &velocity)
{
  std::vector<const parallel::distributed::Vector<double> *> vecs;
  for (unsigned int i=0; i<velocity.size(); ++i)
    vecs.push_back(&velocity[i]);
  do_evaluate(vecs);
}

template <int dim>
void
StatisticsManager<dim>::write_output(const std::string output_prefix,
                                     const double      viscosity)
{
  if(Utilities::MPI::this_mpi_process(communicator)==0)
  {
    std::ofstream f;

    f.open((output_prefix + ".flow_statistics").c_str(),std::ios::trunc);
    f<<"statistics of turbulent channel flow  "<<std::endl;
    f<<"number of samples:   " << numchsamp << std::endl;
    f<<"friction Reynolds number:   " << sqrt(viscosity*((velx_glob.at(1)-velx_glob.at(0))/((double)numchsamp)/(y_glob.at(1)+1.)))/viscosity << std::endl;
    f<<"wall shear stress:   " << viscosity*((velx_glob.at(1)-velx_glob.at(0))/numchsamp/(y_glob.at(1)+1.)) << std::endl;

    f<< "       y       |       u      |       v      |       w      |   rms(u')    |   rms(v')    |   rms(w')    |     u'v'     " << std::endl;
    for (unsigned int idx = 0; idx<y_glob.size(); idx++)
    {
      f<<std::scientific<<std::setprecision(7) << std::setw(15) << y_glob.at(idx);
      f<<std::scientific<<std::setprecision(7) << std::setw(15) << velx_glob.at(idx)/(double)numchsamp;
      f<<std::scientific<<std::setprecision(7) << std::setw(15) << vely_glob.at(idx)/(double)numchsamp;
      f<<std::scientific<<std::setprecision(7) << std::setw(15) << velz_glob.at(idx)/(double)numchsamp;
      f<<std::scientific<<std::setprecision(7) << std::setw(15) << std::sqrt(std::abs((velxsq_glob.at(idx)/(double)(numchsamp)
             -velx_glob.at(idx)*velx_glob.at(idx)/(((double)numchsamp)*((double)numchsamp)))));
      f<<std::scientific<<std::setprecision(7) << std::setw(15) << sqrt(velysq_glob.at(idx)/(double)(numchsamp));
      f<<std::scientific<<std::setprecision(7) << std::setw(15) << sqrt(velzsq_glob.at(idx)/(double)(numchsamp));
      f<<std::scientific<<std::setprecision(7) << std::setw(15) << (veluv_glob.at(idx))/(double)(numchsamp);
      f << std::endl;
    }
    f.close();
  }
}



template <int dim>
void
StatisticsManager<dim>::reset()
{
  std::fill(velx_glob.begin(), velx_glob.end(), 0.);
  std::fill(vely_glob.begin(), vely_glob.end(), 0.);
  std::fill(velz_glob.begin(), velz_glob.end(), 0.);
  std::fill(velxsq_glob.begin(), velxsq_glob.end(), 0.);
  std::fill(velysq_glob.begin(), velysq_glob.end(), 0.);
  std::fill(velzsq_glob.begin(), velzsq_glob.end(), 0.);
  std::fill(veluv_glob.begin(), veluv_glob.end(), 0.);
  numchsamp = 0;
}



template <int dim>
void
StatisticsManager<dim>::do_evaluate(const std::vector<const parallel::distributed::Vector<double> *> &velocity)
{
  std::vector<double> area_loc(velx_glob.size());
  std::vector<double> velx_loc(velx_glob.size());
  std::vector<double> vely_loc(velx_glob.size());
  std::vector<double> velz_loc(velx_glob.size());
  std::vector<double> velxsq_loc(velx_glob.size());
  std::vector<double> velysq_loc(velx_glob.size());
  std::vector<double> velzsq_loc(velx_glob.size());
  std::vector<double> veluv_loc(velx_glob.size());

  const unsigned int fe_degree = dof_handler.get_fe().degree;
  std::vector<std_cxx11::shared_ptr<FEValues<dim,dim> > > fe_values(n_points_y);
  QGauss<dim-1> gauss_2d(fe_degree+1);

  for (unsigned int i=0; i<n_points_y; ++i)
  {
    std::vector<Point<dim> > points(gauss_2d.size());
    std::vector<double> weights(gauss_2d.size());
    for (unsigned int j=0; j<gauss_2d.size(); ++j)
    {
      points[j][0] = gauss_2d.point(j)[0];
      points[j][2] = gauss_2d.point(j)[1];
      points[j][1] = (double)i/(n_points_y-1);
      weights[j] = gauss_2d.weight(j);
    }
    fe_values[i].reset(new FEValues<dim>(dof_handler.get_fe().base_element(0),
                                         Quadrature<dim>(points, weights),
                                         update_values | update_jacobians |
                                         update_quadrature_points));
  }

  const unsigned int scalar_dofs_per_cell = dof_handler.get_fe().base_element(0).dofs_per_cell;
  std::vector<double> vel_values(fe_values[0]->n_quadrature_points);
  std::vector<Tensor<1,dim> > velocity_vector(scalar_dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dof_handler.get_fe().dofs_per_cell);

  for (typename DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active(); cell!=dof_handler.end(); ++cell)
    if (cell->is_locally_owned())
    {
      cell->get_dof_indices(dof_indices);
      // vector-valued FE where all components are explicitly listed in the
      // DoFHandler
      if (dof_handler.get_fe().element_multiplicity(0) >= dim)
        for (unsigned int j=0; j<dof_indices.size(); ++j)
        {
          const std::pair<unsigned int,unsigned int> comp =
            dof_handler.get_fe().system_to_component_index(j);
          if (comp.first < dim)
            velocity_vector[comp.second][comp.first] = (*velocity[0])(dof_indices[j]);
        }
      else
      // scalar FE where we have several vectors referring to the same
      // DoFHandler
      {
        AssertDimension(dof_handler.get_fe().element_multiplicity(0), 1);
        for (unsigned int j=0; j<scalar_dofs_per_cell; ++j)
          for (unsigned int d=0; d<dim; ++d)
            velocity_vector[j][d] = (*velocity[d])(dof_indices[j]);
      }

      for (unsigned int i=0; i<n_points_y; ++i)
      {
        fe_values[i]->reinit(typename Triangulation<dim>::active_cell_iterator(cell));

        double area = 0, velx = 0, vely = 0, velz = 0,
          velxsq = 0, velysq = 0, velzsq = 0, veluv = 0;

        for (unsigned int q=0; q<fe_values[i]->n_quadrature_points; ++q)
        {
          // interpolate velocity to the quadrature point
          Tensor<1,dim> velocity;
          for (unsigned int j=0; j<velocity_vector.size(); ++j)
            velocity += fe_values[i]->shape_value(j,q) * velocity_vector[j];

          Tensor<2,2> reduced_jacobian;
          reduced_jacobian[0][0] = fe_values[i]->jacobian(q)[0][0];
          reduced_jacobian[0][1] = fe_values[i]->jacobian(q)[0][dim-1];
          reduced_jacobian[1][0] = fe_values[i]->jacobian(q)[dim-1][0];
          reduced_jacobian[1][1] = fe_values[i]->jacobian(q)[dim-1][dim-1];
          double area_ele = determinant(reduced_jacobian) * fe_values[i]->get_quadrature().weight(q);
          area += area_ele;
          velx += velocity[0] * area_ele;
          vely += velocity[1] * area_ele;
          velz += velocity[dim-1] * area_ele;
          velxsq += velocity[0] * velocity[0] * area_ele;
          velysq += velocity[1] * velocity[1] * area_ele;
          velzsq += velocity[dim-1] * velocity[dim-1] * area_ele;
          veluv += velocity[0] * velocity[1] * area_ele;
        }

        // find index within the y-values: first do a binary search to find
        // the next larger value of y in the list...
        const double y = fe_values[i]->quadrature_point(0)[1];
        unsigned int idx = std::distance(y_glob.begin(),
                                         std::lower_bound(y_glob.begin(), y_glob.end(),
                                                          y));
        // ..., then, check whether the point before was closer (but off
        // by 1e-13 or less)
        if (idx > 0 && std::abs(y_glob[idx-1]-y) < std::abs(y_glob[idx]-y))
          idx--;
        AssertThrow(std::abs(y_glob[idx]-y)<1e-13,
                    ExcMessage("Could not locate " + std::to_string(y) + " among "
                               "pre-evaluated points. Closest point is " +
                               std::to_string(y_glob[idx]) + " at distance " +
                               std::to_string(std::abs(y_glob[idx]-y)) +
                               ". Check transform() function given to constructor."));
        velx_loc.at(idx) += velx;
        vely_loc.at(idx) += vely;
        velz_loc.at(idx) += velz;
        velxsq_loc.at(idx) += velxsq;
        velysq_loc.at(idx) += velysq;
        velzsq_loc.at(idx) += velzsq;
        veluv_loc.at(idx) += veluv;
        area_loc.at(idx) += area;
      }
    }
  // accumulate data over all processors overwriting the processor-local data
  // in xxx_loc
  Utilities::MPI::sum(velx_loc, communicator, velx_loc);
  Utilities::MPI::sum(vely_loc, communicator, vely_loc);
  Utilities::MPI::sum(velz_loc, communicator, velz_loc);
  Utilities::MPI::sum(velxsq_loc, communicator, velxsq_loc);
  Utilities::MPI::sum(velysq_loc, communicator, velysq_loc);
  Utilities::MPI::sum(velzsq_loc, communicator, velzsq_loc);
  Utilities::MPI::sum(veluv_loc, communicator, veluv_loc);
  Utilities::MPI::sum(area_loc, communicator, area_loc);

  for (unsigned int idx = 0; idx<y_glob.size(); idx++)
  {
    velx_glob.at(idx) += velx_loc[idx]/area_loc[idx];
    vely_glob.at(idx) += vely_loc[idx]/area_loc[idx];
    velz_glob.at(idx) += velz_loc[idx]/area_loc[idx];
    velxsq_glob.at(idx) += velxsq_loc[idx]/area_loc[idx];
    velysq_glob.at(idx) += velysq_loc[idx]/area_loc[idx];
    velzsq_glob.at(idx) += velzsq_loc[idx]/area_loc[idx];
    veluv_glob.at(idx) += veluv_loc[idx]/area_loc[idx];
  }
  numchsamp++;
}

template <int dim>
void
StatisticsManager<dim>::evaluate_xwall(const parallel::distributed::Vector<double> &velocity,
                                       const DoFHandler<dim> &dof_handler_wdist,
                                       const FEParameters<dim> & fe_param,
                                       const double viscosity)
{
  std::vector<const parallel::distributed::Vector<double> *> vecs;
  vecs.push_back(&velocity);
  do_evaluate_xwall(vecs,dof_handler_wdist,fe_param,viscosity);
}

template <int dim>
void
StatisticsManager<dim>::do_evaluate_xwall(const std::vector<const parallel::distributed::Vector<double> *> &velocity,
                                          const DoFHandler<dim> &dof_handler_wdist,
                                          const FEParameters<dim> & fe_param,
                                          const double viscosity)
{
  std::vector<double> area_loc(velx_glob.size());
  std::vector<double> velx_loc(velx_glob.size());
  std::vector<double> vely_loc(velx_glob.size());
  std::vector<double> velz_loc(velx_glob.size());
  std::vector<double> velxsq_loc(velx_glob.size());
  std::vector<double> velysq_loc(velx_glob.size());
  std::vector<double> velzsq_loc(velx_glob.size());
  std::vector<double> veluv_loc(velx_glob.size());

  const unsigned int fe_degree = dof_handler.get_fe().degree;
  std::vector<std_cxx11::shared_ptr<FEValues<dim,dim> > > fe_values(n_points_y);
  std::vector<std_cxx11::shared_ptr<FEValues<dim,dim> > > fe_values_xwall(n_points_y);
  std::vector<std_cxx11::shared_ptr<FEValues<dim,dim> > > fe_values_tauw(n_points_y);
  QGauss<dim-1> gauss_2d(fe_degree+1);

  for (unsigned int i=0; i<n_points_y; ++i)
  {
    std::vector<Point<dim> > points(gauss_2d.size());
    std::vector<double> weights(gauss_2d.size());
    for (unsigned int j=0; j<gauss_2d.size(); ++j)
    {
      points[j][0] = gauss_2d.point(j)[0];
      points[j][2] = gauss_2d.point(j)[1];
      points[j][1] = (double)i/(n_points_y-1);
      weights[j] = gauss_2d.weight(j);
    }
    fe_values[i].reset(new FEValues<dim>(dof_handler.get_fe().base_element(0),
                                         Quadrature<dim>(points, weights),
                                         update_values | update_jacobians |
                                         update_quadrature_points));
    fe_values_xwall[i].reset(new FEValues<dim>(dof_handler.get_fe().base_element(1),
                                         Quadrature<dim>(points, weights),
                                         update_values | update_jacobians |
                                         update_quadrature_points));
    fe_values_tauw[i].reset(new FEValues<dim>(dof_handler_wdist.get_fe().base_element(0),
                                         Quadrature<dim>(points, weights),
                                         update_values | update_jacobians |
                                         update_quadrature_points));
  }

  const unsigned int scalar_dofs_per_cell = dof_handler.get_fe().base_element(0).dofs_per_cell;
  const unsigned int scalar_dofs_per_cell_xwall = dof_handler.get_fe().base_element(1).dofs_per_cell;
  const unsigned int scalar_dofs_per_cell_tauw = dof_handler_wdist.get_fe().base_element(0).dofs_per_cell;
  std::vector<double> vel_values(fe_values[0]->n_quadrature_points);
  std::vector<Tensor<1,dim> > velocity_vector(scalar_dofs_per_cell);
  std::vector<Tensor<1,dim> > velocity_vector_xwall(scalar_dofs_per_cell_xwall);
  std::vector<Tensor<1,1> > tauw_vector(scalar_dofs_per_cell_tauw);
  std::vector<Tensor<1,1> > wdist_vector(scalar_dofs_per_cell_tauw);
  std::vector<types::global_dof_index> dof_indices(dof_handler.get_fe().dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices_tauw(dof_handler_wdist.get_fe().dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator cell_tauw=dof_handler_wdist.begin_active();
  for (typename DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active(); cell!=dof_handler.end(); ++cell, ++cell_tauw)
    if (cell->is_locally_owned())
    {
      cell->get_dof_indices(dof_indices);

      cell_tauw->get_dof_indices(dof_indices_tauw);

      { //read dofs from vector
        for (unsigned int j=0; j<dof_indices.size(); ++j)
        {
          const std::pair<unsigned int,unsigned int> comp =
            dof_handler.get_fe().system_to_component_index(j);
          if (comp.first < dim)
            velocity_vector[comp.second][comp.first] = (*velocity[0])(dof_indices[j]);
          else
            velocity_vector_xwall[comp.second][comp.first-dim] = (*velocity[0])(dof_indices[j]);
        }
        for (unsigned int j=0; j<scalar_dofs_per_cell_tauw; ++j)
          wdist_vector[j][0] = (*fe_param.wdist)(dof_indices_tauw[j]);
        for (unsigned int j=0; j<scalar_dofs_per_cell_tauw; ++j)
          tauw_vector[j][0] = (*fe_param.tauw)(dof_indices_tauw[j]);
      }

      for (unsigned int i=0; i<n_points_y; ++i)
      {
        fe_values[i]->reinit(typename Triangulation<dim>::active_cell_iterator(cell));
        fe_values_xwall[i]->reinit(typename Triangulation<dim>::active_cell_iterator(cell));
        fe_values_tauw[i]->reinit(typename Triangulation<dim>::active_cell_iterator(cell));

        double area = 0, velx = 0, vely = 0, velz = 0,
          velxsq = 0, velysq = 0, velzsq = 0, veluv = 0;

        AlignedVector<double > wdist(fe_values[i]->n_quadrature_points,0.);
        AlignedVector<double > tauw(fe_values[i]->n_quadrature_points,0.);
        for (unsigned int q=0; q<fe_values[i]->n_quadrature_points; ++q)
        {
          for (unsigned int j=0; j<wdist_vector.size(); ++j)
            wdist[q] += (fe_values_tauw[i]->shape_value(j,q) * wdist_vector[j])[0];
          for (unsigned int j=0; j<tauw_vector.size(); ++j)
            tauw[q] += (fe_values_tauw[i]->shape_value(j,q) * tauw_vector[j])[0];
        }
        SpaldingsLawEvaluation<dim, double, double > spalding(viscosity);
        spalding.reinit(wdist,tauw,fe_values[i]->n_quadrature_points);

        for (unsigned int q=0; q<fe_values[i]->n_quadrature_points; ++q)
        {
          // interpolate velocity to the quadrature point
          Tensor<1,dim> velocity;
          for (unsigned int j=0; j<velocity_vector.size(); ++j)
            velocity += fe_values[i]->shape_value(j,q) * velocity_vector[j];

          for (unsigned int j=0; j<velocity_vector_xwall.size(); ++j)
            velocity += fe_values_xwall[i]->shape_value(j,q) * velocity_vector_xwall[j] * spalding.enrichment(q);

          Tensor<2,2> reduced_jacobian;
          reduced_jacobian[0][0] = fe_values[i]->jacobian(q)[0][0];
          reduced_jacobian[0][1] = fe_values[i]->jacobian(q)[0][dim-1];
          reduced_jacobian[1][0] = fe_values[i]->jacobian(q)[dim-1][0];
          reduced_jacobian[1][1] = fe_values[i]->jacobian(q)[dim-1][dim-1];
          double area_ele = determinant(reduced_jacobian) * fe_values[i]->get_quadrature().weight(q);
          area += area_ele;
          velx += velocity[0] * area_ele;
          vely += velocity[1] * area_ele;
          velz += velocity[dim-1] * area_ele;
          velxsq += velocity[0] * velocity[0] * area_ele;
          velysq += velocity[1] * velocity[1] * area_ele;
          velzsq += velocity[dim-1] * velocity[dim-1] * area_ele;
          veluv += velocity[0] * velocity[1] * area_ele;
        }

        // find index within the y-values: first do a binary search to find
        // the next larger value of y in the list...
        const double y = fe_values[i]->quadrature_point(0)[1];
        unsigned int idx = std::distance(y_glob.begin(),
                                         std::lower_bound(y_glob.begin(), y_glob.end(),
                                                          y));
        // ..., then, check whether the point before was closer (but off
        // by 1e-13 or less)
        if (idx > 0 && std::abs(y_glob[idx-1]-y) < std::abs(y_glob[idx]-y))
          idx--;
        AssertThrow(std::abs(y_glob[idx]-y)<1e-13,
                    ExcMessage("Could not locate " + std::to_string(y) + " among "
                               "pre-evaluated points. Closest point is " +
                               std::to_string(y_glob[idx]) + " at distance " +
                               std::to_string(std::abs(y_glob[idx]-y)) +
                               ". Check transform() function given to constructor."));
        velx_loc.at(idx) += velx;
        vely_loc.at(idx) += vely;
        velz_loc.at(idx) += velz;
        velxsq_loc.at(idx) += velxsq;
        velysq_loc.at(idx) += velysq;
        velzsq_loc.at(idx) += velzsq;
        veluv_loc.at(idx) += veluv;
        area_loc.at(idx) += area;
      }
    }
  // accumulate data over all processors overwriting the processor-local data
  // in xxx_loc
  Utilities::MPI::sum(velx_loc, communicator, velx_loc);
  Utilities::MPI::sum(vely_loc, communicator, vely_loc);
  Utilities::MPI::sum(velz_loc, communicator, velz_loc);
  Utilities::MPI::sum(velxsq_loc, communicator, velxsq_loc);
  Utilities::MPI::sum(velysq_loc, communicator, velysq_loc);
  Utilities::MPI::sum(velzsq_loc, communicator, velzsq_loc);
  Utilities::MPI::sum(veluv_loc, communicator, veluv_loc);
  Utilities::MPI::sum(area_loc, communicator, area_loc);

  for (unsigned int idx = 0; idx<y_glob.size(); idx++)
  {
    velx_glob.at(idx) += velx_loc[idx]/area_loc[idx];
    vely_glob.at(idx) += vely_loc[idx]/area_loc[idx];
    velz_glob.at(idx) += velz_loc[idx]/area_loc[idx];
    velxsq_glob.at(idx) += velxsq_loc[idx]/area_loc[idx];
    velysq_glob.at(idx) += velysq_loc[idx]/area_loc[idx];
    velzsq_glob.at(idx) += velzsq_loc[idx]/area_loc[idx];
    veluv_glob.at(idx) += veluv_loc[idx]/area_loc[idx];
  }
  numchsamp++;
}

// explicit instantiation
template class StatisticsManager<2>;
template class StatisticsManager<3>;
