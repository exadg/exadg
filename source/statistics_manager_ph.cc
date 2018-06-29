
#include "../include/incompressible_navier_stokes/postprocessor/statistics_manager_ph.h"

#include <deal.II/fe/fe_values.h>
#include <deal.II/distributed/tria_base.h>

#include "../include/incompressible_navier_stokes/xwall/SpaldingsLaw.h"


//#define DEBUG_Y
//#define DEBUG_WRITE_OUTPUT
//#define DEBUG_TAU_W
#define DOUBLE_REFINEMENT_X
//#define ADDITIONAL_ADJUSTMENT

using namespace IncNS;

namespace patch
{
  template < typename T > std::string to_string( const T& n )
  {
    std::ostringstream stm ;
    stm << n ;
    return stm.str() ;
  }
}

template <int dim>
//void StatisticsManagerPH<dim>::setup(const std::function<Point<dim>(const Point<dim> &)> &grid_transform)
void StatisticsManagerPH<dim>::setup(const Function< dim > &push_forward_function, const std::string output_prefix,const bool enriched)
{
  // note: this code only works on structured meshes where the faces in
  // y-direction are faces 2 and 3

  // note: this code only works with discretizations that approximate the geometry adequate. That means
  // you need a refinement level of minimum  3 and fe_degree of minimum  2 to ensure the statistics manager to work robust


  // ---------------------------------------------------
  // get y positions for given x_over_h positions
  // ---------------------------------------------------

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
  const unsigned int fe_degree = dof_handler.get_fe().degree;

  n_cells_y_dir *= std::pow(2, dof_handler.get_triangulation().n_global_levels()-1);

  n_points_y = fe_degree + 1;
  if(enriched)
    n_points_y = 50;
  n_points_y_glob =  n_cells_y_dir*(n_points_y-1)+1;
//  n_points_x = 3.0*n_points_y;

  // initialize data
  numchsamp = 0;
  x_over_h = {0.05, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},

  y_vec_glob.resize(x_over_h.size());
  y_h_glob.resize(x_over_h.size());
  vel_glob.resize(3);
  for(unsigned int i=0; i<3;i++)
    vel_glob[i].resize(x_over_h.size());
  velsq_glob.resize(3);
  for(unsigned int i=0; i<3;i++)
    velsq_glob[i].resize(x_over_h.size());
  veluv_glob.resize(x_over_h.size());
  epsii_glob.resize(3);
  for(unsigned int i=0; i<3;i++)
    epsii_glob[i].resize(x_over_h.size());
  for (unsigned int i=0; i<x_over_h.size(); i++)
  {
    for(unsigned int j=0; j<3;j++)
      vel_glob[j][i].resize(n_points_y_glob,0.);
    for(unsigned int j=0; j<3;j++)
      velsq_glob[j][i].resize(n_points_y_glob,0.);
    for(unsigned int j=0; j<3;j++)
      epsii_glob[j][i].resize(n_points_y_glob,0.);
    veluv_glob[i].resize(n_points_y_glob);
  }


//  std::vector<double> delta_y(x_over_h.size(),100.0); // just for AssertThrow

  // get an estimate for all y-values
  // that is an estimate, since the push_forward_function assigns to the optimal geometry,
  // but the geometry is approximated. Thus y-values are different
  for (unsigned int i=0; i<x_over_h.size(); i++)
  {
    y_vec_glob[i].reserve(n_points_y_glob);
    y_h_glob[i].reserve(n_points_y_glob);
    const double x_pos = x_over_h[i]*h;
    for (unsigned int ele = 0; ele < n_cells_y_dir;ele++)
    {
      double elelower = y_min + (y_max-y_min)/(double)n_cells_y_dir*(double)ele;
      double eleupper = y_min + (y_max-y_min)/(double)n_cells_y_dir*(double)(ele+1);
      Point<dim> pointlower;
      pointlower[0]=x_pos;
      pointlower[1]=elelower;
      Point<dim> pointupper;
      pointupper[0]=x_pos;
      pointupper[1]=eleupper;
      double ylower = push_forward_function.value(pointlower,1);
      double yupper = push_forward_function.value(pointupper,1);
      for (unsigned int z_line = 0; z_line<n_points_y-1;z_line++)
      {
        double coord = ylower + (yupper-ylower)/(n_points_y-1)*z_line;
        y_vec_glob[i].push_back(coord);
        if(y_h_glob[i].size()>0)
          y_h_glob[i].push_back((yupper-ylower + y_h_glob[i].back())*0.5);//take the average between the current and the previous element, results in better consistent behavior at element interfaces
        else
          y_h_glob[i].push_back(yupper-ylower);
      }
    }
    //push back last missing coordinate at upper wall
    Point<dim> upper;
    upper[1] = y_max;
    upper[0] = x_pos;
    y_vec_glob[i].push_back(push_forward_function.value(upper,1));
    AssertThrow(y_vec_glob[i].size() == n_points_y_glob, ExcInternalError());
    y_h_glob[i].push_back(y_h_glob[i].back());
  }
  //compute real volume for eta
  for (unsigned int i_x=0; i_x<x_over_h.size(); i_x++)
  {
    for (unsigned int j=0; j<y_h_glob[i_x].size(); j++)
      y_h_glob[i_x][j] = std::pow(y_h_glob[i_x][j]*(4.5*0.028/(double)n_cells_y_dir)*(0.5*9.*0.028/(double)n_cells_y_dir),1./3.)/(double)(fe_degree + 1);
  }

  // get the real y-values (y_loc) on each processor, respectively
  for (unsigned int i_x=0; i_x<x_over_h.size(); i_x++)
   {
#ifdef DEBUG_Y
    std::cout << "### current x/h position (i_x=" << i_x << "): " << x_over_h[i_x] << std::endl;
#endif
    std::vector<std::shared_ptr<FEValues<dim,dim> > > fe_values(n_points_y);
    QGauss<1> gauss_1d(fe_degree+1);
    std::vector<double> y_loc;
    y_loc.resize(n_points_y_glob,-1);

    double x_pos = x_over_h[i_x]*h;
    // get xi1 position to the considered x position x_pos
    for (typename DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active(); cell!=dof_handler.end(); ++cell)
      if (cell->is_locally_owned())
        {
        std::vector<double> vertex_x_pos(GeometryInfo<dim>::vertices_per_cell);
        std::vector<double> vertex_z_pos(GeometryInfo<dim>::vertices_per_cell);

        // get vertices of cell
        for (unsigned int v=0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          vertex_x_pos[v] = cell->vertex(v)[0];
          if(dim==3)
            vertex_z_pos[v] = cell->vertex(v)[2];
          else
            vertex_z_pos[v] = 0.;
        }

        // determine if the cell is relevant for getting a y-value
        std::vector<double>::iterator min_x = std::min_element(vertex_x_pos.begin(), vertex_x_pos.end());
        std::vector<double>::iterator max_x = std::max_element(vertex_x_pos.begin(), vertex_x_pos.end());
        std::vector<double>::iterator min_z = std::min_element(vertex_z_pos.begin(), vertex_z_pos.end());
        std::vector<double>::iterator max_z = std::max_element(vertex_z_pos.begin(), vertex_z_pos.end());

        const double EPSILON = 1e-12;
        if (*min_x <= x_pos && *max_x > x_pos && *min_z-EPSILON <= 0 && *max_z+EPSILON > 0 )
          {
#ifdef DEBUG_Y
          std::cout << "new cell" << std::endl;
#endif
          double xi1_pos = x_pos/(*max_x-*min_x) - *min_x/(*max_x-*min_x);

          std::vector<int> old_idx(n_points_y,y_vec_glob[i_x].size()+10);

          for (unsigned int i=0; i<n_points_y; ++i)
          {
            std::vector<Point<dim> > points(gauss_1d.size());
            std::vector<double> weights(gauss_1d.size());
            if(dim == 2)
            {
              points.resize(1);
              weights.resize(1);
            }
            for (unsigned int j=0; j<weights.size(); ++j)
            {
              points[j][0] = xi1_pos;
              points[j][1] = (double)i/(n_points_y-1); // this are the "real" y-values in parameter space
              if(dim==3)
              {
                points[j][2] = gauss_1d.point(j)[0];
                weights[j] = gauss_1d.weight(j);
              }
              else
                weights[j] = 1.;
            }
            fe_values[i].reset(new FEValues<dim>(mapping_,
                                                 dof_handler.get_fe().base_element(0),
                                                 Quadrature<dim>(points, weights),
                                                 update_values | update_jacobians |
                                                 update_quadrature_points));

            fe_values[i]->reinit(typename Triangulation<dim>::active_cell_iterator(cell));

            const double y = fe_values[i]->quadrature_point(0)[1]; // this are the "real" y-values in real space
#ifdef DEBUG_Y
            std::cout << "y = " << y << " " << std::endl;
#endif

            // find index within the y-values: first do a binary search to find
            // the next larger value of y in the list...
            unsigned int idx = std::distance(y_vec_glob[i_x].begin(),
                                             std::lower_bound(y_vec_glob[i_x].begin(), y_vec_glob[i_x].end(),
                                                              y));

            // ..., then, check whether the point before was closer (but off
            // by 1e-13 or less)
            if (idx > 0 && std::abs(y_vec_glob[i_x][idx-1]-y) < std::abs(y_vec_glob[i_x][idx]-y))
              idx--;

#ifdef ADDITIONAL_ADJUSTMENT
            if (i > 0)
            {
              // within an element idx is unique !!!
              if ( std::find(old_idx.begin(), old_idx.end(), idx) != old_idx.end() )
              {
  #ifdef DEBUG_Y
                std::cout << "something went wrong when sorting y in vector y_loc. The algorythm sees that index idx = " << idx << " is appropriate according to its distance to y, but this index has already been used !!! -> the code sort this value in next higher position of y_loc (idx = idx+1) " << std::endl;
  #endif
                // the y-value before was sorted wrong
                if (idx > 0 && y_loc.at(idx-1) < -0.9)
                {
                  y_loc.at(idx-1) = y_loc.at(idx);
                  old_idx.at(i) = idx; // idx-1 should be already in old_idx
                }
                else
                {
                  idx++;
                  old_idx.at(i) = idx;
                }
              }
              else
                old_idx.at(i) = idx;
            }
            else
              old_idx.at(i) = idx;
#endif
            y_loc.at(idx)=y;

#ifdef DEBUG_Y
            std::cout << ".. is sorted in y_loc.at(" << idx << ") " << std::endl;
#endif
          }
          }
        }

      std::vector<double> tmp(y_vec_glob[i_x]);
      Utilities::MPI::max(y_loc, communicator, y_vec_glob[i_x]); // after this step (*), all entries in y_vec_glob have been changed

//      std::vector<double> diff(tmp - y_vec_glob[i_x]);
//      geom_appr_error = std::max_element(diff.);
#ifdef DEBUG_Y
      std::cout << "y_loc = ";
      for (unsigned int i=0; i<n_points_y_glob; ++i)
        std::cout  << y_loc.at(i) << " ";
      std::cout << std::endl;
      if(Utilities::MPI::this_mpi_process(communicator)==0)
        for (unsigned int i=0; i<n_points_y_glob; ++i)
          std::cout << "new  " << y_vec_glob[i_x].at(i) <<"  old  " << tmp.at(i)<< std::endl;
#endif
      for (unsigned int i=0; i<n_points_y_glob; ++i)
      {
        AssertThrow(y_vec_glob[i_x].at(i)>-0.9,
            ExcMessage("The geometry approximation is too bad for the statistics_manager_PH to work. Use minimum l3_p2 for a GRID_STRETCH_FAC up to 2.5 (using the double amount of cells in x-direction). "
                "If GRID_STRETCH_FAC is higher a more accurate spatial discretization is necessary to make the statistics manager work")); // after step (*) ... . Thus, a "-1" idicates that something wents wrong
//        AssertThrow(std::abs(y_vec_glob[i_x].at(i) - tmp.at(i)) <= delta_y.at(i_x),
//            ExcMessage("WARNING (If you think its OK comment this assert out!): The geometry approximation is too bad for the statistics_manager_PH to work. Use minimum l3_p2 or l2_p4."
//                       "std::abs(y_vec_glob[i_x].at(i) - tmp.at(i)) = " + patch::to_string(std::abs(y_vec_glob[i_x].at(i) - tmp.at(i))) + " and delta_y.at(i) = " + patch::to_string(delta_y.at(i))));
      }

      AssertThrow(y_vec_glob[i_x].size() == n_points_y_glob, ExcInternalError());

   } // loop over x_pos


  // ---------------------------------------------------
  // get x positions where tau_w and y+ are evaluated.
  // additionally get y1 for the calculation of y+
  // ---------------------------------------------------

  // variable to determine the geometry approximation
  std::vector<double> y_hill_contour_nominal;


#ifdef DOUBLE_REFINEMENT_X
  // it is ASSUMED that the refinement level is applied to TWO INITIAL CELLS !!!!!!!
  unsigned int n_cells_x_dir = 2*std::pow(2, dof_handler.get_triangulation().n_global_levels()-1);
#else
  // it is ASSUMED that the refinement level is applied to ONE INITIAL CELL !!!!!!!
    unsigned int n_cells_x_dir = std::pow(2, dof_handler.get_triangulation().n_global_levels()-1);
#endif
    n_points_x_glob = n_cells_x_dir*(n_points_x-1)+1;

    // initialize data
    dudy_bottom_glob.resize(n_points_x_glob);
    p_bottom_glob.resize(n_points_x_glob);
    dudy_top_glob.resize(n_points_x_glob);
    p_top_glob.resize(n_points_x_glob);

    // get an estimate for all x-values
    x_glob.reserve(n_points_x_glob);
    y1_bottom_glob.reserve(n_points_x_glob);
    y1_top_glob.reserve(n_points_x_glob);

    y_hill_contour_nominal.reserve(n_points_x_glob);

#ifdef DEBUG_TAU_W
    y_glob_debug.reserve(n_points_x_glob);
#endif
    for (unsigned int i_x = 0; i_x < n_points_x_glob;i_x++)
    {
      double x = x_max/((double)n_points_x_glob - 1.0)*(double)i_x; // linear distributed before AND after transform with push forward (opposed to y-values in setup code before)
      x_glob.push_back(x);

      // points at the bottom
      Point<dim> lower_point;
      lower_point[0] = x;
      lower_point[1] = y_min;
      if(dim==3)
        lower_point[2] = 0.0;
      double y_lower = push_forward_function.value(lower_point,1);
      Point<dim> upper_point;
      upper_point[0] = x;
      upper_point[1] = y_min + (y_max - y_min)/n_cells_y_dir;
      if(dim==3)
        upper_point[2] = 0.0;
      double y_upper = push_forward_function.value(upper_point,1);
      double y1 = (y_upper - y_lower)/(fe_degree + 1);
      y1_bottom_glob.push_back(y1);
      y_hill_contour_nominal.push_back(y_lower);

      // points at the bottom
//      Point<dim> lower_point;
      lower_point[0] = x;
      lower_point[1] = y_max - (y_max - y_min)/n_cells_y_dir;
      if(dim==3)
        lower_point[2] = 0.0;
      y_lower = push_forward_function.value(lower_point,1);
//      Point<dim> upper_point;
      upper_point[0] = x;
      upper_point[1] = y_max;
      if(dim==3)
        upper_point[2] = 0.0;
      y_upper = push_forward_function.value(upper_point,1);
      y1 = (y_upper - y_lower)/(fe_degree + 1);
      y1_top_glob.push_back(y1);


#ifdef DEBUG_TAU_W
      y_glob_debug.push_back(y_lower);
#endif
    }

    AssertThrow(x_glob.size() == n_points_x_glob,
        ExcMessage("x_glob.size() = " + patch::to_string(x_glob.size()) + " and n_points_x_glob = " + patch::to_string(n_points_x_glob)));
#ifdef DEBUG_TAU_W
    AssertThrow(y_glob_debug.size() == n_points_x_glob,
            ExcMessage("y_glob_debug.size() = " + patch::to_string(y_glob_debug.size()) + " and n_points_x_glob = " + patch::to_string(n_points_x_glob)));
#endif


    // ---------------------------------------------------
    // get actual y positions for given x positions
    // in order to evaluate the geometry approximation error
    // ---------------------------------------------------

    std::vector<std::shared_ptr<FEValues<dim,dim> > > fe_values(n_points_x);

    for (unsigned int i=0; i<n_points_x; ++i)
     {
       std::vector<Point<dim> > gauss_points(1);
  //     Point<dim> gauss_point;
       gauss_points[0][0] = (double)i/(n_points_x-1);
       gauss_points[0][1] = 0.0;            // lower domain boundary
       if(dim==3)
         gauss_points[0][2] = 0.0;
  //     double weight_dummy = 0.0;
       std::vector<double> weights_dummy(1);
       weights_dummy.at(0) = 0.0;

       fe_values[i].reset(new FEValues<dim>(mapping_,
                                            dof_handler.get_fe().base_element(0),
                                            Quadrature<dim>(gauss_points, weights_dummy),
                                            update_quadrature_points));
     }

    std::vector<double> y_hill_contour_actual(y_hill_contour_nominal);
    std::vector<double> y_loc;
    y_loc.resize(n_points_x_glob,-1);

    for (typename DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active(); cell!=dof_handler.end(); ++cell)
    {
      if (cell->is_locally_owned())
      {
        if (cell->at_boundary(2)) // just evaluate cells at the bottom of the domain
        {
          for (unsigned int i=0; i<n_points_x; ++i)
          {
           fe_values[i]->reinit(typename Triangulation<dim>::active_cell_iterator(cell));

           const double x = fe_values[i]->quadrature_point(0)[0];
           const double y = fe_values[i]->quadrature_point(0)[1];

           // find index within the x-values: first do a binary search to find
           // the next larger value of x in the list...
           unsigned int idx = std::distance(x_glob.begin(),
                                            std::lower_bound(x_glob.begin(), x_glob.end(),
                                                             x));
           // ..., then, check whether the point before was closer (but off
           // by 1e-13 or less)
           if (idx > 0 && std::abs(x_glob[idx-1]-x) < std::abs(x_glob[idx]-x))
             idx--;
           AssertThrow(std::abs(x_glob[idx]-x)<1e-13,
                       ExcMessage("Could not locate " + patch::to_string(x) + " among "
                                  "pre-evaluated points. Closest point is " +
                                  patch::to_string(x_glob[idx]) + " at distance " +
                                  patch::to_string(std::abs(x_glob[idx]-x))));

           y_loc.at(idx) = y;
          }
        }
      }
    }

     Utilities::MPI::max(y_loc, communicator, y_hill_contour_actual); // after this step (*), all entries in y_vec_glob have been changed

      for (unsigned int i=0; i<n_points_y_glob; ++i)
      {
        AssertThrow(y_hill_contour_actual.at(i)>-0.9, ExcInternalError());
      }

      // ---------------------------------------------------
      // write geometry approximation error
      // ---------------------------------------------------
      if(Utilities::MPI::this_mpi_process(communicator)==0)
      {
        std::ofstream f;

        f.open((output_prefix + ".geometry_approximation_error").c_str(),std::ios::trunc);
        f<<"geometry_approximation_error of periodic hill flow "<<std::endl;
        f<<"refinement level:   " << dof_handler.get_triangulation().n_global_levels() - 1 << std::endl;
        f<<"fe degree:   " << fe_degree << std::endl;
    //    f<<"friction Reynolds number:   " << sqrt(viscosity*(velx_glob.at(1)/numchsamp/(x_glob.at(1)+1.)))/viscosity << std::endl;

        f<< "       x       |    y_actual   |   y_nominal   |   abs(error)  " << std::endl;
        for (unsigned int idx = 0; idx<x_glob.size(); idx++)
        {
          f<<std::scientific<<std::setprecision(7) << std::setw(15) << x_glob.at(idx);
          f<<std::scientific<<std::setprecision(7) << std::setw(15) << y_hill_contour_actual.at(idx);
          f<<std::scientific<<std::setprecision(7) << std::setw(15) << y_hill_contour_nominal.at(idx);
          f<<std::scientific<<std::setprecision(7) << std::setw(15) << std::abs(y_hill_contour_actual.at(idx) - y_hill_contour_nominal.at(idx));
          f << std::endl;
        }
        f.close();
      }

}



template <int dim>
void
StatisticsManagerPH<dim>::evaluate(const parallel::distributed::Vector<double> &velocity,const parallel::distributed::Vector<double> &pressure)
{
  std::vector<const parallel::distributed::Vector<double> *> vecs;
  vecs.push_back(&velocity);
  do_evaluate(vecs,pressure);
}



template <int dim>
void
StatisticsManagerPH<dim>::evaluate(const std::vector<parallel::distributed::Vector<double> > &velocity,const parallel::distributed::Vector<double> &pressure)
{
  std::vector<const parallel::distributed::Vector<double> *> vecs;
  for (unsigned int i=0; i<velocity.size(); ++i)
    vecs.push_back(&velocity[i]);
  do_evaluate(vecs,pressure);
}

template <int dim>
void
StatisticsManagerPH<dim>::evaluate(const parallel::distributed::BlockVector<double> &velocity,const parallel::distributed::Vector<double> &pressure)
{
  std::vector<const parallel::distributed::Vector<double> *> vecs;
  for (unsigned int i=0; i<velocity.n_blocks(); ++i)
    vecs.push_back(&(velocity.block(i)));
  do_evaluate(vecs,pressure);
}

template <int dim>
void
StatisticsManagerPH<dim>::evaluate_xwall(const parallel::distributed::Vector<double> &velocity,
                                         const parallel::distributed::Vector<double> &pressure,
                                         const DoFHandler<dim>                       &dof_handler_wdist,
                                         const FEParameters<dim>                     &fe_param)
{
  std::vector<const parallel::distributed::Vector<double> *> vecs;
  vecs.push_back(&velocity);
  do_evaluate_xwall(vecs,pressure,dof_handler_wdist,fe_param);
}

template <int dim>
void
StatisticsManagerPH<dim>::write_output(const std::string output_prefix,
                                       const double      viscosity,
                                       unsigned int statistics_number)
{
  if(Utilities::MPI::this_mpi_process(communicator)==0)
  {
    //specify values for normalization
    const double H = 0.028;
    const double ub = 5.621;
    const double ubsq = ub*ub;

    std::string new_output_prefix;
    std::ostringstream filename;
    filename.str("");
    filename.clear();
    filename << output_prefix
             << "_slice"
             << statistics_number;
    new_output_prefix = filename.str();

    // write velocitys at certain x_over_h -positions
    for (unsigned int i_x=0; i_x<x_over_h.size(); i_x++)
    {
      std::ofstream f;
      f.open((new_output_prefix + "_" + patch::to_string(i_x) + ".flow_statistics").c_str(),std::ios::trunc);
      f<<"statistics of periodic hill flow for x/h = " << x_over_h[i_x] <<std::endl;
      f<<"number of samples:   " << numchsamp << std::endl;
      f<<"viscosity:           " << std::scientific<<std::setprecision(9) << viscosity << std::endl;

      f<< "       y/H       |      u/U_b     |      v/U_b     |      w/U_b     |   u'u'/U_b^2   |   v'v'/U_b^2   |   w'w'/U_b^2   |   u'v'/U_b^2   |   TKE/U_b^2    |      eps       |     eta/H      |     h/eta      |       h      " << std::endl;
      for (unsigned int idx = 0; idx<y_vec_glob[i_x].size(); idx++)
      {
        double velx = vel_glob[0][i_x].at(idx)/(double)numchsamp;
        double vely = vel_glob[1][i_x].at(idx)/(double)numchsamp;
        double velz = vel_glob[2][i_x].at(idx)/(double)numchsamp;

        double veluu = velsq_glob[0][i_x].at(idx)/((double)(numchsamp))
        - velx*velx;
        double velvv = velsq_glob[1][i_x].at(idx)/((double)(numchsamp))
        - vely*vely;
        double velww = velsq_glob[2][i_x].at(idx)/((double)(numchsamp))
        - velz*velz;
        double veluv = veluv_glob[i_x].at(idx)/((double)(numchsamp))
        - velx*vely;
        //eps = 1/2 epsii -> factor 2 cancels out
        double eps = viscosity  *(epsii_glob[0][i_x].at(idx)/((double)(numchsamp))
                                + epsii_glob[1][i_x].at(idx)/((double)(numchsamp))
                                + epsii_glob[2][i_x].at(idx)/((double)(numchsamp)));
        //to prevent devision by 0
        double eta = -1.;
        if(DNS)
          eta = std::pow(viscosity*viscosity*viscosity/eps,0.25);
        double h_eta = y_h_glob[i_x][idx] / eta;

#ifdef DEBUG_WRITE_OUTPUT
        std::cout << "velxssq using std::abs: " << velxssq << std::endl;
        std::cout << "velxssq using abs: " << abs(velsq_glob[0][i_x].at(idx)/((double)(numchsamp))
            -vel_glob[0][i_x].at(idx)*vel_glob[0][i_x].at(idx)/(((double)numchsamp)*((double)numchsamp))) << std::endl;
        std::cout << "sqrt(velyssq) = " << sqrt(velyssq) << std::endl;
        std::cout << "std::sqrt(velyssq) = " << std::sqrt(velyssq)<<  std::endl;
#endif

        f<<std::scientific<<std::setprecision(9) << std::setw(17) << y_vec_glob[i_x].at(idx)/H;
        f<<std::scientific<<std::setprecision(9) << std::setw(17) << velx/ub;
        f<<std::scientific<<std::setprecision(9) << std::setw(17) << vely/ub;
        f<<std::scientific<<std::setprecision(9) << std::setw(17) << velz/ub;
        f<<std::scientific<<std::setprecision(9) << std::setw(17) << veluu/ubsq;
        f<<std::scientific<<std::setprecision(9) << std::setw(17) << velvv/ubsq;
        f<<std::scientific<<std::setprecision(9) << std::setw(17) << velww/ubsq;
        f<<std::scientific<<std::setprecision(9) << std::setw(17) << veluv/ubsq;
        f<<std::scientific<<std::setprecision(9) << std::setw(17) << 0.5*( veluu + velvv + velww )/ubsq;
        f<<std::scientific<<std::setprecision(9) << std::setw(17) << eps;
        f<<std::scientific<<std::setprecision(9) << std::setw(17) << eta/H;
        f<<std::scientific<<std::setprecision(9) << std::setw(17) << h_eta;
        f<<std::scientific<<std::setprecision(9) << std::setw(17) << y_h_glob[i_x][idx];
        f << std::endl;
      }
      f.close();
    }

    // write tau_w, yplus at bottom
    std::ofstream f;

    f.open((new_output_prefix + ".tauw_Yplus_flow_statistics_bottom").c_str(),std::ios::trunc);
    f<<"statistics of periodic hill flow  "<<std::endl;
    f<<"number of samples:   " << numchsamp << std::endl;
    f<<"fe degree:           " << dof_handler.get_fe().degree << std::endl;

    double pnorm = p_top_glob.at(0)/(double)numchsamp;
    f<< "       x/H      |      Yplus     |       cf       |        cp      " << std::endl;
    for (unsigned int idx = 0; idx<x_glob.size(); idx++)
    {
      double tau_w = viscosity*dudy_bottom_glob.at(idx)/(double)numchsamp;
      double p = p_bottom_glob.at(idx)/(double)numchsamp;

      const double norm = 0.5*ubsq;
      const double cf = tau_w/norm;
      const double cp = (p-pnorm)/norm;
      f<<std::scientific<<std::setprecision(9) << std::setw(17) << x_glob.at(idx)/H;
      f<<std::scientific<<std::setprecision(9) << std::setw(17) << std::sqrt(std::abs(tau_w))*y1_bottom_glob.at(idx)/viscosity;
      f<<std::scientific<<std::setprecision(9) << std::setw(17) << cf;
      f<<std::scientific<<std::setprecision(9) << std::setw(17) << cp;
      f << std::endl;
    }
    f.close();


    // write tau_w, yplus at top
    f.open((new_output_prefix + ".tauw_Yplus_flow_statistics_top").c_str(),std::ios::trunc);
    f<<"statistics of periodic hill flow  "<<std::endl;
    f<<"number of samples:   " << numchsamp << std::endl;
    f<<"fe degree:           " << dof_handler.get_fe().degree << std::endl;

    f<< "       x/H      |      Yplus     |       cf       |        cp      " << std::endl;
    for (unsigned int idx = 0; idx<x_glob.size(); idx++)
    {
      double tau_w = -viscosity*dudy_top_glob.at(idx)/(double)numchsamp; // minus is set because on the top of the domain, the domain the gradient is negative for unseparated u-profiles
      double p = p_top_glob.at(idx)/(double)numchsamp;

      const double norm = 0.5*ubsq;
      const double cf = tau_w/norm;
      const double cp = (p-pnorm)/norm;
      f<<std::scientific<<std::setprecision(9) << std::setw(17) << x_glob.at(idx)/H;
      f<<std::scientific<<std::setprecision(9) << std::setw(17) << std::sqrt(std::abs(tau_w))*y1_top_glob.at(idx)/viscosity;
      f<<std::scientific<<std::setprecision(9) << std::setw(17) << cf;
      f<<std::scientific<<std::setprecision(9) << std::setw(17) << cp;
      f << std::endl;
    }
    f.close();
  }
}

template <int dim>
void
StatisticsManagerPH<dim>::reset()
{
  for (unsigned int i=0; i<x_over_h.size(); i++)
  {
    for(unsigned int j=0;j<dim;j++)
      std::fill(vel_glob[j][i].begin(), vel_glob[j][i].end(), 0.);
    for(unsigned int j=0;j<dim;j++)
      std::fill(velsq_glob[j][i].begin(), velsq_glob[j][i].end(), 0.);
    std::fill(veluv_glob[i].begin(), veluv_glob[i].end(), 0.);
  }
  numchsamp = 0;
}



template <int dim>
void
StatisticsManagerPH<dim>::do_evaluate(const std::vector<const parallel::distributed::Vector<double> *> &velocity,const parallel::distributed::Vector<double> &pressure)
{
// ---------------------------------------------------
// evaluate velocity at given x_over_h positions
// ---------------------------------------------------
  for (unsigned int i_x=0; i_x<x_over_h.size(); i_x++)
  {
    const double x_pos = x_over_h[i_x]*h;

    std::vector<double> length_loc(vel_glob[0][0].size());
    std::vector<std::vector<double> > vel_loc(dim);
    for(unsigned int i=0;i<dim;i++)
      vel_loc[i].resize(vel_glob[0][0].size());
    std::vector<std::vector<double> > velsq_loc(dim);
    for(unsigned int i=0;i<dim;i++)
      velsq_loc[i].resize(vel_glob[0][0].size());
    std::vector<double> veluv_loc(vel_glob[0][0].size());
    std::vector<std::vector<double> > epsii_loc(dim);
    if(DNS)
      for(unsigned int i=0;i<dim;i++)
        epsii_loc[i].resize(vel_glob[0][0].size());
    const unsigned int fe_degree = dof_handler.get_fe().degree;
    std::vector<std::shared_ptr<FEValues<dim,dim> > > fe_values(n_points_y);
    QGauss<1> gauss_1d(fe_degree+1);

    // get xi1 position to the considered x position x_pos
    double xi1_pos = 0.0;
    for (typename DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active(); cell!=dof_handler.end(); ++cell)
      if (cell->is_locally_owned())
        {
        std::vector<double> vertex_x_pos(GeometryInfo<dim>::vertices_per_cell);

        // get vertices of cell
        for (unsigned int v=0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          vertex_x_pos[v] = cell->vertex(v)[0];

        // determine if the cell is relevant for statistics
        std::vector<double>::iterator min = std::min_element(vertex_x_pos.begin(), vertex_x_pos.end());
        std::vector<double>::iterator max = std::max_element(vertex_x_pos.begin(), vertex_x_pos.end());

        if (*min <= x_pos && *max > x_pos)
          {
          xi1_pos = x_pos/(*max-*min) - *min/(*max-*min);
          break;
          }
        }

    for (unsigned int i=0; i<n_points_y; ++i)
    {
      std::vector<Point<dim> > points(gauss_1d.size());
      std::vector<double> weights(gauss_1d.size());
      if(dim == 2)
      {
        points.resize(1);
        weights.resize(1);
      }
      for (unsigned int j=0; j<weights.size(); ++j)
      {
        points[j][0] = xi1_pos;
        points[j][1] = (double)i/(n_points_y-1); // this are the "real" y-values in parameter space
        if(dim==3)
        {
          points[j][2] = gauss_1d.point(j)[0];
          weights[j] = gauss_1d.weight(j);
        }
        else
          weights[j] = 1.;
      }
      if(DNS)
      {
        fe_values[i].reset(new FEValues<dim>(mapping_,
                                             dof_handler.get_fe().base_element(0),
                                             Quadrature<dim>(points, weights),
                                             update_values | update_jacobians |
                                             update_quadrature_points | update_gradients));
      }
      else
      {
        fe_values[i].reset(new FEValues<dim>(mapping_,
                                             dof_handler.get_fe().base_element(0),
                                             Quadrature<dim>(points, weights),
                                             update_values | update_jacobians |
                                             update_quadrature_points));
      }
    }

    const unsigned int scalar_dofs_per_cell = dof_handler.get_fe().base_element(0).dofs_per_cell;
//    std::vector<double> vel_values(fe_values[0]->n_quadrature_points);
    std::vector<Tensor<1,dim> > velocity_vector(scalar_dofs_per_cell); // \vec{u}^e_h

    std::vector<types::global_dof_index> dof_indices(dof_handler.get_fe().dofs_per_cell);

    for (typename DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active(); cell!=dof_handler.end(); ++cell)
      if (cell->is_locally_owned())
        {

        cell->get_dof_indices(dof_indices);
        std::vector<double> vertex_x_pos(GeometryInfo<dim>::vertices_per_cell);

        // get vertices of cell
        for (unsigned int v=0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          vertex_x_pos[v] = cell->vertex(v)[0];

        // determine if the cell is relevant for statistics
        std::vector<double>::iterator min = std::min_element(vertex_x_pos.begin(), vertex_x_pos.end());
        std::vector<double>::iterator max = std::max_element(vertex_x_pos.begin(), vertex_x_pos.end());
        if (*min <= x_pos && *max > x_pos)
          {
          // get velocity_vector[element nodes][components] in element

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

          // perform quadrature
          for (unsigned int i=0; i<n_points_y; ++i)
          {
            fe_values[i]->reinit(typename Triangulation<dim>::active_cell_iterator(cell));

            double length = 0, veluv = 0;
            std::vector<double> vel(dim,0.);
            std::vector<double> velsq(dim,0.);
            std::vector<double> epsii(dim,0.);
            for (unsigned int q=0; q<fe_values[i]->n_quadrature_points; ++q)
              {
              // Assert
              const Point<dim> point_assert = fe_values[i]->quadrature_point(q); // all quadrature points have the same x-value
              (void) point_assert;
              Assert(std::abs(point_assert[0]-x_pos)<1e-13,
                          ExcMessage("Check calculation of xi1-value for quadrature. "
                                     "x_pos = " + patch::to_string(x_pos) + " and point_assert[0] = " + patch::to_string(point_assert[0])));

                // interpolate velocity to the quadrature point
                Tensor<1,dim> velocity;
                Tensor<1,dim> gradu;
                for (unsigned int j=0; j<velocity_vector.size(); ++j)
                  velocity += fe_values[i]->shape_value(j,q) * velocity_vector[j];
                if(DNS)
                  for (unsigned int j=0; j<velocity_vector.size(); ++j)
                    for (unsigned int idim = 0; idim < dim;idim++)
                      gradu[idim] += fe_values[i]->shape_grad(j,q)[idim] * velocity_vector[j][idim];
                double reduced_jacobian;
                if(dim==3)
                  reduced_jacobian = fe_values[i]->jacobian(q)[2][2];
                else
                  reduced_jacobian = 1.;

                double length_ele = reduced_jacobian * fe_values[i]->get_quadrature().weight(q);
                length += length_ele; // length is the length of one element. length_ele is one summand of the quadrature
                for(unsigned int j=0;j<dim;j++)
                  vel[j] += velocity[j] * length_ele;
                for(unsigned int j=0;j<dim;j++)
                  velsq[j] += velocity[j] * velocity[j] * length_ele;
                veluv += velocity[0] * velocity[1] * length_ele;
                if(DNS)
                  for(unsigned int j=0;j<dim;j++)
                    epsii[j] += gradu[j] * gradu[j] * length_ele;
              }

            // check quadrature
            double n_cells_z_dir = (n_points_y_glob-1)/(n_points_y-1); // ASSUMED: n_cells_y_dir = n_cells_z_dir
            (void) n_cells_z_dir;
            if(dim == 3)
              Assert(std::abs(length - 4.5*h/n_cells_z_dir) < 1e-13,
                      ExcMessage("check quadrature code. length = " + patch::to_string(length) + " lz_ele = " + patch::to_string(4.5*h/n_cells_z_dir) + "(element-length in z direction)"));


            // find index within the y-values: first do a binary search to find
            // the next larger value of y in the list...
            const double y = fe_values[i]->quadrature_point(0)[1];
            unsigned int idx = std::distance(y_vec_glob[i_x].begin(),
                                             std::lower_bound(y_vec_glob[i_x].begin(), y_vec_glob[i_x].end(),
                                                              y));
            // ..., then, check whether the point before was closer (but off
            // by 1e-13 or less)
            if (idx > 0 && std::abs(y_vec_glob[i_x][idx-1]-y) < std::abs(y_vec_glob[i_x][idx]-y))
              idx--;
            Assert(idx < n_points_y_glob,
                        ExcMessage("idx is out of range. The current x/h-position is " + patch::to_string(x_over_h[i_x]) +
                                   ". idx = " + patch::to_string(idx) + " n_points_y_glob = " + patch::to_string(n_points_y_glob)));

            Assert(std::abs(y_vec_glob[i_x][idx]-y)<1e-13,//1e-13,
                        ExcMessage("Could not locate " + patch::to_string(y) + " among "
                                   "pre-evaluated points. Closest point is " +
                                   patch::to_string(y_vec_glob[i_x][idx]) + " at distance " +
                                   patch::to_string(std::abs(y_vec_glob[i_x][idx]-y)) +
                                   ". The current x/h-position is " + patch::to_string(x_over_h[i_x]) +
                                   " Check your discretization. It should have a refinement level of minimum 3 and a fe-degree of minimum 2."));


            for(unsigned int j=0;j<dim;j++)
              vel_loc[j][idx] += vel[j];
            for(unsigned int j=0;j<dim;j++)
              velsq_loc[j][idx] += velsq[j];
            veluv_loc.at(idx) += veluv;
            length_loc.at(idx) += length;
            if(DNS)
              for(unsigned int j=0;j<dim;j++)
                epsii_loc[j][idx] += epsii[j];
          } // loop over y-points within one element
          }
        }
        // loop over cells

    // accumulate data over all processors overwriting the processor-local data
    // in xxx_loc
    for(unsigned int j=0;j<dim;j++)
      Utilities::MPI::sum(vel_loc[j], communicator, vel_loc[j]);
    for(unsigned int j=0;j<dim;j++)
      Utilities::MPI::sum(velsq_loc[j], communicator, velsq_loc[j]);
    Utilities::MPI::sum(veluv_loc, communicator, veluv_loc);
    Utilities::MPI::sum(length_loc, communicator, length_loc);
    if(DNS)
      for(unsigned int j=0;j<dim;j++)
        Utilities::MPI::sum(epsii_loc[j], communicator, epsii_loc[j]);

    // check quadrature
    for (unsigned int idx = 0; idx<y_vec_glob[i_x].size(); idx++)
      if(dim == 3)
        Assert(std::abs(length_loc.at(idx) - 4.5*h) < 1e-13 || std::abs(length_loc.at(idx) - 2*4.5*h) < 1e-13,
            ExcMessage("check quadrature code. length_loc.at(" + patch::to_string(idx) + ") = " + patch::to_string(length_loc.at(idx)) + " 4.5*h = " + patch::to_string(4.5*h)));



    for (unsigned int idx = 0; idx<y_vec_glob[i_x].size(); idx++)
    {
      for(unsigned int j=0;j<dim;j++)
        vel_glob[j][i_x][idx] += vel_loc[j][idx]/length_loc[idx];
      for(unsigned int j=0;j<dim;j++)
        velsq_glob[j][i_x][idx] += velsq_loc[j][idx]/length_loc[idx];
      veluv_glob[i_x].at(idx) += veluv_loc[idx]/length_loc[idx];
      if(DNS)
        for(unsigned int j=0;j<dim;j++)
          epsii_glob[j][i_x][idx] += epsii_loc[j][idx]/length_loc[idx];
    }
  } // for loop over x

  // -----------------------------------------------------------------
  // evaluate velocity an certain x-positions for calculation of tau_w
  // -----------------------------------------------------------------

  // on the bottom
  std::vector<double> length_loc(dudy_bottom_glob.size(),0.0);
  std::vector<double> dudy_w_loc(dudy_bottom_glob.size(),0.0);
  std::vector<double> p_w_loc(dudy_bottom_glob.size(),0.0);

  const unsigned int fe_degree = dof_handler.get_fe().degree;
  std::vector<std::shared_ptr<FEValues<dim,dim> > > fe_values(n_points_x);
  std::vector<std::shared_ptr<FEFaceValues<dim,dim> > > fe_face_values(n_points_x);
  QGauss<1> gauss_1d(fe_degree+1);

  for (unsigned int i=0; i<n_points_x; ++i)
  {
    std::vector<Point<dim> > points(gauss_1d.size());
    std::vector<Point<dim-1> > points_face(gauss_1d.size());
    std::vector<double> weights(gauss_1d.size());
    if(dim == 2)
    {
      points.resize(1);
      points_face.resize(1);
      weights.resize(1);
    }
    for (unsigned int j=0; j<weights.size(); ++j)
    {
      points[j][0] = (double)i/(n_points_x-1);  // linear distributed form 0 to 1
      points[j][1] = 0.0;                       // assumed to be the bottom of the cells
      points_face[j][0] = (double)i/(n_points_x-1);  // linear distributed form 0 to 1
      if(dim==3)
      {
        points[j][2] = gauss_1d.point(j)[0];
        points_face[j][1] = gauss_1d.point(j)[0];
        weights[j] = gauss_1d.weight(j);
      }
      else
        weights[j] = 1.;
    }
    fe_values[i].reset(new FEValues<dim>(mapping_,
                                         dof_handler.get_fe().base_element(0),
                                         Quadrature<dim>(points, weights),
                                         update_values | update_jacobians |
                                         update_quadrature_points | update_gradients));
    fe_face_values[i].reset(new FEFaceValues<dim>(mapping_,
                                         dof_handler.get_fe().base_element(0),
                                         Quadrature<dim-1>(points_face, weights),
                                         update_jacobians |
                                         update_quadrature_points | update_normal_vectors));
  }

  const unsigned int scalar_dofs_per_cell = dof_handler.get_fe().base_element(0).dofs_per_cell;
  std::vector<double> vel_values(fe_values[0]->n_quadrature_points);
  std::vector<Tensor<1,dim> > velocity_vector(scalar_dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dof_handler.get_fe().dofs_per_cell);

  std::vector<double > pressure_vector(scalar_dofs_per_cell); // \vec{p}^e_h
  std::vector<types::global_dof_index> dof_indices_p(dof_handler_p.get_fe().dofs_per_cell);
//  FEValuesExtractors::Vector v_extract(0);
  typename DoFHandler<dim>::active_cell_iterator cell_p=dof_handler_p.begin_active();
  for (typename DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active(); cell!=dof_handler.end(); ++cell, ++cell_p)
    if (cell->is_locally_owned())
      {
      if (cell->at_boundary(2)) // just evaluate cells at the bottom of the domain
      {
        cell->get_dof_indices(dof_indices);
        cell_p->get_dof_indices(dof_indices_p);
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
        for (unsigned int j=0; j<dof_indices_p.size(); ++j)
          {
            pressure_vector[j] = pressure(dof_indices_p[j]);
          }

        std::vector<Tensor<1,dim> > n;
        for (unsigned int i=0; i<n_points_x; ++i)
        {
          fe_values[i]->reinit(typename Triangulation<dim>::active_cell_iterator(cell));

          double length = 0, dudy_w = 0, p_w = 0;

          for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
            if (cell->face(face_number)->at_boundary())
              {
                fe_face_values[i]->reinit(cell,face_number);
                n.resize(fe_face_values[i]->n_quadrature_points);
                for (unsigned int q=0; q<fe_face_values[i]->n_quadrature_points; ++q)
                  n[q]=fe_face_values[i]->normal_vector(q);
              }
          for (unsigned int q=0; q<fe_values[i]->n_quadrature_points; ++q)
          {
            // calculate dudy at the quadrature point
            Tensor<1,dim,double > du = Tensor<1,dim,double >();
            Tensor<1,dim,double > dv = Tensor<1,dim,double >();
            double dudn = 0.0;
            double p = 0.;
            for (unsigned int j=0; j<velocity_vector.size(); ++j)
              du += fe_values[i]->shape_grad(j,q) * velocity_vector[j][0];
            double dudn2 = (du*n[q])*(du*n[q]) + (dv*n[q])*(dv*n[q]);
            dudn = std::sqrt(std::abs(dudn2));
            if(du[1]<0.)
              dudn *= -1.;

            for (unsigned int j=0; j<velocity_vector.size(); ++j)
              p += fe_values[i]->shape_value(j,q) * pressure_vector[j];

            double reduced_jacobian;
            if(dim==3)
              reduced_jacobian = fe_values[i]->jacobian(q)[2][2];
            else
              reduced_jacobian = 1.;

            double length_ele = reduced_jacobian * fe_values[i]->get_quadrature().weight(q);
            length += length_ele;
            dudy_w += dudn*length_ele;
            p_w += p*length_ele;

          }

          double n_cells_z_dir = (n_points_y_glob-1)/(n_points_y-1); // ASSUMED: n_cells_y_dir = n_cells_z_dir
          (void) n_cells_z_dir;
          if(dim == 3)
          Assert(std::abs(length - 4.5*h/n_cells_z_dir) < 1e-13,
                  ExcMessage("check quadrature code. length = " + patch::to_string(length) + " lz_ele = " + patch::to_string(4.5*h/n_cells_z_dir) + "(element-length in z direction)"));

          // find index within the x-values: first do a binary search to find
          // the next larger value of x in the list...
          const double x = fe_values[i]->quadrature_point(0)[0];
          unsigned int idx = std::distance(x_glob.begin(),
                                           std::lower_bound(x_glob.begin(), x_glob.end(),
                                                            x));
          // ..., then, check whether the point before was closer (but off
          // by 1e-13 or less)
          if (idx > 0 && std::abs(x_glob[idx-1]-x) < std::abs(x_glob[idx]-x))
            idx--;
          Assert(std::abs(x_glob[idx]-x)<1e-13,
                      ExcMessage("Could not locate " + patch::to_string(x) + " among "
                                 "pre-evaluated points. Closest point is " +
                                 patch::to_string(x_glob[idx]) + " at distance " +
                                 patch::to_string(std::abs(x_glob[idx]-x))));

          dudy_w_loc.at(idx) += dudy_w;
          length_loc.at(idx) += length;
          p_w_loc.at(idx) += p_w;

        }
      }
      }
  // accumulate data over all processors overwriting the processor-local data
  // in xxx_loc
  Utilities::MPI::sum(dudy_w_loc, communicator, dudy_w_loc);
  Utilities::MPI::sum(length_loc, communicator, length_loc);
  Utilities::MPI::sum(p_w_loc, communicator, p_w_loc);

  // check quadrature
  for (unsigned int idx = 0; idx<x_glob.size(); idx++)
    if(dim == 3)
      Assert(std::abs(length_loc.at(idx) - 4.5*h) < 1e-13 || std::abs(length_loc.at(idx) - 2*4.5*h) < 1e-13,
          ExcMessage("check quadrature code. length_loc.at(" + patch::to_string(idx) + ") = " + patch::to_string(length_loc.at(idx)) + " 4.5*h = " + patch::to_string(4.5*h)));

  for (unsigned int idx = 0; idx<x_glob.size(); idx++)
  {
    dudy_bottom_glob.at(idx) += dudy_w_loc[idx]/length_loc[idx];
    p_bottom_glob.at(idx) += p_w_loc[idx]/length_loc[idx];
  }



  // on the TOP
    std::vector<double> length_top_loc(dudy_top_glob.size(),0.0);
    std::vector<double> dudy_top_loc(dudy_top_glob.size(),0.0);
    std::vector<double> p_top_loc(p_top_glob.size(),0.0);

//    const unsigned int fe_degree = dof_handler.get_fe().degree;
    std::vector<std::shared_ptr<FEValues<dim,dim> > > fe_values_top(n_points_x);
//    QGauss<1> gauss_1d(fe_degree+1);

    for (unsigned int i=0; i<n_points_x; ++i)
    {
      std::vector<Point<dim> > points(gauss_1d.size());
      std::vector<double> weights(gauss_1d.size());
      if(dim == 2)
      {
        points.resize(1);
        weights.resize(1);
      }
      for (unsigned int j=0; j<weights.size(); ++j)
      {
        points[j][0] = (double)i/(n_points_x-1);  // linear distributed form 0 to 1
        points[j][1] = 1.0;                       // assumed to be the top of the cells
        if(dim==3)
        {
          points[j][2] = gauss_1d.point(j)[0];
          weights[j] = gauss_1d.weight(j);
        }
        else
          weights[j] = 1.;
      }
      fe_values_top[i].reset(new FEValues<dim>(mapping_,
                                           dof_handler.get_fe().base_element(0),
                                           Quadrature<dim>(points, weights),
                                           update_values | update_jacobians |
                                           update_quadrature_points | update_gradients));
    }

    std::vector<Tensor<1,dim> > velocity_vector_top(scalar_dofs_per_cell);
    std::vector<double > pressure_vector_top(scalar_dofs_per_cell); // \vec{p}^e_h

    cell_p=dof_handler_p.begin_active();

    for (typename DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active(); cell!=dof_handler.end(); ++cell, ++cell_p)
      if (cell->is_locally_owned())
        {
        if (cell->at_boundary(3)) // just evaluate cells at the bottom of the domain
        {
          cell->get_dof_indices(dof_indices);
          cell_p->get_dof_indices(dof_indices_p);
          // vector-valued FE where all components are explicitly listed in the
          // DoFHandler
          if (dof_handler.get_fe().element_multiplicity(0) >= dim)
            for (unsigned int j=0; j<dof_indices.size(); ++j)
              {
                const std::pair<unsigned int,unsigned int> comp =
                  dof_handler.get_fe().system_to_component_index(j);
                if (comp.first < dim)
                  velocity_vector_top[comp.second][comp.first] = (*velocity[0])(dof_indices[j]);
              }
          else
            // scalar FE where we have several vectors referring to the same
            // DoFHandler
            {
              AssertDimension(dof_handler.get_fe().element_multiplicity(0), 1);
              for (unsigned int j=0; j<scalar_dofs_per_cell; ++j)
                for (unsigned int d=0; d<dim; ++d)
                  velocity_vector_top[j][d] = (*velocity[d])(dof_indices[j]);
            }

          for (unsigned int j=0; j<dof_indices_p.size(); ++j)
            {
              pressure_vector_top[j] = pressure(dof_indices_p[j]);
            }

          for (unsigned int i=0; i<n_points_x; ++i)
          {
            fe_values_top[i]->reinit(typename Triangulation<dim>::active_cell_iterator(cell));

            double length = 0, dudy_w = 0, p_w = 0;

            for (unsigned int q=0; q<fe_values_top[i]->n_quadrature_points; ++q)
            {
              // calculate dudy at the quadrature point
              double dudy = 0.0;
              double p = 0.;
              for (unsigned int j=0; j<velocity_vector_top.size(); ++j)
                dudy += fe_values_top[i]->shape_grad(j,q)[1] * velocity_vector_top[j][0];

              for (unsigned int j=0; j<velocity_vector_top.size(); ++j)
                p += fe_values_top[i]->shape_value(j,q) * pressure_vector_top[j];

              double reduced_jacobian;
              if(dim==3)
                reduced_jacobian = fe_values_top[i]->jacobian(q)[2][2];
              else
                reduced_jacobian = 1.;

              double length_ele = reduced_jacobian * fe_values_top[i]->get_quadrature().weight(q);
              length += length_ele;
              dudy_w += dudy*length_ele;
              p_w += p*length_ele;

            }

            double n_cells_z_dir = (n_points_y_glob-1)/(n_points_y-1); // ASSUMED: n_cells_y_dir = n_cells_z_dir
            (void) n_cells_z_dir;
            if(dim == 3)
              Assert(std::abs(length - 4.5*h/n_cells_z_dir) < 1e-13,
                      ExcMessage("check quadrature code. length = " + patch::to_string(length) + " lz_ele = " + patch::to_string(4.5*h/n_cells_z_dir) + "(element-length in z direction)"));

            // find index within the x-values: first do a binary search to find
            // the next larger value of x in the list...
            const double x = fe_values_top[i]->quadrature_point(0)[0];
            unsigned int idx = std::distance(x_glob.begin(),
                                             std::lower_bound(x_glob.begin(), x_glob.end(),
                                                              x));
            // ..., then, check whether the point before was closer (but off
            // by 1e-13 or less)
            if (idx > 0 && std::abs(x_glob[idx-1]-x) < std::abs(x_glob[idx]-x))
              idx--;
            Assert(std::abs(x_glob[idx]-x)<1e-13,
                        ExcMessage("Could not locate " + patch::to_string(x) + " among "
                                   "pre-evaluated points. Closest point is " +
                                   patch::to_string(x_glob[idx]) + " at distance " +
                                   patch::to_string(std::abs(x_glob[idx]-x))));

            dudy_top_loc.at(idx) += dudy_w;
            length_top_loc.at(idx) += length;
            p_top_loc.at(idx) += p_w;
          }
        }
        }
    // accumulate data over all processors overwriting the processor-local data
    // in xxx_loc
    Utilities::MPI::sum(dudy_top_loc, communicator, dudy_top_loc);
    Utilities::MPI::sum(length_top_loc, communicator, length_top_loc);
    Utilities::MPI::sum(p_top_loc, communicator, p_top_loc);

    // check quadrature
    for (unsigned int idx = 0; idx<x_glob.size(); idx++)
      if(dim == 3)
        Assert(std::abs(length_top_loc.at(idx) - 4.5*h) < 1e-13 || std::abs(length_top_loc.at(idx) - 2*4.5*h) < 1e-13,
            ExcMessage("check quadrature code. length_loc.at(" + patch::to_string(idx) + ") = " + patch::to_string(length_top_loc.at(idx)) + " 4.5*h = " + patch::to_string(4.5*h)));

    for (unsigned int idx = 0; idx<x_glob.size(); idx++)
    {
      dudy_top_glob.at(idx) += dudy_top_loc[idx]/length_top_loc[idx];
      p_top_glob.at(idx) += p_top_loc[idx]/length_top_loc[idx];
    }

  numchsamp++;
}

template <int dim>
void
StatisticsManagerPH<dim>::do_evaluate_xwall(const std::vector<const parallel::distributed::Vector<double> *> &velocity,
                                            const parallel::distributed::Vector<double> &pressure,
                                            const DoFHandler<dim>                       &dof_handler_wdist,
                                            const FEParameters<dim>                     &fe_param)
{
// ---------------------------------------------------
// evaluate velocity at given x_over_h positions
// ---------------------------------------------------
  for (unsigned int i_x=0; i_x<x_over_h.size(); i_x++)
  {
    const double x_pos = x_over_h[i_x]*h;

    std::vector<double> length_loc(vel_glob[0][0].size());
    std::vector<std::vector<double> > vel_loc(dim);
    for(unsigned int i=0;i<dim;i++)
      vel_loc[i].resize(vel_glob[0][0].size());
    std::vector<std::vector<double> > velsq_loc(dim);
    for(unsigned int i=0;i<dim;i++)
      velsq_loc[i].resize(vel_glob[0][0].size());
    std::vector<double> veluv_loc(vel_glob[0][0].size());

    const unsigned int fe_degree = dof_handler.get_fe().degree;
    std::vector<std::shared_ptr<FEValues<dim,dim> > > fe_values(n_points_y);
    std::vector<std::shared_ptr<FEValues<dim,dim> > > fe_values_xwall(n_points_y);
    std::vector<std::shared_ptr<FEValues<dim,dim> > > fe_values_tauw(n_points_y);
    QGauss<1> gauss_1d(fe_degree+1);

    // get xi1 position to the considered x position x_pos
    double xi1_pos = 0.0;
    for (typename DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active(); cell!=dof_handler.end(); ++cell)
      if (cell->is_locally_owned())
        {
        std::vector<double> vertex_x_pos(GeometryInfo<dim>::vertices_per_cell);

        // get vertices of cell
        for (unsigned int v=0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          vertex_x_pos[v] = cell->vertex(v)[0];

        // determine if the cell is relevant for statistics
        std::vector<double>::iterator min = std::min_element(vertex_x_pos.begin(), vertex_x_pos.end());
        std::vector<double>::iterator max = std::max_element(vertex_x_pos.begin(), vertex_x_pos.end());

        if (*min <= x_pos && *max > x_pos)
          {
          xi1_pos = x_pos/(*max-*min) - *min/(*max-*min);
          break;
          }
        }

    for (unsigned int i=0; i<n_points_y; ++i)
    {
      std::vector<Point<dim> > points(gauss_1d.size());
      std::vector<double> weights(gauss_1d.size());
      if(dim == 2)
      {
        points.resize(1);
        weights.resize(1);
      }
      for (unsigned int j=0; j<weights.size(); ++j)
      {
        points[j][0] = xi1_pos;
        points[j][1] = (double)i/(n_points_y-1); // this are the "real" y-values in parameter space
        if(dim==3)
        {
          points[j][2] = gauss_1d.point(j)[0];
          weights[j] = gauss_1d.weight(j);
        }
        else
          weights[j] = 1.;
      }
      fe_values[i].reset(new FEValues<dim>(mapping_,
                                           dof_handler.get_fe().base_element(0),
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
//    std::vector<double> vel_values(fe_values[0]->n_quadrature_points);
    std::vector<Tensor<1,dim> > velocity_vector(scalar_dofs_per_cell); // \vec{u}^e_h
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
        std::vector<double> vertex_x_pos(GeometryInfo<dim>::vertices_per_cell);

        // get vertices of cell
        for (unsigned int v=0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          vertex_x_pos[v] = cell->vertex(v)[0];

        // determine if the cell is relevant for statistics
        std::vector<double>::iterator min = std::min_element(vertex_x_pos.begin(), vertex_x_pos.end());
        std::vector<double>::iterator max = std::max_element(vertex_x_pos.begin(), vertex_x_pos.end());
        if (*min <= x_pos && *max > x_pos)
          {
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

          // perform quadrature
          for (unsigned int i=0; i<n_points_y; ++i)
          {
            fe_values[i]->reinit(typename Triangulation<dim>::active_cell_iterator(cell));
            fe_values_xwall[i]->reinit(typename Triangulation<dim>::active_cell_iterator(cell));
            fe_values_tauw[i]->reinit(typename Triangulation<dim>::active_cell_iterator(cell));

            double length = 0, veluv = 0;
            std::vector<double> vel(dim,0.);
            std::vector<double> velsq(dim,0.);

            AlignedVector<double > wdist(fe_values[i]->n_quadrature_points,0.);
            AlignedVector<double > tauw(fe_values[i]->n_quadrature_points,0.);
            for (unsigned int q=0; q<fe_values[i]->n_quadrature_points; ++q)
            {
              for (unsigned int j=0; j<wdist_vector.size(); ++j)
                wdist[q] += (fe_values_tauw[i]->shape_value(j,q) * wdist_vector[j])[0];
              for (unsigned int j=0; j<tauw_vector.size(); ++j)
                tauw[q] += (fe_values_tauw[i]->shape_value(j,q) * tauw_vector[j])[0];
            }
            SpaldingsLawEvaluation<dim, double, double > spalding(fe_param.viscosity);
            spalding.reinit(wdist,tauw,fe_values[i]->n_quadrature_points);
            for (unsigned int q=0; q<fe_values[i]->n_quadrature_points; ++q)
              {
              // Assert
              const Point<dim> point_assert = fe_values[i]->quadrature_point(q); // all quadrature points have the same x-value
              AssertThrow(std::abs(point_assert[0]-x_pos)<1e-13,
                          ExcMessage("Check calculation of xi1-value for quadrature. "
                                     "x_pos = " + patch::to_string(x_pos) + " and point_assert[0] = " + patch::to_string(point_assert[0])));


                // interpolate velocity to the quadrature point
                Tensor<1,dim> velocity;
                for (unsigned int j=0; j<velocity_vector.size(); ++j)
                  velocity += fe_values[i]->shape_value(j,q) * velocity_vector[j];
                for (unsigned int j=0; j<velocity_vector_xwall.size(); ++j)
                  velocity += fe_values_xwall[i]->shape_value(j,q) * velocity_vector_xwall[j] * spalding.enrichment(q);

                double reduced_jacobian;
                if(dim==3)
                  reduced_jacobian = fe_values[i]->jacobian(q)[2][2];
                else
                  reduced_jacobian = 1.;

                double length_ele = reduced_jacobian * fe_values[i]->get_quadrature().weight(q);
                length += length_ele; // length is the length of one element. length_ele is one summand of the quadrature
                for(unsigned int j=0;j<dim;j++)
                  vel[j] += velocity[j] * length_ele;
                for(unsigned int j=0;j<dim;j++)
                  velsq[j] += velocity[j] * velocity[j] * length_ele;
                veluv += velocity[0] * velocity[1] * length_ele;
              }

            // check quadrature
            double n_cells_z_dir = (n_points_y_glob-1)/(n_points_y-1); // ASSUMED: n_cells_y_dir = n_cells_z_dir
            if(dim == 3)
              AssertThrow(std::abs(length - 4.5*h/n_cells_z_dir) < 1e-13,
                      ExcMessage("check quadrature code. length = " + patch::to_string(length) + " lz_ele = " + patch::to_string(4.5*h/n_cells_z_dir) + "(element-length in z direction)"));


            // find index within the y-values: first do a binary search to find
            // the next larger value of y in the list...
            const double y = fe_values[i]->quadrature_point(0)[1];
            unsigned int idx = std::distance(y_vec_glob[i_x].begin(),
                                             std::lower_bound(y_vec_glob[i_x].begin(), y_vec_glob[i_x].end(),
                                                              y));
            // ..., then, check whether the point before was closer (but off
            // by 1e-13 or less)
            if (idx > 0 && std::abs(y_vec_glob[i_x][idx-1]-y) < std::abs(y_vec_glob[i_x][idx]-y))
              idx--;
            AssertThrow(idx < n_points_y_glob,
                        ExcMessage("idx is out of range. The current x/h-position is " + patch::to_string(x_over_h[i_x]) +
                                   ". idx = " + patch::to_string(idx) + " n_points_y_glob = " + patch::to_string(n_points_y_glob)));

            AssertThrow(std::abs(y_vec_glob[i_x][idx]-y)<1e-13,//1e-13,
                        ExcMessage("Could not locate " + patch::to_string(y) + " among "
                                   "pre-evaluated points. Closest point is " +
                                   patch::to_string(y_vec_glob[i_x][idx]) + " at distance " +
                                   patch::to_string(std::abs(y_vec_glob[i_x][idx]-y)) +
                                   ". The current x/h-position is " + patch::to_string(x_over_h[i_x]) +
                                   " Check your discretization. It should have a refinement level of minimum 3 and a fe-degree of minimum 2."));


            for(unsigned int j=0;j<dim;j++)
              vel_loc[j][idx] += vel[j];
            for(unsigned int j=0;j<dim;j++)
              velsq_loc[j][idx] += velsq[j];
            veluv_loc.at(idx) += veluv;
            length_loc.at(idx) += length;
          } // loop over y-points within one element
          }
        }
        // loop over cells

    // accumulate data over all processors overwriting the processor-local data
    // in xxx_loc
    for(unsigned int j=0;j<dim;j++)
      Utilities::MPI::sum(vel_loc[j], communicator, vel_loc[j]);
    for(unsigned int j=0;j<dim;j++)
      Utilities::MPI::sum(velsq_loc[j], communicator, velsq_loc[j]);
    Utilities::MPI::sum(veluv_loc, communicator, veluv_loc);
    Utilities::MPI::sum(length_loc, communicator, length_loc);

    // check quadrature
    for (unsigned int idx = 0; idx<y_vec_glob[i_x].size(); idx++)
      if(dim == 3)
        AssertThrow(std::abs(length_loc.at(idx) - 4.5*h) < 1e-13 || std::abs(length_loc.at(idx) - 2*4.5*h) < 1e-13,
            ExcMessage("check quadrature code. length_loc.at(" + patch::to_string(idx) + ") = " + patch::to_string(length_loc.at(idx)) + " 4.5*h = " + patch::to_string(4.5*h)));



    for (unsigned int idx = 0; idx<y_vec_glob[i_x].size(); idx++)
    {
      for(unsigned int j=0;j<dim;j++)
        vel_glob[j][i_x][idx] += vel_loc[j][idx]/length_loc[idx];
      for(unsigned int j=0;j<dim;j++)
        velsq_glob[j][i_x][idx] += velsq_loc[j][idx]/length_loc[idx];
      veluv_glob[i_x].at(idx) += veluv_loc[idx]/length_loc[idx];
    }
  } // for loop over x

  // -----------------------------------------------------------------
  // evaluate velocity an certain x-positions for calculation of tau_w
  // -----------------------------------------------------------------

  // on the bottom
  std::vector<double> length_loc(dudy_bottom_glob.size(),0.0);
  std::vector<double> dudy_w_loc(dudy_bottom_glob.size(),0.0);
  std::vector<double> p_w_loc(dudy_bottom_glob.size(),0.0);

  const unsigned int fe_degree = dof_handler.get_fe().degree;
  std::vector<std::shared_ptr<FEValues<dim,dim> > > fe_values(n_points_x);
  std::vector<std::shared_ptr<FEValues<dim,dim> > > fe_values_xwall(n_points_x);
  std::vector<std::shared_ptr<FEValues<dim,dim> > > fe_values_p(n_points_x);
  std::vector<std::shared_ptr<FEValues<dim,dim> > > fe_values_tauw(n_points_x);
  QGauss<1> gauss_1d(fe_degree+1);

  for (unsigned int i=0; i<n_points_x; ++i)
  {
    std::vector<Point<dim> > points(gauss_1d.size());
    std::vector<double> weights(gauss_1d.size());
    if(dim == 2)
    {
      points.resize(1);
      weights.resize(1);
    }
    for (unsigned int j=0; j<weights.size(); ++j)
    {
      points[j][0] = (double)i/(n_points_x-1);  // linear distributed form 0 to 1
      points[j][1] = 0.0;                       // assumed to be the bottom of the cells
      if(dim==3)
      {
        points[j][2] = gauss_1d.point(j)[0];
        weights[j] = gauss_1d.weight(j);
      }
      else
        weights[j] = 1.;
    }
    fe_values[i].reset(new FEValues<dim>(mapping_,
                                         dof_handler.get_fe().base_element(0),
                                         Quadrature<dim>(points, weights),
                                         update_values | update_jacobians |
                                         update_quadrature_points | update_gradients));
    fe_values_xwall[i].reset(new FEValues<dim>(mapping_,
                                         dof_handler.get_fe().base_element(1),
                                         Quadrature<dim>(points, weights),
                                         update_values | update_jacobians |
                                         update_quadrature_points | update_gradients));
    fe_values_p[i].reset(new FEValues<dim>(mapping_,
                                         dof_handler_p.get_fe(),
                                         Quadrature<dim>(points, weights),
                                         update_values | update_jacobians |
                                         update_quadrature_points | update_gradients));
    fe_values_tauw[i].reset(new FEValues<dim>(mapping_,
                                         dof_handler_wdist.get_fe(),
                                         Quadrature<dim>(points, weights),
                                         update_values | update_jacobians |
                                         update_quadrature_points | update_gradients));
  }

  const unsigned int scalar_dofs_per_cell = dof_handler.get_fe().base_element(0).dofs_per_cell;
  const unsigned int scalar_dofs_per_cell_p = dof_handler_p.get_fe().dofs_per_cell;
  const unsigned int scalar_dofs_per_cell_xwall = dof_handler.get_fe().base_element(1).dofs_per_cell;
  const unsigned int scalar_dofs_per_cell_tauw = dof_handler_wdist.get_fe().base_element(0).dofs_per_cell;
//    std::vector<double> vel_values(fe_values[0]->n_quadrature_points);
  std::vector<Tensor<1,dim> > velocity_vector(scalar_dofs_per_cell); // \vec{u}^e_h
  std::vector<Tensor<1,dim> > velocity_vector_xwall(scalar_dofs_per_cell_xwall);
  std::vector<Tensor<1,1> > tauw_vector(scalar_dofs_per_cell_tauw);
  std::vector<Tensor<1,1> > wdist_vector(scalar_dofs_per_cell_tauw);
  std::vector<types::global_dof_index> dof_indices(dof_handler.get_fe().dofs_per_cell);

  std::vector<double > pressure_vector(scalar_dofs_per_cell_p); // \vec{p}^e_h
  std::vector<types::global_dof_index> dof_indices_p(scalar_dofs_per_cell_p);
  typename DoFHandler<dim>::active_cell_iterator cell_p=dof_handler_p.begin_active();
  std::vector<types::global_dof_index> dof_indices_tauw(dof_handler_wdist.get_fe().dofs_per_cell);
  typename DoFHandler<dim>::active_cell_iterator cell_tauw=dof_handler_wdist.begin_active();
  for (typename DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active(); cell!=dof_handler.end(); ++cell, ++cell_p, ++cell_tauw)
    if (cell->is_locally_owned())
      {
      if (cell->at_boundary(2)) // just evaluate cells at the bottom of the domain
      {
        cell->get_dof_indices(dof_indices);
        cell_p->get_dof_indices(dof_indices_p);
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

        for (unsigned int j=0; j<dof_indices_p.size(); ++j)
          {
            pressure_vector[j] = pressure(dof_indices_p[j]);
          }

        for (unsigned int i=0; i<n_points_x; ++i)
        {
          fe_values[i]->reinit(typename Triangulation<dim>::active_cell_iterator(cell));
          fe_values_p[i]->reinit(typename Triangulation<dim>::active_cell_iterator(cell));
          fe_values_xwall[i]->reinit(typename Triangulation<dim>::active_cell_iterator(cell));
          fe_values_tauw[i]->reinit(typename Triangulation<dim>::active_cell_iterator(cell));

          double length = 0, dudy_w = 0, p_w = 0;

          AlignedVector<double > wdist(fe_values[i]->n_quadrature_points,0.);
          AlignedVector<double > tauw(fe_values[i]->n_quadrature_points,0.);
          AlignedVector<Tensor<1,dim,double > > gradwdist(fe_values[i]->n_quadrature_points);
          AlignedVector<Tensor<1,dim,double > > gradtauw(fe_values[i]->n_quadrature_points);
          for (unsigned int q=0; q<fe_values[i]->n_quadrature_points; ++q)
          {
            for (unsigned int j=0; j<wdist_vector.size(); ++j)
              wdist[q] += (fe_values_tauw[i]->shape_value(j,q) * wdist_vector[j])[0];
            for (unsigned int j=0; j<tauw_vector.size(); ++j)
              tauw[q] += (fe_values_tauw[i]->shape_value(j,q) * tauw_vector[j])[0];

            for (unsigned int j=0; j<tauw_vector.size(); ++j)
              gradtauw[q] += fe_values_tauw[i]->shape_grad(j,q) * tauw_vector[j][0];
            for (unsigned int j=0; j<wdist_vector.size(); ++j)
              gradwdist[q] += fe_values_tauw[i]->shape_grad(j,q) * wdist_vector[j][0];
          }
          SpaldingsLawEvaluation<dim, double, double > spalding(fe_param.viscosity);
          std::vector<bool> enrcomp(1);
          enrcomp[0] = true;
          spalding.reinit(wdist,tauw,gradwdist,gradtauw,fe_values[i]->n_quadrature_points,enrcomp);

          for (unsigned int q=0; q<fe_values[i]->n_quadrature_points; ++q)
          {
            // calculate dudy at the quadrature point
            Tensor<1,dim,double > du = Tensor<1,dim,double >();
            Tensor<1,dim,double > dv = Tensor<1,dim,double >();
            double dudn = 0.0;
            double p = 0.;
            for (unsigned int j=0; j<velocity_vector.size(); ++j)
              du += fe_values[i]->shape_grad(j,q) * velocity_vector[j][0];
            for (unsigned int j=0; j<velocity_vector.size(); ++j)
              dv += fe_values[i]->shape_grad(j,q) * velocity_vector[j][1];

            for (unsigned int j=0; j<velocity_vector_xwall.size(); ++j)
              du += fe_values_xwall[i]->shape_grad(j,q) * velocity_vector_xwall[j][0]* spalding.enrichment(q)
                    + fe_values_xwall[i]->shape_value(j,q) * velocity_vector_xwall[j][0]* spalding.enrichment_gradient(q);
            for (unsigned int j=0; j<velocity_vector_xwall.size(); ++j)
              dv += fe_values_xwall[i]->shape_grad(j,q) * velocity_vector_xwall[j][1]* spalding.enrichment(q)
                    + fe_values_xwall[i]->shape_value(j,q) * velocity_vector_xwall[j][1]* spalding.enrichment_gradient(q);

            Tensor<1,dim,double > n = gradwdist[q]/gradwdist[q].norm(); //normalized normal vector in r-s coordinates (element cordinates)
            double dudn2 = (du*n)*(du*n) + (dv*n)*(dv*n);
            dudn = std::sqrt(std::abs(dudn2));
            if(du[1]<0.)
              dudn *= -1.;
            for (unsigned int j=0; j<velocity_vector.size(); ++j)
              p += fe_values_p[i]->shape_value(j,q) * pressure_vector[j];

            double reduced_jacobian;
            if(dim==3)
              reduced_jacobian = fe_values[i]->jacobian(q)[2][2];
            else
              reduced_jacobian = 1.;

            double length_ele = reduced_jacobian * fe_values[i]->get_quadrature().weight(q);
            length += length_ele;
            dudy_w += dudn*length_ele;
            p_w += p*length_ele;

          }

          double n_cells_z_dir = (n_points_y_glob-1)/(n_points_y-1); // ASSUMED: n_cells_y_dir = n_cells_z_dir
          if(dim == 3)
          AssertThrow(std::abs(length - 4.5*h/n_cells_z_dir) < 1e-13,
                  ExcMessage("check quadrature code. length = " + patch::to_string(length) + " lz_ele = " + patch::to_string(4.5*h/n_cells_z_dir) + "(element-length in z direction)"));

          // find index within the x-values: first do a binary search to find
          // the next larger value of x in the list...
          const double x = fe_values[i]->quadrature_point(0)[0];
          unsigned int idx = std::distance(x_glob.begin(),
                                           std::lower_bound(x_glob.begin(), x_glob.end(),
                                                            x));
          // ..., then, check whether the point before was closer (but off
          // by 1e-13 or less)
          if (idx > 0 && std::abs(x_glob[idx-1]-x) < std::abs(x_glob[idx]-x))
            idx--;
          AssertThrow(std::abs(x_glob[idx]-x)<1e-13,
                      ExcMessage("Could not locate " + patch::to_string(x) + " among "
                                 "pre-evaluated points. Closest point is " +
                                 patch::to_string(x_glob[idx]) + " at distance " +
                                 patch::to_string(std::abs(x_glob[idx]-x))));

          dudy_w_loc.at(idx) += dudy_w;
          length_loc.at(idx) += length;
          p_w_loc.at(idx) += p_w;

        }
      }
      }
  // accumulate data over all processors overwriting the processor-local data
  // in xxx_loc
  Utilities::MPI::sum(dudy_w_loc, communicator, dudy_w_loc);
  Utilities::MPI::sum(length_loc, communicator, length_loc);
  Utilities::MPI::sum(p_w_loc, communicator, p_w_loc);

  // check quadrature
  for (unsigned int idx = 0; idx<x_glob.size(); idx++)
    if(dim == 3)
      AssertThrow(std::abs(length_loc.at(idx) - 4.5*h) < 1e-13 || std::abs(length_loc.at(idx) - 2*4.5*h) < 1e-13,
          ExcMessage("check quadrature code. length_loc.at(" + patch::to_string(idx) + ") = " + patch::to_string(length_loc.at(idx)) + " 4.5*h = " + patch::to_string(4.5*h)));

  for (unsigned int idx = 0; idx<x_glob.size(); idx++)
  {
    dudy_bottom_glob.at(idx) += dudy_w_loc[idx]/length_loc[idx];
    p_bottom_glob.at(idx) += p_w_loc[idx]/length_loc[idx];
  }



// on the TOP
  std::vector<double> length_top_loc(dudy_top_glob.size(),0.0);
  std::vector<double> dudy_top_loc(dudy_top_glob.size(),0.0);
  std::vector<double> p_top_loc(p_top_glob.size(),0.0);

  std::vector<std::shared_ptr<FEValues<dim,dim> > > fe_values_top(n_points_x);
  std::vector<std::shared_ptr<FEValues<dim,dim> > > fe_values_p_top(n_points_x);
  std::vector<std::shared_ptr<FEValues<dim,dim> > > fe_values_xwall_top(n_points_x);
  std::vector<std::shared_ptr<FEValues<dim,dim> > > fe_values_tauw_top(n_points_x);

  for (unsigned int i=0; i<n_points_x; ++i)
  {
    std::vector<Point<dim> > points(gauss_1d.size());
    std::vector<double> weights(gauss_1d.size());
    if(dim == 2)
    {
      points.resize(1);
      weights.resize(1);
    }
    for (unsigned int j=0; j<weights.size(); ++j)
    {
      points[j][0] = (double)i/(n_points_x-1);  // linear distributed form 0 to 1
      points[j][1] = 1.0;                       // assumed to be the top of the cells
      if(dim==3)
      {
        points[j][2] = gauss_1d.point(j)[0];
        weights[j] = gauss_1d.weight(j);
      }
      else
        weights[j] = 1.;
    }
    fe_values_top[i].reset(new FEValues<dim>(mapping_,
                                         dof_handler.get_fe().base_element(0),
                                         Quadrature<dim>(points, weights),
                                         update_values | update_jacobians |
                                         update_quadrature_points | update_gradients));
    fe_values_xwall_top[i].reset(new FEValues<dim>(mapping_,
                                         dof_handler.get_fe().base_element(1),
                                         Quadrature<dim>(points, weights),
                                         update_values | update_jacobians |
                                         update_quadrature_points | update_gradients));
    fe_values_p_top[i].reset(new FEValues<dim>(mapping_,
                                         dof_handler_p.get_fe(),
                                         Quadrature<dim>(points, weights),
                                         update_values | update_jacobians |
                                         update_quadrature_points | update_gradients));
    fe_values_tauw_top[i].reset(new FEValues<dim>(mapping_,
                                         dof_handler_wdist.get_fe(),
                                         Quadrature<dim>(points, weights),
                                         update_values | update_jacobians |
                                         update_quadrature_points | update_gradients));
  }

  std::vector<double > pressure_vector_top(scalar_dofs_per_cell); // \vec{p}^e_h
  std::vector<Tensor<1,dim> > velocity_vector_xwall_top(scalar_dofs_per_cell_xwall);
  std::vector<Tensor<1,1> > tauw_vector_top(scalar_dofs_per_cell_tauw);
  std::vector<Tensor<1,1> > wdist_vector_top(scalar_dofs_per_cell_tauw);
  cell_p=dof_handler_p.begin_active();
  cell_tauw=dof_handler_wdist.begin_active();
  for (typename DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active(); cell!=dof_handler.end(); ++cell, ++cell_p, ++cell_tauw)
    if (cell->is_locally_owned())
      {
      if (cell->at_boundary(3)) // just evaluate cells at the bottom of the domain
      {
        cell->get_dof_indices(dof_indices);
        cell_p->get_dof_indices(dof_indices_p);
        cell_tauw->get_dof_indices(dof_indices_tauw);
        // vector-valued FE where all components are explicitly listed in the
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

        for (unsigned int j=0; j<dof_indices_p.size(); ++j)
          {
            pressure_vector_top[j] = pressure(dof_indices_p[j]);
          }

        for (unsigned int i=0; i<n_points_x; ++i)
        {
          fe_values_top[i]->reinit(typename Triangulation<dim>::active_cell_iterator(cell));
          fe_values_p_top[i]->reinit(typename Triangulation<dim>::active_cell_iterator(cell));
          fe_values_xwall_top[i]->reinit(typename Triangulation<dim>::active_cell_iterator(cell));
          fe_values_tauw_top[i]->reinit(typename Triangulation<dim>::active_cell_iterator(cell));

          double length = 0, dudy_w = 0, p_w = 0;
          AlignedVector<double > wdist(fe_values_top[i]->n_quadrature_points,0.);
          AlignedVector<double > tauw(fe_values_top[i]->n_quadrature_points,0.);
          AlignedVector<Tensor<1,dim,double > > gradwdist(fe_values_top[i]->n_quadrature_points);
          AlignedVector<Tensor<1,dim,double > > gradtauw(fe_values_top[i]->n_quadrature_points);
          for (unsigned int q=0; q<fe_values_top[i]->n_quadrature_points; ++q)
          {
            for (unsigned int j=0; j<wdist_vector.size(); ++j)
              wdist[q] += (fe_values_tauw_top[i]->shape_value(j,q) * wdist_vector[j])[0];
            for (unsigned int j=0; j<tauw_vector.size(); ++j)
              tauw[q] += (fe_values_tauw_top[i]->shape_value(j,q) * tauw_vector[j])[0];

            for (unsigned int j=0; j<tauw_vector.size(); ++j)
              gradtauw[q] += fe_values_tauw_top[i]->shape_grad(j,q) * tauw_vector[j][0];
            for (unsigned int j=0; j<wdist_vector.size(); ++j)
              gradwdist[q] += fe_values_tauw_top[i]->shape_grad(j,q) * wdist_vector[j][0];
          }
          SpaldingsLawEvaluation<dim, double, double > spalding(fe_param.viscosity);
          std::vector<bool> enrcomp(1);
          enrcomp[0] = true;
          spalding.reinit(wdist,tauw,gradwdist,gradtauw,fe_values_top[i]->n_quadrature_points,enrcomp);

          for (unsigned int q=0; q<fe_values_top[i]->n_quadrature_points; ++q)
          {
            // calculate dudy at the quadrature point
            double dudy = 0.0;
            double p = 0.;
            for (unsigned int j=0; j<velocity_vector.size(); ++j)
              dudy += fe_values_top[i]->shape_grad(j,q)[1] * velocity_vector[j][0];
            for (unsigned int j=0; j<velocity_vector_xwall.size(); ++j)
              dudy += fe_values_xwall_top[i]->shape_grad(j,q)[1] * velocity_vector_xwall[j][0]*spalding.enrichment(q)
                    + fe_values_xwall_top[i]->shape_value(j,q) * velocity_vector_xwall[j][0]*spalding.enrichment_gradient(q)[1];

            for (unsigned int j=0; j<velocity_vector.size(); ++j)
              p += fe_values_p_top[i]->shape_value(j,q) * pressure_vector_top[j];

            double reduced_jacobian;
            if(dim==3)
              reduced_jacobian = fe_values_top[i]->jacobian(q)[2][2];
            else
              reduced_jacobian = 1.;

            double length_ele = reduced_jacobian * fe_values_top[i]->get_quadrature().weight(q);
            length += length_ele;
            dudy_w += dudy*length_ele;
            p_w += p*length_ele;

          }

          double n_cells_z_dir = (n_points_y_glob-1)/(n_points_y-1); // ASSUMED: n_cells_y_dir = n_cells_z_dir
          if(dim == 3)
            AssertThrow(std::abs(length - 4.5*h/n_cells_z_dir) < 1e-13,
                    ExcMessage("check quadrature code. length = " + patch::to_string(length) + " lz_ele = " + patch::to_string(4.5*h/n_cells_z_dir) + "(element-length in z direction)"));

          // find index within the x-values: first do a binary search to find
          // the next larger value of x in the list...
          const double x = fe_values_top[i]->quadrature_point(0)[0];
          unsigned int idx = std::distance(x_glob.begin(),
                                           std::lower_bound(x_glob.begin(), x_glob.end(),
                                                            x));
          // ..., then, check whether the point before was closer (but off
          // by 1e-13 or less)
          if (idx > 0 && std::abs(x_glob[idx-1]-x) < std::abs(x_glob[idx]-x))
            idx--;
          AssertThrow(std::abs(x_glob[idx]-x)<1e-13,
                      ExcMessage("Could not locate " + patch::to_string(x) + " among "
                                 "pre-evaluated points. Closest point is " +
                                 patch::to_string(x_glob[idx]) + " at distance " +
                                 patch::to_string(std::abs(x_glob[idx]-x))));

          dudy_top_loc.at(idx) += dudy_w;
          length_top_loc.at(idx) += length;
          p_top_loc.at(idx) += p_w;
        }
      }
      }
  // accumulate data over all processors overwriting the processor-local data
  // in xxx_loc
  Utilities::MPI::sum(dudy_top_loc, communicator, dudy_top_loc);
  Utilities::MPI::sum(length_top_loc, communicator, length_top_loc);
  Utilities::MPI::sum(p_top_loc, communicator, p_top_loc);

  // check quadrature
  for (unsigned int idx = 0; idx<x_glob.size(); idx++)
    if(dim == 3)
      AssertThrow(std::abs(length_top_loc.at(idx) - 4.5*h) < 1e-13 || std::abs(length_top_loc.at(idx) - 2*4.5*h) < 1e-13,
          ExcMessage("check quadrature code. length_loc.at(" + patch::to_string(idx) + ") = " + patch::to_string(length_top_loc.at(idx)) + " 4.5*h = " + patch::to_string(4.5*h)));

  for (unsigned int idx = 0; idx<x_glob.size(); idx++)
  {
    dudy_top_glob.at(idx) += dudy_top_loc[idx]/length_top_loc[idx];
    p_top_glob.at(idx) += p_top_loc[idx]/length_top_loc[idx];
  }

  numchsamp++;
}


// explicit instantiation
template class StatisticsManagerPH<2>;
template class StatisticsManagerPH<3>;
