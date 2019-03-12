/*
 * grid_functions_turbulent_channel.h
 *
 *  Created on: Apr 20, 2018
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_GRID_FUNCTIONS_TURBULENT_CHANNEL_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_GRID_FUNCTIONS_TURBULENT_CHANNEL_H_

/*
 *  maps eta in [0,1] --> y in [-1,1]*length_y/2.0 (using a hyperbolic mesh stretching)
 */
double grid_transform_y(const double &eta)
{
 double y = 0.0;

 if(GRID_STRETCH_FAC >= 0)
   y = DIMENSIONS_X2/2.0*std::tanh(GRID_STRETCH_FAC*(2.*eta-1.))/std::tanh(GRID_STRETCH_FAC);
 else // use a negative GRID_STRETCH_FACTOR deactivate grid stretching
   y = DIMENSIONS_X2/2.0*(2.*eta-1.);

 return y;
}

/*
* inverse mapping:
*
*  maps y in [-1,1]*length_y/2.0 --> eta in [0,1]
*/
double inverse_grid_transform_y(const double &y)
{
 double eta = 0.0;

 if(GRID_STRETCH_FAC >= 0)
   eta = (std::atanh(y*std::tanh(GRID_STRETCH_FAC)*2.0/DIMENSIONS_X2)/GRID_STRETCH_FAC+1.0)/2.0;
 else // use a negative GRID_STRETCH_FACTOR deactivate grid stretching
   eta = (2.*y/DIMENSIONS_X2+1.)/2.0;

 return eta;
}

template <int dim>
Point<dim> grid_transform (const Point<dim> &in)
{
 Point<dim> out = in;

 out[0] = in(0)-DIMENSIONS_X1/2.0;
 out[1] = grid_transform_y(in[1]);

 if(dim==3)
   out[2] = in(2)-DIMENSIONS_X3/2.0;

 return out;
}

#include <deal.II/grid/manifold_lib.h>

template <int dim>
class ManifoldTurbulentChannel : public ChartManifold<dim,dim,dim>
{
public:
 ManifoldTurbulentChannel(Tensor<1,dim> const &dimensions_in)
 {
   dimensions = dimensions_in;
 }

 /*
  *  push_forward operation that maps point xi in reference coordinates [0,1]^d to
  *  point x in physical coordinates
  */
 Point<dim> push_forward(const Point<dim> &xi) const
 {
   Point<dim> x;

   x[0] = xi[0]*dimensions[0]-dimensions[0]/2.0;
   x[1] = grid_transform_y(xi[1]);

   if(dim==3)
     x[2] = xi[2]*dimensions[2]-dimensions[2]/2.0;

   return x;
 }

 /*
  *  pull_back operation that maps point x in physical coordinates
  *  to point xi in reference coordinates [0,1]^d
  */
 Point<dim> pull_back(const Point<dim> &x) const
 {
   Point<dim> xi;

   xi[0] = x[0]/dimensions[0]+0.5;
   xi[1] = inverse_grid_transform_y(x[1]);

   if(dim==3)
     xi[2] = x[2]/dimensions[2]+0.5;

   return xi;
 }

 std::unique_ptr<Manifold<dim>>
 clone() const override
 {
   return std_cxx14::make_unique<ManifoldTurbulentChannel<dim>>(dimensions);
 }

private:
Tensor<1,dim> dimensions;
};



#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_GRID_FUNCTIONS_TURBULENT_CHANNEL_H_ */
