/*
 * cube_moving_manifold.h
 *
 *  Created on: Mai 13, 2019
 *      Author: heinz and fehn
 */

#ifndef APPLICATIONS_GRID_TOOLS_CUBE_MOVING_MANIFOLD_H_
#define APPLICATIONS_GRID_TOOLS_CUBE_MOVING_MANIFOLD_H_


template <int dim>
class CubeMovingManifold : public ChartManifold<dim,dim,dim>
{
public:
  CubeMovingManifold( double       *time,
                      const double       left,
                      const double       right,
                      const double       amplitude,
                      const double     delta_t,
                      const double frequency = 1.0)
    :
    time(time),
    left(left),
    right(right),
    amplitude(amplitude),
    delta_t(delta_t),
    frequency(frequency)
  {
  }

  //TODO: Change Formulation to cover both, 2 and 3D.

  Point<dim> push_forward(const Point<dim> &chart_point) const
    {

      Point<dim> space_point;
      Point<dim> displacement;

      displacement(0)=std::sin(2* numbers::PI*(chart_point(1)-left)/width)*sin_t*amplitude;
      displacement(1)=std::sin(2* numbers::PI*(chart_point(0)-left)/width)*sin_t*amplitude;

      //Damping---------------------------------------
      //linear
      //double damp0 = (1- std::abs(chart_point(0))/right );
      //double damp1 = (1- std::abs(chart_point(1))/right );
      //quadratic
      double damp0 = (1- std::pow(chart_point(0)/right,2));
      double damp1 = (1- std::pow(chart_point(1)/right,2));
      //----------------------------------------------

      displacement(0) = displacement(0) *damp0;
      displacement(1) = displacement(1) *damp1;

      space_point = chart_point+ displacement;

     return space_point;

    }

    Point<dim> pull_back(const Point<dim> &space_point) const
    {

      Point<dim> X = space_point;

      Tensor<1,dim> R; //Residual

      //Damping---------------------------------------
      //linear
      //double damp0 = (1- std::abs(X(0))/right );
      //double damp1 = (1- std::abs(X(1))/right );
      //quadratic
      double damp0 = (1- std::pow(X(0)/right,2) );
      double damp1 = (1- std::pow(X(1)/right,2) );
      //----------------------------------------------

        R[0] = space_point(0)- X(0)- amplitude*sin_t*std::sin(2*numbers::PI*(X(1)-left)/width)*damp0;
        R[1] = space_point(1)- X(1)- amplitude*sin_t*std::sin(2*numbers::PI*(X(0)-left)/width)*damp1;


      unsigned int its = 0;
      while (R.norm() > 1e-12 && its < 100)
      {

        //Damping---------------------------------------
        //linear
        //double damp0 = (1- std::abs(X(0))/right );
        //double damp1 = (1- std::abs(X(1))/right );
        //double damp0_dX0 =(-X(0)/std::abs(X(0)) *right);
        //double damp1_dX1 =(-X(1)/std::abs(X(0)) *right);
        //quadratic
        double damp0 = (1- std::pow(X(0)/right,2) );
        double damp1 = (1- std::pow(X(1)/right,2) );
        double damp0_dX0 =(-2*X(0)/std::pow(right,2));
        double damp1_dX1 =(-2*X(1)/std::pow(right,2));
        //----------------------------------------------

        Tensor<2,dim> J;

        J[0][0]= - 1 - amplitude*sin_t*std::sin(2*numbers::PI*(X(1)-left)/width) * damp0_dX0; //dR0/dX0
        J[0][1]= - amplitude*sin_t*2*numbers::PI/width* std::cos(2*numbers::PI*(X(1)-left)/width)*damp0; //dR0/dX1
        J[1][0]= - amplitude*sin_t*2*numbers::PI/width* std::cos(2*numbers::PI*(X(0)-left)/width)*damp1; //dR1/dX0
        J[1][1]= - 1 - amplitude*sin_t*std::sin(2*numbers::PI*(X(0)-left)/width) * damp1_dX1; //dR1/dX1

        Tensor<1,dim> D; //Displacement
        D = invert(J)*-1*R;
        X+=D;

        R[0] = space_point(0)- X(0)- amplitude*sin_t*std::sin(2*numbers::PI*(X(1)-left)/width)*damp0;
        R[1] = space_point(1)- X(1)- amplitude*sin_t*std::sin(2*numbers::PI*(X(0)-left)/width)*damp1;


        ++its;
      }
      AssertThrow (R.norm() < 1e-12,
                   ExcMessage("Newton for point did not converge."));

      return X;


    }

  std::unique_ptr<Manifold<dim>>
  clone() const override
  {
    return std_cxx14::make_unique<CubeMovingManifold<dim>>(time,left,right,amplitude,delta_t,frequency);
  }

private:

  double *time;
  const double left;
  const double right;
  const double amplitude;
  const double delta_t;
  const double frequency;
  const double width = right-left;
  const double T = delta_t/frequency; //duration of a period
  const double t = *(time);
  const double sin_t=std::pow(std::sin(2*numbers::PI*t/T),2);
  //const double sin_t = std::sin(2*numbers::PI*t/T);


};

#endif /* APPLICATIONS_GRID_TOOLS_CUBE_MOVING_MANIFOLD_H_ */
