/*
 * explicit_runge_kutta.h
 *
 *  Created on: Jul 10, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_EXPLICIT_RUNGE_KUTTA_H_
#define INCLUDE_CONVECTION_DIFFUSION_EXPLICIT_RUNGE_KUTTA_H_

template<typename Operator, typename Vector>
class ExplicitRungeKuttaTimeIntegrator
{
public:
  // Constructor
  ExplicitRungeKuttaTimeIntegrator(unsigned int              order_time_integrator,
                                   std::shared_ptr<Operator> underlying_operator_in)
    :
    order(order_time_integrator),
    underlying_operator(underlying_operator_in)
  {
    // initialize vectors
    if(order >= 2)
      underlying_operator->initialize_dof_vector(vec_rhs);
    if(order >= 3)
      underlying_operator->initialize_dof_vector(vec_temp);
  }

  void solve_timestep(Vector       &dst,
                      Vector const &src,
                      double       time,
                      double       time_step)
  {
    if(order == 1) // explicit Euler method
    {
      underlying_operator->evaluate(dst,src,time);
      dst *= time_step;
      dst.add(1.0,src);
    }
    else if(order == 2) // Runge-Kutta method of order 2
    {
      // stage 1
      underlying_operator->evaluate(vec_rhs,src,time);

      // stage 2
      vec_rhs *= time_step/2.;
      vec_rhs.add(1.0,src);
      underlying_operator->evaluate(dst,vec_rhs,time + time_step/2.);
      dst *= time_step;
      dst.add(1.0,src);
    }
    else if(order == 3) //Heun's method of order 3
    {
      dst = src;

      // stage 1
      underlying_operator->evaluate(vec_temp,src,time);
      dst.add(1.*time_step/4.,vec_temp);

      // stage 2
      vec_rhs.equ(1.,src);
      vec_rhs.add(time_step/3.,vec_temp);
      underlying_operator->evaluate(vec_temp,vec_rhs,time+time_step/3.);

      // stage 3
      vec_rhs.equ(1.,src);
      vec_rhs.add(2.0*time_step/3.0,vec_temp);
      underlying_operator->evaluate(vec_temp,vec_rhs,time+2.*time_step/3.);
      dst.add(3.*time_step/4.,vec_temp);
    }
    else if(order == 4) //classical 4th order Runge-Kutta method
    {
      dst = src;

      // stage 1
      underlying_operator->evaluate(vec_temp,src,time);
      dst.add(time_step/6., vec_temp);

      // stage 2
      vec_rhs.equ(1.,src);
      vec_rhs.add(time_step/2., vec_temp);
      underlying_operator->evaluate(vec_temp,vec_rhs,time+time_step/2.);
      dst.add(time_step/3., vec_temp);

      // stage 3
      vec_rhs.equ(1., src);
      vec_rhs.add(time_step/2., vec_temp);
      underlying_operator->evaluate(vec_temp,vec_rhs,time+time_step/2.);
      dst.add(time_step/3., vec_temp);

      // stage 4
      vec_rhs.equ(1., src);
      vec_rhs.add(time_step, vec_temp);
      underlying_operator->evaluate(vec_temp,vec_rhs,time+time_step);
      dst.add(time_step/6., vec_temp);
    }

    else
    {
      AssertThrow(order <= 4,ExcMessage("Explicit Runge-Kutta method only implemented for order <= 4!"));
    }
  }

private:
  unsigned int order;
  std::shared_ptr<Operator> underlying_operator;

  Vector vec_rhs, vec_temp;
};


#endif /* INCLUDE_CONVECTION_DIFFUSION_EXPLICIT_RUNGE_KUTTA_H_ */
