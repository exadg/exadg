/*
 * TimeIntBDFBase.h
 *
 *  Created on: Aug 22, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_TIMEINTBDFBASE_H_
#define INCLUDE_TIMEINTBDFBASE_H_


class TimeIntBDFBase
{
public:
  TimeIntBDFBase(unsigned int const order_time_integrator,
                 bool const         start_with_low_order_method,
                 bool const         use_adaptive_time_stepping)
    :
    time_step_number(1),
    order(order_time_integrator),
    start_with_low_order(start_with_low_order_method),
    adaptive_time_stepping(use_adaptive_time_stepping),
    time_steps(order),
    gamma0(1.0),
    alpha(order),
    beta(order)
  {}

  virtual ~TimeIntBDFBase(){}

  virtual void setup(bool do_restart = false) = 0;

  virtual void timeloop() = 0;

protected:

  /*
   *  This function initializes the time integrator constants. The default case is
   *  start_with_low_order = false.
   */
  void initialize_time_integrator_constants();

  /*
   *  This function updates the time integrator constants which is necessary when
   *  using start_with_low_order = true or when using adaptive time stepping
   */
  virtual void update_time_integrator_constants();

  /*
   *  This function calculates time integrator constants
   *  in case of varying time step sizes (adaptive time stepping).
   */
  void set_adaptive_time_integrator_constants(unsigned int const              current_order,
                                              std::vector<double> const     & time_steps,
                                              std::vector<double>           & alpha,
                                              std::vector<double>           & beta,
                                              double                        & gamma0);

  // the number of the current time step starting with time_step_number = 1
  unsigned int time_step_number;

  // order of time integrator
  unsigned int const order;

  // use a low order time integration scheme to start the time integrator?
  bool const start_with_low_order;

  // use adaptive time stepping?
  bool const adaptive_time_stepping;

  // Vector that stores time step sizes. This vector is necessary
  // if adaptive_time_stepping = true. For constant time step sizes
  // one double for the time step size would be sufficient.
  std::vector<double> time_steps;

  // time integrator constants: discrete time derivative term
  double gamma0;
  std::vector<double> alpha;

  // time integrator constants: extrapolation scheme
  std::vector<double> beta;

private:
  /*
   *  This function calculates the time integrator constants
   *  in case of constant time step sizes.
   */
  void set_constant_time_integrator_constants (unsigned int const current_order);

  /*
   *  This function prints the time integrator constants in order to check the
   *  correctness of the time integrator constants.
   */
  void check_time_integrator_constants (unsigned int const number) const;

};


void TimeIntBDFBase::
initialize_time_integrator_constants()
{
  AssertThrow(order == 1 || order == 2 || order == 3 || order == 4,
      ExcMessage("Specified order of time integration scheme is not implemented."));

  // the default case is start_with_low_order == false
  set_constant_time_integrator_constants(order);
}


void TimeIntBDFBase::
set_constant_time_integrator_constants (unsigned int const current_order)
{
  AssertThrow(current_order <= order,
      ExcMessage("There is a logical error when updating the time integrator constants."));

  if(current_order == 1)   //BDF 1
  {
    gamma0 = 1.0;

    alpha[0] = 1.0;

    beta[0] = 1.0;
  }
  else if(current_order == 2) //BDF 2
  {
    gamma0 = 3.0/2.0;

    alpha[0] = 2.0;
    alpha[1] = -0.5;

    beta[0] = 2.0;
    beta[1] = -1.0;
  }
  else if(current_order == 3) //BDF 3
  {
    gamma0 = 11./6.;

    alpha[0] = 3.;
    alpha[1] = -1.5;
    alpha[2] = 1./3.;

    beta[0] = 3.0;
    beta[1] = -3.0;
    beta[2] = 1.0;
  }
  else if(current_order == 4) // BDF 4
  {
    gamma0 = 25./12.;

    alpha[0] = 4.;
    alpha[1] = -3.;
    alpha[2] = 4./3.;
    alpha[3] = -1./4.;

    beta[0] = 4.;
    beta[1] = -6.;
    beta[2] = 4.;
    beta[3] = -1.;
  }

  /*
   * Fill the rest of the vectors with zeros since current_order might be
   * smaller than order, e.g. when using start_with_low_order = true
   */
  for(unsigned int i=current_order;i<order;++i)
  {
    alpha[i] = 0.0;
    beta[i] = 0.0;
  }
}

void TimeIntBDFBase::
set_adaptive_time_integrator_constants (unsigned int const              current_order,
                                        std::vector<double> const     & time_steps,
                                        std::vector<double>           & alpha,
                                        std::vector<double>           & beta,
                                        double                        & gamma0)
{
  AssertThrow(current_order <= order,
    ExcMessage("There is a logical error when updating the time integrator constants."));

  if(current_order == 1)   // BDF 1
  {
    gamma0 = 1.0;

    alpha[0] = 1.0;

    beta[0] = 1.0;
  }
  else if(current_order == 2) // BDF 2
  {
    gamma0 = (2*time_steps[0]+time_steps[1])/(time_steps[0]+time_steps[1]);

    alpha[0] = (time_steps[0]+time_steps[1])/time_steps[1];
    alpha[1] = - time_steps[0]*time_steps[0]/((time_steps[0]+time_steps[1])*time_steps[1]);

    beta[0] = (time_steps[0]+time_steps[1])/time_steps[1];
    beta[1] = -time_steps[0]/time_steps[1];
  }
  else if(current_order == 3) // BDF 3
  {
    gamma0 = 1.0 + time_steps[0]/(time_steps[0]+time_steps[1])
                 + time_steps[0]/(time_steps[0]+time_steps[1]+time_steps[2]);

    alpha[0] = +(time_steps[0]+time_steps[1])*(time_steps[0]+time_steps[1]+time_steps[2])/
                (time_steps[1]*(time_steps[1]+time_steps[2]));
    alpha[1] = -time_steps[0]*time_steps[0]*(time_steps[0]+time_steps[1]+time_steps[2])/
                ((time_steps[0]+time_steps[1])*time_steps[1]*time_steps[2]);
    alpha[2] = +time_steps[0]*time_steps[0]*(time_steps[0]+time_steps[1])/
                ((time_steps[0]+time_steps[1]+time_steps[2])*(time_steps[1]+time_steps[2])*time_steps[2]);

    beta[0] = +(time_steps[0]+time_steps[1])*(time_steps[0]+time_steps[1]+time_steps[2])/
               (time_steps[1]*(time_steps[1]+time_steps[2]));
    beta[1] = -time_steps[0]*(time_steps[0]+time_steps[1]+time_steps[2])/
               (time_steps[1]*time_steps[2]);
    beta[2] = +time_steps[0]*(time_steps[0]+time_steps[1])/
               ((time_steps[1]+time_steps[2])*time_steps[2]);
  }
  else if(current_order == 4) // BDF 4
  {
    gamma0 = 1.0 + time_steps[0]/(time_steps[0]+time_steps[1])
                 + time_steps[0]/(time_steps[0]+time_steps[1]+time_steps[2])
                 + time_steps[0]/(time_steps[0]+time_steps[1]+time_steps[2]+time_steps[3]);

    alpha[0] = (time_steps[0]+time_steps[1])*(time_steps[0]+time_steps[1]+time_steps[2])*
               (time_steps[0]+time_steps[1]+time_steps[2]+time_steps[3]) /
               ( time_steps[1]*(time_steps[1]+time_steps[2])*(time_steps[1]+time_steps[2]+time_steps[3]) );

    alpha[1] = - time_steps[0]*time_steps[0]*(time_steps[0]+time_steps[1]+time_steps[2])
                 *(time_steps[0]+time_steps[1]+time_steps[2]+time_steps[3]) /
                 ( (time_steps[0]+time_steps[1])*time_steps[1]*time_steps[2]*(time_steps[2]+time_steps[3]) );

    alpha[2] = time_steps[0]*time_steps[0]*(time_steps[0]+time_steps[1])*
               (time_steps[0]+time_steps[1]+time_steps[2]+time_steps[3]) /
               ( (time_steps[0]+time_steps[1]+time_steps[2])*(time_steps[1]+time_steps[2])*time_steps[2]*time_steps[3] );

    alpha[3] = - time_steps[0]*time_steps[0]*(time_steps[0]+time_steps[1])*
                 (time_steps[0]+time_steps[1]+time_steps[2]) /
                 ( (time_steps[0]+time_steps[1]+time_steps[2]+time_steps[3])*(time_steps[1]+time_steps[2]+time_steps[3])*
                    (time_steps[2]+time_steps[3])*time_steps[3] );

    beta[0] = (time_steps[0]+time_steps[1])*
              (time_steps[0]+time_steps[1]+time_steps[2])*
              (time_steps[0]+time_steps[1]+time_steps[2]+time_steps[3]) /
              ( time_steps[1]*(time_steps[1]+time_steps[2])*(time_steps[1]+time_steps[2]+time_steps[3]) );

    beta[1] = - time_steps[0]*
                (time_steps[0]+time_steps[1]+time_steps[2])*
                (time_steps[0]+time_steps[1]+time_steps[2]+time_steps[3]) /
                ( time_steps[1]*time_steps[2]*(time_steps[2]+time_steps[3]) );

    beta[2] = time_steps[0]*
              (time_steps[0]+time_steps[1])*
              (time_steps[0]+time_steps[1]+time_steps[2]+time_steps[3]) /
              ( (time_steps[1]+time_steps[2])*time_steps[2]*time_steps[3] );

    beta[3] = - time_steps[0]*
                (time_steps[0]+time_steps[1])*
                (time_steps[0]+time_steps[1]+time_steps[2]) /
                ( (time_steps[1]+time_steps[2]+time_steps[3])*(time_steps[2]+time_steps[3])*time_steps[3] );
  }

  /*
   * Fill the rest of the vectors with zeros since current_order might be
   * smaller than order, e.g. when using start_with_low_order = true
   */
  for(unsigned int i=current_order;i<order;++i)
  {
    alpha[i] = 0.0;
    beta[i] = 0.0;
  }
}

void TimeIntBDFBase::
update_time_integrator_constants()
{
  if(adaptive_time_stepping == true)
  {
    // when starting the time integrator with a low order method, ensure that
    // the time integrator constants are set properly
    if(time_step_number <= order && start_with_low_order == true)
    {
      set_adaptive_time_integrator_constants(time_step_number, time_steps, alpha, beta, gamma0);
    }
    else // otherwise, adjust time integrator constants since this is adaptive time stepping
    {
      set_adaptive_time_integrator_constants(order, time_steps, alpha, beta, gamma0);
    }
  }
  else // adaptive_time_stepping == false, i.e., constant time step sizes
  {
    // when starting the time integrator with a low order method, ensure that
    // the time integrator constants are set properly
    if(time_step_number <= order && start_with_low_order == true)
    {
      set_constant_time_integrator_constants(time_step_number);
    }
  }

//   check_time_integrator_constants(time_step_number);
}


void TimeIntBDFBase::
check_time_integrator_constants(unsigned int current_time_step_number) const
{
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << "Time integrator constants: time step " << current_time_step_number << std::endl;

    std::cout << "Gamma0   = " << gamma0 << std::endl;

    for(unsigned int i=0;i<order;++i)
      std::cout << "Alpha[" << i <<"] = " << alpha[i] << std::endl;

    for(unsigned int i=0;i<order;++i)
      std::cout << "Beta[" << i <<"]  = " << beta[i] << std::endl;
  }
}

#endif /* INCLUDE_TIMEINTBDFBASE_H_ */
