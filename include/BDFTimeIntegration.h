/*
 * BDFTimeIntegration.h
 *
 *  Created on: Jan 30, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_BDFTIMEINTEGRATION_H_
#define INCLUDE_BDFTIMEINTEGRATION_H_


class BDFTimeIntegratorConstants
{
public:
  BDFTimeIntegratorConstants(unsigned int const order_time_integrator,
                             bool const         start_with_low_order_method)
    :
    order(order_time_integrator),
    start_with_low_order(start_with_low_order_method),
    gamma0(1.0),
    alpha(order)
  {
    AssertThrow(order >= 1 && order <= 4, ExcMessage("Specified order of BDF scheme not implemented."));
  }

  double get_gamma0() const
  {
    return gamma0;
  }

  double get_alpha(unsigned int const i) const
  {
    AssertThrow(i < order,
        ExcMessage("In order to access BDF time integrator constants the index has to be smaller than the order of the time integration scheme."));

    return alpha[i];
  }

  /*
   *  This function initializes the time integrator constants.
   */
  void initialize();

  /*
   *  This function updates the time integrator constants of the BDF scheme
   *  in case of constant time step sizes.
   */
  void update(unsigned int const time_step_number);

  /*
   *  This function updates the time integrator constants of the BDF scheme
   *  in case of adaptive time step sizes.
   */
  void update(unsigned int const        time_step_number,
              std::vector<double> const &time_steps);

  /*
   *  This function prints the time integrator constants
   */
  void print() const;


private:
  /*
   *  This function calculates the time integrator constants of the BDF scheme
   *  in case of constant time step sizes.
   */
  void set_constant_time_step(unsigned int const  current_order);

  /*
   *  This function calculates time integrator constants
   *  in case of varying time step sizes (adaptive time stepping).
   */
  void set_adaptive_time_step(unsigned int const        current_order,
                              std::vector<double> const &time_steps);

  // order of time integrator
  unsigned int const order;

  // use a low order time integration scheme to start the time integrator?
  bool const start_with_low_order;

  /*
   *  BDF time integrator constants:
   *
   *  du/dt = (gamma_0 u^{n+1} - alpha_0 u^{n} - alpha_1 u^{n-1} ... - alpha_{J-1} u^{n-J+1})/dt
   */
  double gamma0;
  std::vector<double> alpha;
};



void BDFTimeIntegratorConstants::
set_constant_time_step(unsigned int const current_order)
{
  AssertThrow(current_order <= order,
      ExcMessage("There is a logical error when updating the BDF time integrator constants."));

  if(current_order == 1)   //BDF 1
  {
    gamma0 = 1.0;

    alpha[0] = 1.0;
  }
  else if(current_order == 2) //BDF 2
  {
    gamma0 = 3.0/2.0;

    alpha[0] = 2.0;
    alpha[1] = -0.5;
  }
  else if(current_order == 3) //BDF 3
  {
    gamma0 = 11./6.;

    alpha[0] = 3.;
    alpha[1] = -1.5;
    alpha[2] = 1./3.;
  }
  else if(current_order == 4) // BDF 4
  {
    gamma0 = 25./12.;

    alpha[0] = 4.;
    alpha[1] = -3.;
    alpha[2] = 4./3.;
    alpha[3] = -1./4.;
  }

  /*
   * Fill the rest of the vectors with zeros since current_order might be
   * smaller than order, e.g., when using start_with_low_order = true
   */
  for(unsigned int i=current_order;i<order;++i)
  {
    alpha[i] = 0.0;
  }
}


void BDFTimeIntegratorConstants::
set_adaptive_time_step (unsigned int const        current_order,
                        std::vector<double> const &time_steps)
{
  AssertThrow(current_order <= order,
    ExcMessage("There is a logical error when updating the time integrator constants."));

  AssertThrow(time_steps.size() == order,
    ExcMessage("Length of vector containing time step sizes has to be equal to order of time integration scheme."));

  if(current_order == 1)   // BDF 1
  {
    gamma0 = 1.0;

    alpha[0] = 1.0;
  }
  else if(current_order == 2) // BDF 2
  {
    gamma0 = (2*time_steps[0]+time_steps[1])/(time_steps[0]+time_steps[1]);

    alpha[0] = (time_steps[0]+time_steps[1])/time_steps[1];
    alpha[1] = - time_steps[0]*time_steps[0]/((time_steps[0]+time_steps[1])*time_steps[1]);
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
  }

  /*
   * Fill the rest of the vectors with zeros since current_order might be
   * smaller than order, e.g. when using start_with_low_order = true
   */
  for(unsigned int i=current_order;i<order;++i)
  {
    alpha[i] = 0.0;
  }
}

void BDFTimeIntegratorConstants::
initialize()
{
  // The default case is start_with_low_order = false.
  set_constant_time_step(order);
}


void BDFTimeIntegratorConstants::
update(unsigned int const time_step_number)
{
  // when starting the time integrator with a low order method, ensure that
  // the time integrator constants are set properly
  if(time_step_number <= order && start_with_low_order == true)
  {
    set_constant_time_step(time_step_number);
  }
}

void BDFTimeIntegratorConstants::
update(unsigned int const        time_step_number,
       std::vector<double> const &time_steps)
{
  // when starting the time integrator with a low order method, ensure that
  // the time integrator constants are set properly
  if(time_step_number <= order && start_with_low_order == true)
  {
    set_adaptive_time_step(time_step_number, time_steps);
  }
  else // adjust time integrator constants since this is adaptive time stepping
  {
    set_adaptive_time_step(order, time_steps);
  }
}

void BDFTimeIntegratorConstants::
print() const
{
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << "Gamma0   = " << gamma0 << std::endl;

    for(unsigned int i=0;i<order;++i)
      std::cout << "Alpha[" << i <<"] = " << alpha[i] << std::endl;
  }
}


#endif /* INCLUDE_BDFTIMEINTEGRATION_H_ */
