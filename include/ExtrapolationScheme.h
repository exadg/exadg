/*
 * ExtrapolationScheme.h
 *
 *  Created on: Jan 30, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_EXTRAPOLATIONSCHEME_H_
#define INCLUDE_EXTRAPOLATIONSCHEME_H_


class ExtrapolationConstants
{
public:
  ExtrapolationConstants(unsigned int const order_extrapolation_scheme,
                         bool const         start_with_low_order_method)
    :
    order(order_extrapolation_scheme),
    start_with_low_order(start_with_low_order_method),
    beta(order)
  {
    AssertThrow(order <= 4, ExcMessage("Specified order of extrapolation scheme not implemented."));
  }

  double get_beta(unsigned int const i) const
  {
    AssertThrow(i < order,
        ExcMessage("In order to access constants of extrapolation scheme the index has to be smaller than the order of the scheme."));

    return beta[i];
  }

  unsigned int get_order()
  {
    return order;
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
   *  This function calculates constants of extrapolation scheme
   *  in case of constant time step sizes.
   */
  void set_constant_time_step(unsigned int const  current_order);

  /*
   *  This function calculates constants of extrapolation scheme
   *  in case of varying time step sizes (adaptive time stepping).
   */
  void set_adaptive_time_step(unsigned int const        current_order,
                              std::vector<double> const &time_steps);

  /*
   *  order of extrapolation scheme
   */
  unsigned int const order;

  // use a low order scheme in the first time steps?
  bool const start_with_low_order;

  /*
   *  Constants of extrapolation scheme
   */
  std::vector<double> beta;
};



void ExtrapolationConstants::
set_constant_time_step(unsigned int const current_order)
{
  AssertThrow(current_order <= order,
      ExcMessage("There is a logical error when updating the constants of the extrapolation scheme."));

  if(current_order == 1) // EX 1
  {
    beta[0] = 1.0;
  }
  else if(current_order == 2) // EX 2
  {
    beta[0] = 2.0;
    beta[1] = -1.0;
  }
  else if(current_order == 3) // EX 3
  {
    beta[0] = 3.0;
    beta[1] = -3.0;
    beta[2] = 1.0;
  }
  else if(current_order == 4) // EX 4
  {
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
    beta[i] = 0.0;
  }
}


void ExtrapolationConstants::
set_adaptive_time_step (unsigned int const        current_order,
                        std::vector<double> const &time_steps)
{
  AssertThrow(current_order <= order,
    ExcMessage("There is a logical error when updating the constants of the extrapolation scheme."));

  AssertThrow(time_steps.size() == order,
    ExcMessage("Length of vector containing time step sizes has to be equal to order of extrapolation scheme."));

  if(current_order == 1)   // EX 1
  {
    beta[0] = 1.0;
  }
  else if(current_order == 2) // EX 2
  {
    beta[0] = (time_steps[0]+time_steps[1])/time_steps[1];
    beta[1] = -time_steps[0]/time_steps[1];
  }
  else if(current_order == 3) // EX 3
  {
    beta[0] = +(time_steps[0]+time_steps[1])*(time_steps[0]+time_steps[1]+time_steps[2])/
               (time_steps[1]*(time_steps[1]+time_steps[2]));
    beta[1] = -time_steps[0]*(time_steps[0]+time_steps[1]+time_steps[2])/
               (time_steps[1]*time_steps[2]);
    beta[2] = +time_steps[0]*(time_steps[0]+time_steps[1])/
               ((time_steps[1]+time_steps[2])*time_steps[2]);
  }
  else if(current_order == 4) // EX 4
  {
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
   * smaller than order, e.g. when using start_with_low_order = true,
   * if current_order == 0, all coefficients are set to zero, i.e., no extrapolation
   */
  for(unsigned int i=current_order;i<order;++i)
  {
    beta[i] = 0.0;
  }
}

void ExtrapolationConstants::
initialize()
{
  // The default case is start_with_low_order = false.
  set_constant_time_step(order);
}


void ExtrapolationConstants::
update(unsigned int const current_order)
{
  // when starting the time integrator with a low order method, ensure that
  // the time integrator constants are set properly
  if(current_order <= order && start_with_low_order == true)
  {
    set_constant_time_step(current_order);
  }
}

void ExtrapolationConstants::
update(unsigned int const        current_order,
       std::vector<double> const &time_steps)
{
  // when starting the time integrator with a low order method, ensure that
  // the time integrator constants are set properly
  if(current_order <= order && start_with_low_order == true)
  {
    set_adaptive_time_step(current_order, time_steps);
  }
  else // adjust time integrator constants since this is adaptive time stepping
  {
    set_adaptive_time_step(order, time_steps);
  }
}

void ExtrapolationConstants::
print() const
{
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    for(unsigned int i=0;i<order;++i)
      std::cout << "Beta[" << i <<"]  = " << beta[i] << std::endl;
  }
}


#endif /* INCLUDE_EXTRAPOLATIONSCHEME_H_ */
