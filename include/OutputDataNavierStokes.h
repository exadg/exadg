/*
 * OutputDataNavierStokes.h
 *
 *  Created on: Oct 12, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_OUTPUTDATANAVIERSTOKES_H_
#define INCLUDE_OUTPUTDATANAVIERSTOKES_H_

#include "../include/OutputData.h"

struct OutputDataNavierStokes : public OutputData
{
  OutputDataNavierStokes()
    :
    compute_divergence(false)
  {}

  void print(ConditionalOStream &pcout, bool unsteady)
  {
    OutputData::print(pcout,unsteady);

    print_parameter(pcout,"Compute divergence",compute_divergence);
  }

  // compute divergence of velocity field
  bool compute_divergence;

};

#endif /* INCLUDE_OUTPUTDATANAVIERSTOKES_H_ */
