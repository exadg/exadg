/*
 * AnalyticalSolutionConvDiff.h
 *
 *  Created on: Oct 12, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_ANALYTICALSOLUTIONCONVDIFF_H_
#define INCLUDE_ANALYTICALSOLUTIONCONVDIFF_H_


template<int dim>
struct AnalyticalSolutionConvDiff
{
  std_cxx11::shared_ptr<Function<dim> > solution;
};


#endif /* INCLUDE_ANALYTICALSOLUTIONCONVDIFF_H_ */
