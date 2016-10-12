/*
 * PressureDifferenceData.h
 *
 *  Created on: Oct 11, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_PRESSUREDIFFERENCEDATA_H_
#define INCLUDE_PRESSUREDIFFERENCEDATA_H_


template<int dim>
struct PressureDifferenceData
{
  PressureDifferenceData()
    :
    calculate_pressure_difference(false),
    filename_prefix_pressure_difference("indexa")
  {}

  /*
   *  active or not
   */
  bool calculate_pressure_difference;

  /*
   *  Points:
   *  calculation of pressure difference: p(point_1) - p(point_2)
   */
  Point<dim> point_1;
  Point<dim> point_2;

  /*
   *  filenames
   */
  std::string filename_prefix_pressure_difference;
};


#endif /* INCLUDE_PRESSUREDIFFERENCEDATA_H_ */
