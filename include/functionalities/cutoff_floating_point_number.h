/*
 * cutoff_floating_point_number.h
 *
 *  Created on: Jun 27, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_FUNCTIONALITIES_CUTOFF_FLOATING_POINT_NUMBER_H_
#define INCLUDE_FUNCTIONALITIES_CUTOFF_FLOATING_POINT_NUMBER_H_

/*
 * This function allows to cutoff floating point numbers after n_digits of accuracy.
 */
template<typename Number>
Number
cutoff(Number const number, int const n_digits)
{
  AssertThrow(n_digits >= 1, ExcMessage("invalid parameter."));

  int order = std::floor(std::log10(std::fabs(number)));

  Number factor = std::pow(10., (Number)(-order + n_digits - 1));

  Number number_cutoff = int(number * factor) / factor;

  return number_cutoff;
}



#endif /* INCLUDE_FUNCTIONALITIES_CUTOFF_FLOATING_POINT_NUMBER_H_ */
