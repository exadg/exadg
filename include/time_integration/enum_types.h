/*
 * enum_types.h
 *
 *  Created on: Apr 1, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_TIME_INTEGRATION_ENUM_TYPES_H_
#define INCLUDE_TIME_INTEGRATION_ENUM_TYPES_H_

#include <string>

namespace ExaDG
{
using namespace dealii;

enum class CFLConditionType
{
  VelocityNorm,
  VelocityComponents
};

std::string
enum_to_string(CFLConditionType const enum_type);

enum class GenAlphaType
{
  Newmark,
  GenAlpha,
  HHTAlpha,
  BossakAlpha
};

std::string
enum_to_string(GenAlphaType const enum_type);

} // namespace ExaDG

#endif /* INCLUDE_TIME_INTEGRATION_ENUM_TYPES_H_ */
