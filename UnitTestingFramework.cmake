#########################################################################
#
#                 #######               ######  #######
#                 ##                    ##   ## ##
#                 #####   ##  ## #####  ##   ## ## ####
#                 ##       ####  ## ##  ##   ## ##   ##
#                 ####### ##  ## ###### ######  #######
#
#  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
#
#  Copyright (C) 2021 by the ExaDG authors
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#########################################################################

OPTION(EXADG_WITH_GOOGLETEST "Use GoogleTest for unit testing" ON)
IF(EXADG_WITH_GOOGLETEST)
  MESSAGE(STATUS "Unit tests with GoogleTest: enabled")

  ADD_CUSTOM_TARGET(unittests)
  ADD_DEPENDENCIES(exadg unittests)

  IF(TARGET gtest)
    MESSAGE(FATAL_ERROR "A target <gtest> has already been included by a TPL." "This is not supported.")
  ENDIF()

  INCLUDE(FetchContent)
  FETCHCONTENT_DECLARE(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG release-1.12.0
      )
  FETCHCONTENT_MAKEAVAILABLE(googletest)

  ADD_SUBDIRECTORY(tests)
ELSE()
  MESSAGE(STATUS "Unit tests with GoogleTest: disabled")
ENDIF()
