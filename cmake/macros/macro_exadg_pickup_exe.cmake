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

MACRO(EXADG_PICKUP_EXE FILE_NAME TARGET_NAME EXE_NAME)

    ADD_EXECUTABLE(${TARGET_NAME} ${FILE_NAME})
    DEAL_II_SETUP_TARGET(${TARGET_NAME})
    SET_TARGET_PROPERTIES(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${EXE_NAME})
    TARGET_LINK_LIBRARIES(${TARGET_NAME} exadg)

    IF(${EXADG_WITH_FFTW})
       TARGET_LINK_FFTW(${TARGET_NAME})
    ENDIF()

ENDMACRO(EXADG_PICKUP_EXE)
