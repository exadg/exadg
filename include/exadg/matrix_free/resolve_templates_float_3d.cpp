/*
 * resolve_templates_float_3d.cpp
 *
 *  Created on: May 2, 2019
 *      Author: fehn
 */

#include <exadg/matrix_free/evaluation_template_factory.templates.h>

template struct dealii::internal::FEEvaluationFactory<3, float>;

template struct dealii::internal::FEFaceEvaluationFactory<3, float>;

template struct dealii::internal::CellwiseInverseMassFactory<3, float>;
