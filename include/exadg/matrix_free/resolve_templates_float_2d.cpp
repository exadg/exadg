/*
 * resolve_templates_float_2d.cpp
 *
 *  Created on: May 2, 2019
 *      Author: fehn
 */

#include <exadg/matrix_free/evaluation_template_factory.templates.h>

template struct dealii::internal::FEEvaluationFactory<2, float>;

template struct dealii::internal::FEFaceEvaluationFactory<2, float>;

template struct dealii::internal::CellwiseInverseMassFactory<2, float>;
