#ifndef LUNG_ALGEBRA_UTIL
#define LUNG_ALGEBRA_UTIL

Tensor<2, 3> compute_rotation_matrix(dealii::Tensor<1, 3> a0,
                                     dealii::Tensor<1, 3> a1,
                                     dealii::Tensor<1, 3> b0,
                                     dealii::Tensor<1, 3> b1)
{
  // assemble source system
  Tensor<2, 3> A;
  for(int i = 0; i < 3; i++)
    A[i][0] = a0[i];
  for(int i = 0; i < 3; i++)
    A[i][1] = a1[i];

  dealii::Tensor<1, 3> a2 = cross_product_3d(a0, a1);
  for(int i = 0; i < 3; i++)
    A[i][2] = a2[i];

  // assemble target system
  Tensor<2, 3> B;
  for(int i = 0; i < 3; i++)
    B[i][0] = b0[i];
  for(int i = 0; i < 3; i++)
    B[i][1] = b1[i];

  dealii::Tensor<1, 3> b2 = cross_product_3d(b0, b1);
  for(int i = 0; i < 3; i++)
    B[i][2] = b2[i];

  // compute rotation matrix: R = B*A^{-1}
  return B * invert(A);
}

#endif