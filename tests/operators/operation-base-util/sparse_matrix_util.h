#ifndef OPERATOR_BASE_SM_UTIL
#define OPERATOR_BASE_SM_UTIL

void
print_ascii(TrilinosWrappers::SparseMatrix & system_matrix)
{
  int prev_row = -1;
  int prev_col = -1;

  for(auto i : system_matrix)
  {
    double val      = i.value();
    int    curr_row = i.row();
    int    curr_col = i.column();

    if(prev_row != curr_row)
    {
      prev_col = -1;
      printf("\n");
    }

    if((curr_col - prev_col - 1) != 0)
      printf("%*c", 8 * (curr_col - prev_col - 1) + 1, (char)0);
    printf("%8.2f", val);

    prev_col = curr_col;
    prev_row = curr_row;
  }

  std::cout << std::endl;
}

void
print_matlab(TrilinosWrappers::SparseMatrix & system_matrix)
{
  for(auto i : system_matrix)
  {
    double val      = i.value();
    int    curr_row = i.row() + 1;
    int    curr_col = i.column() + 1;

    if(std::abs(val) > 1e-12)
    {
      printf("%d %d %15.10f\n", curr_row, curr_col, val);
    }
  }

  printf("\n\n");
}

#endif