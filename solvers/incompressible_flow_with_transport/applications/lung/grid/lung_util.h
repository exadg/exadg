#ifndef LUNG_LUNG_UTIL
#define LUNG_LUNG_UTIL

namespace ExaDG
{

struct CellAdditionalInfo
{
  CellAdditionalInfo() : cell_id(0), radius(0), generation(0)
  {
  }

  CellAdditionalInfo(unsigned int cell_id, double radius, int generation)
    : cell_id(cell_id), radius(radius), generation(generation)
  {
  }

  unsigned int cell_id;
  double       radius;
  int          generation;
};

class Node
{
public:
  Node(Node *                            parent,
       unsigned int                      id,
       int *                             xadj_vertex,
       int *                             adjncy_vertex,
       std::vector<CellAdditionalInfo> & cells_additional_data,
       std::vector<CellData<1>> &        cells,
       std::vector<Point<3>> &           points,
       unsigned int                      todo,
       bool                              _is_left)
    : id(cells_additional_data[id].cell_id),
      generation(cells_additional_data[id].generation),
      radius(cells_additional_data[id].radius),
      from(points[cells[id].vertices[0]]),
      to(points[cells[id].vertices[1]]),
      _is_left(_is_left),
      parent(parent),
      left_child(nullptr),
      right_child(nullptr)
  {
    // enough generations have been processed
    if(todo == 0)
      return;

    // process children
    int i = xadj_vertex[id];
    // ... left child
    for(; i < xadj_vertex[id + 1]; i++)
      if(cells_additional_data[adjncy_vertex[i]].generation > this->generation)
      {
        left_child = new Node(this,
                              adjncy_vertex[i],
                              xadj_vertex,
                              adjncy_vertex,
                              cells_additional_data,
                              cells,
                              points,
                              todo - 1,
                              true);
        break;
      }
    i++;
    // ... right child
    for(; i < xadj_vertex[id + 1]; i++)
      if(cells_additional_data[adjncy_vertex[i]].generation > this->generation)
      {
        right_child = new Node(this,
                               adjncy_vertex[i],
                               xadj_vertex,
                               adjncy_vertex,
                               cells_additional_data,
                               cells,
                               points,
                               todo - 1,
                               false);
        break;
      }
  }

  static Node *
  create_root(unsigned int                      id,
              int *                             xadj_vertex,
              int *                             adjncy_vertex,
              std::vector<CellAdditionalInfo> & cells_additional_data,
              std::vector<CellData<1>> &        cells,
              std::vector<Point<3>> &           points,
              unsigned int                      todo)
  {
    return new Node(
      nullptr, id, xadj_vertex, adjncy_vertex, cells_additional_data, cells, points, todo, false);
  }

  virtual ~Node()
  {
    if(left_child != nullptr)
      delete left_child;
    if(right_child != nullptr)
      delete right_child;
  }

  static bool
  check_if_planar(dealii::Tensor<1, 3> v1, dealii::Tensor<1, 3> v2, dealii::Tensor<1, 3> v3)
  {
    dealii::Tensor<2, 3> A;

    for(int i = 0; i < 3; i++)
      A[i][0] = v1[i];

    for(int i = 0; i < 3; i++)
      A[i][1] = v2[i];

    for(int i = 0; i < 3; i++)
      A[i][2] = v3[i];

    double det = determinant(A);

    return std::abs(det) < 1e-10;
  }

  bool
  check_if_planar()
  {
    if(left_child != nullptr && !left_child->check_if_planar())
      return false;
    else if(left_child == nullptr)
      return true;

    if(right_child != nullptr && !right_child->check_if_planar())
      return false;
    else if(right_child == nullptr)
      return true;

    Point<3> & p1 = this->from;
    Point<3> & p2 = this->to;
    Point<3> & p3 = this->left_child->to;
    Point<3> & p4 = this->right_child->to;

    dealii::Tensor<1, 3> v1 = p2 - p1;
    dealii::Tensor<1, 3> v2 = p3 - p2;
    dealii::Tensor<1, 3> v3 = p4 - p2;

    return check_if_planar(v1, v2, v3);
  }

  static dealii::Tensor<1, 3>
  get_normal_vector(dealii::Tensor<1, 3> v1, dealii::Tensor<1, 3> v2)
  {
    double scalar_product = v1 * v2;

    double deg = std::acos(scalar_product / v1.norm() / v2.norm());

    if(deg < 1e-10 || (numbers::PI - deg) < 1e-10)
      AssertThrow(false, ExcMessage("Given vectors are collinear!"))

        return cross_product_3d(v1, v2);
  }

  [[nodiscard]] dealii::Tensor<1, 3>
  get_normal_vector_children() const
  {
    Point<3> &           p1 = this->right_child->from;
    Point<3> &           p2 = this->right_child->to;
    Point<3> &           p3 = this->left_child->from;
    Point<3> &           p4 = this->left_child->to;
    dealii::Tensor<1, 3> v1 = p2 - p1;
    dealii::Tensor<1, 3> v2 = p4 - p3;

    double argument = v1 * v2 / v1.norm() / v2.norm();

    if((1.0 - std::abs(argument)) < 1e-10)
      argument = std::copysign(1.0, argument);

    double deg = std::acos(argument);

    dealii::Tensor<1, 3> normal;

    if(deg < 1e-10 || (numbers::PI - deg) < 1e-10)
    {
      Point<3>             p5       = this->from;
      dealii::Tensor<1, 3> v_parent = p1 - p5;
      normal                        = get_normal_vector(v1 / v1.norm(), v_parent / v_parent.norm());
    }
    else
    {
      normal = get_normal_vector(v1 / v1.norm(), v2 / v2.norm());
    }

    return normal;
  }

  dealii::Tensor<1, 3>
  get_normal_vector_parent_left()
  {
    Point<3> &           p1 = this->from;
    Point<3> &           p2 = this->to;
    Point<3> &           p3 = this->left_child->from;
    Point<3> &           p4 = this->left_child->to;
    dealii::Tensor<1, 3> v1 = p1 - p2;
    dealii::Tensor<1, 3> v2 = p4 - p3;

    double argument = v1 * v2 / v1.norm() / v2.norm();

    if((1.0 - std::abs(argument)) < 1e-10)
      argument = std::copysign(1.0, argument);

    double deg = std::acos(argument);

    dealii::Tensor<1, 3> normal;

    if(deg < 1e-10 || (numbers::PI - deg) < 1e-10)
    {
      Point<3>             p5            = this->right_child->to;
      dealii::Tensor<1, 3> v_right_child = p5 - p3;

      normal = get_normal_vector(v1 / v1.norm(), v_right_child / v_right_child.norm());
    }
    else
    {
      normal = get_normal_vector(v1 / v1.norm(), v2 / v2.norm());
    }

    return normal;
  }

  dealii::Tensor<1, 3>
  get_normal_vector_parent_right()
  {
    Point<3> &           p1 = this->from;
    Point<3> &           p2 = this->to;
    Point<3> &           p3 = this->right_child->from;
    Point<3> &           p4 = this->right_child->to;
    dealii::Tensor<1, 3> v1 = p1 - p2;
    dealii::Tensor<1, 3> v2 = p4 - p3;

    double argument = v1 * v2 / v1.norm() / v2.norm();

    if((1.0 - std::abs(argument)) < 1e-10)
      argument = std::copysign(1.0, argument);

    double deg = std::acos(argument);

    dealii::Tensor<1, 3> normal;

    if(deg < 1e-10 || (numbers::PI - deg) < 1e-10)
    {
      Point<3>             p5           = this->left_child->to;
      dealii::Tensor<1, 3> v_left_child = p5 - p3;

      normal = get_normal_vector(v1 / v1.norm(), v_left_child / v_left_child.norm());
    }
    else
    {
      normal = get_normal_vector(v1 / v1.norm(), v2 / v2.norm());
    }

    return normal;
  }

  virtual dealii::Tensor<1, 3>
  get_tangential_vector()
  {
    Point<3> &           p1 = this->from;
    Point<3> &           p2 = this->to;
    dealii::Tensor<1, 3> v1 = p2 - p1;

    return v1;
  }

  [[nodiscard]] double
  get_length() const
  {
    auto temp = to;
    temp -= from;
    return temp.norm();
  }

  static double
  get_degree(dealii::Tensor<1, 3> t1, dealii::Tensor<1, 3> t2)
  {
    double argument = t1 * t2 / t1.norm() / t2.norm();

    if((1.0 - std::abs(argument)) < 1e-10)
      argument = std::copysign(1.0, argument);

    return std::acos(argument);
  }

  static double
  get_degree(Point<3> & p1, Point<3> & p2, Point<3> & p3, Point<3> & p4)
  {
    return get_degree(p2 - p1, p4 - p3);
  }

  void
  print()
  {
    printf("%7d %2d %.4e %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e\n",
           id,
           generation,
           radius,
           from[0],
           from[1],
           from[2],
           to[0],
           to[1],
           to[2]);
    if(left_child != nullptr)
      left_child->print();
    if(right_child != nullptr)
      right_child->print();
  }

  [[nodiscard]] bool
  has_children() const
  {
    return left_child != nullptr && right_child != nullptr;
  }

  [[nodiscard]] bool
  has_child() const
  {
    if(left_child == nullptr and right_child != nullptr)
      AssertThrow(false, ExcMessage("single child is only allowed as left child"));

    return (left_child != nullptr and right_child == nullptr);
  }

  [[nodiscard]] Point<3>
  get_source() const
  {
    return from;
  }

  [[nodiscard]] Point<3>
  get_target() const
  {
    return to;
  }

  [[nodiscard]] bool
  is_root() const
  {
    return parent == nullptr;
  }

  [[nodiscard]] bool
  is_left() const
  {
    return _is_left;
  }

  [[nodiscard]] double
  get_radius() const
  {
    return radius;
  }

  [[nodiscard]] Node *
  get_parent() const
  {
    return parent;
  }

  [[nodiscard]] Node *
  get_left_child() const
  {
    return left_child;
  }

  [[nodiscard]] Node *
  get_right_child() const
  {
    return right_child;
  }

  [[maybe_unused]] [[nodiscard]] int
  get_generation() const
  {
    return generation;
  }

  virtual int
  get_intersections()
  {
    return std::max(2.0, get_length() / 2.0 / get_radius());
  }

public:
  unsigned int id;
  int          generation;
  double       radius;
  Point<3>     from;
  Point<3>     to;
  bool         _is_left;
  Node *       parent;
  Node *       left_child;
  Node *       right_child;
  bool         do_rot   = false;

  std::vector<Point<3>> skeleton;
};

} // namespace ExaDG

#endif
