#ifndef LUNG_LUNG_UTIL
#define LUNG_LUNG_UTIL

namespace LungID
{
int
create_root()
{
  return 0;
}

int
generate(int num, bool left)
{
  int generation_parent = (num << 27) >> 27;
  int generation        = generation_parent + 1;

  if(left)
    return (num + 1) | (1 << (31 - generation));
  else
    return (num + 1);
}

int 
get_generation(int num)
{
  return (num << 27) >> 27;
}

template<typename T>
std::string to_binary(T val)
{
  std::size_t sz = sizeof(val)*CHAR_BIT;
  std::string ret(sz, ' ');
  while( sz-- )
  {
    ret[sz] = '0'+(val&1);
    val >>= 1;
  }
  return ret;
}

std::string
to_string(int num)
{
    return to_binary(num);
    
}

} // namespace LungID


struct CellAdditionalInfo
{
  CellAdditionalInfo() : radius(0), generation(0)
  {
  }

  CellAdditionalInfo(double radius, int generation) : radius(radius), generation(generation)
  {
  }

  double radius;
  int    generation;
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
       int                               todo,
       bool                              _is_left)
    : id(id),
      generation(cells_additional_data[id].generation),
      radius(cells_additional_data[id].radius),
      from(points[cells[id].vertices[0]]),
      to(points[cells[id].vertices[1]]),
      _is_left(_is_left),
      parent(parent),
      left_child(nullptr),
      right_child(nullptr),
      _is_dummy(false)
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

  Node(Node * left_child, Node * right_child)
    : id(0),
      generation(left_child->get_generation() - 1),
      radius((left_child->get_radius() + right_child->get_radius()) / 2 / 0.8),
      _is_left(false),
      parent(nullptr),
      left_child(left_child),
      right_child(right_child),
      _is_dummy(false)
  {
    left_child->set_parent(this);
    left_child->_is_left = true;
    right_child->set_parent(this);
    right_child->_is_left = false;
  }

  Node(Node * left_child, Node * right_child, Point<3> from, bool _is_left, bool do_twist = true, bool do_rot = false)
    : id(0),
      generation(left_child->get_generation() - 1),
      radius((left_child->get_radius() + right_child->get_radius()) / 2 / 0.7),
      _is_left(_is_left),
      parent(nullptr),
      left_child(left_child),
      right_child(right_child),
      _is_dummy(false)
  {
    left_child->set_parent(this);
    left_child->_is_left = true;
    right_child->set_parent(this);
    right_child->_is_left = false;

    this->to   = left_child->from;
    this->from = from;
    
    this->do_twist = do_twist;
    this->do_rot   = do_rot;
  }

  static Node *
  create_root(unsigned int                      id,
              int *                             xadj_vertex,
              int *                             adjncy_vertex,
              std::vector<CellAdditionalInfo> & cells_additional_data,
              std::vector<CellData<1>> &        cells,
              std::vector<Point<3>> &           points,
              int                               todo)
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
    Point<3> & p4 = this->left_child->from;

    dealii::Tensor<1, 3> v1 = p1 - p2;
    dealii::Tensor<1, 3> v2 = p2 - p3;
    dealii::Tensor<1, 3> v3 = p2 - p4;

    dealii::Tensor<1, 3> n1 = cross_product_3d(v1, v2);
    dealii::Tensor<1, 3> n2 = cross_product_3d(v1, v3);

    double n3 = scalar_product(n1, n2);
    return n3 == 0;
  }

  dealii::Tensor<1, 3>
  get_normal_vector()
  {
    Point<3> &           p1 = this->right_child->from;
    Point<3> &           p2 = this->right_child->to;
    Point<3> &           p3 = this->left_child->from;
    Point<3> &           p4 = this->left_child->to;
    dealii::Tensor<1, 3> v1 = p1 - p2;
    dealii::Tensor<1, 3> v2 = p3 - p4;

    double n3 = std::acos(v1 * v2 / v1.norm() / v2.norm());
    if(true || abs(n3) < 1e-10 || abs(abs(n3) - numbers::PI) < 1e-10)
    {
      Point<3> &           p0 = this->from;
      dealii::Tensor<1, 3> v1 = p0 - p1;
      return cross_product_3d(v1, v2);
    }
    else
    {
      return cross_product_3d(v1, v2);
    }
  }
  //
  //    dealii::Tensor<1, 3> get_normal_vector() {
  //        Point<3>& p1 = this->right_child->from;
  //        Point<3>& p2 = this->right_child->to;
  //        Point<3>& p3 = this->left_child->from;
  //        Point<3>& p4 = this->left_child->to;
  //        dealii::Tensor<1, 3> v1 = p1 - p2;
  //        dealii::Tensor<1, 3> v2 = p3 - p4;
  //
  //        std::cout << "AAAAAAAAAAAAAA1" << std::endl;
  //        double n3 = std::acos( v1*v2  /v1.norm()/v2.norm());
  //        std::cout << n3 << std::endl;
  //        if(abs(n3)<1e-10 || abs(abs(n3)-numbers::PI)<1e-10){
  //            std::cout << "AAAAAAAAAAAAAA2" << std::endl;
  //            Point<3>& p0 = this->from;
  //            dealii::Tensor<1, 3> v1 = p0 - p1;
  //            return cross_product_3d(v1, v2);
  //        } else
  //            return cross_product_3d(v1, v2);
  //    }

  virtual dealii::Tensor<1, 3>
  get_tangential_vector()
  {
    Point<3> &           p1 = this->from;
    Point<3> &           p2 = this->to;
    dealii::Tensor<1, 3> v1 = p1 - p2;

    return v1;
  }

  double
  get_length()
  {
    auto temp = to;
    temp -= from;
    return temp.norm();
  }

  double get_degree(dealii::Tensor<1, 3> t1, dealii::Tensor<1, 3> t2)
  {
    return std::acos(t1 * t2 / t1.norm() / t2.norm());
  }

  double get_degree(Point<3> & p1, Point<3> & p2, Point<3> & p3, Point<3> & p4)
  {
    return get_degree(p2 - p1, p4 - p3);
  }

  virtual double
  get_degree_1()
  {
    return get_degree(this->from, this->to, left_child->from, left_child->to);
  }

  virtual double
  get_degree_2()
  {
    return get_degree(this->from, this->to, right_child->from, right_child->to);
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

  bool
  has_children()
  {
    return left_child != nullptr && right_child != nullptr;
  }

  Point<3>
  get_source()
  {
    return from;
  }

  Point<3>
  get_target()
  {
    return to;
  }

  bool
  is_root()
  {
    return parent == nullptr;
  }

  bool
  is_left()
  {
    return _is_left;
  }

  double
  get_radius()
  {
    return radius;
  }

  Node *
  get_parent()
  {
    return parent;
  }

  void
  set_parent(Node * parent)
  {
    this->parent = parent;
  }

  Node *
  get_left_child()
  {
    return left_child;
  }

  Node *
  get_right_child()
  {
    return right_child;
  }

  int
  get_generation()
  {
    return generation;
  }

  bool
  is_dummy()
  {
    return _is_dummy;
  }

  virtual int
  get_intersections()
  {
    return std::max(2.0,get_length() / 2 / get_radius());
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
  bool         _is_dummy;
  bool         do_twist = true;
  bool         do_rot = false;
  
  std::vector<Point<3>> skeleton;
};

class DummyNode : public Node
{
public:
  DummyNode(Node * left_child, Node * right_child) : Node(left_child, right_child)
  {
    this->_is_dummy = true;
  }

  virtual double
  get_degree_1()
  {
    return get_degree(this->right_child->from,
                      this->right_child->to,
                      this->left_child->from,
                      this->left_child->to) /
           2;
  }

  virtual double
  get_degree_2()
  {
    return get_degree(this->left_child->from,
                      this->left_child->to,
                      this->right_child->from,
                      this->right_child->to) /
           2;
  }


  dealii::Tensor<1, 3>
  get_tangential_vector()
  {
    auto right_dir = this->right_child->from - this->right_child->to;
    auto left_dir  = this->left_child->from - this->left_child->to;

    return left_dir / left_dir.norm() + right_dir / right_dir.norm();
  }


  virtual int
  get_intersections()
  {
    return 0;
  }
};

#endif