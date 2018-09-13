#ifndef LAZY_PTR_H_
#define LAZY_PTR_H_

template<typename T>
class lazy_ptr
{
public:
  lazy_ptr() : tp(&t)
  {
  }

  void
  reset()
  {
    this->tp = &this->t;
  }
  
  void
  reinit(T const & t)
  {
    this->tp = &t;
  }
  
  T &
  own()
  {
    return t;
  }
  
  T const * operator->()
  {
    return tp;
  }
  
  T const & operator*()
  {
    return *tp;
  }

private:
  T         t;
  T const * tp;
};

#endif