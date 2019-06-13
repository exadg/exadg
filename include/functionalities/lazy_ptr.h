#ifndef LAZY_PTR_H_
#define LAZY_PTR_H_

template<typename T>
class lazy_ptr
{
public:
  lazy_ptr() : t_ptr(&t)
  {
  }

  // resets the pointer (using own data)
  void
  reset()
  {
    this->t_ptr = &this->t;
  }

  // resets the pointer (using external data)
  void
  reset(T const & t_other)
  {
    this->t_ptr = &t_other;
  }

  // provides access to own data storage, e.g., in order to overwrite the data
  T &
  own()
  {
    return t;
  }

  T const * operator->()
  {
    return t_ptr;
  }

  T const & operator*()
  {
    return *t_ptr;
  }

private:
  T         t;
  T const * t_ptr;
};

#endif
