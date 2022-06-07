#include <functional>
#include <vector>
#include <numeric>
#include <cstdint>
#include <cassert>
#include <cstdlib>

template<typename T>
struct ndarray_view;
template<typename T>
struct ndarray {
  std::vector<uint32_t> _shape;
  uint32_t _num_shape, _num_elem;
  std::vector<uint32_t> _len_of_each_dim;
  T* _data;
  ndarray() {}
  ndarray(const std::vector<uint32_t> & shape) {
    _shape = shape;
    _num_elem = std::accumulate(shape.begin(), shape.end(), 1ul, std::multiplies<size_t>());
    _len_of_each_dim.resize(shape.size());
    _len_of_each_dim.back() = 1;
    for (int i = _shape.size() - 1; i > 0; i--) {
      _len_of_each_dim[i - 1] = _len_of_each_dim[i] * _shape[i];
    }
    _data = new T[_num_elem] {};
    _num_shape = _shape.size();
  }
  void reshape(const std::vector<uint32_t> & shape) {
    uint32_t new_num_elem = std::accumulate(shape.begin(), shape.end(), 1ul, std::multiplies<size_t>());
    assert(new_num_elem == _num_elem);
    _shape = shape;
    _num_elem = new_num_elem;
    _len_of_each_dim.resize(shape.size());
    _len_of_each_dim.back() = 1;
    for (int i = _shape.size() - 1; i > 0; i--) {
      _len_of_each_dim[i - 1] = _len_of_each_dim[i] * _shape[i];
    }
    _num_shape = _shape.size();
  }
  T& at(const std::vector<uint32_t> & idx) {
    assert(idx.size() <= _num_shape);
    return this->at(idx.data(), idx.size());
  }
  T& at(const uint32_t * idx) {
    return this->at(idx, _num_shape);
  }
  T& at(const uint32_t * idx, const uint32_t idx_len) {
    assert(idx_len > 0);
    uint32_t offset = idx[0];
    for (uint32_t dim_idx = 1; dim_idx < _num_shape; dim_idx++) {
      offset *= _shape[dim_idx];
      offset += (dim_idx < idx_len) ? idx[dim_idx] : 0;
    }
    return _data[offset];
  }

  ndarray_view<T> operator[](const uint32_t idx);
  ndarray_view<T> operator[](const std::vector<uint32_t> & idx_array);
  ndarray_view<T> operator[](const ndarray_view<uint32_t> & idx_array);
  //  {
  //   return ndarray_view<T>(&this)[idx];
  // }
};

template<typename T>
struct ndarray_view {
  uint32_t* _shape;
  uint32_t* _len_of_each_dim;
  uint32_t _num_shape;
  T* _data;
  ndarray_view(ndarray<T> & array) {
    _data = array._data;
    _shape = array._shape.data();
    _num_shape = array._shape.size();
    _len_of_each_dim = array._len_of_each_dim.data();
  }
  ndarray_view(const ndarray_view<T> & view, const uint32_t first_idx) {
    _data = view._data + first_idx * view._len_of_each_dim[0];
    _shape = view._shape + 1;
    _len_of_each_dim = view._len_of_each_dim + 1;
    _num_shape = view._num_shape - 1;
  }
  void rebuild(ndarray<T> & array) {
    _data = array._data;
    _shape =  array._shape.data();
    _num_shape = array._shape.size();
    _len_of_each_dim = array._len_of_each_dim.data();
  }
  ndarray_view<T> operator[](const uint32_t idx) {
    return ndarray_view<T>(*this, idx);
  }
  ndarray_view<T> operator[](const std::vector<uint32_t> & idx_array) {
    return _sub_array(idx_array.data(), idx_array.size());
  }
  template<typename IDX_T>
  ndarray_view<T> operator[](std::initializer_list<IDX_T> idx_array) {
    return _sub_array(idx_array.begin(), idx_array.size());
  }
  ndarray_view<T> operator[](const ndarray_view<uint32_t> & idx_array) {
    assert(idx_array._num_shape == 1);
    return _sub_array(idx_array._data, *idx_array._shape);
  }

  T& ref() {
    assert(_num_shape == 0);
    return *_data;
  }
 private:
  template<typename IDX_T>
  ndarray_view<T> _sub_array(const IDX_T * const idx_array, const uint32_t num_idx) {
    ndarray_view<T> ret = *this;
    ret._shape += num_idx;
    ret._len_of_each_dim += num_idx;
    ret._num_shape -= num_idx;
    for (uint i = 0; i < num_idx; i++) {
      ret._data += idx_array[i] * _len_of_each_dim[i];
    }
    {
      size_t offset = idx_array[0];
      for (uint32_t dim_idx = 1; dim_idx < _num_shape; dim_idx++) {
        offset *= _shape[dim_idx];
        offset += (dim_idx < num_idx) ? idx_array[dim_idx] : 0;
      }
      if ((size_t)(ret._data - this->_data) != offset) {
        abort();
      }
    }
    return ret;
  }
};

template<typename T>
ndarray_view<T>ndarray<T>::operator[](const uint32_t idx){
  return ndarray_view<T>(*this)[idx];
}
template<typename T>
ndarray_view<T>ndarray<T>::operator[](const std::vector<uint32_t> & idx_array){
  return ndarray_view<T>(*this)[idx_array];
}
template<typename T>
ndarray_view<T>ndarray<T>::operator[](const ndarray_view<uint32_t> & idx_array){
  return ndarray_view<T>(*this)[idx_array];
}