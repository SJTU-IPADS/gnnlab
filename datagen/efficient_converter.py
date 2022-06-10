import cupy, numpy

class TinyCOO:
  def __init__(self, row, col):
    self.row = row
    self.col = col
    self.location = None
    if isinstance(self.row, cupy.ndarray):
      self.location = 'gpu'
    elif row is not None:
      self.location = 'cpu'


def clean_cuda_cache():
  mempool = cupy.get_default_memory_pool()
  mempool.free_all_blocks()
def print_cuda_memory():
  print(cupy.cuda.runtime.memGetInfo())

class COOUtil:
  @staticmethod
  def release_coo(coo):
    coo.row = None
    coo.col = None
    coo.location = None
    clean_cuda_cache()

  @staticmethod
  def to_cpu(coo):
    if coo.location == 'gpu':
      coo.row = coo.row.get()
      coo.col = coo.col.get()
      coo.location = 'cpu'
    else:
      assert(coo.row is None or isinstance(coo.row, numpy.ndarray))
      assert(coo.col is None or isinstance(coo.col, numpy.ndarray))
    clean_cuda_cache()

  @staticmethod
  def to_gpu(coo):
    if coo.location == 'cpu':
      coo.row = cupy.asarray(coo.row)
      coo.col = cupy.asarray(coo.col)
      coo.location = 'gpu'
    clean_cuda_cache()

  @staticmethod
  def dedup_sorted_coo(coo):
    src_row, src_col = coo.row, coo.col
    diff = cupy.sparse.coo_matrix._sum_duplicates_diff(src_row, src_col, size=src_row.size)
    index = cupy.cumsum(diff, dtype='int64')
    del diff
    clean_cuda_cache()
    size = int(index[-1]) + 1 
    extractor = cupy.ElementwiseKernel(
        'T src, int64 index',
        'raw T dst',
        '''
        dst[index] = src;
        ''',
        'fastgraph_extractor'
    )
    row = cupy.empty(size, dtype=src_row.dtype)
    extractor(src_row, index, row)
    coo.row = row
    del src_row
    clean_cuda_cache()

    col = cupy.empty(size, dtype=src_col.dtype)
    extractor(src_col, index, col)
    del src_col
    coo.col = col
    del index
    coo.location = 'gpu'
    clean_cuda_cache()

  @staticmethod
  def sort_coo(coo):
    # 3. sort (row,col)
    assert(isinstance(coo.row, cupy.ndarray))
    assert(isinstance(coo.col, cupy.ndarray))
    keys = cupy.stack([coo.col, coo.row])
    # 3.1 move row, col back to cpu to save space
    COOUtil.to_cpu(coo)
    clean_cuda_cache()
    # 3.2 actual sort
    order = cupy.lexsort(keys)
    del keys
    clean_cuda_cache()
    # 3.3 reorder row & col
    coo.row = cupy.asarray(coo.row)[order]
    clean_cuda_cache()
    coo.col = cupy.asarray(coo.col)[order]
    coo.location = 'gpu'
    del order
    clean_cuda_cache()

  @staticmethod
  def sort_coo_row_only(coo):
    # 3. sort (row,col)
    keys = coo.row
    if isinstance(keys, cupy.ndarray):
      # 3.1 move row, col back to cpu to save space
      COOUtil.to_cpu(coo)
      clean_cuda_cache()
    else:
      keys = cupy.asarray(keys)
    keys = keys.reshape((1,-1))
    # 3.2 actual sort
    order = cupy.lexsort(keys)
    del keys
    clean_cuda_cache()
    # 3.3 reorder row & col
    coo.row = cupy.asarray(coo.row)[order]
    clean_cuda_cache()
    coo.col = cupy.asarray(coo.col)[order]
    coo.location = 'gpu'
    del order
    clean_cuda_cache()
  @staticmethod
  def slice_coo(coo, row_lb, row_ub):
    new_coo = TinyCOO(None, None)
    new_coo.row = coo.row[row_lb:row_ub]
    new_coo.col = coo.col[row_lb:row_ub]
    new_coo.location = coo.location
    return new_coo

def get_indptr_from_sorted_coo(row, num_nodes):
  if not isinstance(row, cupy.ndarray):
    row = cupy.asarray(row)
  indptr = cupy.bincount(row, minlength=num_nodes).cumsum()
  indptr = cupy.pad(indptr, (1, 0), 'constant', constant_values=(0,))
  del row
  clean_cuda_cache()
  return indptr

# def get_row_from_indptr(indptr, size):
#   row = cupy.zeros(shape=(size), dtype='int32')
#   extractor = cupy.ElementwiseKernel(
#       'T src, int64 index',
#       'raw T dst',
#       '''
#       dst[index] = src;
#       ''',
#       'fastgraph_extractor'
#   )

def concat_coo(coo_list):
  new_coo = TinyCOO(None, None)
  new_coo.location = 'gpu'
  new_coo.row = cupy.concatenate([cupy.asarray(coo.row) for coo in coo_list])
  clean_cuda_cache()
  new_coo.col = cupy.concatenate([cupy.asarray(coo.col) for coo in coo_list])
  clean_cuda_cache()
  return new_coo

# both coo must be sorted by row (col may be disordered)
def chunk_sparse_add(coo1, coo2, num_rows, to_cpu = True, do_dedup = True, chunk_size = 1024):
  indptr1 = get_indptr_from_sorted_coo(coo1.row, num_rows)
  print(coo2.row.shape)
  indptr2 = get_indptr_from_sorted_coo(coo2.row, num_rows)
  clean_cuda_cache()
  assert(indptr1.shape[0] == num_rows + 1)
  assert(indptr2.shape[0] == num_rows + 1)

  slice_sum_list = []

  for node_lb in range(0, num_rows, chunk_size):
    node_ub = min(node_lb + chunk_size, num_rows)
    print(f"chunk {node_lb},{node_ub}")
    print_cuda_memory()
    # handle row number in range [node_lb, node_ub)
    coo_slice_1 = COOUtil.slice_coo(coo1, indptr1[node_lb].item(), indptr1[node_ub].item())
    coo_slice_2 = COOUtil.slice_coo(coo2, indptr2[node_lb].item(), indptr2[node_ub].item())
    coo_slice_sum = efficient_sparse_add(coo_slice_1, coo_slice_2, to_cpu=True, do_dedup=do_dedup)
    COOUtil.to_cpu(coo_slice_sum)
    del coo_slice_1, coo_slice_2
    slice_sum_list.append(coo_slice_sum)
    clean_cuda_cache()
  del indptr1, indptr2
  new_coo = concat_coo(slice_sum_list)
  del slice_sum_list
  if to_cpu:
    COOUtil.to_cpu(new_coo)
  clean_cuda_cache()
  return new_coo

def efficient_sparse_add(coo1, coo2, to_cpu = True, do_dedup = True):
  coo = TinyCOO(None, None)
  coo.row = cupy.concatenate((cupy.asarray(coo1.row), cupy.asarray(coo2.row)))
  coo1.row = None
  coo2.row = None
  clean_cuda_cache()
  coo.col = cupy.concatenate((cupy.asarray(coo1.col), cupy.asarray(coo2.col)))
  coo1.col = None
  coo2.col = None
  coo.location = 'gpu'
  clean_cuda_cache()

  COOUtil.sort_coo(coo)
  if do_dedup:
    COOUtil.dedup_sorted_coo(coo)
  if to_cpu:
    COOUtil.to_cpu(coo)
  clean_cuda_cache()
  return coo

# def efficient_sparse_multi_add(coo_list, to_cpu = True, do_dedup = True):
#   coo = concat_coo(coo_list)
#   clean_cuda_cache()

#   COOUtil.sort_coo(coo)
#   if do_dedup:
#     COOUtil.dedup_sorted_coo(coo)
#   if to_cpu:
#     COOUtil.to_cpu(coo)
#   clean_cuda_cache()
#   return coo



# # both coo must be sorted by row (col may be disordered)
# def chunk_sparse_multi_add(coo_list, num_rows, to_cpu = True, do_dedup = True, chunk_size = 1024):
#   indptr_list = [get_indptr_from_sorted_coo(coo.row, num_rows) for coo in coo_list]
#   clean_cuda_cache()
#   assert(indptr_list[0].shape[0] == num_rows + 1)

#   slice_sum_list = []

#   for node_lb in range(0, num_rows, chunk_size):
#     node_ub = min(node_lb + chunk_size, num_rows)
#     print(f"chunk {node_lb},{node_ub}")
#     print_cuda_memory()
#     # handle row number in range [node_lb, node_ub)
#     coo_slice_list = [COOUtil.slice_coo(coo_list[i], indptr_list[i][node_lb].item(), indptr_list[i][node_ub].item()) for i in range(len(coo_list))]
#     coo_slice_sum = efficient_sparse_multi_add(coo_slice_list, to_cpu=True, do_dedup=do_dedup)
#     COOUtil.to_cpu(coo_slice_sum)
#     del coo_slice_list
#     slice_sum_list.append(coo_slice_sum)
#     clean_cuda_cache()
#   del indptr_list
#   new_coo = concat_coo(slice_sum_list)
#   del slice_sum_list
#   if to_cpu:
#     COOUtil.to_cpu(new_coo)
#   clean_cuda_cache()
#   return new_coo