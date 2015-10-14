"""
Do a simple test of parallel processing of a numpy array.
"""
import numpy as np
import multiprocessing as mp

def testfun(args):
    """
    Take data between start and end from inarr, do something,
    and save to outarr.
    """
    start, end = args
    print start, end
    outarr[start:end] = np.mean(inarr[start:end]) * np.ones(end - start)
    return None

if __name__ == '__main__':
    # allocate some memory
    # no need to lock, since we're operating on separate sections of data
    sz = 10000
    bigarr = mp.Array('d', sz, lock=False)
    result = mp.Array('d', sz, lock=False)

    # Wrap the data as shared arrays
    inarr = np.frombuffer(bigarr)
    outarr = np.frombuffer(result)

    # make some data
    inarr[:] = np.random.rand(sz)
    outarr[:] = np.zeros(sz)

    # set up a worker pool
    pool = mp.Pool(processes=4)

    # make a list of (nonoverlapping) (start, end) tuples
    gap = 100
    starts = xrange(0, sz, gap)
    ends = (s + gap for s in starts)
    tuplist = zip(starts, ends)

    pool.map(testfun, tuplist)