"""
Do a simple test of parallel processing of a numpy array. In this case,
wrap processing in a class.
"""
import numpy as np
import multiprocessing as mp

def mapfun(args):
    return Worker._mapfun(args)

class Worker():
    """
    This class does some work.
    """
    def __init__(self, tuplist):
        self.slices = tuplist

    @staticmethod
    def _mapfun(args):
        """
        Function to be mapped over things.
        """
        start, end = args
        print start, end
        mn = np.mean(inarr[start:end])
        outarr[start:end] = mn * np.ones(end - start)
        return mn

    def calc_to_do(self, indata):
        """
        Function that gets called.
        """
        global inarr, outarr
        print indata
        shared_in = mp.Array('d', indata, lock=False)
        inarr = np.frombuffer(shared_in)
        shared_out = mp.Array('d', indata.size, lock=False)
        outarr = np.frombuffer(shared_out)

        # set up a worker pool
        pool = mp.Pool(processes=4)

        # map over tuplist
        result_list = pool.map(mapfun, self.slices)

        return result_list

if __name__ == '__main__':
    # allocate some memory
    # no need to lock, since we're operating on separate sections of data
    sz = 10000

    # make some data
    datarr = np.random.rand(sz)

    # make a list of (nonoverlapping) (start, end) tuples
    gap = 100
    starts = xrange(0, sz, gap)
    ends = (s + gap for s in starts)
    tuplist = zip(starts, ends)

    obj = Worker(tuplist)
    print obj.calc_to_do(datarr)