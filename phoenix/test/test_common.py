# -*- coding: utf8
from __future__ import print_function, division

import unittest
import random

from phoenix.common import ContiguousID

class TestContiguousID(unittest.TestCase):

    def test_all(self):
        cont_id = ContiguousID()
        self.assertEquals(cont_id['a'], 0)
        self.assertEquals(cont_id['b'], 1)
        self.assertEquals(cont_id[0], 2)

        self.assertEquals(cont_id['a'], 0)
        self.assertEquals(cont_id[int], 3)

        new_keys = range(1, 200)
        random.shuffle(new_keys)
        iter_vals = iter(range(4, 204))
        for i in new_keys:
            self.assertEquals(cont_id[i], next(iter_vals))

        for i, d in enumerate(sorted(cont_id.itervalues())):
            self.assertEqual(i, d)

        self.assertEquals(203, len(cont_id))

    def test_boost(self):
        cont_id = ContiguousID()
        self.assertEquals(cont_id['a'], 0)
        self.assertEquals(cont_id['b'], 1)
        self.assertEquals(cont_id[0], 2)
        
        cont_id.boost(10)
        self.assertEquals(cont_id['a'], 10)
        self.assertEquals(cont_id['b'], 11)
        self.assertEquals(cont_id[0], 12)
        
if __name__ == "__main__":
    unittest.main()
