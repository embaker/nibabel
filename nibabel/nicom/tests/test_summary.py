""" Tests for Stacker
"""
from ..summary import (ElementSummary, VaryingElement, determine_variation,
                       find_axes, slice_to_bitarray, iterable_to_bitarray)

from unittest import TestCase
from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_not_equal, assert_raises)

from bitarray import bitarray
import numpy as np
import itertools
import random
import collections
import six


def test_slice_to_bitarray():
    barray = slice_to_bitarray(slice(1,4), 6)
    assert_equal(bitarray('011100'), barray)
    barray = slice_to_bitarray(slice(2,8,2), 6)
    assert_equal(bitarray('001010'), barray)


def test_iterable_to_bitarray():
    barray = iterable_to_bitarray([0,1,4], 6)
    assert_equal(bitarray('110010'), barray)
    barray = iterable_to_bitarray({3,2,5}, 6)
    assert_equal(bitarray('001101'), barray)
    with assert_raises(IndexError) as cm:
        barray = iterable_to_bitarray([0,2,8], 6)


def fnear(r):
    def func(x, y):
        return abs(x - y) <= r
    return func

#===============================================================================
#
# VaryingElement tests
#
#===============================================================================
class TestVaryingElement(TestCase):

    def test_get(self):
        # without comparision function
        elem = VaryingElement.make_from_list([1, 1, 5, 1])
        assert_equal(elem.get(1), bitarray('1101'))
        assert_equal(elem.get(5), 2)
        assert_equal(elem.get(9), None)
        assert_equal(elem.get(9, default='x'), 'x')
        # with comparision function
        elem = VaryingElement.make_from_list([1, 1, 5, 1],
                                                cmp_func=fnear(1))
        #assert_equal(elem.get(2), bitarray('1101'))
        assert_equal(elem.get(1), bitarray('1101'))
        assert_equal(elem.get(0), bitarray('1101'))
        assert_equal(elem.get(4), 2)
        assert_equal(elem.get(5), 2)
        assert_equal(elem.get(6), 2)
        assert_equal(elem.get(9), None)
        assert_equal(elem.get(9, default='x'), 'x')


    def test__getitem__(self):
        # without comparision function
        elem = VaryingElement.make_from_list([1, 1, 5, 1])
        assert_equal(elem[1], bitarray('1101'))
        assert_equal(elem[5], 2)
        assert_raises(KeyError, elem.__getitem__, 9)
        # with comparision function
        elem = VaryingElement.make_from_list([1, 1, 5, 1],
                                                cmp_func=fnear(1))
        assert_equal(elem[2], bitarray('1101'))
        assert_equal(elem[1], bitarray('1101'))
        assert_equal(elem[0], bitarray('1101'))
        assert_equal(elem[4], 2)
        assert_equal(elem[5], 2)
        assert_equal(elem[6], 2)
        assert_raises(KeyError, elem.__getitem__, 9)


    def test__unsafe_set(self):
        result = VaryingElement(size=4)
        result._unsafe_set(3, 2)
        assert_equal(1, len(result))
        assert_equal(2, result[3])
        result._unsafe_set(3, 1)
        assert_equal(1, len(result))
        assert_equal(1, result[3])


    def test__unsafe_del(self):
        result = VaryingElement(size=6)
        result[1] = bitarray('001101')
        result[2] = bitarray('100010')
        result[3] = 1
        result._unsafe_del(3)
        assert_equal(2, len(result))
        assert_false(3 in result)
        result._unsafe_del(1)
        assert_equal(1, len(result))
        assert_false(1 in result)


    def test__unassign_int(self):
        result = VaryingElement(size=6)
        result[1] = bitarray('011000')
        result[2] = bitarray('000111')
        result[3] = 0

        result._unassign_int(1)
        assert_equal(3, len(result))
        assert_equal(result[1], 2)
        assert_equal(result[2], bitarray('000111'))
        assert_equal(result[3], 0)

        result._unassign_int(4)
        assert_equal(3, len(result))
        assert_equal(result[1], 2)
        assert_equal(result[2], bitarray('000101'))
        assert_equal(result[3], 0)

        result._unassign_int(2)
        assert_equal(2, len(result))
        assert_equal(result[2], bitarray('000101'))
        assert_equal(result[3], 0)

        result._unassign_int(0, ignore_keys={3})
        assert_equal(2, len(result))
        assert_equal(result[2], bitarray('000101'))
        assert_equal(result[3], 0)


    def test__unassign_bitarray(self):
        result = VaryingElement(size=6)
        result[1] = bitarray('011000')
        result[2] = bitarray('000111')
        result[3] = 0

        result._unassign_bitarray(bitarray('001100'), ignore_keys={1,2})
        assert_equal(3, len(result))
        assert_equal(result[1], bitarray('011000'))
        assert_equal(result[2], bitarray('000111'))
        assert_equal(result[3], 0)

        result._unassign_bitarray(bitarray('001100'))
        assert_equal(3, len(result))
        assert_equal(result[1], 1)
        assert_equal(result[2], bitarray('000011'))
        assert_equal(result[3], 0)

        result._unassign_bitarray(bitarray('110000'))
        assert_equal(1, len(result))
        assert_equal(result[2], bitarray('000011'))


    def test_unassign(self):
        result = VaryingElement(size=10)
        result[1] = bitarray('0110000000')
        result[2] = bitarray('0001110000')
        result[3] = bitarray('0000001110')
        result[4] = 0
        result[5] = 9

        result.unassign(slice(3), ignore_keys={1})
        assert_equal(4, len(result))
        assert_equal(result[1], bitarray('0110000000'))
        assert_equal(result[2], bitarray('0001110000'))
        assert_equal(result[3], bitarray('0000001110'))
        assert_equal(result[5], 9)

        result.unassign([1,4], ignore_keys={2})
        assert_equal(4, len(result))
        assert_equal(result[1], 2)
        assert_equal(result[2], bitarray('0001110000'))
        assert_equal(result[3], bitarray('0000001110'))
        assert_equal(result[5], 9)


    def test___setitem__(self):
        result = VaryingElement(size=5)

        result['a'] = bitarray('10101')
        assert_equal(result['a'], bitarray('10101'))

        result['b'] = 1
        assert_equal(result['a'], bitarray('10101'))
        assert_equal(result['b'], 1)

        result['c'] = bitarray('10100')
        assert_equal(result['a'], 4)
        assert_equal(result['b'], 1)
        assert_equal(result['c'], bitarray('10100'))

        result['d'] = bitarray('00011')
        assert_equal(result['b'], 1)
        assert_equal(result['c'], bitarray('10100'))
        assert_equal(result['d'], bitarray('00011'))

        result['b'] = 1
        assert_equal(result['b'], 1)
        assert_equal(result['c'], bitarray('10100'))
        assert_equal(result['d'], bitarray('00011'))

        result['d'] = bitarray('00011')
        assert_equal(result['b'], 1)
        assert_equal(result['c'], bitarray('10100'))
        assert_equal(result['d'], bitarray('00011'))

        result['b'] = 3
        assert_equal(result['b'], 3)
        assert_equal(result['c'], bitarray('10100'))
        assert_equal(result['d'], 4)

        with assert_raises(IndexError) as cm:
            result['e'] = 5

        with assert_raises(ValueError) as cm:
            result['f'] = bitarray('001000')


    def test_assign(self):
        result = VaryingElement(size=8)
        result['a'] = 0
        result['b'] = bitarray('01110000')
        result['c'] = bitarray('00001111')

        result.assign('a', 0)
        assert_equal(result['a'], 0)
        assert_equal(result['b'], bitarray('01110000'))
        assert_equal(result['c'], bitarray('00001111'))

        result.assign('b', bitarray('01110000'))
        assert_equal(result['a'], 0)
        assert_equal(result['b'], bitarray('01110000'))
        assert_equal(result['c'], bitarray('00001111'))

        result.assign('b', bitarray('00000000'))
        assert_equal(result['a'], 0)
        assert_equal(result['b'], bitarray('01110000'))
        assert_equal(result['c'], bitarray('00001111'))

        result.assign('a', slice(1,3))
        assert_equal(result['a'], bitarray('11100000'))
        assert_equal(result['b'], 3)
        assert_equal(result['c'], bitarray('00001111'))

        result.assign('a', {3, 6})
        assert_true('b' not in result)
        assert_equal(result['a'], bitarray('11110010'))
        assert_equal(result['c'], bitarray('00001101'))

        result.assign('d', bitarray('00011000'))
        assert_equal(result['a'], bitarray('11100010'))
        assert_equal(result['c'], bitarray('00000101'))
        assert_equal(result['d'], bitarray('00011000'))

        result.assign('b', 0)
        assert_equal(result['a'], bitarray('01100010'))
        assert_equal(result['b'], 0)
        assert_equal(result['c'], bitarray('00000101'))
        assert_equal(result['d'], bitarray('00011000'))


    def test_resize(self):
        result = VaryingElement(size=3)
        result['a'] = bitarray('101')
        result['b'] = 1

        result.resize(5)
        assert_equal(result._size, 5)
        assert_equal(result['a'], bitarray('10100'))
        assert_equal(result['b'], 1)

        result.resize(3)
        assert_equal(result._size, 3)
        assert_equal(result['a'], bitarray('101'))
        assert_equal(result['b'], 1)

        result.resize(2)
        assert_equal(result._size, 2)
        assert_equal(result['a'], 0)
        assert_equal(result['b'], 1)


    def test_extend(self):
        result = VaryingElement(size=9)
        result['a+a'] = bitarray('001100000')
        result['i+i'] = 1
        result['i+a'] = 0
        result['i+_'] = 8
        result['a+i'] = bitarray('000011000')
        result['a+_'] = bitarray('000000110')

        other = VaryingElement(size=10)
        other['a+a'] = bitarray('0011000000')
        other['i+i'] = 0
        other['i+a'] = bitarray('0000110000')
        other['_+i'] = 1
        other['a+i'] = 6
        other['_+a'] = bitarray('0000000111')

        result.extend(other)

        assert_equal(19, result.size)
        assert_equal(8, len(result))
        assert_equal(bitarray('0011000000011000000'), result['a+a'])
        assert_equal(bitarray('0100000001000000000'), result['i+i'])
        assert_equal(bitarray('1000000000000110000'), result['i+a'])
        assert_equal(8, result['i+_'])
        assert_equal(10, result['_+i'])
        assert_equal(bitarray('0000110000000001000'), result['a+i'])
        assert_equal(bitarray('0000001100000000000'), result['a+_'])
        assert_equal(bitarray('0000000000000000111'), result['_+a'])


    def test_subset(self):
        velem = VaryingElement.make_from_list([1, 3, 2, 1, 2, 3, 1])
        selem = velem.subset([0, 2, 3, 4])
        assert_equal([1, 2, 1, 2], selem.to_list())
        assert_equal([1, 3, 2, 1, 2, 3, 1], velem.to_list())
        # with unfilled values
        velem = VaryingElement.make_from_list([1, 3, None, 1, None, 3, 1])
        selem = velem.subset([0, 2, 3, 4])
        assert_equal([1, None, 1, None], selem.to_list())
        assert_equal([1, 3, None, 1, None, 3, 1], velem.to_list())
        # empty subset
        velem = VaryingElement.make_from_list([None, 3, None, 1, 0, 3, 1])
        selem = velem.subset([0, 2])
        assert_equal([None, None], selem.to_list())
        assert_true(selem.is_empty())
        assert_equal([None, 3, None, 1, 0, 3, 1], velem.to_list())
        # constant subset
        velem = VaryingElement.make_from_list([1, 3, 2, 1, 2, 3, 1])
        selem = velem.subset([0, 3, 6])
        assert_equal([1, 1, 1], selem.to_list())
        assert_true(selem.is_constant())
        assert_equal([1, 3, 2, 1, 2, 3, 1], velem.to_list())


    def test_insert(self):
        result = VaryingElement(size=4)
        result['a'] = bitarray('0011')
        result['b'] = 0
        result['c'] = 1

        result.insert('a', 1)
        assert_equal(5, result.size)
        assert_equal(bitarray('01011'), result['a'])
        assert_equal(0, result['b'])
        assert_equal(2, result['c'])

        result.insert('b', 4)
        assert_equal(6, result.size)
        assert_equal(bitarray('010101'), result['a'])
        assert_equal(bitarray('100010'), result['b'])
        assert_equal(2, result['c'])

        result.insert('d', 1)
        assert_equal(7, result.size)
        assert_equal(bitarray('0010101'), result['a'])
        assert_equal(bitarray('1000010'), result['b'])
        assert_equal(3, result['c'])
        assert_equal(1, result['d'])


    def test_get_key(self):
        def cmp_func(x, y):
            if isinstance(x, float) and isinstance(y, float):
                return abs(x - y) < 0.5
            else:
                return False
        result = VaryingElement(size=5, cmp_func=cmp_func)
        result[1.2] = 4
        key = result.get_key(1.4)
        assert_equal(1.2, key)

        result = VaryingElement(size=5)
        result[1.2] = 4
        key = result.get_key(1.4)
        assert_equal(1.4, key)


    def test_make_from_list(self):
        result = VaryingElement.make_from_list([1,2,1,3,2,'NA',1],
                                               unfilled_value='NA')
        assert_equal(7, result.size)
        assert_equal(result[1], bitarray('1010001'))
        assert_equal(result[2], bitarray('0100100'))
        assert_equal(result[3], 3)
        assert_equal(result.unfilled, bitarray('0000010'))
        # test cretion with a list of length 1
        result = VaryingElement.make_from_list([1])
        assert_equal(1, result.size)
        assert_equal(result[1], 0)
        assert_equal(result.unfilled, None)
        # test empty creation
        result = VaryingElement.make_from_list([0,0,0,0],
                                               unfilled_value=0)
        assert_equal(4, result.size)
        assert_equal(0, len(result))
        assert_true(result.is_empty())
        assert_equal(result.unfilled, bitarray('1111'))


    def test_make_constant(self):
        # size > 1
        velem = VaryingElement.make_constant('x', 4)
        assert_true(velem.is_constant())
        assert_equal(velem['x'], bitarray('1111'))
        # size == 1
        velem = VaryingElement.make_constant('x', size=1)
        assert_true(velem.is_constant())
        assert_equal(velem['x'], 0)


    def test_make_combined(self):
        # test completely filled
        list1 = ['i_', 'ii', 'ib', 'bi', 'bi', 'bb', 'bb', 'b_', 'b_']
        list2 = ['_i', 'ii', 'ib', 'ib', 'bi', 'bb', 'bb', '_b', '_b']
        elems = [VaryingElement.make_from_list(list1),
                 VaryingElement.make_from_list(list2)]
        result = VaryingElement.make_combined(elems)
        expected = VaryingElement.make_from_list(list1 + list2)
        assert_equal(result, expected)
        # test with comparison function
        list1 = [1, 1, 2, 4, 7, 7]
        list2 = [2, 2, 1, 5, 9]
        elems = [VaryingElement.make_from_list(list1),
                 VaryingElement.make_from_list(list2)]
        result = VaryingElement.make_combined(elems, cmp_func=fnear(1))
        full_list = list1 + list2
        expected = VaryingElement.make_from_list(full_list, cmp_func=fnear(1))
        assert_equal(result, expected)
        # test with unfilled values
        list1 = [0, 0, 1, 1, 2]
        list2 = [0, 1, 2, 2]
        elems = [VaryingElement.make_from_list(list1, unfilled_value=0),
                 VaryingElement.make_from_list(list2, unfilled_value=0)]
        result = VaryingElement.make_combined(elems)
        full_list = list1 + list2
        expected = VaryingElement.make_from_list(full_list, unfilled_value=0)
        assert_equal(result, expected)


    def test_clean(self):
        result = VaryingElement(size=5)
        result._unsafe_set('a', bitarray('01010'))
        result._unsafe_set('b', bitarray('00100'))
        result._unsafe_set('c', bitarray('00000'))
        result._unsafe_set('d', 4)
        result._unsafe_set('e', 7)
        result._unsafe_set('f', -1)

        result.clean()
        assert_equal(3, len(result))
        assert_equal(result['a'], bitarray('01010'))
        assert_equal(result['b'], 2)
        assert_equal(result['d'], 4)


    def test_to_list(self):
        result = VaryingElement(size=5)
        result['a'] = bitarray('01100')
        result['b'] = 4

        expected_inverse = ['NA', 'a', 'a', 'NA', 'b']
        result_inverse = result.to_list(unfilled_key='NA')
        assert_equal(expected_inverse, result_inverse)


    def test_which(self):
        result = VaryingElement(size=5)
        result['a'] = bitarray('01100')
        result['b'] = 4

        result_key = result.which(2)
        assert_equal('a', result_key)

        result_key = result.which(4)
        assert_equal('b', result_key)

        result_key = result.which(3, unfilled_key="Unfilled")
        assert_equal("Unfilled", result_key)


    def test_append(self):
        result = VaryingElement(size=5)
        result['a'] = bitarray('01100')
        result['b'] = 4

        result.append('c')
        assert_equal(6, result.size)
        assert_equal(result['a'], bitarray('011000'))
        assert_equal(result['b'], 4)
        assert_equal(result['c'], 5)

        result.append('b')
        assert_equal(7, result.size)
        assert_equal(result['a'], bitarray('0110000'))
        assert_equal(result['b'], bitarray('0000101'))
        assert_equal(result['c'], 5)

        result.append('a')
        assert_equal(8, result.size)
        assert_equal(result['a'], bitarray('01100001'))
        assert_equal(result['b'], bitarray('00001010'))
        assert_equal(result['c'], 5)


    def test_clear(self):
        def cmpf(x, y):
            return x == y
        result = VaryingElement(size=5, cmp_func=cmpf)
        result['a'] = bitarray('01100')
        result['b'] = 4

        result.clear()
        assert_equal(0, len(result))
        assert_equal(5, result.size)
        assert_true(result.cmp_func is cmpf)


    def test_is_empty(self):
        result = VaryingElement(size=7)
        assert_true(result.is_empty())
        result[1] = 3
        assert_false(result.is_empty())


    def test_is_full(self):
        result = VaryingElement(size=7)
        result[1] = bitarray('0011100')
        assert_false(result.is_full())
        result[2] = 1
        result[3] = bitarray('1000011')
        assert_true(result.is_full())


    def test_is_constant(self):
        result = VaryingElement(size=1)
        result[1] = 0
        assert_true(result.is_constant())

        result = VaryingElement(size=7)
        result[1] = bitarray('0011100')
        assert_false(result.is_constant())
        result[2] = 1
        result[3] = bitarray('1000011')
        assert_false(result.is_constant())

        result = VaryingElement(size=7)
        result[1] = bitarray('1111111')
        assert_true(result.is_constant())


    def test_is_square(self):
        result = VaryingElement(size=12)
        result[1] = bitarray('111100000000')
        result[2] = bitarray('000011110000')
        result[3] = bitarray('000000001111')
        assert_true(result.is_square())

        result = VaryingElement(size=12)
        result[1] = bitarray('111100000000')
        result[2] = bitarray('000011100000')
        result[3] = bitarray('000000011111')
        assert_false(result.is_square())


    def test_is_equivalent(self):
        result1 = VaryingElement(size=5)
        result1[1] = bitarray('01100')
        result1[2] = bitarray('00011')
        result1[3] = 0
        assert_true(result1.is_equivalent(result1))

        result2 = VaryingElement(size=5)
        result2['a'] = 0
        result2['b'] = bitarray('01100')
        result2['c'] = bitarray('00011')
        assert_true(result1.is_equivalent(result2))

        result2['b'] = 1
        assert_false(result1.is_equivalent(result2))


    def test_is_complementary(self):
        # v[i] is a n-dimensional array that varies only over index i
        v2 = [VaryingElement.make_from_list(a.flatten())
              for a in np.mgrid[0:3, 0:4]]

        assert_true(v2[0].is_complementary(v2[1]))
        assert_true(v2[1].is_complementary(v2[0]))
        assert_false(v2[0].is_complementary(v2[0]))
        assert_false(v2[1].is_complementary(v2[1]))

        m2 = VaryingElement.make_from_list(([0]*4) + ([1]*8))
        assert_false(v2[0].is_complementary(m2))
        assert_false(v2[1].is_complementary(m2))
        assert_false(m2.is_complementary(v2[0]))
        assert_false(m2.is_complementary(v2[1]))

        v3 = [VaryingElement.make_from_list(a.flatten())
              for a in np.mgrid[0:2, 0:3, 0:4]]

        assert_true(v3[0].is_complementary(v3[1]))
        assert_true(v3[1].is_complementary(v3[0]))


    def test_is_permutation(self):
        # test 1-to-1 mapping
        velem1 = VaryingElement.make_from_list(['a', 'b', 'c', 'd'])
        velem2 = VaryingElement.make_from_list(['b', 'd', 'c', 'a'])
        # any permutation
        assert_true(velem1.is_permutation(velem2))
        # specific permutation
        assert_true(velem1.is_permutation(velem2, [1, 3, 2, 0]))
        assert_false(velem1.is_permutation(velem2, [1, 3, 0, 2]))
        # test unfilled
        velem1 = VaryingElement.make_from_list(['a', None, 'c', 'd'])
        velem2 = VaryingElement.make_from_list([None, 'd', 'c', 'a'])
        # any permutation
        assert_true(velem1.is_permutation(velem2))
        # specific permutation
        assert_true(velem1.is_permutation(velem2, [1, 3, 2, 0]))
        assert_false(velem1.is_permutation(velem2, [1, 3, 0, 2]))
        # test 2-to-1 mapping
        velem1 = VaryingElement.make_from_list(['a', 'b', 'b', 'c'])
        velem2 = VaryingElement.make_from_list(['b', 'c', 'b', 'a'])
        # any permutation
        assert_true(velem1.is_permutation(velem2))
        # specific permutation
        assert_true(velem1.is_permutation(velem2, [1, 3, 2, 0]))
        assert_true(velem1.is_permutation(velem2, [2, 3, 1, 0]))
        assert_false(velem1.is_permutation(velem2, [1, 3, 0, 2]))


    def test_find_permutation(self):
        # test 1-to-1 mapping
        velem1 = VaryingElement.make_from_list(['a', 'b', 'c', 'd'])
        velem2 = VaryingElement.make_from_list(['b', 'd', 'c', 'a'])
        assert_equal(velem1.find_permutation(velem2), [1, 3, 2, 0])
        # test unfilled
        velem1 = VaryingElement.make_from_list(['a', None, 'c', 'd'])
        velem2 = VaryingElement.make_from_list([None, 'd', 'c', 'a'])
        assert_equal(velem1.find_permutation(velem2), [1, 3, 2, 0])
        # test 2-to-1 mapping
        velem1 = VaryingElement.make_from_list(['a', 'b', 'b', 'c'])
        velem2 = VaryingElement.make_from_list(['b', 'c', 'b', 'a'])
        result_2to1 = velem1.find_permutation(velem2)
        assert_true(result_2to1 == [1, 3, 2, 0] or
                    result_2to1 == [2, 3, 1, 0])
        # test no permutation
        velem1 = VaryingElement.make_from_list(['a', 'b', 'c', 'd'])
        velem2 = VaryingElement.make_from_list(['b', 'e', 'c', 'a'])
        assert_true(velem1.find_permutation(velem2) is None)


#===============================================================================
#
# other
#
#===============================================================================


def test_find_axes():
    axes_data = np.mgrid[0:3, 0:4, 0:5]
    size = np.prod(axes_data.shape[1:])

    var_elems = {}

    other_data = np.ones(size)
    other_data[:size/2] = 2
    other_data[0] = 3
    other_data[-1] = 4
    var_elems['a'] = VaryingElement.make_from_list(other_data)
    var_elems['b'] = VaryingElement.make_from_list(np.ravel(axes_data[0]))

    np.random.shuffle(other_data)
    var_elems['c'] = VaryingElement.make_from_list(other_data)
    var_elems['d'] = VaryingElement.make_from_list(np.ravel(axes_data[1]))
    var_elems['e'] = VaryingElement.make_from_list(np.ravel(axes_data[2]))
    var_elems['f'] = VaryingElement.make_from_list(np.ravel(axes_data[1]))

    axes = find_axes(['a','b','c','e','f'], var_elems)
    assert_equal(('b','e','f'), axes)


def test_determine_variation():
    n = 4
    shape = np.arange(2, n+2)
    size = np.prod(shape)
    vary = np.mgrid[tuple(slice(0, i, 1) for i in shape)]
    axes = [VaryingElement.make_from_list(v.flatten())
            for v in vary]
    single_axis = VaryingElement.make_from_list(np.arange(size))
    const_elem = VaryingElement.make_from_list(np.ones(size))

    # multiple axes with varying element
    for n_axes in range(1, len(axes)-1):
        for var_axes in itertools.combinations(range(len(axes)), n_axes):
            inv = np.ravel(np.sum(vary[np.array(var_axes)], axis=0))
            elem = VaryingElement.make_from_list(inv)
            result = determine_variation(elem, axes)
            assert_equal(set(var_axes), set(result))

    # multiple axes with a constant element
    result = determine_variation(const_elem, axes)
    assert_equal(0, len(result))

    # single axis with a constant element
    result = determine_variation(const_elem, [single_axis])
    assert_equal(0, len(result))

    # single axis with a varying element
    result = determine_variation(axes[0], [single_axis])
    assert_equal({0}, set(result))


#===============================================================================
#
# DicomSummary tests
#
#===============================================================================
def make_data_dict(vals, rkeys, ckeys, unfilled_value=None):
    result = dict((ckey, dict()) for ckey in ckeys)
    for rkey, row in zip(rkeys, vals):
        for ckey, val in zip(ckeys, row):
            if val != unfilled_value:
                result[ckey][rkey] = val
    return result


def assert_equal_summaries(result, expected):
    assert_equal(result.size, expected.size)
    assert_equal(set(result._const_elems.keys()),
                 set(expected._const_elems.keys()))
    assert_equal(set(result._varying_elems.keys()),
                 set(expected._varying_elems.keys()))
    msg = "Constant Element [%s]:\n  expect: %s\n  result: %s"
    for key, val in six.iteritems(result._const_elems):
        assert_equal(val, expected[key], msg % (key, expected[key], val))
    msg = "Varing Element [%s]:\n  expect: %s\n  result: %s"
    for key, val in six.iteritems(result._varying_elems):
        assert_equal(val, expected[key], msg % (key, expected[key], val))


class TestElementSummary(TestCase):

    def test_make_from_lists(self):
        data = {'a' : [5, 5, 5, 5, 5, 5, 5],
                'b' : [1, 1, 2, 2, 1, 1, 1],
                'c' : [3, 4, 3, 3, 4, 4, 3],
                'd' : [0, 0, 1, 1, 0, 2, 0],
                'e' : [5, 0, 1, 1, 0, 4, 3],
                'f' : [0, 0, 0, 0, 0, 0, 0]}
        cmps = {'b' : fnear(1),
                'e' : fnear(1)}
        uval = 0
        summary = ElementSummary.make_from_lists(data,
                                                 unfilled_value=uval,
                                                 compare_rules=cmps)
        # check dimensions
        assert_equal(summary.size, 7)
        assert_equal(len(summary), 5)
        # check constant elements
        assert_equal(summary['a'], 5)
        assert_true(summary['b'] in (1, 2))
        # check varying elements
        assert_equal(summary['c'],
                     VaryingElement.make_from_list(data['c'],
                                                   unfilled_value=uval))
        assert_equal(summary['d'],
                     VaryingElement.make_from_list(data['d'],
                                                   unfilled_value=uval))
        assert_equal(summary['e'],
                     VaryingElement.make_from_list(data['e'],
                                                   unfilled_value=uval,
                                                   cmp_func=cmps['e']))
        # check empty elements
        assert_false('f' in summary)
        # test size check
        data = {'a' : [1] * 4,
                'b' : [1] * 5}
        assert_raises(ValueError, ElementSummary.make_from_lists, data)


    # def test_remove(self):
    #
    #     result = ElementSummary()
    #     result._tags = ['0','1','2','3','4','5','6']
    #     result._const_elems = {}
    #     result._varying_elems["VKey1"] = VaryingElement(size=7)
    #     result._varying_elems["VKey1"][2.0] = bitarray('0001010')
    #     result._varying_elems["VKey1"][6.2] = bitarray('0110000')
    #     result._varying_elems["VKey1"][7.8] = bitarray('1000101')
    #
    #     result._varying_elems["VKey2"] = VaryingElement(size=7)
    #     result._varying_elems["VKey2"][11] = bitarray('1110010')
    #     result._varying_elems["VKey2"][12] = bitarray('0001001')
    #     result._varying_elems["VKey2"][14] = 4
    #
    #     result._varying_elems["VKey3"] = VaryingElement(size=7)
    #     result._varying_elems["VKey3"][23] = 3
    #     result._varying_elems["VKey3"][25] = bitarray('1110101')
    #     result._varying_elems["VKey3"][21] = 5
    #
    #     result.remove(['3','5'])
    #
    #     assert_equal(['0','1','2','4','6'], result._tags)
    #     assert_equal({"VKey3" : 25}, result._const_elems)
    #     assert_equal({"VKey1", "VKey2"}, set(result._varying_elems.keys()))
    #
    #     assert_equal(bitarray('01100'), result._varying_elems["VKey1"][6.2])
    #     assert_equal(bitarray('10011'), result._varying_elems["VKey1"][7.8])
    #
    #     assert_equal(bitarray('11100'), result._varying_elems["VKey2"][11])
    #     assert_equal(4, result._varying_elems["VKey2"][12])
    #     assert_equal(3, result._varying_elems["VKey2"][14])


    def test_subset(self):
        parent_data = [('c->c', [5, 5, 5, 5, 5, 5, 5]),
                       ('v->c', [1, 1, 2, 2, 1, 1, 1]),
                       ('v->v', [3, 4, 3, 3, 4, 4, 3]),
                       ('v->_', [None, None, 1, 1, None, 2, None])]
        parent = ElementSummary.make_from_lists(parent_data)
        # multi index subset
        indices = [0, 1, 4, 6]
        child_data = [('c->c', [5, 5, 5, 5]),
                      ('v->c', [1, 1, 1, 1]),
                      ('v->v', [3, 4, 4, 3])]
        child = ElementSummary.make_from_lists(child_data)
        assert_equal(parent.subset(indices), child)
        assert_equal(parent, ElementSummary.make_from_lists(parent_data))
        # single index subset
        indices = [1]
        child_data = [('c->c', [5]),
                      ('v->c', [1]),
                      ('v->v', [4])]
        child = ElementSummary.make_from_lists(child_data)
        assert_equal(parent.subset(indices), child)
        assert_equal(parent, ElementSummary.make_from_lists(parent_data))


    def test_split(self):
        split_key = 'b'
        parent_data = {'a' : [5, 5, 5, 5, 5, 5, 5, 5, 5],
                       'b' : [1, 1, 2, 2, 2, 1, 0, 1, 0],
                       'c' : [3, 4, 3, 3, 4, 4, 3, 3, 4],
                       'd' : [0, 0, 0, 2, 2, 0, 2, 0, 1],
                       'e' : [1, 1, 0, 0, 0, 1, 2, 1, 2]}
        child_data = {1 : {'a' : [5, 5, 5, 5],
                           'b' : [1, 1, 1, 1],
                           'c' : [3, 4, 4, 3],
                           'e' : [1, 1, 1, 1]},
                      2 : {'a' : [5, 5, 5],
                           'b' : [2, 2, 2],
                           'c' : [3, 3, 4],
                           'd' : [0, 2, 2]},
                      0 : {'a' : [5, 5],
                           'c' : [3, 4],
                           'd' : [2, 1],
                           'e' : [2, 2]},
                     }
        parent = ElementSummary.make_from_lists(parent_data,
                                                unfilled_value=0)
        child = {key : ElementSummary.make_from_lists(val, unfilled_value=0)
                 for key, val in six.iteritems(child_data)}
        # check using default default
        result = dict(item for item in parent.split(split_key))
        assert_equal(result[1], child[1])
        assert_equal(result[2], child[2])
        assert_equal(result[None], child[0])
        # check using given default
        result = dict(item for item in parent.split(split_key, default='x'))
        assert_equal(result[1], child[1])
        assert_equal(result[2], child[2])
        assert_equal(result['x'], child[0])


    def test_group(self):
        group_keys = ('b', 'f')
        parent_data = {'a' : [5, 5, 5, 5, 5, 5, 5, 5, 5],
                       'b' : [1, 1, 2, 2, 2, 1, 0, 1, 0],
                       'c' : [3, 4, 3, 3, 4, 4, 3, 3, 4],
                       'd' : [0, 0, 0, 2, 2, 0, 2, 0, 1],
                       'e' : [1, 1, 0, 0, 0, 1, 2, 1, 2],
                       'f' : [9, 7, 8, 8, 8, 7, 9, 7, 9]}
        child_data = {(1, 7) : {'a' : [5, 5, 5],
                                'b' : [1, 1, 1],
                                'c' : [4, 4, 3],
                                'e' : [1, 1, 1],
                                'f' : [7, 7, 7]},
                      (1, 9) : {'a' : [5],
                                'b' : [1],
                                'c' : [3],
                                'e' : [1],
                                'f' : [9]},
                      (2, 8) : {'a' : [5, 5, 5],
                                'b' : [2, 2, 2],
                                'c' : [3, 3, 4],
                                'd' : [0, 2, 2],
                                'f' : [8, 8, 8]},
                      (0, 9) : {'a' : [5, 5],
                                'c' : [3, 4],
                                'd' : [2, 1],
                                'e' : [2, 2],
                                'f' : [9, 9]}}
        parent = ElementSummary.make_from_lists(parent_data,
                                                unfilled_value=0)
        child = {key : ElementSummary.make_from_lists(val, unfilled_value=0)
                 for key, val in six.iteritems(child_data)}
        result = dict(item for item in parent.group(group_keys))
        assert_equal(result[(1, 7)], child[(1, 7)])
        assert_equal(result[(1, 9)], child[(1, 9)])
        assert_equal(result[(2, 8)], child[(2, 8)])
        assert_equal(result[(None, 9)], child[(0, 9)])
        # none None default
        result = dict(item for item in parent.group(group_keys, default='x'))
        assert_equal(result[(1, 7)], child[(1, 7)])
        assert_equal(result[(1, 9)], child[(1, 9)])
        assert_equal(result[(2, 8)], child[(2, 8)])
        assert_equal(result[('x', 9)], child[(0, 9)])


    def test_find_axes(self):
        # make fake data
        axes_names = ('x', 'y', 'z')
        axes_grid = np.mgrid[0:3, 0:4, 0:5]
        data = {key : np.ravel(axes_grid[i])
                for i, key in enumerate(axes_names)}
        size = len(data['x'])
        data['a'] = np.array([1, 2, 3] + ([0] * (size - 3)))
        data['b'] = np.array([1, 1, 2, 2, 3] + ([0] * (size - 5)))
        data['c'] = np.array([5] * size)
        # shuffle data
        mix_idx = np.random.permutation(size)
        for key, val in six.iteritems(data):
            data[key] = val[mix_idx]
        # create summary containing one set of axes
        summary = ElementSummary.make_from_lists(data)
        # test finding no axes from guess keys
        guess_keys = ['y', 'z', 'a', 'b', 'c']
        random.shuffle(guess_keys)
        axes_keys = summary.find_axes(guess_keys)
        assert_true(axes_keys is None)
        # test finding axes with input previous elements with exact guess_keys
        prv_elems = (summary.get('x'),)
        guess_keys = ['y', 'z', 'a', 'b', 'c']
        expected_axes_keys = {'y', 'z'}
        returned_axes_keys = set(summary.find_axes(guess_keys, prv_elems))
        assert_equal(expected_axes_keys, returned_axes_keys)
        # test finding axes with input previous elements with guess_keys
        # having axis already given in previous elemnts
        prv_elems = (summary.get('x'),)
        guess_keys = ['x', 'y', 'z', 'a', 'b', 'c']
        expected_axes_keys = {'y', 'z'}
        returned_axes_keys = set(summary.find_axes(guess_keys, prv_elems))
        assert_equal(expected_axes_keys, returned_axes_keys)
        # test finding all axes
        guess_keys = ['b', 'a', 'y', 'c', 'z', 'x']
        expected_axes_keys = {'x', 'y', 'z'}
        returned_axes_keys = set(summary.find_axes(guess_keys))
        assert_equal(expected_axes_keys, returned_axes_keys)
        # test finding axes when there is a duplicate of an axis
        data['w'] = data['x']
        summary = ElementSummary.make_from_lists(data)
        guess_keys = ['w', 'b', 'a', 'y', 'c', 'z', 'x']
        expected_axes_keys = {'w', 'y', 'z'}
        returned_axes_keys = set(summary.find_axes(guess_keys))
        assert_equal(expected_axes_keys, returned_axes_keys)


    def test_resize(self):
        # resieze from empty
        summary = ElementSummary()
        summary.resize(5)
        assert_equal(summary.size, 5)
        # increase resize of non-empty
        summary = ElementSummary.make_from_lists({'a' : [0, 1, 0, 1]})
        assert_equal(summary.size, 4)
        summary.resize(6)
        assert_equal(summary.size, 6)
        assert_equal(summary['a'].size, 6)
        # increase resize of non-empty
        summary = ElementSummary.make_from_lists({'a' : [0, 1, 0, 1]})
        assert_equal(summary.size, 4)
        summary.resize(2)
        assert_equal(summary.size, 2)
        assert_equal(summary['a'].size, 2)


    def test__append_summary(self):
        initial_size = 7
        append_size = 6
        initial_data = {'v,v' : [0, 1, 0, 2, 2, 0, 5],
                        'c,c' : [0, 0, 0, 0, 0, 0, 0],
                        'c,v' : [1, 1, 1, 1, 1, 1, 1],
                        'v,c' : [1, 0, 2, 2, None, 1, 2],
                        'v,_' : [0, 1, 3, None, 2, 2, 3],
                        'c,_' : [3, 3, 3, 3, 3, 3, 3],
                        'cmp:v,v' : [0, 3, 3, 6, 9, 9, 9],
                        'cmp:c,v' : [6, 6, 6, 6, 6, 6, 6],
                        'cmp:v,c' : [5, 5, 1, 2, None, 5, 5]}
        append_data = {'v,v' : [0, 1, 1, 1, 0, 3],
                       'c,c' : [0, 0, 0, 0, 0, 0],
                       'c,v' : [1, 0, 2, 0, 2, 2],
                       'v,c' : [1, 1, 1, 1, 1, 1],
                       '_,v' : [1, 1, None, 2, None, 1],
                       '_,c' : [5, 5, 5, 5, 5, 5],
                       'cmp:v,v' : [7, 4, 1, 1, 10, 10],
                       'cmp:c,v' : [5, 7, 1, 2, None, 7],
                       'cmp:v,c' : [6, 6, 6, 6, 6, 6]}
        expected_data = {'v,v' : initial_data['v,v'] + append_data['v,v'],
                         'c,c' : initial_data['c,c'] + append_data['c,c'],
                         'c,v' : initial_data['c,v'] + append_data['c,v'],
                         'v,c' : initial_data['v,c'] + append_data['v,c'],
                         'v,_' : initial_data['v,_'] + [None]*append_size,
                         'c,_' : initial_data['c,_'] + [None]*append_size,
                         '_,v' : [None]*initial_size + append_data['_,v'],
                         '_,c' : [None]*initial_size + append_data['_,c'],
                        'cmp:v,v' : [0, 3, 3, 6, 9, 9, 9, 6, 3, 0, 0, 9, 9],
                        'cmp:c,v' : [6, 6, 6, 6, 6, 6, 6, 5, 7, 1, 2, None, 7],
                        'cmp:v,c' : [5, 5, 1, 2, None, 5, 5, 5, 5, 5, 5, 5, 5]}
        cmp_rules = {'cmp:v,v' : fnear(1),
                     'cmp:c,v' : fnear(1),
                     'cmp:v,c' : fnear(1)}
        append = ElementSummary.make_from_lists(append_data)
        expected = ElementSummary.make_from_lists(expected_data,
                                                     compare_rules=cmp_rules)
        result = ElementSummary.make_from_lists(initial_data,
                                                   compare_rules=cmp_rules)
        result.resize(initial_size + append_size)
        result._append_summary(initial_size, append)
        assert_equal_summaries(result, expected)


    def test_append(self):
        summary = ElementSummary(compare_rules={'c' : lambda x,y: abs(x-y) < 2})
        # first append
        summary.append({'a':2, 'b':0, 'c':3, 'd':4})
        assert_equal(summary.size, 1)
        assert_equal(set(summary._const_elems.keys()), {'a', 'b', 'c', 'd'})
        assert_equal(summary['a'], 2)
        assert_equal(summary['b'], 0)
        assert_equal(summary['c'], 3)
        assert_equal(summary['d'], 4)
        # second append
        summary.append({'a':1, 'b':0, 'c':4, 'd':0})
        assert_equal(summary.size, 2)
        assert_equal(set(summary._const_elems.keys()), {'b', 'c'})
        assert_equal(set(summary._varying_elems.keys()), {'a', 'd'})
        assert_equal(summary['a'], VaryingElement.make_from_list([2, 1]))
        assert_equal(summary['b'], 0)
        assert_equal(summary['c'], 3)
        assert_equal(summary['d'], VaryingElement.make_from_list([4, 0]))
        # third append
        summary.append({'b':0, 'c':2, 'd':0, 'e':5})
        assert_equal(summary.size, 3)
        assert_equal(set(summary._const_elems.keys()), {'b', 'c'})
        assert_equal(summary['b'], 0)
        assert_equal(summary['c'], 3)
        assert_equal(set(summary._varying_elems.keys()), {'a', 'd', 'e'})
        assert_equal(summary['a'],
                     VaryingElement.make_from_list([2, 1, None]))
        assert_equal(summary['d'],
                     VaryingElement.make_from_list([4, 0, 0]))
        assert_equal(summary['e'],
                     VaryingElement.make_from_list([None, None, 5]))
        # test appending of another Element Summary
        initial_data = {'a' : [0, 1, 0, 2, 2, 0],
                        'b' : [0, 0, 0, 0, 0, 0],
                        'c' : [1, 1, 1, 1, 1, 1],
                        'd' : [1, 0, 2, 2, 1, 1],
                        'e' : [0, 1, 3, None, 2, 2],
                        'f' : [1, 1, None, None, None, 1]}
        extended_data = {'a' : [0, 1, 1, 1, 0],
                         'b' : [0, 0, 0, 0, 0],
                         'c' : [1, 0, 2, 0, 2],
                         'd' : [1, 1, 1, 1, 1],
                         'e' : [1, 1, None, None, None],
                         'g' : [None, None, 1, 2, 1]}
        full_data = {'a' : initial_data['a'] + extended_data['a'],
                     'b' : initial_data['b'] + extended_data['b'],
                     'c' : initial_data['c'] + extended_data['c'],
                     'd' : initial_data['d'] + extended_data['d'],
                     'e' : initial_data['e'] + extended_data['e'],
                     'f' : initial_data['f'] + ([None]*5),
                     'g' : ([None]*6) + extended_data['g']}
        expected = ElementSummary.make_from_lists(full_data)
        appended = ElementSummary.make_from_lists(initial_data)
        appended.append(ElementSummary.make_from_lists(extended_data))
        assert_equal(appended.size, 11)
        assert_equal(appended, expected)


    def test_extend(self):
        def fake_inner(data, idx, unfilled_value=None):
            for key, val in six.iteritems(data):
                if val[idx] != unfilled_value:
                    yield key, val[idx]
        def fake_outer(data, size, unfilled_value=None):
            for idx in range(size):
                yield fake_inner(data, idx, unfilled_value)
        initial_data = {'a' : [0, 1, 0, 2, 2, 0],
                        'b' : [0, 0, 0, 0, 0, 0],
                        'c' : [1, 1, 1, 1, 1, 1],
                        'd' : [1, 0, 2, 2, 1, 1],
                        'e' : [0, 1, 3, None, 2, 2],
                        'f' : [1, 1, None, None, None, 1]}
        extended_data = {'a' : [0, 1, 1, 1, 0],
                         'b' : [0, 0, 0, 0, 0],
                         'c' : [1, 0, 2, 0, 2],
                         'd' : [1, 1, 1, 1, 1],
                         'e' : [1, 1, None, None, None],
                         'g' : [None, None, 1, 2, 1]}
        full_data = {'a' : initial_data['a'] + extended_data['a'],
                     'b' : initial_data['b'] + extended_data['b'],
                     'c' : initial_data['c'] + extended_data['c'],
                     'd' : initial_data['d'] + extended_data['d'],
                     'e' : initial_data['e'] + extended_data['e'],
                     'f' : initial_data['f'] + ([None]*5),
                     'g' : ([None]*6) + extended_data['g']}
        # test inital extend on empty summary
        summary = ElementSummary()
        summary.extend(fake_outer(initial_data, 6))
        # test size
        assert_equal(summary.size, 6)
        # test constant elements
        assert_equal(set(summary._const_elems.keys()), {'b', 'c'})
        assert_equal(summary['b'], 0)
        assert_equal(summary['c'], 1)
        # test varying elements
        varying_keys = {'a','d','e','f'}
        assert_equal(set(summary._varying_elems.keys()), varying_keys)
        for key in varying_keys:
            true_elem = VaryingElement.make_from_list(initial_data[key])
            assert_equal(summary[key], true_elem)
        # test addtional extend_data
        summary.extend(fake_outer(extended_data, 5))
        # test size
        assert_equal(summary.size, 11)
        # test constant elements
        assert_equal(set(summary._const_elems.keys()), {'b'})
        assert_equal(summary['b'], 0)
        # test varying elements
        varying_keys = {'a','c','d','e','f','g'}
        assert_equal(set(summary._varying_elems.keys()), varying_keys)
        for key in varying_keys:
            true_elem = VaryingElement.make_from_list(full_data[key])
            assert_equal(summary[key], true_elem)
        # test using guess_size
        # test inital extend on empty summary
        true_summary = ElementSummary.make_from_lists(initial_data)
        # high guess
        summary = ElementSummary()
        summary.extend(fake_outer(initial_data, 6), guess_size=9)
        assert_equal(summary, true_summary)
        # exact guess
        summary = ElementSummary()
        summary.extend(fake_outer(initial_data, 6), guess_size=6)
        assert_equal(summary, true_summary)
        # low guess
        summary = ElementSummary()
        summary.extend(fake_outer(initial_data, 6), guess_size=4)
        assert_equal(summary, true_summary)
        # test addtional extend_data
        true_summary = ElementSummary.make_from_lists(full_data)
        # high guess
        summary = ElementSummary.make_from_lists(initial_data)
        summary.extend(fake_outer(extended_data, 5), guess_size=14)
        assert_equal(summary, true_summary)
        # exact guess
        summary = ElementSummary.make_from_lists(initial_data)
        summary.extend(fake_outer(extended_data, 5), guess_size=11)
        assert_equal(summary, true_summary)
        # low guess
        summary = ElementSummary.make_from_lists(initial_data)
        summary.extend(fake_outer(extended_data, 5), guess_size=9)
        assert_equal(summary, true_summary)
        # test the inclusion of summaries in input sequence
        start_data = {'a' : [0, 1, 0],
                      'b' : [0, 0, 0],
                      'c' : [1, 1, 1],
                      'd' : [2, 2, 2],
                      'e' : [0, 1, 3],
                      'f' : [1, 1, None]}
        middle_data = {'a' : [2, 0, 1, 1, 1],
                       'b' : [0, 0, 0, 0, 0],
                       'c' : [1, 1, 1, 1, 1],
                       'd' : [2, 1, 3, 2, 1],
                       'e' : [None, 2, 2, 1, 1],
                       'f' : [None, None, 1, None, None]}
        end_data = {'a' : [0, 1, 1, 2, 0],
                    'b' : [0, 0, 0, 0, 0],
                    'c' : [1, 0, 2, 0, 2],
                    'd' : [2, 2, 2, 2, 2],
                    'e' : [1, 1, None, None, None],
                    'g' : [None, None, 1, 2, 1]}
