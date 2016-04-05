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

#===============================================================================
#
# VaryingElement tests
#
#===============================================================================
class TestVaryingElement(TestCase):

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


    def test__convert_to_array(self):
        result = VaryingElement(size=5)

        barr = bitarray('00101')
        result['a'] = barr
        result._convert_to_array('a')
        assert_equal(result['a'], barr)

        result['b'] = 1
        result._convert_to_array('b')
        assert_equal(result['b'], bitarray('01000'))

        result._convert_to_array('c')
        assert_equal(result['c'], bitarray('00000'))


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


    def test_reduce(self):
        result = VaryingElement(size=9)
        result[1] = bitarray('001100100')
        result[2] = 1
        result[3] = bitarray('100010000')
        result[4] = 5
        result[5] = bitarray('000000011')

        result.reduce([2, 3, 5, 6, 7])

        assert_equal(5, result.size)
        assert_equal(3, len(result))
        assert_equal(bitarray('11010'), result[1])
        assert_equal(2, result[4])
        assert_equal(4, result[5])


    def test_subset(self):
        velem = VaryingElement.make_from_inverse([1, 3, 2, 1, 2, 3, 1])
        selem = velem.subset([0, 2, 3, 4])
        assert_equal([1, 2, 1, 2], selem.inverse())
        # with unfilled values
        velem = VaryingElement.make_from_inverse([1, 3, None, 1, None, 3, 1])
        selem = velem.subset([0, 2, 3, 4])
        assert_equal([1, None, 1, None], selem.inverse())
        # empty subset
        velem = VaryingElement.make_from_inverse([None, 3, None, 1, 0, 3, 1])
        selem = velem.subset([0, 2])
        assert_equal([None, None], selem.inverse())
        assert_true(selem.is_empty())
        # constant subset
        velem = VaryingElement.make_from_inverse([1, 3, 2, 1, 2, 3, 1])
        selem = velem.subset([0, 3, 6])
        assert_equal([1, 1, 1], selem.inverse())
        assert_true(selem.is_constant())


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


    def test__find_repr_key(self):
        def cmp_func(x, y):
            if isinstance(x, float) and isinstance(y, float):
                return abs(x - y) < 0.5
            else:
                return False
        result = VaryingElement(size=5, cmp_func=cmp_func)
        result[1.2] = 4
        key = result._find_repr_key(1.4)
        assert_equal(1.2, key)

        result = VaryingElement(size=5)
        result[1.2] = 4
        key = result._find_repr_key(1.4)
        assert_equal(1.4, key)


    def test_make_from_inverse(self):
        result = VaryingElement.make_from_inverse([1,2,1,3,2,'NA',1],
                                                  unfilled_key='NA')
        assert_equal(7, result.size)
        assert_equal(result[1], bitarray('1010001'))
        assert_equal(result[2], bitarray('0100100'))
        assert_equal(result[3], 3)

        result = VaryingElement.make_from_inverse([1])
        assert_equal(1, result.size)
        assert_equal(result[1], 0)


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


    def test_inverse(self):
        result = VaryingElement(size=5)
        result['a'] = bitarray('01100')
        result['b'] = 4

        expected_inverse = ['NA', 'a', 'a', 'NA', 'b']
        result_inverse = result.inverse(unfilled_key='NA')
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


    def test_change_compare_function(self):
        def cmpf(x, y):
            return abs(x-y) < 0.5
        result = VaryingElement(size=7)

        result[1.0] = bitarray('1010000')
        result[1.2] = 1
        result[1.5] = 3
        result[3.0] = bitarray('0000111')

        result.change_compare_function(cmpf)

        assert_equal(2, len(result))
        assert_equal(bitarray('0000111'), result[3.0])

        key = result._find_repr_key(1.2)
        assert_true(key in (1.0, 1.2, 1.5))
        assert_equal(bitarray('1111000'), result[key])


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
        v2 = [VaryingElement.make_from_inverse(a.flatten())
              for a in np.mgrid[0:3, 0:4]]

        assert_true(v2[0].is_complementary(v2[1]))
        assert_true(v2[1].is_complementary(v2[0]))
        assert_false(v2[0].is_complementary(v2[0]))
        assert_false(v2[1].is_complementary(v2[1]))

        m2 = VaryingElement.make_from_inverse(([0]*4) + ([1]*8))
        assert_false(v2[0].is_complementary(m2))
        assert_false(v2[1].is_complementary(m2))
        assert_false(m2.is_complementary(v2[0]))
        assert_false(m2.is_complementary(v2[1]))

        v3 = [VaryingElement.make_from_inverse(a.flatten())
              for a in np.mgrid[0:2, 0:3, 0:4]]

        assert_true(v3[0].is_complementary(v3[1]))
        assert_true(v3[1].is_complementary(v3[0]))


    def test_is_permutation(self):
        # test 1-to-1 mapping
        velem1 = VaryingElement.make_from_inverse(['a', 'b', 'c', 'd'])
        velem2 = VaryingElement.make_from_inverse(['b', 'd', 'c', 'a'])
        # any permutation
        assert_true(velem1.is_permutation(velem2))
        # specific permutation
        assert_true(velem1.is_permutation(velem2, [1, 3, 2, 0]))
        assert_false(velem1.is_permutation(velem2, [1, 3, 0, 2]))
        # test unfilled
        velem1 = VaryingElement.make_from_inverse(['a', None, 'c', 'd'])
        velem2 = VaryingElement.make_from_inverse([None, 'd', 'c', 'a'])
        # any permutation
        assert_true(velem1.is_permutation(velem2))
        # specific permutation
        assert_true(velem1.is_permutation(velem2, [1, 3, 2, 0]))
        assert_false(velem1.is_permutation(velem2, [1, 3, 0, 2]))
        # test 2-to-1 mapping
        velem1 = VaryingElement.make_from_inverse(['a', 'b', 'b', 'c'])
        velem2 = VaryingElement.make_from_inverse(['b', 'c', 'b', 'a'])
        # any permutation
        assert_true(velem1.is_permutation(velem2))
        # specific permutation
        assert_true(velem1.is_permutation(velem2, [1, 3, 2, 0]))
        assert_true(velem1.is_permutation(velem2, [2, 3, 1, 0]))
        assert_false(velem1.is_permutation(velem2, [1, 3, 0, 2]))


    def test_find_permutation(self):
        # test 1-to-1 mapping
        velem1 = VaryingElement.make_from_inverse(['a', 'b', 'c', 'd'])
        velem2 = VaryingElement.make_from_inverse(['b', 'd', 'c', 'a'])
        assert_equal(velem1.find_permutation(velem2), [1, 3, 2, 0])
        # test unfilled
        velem1 = VaryingElement.make_from_inverse(['a', None, 'c', 'd'])
        velem2 = VaryingElement.make_from_inverse([None, 'd', 'c', 'a'])
        assert_equal(velem1.find_permutation(velem2), [1, 3, 2, 0])
        # test 2-to-1 mapping
        velem1 = VaryingElement.make_from_inverse(['a', 'b', 'b', 'c'])
        velem2 = VaryingElement.make_from_inverse(['b', 'c', 'b', 'a'])
        result_2to1 = velem1.find_permutation(velem2)
        assert_true(result_2to1 == [1, 3, 2, 0] or
                    result_2to1 == [2, 3, 1, 0])
        # test no permutation
        velem1 = VaryingElement.make_from_inverse(['a', 'b', 'c', 'd'])
        velem2 = VaryingElement.make_from_inverse(['b', 'e', 'c', 'a'])
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
    var_elems['a'] = VaryingElement.make_from_inverse(other_data)
    var_elems['b'] = VaryingElement.make_from_inverse(np.ravel(axes_data[0]))

    np.random.shuffle(other_data)
    var_elems['c'] = VaryingElement.make_from_inverse(other_data)
    var_elems['d'] = VaryingElement.make_from_inverse(np.ravel(axes_data[1]))
    var_elems['e'] = VaryingElement.make_from_inverse(np.ravel(axes_data[2]))
    var_elems['f'] = VaryingElement.make_from_inverse(np.ravel(axes_data[1]))

    axes = find_axes(['a','b','c','e','f'], var_elems)
    assert_equal(('b','e','f'), axes)


def test_determine_variation():
    n = 4
    shape = np.arange(2, n+2)
    size = np.prod(shape)
    vary = np.mgrid[tuple(slice(0, i, 1) for i in shape)]
    axes = [VaryingElement.make_from_inverse(v.flatten())
            for v in vary]
    single_axis = VaryingElement.make_from_inverse(np.arange(size))
    const_elem = VaryingElement.make_from_inverse(np.ones(size))

    # multiple axes with varying element
    for n_axes in range(1, len(axes)-1):
        for var_axes in itertools.combinations(range(len(axes)), n_axes):
            inv = np.ravel(np.sum(vary[np.array(var_axes)], axis=0))
            elem = VaryingElement.make_from_inverse(inv)
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

class TestElementSummary(TestCase):

    def test_remove(self):

        result = ElementSummary()
        result._tags = ['0','1','2','3','4','5','6']
        result._const_elems = {}
        result._varying_elems["VKey1"] = VaryingElement(size=7)
        result._varying_elems["VKey1"][2.0] = bitarray('0001010')
        result._varying_elems["VKey1"][6.2] = bitarray('0110000')
        result._varying_elems["VKey1"][7.8] = bitarray('1000101')

        result._varying_elems["VKey2"] = VaryingElement(size=7)
        result._varying_elems["VKey2"][11] = bitarray('1110010')
        result._varying_elems["VKey2"][12] = bitarray('0001001')
        result._varying_elems["VKey2"][14] = 4

        result._varying_elems["VKey3"] = VaryingElement(size=7)
        result._varying_elems["VKey3"][23] = 3
        result._varying_elems["VKey3"][25] = bitarray('1110101')
        result._varying_elems["VKey3"][21] = 5

        result.remove(['3','5'])

        assert_equal(['0','1','2','4','6'], result._tags)
        assert_equal({"VKey3" : 25}, result._const_elems)
        assert_equal({"VKey1", "VKey2"}, set(result._varying_elems.keys()))

        assert_equal(bitarray('01100'), result._varying_elems["VKey1"][6.2])
        assert_equal(bitarray('10011'), result._varying_elems["VKey1"][7.8])

        assert_equal(bitarray('11100'), result._varying_elems["VKey2"][11])
        assert_equal(4, result._varying_elems["VKey2"][12])
        assert_equal(3, result._varying_elems["VKey2"][14])


    def test_subset(self):
        parent = ElementSummary()
        parent._tags = ['0','1','2','3','4','5','6']
        parent._const_elems = {'c->c' : 5}

        parent._varying_elems['v->c'] = \
            VaryingElement.make_from_inverse([1, 1, 2, 2, 1, 1, 1])
        parent._varying_elems['v->v'] = \
            VaryingElement.make_from_inverse([3, 4, 3, 3, 4, 4, 3])
        parent._varying_elems['v->_'] = \
            VaryingElement.make_from_inverse([None, None, 1, 1, None, 2, None])

        child = parent.subset([0, 1, 4, 6])

        assert_equal(['0', '1', '4', '6'], child._tags)
        assert_equal(2, len(child._const_elems))
        assert_equal(5, child._const_elems['c->c'])
        assert_equal(1, child._const_elems['v->c'])
        assert_equal(1, len(child._varying_elems))
        assert_equal(
             VaryingElement.make_from_inverse([3, 4, 4, 3]),
             child._varying_elems['v->v'])


    def test_split(self):
        parent = ElementSummary()
        data = [('0', {'1,2->c' : 5, 'key' : 1, '1,2->v' : 3, '2->_' : 1}),
                ('1', {'1,2->c' : 5, 'key' : 1, '1,2->v' : 4, '2->_' : 1}),
                ('2', {'1,2->c' : 5, 'key' : 2, '1,2->v' : 3, '1->_' : 1}),
                ('3', {'1,2->c' : 5, 'key' : 2, '1,2->v' : 3, '1->_' : 2}),
                ('4', {'1,2->c' : 5, 'key' : 2, '1,2->v' : 4, '1->_' : 2}),
                ('5', {'1,2->c' : 5, 'key' : 1, '1,2->v' : 4, '2->_' : 2}),
                ('6', {'1,2->c' : 5, 'key' : 1, '1,2->v' : 3, '2->_' : 2}),
                ]
        parent.extend(data)
        # parent._tags = ['0','1','2','3','4','5','6']
        # parent._const_elems = {'1,2->c' : 5}
        #
        # parent._varying_elems['key'] = \
        #     VaryingElement.make_from_inverse([1, 1, 2, 2, 2, 1, 1])
        # parent._varying_elems['1,2->v'] = \
        #     VaryingElement.make_from_inverse([3, 4, 3, 3, 4, 4, 3])
        # parent._varying_elems['1->_'] = \
        #     VaryingElement.make_from_inverse([None, None, 1, 2, 2, None, None])
        # parent._varying_elems['2->_'] = \
        #     VaryingElement.make_from_inverse([1, 1, None, None, None, 2, 2])

        result = dict(item for item in parent.split('key'))
        assert_equal(2, len(result))

        assert_true(1 in result)
        assert_equal(2, len(result[1]._const_elems))
        assert_equal(2, len(result[1]._varying_elems))
        assert_equal(5, result[1]._const_elems['1,2->c'])
        assert_equal(1, result[1]._const_elems['key'])
        assert_equal(set(['0','1','5','6']), set(result[1]._tags))
        assert_equal(
            VaryingElement.make_from_inverse([3, 4, 4, 3]),
            result[1]._varying_elems['1,2->v'])
        assert_equal(
            VaryingElement.make_from_inverse([1, 1, 2, 2]),
            result[1]._varying_elems['2->_'])

        assert_true(2 in result)
        print result[2]._const_elems.keys()
        assert_equal(2, len(result[2]._const_elems))
        assert_equal(2, len(result[2]._varying_elems))
        assert_equal(5, result[2]._const_elems['1,2->c'])
        assert_equal(2, result[2]._const_elems['key'])
        assert_equal(['2','3','4'], result[2]._tags)
        assert_equal(
            VaryingElement.make_from_inverse([3, 3, 4]),
            result[2]._varying_elems['1,2->v'])
        assert_equal(
            VaryingElement.make_from_inverse([1, 2, 2]),
            result[2]._varying_elems['1->_'])


    def test_group(self):
        parent = ElementSummary()
        parent._tags = ['0','1','2','3','4','5','6','7']
        parent._const_elems = {'ckey' : 5}

        parent._varying_elems['key1'] = \
            VaryingElement.make_from_inverse([1, 1, 2, 2, 2, 1, 1, 1])
        parent._varying_elems['key2'] = \
            VaryingElement.make_from_inverse([3, 4, 3, 3, 4, 4, 3, 2])
        parent._varying_elems['key3'] = \
            VaryingElement.make_from_inverse([5, 6, 5, 6, 5, 7, 7, 5])
        parent._varying_elems['key4'] = \
            VaryingElement.make_from_inverse([8, 9, 9, 9, 8, 9, 8, 8])

        result = dict(item for item in parent.group(('key1', 'key4')))
        assert_equal(4, len(result))

        assert_true((1, 8) in result)
        assert_equal(3, len(result[(1, 8)]._const_elems))
        assert_equal(2, len(result[(1, 8)]._varying_elems))
        assert_equal(5, result[(1, 8)]._const_elems['ckey'])
        assert_equal(1, result[(1, 8)]._const_elems['key1'])
        assert_equal(8, result[(1, 8)]._const_elems['key4'])
        assert_equal(
            VaryingElement.make_from_inverse([3, 3, 2]),
            result[(1, 8)]._varying_elems['key2'])
        assert_equal(
            VaryingElement.make_from_inverse([5, 7, 5]),
            result[(1, 8)]._varying_elems['key3'])

        assert_true((1, 9) in result)
        assert_equal(4, len(result[(1, 9)]._const_elems))
        assert_equal(1, len(result[(1, 9)]._varying_elems))
        assert_equal(5, result[(1, 9)]._const_elems['ckey'])
        assert_equal(1, result[(1, 9)]._const_elems['key1'])
        assert_equal(4, result[(1, 9)]._const_elems['key2'])
        assert_equal(9, result[(1, 9)]._const_elems['key4'])
        assert_equal(
            VaryingElement.make_from_inverse([6, 7]),
            result[(1, 9)]._varying_elems['key3'])

        assert_true((2, 8) in result)
        assert_equal(5, len(result[(2, 8)]._const_elems))
        assert_equal(0, len(result[(2, 8)]._varying_elems))
        assert_equal(5, result[(2, 8)]._const_elems['ckey'])
        assert_equal(2, result[(2, 8)]._const_elems['key1'])
        assert_equal(4, result[(2, 8)]._const_elems['key2'])
        assert_equal(5, result[(2, 8)]._const_elems['key3'])
        assert_equal(8, result[(2, 8)]._const_elems['key4'])

        assert_true((2, 9) in result)
        assert_equal(4, len(result[(2, 9)]._const_elems))
        assert_equal(1, len(result[(2, 9)]._varying_elems))
        assert_equal(5, result[(2, 9)]._const_elems['ckey'])
        assert_equal(2, result[(2, 9)]._const_elems['key1'])
        assert_equal(3, result[(2, 9)]._const_elems['key2'])
        assert_equal(9, result[(2, 9)]._const_elems['key4'])
        assert_equal(
            VaryingElement.make_from_inverse([5, 6]),
            result[(2, 9)]._varying_elems['key3'])


    def test_merge(self):
        result = ElementSummary()
        result._tags = ['0','1','2','3']
        result._const_elems = {'c+c=c' : 1,
                              'c+c=v' : 2,
                              'c+_' : 3,
                              'c+v' : 4}
        result._varying_elems['v+v'] = \
            VaryingElement.make_from_inverse([1, 1, 2, 2])
        result._varying_elems['v+c'] = \
            VaryingElement.make_from_inverse([3, 4, 3, 3])
        result._varying_elems['v+_'] = \
            VaryingElement.make_from_inverse([5, 5, 5, 6])

        other = ElementSummary()
        other._tags = ['4','5','6']
        other._const_elems = {'c+c=c' : 1,
                             'c+c=v' : 3,
                             '_+c' : 6,
                             'v+c' : 4}
        other._varying_elems['v+v'] = \
            VaryingElement.make_from_inverse([1, 2, 3])
        other._varying_elems['c+v'] = \
            VaryingElement.make_from_inverse([3, 4, 3])
        other._varying_elems['_+v'] = \
            VaryingElement.make_from_inverse([6, 5, 5])

        result.merge(other)

        assert_equal(['0','1','2','3','4','5','6'], result._tags)
        assert_equal({'c+c=c' : 1}, result._const_elems)
        assert_equal(
            VaryingElement.make_from_inverse([2,2,2,2,3,3,3]),
            result._varying_elems['c+c=v'])
        assert_equal(
            VaryingElement.make_from_inverse([3,3,3,3,None,None,None]),
            result._varying_elems['c+_'])
        assert_equal(
            VaryingElement.make_from_inverse([4,4,4,4,3,4,3]),
            result._varying_elems['c+v'])
        assert_equal(
            VaryingElement.make_from_inverse([None,None,None,None,6,6,6]),
            result._varying_elems['_+c'])
        assert_equal(
            VaryingElement.make_from_inverse([1,1,2,2,1,2,3]),
            result._varying_elems['v+v'])
        assert_equal(
            VaryingElement.make_from_inverse([None,None,None,None,6,5,5]),
            result._varying_elems['_+v'])


    def test_find_axes(self):

        axes_names = ('x', 'y', 'z')
        axes_data = np.mgrid[0:3, 0:4, 0:5]
        axes_data = tuple(np.ravel(axes_data[i]) for i in range(axes_data.shape[0]))

        n = len(axes_data[0])

        a = [1, 2, 3]
        a.extend([0] * (n-len(a)))
        random.shuffle(a)

        b = [1, 1, 1, 2, 2]
        b.extend([0] * (n-len(b)))
        random.shuffle(b)

        c = [5] * n

        names = axes_names + ('a', 'b', 'c')
        values = axes_data + (a, b, c)
        data = [(i, dict(zip(names, val)))
                for i, val in enumerate(zip(*values))]
        random.shuffle(data)

        summary = ElementSummary()
        summary.extend(data)

        guess_keys = [val for val in names if val != 'x']
        random.shuffle(guess_keys)
        axes_keys = summary.find_axes(guess_keys)
        assert_true(axes_keys is None)

        prv_elems = (summary.get('x'),)
        expected_axes_keys = set(ax for ax in axes_names if ax != 'x')
        returned_axes_keys = set(summary.find_axes(guess_keys, prv_elems))
        assert_equal(expected_axes_keys, returned_axes_keys)

        guess_keys = ['x'] + guess_keys
        returned_axes_keys = set(summary.find_axes(guess_keys, prv_elems))
        assert_equal(expected_axes_keys, returned_axes_keys)

        random.shuffle(guess_keys)
        expected_axes_keys = set(axes_names)
        returned_axes_keys = set(summary.find_axes(guess_keys))
        assert_equal(expected_axes_keys, returned_axes_keys)


    def test_append(self):
        summary = ElementSummary()

        summary.append(0, {'a':2, 'b':0, 'c':3, 'd':4})
        assert_equal(set(summary._const_elems.keys()), {'a', 'b', 'c', 'd'})
        assert_equal(summary['a'], 2)
        assert_equal(summary['b'], 0)
        assert_equal(summary['c'], 3)
        assert_equal(summary['d'], 4)

        summary.append(1, {'a':1, 'b':0, 'c':1, 'd':0})
        assert_equal(set(summary._const_elems.keys()), {'b'})
        assert_equal(set(summary._varying_elems.keys()), {'a', 'c', 'd'})
        assert_equal(summary['a'], VaryingElement.make_from_inverse([2, 1]))
        assert_equal(summary['b'], 0)
        assert_equal(summary['c'], VaryingElement.make_from_inverse([3, 1]))
        assert_equal(summary['d'], VaryingElement.make_from_inverse([4, 0]))


    def test_extend(self):
        inverses = {'a' : [0, 1, 1, 1, 0, 0],
                    'b' : [0, 0, 0, 0, 0, 0],
                    'c' : [1, 0, 2, 0, 2, 0],
                    'd' : [0, 1, 3, 2, 2, 2],
                    'e' : [1, 1, None, None, None, 1],
                    'f' : [None, None, 1, 2, 1, None]}
        data_pairs = [{'a':0, 'b':0, 'c':1, 'd':0, 'e':1},
                      {'a':1, 'b':0, 'c':0, 'd':1, 'e':1},
                      {'a':1, 'b':0, 'c':2, 'd':3, 'f':1},
                      {'a':1, 'b':0, 'c':0, 'd':2, 'f':2},
                      {'a':0, 'b':0, 'c':2, 'd':2, 'f':1},
                      {'a':0, 'b':0, 'c':0, 'd':2, 'e':1}]
        data_pairs = zip(range(len(data_pairs)), data_pairs)

        summary = ElementSummary()
        summary.extend(data_pairs)
        # test constant elems
        assert_equal(set(summary._const_elems.keys()), {'b'})
        assert_equal(summary['b'], 0)
        # test varying elements
        assert_equal(set(summary._varying_elems.keys()), {'a','c','d','e','f'})
        assert_equal(summary['a'],
                     VaryingElement.make_from_inverse(inverses['a']))
        assert_equal(summary['c'],
                     VaryingElement.make_from_inverse(inverses['c']))
        assert_equal(summary['d'],
                     VaryingElement.make_from_inverse(inverses['d']))
        assert_equal(summary['a'],
                     VaryingElement.make_from_inverse(inverses['a']))
        assert_equal(summary['e'],
                     VaryingElement.make_from_inverse(inverses['e']))
        assert_equal(summary['f'],
                     VaryingElement.make_from_inverse(inverses['f']))
