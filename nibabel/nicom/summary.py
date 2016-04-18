"""Data structures for summarizing (key,value) pairs across many instances"""
import numpy as np
import six
import collections
import itertools
import operator
from bitarray import bitarray
from copy import deepcopy
from numbers import Integral


#===============================================================================
#  Utility functions for working with bitarrays in the context of how they are
#  utilized in this module.
#===============================================================================

def zeros_bitarray(size):
    """Create a bitarray of filled with 0s.
    """
    barray = bitarray(size)
    barray.setall(0)
    return barray


def ones_bitarray(size):
    """Create a bitarray of filled with 1s.
    """
    barray = bitarray(size)
    barray.setall(1)
    return barray


def slice_to_bitarray(slc, size):
    """Creates a bitarray and sets the bits to 1 for the indices given by a
    slice.
    """
    barray = bitarray(size)
    barray.setall(0)
    barray[slc] = 1
    return barray


def iterable_to_bitarray(itr, size):
    """Creates a bitarray and sets the bits to 1 for the indices given by an
    iterable container.
    """
    barray = bitarray(size)
    barray.setall(0)
    for i in itr:
        barray[i] = 1
    return barray


def integer_to_bitarray(idx, size):
    """Create a bitarray with a single bit set at the given index.
    """
    barray = bitarray(size)
    barray.setall(0)
    barray[idx] = 1
    return barray


def as_bitarray(x, size):
    """Convert object to bitarray if neccessary, otherwise return object itself
    if it is already a bitarray.
    """
    if isinstance(x, bitarray):
        return x
    elif isinstance(x, Integral):
        return integer_to_bitarray(x, size)
    elif isinstance(x, slice):
        return slice_to_bitarray(x, size)
    elif isinstance(s, colletions.Iterable):
        return iterable_to_bitarray(s, size)
    else:
        raise TypeError


def bitarray_to_indices(barray, start=0):
    """Return the indices of the set bits in the bitarray.
    """
    return tuple(i for i, b in enumerate(barray, start) if b)


#===============================================================================
#
#
#
#===============================================================================
class ElementSummary(object):
    """ Summary of the attributes of multiple DICOM files """


    def __init__(self, compare_rules=None):
        if compare_rules is not None:
            self.compare_rules = compare_rules
        else:
            self.compare_rules = dict()
        self.clear()


    @classmethod
    def make_from_inverses(klass, data, unfilled_value=None, compare_rules=None):
        """

        Example
        -------
        >>> data = [('a', [1, 2, 1]), ('b', [3, 3, 3])]
        >>> summary = ElementSummary.make_from_inverses(data)
        """

        result = ElementSummary(compare_rules=compare_rules)

        # convience conversion to allow for sequence of pairs or dictionaries
        if isinstance(data, dict):
            data_iterator = six.iteritems(data)
        else:
            data_iterator = data

        size = None
        for key, values in data_iterator:
            # ensure the length all values is the same
            if size is None:
                size = len(values)
            elif len(values) != size:
                raise ValueError
            # create varying element for key
            cmp_func = None if compare_rules is None else compare_rules.get(key)
            elem = VaryingElement.make_from_inverse(values,
                                                    unfilled_key=unfilled_value,
                                                    cmp_func=cmp_func)
            # determine how values vary
            if elem.is_empty():
                # if completely unfilled, then do not add
                continue
            elif elem.is_constant():
                # if constant, then add key and value to constant dict
                result._const_elems[key], _ = elem.popitem()
            else:
                # if varying then add to varying dict
                result._varying_elems[key] = elem
        # set size
        result._size = size
        return result

    @property
    def size(self):
        return self._size


    def __eq__(self, other):
        return (self._const_elems == other._const_elems and
                self._varying_elems == other._varying_elems)


    def __ne__(self, other):
        return not self.__eq__(other)


    def __len__(self):
        return len(self._const_elems) + len(self._varying_elems)


    def clear(self):
        """Empty the internal data structures"""
        self._size = 0
        self._const_elems = dict()
        self._varying_elems = dict()


    def resize(self, size):
        self._size = size
        for var_elem in six.itervalues(self._varying_elems):
            var_elem.resize(size)


    def __contains__(self, key):
        return key in self._const_elems or key in self._varying_elems


    def __delitem__(self, key):
        if key in self._const_elems:
            del self._const_elems[key]
        if key in self._varying_elems:
            del self._varying_elems[key]
        raise KeyError


    def __getitem__(self, key):
        if key in self._const_elems:
            return self._const_elems[key]
        if key in self._varying_elems:
            return self._varying_elems[key]
        raise KeyError


    def get(self, key, default=None):
        if key in self._const_elems:
            return self._const_elems[key]
        if key in self._varying_elems:
            return self._varying_elems[key]
        return default


    def is_permutation(self, other):
        """Test if the contents of this object are just a reordering of the
        contents of 'other'
        """
        # check equality of dimesions
        if self.size != other.size:
            return False
        # check that the set of keys are the same
        if (self._const_elems.viewkeys() != other._const_elems.viewkeys() or
            self._varying_elems.viewkeys() != other._varying_elems.viewkeys()):
            return False
        # compare constant elements
        rules = self.compare_rules
        if not all(rules(key, operator.eq)(val, other._const_elems[key])
                   for key, val in six.iteritems(self._const_elems)):
            return False
        # find the strictist mapping between the summaries
        min_deg_of_free = self.size
        for key, this_elem in six.iteritems(self._varying_elems):
            # the number of degrees of freedom that a mapping based on this
            # element will have
            cur_deg_of_free = this_elem.size - len(this_elem)
            # use mapping if it stricter than current strictest mapping
            if cur_deg_of_free < min_deg_of_free:
                min_deg_of_free = cur_deg_of_free
                # get the mapping
                other_elem = other._varying_elems[key]
                other2this = this_elem.find_permutation(other_elem)
                # there is no mapping, therefore summaries are not permutations
                if other2this is None:
                    return False
                # break if we have found the strictest mapping that we can
                if min_deg_of_free <= 1:
                    break
        # compare varying elements are the same under the found mapping
        for key, this_elem in six.iteritems(self._varying_elems):
            other_elem = other._varying_elems[key]
            if not this_elem.is_permutation(other_elem, other2this):
                return False
        return True


    def _append_single(self, idx, data):
        # this function is considered unsafe because changing the sizes of the
        # varying elements to accommodate the inserted data must be done
        # externally.

        # all elements are constant if this the first dict
        if idx == 0:
            if isinstance(data, dict):
                self._const_elems = deepcopy(data)
            else:
                for key, value in data:
                    self._const_elems[key] = value
            return

        # convenience coversion for dictionaries
        if isinstance(data, dict):
            data_generator = six.iteritems(data)
        else:
            data_generator = data

        # keep track of visited constant keys, not visited constant keys need
        # to be moved to varying elements
        unvisited_const_keys = set(self._const_elems.keys())

        for key, value in data_generator:
            # Retrieve the comparison function
            cmp_func = self.compare_rules.get(key)

            if key in self._const_elems:
                unvisited_const_keys.discard(key)
                const_val = self._const_elems[key]
                cmp_rep = operator.eq if cmp_func is None else cmp_func
                if not cmp_rep(const_val, value):
                    # The element is no longer const, move it to the varying
                    var_elem = VaryingElement(size=self.size, cmp_func=cmp_func)
                    var_elem[const_val] = slice(0, idx, 1)
                    var_elem._unsafe_assign_int(value, idx)
                    self._varying_elems[key] = var_elem
                    del self._const_elems[key]
                # The value is still const, continue
                continue

            if key in self._varying_elems:
                # Mark location in varying element object
                self._varying_elems[key]._unsafe_assign_int(value, idx)
            else:
                # Haven't seen this key before, add to varying_elems
                # because missing values are not considered const
                var_elem = VaryingElement(size=self.size, cmp_func=cmp_func)
                var_elem[value] = idx
                self._varying_elems[key] = var_elem

        # move unvisited constant elements to varying
        for key in unvisited_const_keys:
            var_elem = VaryingElement(size=self.size,
                                      cmp_func=self.compare_rules.get(key))
            var_elem[self._const_elems[key]] = slice(0, idx, 1)
            self._varying_elems[key] = var_elem
            del self._const_elems[key]


    def _append_summary(self, idx, other):
        # extensively used structures
        if other.size == 1:
            new_part = idx
        else:
            new_part = slice_to_bitarray(slice(idx, idx+other.size), self.size)

        if idx == 1:
            old_part = 0
        else:
            old_part = slice_to_bitarray(slice(0, idx), self.size)

        # creating the 8 possible key set combinations
        other_const = set(other._const_elems.keys())
        other_vary = set(other._varying_elems.keys())
        this_const = set(self._const_elems.keys())
        this_vary = set(self._varying_elems.keys())

        # overlaping key sets
        this_vary_other_vary = this_vary & other_vary
        this_vary_other_const = this_vary & other_const
        this_const_other_vary = this_const & other_vary
        this_const_other_const = this_const & other_const

        # if the paths are overlaping then there should be no exclusive keys
        other_const -= this_const_other_const | this_vary_other_const
        other_vary -= this_const_other_vary | this_vary_other_vary
        this_const -= this_const_other_vary | this_const_other_const
        this_vary -= this_vary_other_vary | this_vary_other_const

        # keys that are only in other.const_elems
        for key in other_const:
            var_elem = VaryingElement(size=self.size,
                                      cmp_func=self.compare_rules.get(key))
            var_elem._unsafe_set(other._const_elems[key], new_part.copy())
            self._varying_elems[key] = var_elem

        # keys that are only in other.varying_elems
        for key in other_vary:
            var_elem = VaryingElement(size=idx,
                                      cmp_func=self.compare_rules.get(key))
            var_elem.extend(other._varying_elems[key])
            if var_elem.size != self.size:
                var_elem.resize(self.size)
            self._varying_elems[key] = var_elem

        # keys that are only in self.const_elems
        for key in this_const:
            var_elem = VaryingElement(size=self.size,
                                      cmp_func=self.compare_rules.get(key))
            var_elem._unsafe_set(self._const_elems[key], old_part.copy())
            self._varying_elems[key] = var_elem
            del self._const_elems[key]

        # intersection of self.const_elems and other.varying_elems
        for key in this_const_other_vary:
            var_elem = VaryingElement(size=self.size,
                                      cmp_func=self.compare_rules.get(key))
            var_elem._unsafe_set(self._const_elems[key], old_part.copy())
            self._varying_elems[key] = var_elem
            del self._const_elems[key]

        # intersection of varying elements
        wrk_array = bitarray(self.size)
        for key in this_vary_other_vary | this_const_other_vary:
            other_elem = other._varying_elems[key]
            this_elem = self._varying_elems[key]
            for other_key, other_val in six.iteritems(other_elem):
                repr_key = this_elem._find_repr_key(other_key)
                if isinstance(other_val, bitarray):
                    wrk_array.setall(0)
                    wrk_array[idx : idx + other.size] = other_val
                    this_elem._unsafe_assign_bitarray(repr_key, wrk_array)
                else:
                    this_elem._unsafe_assign_int(repr_key, idx + other_val)

        # intersection of constant elements
        for key in this_const_other_const:
            cmp_func = self.compare_rules.get(key)
            cmp_rep = operator.eq if cmp_func is None else cmp_func
            if not cmp_rep(self._const_elems[key], other._const_elems[key]):
                var_elem = VaryingElement(size=self.size, cmp_func=cmp_func)
                var_elem._unsafe_set(self._const_elems[key], old_part.copy())
                var_elem._unsafe_set(other._const_elems[key], new_part.copy())
                self._varying_elems[key] = var_elem
                del self._const_elems[key]

        # intersection of self.varying_elems and other.const_elems
        for key in this_vary_other_const:
            var_elem = self._varying_elems[key]
            repr_key = var_elem._find_repr_key(other._const_elems[key])
            var_elem._unsafe_assign_bitarray(repr_key, new_part.copy())


    def append(self, data):
        idx = self.size
        if not isinstance(data, ElementSummary):
            # append regular sequence
            self.resize(self.size + 1)
            self._append_single(idx, data)
        elif data.size > 0:
            # append ElementSummary if it is not empty
            self.resize(self.size + data.size)
            self._append_summary(idx, data)


    def extend(self, data_seq, guess_size=None):

        # amount by which to increment the size to reduce the amount of
        # resizing
        chunk_size = 8

        # loading of default guess_size
        if guess_size is None:
            if hasattr(data_seq, '__len__'):
                guess_size = len(data_seq)
            else:
                # will cause the size to be increased to the next largest
                # multiple of chunk_size
                guess_size = chunk_size - (self.size % chunk_size)

        # uesd to keep track of at what index the data is currently being
        # inserted at and what the final size is
        cur_idx = self.size
        # increase size to allow for new data to be added
        self.resize(self.size + guess_size)

        for data in data_seq:
            if isinstance(data, ElementSummary):
                req_size = data.size + cur_idx
                # increase the varying elements' sizes when more space
                # is needed
                if req_size > self.size:
                    # find next largest size that falls on chunck_size
                    new_size = req_size + chunk_size - (req_size % chunk_size)
                    self.resize(new_size)
                # append data
                self._append_summary(cur_idx, data)
                # increment current index
                cur_idx += data.size
            else:
                # increase the varying elements' sizes when more space
                # is needed
                if cur_idx >= self.size:
                    self.resize(self.size + chunk_size)
                # append data
                self._append_single(cur_idx, data)
                # increment current index
                cur_idx += 1

        # shrink the varying elements' sizes to the true size if neccessary
        if cur_idx != self.size:
            self.resize(cur_idx)


    def reduce(self, indices):
        new_size = len(indices)
        if new_size == 0:
            self.clear()
        else:
            self._size = new_size
            # remove indices from all the varying elements
            del_keys = []
            for key, var_elem in six.iteritems(self._varying_elems):
                # reduce the varying element
                var_elem.reduce(indices)
                # ghandle elements that are not still varying
                if var_elem.is_empty():
                    del_keys.append(key)
                elif var_elem.is_constant():
                    self._const_elems[key], _ = var_elem.popitem()
                    del_keys.append(key)
            # delete elements that are no longer varying
            for key in del_keys:
                del self._varying_elems[key]


    def remove(self, indices):
        """ Remove the given indices from the summary """
        self.reduce([i for i in range(self.size) if i not in indices])


    def empty_copy(self):
        return ElementSummary(compare_rules=self.compare_rules)


    def subset(self, indices):
        result = self.empty_copy()
        # return empty ElementSummary if there are no indices to fill it with
        if len(indices) == 0:
            return result

        self._size = len(indices)
        result._const_elems = self._const_elems.copy()

        if len(indices) > 1:
            for key, old_var_elem in six.iteritems(self._varying_elems):
                new_var_elem = old_var_elem.subset(indices)
                if new_var_elem.is_empty():
                    # do not include empty varying element in results
                    continue
                elif new_var_elem.is_constant():
                    # reduction caused element to become constant
                    result._const_elems[key], _ = new_var_elem.popitem()
                else:
                    # element is still varying
                    result._varying_elems[key] = new_var_elem
        else:
            # if there is a single index, then all elements are constant
            idx = indices[0]
            for key, var_elem in six.iteritems(self._varying_elems):
                try:
                    result._const_elems[key] = var_elem.at(idx)
                except KeyError:
                    # index is unfilled for key, do not add to subset
                    continue
        return result


    def split_mask(self, key, mask=None, default=None):
        if mask is None:
            mask = ones_bitarray(self.size)
        if key in self._const_elems:
            yield self._const_elems[key], mask.copy()
        elif key not in self._varying_elems:
            yield default, mask.copy()
        else:
            pvt_elem = self._varying_elems[key]
            filled = zeros_bitarray(self.size)
            for pvt_key, pvt_val in six.iteritems(pvt_elem):
                if isinstance(pvt_val, bitarray):
                    filled |= pvt_val
                    new_mask = mask & pvt_val
                    if any(new_mask):
                        yield pvt_key, new_mask
                else:
                    filled[pvt_val] = 1
                    if mask[pvt_val] == 1:
                        yield pvt_key, integer_to_bitarray(pvt_val, self.size)
            # yield sub-summary for unfilled indices
            new_mask = mask & ~filled
            if any(new_mask):
                yield default, new_mask


    def split(self, key, default=None):
        for val, val_mask in self.split_mask(key, default):
            yield val, self.subset(bitarray_to_indices(val_mask))


    def group_mask(self, keys, mask=None, default=None):
        if mask is None:
            mask = ones_bitarray(self.size)
        if len(keys) == 1:
            for pvt_val, pvt_msk in self.split_mask(keys[0], mask, default):
                yield (pvt_val,), pvt_msk
        else:
            for pvt_val, pvt_msk in self.split_mask(keys[0], mask, default):
                for vals, msk in self.group_mask(keys[1:], pvt_msk, default):
                    yield (pvt_val,) + vals, msk


    def group(self, keys, default=None):
        """Generator that splits the summary into sub-summaries that share the
        same values for the given set of keys."""
        for grp_vals, grp_mask in self.group_mask(keys, default):
            yield grp_vals, self.subset(bitarray_to_indices(grp_mask))


    def find_axes(self, guess_keys, prv_elems=None):
        #TODO; better description
        """Find a set of elements whose values form a coordinate basis.
        """
        # filter guess_keys to only those that are possible
        possible = []
        for key in guess_keys:
            elem = self._varying_elems.get(key)
            if elem is not None and len(elem) > 1 and elem.is_square():
                possible.append((key, elem))

        # handle input for prv_elems
        if prv_elems is None:
            prv_elems = tuple()
            prv_prod = 1
        else:
            prv_elems = tuple(prv_elems)
            prv_prod = np.prod([len(prv) for prv in prv_elems])

        # search for a set of keys that make a complete set of axes
        axes_keys = self._find_axes(possible, prv_elems, prv_prod)

        return axes_keys


    def _find_axes(self, possible, prv_elems, prv_prod):
        # helper function which contains majority of the logic for find_axes
        for i, (key, elem) in enumerate(possible):
            # the current size of the space that can be spanned by the previous
            # elements plus this element
            cur_prod = len(elem) * prv_prod
            # quick check that the addition of the current elem could lead to a
            # complete set of axes
            if elem.size % cur_prod != 0:
                continue
            # check that the varability of the current element is contained
            # in the null space of all previous elements
            if all(elem.is_complementary(prv) for prv in prv_elems):
                # if the current element completes the set of axes, then
                # return its key
                if elem.size == cur_prod:
                    return (key,)
                # if there is still variablity to account for then recurse
                # in search of further axes
                cur_elems = prv_elems + (elem,)
                sub_keys = self._find_axes(possible[i+1:], cur_elems, cur_prod)
                # if sub_keys is not None then a complete set of axes was
                # found that contains the current element
                if sub_keys is not None:
                    # prepend the current elements key and send it up the
                    # recursion stack
                    return (key,) + sub_keys
        # return None if no complete set of axes could be found
        return None


    def to_axis_metadata(self, axes_elems):

        # mapping of axes coordinates to flat index
        ndim = len(axes_elems)
        shape = tuple(len(ax) for ax in axes)
        coord_to_idx = np.ndarray(shape, dtype=int, order='C')
        ax_val_iters = tuple(six.itervalues(ax) for ax in axes_elems)
        for fidx, ax_vals in enumerate(itertools.product(*ax_val_iters)):
            # determine the set of indices that have these axes coordinates
            overlap = ax_vals[0].copy()
            for ax_val in ax_vals[1:]:
                overlap &= ax_val
            # a single 1 in bitarray at the index for these coordinates
            coord_to_idx.flat[fidx] = overlap.index(1)

        # determine how each varying element is varying with respect to the
        # given axes, and group similar varying elements together
        variation_groups = dict()
        for key, vary_elem in six.iteritems(self._varying_elems):
            var_axes = vary_elem.variation_wrt_axes(axes_elems)
            if var_axes not in variation_groups:
                variation_groups[var_axes] = dict()
            # use VaryElement instance for key as a placeholder that will be
            # replaced by the array of values
            variation_groups[var_axes][key] = vary_elem

        # replace VaryingElement instances with arrays collapsed to only those
        # axes for which the values depend
        for vary_axes, vary_elems in six.iteritems(variation_groups):
            # get a mapping reduced to only the axes the values vary over
            dim_diff = len(axes_elems) - len(vary_axes)
            if dim_diff > 0:
                # move varying axes to the end to facilitate extraction
                tp = tuple(i for i in range(ndim) if i not in vary_axes)
                tp += vary_axes
                grp_mapping = np.transpose(coord_to_idx, tp)[(0,)*dim_diff]
            else:
                grp_mapping = np.transpose(coord_to_idx, vary_axes)
            # use mapping to create array of values
            for key, vary_elem in six.iteritems(vary_elems):
                vary_elems[key] = np.asarray(vary_elem.inverse())[grp_mapping]

        # add constant elems
        variation_groups['const'] = self._const_elems

        return variation_groups


    def __str__(self):
        return "%s, %s" % (self._const_elems, self._varying_elems)

#===============================================================================
#
#
#
#===============================================================================
class VaryingElement(collections.OrderedDict):
    """A memory efficient structure for a one-to-many mapping of a value to a
    positional index.

    This mapping is the inverse of that of a list or array:

    >>> values = ['a', 'a', 'b', 'a', 'b']
    >>> velem = VaryingElement.make_from_inverse(values)
    >>> velem['a']
    [0, 1, 3]
    >>> velem['b']
    [2, 4]
    """

    def __init__(self, size=0, cmp_func=None):
        """Initialize a VaryingElement instance.

        Parameters
        ----------
        size : int
            the number of indices to initially start with.

        cmp_func : binary function
            a comparision function to use to test whether two keys should be
            considered equivalent.
        """
        if size < 0:
            raise ValueError
        super(VaryingElement, self).__init__()
        self._size = size
        self.cmp_func = cmp_func


    @property
    def size(self):
        """The number indices. Not the number of keys"""
        return self._size


    #----------------------------------------------------------------------
    # logic functions
    #----------------------------------------------------------------------

    def _unsafe_set(self, repr_key, value):
        super(VaryingElement, self).__setitem__(repr_key, value)


    def _unsafe_del(self, repr_key):
        super(VaryingElement, self).__delitem__(repr_key)

    def _find_repr_key(self, key):
        if self.cmp_func is None:
            return key
        else:
            for repr_key in six.iterkeys(super(VaryingElement, self)):
                if self.cmp_func(key, repr_key):
                    return repr_key
            return key

    def __getitem__(self, key):
        repr_key = self._find_repr_key(key)
        return super(VaryingElement, self).__getitem__(repr_key)


    def get(self, key, default=None):
        repr_key = self._find_repr_key(key)
        return super(VaryingElement, self).get(repr_key, default)


    def _unassign_int(self, idx, ignore_keys={}):
        del_needed = False
        for key, val in six.iteritems(self):
            if key in ignore_keys:
                continue
            if isinstance(val, bitarray) and val[idx]:
                val[idx] = 0
                if val.count(1) == 1:
                    self._unsafe_set(key, val.index(1))
                break
            elif val == idx:
                del_needed = True
                del_key = key
                break
        if del_needed:
            self._unsafe_del(del_key)


    def _unassign_bitarray(self, barray, ignore_keys={}):
        not_barray = ~barray
        # filter out indices in other elements that are part of the new element
        del_keys = []
        for old_key, old_val in six.iteritems(super(VaryingElement, self)):
            if old_key in ignore_keys:
                continue
            if isinstance(old_val, bitarray):
                old_val &= not_barray
                count = old_val.count(1)
                if count == 0:
                    del_keys.append(old_key)
                elif count == 1:
                    self._unsafe_set(old_key, old_val.index(1))
            elif not not_barray[old_val]:
                del_keys.append(old_key)
        # deleting empty entries
        for del_key in del_keys:
            self._unsafe_del(del_key)


    def unassign(self, indices, ignore_keys={}):
        """Unassign indices from their associated keys.

        Parameters
        ----------
        indices : bitarray, int, slice, iterable
            indices to disassociate with any key.

        ignore_keys : collection
            a set of keys to not disassociate the the indices to with

        Example
        -------
        >>> velem = VaryingElement.make_from_inverse(['a', 'a', 'b', 'a', 'b'])
        >>> velem.unassign(indices=[0, 1, 2], ignore_keys={'b'})
        >>> velem['a']
        [3]
        >>> velem['b']
        [2, 4]
        """
        if isinstance(indices, bitarray):
            self._unassign_bitarray(indices, ignore_keys)
        elif isinstance(indices, Integral):
            self._unassign_int(indices, ignore_keys)
        elif isinstance(indices, slice):
            barray = slice_to_bitarray(indices, self.size)
            self._unassign_bitarray(barray, ignore_keys)
        elif isinstance(indices, collections.Iterable):
            barray = iterable_to_bitarray(indices, self.size)
            self._unassign_bitarray(barray, ignore_keys)
        else:
            raise TypeError


    def _setitem_int(self, key, index):
        repr_key = self._find_repr_key(key)
        old_key = self._unassign_int(index, {repr_key})
        if repr_key != key:
            self.__delitem__(repr_key)
        # add the new element
        self._unsafe_set(key, index)


    def _setitem_bitarray(self, key, indices):
        # handle special cases
        count = indices.count(1)
        if count == 1:
            self._setitem_int(key, indices.index(1))
            return

        repr_key = self._find_repr_key(key)

        # handle special empty case
        if count == 0:
            try:
                self._unsafe_del(repr_key)
            except KeyError:
                pass
            return

        if repr_key != key:
            self.__delitem__(repr_key)

        # unassign shared indices in other entries
        self._unassign_bitarray(indices, {repr_key})
        # add the new element
        self._unsafe_set(key, indices)


    def __setitem__(self, key, indices):
        """Set the indices associated with a key.

        Any indices previously associated with the key that are not in the newly
        given indices will be disassociated with the key. The newly given
        indices will be unassigned from all other keys.

        The given indices must agree with the current size. NO automatic
        resizing is done.

        Example
        -------
        >>> velem = VaryingElement.make_from_inverse(['a', 'a', 'b', 'a', 'b'])
        >>> velem['a']
        [0, 1, 3]
        >>> velem['b']
        [2, 4]
        >>> velem['a'] = [1, 2, 3]
        >>> velem['a']
        [1, 2, 3]
        >>> velem['b']
        [4]
        """
        if isinstance(indices, bitarray):
            # ensure that the length is the same
            if indices.length() != self.size:
                raise ValueError
            self._setitem_bitarray(key, indices)
        elif isinstance(indices, Integral):
            # ensure that the index is with in the range
            if indices < 0 or indices >= self.size:
                raise IndexError
            self._setitem_int(key, indices)
        elif isinstance(indices, slice):
            barray = slice_to_bitarray(indices, self.size)
            self._setitem_bitarray(key, barray)
        elif isinstance(indices, collections.Iterable):
            barray = iterable_to_bitarray(indices, self.size)
            self._setitem_bitarray(key, barray)
        else:
            raise TypeError


    def _unsafe_assign_int(self, repr_key, idx):
        # this associates the index 'idx' with the value 'repr_key' without
        # performing the work of removing any previous association that 'idx'.
        # This function is unsafe because the caller must ensure that the
        # passed in index will not cause the constraint that each index is only
        # associated with one value be broken. This is used when it is known
        # that the passed in index is not currently associated with any value
        old_val = super(VaryingElement, self).get(repr_key)
        if old_val is None:
            self._unsafe_set(repr_key, idx)
        elif isinstance(old_val, bitarray):
            old_val[idx] = 1
        else:
            new_val = zeros_bitarray(self.size)
            new_val[old_val] = 1
            new_val[idx] = 1
            self._unsafe_set(repr_key, new_val)


    def _unsafe_assign_bitarray(self, repr_key, barray):
        # this associates the indices in 'barray' with the value 'repr_key'
        # without performing the work of removing any previous association that
        # the indices in 'barray'. This function is unsafe because the caller
        # must ensure that the passed in indices will not cause the constraint
        # that each index is only associated with one value be broken. This is
        # used when it is known that the passed in indices are not currently
        # associated with any value
        old_val = super(VaryingElement, self).get(repr_key)
        if old_val is None:
            self._unsafe_set(repr_key, barray.copy())
        elif isinstance(old_val, bitarray):
            old_val |= barray
        else:
            new_val = barray.copy()
            new_val[old_val] = 1
            self._unsafe_set(repr_key, new_val)


    def assign(self, key, indices):
        """Associate additional indices with a key.

        The new indices associated with the key will be the union of the key's
        current indices and the given indices. The given indices will be
        unassigned from the other keys.

        The given indices must agree with the current size. NO automatic
        resizing is done.

        Parameters:
        -----------
        key : hashable object
            the key to assign the given indices to

        indices : int, set of ints, slice, bitarray
            the indices to assign to the given key

        Example
        -------
        >>> velem = VaryingElement.make_from_inverse(['a', 'a', 'b', 'a', 'b'])
        >>> velem['a']
        [0, 1, 3]
        >>> velem['b']
        [2, 4]
        >>> velem.assign('b', [1, 2])
        >>> velem['a']
        [0, 3]
        >>> velem['b']
        [1, 2, 4]
        """
        if isinstance(indices, bitarray):
            count = indices.count(1)
            if count > 1:
                repr_key = self._find_repr_key(key)
                self._unassign_bitarray(indices, {repr_key})
                self._unsafe_assign_bitarray(key, indices)
            elif count == 1:
                self.assign(key, indices.index(1))
        elif isinstance(indices, Integral):
            repr_key = self._find_repr_key(key)
            self._unassign_int(indices)
            self._unsafe_assign_int(repr_key, indices)
        elif isinstance(indices, slice):
            barray = slice_to_bitarray(indices, self.size)
            self.assign(key, barray)
        elif isinstance(indices, collections.Iterable):
            barray = iterable_to_bitarray(indices, self.size)
            self.assign(key, barray)
        else:
            raise ValueError


    def _convert_to_array(self, repr_key):
        prev_val = super(VaryingElement, self).get(repr_key)
        if isinstance(prev_val, bitarray):
            return prev_val
        else:
            new_val = bitarray(self.size)
            new_val.setall(0)
            if isinstance(prev_val, Integral):
                new_val[prev_val] = 1
            self._unsafe_set(repr_key, new_val)
            return new_val

    #----------------------------------------------------------------------
    # logic functions
    #----------------------------------------------------------------------
    def __contains__(self, key):
        if self.cmp_func is None:
            return super(VaryingElement, self).__contains__(key)
        else:
            return any(self.cmp_func(k, key) for k in six.iterkeys(self))


    def __eq__(self, other):
        # quick dimension equality check
        if len(self) != len(other) or self.size != other.size:
            return False
        return all(val == self.get(key) for key, val in six.iteritems(other))


    def __ne__(self, other):
        return not self.__eq__(other)


    def are_keys_equal(self, other):
        if len(self) != len(other):
            return False
        if self.cmp_func is None:
            return self.viewkeys() == other.viewkeys()
        else:
            return all(self.__contains__(key) for key in six.iterkeys(other))


    def is_empty(self):
        """Test if there are no indices mapped to a key"""
        return len(self) == 0


    def is_full(self):
        """Test if every index has a key mapped to it"""
        return self.unfilled is None


    def is_constant(self):
        """Test if all indices are mapped to the same key"""
        return len(self) == 1 and self.is_full()


    def is_square(self):
        """Test if every key has the same number of associated indices."""
        if self.size % len(self) != 0:
            return False
        filled_per_key = self.size // len(self)
        for val in six.itervalues(self):
            if isinstance(val, bitarray):
                if val.count(1) != filled_per_key:
                    return False
            else:
                return False
        return True


    def is_equivalent(self, other):
        #TODO: better explanation
        """Test if this object has the same mapping of indices as that of
        'other'.

        Example
        -------
        >>> velem1 = VaryingElement.make_from_inverse(['a', 'a', 'b', 'a', 'b'])
        >>> velem2 = VaryingElement.make_from_inverse([5.0, 5.0, 2.0, 5.0, 2.0])
        >>> velem1.is_equivalent(velem2)
        True
        >>> velem2.is_equivalent(velem1)
        True
        >>> velem3 = VaryingElement.make_from_inverse([1, 2, 2, 2, 1])
        >>> velem1.is_equivalent(velem3)
        False
        """
        if len(self) != len(other) or self.size != other.size:
            return False
        return all(any(v1 == v2 for v1 in six.itervalues(other))
                   for v2 in six.itervalues(self))


    def is_complementary(self, other, allow_trivial=True):
        #TODO: explanation
        """Test if
        """
        if self.size != other.size:
            raise ValueError("unequal sizes")

        # the size of the coordinate space 'self' and 'other' can define
        rank = len(self) * len(other)

        # quick check that
        if self.size % rank != 0:
            return False

        # check for trivial complement where one VaryElement has all them
        # varability and the other has none
        if ((len(self) == 1 or len(other) == 1)
            and self.size == rank and allow_trivial):
            return True

        # size of space that can not be accounted for by a basis formed by
        # 'self' and 'other'
        null_rank = self.size / rank

        # if not considering the trivial complement then complementary elements
        # will only consist of bitarrays
        if not all(isinstance(val, bitarray) for val in six.itervalues(self)):
            return False
        if not all(isinstance(val, bitarray) for val in six.itervalues(other)):
            return False

        # check that the overlap for each combination of values is the size
        # of the null rank
        value_iters = (six.itervalues(self), six.itervalues(other))
        for v1, v2 in itertools.product(*value_iters):
            if null_rank != (v1 & v2).count(1):
                return False

        return True


    def is_permutation(self, other, other2this=None):
        """Test
        """
        # quick check is permuation is possible
        if len(self) != len(other):
            return False
        # thorough check if a permuation
        if other2this is not None:
            # check for specific permutation
            for key in six.iterkeys(self):
                this_indices = set(self.where(key))
                other_indices = set(other2this[i] for i in other.where(key))
                if this_indices != other_indices:
                    return False
        else:
            # check for any permutation
            for key in six.iterkeys(self):
                this_indices = self.where(key)
                other_indices = other.where(key)
                if len(this_indices) != len(other_indices):
                    return False
        return True

    #----------------------------------------------------------------------
    # logic functions
    #----------------------------------------------------------------------

    def find_permutation(self, other):
        """Determine the reordering of this VaryingElement that would cause it
        to be the same as the VaryingElement 'other'.

        Returns
        -------
        """
        other2this = [0] * self.size
        fill_count = 0
        for key in six.iterkeys(self):
            this_indices = self.where(key)
            other_indices = other.where(key)
            if len(other_indices) != len(this_indices):
                return None
            for other_idx, this_idx in zip(other_indices, this_indices):
                other2this[other_idx] = this_idx
            fill_count += len(this_indices)
        # handle mapping for unfilled indices
        if self.size == fill_count + 1:
            # if there is a single unfilled index then VaryingElement.unfilled
            # will that index
            other2this[other.unfilled] = self.unfilled
        elif fill_count != self.size:
            # if there are multiple unfilled indices then
            # VaryingElement.unfilled will return a bitarray
            this_indices = [i for i, b in self.unfilled if b]
            other_indices = [i for i, b in other.unfilled if b]
            for other_idx, this_idx in zip(other_indices, this_indices):
                other2this[other_idx] = this_idx
        return other2this


    @property
    def unfilled(self):
        """Which indices do not have a key associated with them.

        Example
        -------
        >>> velem = VaryingElement.make_from_inverse(['a', 'a', 'b', 'a', 'b'])
        >>> velem.unassign(indices=[0, 1, 2], ignore_keys={'b'})
        >>> velem['a']
        [3]
        >>> velem['b']
        [2, 4]
        >>> velem.unfilled
        '10010'
        """
        unfilled_val = bitarray(self._size)
        unfilled_val.setall(0)
        for key, val in six.iteritems(super(VaryingElement, self)):
            if isinstance(val, bitarray):
                unfilled_val |= val
            else:
                unfilled_val[val] = 1
        unfilled_val.invert()
        # determing storage type
        count = unfilled_val.count(1)
        if count > 1:
            return unfilled_val
        elif count == 1:
            return unfilled_val.index(1)
        else:
            return None

    #----------------------------------------------------------------------
    # modifier functions
    #----------------------------------------------------------------------

    def resize(self, n):
        """Change the number of indices"""
        if n > self._size:
            size_dif = n - self._size
            self._size = n
            padding = bitarray(size_dif)
            padding.setall(0)
            for key, val in six.iteritems(super(VaryingElement, self)):
                if isinstance(val, bitarray):
                    val.extend(padding)
        elif n < self._size:
            self._size = n
            for key, val in six.iteritems(super(VaryingElement, self)):
                if isinstance(val, bitarray):
                    self._unsafe_set(key, val[:n])
            self.clean()


    def reorder(self, new2old):
        """Reorder the indices.

        Example
        -------
        >>> velem = VaryingElement.make_from_inverse(['a', 'b', 'c', 'd', 'e'])
        >>> velem.reorder([2, 3, 1, 0, 4])
        >>> velem.inverse()
        ['c', 'd', 'b', 'a', 'e']
        """
        # if any((i < 0 or i >= self.size) for i in new2old):
        #     raise IndexError
        if set(range(self._size)) != set(new2old):
            raise ValueError
        for key, val in six.iteritems(super(VaryingElement, self)):
            if isinstance(val, bitarray):
                new_val = bitarray([val[i] for i in new2old])
            else:
                new_val = new2old.index(val)
            self._unsafe_set(key, new_val)


    def change_compare_function(self, cmp_func):
        self.cmp_func = cmp_func
        if self.cmp_func is None:
            return
        rem_keys = set(key for key in six.iterkeys(super(VaryingElement, self)))
        combo = {}
        for key in six.iterkeys(super(VaryingElement, self)):
            rem_keys.discard(key)
            cmp_set = set()
            for cmp_key in rem_keys:
                if self.cmp_func(key, cmp_key):
                    cmp_set.add(cmp_key)
            if len(cmp_set) != 0:
                rem_keys -= cmp_set
                combo[key] = cmp_set

        while len(combo) > 0:
            del_keys = []
            for key, sub_keys in six.iteritems(combo):
                if all((sub_key not in combo) for sub_key in sub_keys):
                    del_keys.append(key)
                    val = self._convert_to_array(key)
                    for sub_key in sub_keys:
                        sub_val = super(VaryingElement, self).get(sub_key)
                        if isinstance(sub_val, bitarray):
                            val |= sub_val
                        else:
                            val[sub_val] = 1
            for del_key in del_keys:
                for sub_key in combo[del_key]:
                    self._unsafe_del(sub_key)
                del combo[del_key]
        self.clean()


    def extend(self, other):
        """Merge a VaryingElement instance into this one.

        Example
        -------
        >>> velem1 = VaryingElement.make_from_inverse(['a', 'a', 'b', 'a'])
        >>> velem2 = VaryingElement.make_from_inverse(['c', 'b', 'b', 'c'])
        >>> velem1.extend(velem2)
        >>> velem1.inverse()
        ['a', 'a', 'b', 'a', 'c', 'b', 'b', 'c']
        >>> velem1['b']
        [2, 5, 6]

        Example
        -------
        >>> velem = VaryingElement.make_from_inverse(['a', 'b'])
        >>> more_values = ['c', 'b']
        >>> velem.extend(more_values)
        >>> velem.inverse()
        [a', 'b', c', 'b']
        >>> velem['b']
        [1, 3]
        """
        old_size = self.size
        if isinstance(other, VaryingElement):
            self.resize(old_size + other.size)

            for other_key, other_val in six.iteritems(other):

                repr_key = self._find_repr_key(other_key)

                if repr_key not in self:
                    if isinstance(other_val, bitarray):
                        new_val = bitarray(self.size)
                        new_val.setall(0)
                        new_val[old_size:] = other_val
                    else:
                        new_val =  other_val + old_size
                    self._unsafe_set(repr_key, new_val)

                else:
                    old_val = super(VaryingElement, self).get(repr_key)
                    if isinstance(old_val, bitarray):
                        if isinstance(other_val, bitarray):
                            old_val[old_size:] |= other_val
                        else:
                            old_val[old_size + other_val] = 1
                    else:
                        new_val = bitarray(self.size)
                        new_val.setall(0)
                        new_val[old_val] = 1
                        if isinstance(other_val, bitarray):
                            new_val[old_size:] |= other_val
                        else:
                            new_val[old_size + other_val] = 1
                        self._unsafe_set(repr_key, new_val)
        else:
            # assume 'other' is a list-like structure of values
            self.resize(old_size + len(other))
            for idx, other_key in enumerate(other, old_size):
                repr_key = self._find_repr_key(other_key)
                self._unsafe_assign_int(repr_key, idx)


    def reduce(self, new2old):
        self._size = len(new2old)
        del_keys = []
        for key, old_val in six.iteritems(super(VaryingElement, self)):
            if isinstance(old_val, bitarray):
                # the occurences of the value are given by bitarray

                # collapse bitarray
                new_val = bitarray([old_val[i] for i in new2old])

                count = new_val.count(1)
                if count > 1:
                    # multiple occurences; add collapsed bitarray
                    self._unsafe_set(key, new_val)
                elif count == 1:
                    # one occurence; add the single position
                    self._unsafe_set(key, new_val.index(1))
                else:
                    # no occurences; add key to list so that it will be removed
                    del_keys.append(key)

            elif isinstance(old_val, Integral):
                count = new2old.count(old_val)
                if count > 1:
                    # multiple occurences; change to bitarray
                    new_val = bitarray([old_val == i for i in new2old])
                    self._unsafe_set(key, new_val)
                elif count == 1:
                    # there is a single occurence of the value
                    self._unsafe_set(key, new2old.index(old_val))
                else:
                    # no occurence; add key to list so that it will be removed
                    del_keys.append(key)

            else:
                raise TypeError

        for del_key in del_keys:
            self._unsafe_del(del_key)


    def insert(self, key, idx):
        """Insert a key index pair.

        Example
        -------
        >>> velem = VaryingElement.make_from_inverse(['a', 'a', 'a', 'a'])
        >>> velem.insert('b', 1)
        >>> velem.inverse()
        ['a', 'b', 'a', 'a', 'a']
        """
        self._size += 1
        repr_key = self._find_repr_key(key)
        # insert into other elements
        for itr_key, itr_val in six.iteritems(super(VaryingElement, self)):
            if isinstance(itr_val, bitarray):
                itr_val.insert(idx, 0)
            elif itr_val >= idx:
                self._unsafe_set(itr_key, itr_val + 1)

        # insert into key element
        val = super(VaryingElement, self).get(repr_key)
        if val is None:
            self._unsafe_set(repr_key, idx)
        elif isinstance(val, bitarray):
            val[idx] = 1
        else:
            new_val = bitarray(self._size)
            new_val.setall(0)
            new_val[val] = 1
            new_val[idx] = 1
            self._unsafe_set(repr_key, new_val)


    def append(self, key):
        """Append a key.

        Example
        -------
        >>> velem = VaryingElement.make_from_inverse(['a', 'b', 'a', 'a'])
        >>> velem.append('b')
        >>> velem.inverse()
        ['a', 'b', 'a', 'a', 'b']
        """
        for val in six.itervalues(super(VaryingElement, self)):
            if isinstance(val, bitarray):
                val.append(0)
        repr_key = self._find_repr_key(key)
        if repr_key in self:
            self._size += 1
            self._convert_to_array(repr_key)
            super(VaryingElement, self).__getitem__(repr_key)[-1] = 1
        else:
            self._unsafe_set(repr_key, self._size)
            self._size += 1


    #----------------------------------------------------------------------
    # factory methods
    #----------------------------------------------------------------------

    @classmethod
    def make_from_inverse(klass, inv, cmp_func=None, unfilled_key=None):
        """Make a VaryingElement instance from an iterable collection of keys.

        Example
        -------

        >>> values = ['a', 'a', 'b', 'a', 'b']
        >>> velem = VaryingElement.make_from_inverse(values)
        >>> velem['a']
        [0, 1, 3]
        >>> velem['b']
        [2, 4]
        """
        out = klass(size=len(inv), cmp_func=cmp_func)
        for idx, key in enumerate(inv):
            if key == unfilled_key:
                continue
            repr_key = out._find_repr_key(key)
            out._unsafe_assign_int(repr_key, idx)
        return out


    @classmethod
    def make_from_dict(klass, other, cmp_func=None):
        #TODO: better description
        """Make a VaryingElement instance from a dictionary"""

        array_lens = set(x.length()
                         for x in six.itervalues(other)
                         if isinstance(x, bitarray))

        if len(array_lens) > 1:
            raise ValueError

        try:
            max_idx = max(x
                          for x in six.itervalues(other)
                          if isinstance(x, (int, long)))
        except ValueError:
            max_idx = -1

        try:
            other_len = array_lens.pop()
            if max_idx >= other_len:
                raise IndexError
        except KeyError:
            other_len = max_idx + 1

        result = klass(size=other_len, cmp_func=cmp_func)

        for key, val in six.iteritems(other):
            result._unsafe_set(key, val)
        result.clean()

        return result


    def subset(self, new2old):
        result = VaryingElement(size=len(new2old),
                                cmp_func=self.cmp_func)
        for key, old_val in six.iteritems(super(VaryingElement, self)):
            if isinstance(old_val, bitarray):
                # the occurences of the value are given by bit array

                # collapse bitarray
                new_val = bitarray([old_val[i] for i in new2old])

                count = new_val.count(1)
                if count > 1:
                    # multiple occurences; add collapsed bitarray
                    result._unsafe_set(key, new_val)
                elif count == 1:
                    # one occurence; add the single position
                    result._unsafe_set(key, new_val.index(1))

            elif old_val in new2old:
                # there is a single occurence of the value that is given index
                result._unsafe_set(key, new2old.index(old_val))
        return result


    def check(self):
        if super(VaryingElement, self).__len__() > self._size:
            return False

        found = bitarray(self._size)
        found.setall(0)
        total = 0
        for val in six.itervalues(super(VaryingElement, self)):
            if isinstance(val, bitarray):
                if val.length() != self._size:
                    return False
                found |= val
                total += val.count(1)

            elif isinstance(val, (int, long)):
                if val >= self._size:
                    return False
                found[val] = 1
                total += 1
            else:
                raise TypeError
        return total == found.count(1)


    def clean(self):
        del_keys = []
        for key, val in six.iteritems(self):
            if isinstance(val, bitarray):
                count = val.count(1)
                if count == 1:
                    self._unsafe_set(key, val.index(1))
                elif count == 0:
                    del_keys.append(key)
            elif self._size <= val or val < 0:
                del_keys.append(key)

        for key in del_keys:
            self._unsafe_del(key)


    def inverse(self, unfilled_key=None):
        """Produce the inverse mapping of index to value.

        Parameters
        ----------
        unfilled_key : object
            value to use when an index does not have an associated key

        Example
        -------
        >>> velem = VaryingElement(size=6)
        >>> velem['a'] = [0, 3, 5]
        >>> velem['b'] = [1, 2]
        >>> velem.inverse(unfilled_key='c')
        ['a', 'b', 'b', 'a', 'c', 'a']
        """
        inv = [unfilled_key] * self.size
        for key, val in six.iteritems(super(VaryingElement, self)):
            if isinstance(val, bitarray):
                for i, _ in itertools.ifilter(lambda x: x[1], enumerate(val)):
                    inv[i] = key
            else:
                inv[val] = key
        return inv


    def where(self, key):
        """Get the positional indices for the key"""
        repr_key = self._find_repr_key(key)
        val = self.get(repr_key)
        if val is None:
            return list()
        if isinstance(val, bitarray):
            return [i for i, b in enumerate(val) if b]
        return [val]


    def at(self, idx):
        """Find the value associated with an index.

        Parameters
        ----------
        idx : int
            index to find the associated value of

        Raises
        ------
        KeyError :
            if no value is associated with the given index.
        """
        for key, val in six.iteritems(super(VaryingElement, self)):
            if isinstance(val, bitarray):
                if val[idx]:
                    return key
            else:
                if val == idx:
                    return key
        raise KeyError("Index %d is unfilled" % idx)


    def which(self, idx, unfilled_key=None):
        """Find the value associated with an index.

        Parameters
        ----------
        idx : int
            index to find the associated value of

        unfilled_key : object
            Value to return if there is no value associated with the index
        """
        if 0 > idx or idx >= self._size:
            raise IndexError("%d out of range [0, %d)" % (idx, self.size))
        for key, val in six.iteritems(super(VaryingElement, self)):
            if isinstance(val, bitarray):
                if val[idx]:
                    return key
            else:
                if val == idx:
                    return key
        return unfilled_key


    def variation_wrt_axes(self, axes_elems):

        # for the special case of a single axis
        if len(axes_elems) == 1:
            if varying_elem.is_constant():
                return []
            else:
                return [0]

        # if any key has only one occurence (value is an int) then there is
        # variation over every axis
        for val in six.itervalues(self):
            if not isinstance(val, bitarray):
                return range(len(axes_elems))

        # it is a theoretical requirment that an axis must be comprised of only
        # bitarrays when there are multiple axes
        for axis_elem in axes_elems:
            for val in six.itervalues(axis_elem):
                if not isinstance(val, bitarray):
                    raise ValueError

        line = bitarray(self.size)
        varying_axes = []
        for cur_axis_idx, cur_axis in enumerate(axes_elems):
            # determine if there are more values than lines for this axis
            if len(varying_elem) > self.size / len(cur_axis):
                varying_axes.append(cur_axis_idx)
                continue

            # loop over all lines
            other_axes_itervalues = tuple(six.itervalues(x)
                                          for i, x in enumerate(axes_elems)
                                          if i != cur_axis_idx)
            for axes_values in itertools.product(*other_axes_itervalues):
                # determine a line of points that are parallel to the current
                # axis
                line.setall(1)
                for axis_value in axes_values:
                    line &= axis_value

                for key, val in six.iteritems(self):
                    # test if any value partially fills the line, thus
                    # implying variation along the line
                    overlap = line & val
                    if any(overlap) and overlap != line:
                        break
                else:
                    # if loop completed without a break then the value along the
                    # line is constant.  Continue to next line
                    continue

                # if loop was broken out of, then the line was found to be
                # varying. Therefore there is variation over this axis
                varying_axes.append(cur_axis_idx)
                break

        return varying_axes


#===============================================================================
#
#
#
#===============================================================================
def find_axes(guess_keys, var_elems):
    #TODO; better description
    """Find a set of elements whose values form a coordinate basis.
    """
    possible = []
    for key in guess_keys:
        elem = var_elems.get(key)
        if elem is not None and elem.is_square():
            if len(elem) == elem.size:
                return (key,)
            possible.append((key, elem))

    for i, (key, elem) in enumerate(possible):
        axes_keys = _find_axes_recurse(possible[i+1:], (elem,), len(elem))
        if axes_keys is not None:
            return (key,) + axes_keys

    return None


def _find_axes_recurse(keys_info, prv_elems, cum_prod):
    # this is a helper function to find_axes
    for i, (key, elem) in enumerate(keys_info):
        if elem.size % (len(elem) * cum_prod) == 0:
            if all(elem.is_complementary(prv_elem) for prv_elem in prv_elems):
                if elem.size == len(elem) * cum_prod:
                    return (key,)
                sub_keys = _find_axes_recurse(keys_info[i+1:],
                                              prv_elems + (elem,),
                                              cum_prod * len(elem))
                if sub_keys is not None:
                    return (key,) + sub_keys
    return None


def determine_variation(varying_elem, axes_elems):

    # for the special case of a single axis
    if len(axes_elems) == 1:
        if varying_elem.is_constant():
            return []
        else:
            return [0]

    # if any key has only one occurence (value is an int) then there is
    # variation over every axis
    for val in six.itervalues(varying_elem):
        if not isinstance(val, bitarray):
            return range(len(axes_elems))

    # it is a theoretical requirment that an axis must be comprised of only
    # bitarrays when there are multiple axes
    for axis_elem in axes_elems:
        for val in six.itervalues(axis_elem):
            if not isinstance(val, bitarray):
                raise ValueError

    line = bitarray(varying_elem.size)
    varying_axes = []
    for cur_axis_idx, cur_axis in enumerate(axes_elems):
        # determine if there are more values than lines for this axis
        if len(varying_elem) > varying_elem.size / len(cur_axis):
            varying_axes.append(cur_axis_idx)
            continue

        # loop over all lines
        other_axes_itervalues = tuple(six.itervalues(x)
                                      for i, x in enumerate(axes_elems)
                                      if i != cur_axis_idx)
        for axes_values in itertools.product(*other_axes_itervalues):
            # determine a line of points that are parallel to the current
            # axis
            line.setall(1)
            for axis_value in axes_values:
                line &= axis_value

            for key, val in six.iteritems(varying_elem):
                # test if any value partially fills the line, thus
                # implying variation along the line
                overlap = line & val
                if any(overlap) and overlap != line:
                    break
            else:
                # if loop completed without a break then the value along the
                # line is constant.  Continue to next line
                continue

            # if loop was broken out of, then the line was found to be
            # varying. Therefore there is variation over this axis
            varying_axes.append(cur_axis_idx)
            break

    return varying_axes


def summary_variation(elem_summary, axes_elems):

    # mapping of axes coordinates to flat index
    ndim = len(axes_elems)
    shape = tuple(len(ax) for ax in axes)
    coord_to_idx = np.ndarray(shape, dtype=int, order='C')
    ax_val_iters = tuple(six.itervalues(ax) for ax in axes_elems)
    for fidx, ax_vals in enumerate(itertools.product(*ax_val_iters)):
        # determine the set of indices that have these axes coordinates
        overlap = ax_vals[0].copy()
        for ax_val in ax_vals[1:]:
            overlap &= ax_val
        # a single 1 in bitarray at the index for these coordinates
        coord_to_idx.flat[fidx] = overlap.index(1)

    # determine how each varying element is varying with respect to the given
    # axes, and group similar varying elements together
    variation_groups = dict()
    for key, vary_elem in six.iteritems(elem_summary._varying_elems):
        var_axes = vary_elem.variation_wrt_axes(axes_elems)
        if var_axes not in variation_groups:
            variation_groups[var_axes] = dict()
        # use VaryElement instance for key as a placeholder that will be
        # replaced by the array of values
        variation_groups[var_axes][key] = vary_elem

    # loop over elements, replacing VaryingElement instances with arrays of
    # values collapsed to only those axes for which the values depend
    for vary_axes, vary_elems in six.iteritems(variation_groups):

        dim_diff = len(axes_elems) - len(vary_axes)
        if dim_diff > 0:
            tp = tuple(i for i in range(ndim) if i not in vary_axes) + vary_axes
            grp_coord_to_idx = np.transpose(coord_to_idx, tp)[(0,)*dim_diff]
        else:
            grp_coord_to_idx = np.transpose(coord_to_idx, vary_axes)

        for key, vary_elem in six.iteritems(vary_elems):
            vary_elems[key] = np.asarray(vary_elem.inverse())[grp_coord_to_idx]

    return variation_groups
