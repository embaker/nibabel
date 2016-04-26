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
    def make_from_lists(cls, data, unfilled_value=None, compare_rules=None):
        """

        Example
        -------
        >>> data = [('a', [1, 2, 1]), ('b', [3, 3, 3])]
        >>> summary = ElementSummary.make_from_lists(data)
        """

        result = cls(compare_rules=compare_rules)

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
            elem = VaryingElement.make_from_list(values,
                                                 unfilled_value=unfilled_value,
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
                    var_elem[const_val] = slice(0, idx)
                    var_elem[value] = idx
                    self._varying_elems[key] = var_elem
                    del self._const_elems[key]
                # The value is still const, continue
                continue

            if key in self._varying_elems:
                # Mark location in varying element object
                self._varying_elems[key].assign(value, idx)
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
            var_elem[self._const_elems[key]] = slice(0, idx)
            self._varying_elems[key] = var_elem
            del self._const_elems[key]


    def _append_summary(self, idx, other):
        # extensively used structures
        if other.size == 1:
            other_part = idx
        else:
            other_part = slice(idx, idx+other.size)

        if idx == 1:
            this_part = 0
        else:
            this_part = slice(0, idx)

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
            var_elem[other._const_elems[key]] = other_part
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
            var_elem[self._const_elems[key]] = this_part
            self._varying_elems[key] = var_elem
            del self._const_elems[key]

        # intersection of self.const_elems and other.varying_elems
        for key in this_const_other_vary:
            var_elem = VaryingElement(size=self.size,
                                      cmp_func=self.compare_rules.get(key))
            var_elem[self._const_elems[key]] = this_part
            self._varying_elems[key] = var_elem
            del self._const_elems[key]

        # intersection of varying elements
        wrk_array = bitarray(self.size)
        for key in this_vary_other_vary | this_const_other_vary:
            other_elem = other._varying_elems[key]
            for other_key, other_val in six.iteritems(other_elem):
                if isinstance(other_val, bitarray):
                    wrk_array.setall(0)
                    wrk_array[other_part] = other_val
                    self._varying_elems[key].assign(other_key, wrk_array)
                else:
                    self._varying_elems[key].assign(other_key, idx + other_val)

        # intersection of constant elements
        for key in this_const_other_const:
            cmp_func = self.compare_rules.get(key)
            cmp_rep = operator.eq if cmp_func is None else cmp_func
            if not cmp_rep(self._const_elems[key], other._const_elems[key]):
                var_elem = VaryingElement(size=self.size, cmp_func=cmp_func)
                var_elem[self._const_elems[key]] = this_part
                var_elem[other._const_elems[key]] = other_part
                self._varying_elems[key] = var_elem
                del self._const_elems[key]

        # intersection of self.varying_elems and other.const_elems
        for key in this_vary_other_const:
            self._varying_elems[key].assign(other._const_elems[key], other_part)


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
        if not indices:
            return result

        result._size = len(indices)
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
            for pvt_key, pvt_val in six.iteritems(pvt_elem):
                if isinstance(pvt_val, bitarray):
                    new_mask = mask & pvt_val
                    if any(new_mask):
                        yield pvt_key, new_mask
                elif mask[pvt_val] == 1:
                    yield pvt_key, integer_to_bitarray(pvt_val, self.size)
            # yield sub-summary for unfilled indices
            if pvt_elem.unfilled is not None:
                new_mask = pvt_elem.unfilled & mask
                if any(new_mask):
                    yield default, new_mask


    def split(self, key, default=None):
        for val, val_mask in self.split_mask(key, default=default):
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
        for grp_vals, grp_mask in self.group_mask(keys, default=default):
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
                vary_elems[key] = np.asarray(vary_elem.to_list())[grp_mapping]

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
class VeryingElementValueTypeError(Exception):
    pass

class VaryingElement(collections.OrderedDict):
    """A memory efficient structure for a one-to-many mapping of a value to a
    positional index.

    This mapping is the inverse of that of a list or array:

    >>> values = ['a', 'a', 'b', 'a', 'b']
    >>> velem = VaryingElement.make_from_list(values)
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
            raise ValueError("size is less than 0")
        super(VaryingElement, self).__init__()
        # initialize the internal atrribute that keeps track of the size
        self._size = size
        # initialize self._unfilled; the attribute that keeps track of the
        # unfilled indices
        if self._size == 0:
            self._unfilled = None
        else:
            self._unfilled = ones_bitarray(self._size)
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


    def __getitem__(self, key):
        repr_key = self.get_key(key)
        return super(VaryingElement, self).__getitem__(repr_key)


    def get(self, key, default=None):
        repr_key = self.get_key(key)
        return super(VaryingElement, self).get(repr_key, default)


    def get_key(self, key):
        if self.cmp_func is None:
            return key
        else:
            for repr_key in six.iterkeys(self):
                if self.cmp_func(key, repr_key):
                    return repr_key
            return key

    def get_item(self, key, default=None):
        """Retrieve a key already contained in the element that is equivalent to
        the supplied key along with its value. If there is no equivalent key
        already in the element, then the supplied key with the value of supplied
        by the default parameter will be returned.
        """
        repr_key = self.get_key(key)
        return repr_key, super(VaryingElement, self).get(repr_key, default)


    def __delitem__(self, key):
        repr_key, val = self.get_item(key)
        if val is not None:
            if self._unfilled is None:
                self._unfilled = as_bitarray(val, self.size)
            elif isinstance(val, bitarray):
                self._unfilled |= val
            elif isinstance(val, Integral):
                self._unfilled[val] = 1
            else:
                raise VeryingElementValueTypeError
            self._unsafe_del(repr_key)
        else:
            raise KeyError


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
        for old_key, old_val in six.iteritems(self):
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
        >>> velem = VaryingElement.make_from_list(['a', 'a', 'b', 'a', 'b'])
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
        # ensure that the index is with in the range
        if index < 0 or index >= self.size:
            raise IndexError
        # retrieve already contained equivalent key and its value
        repr_key, val  = self.get_item(key)
        if self._unfilled is None:
            # element is full
            if val is None:
                # key is not already in the element
                self._unassign_int(index)
                self._unsafe_set(repr_key, index)
            elif isinstance(val, bitarray):
                if val[index] == 0:
                    # only unassign if index is not already associated with key
                    self._unassign_int(index, {repr_key})
                # move the indices being overwritten to unfilled
                self._unfilled = val
                self._unfilled[index] = 0
                # replace key's value with supplied index
                self._unsafe_set(repr_key, index)
            elif isinstance(val, Integral):
                if val != index:
                    # only perform work if changing the key's value
                    self._unassign_int(index, {repr_key})
                    self._unfilled = integer_to_bitarray(val, self.size)
                    self._unsafe_set(repr_key, index)
            else:
                raise VeryingElementValueTypeError
        else:
            # element is not full
            # move the value being replaced to unfilled
            if isinstance(val, bitarray):
                self._unfilled |= val
            elif isinstance(val, Integral):
                self._unfilled[val] = 1
            elif val is not None:
                raise VeryingElementValueTypeError
            # only perform unassignment work if index is associated with another
            # key
            if self._unfilled[index] == 0:
                self._unassign_int(index, {repr_key})
            # replace key's value with supplied index
            self._unsafe_set(repr_key, index)
            # update and clean unfilled index tracker
            self._unfilled[index] = 0
            if not any(self._unfilled):
                self._unfilled = None


    def _setitem_bitarray(self, key, indices):
        # enusre that input bitarray is of valid size
        if indices.length() != self.size:
            raise ValueError
        # handle special cases
        count = indices.count(1)
        if count == 0:
            # treat empty bitarray as a delete
            try:
                self.__delitem__(key)
            except KeyError:
                pass
            return
        if count == 1:
            # use integer specialization if only a single bit is set
            self._setitem_int(key, indices.index(1))
            return

        # retrieve already contained equivalent key and its value
        repr_key, val  = self.get_item(key)

        if self._unfilled is None:
            # element is full
            if val is None:
                # key is not already in the element
                self._unassign_bitarray(indices)
                self._unsafe_set(repr_key, indices)
            elif isinstance(val, bitarray):
                # the key already has bitarray for a value
                if any(indices & ~val):
                    # only unassign if indices are not already associated with
                    # the supplied key
                    self._unassign_bitarray(indices, {repr_key})
                # move to unfilled the indices being overwritten and not in the
                # new set of  indices to unfilled
                self._unfilled = val & ~indices
                if not any(self._unfilled):
                    self._unfilled = None
                # replace key's value with supplied bitarray
                self._unsafe_set(repr_key, indices)
            elif isinstance(val, Integral):
                # the key already has integer for a value
                if indices[val] == 0:
                    # only need to create unfilled if index not in replacement
                    self._unfilled = integer_to_bitarray(val, self.size)
                # always need to unassign
                self._unassign_bitarray(indices, {repr_key})
                self._unsafe_set(repr_key, indices)
            else:
                raise VeryingElementValueTypeError
        else:
            # element is not full
            # move the value being replaced to unfilled
            if isinstance(val, bitarray):
                self._unfilled |= val
            elif isinstance(val, Integral):
                self._unfilled[val] = 1
            elif val is not None:
                raise VeryingElementValueTypeError
            # only perform unassignment work if any indices are associated with
            # another key
            if any(indices & ~self._unfilled):
                self._unassign_bitarray(indices, {repr_key})
            # replace key's value with supplied index
            self._unsafe_set(repr_key, indices)
            # update and clean unfilled index tracker
            self._unfilled &= ~indices
            if not any(self._unfilled):
                self._unfilled = None


    def __setitem__(self, key, indices):
        """Set the indices associated with a key.

        Any indices previously associated with the key that are not in the newly
        given indices will be disassociated with the key. The newly given
        indices will be unassigned from all other keys.

        The given indices must agree with the current size. NO automatic
        resizing is done.

        Example
        -------
        >>> velem = VaryingElement.make_from_list(['a', 'a', 'b', 'a', 'b'])
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
            self._setitem_bitarray(key, indices)
        elif isinstance(indices, Integral):
            self._setitem_int(key, indices)
        elif isinstance(indices, slice):
            barray = slice_to_bitarray(indices, self.size)
            self._setitem_bitarray(key, barray)
        elif isinstance(indices, collections.Iterable):
            barray = iterable_to_bitarray(indices, self.size)
            self._setitem_bitarray(key, barray)
        else:
            raise VeryingElementValueTypeError


    def _assign_int(self, key, index):
        # check that the index is within the bounds
        if index >= self.size or index < 0:
            raise IndexError
        # retrieve already contained equivalent key and its value
        repr_key, val = self.get_item(key)
        if self._unfilled is None or self._unfilled[index] == 0:
            if isinstance(val, Integral):
                if index != val:
                    self._unassign_int(index, {repr_key})
                    new_val = iterable_to_bitarray((val, index), self.size)
                    self._unsafe_set(repr_key, new_val)
            elif isinstance(val, bitarray):
                if val[index] != 1:
                    self._unassign_int(index, {repr_key})
                    val[index] = 1
            elif val is None:
                self._unassign_int(index, {repr_key})
                self._unsafe_set(key, index)
            else:
                raise VeryingElementValueTypeError
        else:
            if isinstance(val, Integral):
                new_val = iterable_to_bitarray((val, index), self.size)
                self._unsafe_set(repr_key, new_val)
            elif isinstance(val, bitarray):
                val[index] = 1
            elif val is None:
                self._unsafe_set(key, index)
            else:
                raise VeryingElementValueTypeError
            # remove newly assigned from unfilled array
            self._unfilled[index] = 0
            if not any(self._unfilled):
                self._unfilled = None


    def _assign_bitarray(self, key, indices):
        # enusre that input bitarray is of valid size
        if indices.length() != self.size:
            raise ValueError
        # handle special cases
        count = indices.count(1)
        if count == 0:
            # don't do anything if bitarray is empty
            return
        if count == 1:
            # use integer specialization if only a single bit is set
            self._assign_int(key, indices.index(1))
            return
        # retrieve already contained equivalent key and its value
        repr_key, val = self.get_item(key)
        if self._unfilled is None:
            self._unassign_bitarray(indices, {repr_key})
            if val is None:
                self._unsafe_set(key, indices.copy())
            elif isinstance(val, bitarray):
                val |= indices
            elif isinstance(val, Integral):
                new_val = indices.copy()
                new_val[val] = 1
                self._unsafe_set(repr_key, new_val)
            else:
                raise VeryingElementValueTypeError
        else:
            if val is None:
                cmp_array = self._unfilled
                self._unsafe_set(repr_key, indices.copy())
            elif isinstance(val, bitarray):
                cmp_array = val | self._unfilled
                val |= indices
            elif isinstance(val, Integral):
                cmp_array = self._unfilled.copy()
                cmp_array[val] = 1
                new_val = indices.copy()
                new_val[val] = 1
                self._unsafe_set(repr_key, new_val)
            else:
                raise VeryingElementValueTypeError
            # unassign bits from other keys if neccessary
            if any(indices & ~cmp_array):
                self._unassign_bitarray(indices, {repr_key})
            # remove newly assigned from unfilled array
            self._unfilled &= ~indices
            if not any(self._unfilled):
                self._unfilled = None


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
        >>> velem = VaryingElement.make_from_list(['a', 'a', 'b', 'a', 'b'])
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
            self._assign_bitarray(key, indices)
        elif isinstance(indices, Integral):
            self._assign_int(key, indices)
        elif isinstance(indices, slice):
            self.assign(key, slice_to_bitarray(indices, self.size))
        elif isinstance(indices, collections.Iterable):
            self.assign(key, iterable_to_bitarray(indices, self.size))
        else:
            raise VeryingElementValueTypeError


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


    def is_empty(self):
        """Test if there are no indices mapped to a key"""
        return len(self) == 0


    def is_full(self):
        """Test if every index has a key mapped to it"""
        return self._unfilled is None


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
        >>> velem1 = VaryingElement.make_from_list(['a', 'a', 'b', 'a', 'b'])
        >>> velem2 = VaryingElement.make_from_list([5.0, 5.0, 2.0, 5.0, 2.0])
        >>> velem1.is_equivalent(velem2)
        True
        >>> velem2.is_equivalent(velem1)
        True
        >>> velem3 = VaryingElement.make_from_list([1, 2, 2, 2, 1])
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

        if (isinstance(self._unfilled, bitarray) and
            isinstance(other._unfilled, bitarray)):
            # map the unfilled elements
            this_indices = bitarray_to_indices(self._unfilled)
            other_indices = bitarray_to_indices(other._unfilled)
            if len(other_indices) != len(this_indices):
                return None
            for other_idx, this_idx in zip(other_indices, this_indices):
                other2this[other_idx] = this_idx
        elif self.unfilled is not None or other.unfilled is not None:
            return None

        for key in six.iterkeys(self):
            this_indices = self.where(key)
            other_indices = other.where(key)
            if len(other_indices) != len(this_indices):
                return None
            for other_idx, this_idx in zip(other_indices, this_indices):
                other2this[other_idx] = this_idx

        return other2this


    @property
    def unfilled(self):
        """Which indices do not have a key associated with them.

        Example
        -------
        >>> velem = VaryingElement.make_from_list(['a', 'a', 'b', 'a', 'b'])
        >>> velem.unassign(indices=[0, 1, 2], ignore_keys={'b'})
        >>> velem['a']
        [3]
        >>> velem['b']
        [2, 4]
        >>> velem.unfilled
        '10010'
        """
        return self._unfilled


    #----------------------------------------------------------------------
    # modifier functions
    #----------------------------------------------------------------------

    def resize(self, new_size):
        """Change the number of indices"""
        # check that the size is not trying to be set to a negative value
        if new_size < 0:
            raise ValueError("size less than 0")

        old_size = self._size
        self._size = new_size
        size_diff = new_size - old_size

        if size_diff > 1:
            # increasing size
            padding = zeros_bitarray(size_diff)
            for key, val in six.iteritems(super(VaryingElement, self)):
                if isinstance(val, bitarray):
                    val.extend(padding)
            # mark all new indices as unfilled
            if isinstance(self._unfilled, bitarray):
                padding.setall(1)
                self._unfilled.extend(padding)
            else:
                slc = slice(old_size, new_size)
                self._unfilled = slice_to_bitarray(slc, new_size)

        elif size_diff == 1:
            # increasing size by one
            for key, val in six.iteritems(super(VaryingElement, self)):
                if isinstance(val, bitarray):
                    val.append(0)
            # mark all new indices as unfilled
            if isinstance(self._unfilled, bitarray):
                self._unfilled.append(1)
            else:
                self._unfilled = integer_to_bitarray(old_size, new_size)

        elif size_diff < 0:
            # decreasing size
            del_keys = []
            for key, val in six.iteritems(super(VaryingElement, self)):
                if isinstance(val, bitarray):
                    new_val = val[:new_size]
                    count = new_val.count(1)
                    if count == 0:
                        del_keys.append(key)
                    elif count == 1:
                        self._unsafe_set(key, new_val.index(1))
                    else:
                        self._unsafe_set(key, new_val)
                elif val > new_size:
                    del_keys.append(key)

            for key in del_keys:
                self._unsafe_del(key)

            # shrink unfilled bitarray
            if isinstance(self._unfilled, bitarray):
                self._unfilled = self._unfilled[:new_size]
                if not any(self._unfilled):
                    self._unfilled = None


    def reorder(self, new2old):
        """Reorder the indices.

        Example
        -------
        >>> velem = VaryingElement.make_from_list(['a', 'b', 'c', 'd', 'e'])
        >>> velem.reorder([2, 3, 1, 0, 4])
        >>> velem.to_list()
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


    def extend_with_list(self, other, unfilled_value=None):
        old_size = self.size
        self.resize(old_size + len(other))
        for idx, key in enumerate(other):
            if key == unfilled_value:
                # do nothing for input indicating unfilled elements
                continue
            # mark as filled in unfilled tracker
            self._unfilled[idx] = 0
            repr_key, val = self.get_item(key)
            if val is None:
                self._unsafe_set(repr_key, idx)
            elif isinstance(val, bitarray):
                val[idx] = 1
            else:
                new_val = iterable_to_bitarray((val, idx), self.size)
                self._unsafe_set(repr_key, new_val)
        # clean unfilled
        if not any(self._unfilled):
            self._unfilled = None


    def extend(self, other, unfilled_value=None):
        """Merge a VaryingElement instance into this one.

        Example
        -------
        >>> velem1 = VaryingElement.make_from_list(['a', 'a', 'b', 'a'])
        >>> velem2 = VaryingElement.make_from_list(['c', 'b', 'b', 'c'])
        >>> velem1.extend(velem2)
        >>> velem1.to_list()
        ['a', 'a', 'b', 'a', 'c', 'b', 'b', 'c']
        >>> velem1['b']
        [2, 5, 6]

        Example
        -------
        >>> velem = VaryingElement.make_from_list(['a', 'b'])
        >>> more_values = ['c', 'b']
        >>> velem.extend(more_values)
        >>> velem.to_list()
        [a', 'b', c', 'b']
        >>> velem['b']
        [1, 3]
        """

        if isinstance(other, VaryingElement):
            old_size = self.size
            self.resize(old_size + other.size)

            # append unfilled
            if other._unfilled is not None:
                self._unfilled[old_size:] = other._unfilled
            # clean unfilled
            if not any(self._unfilled):
                self._unfilled = None

            for other_key, other_val in six.iteritems(other):
                repr_key, val = self.get_item(other_key)

                if val is None:
                    if isinstance(other_val, bitarray):
                        new_val = zeros_bitarray(self.size)
                        new_val[old_size:] = other_val
                    elif isinstance(other_val, Integral):
                        new_val =  other_val + old_size
                    self._unsafe_set(repr_key, new_val)

                elif isinstance(val, bitarray):
                    if isinstance(other_val, bitarray):
                        val[old_size:] |= other_val
                    elif isinstance(other_val, Integral):
                        val[old_size + other_val] = 1

                elif isinstance(val, Integral):
                    new_val = integer_to_bitarray(val, self.size)
                    if isinstance(other_val, bitarray):
                        new_val[old_size:] |= other_val
                    elif isinstance(other_val, Integral):
                        new_val[old_size + other_val] = 1
                    self._unsafe_set(repr_key, new_val)
        else:
            # assume 'other' is a list-like structure of values
            self.extend_with_list(other, unfilled_value)


    def insert(self, key, idx):
        """Insert a key index pair.

        Example
        -------
        >>> velem = VaryingElement.make_from_list(['a', 'a', 'a', 'a'])
        >>> velem.insert('b', 1)
        >>> velem.to_list()
        ['a', 'b', 'a', 'a', 'a']
        """
        self._size += 1
        # insert filled marker bit into unfilled tracker
        if self._unfilled is not None:
            self._unfilled.insert(idx, 0)
        # insert space into all items
        for itr_key, itr_val in six.iteritems(self):
            if isinstance(itr_val, bitarray):
                itr_val.insert(idx, 0)
            elif isinstance(itr_val, Integral):
                if itr_val >= idx:
                    self._unsafe_set(itr_key, itr_val + 1)
        # insert into key element
        repr_key, val = self.get_item(key)
        if val is None:
            self._unsafe_set(repr_key, idx)
        elif isinstance(val, bitarray):
            val[idx] = 1
        else:
            new_val = iterable_to_bitarray((val, idx), self.size)
            self._unsafe_set(repr_key, new_val)


    def append(self, key):
        """Append a key.

        Example
        -------
        >>> velem = VaryingElement.make_from_list(['a', 'b', 'a', 'a'])
        >>> velem.append('b')
        >>> velem.to_list()
        ['a', 'b', 'a', 'a', 'b']
        """
        old_size = self.size
        self._size += 1
        # append filled marker bit into unfilled tracker
        if self._unfilled is not None:
            self._unfilled.append(0)
        # append all bitarrays
        for val in six.itervalues(self):
            if isinstance(val, bitarray):
                val.append(0)
        # actual insertion of value
        repr_key, val = self.get_item(key)
        if val is None:
            self._unsafe_set(repr_key, old_size)
        elif isinstance(val, bitarray):
            val[old_size] = 1
        elif isinstance(val, Integral):
            new_val = zeros_bitarray(self.size)
            new_val[val] = 1
            new_val[old_size] = 1
            self._unsafe_set(repr_key, new_val)


    #----------------------------------------------------------------------
    # factory methods
    #----------------------------------------------------------------------

    @classmethod
    def make_from_list(klass, vals, unfilled_value=None, **kargs):
        """Make a VaryingElement instance from an iterable collection of keys.

        Example
        -------

        >>> values = ['a', 'a', 'b', 'a', 'b']
        >>> velem = VaryingElement.make_from_list(values)
        >>> velem['a']
        [0, 1, 3]
        >>> velem['b']
        [2, 4]
        """
        out = klass(size=0, **kargs)
        out.extend_with_list(vals, unfilled_value)
        return out


    @classmethod
    def make_constant(cls, value, size, **kargs):
        """Makes a new instance that is filled with a single value"""
        out = cls(size=size, **kargs)
        if out.size == 1:
            out._unsafe_set(value, 0)
        elif out.size > 1:
            out._unsafe_set(value, out._unfilled)
        out._unfilled = None
        return out


    @classmethod
    def make_combined(cls, elems, **kargs):
        """Make a new instance by combining multiple instances together.
        """
        if not all(isinstance(elem, VaryingElement) for elem in elems):
            raise ValueError
        out = cls(size=sum(elem.size for elem in elems), **kargs)
        idx0 = 0
        for elem in elems:
            if elem.size == 0:
                continue
            # the range in the new element for which the current element will be
            # inserted
            slc = slice(idx0, idx0 + elem.size)
            # insert unfilled
            if elem.unfilled is None:
                out._unfilled[slc] = 0
            else:
                out._unfilled[slc] = elem.unfilled
            # insert values
            for key, val in six.iteritems(elem):
                okey, oval = out.get_item(key)
                if isinstance(val, bitarray):
                    if oval is None:
                        new_val = zeros_bitarray(out.size)
                        new_val[slc] = val
                        out._unsafe_set(okey, new_val)
                    elif isinstance(oval, bitarray):
                        oval[slc] = val
                    elif isinstance(oval, Integral):
                        new_val = zeros_bitarray(out.size)
                        new_val[slc] = val
                        new_val[oval] = 1
                        out._unsafe_set(okey, new_val)
                elif isinstance(val, Integral):
                    idx = idx0 + val
                    if oval is None:
                        out._unsafe_set(okey, idx)
                    elif isinstance(oval, bitarray):
                        oval[idx] = 1
                    elif isinstance(oval, Integral):
                        new_val = zeros_bitarray(out.size)
                        new_val[oval] = 1
                        new_val[idx] = 1
                        out._unsafe_set(okey, new_val)
            # increment origin index
            idx0 += elem.size
        # clean unfilled tracker
        if not any(out._unfilled):
            out._unfilled = None
        return out


    @classmethod
    def make_from_dict(cls, other, cmp_func=None):
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

        result = cls(size=other_len, cmp_func=cmp_func)

        for key, val in six.iteritems(other):
            result._unsafe_set(key, val)
        result.clean()

        return result


    def subset(self, new2old):
        result = VaryingElement(size=len(new2old),
                                cmp_func=self.cmp_func)
        # collapse unfilled tracker
        if self._unfilled is None:
            result._unfilled = None
        else:
            result._unfilled = bitarray([self._unfilled[i] for i in new2old])
            if not any(result._unfilled):
                result._unfilled = None

        for key, old_val in six.iteritems(self):
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
        self._unfilled = zeros_bitarray(self.size)
        for key, val in six.iteritems(self):
            if isinstance(val, bitarray):
                self._unfilled |= val
                count = val.count(1)
                if count == 1:
                    self._unsafe_set(key, val.index(1))
                elif count == 0:
                    del_keys.append(key)
            elif isinstance(val, Integral):
                if self.size <= val or val < 0:
                    del_keys.append(key)
                else:
                    self._unfilled[val] = 1

        for key in del_keys:
            self._unsafe_del(key)

        self._unfilled.invert()
        if not any(self._unfilled):
            self._unfilled = None


    def to_list(self, unfilled_key=None):
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
        >>> velem.to_list(unfilled_key='c')
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
        repr_key, val = self.get_item(key)
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
def product(*args, **kargs):
    """Iterate """
    mask = kargs.get('mask')
    if mask is None:
        mask = ones_bitarray(args[0].size)
    dflt = kargs.get('default')
    if len(args) == 1:
        # last element in recursion
        for key, val in six.iteritems(args[0]):
            if isinstance(val, Integral):
                if mask[val]:
                    yield (key,), integer_to_bitarray(val, args[0].size)
            elif isinstance(val, bitarray):
                prd = mask & val
                if any(prd):
                    yield (key,), prd
        if args[0].unfilled is not None:
            prd = mask & args[0].unfilled
            if any(prd):
                yield (dflt,), prd
    else:
        for key, val in six.iteritems(args[0]):
            if isinstance(val, Integral):
                if mask[val]:
                    prd = integer_to_bitarray(val, args[0].size)
                    for sky, svl in product(*args[1:], default=dflt, mask=prd):
                        yield (key,) + sk, sv
            elif isinstance(val, bitarray):
                prd = mask & val
                if any(prd):
                    for sky, svl in product(*args[1:], default=dflt, mask=prd):
                        yield (key,) + sky, svl
        if args[0].unfilled is not None:
            prd = mask & args[0].unfilled
            if any(prd):
                for sky, svl in product(*args[1:], default=dflt, mask=prd):
                    yield (dflt,) + sky, svl


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
            vary_elems[key] = np.asarray(vary_elem.to_list())[grp_coord_to_idx]

    return variation_groups
