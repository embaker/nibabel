"""Data structures for summarizing (key,value) pairs across many instances"""
import numpy as np
import six
import collections
import itertools
import operator
from bitarray import bitarray
from copy import deepcopy
from numbers import Integral


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

#===============================================================================
#
#
#
#===============================================================================
class TagCollisionError(Exception):
    pass


class ElementSummary(object):
    """ Summary of the attributes of multiple DICOM files """

    def __init__(self, compare_rules=None):
        if compare_rules is not None:
            self._compare_rules = compare_rules
        else:
            self._compare_rules = dict()
        self.clear()


    @property
    def size(self):
        return len(self._tags)


    def __len__(self):
        return len(self._const_elems) + len(self._varying_elems )


    def clear(self):
        """Empty the internal data structures"""
        self._tags = tuple()
        self._const_elems = dict()
        self._varying_elems = dict()


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


    def get(self, key, defualt=None):
        if key in self._const_elems:
            return self._const_elems[key]
        if key in self._varying_elems:
            return self._varying_elems[key]
        return default


    @property
    def tags(self):
        return self._tags


    def _unsafe_append(self, idx, data_dict, size):
        # This function is considered unsafe because it assumes that the varying
        # elements have been resized to accomidate append data at index 'idx',
        # and it assumes that that it is being called with consecutive indices

        # all elements are constant if this the first dict
        if idx == 0:
            if isinstance(data_dict, dict):
                self._const_elems = deepcopy(data_dict)
            else:
                self._const_elems = dict(data_dict)
            return

        # convenience coversion for dictionaries
        if isinstance(data_dict, dict):
            data_generator = six.iteritems(data_dict)
        else:
            data_generator = data_dict

        for key, value in data_generator:
            # Retrieve the comparison function
            cmp_func = self._compare_rules.get(key)

            if key in self._const_elems:
                const_val = self._const_elems[key]
                cmp_rep = operator.eq if cmp_func is None else cmp_func
                if not cmp_rep(const_val, value):
                    # The element is not const, move it to the varying
                    var_elem = VaryingElement(size=size, cmp_func=cmp_func)
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
                var_elem = VaryingElement(size=size, cmp_func=cmp_func)
                var_elem[value] = idx
                self._varying_elems[key] = var_elem


    def append(self, tag, data):
        # assure no duplicates
        if tag in self._tags:
            raise TagCollisionError

        old_size = self.size
        new_size = old_size + 1
        for var_elem in six.iteritems(self._varying_elems):
            var_elem.resize(new_size)
        self._unsafe_append(old_size, data, new_size)
        self._tags += (tag,)


    def extend(self, tag_data_pairs, guess_size=None):

        chunk_size = 8

        old_size = self.size

        if guess_size is None:
            if hasattr(tag_data_pairs, '__len__'):
                guess_size = len(tag_data_pairs)
            else:
                # will cause the size to be increased to the next largest
                # multiple of chunk_size
                guess_size = chunk_size - (old_size % chunk_size)

        # increase size to allow for new data to be added
        max_size = old_size + guess_size
        for var_elem in six.iteritems(self._varying_elems):
            var_elem.resize(max_size)

        # convenience conversion to iterator for dictionaries
        if isinstance(tag_data_pairs, dict):
            pair_iterator = six.iteritems(tag_data_pairs)
        else:
            pair_iterator = tag_data_pairs

        try:
            for idx, (tag, data) in enumerate(pair_iterator, old_size):
                # assure no duplicates
                if tag in self._tags:
                    raise TagCollisionError

                # increase the varying elements' sizes when more space is needed
                if idx >= max_size:
                    max_size += chunk_size
                    for var_elem in six.itervalues(self._varying_elems):
                        var_elem.resize(max_size)

                self._tags += (tag,)
                self._unsafe_append(idx, data, max_size)

        except TagCollisionError:
            # if a duplicate tag has been found, revert to the prior state
            self._tags = self._tags[:old_size]
            for var_elem in six.itervalues(self._varying_elems):
                var_elem.resize(old_size)
            raise

        # shrink the varying elements' sizes to the true size if neccessary
        if max_size != self.size:
            for var_elem in six.itervalues(self._varying_elems):
                var_elem.resize(self.size)


    def remove(self, tags):
        """ Remove the given tags from the summary """

        # index of tags to keep
        kept_indices = [i for i, tag in enumerate(self._tags)
                        if tag not in tags]

        # return if there are no tags to remove
        if len(kept_indices) == len(self._tags):
            return

        # removing the given tags from self._tags
        self._tags = [self._tags[i] for i in kept_indices]

        # remove indices from all the varying elements
        del_keys = []
        for key, var_elem in six.iteritems(self._varying_elems):
            var_elem.reduce(kept_indices)
            if var_elem.is_empty():
                del_keys.append(key)
            elif var_elem.is_constant():
                self._const_elems[key], _ = var_elem.popitem()
                del_keys.append(key)

        # delete elements that are no longer varying
        for key in del_keys:
            del self._varying_elems[key]


    def merge(self, other):
        """ Merge the summary 'other' into this summary """

        # length of self._tags before merge
        n_old = len(self._tags)

        tags_overlap = set(self._tags) & set(other._tags)
        overlaping = (len(tags_overlap) == 0)
        if overlaping:
            indices = [i for i, tag in enumerate(other._tags)
                       if tag not in tags_overlap]
            self._tags.extend([other._tags[i] for i in indices])
            n_new = n_old + len(indices)
        else:
            self._tags.extend(other._tags)
            n_new = n_old + len(other._tags)

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
            cmp_func = self._compare_rules.get(key)
            var_elem = VaryingElement(size=n_new, cmp_func=cmp_func)
            var_elem[other._const_elems[key]] = slice(n_old, n_new)
            self._varying_elems[key] = var_elem

        # keys that are only in other.varying_elems
        for key in other_vary:
            cmp_func = self._compare_rules.get(key)
            var_elem = VaryingElement(size=n_old, cmp_func=cmp_func)
            var_elem.extend(other._varying_elems[key])
            self._varying_elems[key] = var_elem

        # keys that are only in self.const_elems
        for key in this_const:
            cmp_func = self._compare_rules.get(key)
            var_elem = VaryingElement(size=n_new, cmp_func=cmp_func)
            var_elem[self._const_elems[key]] = slice(0, n_old)
            self._varying_elems[key] = var_elem
            del self._const_elems[key]

        # keys that are only in self.varying_elems
        for key in this_vary:
            self._varying_elems[key].resize(n_new)

        # intersection of varying elements
        for key in this_vary_other_vary:
            other_elem = other._varying_elems[key]
            if overlaping:
                other_elem = other_elem.subset(indices)
            self._varying_elems[key].extend(other_elem)

        # intersection of constant elements
        for key in this_const_other_const:
            cmp_func = self._compare_rules.get(key)
            cmp_rep = operator.eq if cmp_func is None else cmp_func
            if not cmp_rep(self._const_elems[key], other._const_elems[key]):
                var_elem = VaryingElement(size=n_new, cmp_func=cmp_func)
                var_elem[self._const_elems[key]] = slice(0, n_old)
                var_elem[other._const_elems[key]] = slice(n_old, n_new)
                self._varying_elems[key] = var_elem
                del self._const_elems[key]

        # intersection of self.varying_elems and other.const_elems
        for key in this_vary_other_const:
            var_elem = self._varying_elems[key]
            var_elem.resize(n_new)
            var_elem[other._const_elems[key]] = slice(n_old, n_new)

        # intersection of self.const_elems and other.varying_elems
        for key in this_const_other_vary:
            cmp_func = self._compare_rules.get(key)
            if overlaping:
                other_elem = other._varying_elems[key].subset(indices)
                if other_elem.is_constant():
                    other_val = iter(other_elem).next()
                    cmp_rep = operator.eq if cmp_func is None else cmp_func
                    if cmp_rep(self._const_elems[key], other_val):
                        # still constant, continue
                        continue
            else:
                other_elem = other._varying_elems[key]

            var_elem = VaryingElement(size=n_old, cmp_func=cmp_func)
            var_elem[self._const_elems[key]] = slice(0, n_old)
            var_elem.extend(other_elem)
            self._varying_elems[key] = var_elem
            del self._const_elems[key]


    def subset(self, indices):
        result = ElementSummary(compare_rules=self._compare_rules)

        # return empty ElementSummary if there are no indices to fill it with
        if len(indices) == 0:
            return result

        result._tags = [self._tags[i] for i in indices]
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
                result._const_elems[key] = var_elem.which(idx)

        return result


    def split(self, key, defualt=None):
        """Generator that splits the summary into sub-summaries that share the
        same value for the given key.
        """
        if key in self._const_elems:
            yield self._const_elems[key], deepcopy(self)
        elif key not in self._varying_elems:
            yield defualt, deepcopy(self)
        else:
            pvt_elem = self._varying_elems[key]
            filled = bitarray(self.size)
            filled.setall(0)
            for pvt_key, pvt_val in six.iteritems(pvt_elem):
                if isinstance(pvt_val, bitarray):
                    filled |= pvt_val
                    indices = [i for i, b in enumerate(pvt_val) if b]
                else:
                    filled[pvt_val] = 1
                    indices = [pvt_val]
                yield pvt_key, self.subset(indices)
            # yield sub-summary for unfilled indices
            if not all(filled):
                indices = [i for i, b in enumerate(pvt_val) if not b]
                yield defualt, self.subset(indices)


    def group(self, keys, defualt=None):
        """Generator that splits the summary into sub-summaries that share the
        same values for the given set of keys."""
        if len(keys) == 1:
            for val, summary in self.split(keys[0], defualt):
                yield (val,), summary
        else:
            for val, summary in self.split(keys[0], defualt):
                for grp_vals, grp_summary in summary.group(keys[1:], defualt):
                    yield (val,) + grp_vals, grp_summary


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
            # assure that the length is the same
            if indices.length() != self.size:
                raise ValueError
            self._setitem_bitarray(key, indices)
        elif isinstance(indices, Integral):
            # assure that the index is with in the range
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
        # This function is unsafe because the caller must assure that the
        # passed in index will not cause the constraint that each index is only
        # associated with one value be broken. This is used when it is known
        # that the passed in index is not currently associated with any value
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


    def _unsafe_assign_bitarray(self, repr_key, barray):
        # this associates the indices in 'barray' with the value 'repr_key'
        # without performing the work of removing any previous association that
        # the indices in 'barray'. This function is unsafe because the caller
        # must assure that the passed in indices will not cause the constraint
        # that each index is only associated with one value be broken. This is
        # used when it is known that the passed in indices are not currently
        # associated with any value
        old_val = super(VaryingElement, self).get(repr_key)
        if old_val is None:
            self._unsafe_set(repr_key, barray)
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


    def _insert_empty_array(self, key):
        new_array = bitarray(self._size)
        new_array.setall(0)
        self._unsafe_set(key, new_array)


    def _convert_to_array(self, key):
        prev_val = super(VaryingElement, self).get(key)
        if prev_val is None:
            self._insert_empty_array(key)
        elif isinstance(prev_val, (int, long)):
            self._insert_empty_array(key)
            super(VaryingElement, self).__getitem__(key)[prev_val] = 1
        return super(VaryingElement, self).get(key)

    #----------------------------------------------------------------------
    # logic functions
    #----------------------------------------------------------------------

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


    #----------------------------------------------------------------------
    # logic functions
    #----------------------------------------------------------------------

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
        for idx, val in enumerate(inv):
            if val == unfilled_key:
                continue
            elif val in out:
                out._convert_to_array(val)
                out[val][idx] = 1
            else:
                out[val] = idx
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


    def extract(self, out2in):
        check_array = iterable_to_bitarray(out2in, self.size)
        if len(out2in) != check_array.count(1):
            raise ValueError
        result = self.subset(extraced2contained)
        check_array.invert()
        new2old = [i for i, b in enumerate(check_array) if b]
        self.reduce(new2old)
        return results


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


    # def __getitem__(self, key):
    #     """Get the positional indices for the key"""
    #     val = super(VaryingElement, self).__getitem__(key)
    #     if isinstance(val, bitarray):
    #         return [idx for idx, val in emuerate(val)]
    #     else:
    #         return [val]
    #
    #
    def where(self, key):
        """Get the positional indices for the key"""
        val = self.get(key)
        if val is None:
            return list()
        if isinstance(val, bitarray):
            return [i for i, b in enumerate(val) if b]
        return [val]


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


    def __str__(self):
        zeros = ['0'] * self._size
        for key, val in six.iteritems(self):
            if isinstance(val, bitarray):
                print "%s : %s" % (val.to01(), key)
            else:
                zeros[val] = '1'
                "%s : %s" % (''.join(zeros), key)
                zeros[val] = '0'


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
