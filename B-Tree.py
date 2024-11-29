import math, sys
import pandas as pd
import numpy as np
from tqdm import tqdm


class Node:
    def __init__(self, leaf=False):
        self.keys = []         # stores keys in the node
        self.children = []     # stores references to child nodes
        self.leaf = leaf       # boolean flag to check if node is a leaf

# BTree class representing the B-Tree itself
class BTree:
    def __init__(self, t):
        self.root = Node(True) # initializes root as a leaf node
        self.t = t             # the minimum degree of the B-tre
    
    # Splitting a child node when it becomes full
    def split_child(self, x, i):
        y = x.children[i]              # child node to split
        z = Node(y.leaf)               # new node to hold half of y's keys
        x.children.insert(i + 1, z)    # insert new node as a child of x
        x.keys.insert(i, y.keys[self.t - 1])  # move median key to parent node
        z.keys = y.keys[self.t:(2 * self.t - 1)]  # move second half of y's keys to z
        y.keys = y.keys[:self.t - 1]   # retain the first half of y's keys in y

        if not y.leaf:                 # if y is not a leaf, move its children as well
            z.children = y.children[self.t:(2 * self.t)]
            y.children = y.children[:self.t]

    # Insert a new key into the B-tree
    def insert(self, k):
        r = self.root
        if len(r.keys) == (2 * self.t - 1):  # check if root is full
            s = Node(False)                  # create a new root node
            self.root = s
            s.children.append(r)             # make the old root a child of the new root
            self.split_child(s, 0)           # split the old root
            self.insert_key(s, k)            # insert the key into the non-full new root
        else:
            self.insert_key(r, k)            # insert the key into the non-full root

    # Insert a key into a non-full node
    def insert_key(self, x, k):
        i = len(x.keys) - 1                  # start from the rightmost key
        if x.leaf:                           # if x is a leaf node
            x.keys.append(None)              # add space for the new key
            while i >= 0 and k[0] < x.keys[i][0]:
                x.keys[i + 1] = x.keys[i]    # shift keys to the right to make space
                i -= 1
            x.keys[i + 1] = k                # insert the new key at the correct position
        else:                                # if x is an internal node
            while i >= 0 and k[0] < x.keys[i][0]:
                i -= 1
            i += 1
            if len(x.children[i].keys) == (2 * self.t - 1):  # if child is full, split it
                self.split_child(x, i)
                if k[0] > x.keys[i][0]:
                    i += 1
            self.insert_key(x.children[i], k)  # recursively insert into the child

    # Search for a key in the B-tree
    def search_key(self, x, k):
        i = 0
        while i < len(x.keys) and k > x.keys[i][0]:  # find the first key greater than k
            i += 1
        if i < len(x.keys) and k == x.keys[i][0]:    # if key found, return node and index
            return x, i
        elif x.leaf:  # if reached a leaf and not found, return None
            return None
        else:         # recursively search in the appropriate child
            return self.search_key(x.children[i], k)

    # Delete a key from the B-tree
    def delete(self, k):
        self._delete(self.root, k)
        if len(self.root.keys) == 0 and not self.root.leaf:  # if root becomes empty
            self.root = self.root.children[0]                # shrink the tree

            # Helper method to delete a key
    def _delete(self, x, k):
        i = 0
        while i < len(x.keys) and k > x.keys[i][0]:
            i += 1

        if i < len(x.keys) and k == x.keys[i][0]:  # if key is found
            if x.leaf:
                self.delete_leaf_node(x, i)        # delete from leaf node
            else:
                self.delete_internal_node(x, i)    # delete from internal node
        elif not x.leaf:                           # if not found and not leaf
            flag = (i == len(x.keys))
            if len(x.children[i].keys) < self.t:   # ensure child has enough keys
                self._fill(x, i)
            if flag and i > len(x.keys):
                self._delete(x.children[i - 1], k)  # recurse on the left child
            else:
                self._delete(x.children[i], k)      # recurse on the right child

    # Delete a key from a leaf node
    def delete_leaf_node(self, x, i):
        x.keys.pop(i)  # simply remove the key from the leaf

    # Delete a key from an internal node
    def delete_internal_node(self, x, i):
        k = x.keys[i]
        if len(x.children[i].keys) >= self.t:              # if left child has enough keys
            pred = self.get_predecessor(x, i)
            x.keys[i] = pred
            self._delete(x.children[i], pred[0])
        elif len(x.children[i + 1].keys) >= self.t:        # if right child has enough keys
            succ = self.get_successor(x, i)
            x.keys[i] = succ
            self._delete(x.children[i + 1], succ[0])
        else:                                              # merge both children
            self.merge(x, i)
            self._delete(x.children[i], k[0])
    
    # Get predecessor of a key
    def get_predecessor(self, x, i):
        cur = x.children[i]
        while not cur.leaf:
            cur = cur.children[len(cur.keys)]  # move to the rightmost child
        return cur.keys[-1]

    # Get successor of a key
    def get_successor(self, x, i):
        cur = x.children[i + 1]
        while not cur.leaf:
            cur = cur.children[0]  # move to the leftmost child
        return cur.keys[0]

    # Merge two child nodes
    def merge(self, x, i):
        c1 = x.children[i]
        c2 = x.children[i + 1]
        c1.keys.append(x.keys[i])          # move key down from parent
        c1.keys.extend(c2.keys)            # append keys of right sibling
        if not c1.leaf:
            c1.children.extend(c2.children) # append children of right sibling
        x.keys.pop(i)                       # remove key from parent
        x.children.pop(i + 1)               # remove right sibling

    # Fill a child node that has less than minimum keys
    def _fill(self, x, i):
        if i != 0 and len(x.children[i - 1].keys) >= self.t:  # borrow from previous sibling
            self.borrow_from_prev(x, i)
        elif i != len(x.children) - 1 and len(x.children[i + 1].keys) >= self.t:  # borrow from next sibling
            self.borrow_from_next(x, i)
        else:
            if i != len(x.children) - 1:
                self.merge(x, i)   # merge with next sibling
            else:
                self.merge(x, i - 1)  # merge with previous sibling

    # Borrow a key from the previous sibling
    def borrow_from_prev(self, x, i):
        child = x.children[i]
        sibling = x.children[i - 1]
        child.keys.insert(0, x.keys[i - 1])    # move key from parent to child
        if not child.leaf:
            child.children.insert(0, sibling.children.pop())  # move child from sibling
        x.keys[i - 1] = sibling.keys.pop()     # move key from sibling to parent

    # Borrow a key from the next sibling
    def borrow_from_next(self, x, i):
        child = x.children[i]
        sibling = x.children[i + 1]
        child.keys.append(x.keys[i])           # move key from parent to child
        if not child.leaf:
            child.children.append(sibling.children.pop(0))  # move child from sibling
        x.keys[i] = sibling.keys.pop(0)        # move key from sibling to parent

    # Traverse the B-tree and count keys at each level
    def traverse_key(self, x, level=0, level_counts=None):
        if level_counts is None:
            level_counts = {}
        if x:
            level_counts[level] = level_counts.get(level, 0) + len(x.keys)  # count keys at this level
            for child in x.children:
                self.traverse_key(child, level + 1, level_counts)  # recursively count in children
        return level_counts


def get_file():
    file_name = input("Enter the file name you want to insert or delete ▷ (e.g., insert1 or delete1_50 or delete1_90 or ...) ")
    while True:
        try:
            file = pd.read_csv('inputs/' + file_name + '.csv', delimiter='\t', names=['key', 'value'])
            return file
        except FileNotFoundError:
            print("File does not exist.")
            file_name = input("Enter the file name again. ▷ ")

def insertion_test(B, file):
    file_key = file['key']
    file_value = file['value']

    print('===============================')
    print('[ Insertion start ]')

    for i in tqdm(range(len(file_key))):  
        B.insert([file_key[i], file_value[i]])

    print('[ Insertion complete ]')
    print('===============================\n')

    return B

def deletion_test(B, file):
    delete_key = file['key']

    print('===============================')
    print('[ Deletion start ]')

    for i in tqdm(range(len(delete_key))):
        B.delete(delete_key[i])

    print('[ Deletion complete ]')
    print('===============================\n')

    return B

def print_statistic(B):
    print('===============================')
    print('[ Print statistic of tree ]')

    level_counts = B.traverse_key(B.root)

    for level, counts in level_counts.items():
        if level == 0:
            print(f'Level {level} (root): Key Count = {counts}')
        else:
            print(f'Level {level}: Key Count = {counts}')

    total_keys = sum(counts for counts in level_counts.values())
    print(f'Total number of keys across all levels: {total_keys}')
    print('===============================\n')

def main():
    while True:
        try:
            num = int(input("1. Insertion 2. Deletion 3. Statistic 4. End ▶  "))

            if num == 1:
                t = 3  
                B = BTree(t)  
                insert_file = get_file()
                B = insertion_test(B, insert_file)

            elif num == 2:
                delete_file = get_file()
                B = deletion_test(B, delete_file)

            elif num == 3:
                print_statistic(B)

            elif num == 4:
                print("Exiting the program. Goodbye!")
                break  

            else:
                print("Invalid input. Please enter 1, 2, 3, or 4.")

        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == '__main__':
    main()
