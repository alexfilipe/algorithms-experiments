"""
Author: Álex Santos <alex@afcbs.me>
Date: 6 Dec 2020

Experiment 1
============

Given an a h x w array A, where each entry is a character. Design a function
paint(A, i, j, c) that "paints" an image represented by an array A
(substitutes the color for c for each A[i0][j0] starting from A[i][j] for pixels
with the same color).

Example:
>>> A
___________________
___________________
__XXXXXXXX_________
__XXXXXXXXX________
__XXXXXXXXXX_______
XXXXXXXXXXXXXXXXXXX
__XXXXXXXXXX_______
__XXXXXXXXX________
__XXXXXXXX_________
___________________
___________________

>>> paint(A, 0, 0, '*')
*******************
*******************
**XXXXXXXX*********
**XXXXXXXXX********
**XXXXXXXXXX*******
XXXXXXXXXXXXXXXXXXX
__XXXXXXXXXX_______
__XXXXXXXXX________
__XXXXXXXX_________
___________________
___________________

>>> paint(A, 6, 0, '*')
___________________
___________________
__********_________
__*********________
__**********_______
*******************
__**********_______
__*********________
__********_________
___________________
___________________
"""

def file_to_array(filepath):
    with open(filepath) as f:
        array = [list(line[:-1]) for line in f]
    return array

def print_array(array):
    for row in array:
        print("".join(str(s) for s in row))

def successors(array, visited, i, j):
    """Return the successors for the array node i, j."""
    h = len(array)
    w = len(array[0])
    nodes = set()
    if i-1 >= 0 and not visited[i-1][j]:
        nodes.add((i-1, j))
    if j-1 >= 0 and not visited[i][j-1]:
        nodes.add((i, j-1))
    if i+1 < h and not visited[i+1][j]:
        nodes.add((i+1, j))
    if j+1 < w and not visited[i][j+1]:
        nodes.add((i, j+1))
    return nodes

def visit(array, visited, i, j, symbol, color):
    array[i][j] = color
    visited[i][j] = True
    next_nodes = successors(array, visited, i, j)
    for i0, j0 in next_nodes:
        if array[i0][j0] == symbol:
            visit(array, visited, i0, j0, symbol, color)

def paint(array, i, j, color):
    """Paints the picture starting from the position (i, j)."""

    # Defines if the node was visited or not
    visited = [[False for _ in r] for r in array]
    visited[i][j] = True

    # Symbol to substitute for
    symbol = array[i][j]

    # Do breadth first search with graph adjacents and stop when tiles are different
    visit(array, visited, i, j, symbol, color)

    return array


if __name__ == '__main__':
    array = file_to_array("/Users/alex/projs/art1.txt")
    print_array(array)
    paint(array, 5, 0, '*')
    print()
    print_array(array)
