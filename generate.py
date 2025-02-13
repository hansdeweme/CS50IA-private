import sys
from collections import deque
from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generator.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [[None for _ in range(self.crossword.width)] 
                   for _ in range(self.crossword.height)]
        for variable, word in assignment.items():
            for k in range(len(word)):
                i = variable.i + (k if variable.direction == Variable.DOWN else 0)
                j = variable.j + (k if variable.direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                print(letters[i][j] or " " if self.crossword.structure[i][j] else "█", end="")
            print()

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        arcs = [(x, y) for x in self.domains for y in self.crossword.neighbors(x)]
        if not self.ac3(arcs):
            return None
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Ensure each variable's domain only contains words of the correct length.
        """
        for var in self.domains:
            self.domains[var] = {word for word in self.domains[var] if len(word) == var.length}

    def revise(self, x, y):
        """
        Make `x` arc-consistent with `y` by removing values from `x`'s domain
        that have no corresponding value in `y`'s domain.
        """
        if not self.crossword.overlaps[x, y]:
            return False

        i, j = self.crossword.overlaps[x, y]
        letters_y = {w[j] for w in self.domains[y]}  # Use set for O(1) lookup
        revised = False

        words_to_remove = {word for word in self.domains[x] if word[i] not in letters_y}
        if words_to_remove:
            self.domains[x] -= words_to_remove
            revised = True
        
        return revised
        
    def ac3(self, arcs=None):
        """
        Enforce arc consistency using the AC-3 algorithm.
        """
        if arcs == []:  # If explicitly given an empty list, do nothing and return True
            return True
        
        queue = deque(arcs if arcs is not None else [(x, y) 
                                                     for x in self.domains for y in self.crossword.neighbors(x)])
        
        while queue:
            x, y = queue.popleft()
            if self.revise(x, y):
                if not self.domains[x]:
                    return False
                for z in self.crossword.neighbors(x) - {y}:
                    queue.append((z, x))
    
        return all(len(self.domains[var]) > 0 for var in self.domains)
    
    def assignment_complete(self, assignment):
        """
        Check if every crossword variable has a value assigned.
        """
        return set(assignment.keys()) == self.crossword.variables

    def consistent(self, assignment):
        """
        Check if the assignment is consistent.
        """
        words_used = set(assignment.values())
        if len(words_used) != len(assignment):
            return False
        
        for var, word in assignment.items():
            if len(word) != var.length:
                return False
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    i, j = self.crossword.overlaps[var, neighbor]
                    if word[i] != assignment[neighbor][j]:
                        return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Order domain values by least-constraining value heuristic.
        """
        return sorted(self.domains[var], key=lambda word: sum(
            word[self.crossword.overlaps[var, neighbor][0]] != neighbor_word[self.crossword.overlaps[var, neighbor][1]]
            for neighbor in self.crossword.neighbors(var) - assignment.keys()
            for neighbor_word in self.domains[neighbor]
        ))

    def select_unassigned_variable(self, assignment):
        """
        Select an unassigned variable using MRV and degree heuristic.
        """
        unassigned = [var for var in self.crossword.variables if var not in assignment]
        return min(unassigned, key=lambda var: (len(self.domains[var]), -len(self.crossword.neighbors(var))))

    def backtrack(self, assignment):
        """
        Backtracking search to find a solution.
        """
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            assignment[var] = value
            if self.consistent(assignment):
                result = self.backtrack(assignment)
                if result:
                    return result
            del assignment[var]
        return None


def main():
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    structure, words = sys.argv[1], sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
