
import random
import copy


class Minesweeper:
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence:
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if self.count == len(self.cells):
            return {cell for cell in self.cells}  # Use curly braces to return a set
        return set()  # Explicitly return an empty set if no conclusion is possible

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return {cell for cell in self.cells}  # Use curly braces to return a set
        return set()  # Explicitly return an empty set if no conclusion is possible
    
    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells and self.count > 0:
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells and self.count > 0:
            self.cells.remove(cell)


class MinesweeperAI:
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def _get_all_cells(self):
        """
        Returns a set of all cells on the board
        """
        all_cells = set()
        for i in range(self.height):
            for j in range(self.width):
                all_cells.add((i, j))
        return all_cells

    def _get_surrounding_cells(self, cell):
        """
        Returns a set of all surrounding cells of a cell, ensuring valid indices.
        """
        surr_cells = set()
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):
                if 0 <= i < self.height and 0 <= j < self.width and (i, j) != cell:
                    surr_cells.add((i, j))
        return surr_cells
    
    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        # 1) Mark the cell as a move made
        self.moves_made.add(cell)

        # 2) Mark the cell as safe
        self.mark_safe(cell)

        # 3) Get surrounding cells, excluding known safes and mines
        surr_cells = self._get_surrounding_cells(cell)
        new_cells = set()
        for c in surr_cells:
            if c in self.mines:
                count -= 1  # Adjust the count for known mines
            elif c not in self.safes:
                new_cells.add(c)

        # Create and add a new sentence
        new_sentence = Sentence(new_cells, count)
        if new_sentence.cells:
            self.knowledge.append(new_sentence)

        # 4) Update knowledge with safes and mines
        self.update_knowledge()
        
    def update_knowledge(self):
        """
        Updates the knowledge base by:
        1. Marking cells as safe or mines based on sentences.
        2. Inferring new sentences from existing knowledge.
        """
        changes = True  # Continue updating until no more changes
        while changes:
            changes = False

            safe_cells = set()
            mine_cells = set()

            # Identify known safes and mines from sentences
            for sentence in self.knowledge:
                safe_cells.update(sentence.known_safes())
                mine_cells.update(sentence.known_mines())

            # Mark all discovered safe cells
            for cell in safe_cells:
                if cell not in self.safes:
                    changes = True
                    self.mark_safe(cell)

            # Mark all discovered mine cells
            for cell in mine_cells:
                if cell not in self.mines:
                    changes = True
                    self.mark_mine(cell)

            # Remove empty sentences
            self.knowledge = [sentence for sentence in self.knowledge if sentence.cells]

            # Infer new sentences from intersections of existing ones
            for sentence1 in self.knowledge:
                for sentence2 in self.knowledge:
                    if sentence1 == sentence2:
                        continue

                    intersection = sentence1.cells.intersection(sentence2.cells)
                    if intersection:
                        # Infer the non-overlapping parts
                        diff1 = sentence1.cells - intersection
                        diff2 = sentence2.cells - intersection
                        new_count1 = sentence1.count - len(intersection)
                        new_count2 = sentence2.count - len(intersection)

                        # Add new sentences if meaningful
                        if new_count1 >= 0 and diff1:
                            inferred_sentence = Sentence(diff1, new_count1)
                            if inferred_sentence not in self.knowledge:
                                self.knowledge.append(inferred_sentence)
                                changes = True

                        if new_count2 >= 0 and diff2:
                            inferred_sentence = Sentence(diff2, new_count2)
                            if inferred_sentence not in self.knowledge:
                                self.knowledge.append(inferred_sentence)
                                changes = True


    def _update_sentence(self, sentence):
        """
        Updates a given sentence with the knowledge about mines and safes.
        """
        # work with a copy to keep the original sentence
        deepcopy_sentence = copy.deepcopy(sentence)
        if deepcopy_sentence.count > 0 and len(sentence.cells) > 0:

            # get rid of safes to identify mines
            for safe in self.safes:
                deepcopy_sentence.cells.remove(safe) if safe in deepcopy_sentence.cells else None
                if len(deepcopy_sentence.cells) == deepcopy_sentence.count and len(deepcopy_sentence.cells) > 0:
                    temp_set = set() # store mines temporarily in order to avoid KEYERROR
                    for cell in deepcopy_sentence.cells:
                        self.mines.add(cell)
                        temp_set.add(cell)
                    for cell in temp_set:
                        deepcopy_sentence.cells.remove(cell)
                        deepcopy_sentence.count -= 1

            # get rid of mines to find safes
            for mine in self.mines:
                if mine in deepcopy_sentence.cells:
                    deepcopy_sentence.cells.remove(mine)
                    deepcopy_sentence.count -= 1
                    if deepcopy_sentence.count == 0 and len(deepcopy_sentence.cells) > 0:
                        for cell in deepcopy_sentence.cells:
                            self.safes.add(cell)

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        cells = self._get_all_cells()
        save_moves = []
        for cell in cells:
            if cell not in self.moves_made and cell in self.safes:
                save_moves.append(cell)
        if len(save_moves) == 0:
            return None
        else:
            return random.choice(save_moves)

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        if self.make_safe_move() == None:
            # get all moves
            cells = self._get_all_cells()
            potential_moves = []
            for cell in cells:
                if cell not in self.moves_made and cell not in self.mines:
                    potential_moves.append(cell)
            if len(potential_moves) == 0:
                return None
            else:
                return random.choice(potential_moves)
