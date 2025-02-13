import random
import time

class Nim():
    def __init__(self, initial=[1, 3, 5, 7]):
        """
        Initialize game board.
        """
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        """
        Return all available actions (pile index, count) for the given state.
        """
        return {(i, j) for i, pile in enumerate(piles) for j in range(1, pile + 1)}

    @classmethod
    def other_player(cls, player):
        return 0 if player == 1 else 1

    def switch_player(self):
        self.player = Nim.other_player(self.player)

    def move(self, action):
        """
        Perform the given action and switch players.
        """
        pile, count = action
        if self.winner is not None:
            raise Exception("Game already won")
        if pile < 0 or pile >= len(self.piles) or count < 1 or count > self.piles[pile]:
            raise Exception("Invalid move")
        
        self.piles[pile] -= count
        self.switch_player()

        if all(p == 0 for p in self.piles):
            self.winner = self.player


class NimAI():
    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        Initialize AI with an empty Q-learning dictionary,
        an adaptive learning rate (alpha), and an epsilon rate.
        """
        self.q = dict()
        self.alpha = alpha  # Initial learning rate
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        """ Return Q-value for the given (state, action) pair, default to 0. """
        return self.q.get((tuple(state), action), 0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Apply Q-learning formula to update the Q-value.
        """
        new_q = old_q + self.alpha * (reward + future_rewards - old_q)
        self.q[(tuple(state), action)] = new_q

    def best_future_reward(self, state):
        """
        Return the highest Q-value for any action in the given state.
        """
        actions = Nim.available_actions(state)
        return max((self.get_q_value(state, action) for action in actions), default=0)

    def choose_action(self, state, epsilon=True):
        """
        Choose an action based on epsilon-greedy strategy.
        """
        actions = list(Nim.available_actions(state))
        if not actions:
            return None
        
        q_values = {action: self.get_q_value(state, action) for action in actions}
        best_action = max(q_values, key=q_values.get)
        
        if epsilon and random.random() < self.epsilon:
            return random.choice(actions)
        return best_action

    def update(self, old_state, action, new_state, reward):
        """
        Update Q-learning model using the Q-learning update rule.
        """
        old_q = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old_q, reward, best_future)
        
        # Adaptive learning rate (decays over time)
        self.alpha = max(0.1, self.alpha * 0.99)


def train(n):
    """
    Train an AI by playing `n` games against itself.
    """
    player = NimAI()

    for i in range(n):
        print(f"Playing training game {i + 1}")
        game = Nim()
        last = {0: {"state": None, "action": None}, 1: {"state": None, "action": None}}

        while True:
            state = game.piles.copy()
            action = player.choose_action(game.piles)

            last[game.player]["state"] = state
            last[game.player]["action"] = action

            game.move(action)
            new_state = game.piles.copy()

            if game.winner is not None:
                player.update(state, action, new_state, -1)
                player.update(last[game.player]["state"], last[game.player]["action"], new_state, 1)
                break
            elif last[game.player]["state"] is not None:
                player.update(last[game.player]["state"], last[game.player]["action"], new_state, 0)

    print("Done training")
    return player


def play(ai, human_player=None):
    """
    Play human game against the AI.
    """
    if human_player is None:
        human_player = random.randint(0, 1)

    game = Nim()

    while True:
        print("\nPiles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")

        available_actions = Nim.available_actions(game.piles)
        time.sleep(1)

        if game.player == human_player:
            print("Your Turn")
            while True:
                pile = int(input("Choose Pile: "))
                count = int(input("Choose Count: "))
                if (pile, count) in available_actions:
                    break
                print("Invalid move, try again.")
        else:
            print("AI's Turn")
            pile, count = ai.choose_action(game.piles, epsilon=False)
            print(f"AI chose to take {count} from pile {pile}.")

        game.move((pile, count))

        if game.winner is not None:
            print("\nGAME OVER")
            print("Winner is", "Human" if game.winner == human_player else "AI")
            return
