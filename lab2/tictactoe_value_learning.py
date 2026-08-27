"""
Reinforcement Learning Laboratory - Lab 2
Interactive Exploration of Tic-Tac-Toe using Reinforcement Learning
Name: Mahadevi E Malkhed
Roll No: BL.SC.P2CSE25009

Algorithm:
- Tabular state-value learning
- Epsilon-greedy exploration
- Two agents (X and O) trained by self-play
- Alpha = 0.5, epsilon = 0.05
- Training checkpoints: 100, 500, 1,000, 5,000, 10,000 games
"""

import random

WIN_LINES = (
    (0,1,2), (3,4,5), (6,7,8),
    (0,3,6), (1,4,7), (2,5,8),
    (0,4,8), (2,4,6)
)

def check_winner(board):
    for a, b, c in WIN_LINES:
        if board[a] != " " and board[a] == board[b] == board[c]:
            return board[a]
    if " " not in board:
        return "D"
    return None

def available_moves(board):
    return [i for i, cell in enumerate(board) if cell == " "]

def next_state(board, move, mark):
    new_board = list(board)
    new_board[move] = mark
    return tuple(new_board)

class ValueAgent:
    def __init__(self, mark, alpha=0.5, epsilon=0.05):
        self.mark = mark
        self.alpha = alpha
        self.epsilon = epsilon
        self.V = {}

    def value(self, state):
        if state not in self.V:
            result = check_winner(state)
            if result == self.mark:
                self.V[state] = 1.0
            elif result == "D":
                self.V[state] = 0.5
            elif result is not None:
                self.V[state] = 0.0
            else:
                self.V[state] = 0.5
        return self.V[state]

    def choose_action(self, board, training=True):
        moves = available_moves(board)

        if training and random.random() < self.epsilon:
            return random.choice(moves)

        values = []
        for move in moves:
            state = next_state(board, move, self.mark)
            values.append((self.value(state), move))

        best_value = max(value for value, _ in values)
        best_moves = [move for value, move in values
                      if abs(value - best_value) < 1e-12]
        return random.choice(best_moves)

    def update(self, state_history, outcome):
        if outcome == self.mark:
            target = 1.0
        elif outcome == "D":
            target = 0.5
        else:
            target = 0.0

        next_value = target
        for state in reversed(state_history):
            old_value = self.value(state)
            new_value = old_value + self.alpha * (next_value - old_value)
            self.V[state] = new_value
            next_value = new_value

def train_game(x_agent, o_agent):
    board = tuple(" " for _ in range(9))
    histories = {"X": [], "O": []}
    turn = "X"

    while True:
        agent = x_agent if turn == "X" else o_agent
        histories[turn].append(board)

        move = agent.choose_action(board, training=True)
        board = next_state(board, move, turn)

        result = check_winner(board)
        if result is not None:
            x_agent.update(histories["X"], result)
            o_agent.update(histories["O"], result)
            return result

        turn = "O" if turn == "X" else "X"

def train(episodes, seed=42):
    random.seed(seed)
    x_agent = ValueAgent("X")
    o_agent = ValueAgent("O")

    outcomes = {"X": 0, "O": 0, "D": 0}
    for _ in range(episodes):
        result = train_game(x_agent, o_agent)
        outcomes[result] += 1

    return x_agent, o_agent, outcomes

def play_against_random(agent, games=1000, seed=123):
    random.seed(seed)
    results = {"win": 0, "loss": 0, "draw": 0}

    for _ in range(games):
        board = tuple(" " for _ in range(9))
        turn = "X"

        while True:
            if turn == agent.mark:
                move = agent.choose_action(board, training=False)
            else:
                move = random.choice(available_moves(board))

            board = next_state(board, move, turn)
            result = check_winner(board)

            if result is not None:
                if result == agent.mark:
                    results["win"] += 1
                elif result == "D":
                    results["draw"] += 1
                else:
                    results["loss"] += 1
                break

            turn = "O" if turn == "X" else "X"

    return results

if __name__ == "__main__":
    checkpoints = [100, 500, 1000, 5000, 10000]

    print("Tic-Tac-Toe Reinforcement Learning")
    print("alpha = 0.5, epsilon = 0.05")
    print()

    for episodes in checkpoints:
        agent, opponent, training_outcomes = train(episodes)

        evaluation = play_against_random(
            agent,
            games=1000,
            seed=episodes
        )

        print(f"Training episodes: {episodes}")
        print(f"  Self-play outcomes: {training_outcomes}")
        print(f"  Evaluation vs random (1000 games): {evaluation}")
        print(f"  Win %:   {evaluation['win'] / 10:.1f}")
        print(f"  Draw %:  {evaluation['draw'] / 10:.1f}")
        print(f"  Loss %:  {evaluation['loss'] / 10:.1f}")
        print()
