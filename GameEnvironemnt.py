from Actions import Action
from Cards import Deck

class GameEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.deck = Deck().shuffle()
        self.player_hand = [self.deck.pop(), self.deck.pop()]
        self.dealer_card = self.deck.pop()
        self.reward = 0
        return self._get_observation()

    def _get_observation(self):
        return tuple(card.get_card_value() for card in self.player_hand)

    def step(self, action: Action):

        if self.player_hand[0].get_card_value() == self.player_hand[1].get_card_value() and self.player_hand[0].get_card_value():
            self.reward = 2
            return self._get_observation(), self.reward, {"reason: ": "player pair"}

        total = sum(card.get_card_value() for card in self.player_hand + [self.dealer_card])
        is_even = total % 2 == 0

        dealer_val = self.dealer_card.get_card_value()
        player_vals = [card.get_card_value() for card in self.player_hand]

        if dealer_val in player_vals:
            self.reward = -1
            return self._get_observation(), self.reward, {"reason: ": "dealer pair"}

        if (action == Action.GUESS_EVEN and is_even) or (action == Action.GUESS_ODD and not is_even):
            self.reward = 1
            return self._get_observation(), self.reward, {"reason: ": "correct guess"}
        else:
            self.reward = -1
            return self._get_observation(), self.reward, {"reason: ": "incorrect guess"}
