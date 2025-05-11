import random

# Constants
CARD_VALUES = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
    '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 10, 'Q': 10, 'K': 10, 'A': 11
}
DECK = list(CARD_VALUES.keys()) * 4

# Helper functions
def draw_card(deck):
    return deck.pop(random.randint(0, len(deck) - 1))

def hand_value(hand):
    total = sum(CARD_VALUES[card] for card in hand)
    # Adjust for Aces
    aces = hand.count('A')
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total

def is_bust(hand):
    return hand_value(hand) > 21

def dealer_play(deck, dealer_hand):
    while hand_value(dealer_hand) < 17:
        dealer_hand.append(draw_card(deck))
    return dealer_hand

# Core simulation function
def play_blackjack(action_fn):
    """
    Simulates one game of blackjack.
    `action_fn(player_total, dealer_upcard, can_double)` returns action: "hit", "stand", or "double"
    Returns result: "win", "loss", "draw"
    """
    deck = DECK.copy()
    random.shuffle(deck)

    player_hand = [draw_card(deck), draw_card(deck)]
    dealer_hand = [draw_card(deck), draw_card(deck)]

    first_turn = True
    can_double = True

    while True:
        state = (hand_value(player_hand), CARD_VALUES[dealer_hand[0]], can_double)
        action = action_fn(*state)

        if action == "stand":
            break
        elif action == "hit":
            player_hand.append(draw_card(deck))
            if is_bust(player_hand):
                return "loss"
        elif action == "double" and can_double:
            player_hand.append(draw_card(deck))
            if is_bust(player_hand):
                return "loss"
            break
        else:
            # Invalid action (e.g., "double" after first turn), treat as "stand"
            break

        first_turn = False
        can_double = False

    dealer_hand = dealer_play(deck, dealer_hand)

    player_score = hand_value(player_hand)
    dealer_score = hand_value(dealer_hand)

    if is_bust(dealer_hand):
        return "win"
    if player_score > dealer_score:
        return "win"
    if player_score < dealer_score:
        return "loss"
    return "draw"
