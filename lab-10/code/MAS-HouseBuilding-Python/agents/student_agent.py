from typing import List, Dict, Any

from agents import HouseOwnerAgent, CompanyAgent
from communication import NegotiationMessage


class MyACMEAgent(HouseOwnerAgent):
    """
    ACME (house owner / initiator) strategy:

    Auction phase — reversed Dutch (ascending):
      Round 0: 60% of budget  → low opening to save money
      Round 1: 80% of budget  → raise if no bids
      Round 2: 100% of budget → last chance

    Negotiation phase — monotonic concession (initiator must STRICTLY INCREASE):
      Start at 60% of the auction-winning price and work back up to 100%.
      This fulfils the protocol while trying to close below the auction price.
    """

    def __init__(self, role: str, budget_list: List[Dict[str, Any]]):
        super(MyACMEAgent, self).__init__(role, budget_list)
        # Price at which each item's auction succeeded
        self._auction_prices: Dict[str, float] = {}
        # Offers made per (item, partner) to enforce strict increase
        self._negotiation_offers: Dict[tuple, List[float]] = {}

    def propose_item_budget(self, auction_item: str, auction_round: int) -> float:
        budget = self.budget_dict[auction_item]
        round_factors = [0.6, 0.8, 1.0]
        return budget * round_factors[auction_round]

    def notify_auction_round_result(self, auction_item: str, auction_round: int,
                                    responding_agents: List[str]) -> None:
        if responding_agents:
            budget = self.budget_dict[auction_item]
            round_factors = [0.6, 0.8, 1.0]
            self._auction_prices[auction_item] = budget * round_factors[auction_round]

    def provide_negotiation_offer(self, negotiation_item: str, partner_agent: str,
                                  negotiation_round: int) -> float:
        # Ceiling is the price that attracted bids; fall back to full budget if unknown
        ceiling = self._auction_prices.get(negotiation_item, self.budget_dict[negotiation_item])
        key = (negotiation_item, partner_agent)

        if key not in self._negotiation_offers:
            self._negotiation_offers[key] = []

        round_factors = [0.6, 0.8, 1.0]
        
        # --- FIX: Guard against index out of bounds ---
        safe_round_idx = min(negotiation_round, len(round_factors) - 1)
        offer = ceiling * round_factors[safe_round_idx]
        # -----------------------------------------------

        # Guarantee strictly increasing offers (protocol requirement for initiator)
        if self._negotiation_offers[key]:
            offer = max(offer, self._negotiation_offers[key][-1] + 1)

        self._negotiation_offers[key].append(offer)
        return offer
    
    def notify_partner_response(self, response_msg: NegotiationMessage) -> None:
        pass

    def notify_negotiation_winner(self, negotiation_item: str, winning_agent: str,
                                  winning_offer: float) -> None:
        pass


class MyCompanyAgent(CompanyAgent):
    """
    Company (responder) strategy:

    Auction phase:
      Bid only when the announced budget covers the actual cost — no loss accepted.

    Negotiation phase — monotonic concession (responder must STRICTLY DECREASE):
      Ask prices decrease each round from a 20% profit margin down to ~3%:
        Round 0: cost x 1.20
        Round 1: cost x 1.10
        Round 2: cost x 1.03
      The response is also capped one unit below the previous response (protocol
      correctness) and floored at cost (never accept a loss).
      Agreement is triggered whenever ACME's rising offer meets or exceeds the
      company's ask.
    """

    def __init__(self, role: str, specialties: List[Dict[str, Any]]):
        super(MyCompanyAgent, self).__init__(role, specialties)
        # Own responses per conversation to enforce strict decrease
        self._negotiation_responses: Dict[str, List[float]] = {}

    def decide_bid(self, auction_item: str, auction_round: int, item_budget: float) -> bool:
        if not self.has_specialty(auction_item):
            return False
        return item_budget >= self.specialties[auction_item]

    def notify_won_auction(self, auction_item: str, auction_round: int, num_selected: int) -> None:
        pass

    def respond_to_offer(self, initiator_msg: NegotiationMessage) -> float:
        item = initiator_msg.negotiation_item
        round_num = initiator_msg.round
        conv_id = initiator_msg.conversation_id
        my_cost = self.specialties[item]

        if conv_id not in self._negotiation_responses:
            self._negotiation_responses[conv_id] = []

        prev_responses = self._negotiation_responses[conv_id]

        # Profit margins decrease each round: 20% -> 10% -> 3%
        profit_margins = [1.20, 1.10, 1.00]
        idx = min(round_num, len(profit_margins) - 1)
        target = my_cost * profit_margins[idx]

        # Must be strictly below the previous response (protocol requirement for responder)
        if prev_responses:
            target = min(target, prev_responses[-1] - 1)

        # Absolute floor: never accept below cost
        target = max(target, my_cost)

        self._negotiation_responses[conv_id].append(target)
        return target

    def notify_contract_assigned(self, construction_item: str, price: float) -> None:
        pass

    def notify_negotiation_lost(self, construction_item: str) -> None:
        pass
