from typing import Callable, List

from commons import CommonsAgent, CommonsPerception
from communication import AgentAction


class StudentAgent(CommonsAgent):
    def __init__(self, agent_id):
        super(StudentAgent, self).__init__(agent_id)
        self.last_shares = {}       # track shares from the previous round
        self.round_number = 0       # internal round counter

    def specify_share(self, perception: CommonsPerception) -> float:
        n = perception.num_agents
        optimal_share = 1.0 / (2 * n)
        return optimal_share

    def negotiation_response(self, negotiation_round: int, perception: CommonsPerception,
                             utility_func: Callable[[float, float, List[float]], float]) -> AgentAction:
        n = perception.num_agents
        optimal_share = 1.0 / (2 * n)
        my_share = optimal_share

        if perception.resource_shares is None:
            return AgentAction(self.id, resource_share=my_share, no_action=True)

        current_shares = perception.resource_shares

        all_close = all(abs(s - optimal_share) < 1e-4 for s in current_shares.values())
        if all_close:
            return AgentAction(self.id, resource_share=my_share, no_action=True)

        dampening = 0.5
        consumption_adjustment = {}

        for agent_id, share in current_shares.items():
            if agent_id == self.id:
                continue
            diff = optimal_share - share  
            if abs(diff) > 1e-4:
                consumption_adjustment[agent_id] = diff * dampening

        projected_total = my_share
        for agent_id, share in current_shares.items():
            if agent_id == self.id:
                projected_total += 0 
            else:
                adjusted = share + consumption_adjustment.get(agent_id, 0)
                projected_total += max(0, adjusted)

        if projected_total >= 0.95:
            scale = 0.9 / projected_total
            for agent_id in consumption_adjustment:
                consumption_adjustment[agent_id] *= scale

        return AgentAction(self.id, resource_share=my_share,
                           consumption_adjustment=consumption_adjustment, no_action=False)

    def inform_round_finished(self, negotiation_round: int, perception: CommonsPerception):
        self.round_number += 1
        if perception.resource_shares:
            self.last_shares = dict(perception.resource_shares)