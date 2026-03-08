from environment import *
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import time


class AgentDesire(ABC):
    """
    A structured sub-goal (DESIRE) used by the BDI loop.
    Students should select among available desires and build an intention for one desire at a time.

    Students are encouraged to create subclasses with their own logic for:
      - when the desire is achieved
      - when the desire becomes impossible
      - which blocks are relevant for this desire
    """

    def __init__(self, desire_id: str, description: str):
        self.desire_id = desire_id
        self.description = description


    @abstractmethod
    def is_achieved(self, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> bool:
        raise NotImplementedError()


    @abstractmethod
    def is_impossible(self, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> bool:
        raise NotImplementedError()


    @abstractmethod
    def get_desired_blocks(self) -> List[Block]:
        raise NotImplementedError()


    def __str__(self) -> str:
        return f"{self.desire_id}: {self.description}"


@dataclass
class PlaceBlockDesire(AgentDesire):
    block: Block
    support: Optional[Block]
    expected_above: List[Block]  

    def __init__(self, block: Block, support: Optional[Block], expected_above: List[Block] = None):
        if support is None:
            description = f"Place {block} on table"
        else:
            description = f"Place {block} on {support}"

        super().__init__(desire_id=f"place-{block}", description=description)
        self.block = block
        self.support = support
        self.expected_above = expected_above if expected_above else []


    def is_achieved(self, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> bool:
        try:
            stack = current_world.get_stack(self.block)

            if self.support is None:
                if not stack.is_on_table(self.block):
                    return False
            else:
                if not stack.is_on(self.block, self.support):
                    return False

            blocks = stack.get_blocks()
            block_idx = blocks.index(self.block)
            actual_above = blocks[block_idx + 1:]

            # cannot have more blocks than target
            if len(actual_above) > len(self.expected_above):
                return False

            # must match prefix of expected
            for i in range(len(actual_above)):
                if actual_above[i] != self.expected_above[i]:
                    return False

            return True

        except Exception:
            return False

    def is_impossible(self, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> bool:
        try:
            stack = current_world.get_stack(self.block)
        except Exception:
            return holding_block != self.block

        if self.support is not None:
            try:
                current_world.get_stack(self.support)
            except Exception:
                return holding_block != self.support

        if stack.is_locked(self.block) and not self.is_achieved(current_world, holding_block):
            return True

        return False


    def get_desired_blocks(self) -> List[Block]:
        return [self.block]
    

    def get_support(self):
        return self.support


@dataclass
class BuildStackDesire(AgentDesire):
    stack_blocks: List[Block]

    def __init__(self, stack_blocks: List[Block]):
        super().__init__(
            desire_id="stack-" + "-".join([str(b) for b in stack_blocks]),
            description="Build stack: " + "-".join([str(b) for b in stack_blocks]),
        )
        self.stack_blocks = list(stack_blocks)


    def is_achieved(self, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> bool:
        if len(self.stack_blocks) == 0:
            return False

        try:
            bottom = self.stack_blocks[0]
            stack = current_world.get_stack(bottom)
            if not stack.is_on_table(bottom):
                return False

            for idx in range(1, len(self.stack_blocks)):
                if not stack.is_on(self.stack_blocks[idx], self.stack_blocks[idx - 1]):
                    return False

            return True
        except Exception:
            return False


    def is_impossible(self, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> bool:
        if len(self.stack_blocks) == 0:
            return True

        for b in self.stack_blocks:
            try:
                current_world.get_stack(b)
            except Exception:
                if holding_block != b:
                    return True

        if self.is_achieved(current_world, holding_block):
            return False

        for idx, b in enumerate(self.stack_blocks):
            try:
                block_stack = current_world.get_stack(b)
                if block_stack.is_locked(b):
                    # A locked block is impossible to move — if it's not in
                    # its correct target position, the stack can never be built.
                    expected_support = None if idx == 0 else self.stack_blocks[idx - 1]
                    if expected_support is None:
                        if not block_stack.is_on_table(b):
                            return True
                    else:
                        if not block_stack.is_on(b, expected_support):
                            return True
            except Exception:
                return True

        return False


    def get_desired_blocks(self) -> List[Block]:
        return list(self.stack_blocks)


@dataclass
class BuildRowDesire(AgentDesire):
    row_blocks: List[Block]
    row_level: int

    def __init__(self, row_blocks: List[Block], row_level: int):
        super().__init__(
            desire_id=f"row-{row_level}-" + "-".join([str(b) for b in row_blocks]),
            description=f"Build row at level {row_level}: " + "-".join([str(b) for b in row_blocks]),
        )
        self.row_blocks = list(row_blocks)
        self.row_level = row_level


    def is_achieved(self, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> bool:
        if len(self.row_blocks) == 0:
            return False

        used_stack_bases = set()
        for block in self.row_blocks:
            try:
                stack = current_world.get_stack(block)
            except Exception:
                return False

            blocks = stack.get_blocks()
            if block not in blocks:
                return False

            if blocks.index(block) != self.row_level:
                return False

            stack_base = stack.get_bottom_block()
            if stack_base in used_stack_bases:
                return False

            used_stack_bases.add(stack_base)

        return True


    def is_impossible(self, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> bool:
        if len(self.row_blocks) == 0:
            return True

        stack_to_row_blocks = {}
        for b in self.row_blocks:
            try:
                block_stack = current_world.get_stack(b)
            except Exception:
                if holding_block != b:
                    return True
                continue

            if block_stack.is_locked(b):
                block_level = block_stack.get_blocks().index(b)
                if block_level != self.row_level:
                    return True

            stack_base = block_stack.get_bottom_block()
            if stack_base not in stack_to_row_blocks:
                stack_to_row_blocks[stack_base] = []
            stack_to_row_blocks[stack_base].append(b)

        if self.is_achieved(current_world, holding_block):
            return False

        for stack_base, row_blocks_in_stack in stack_to_row_blocks.items():
            if len(row_blocks_in_stack) <= 1:
                continue

            locked_row_targets_in_same_stack = []
            for b in row_blocks_in_stack:
                try:
                    if current_world.get_stack(b).is_locked(b):
                        locked_row_targets_in_same_stack.append(b)
                except Exception:
                    return True

            # Row semantics require target row blocks to end up on distinct stacks.
            # If at least two row-target blocks are already locked in the same stack,
            # they cannot be separated anymore, so the row desire is impossible.
            # Note: this does NOT count locked support blocks under the row level.
            if len(locked_row_targets_in_same_stack) >= 2:
                return True

        return False


    def get_desired_blocks(self) -> List[Block]:
        return list(self.row_blocks)


##### here  
class MyAgent(BlocksWorldAgent):

    MODE_IDLE = "IDLE"

    # commited to a plan 
    MODE_COMMITTED = "COMMITTED"

    def __init__(self, name: str, target_state: BlocksWorld):
        super(MyAgent, self).__init__(name=name)

        self.target_state = target_state

        """
        The agent's belief about the world state. Initially, the agent has no belief about the world state.
        """
        self.belief: BlocksWorld = None

        """
        The agent's current desire. It is expressed as a list of blocks for which the agent wants to make a plan to bring to their corresponding
        configuration in the target state. 
        The list can contain a single block or a sequence of blocks that represent: (i) a stack of blocks, (ii) a row of blocks (e.g. going level by level).
        """
        self.current_desire: Optional[AgentDesire] = None

        """
        The set of possible desires (sub-goals) extracted from the target world.
        One desire corresponds to putting one block into its target relation (on table or on another block).
        """
        self.desire_pool: List[AgentDesire] = []

        """
        The current intention is the agent plan (sequence of actions) that the agent is executing to achieve the current desire.
        """
        self.current_intention: List[BlocksWorldAction] = []

        self.mode: str = MyAgent.MODE_IDLE
        self.last_action: Optional[BlocksWorldAction] = None
        self.last_failure_reason: Optional[str] = None

        self._initialize_desire_pool_from_target()


    def response(self, perception: BlocksWorldPerception) -> BlocksWorldAction:
        ## if the perceived state contains the target state, the agent has achieved its goal
        if perception.current_world.contains_world(self.target_state):
            return AgentCompleted()
        
        ## revise the agents beliefs based on the perceived state
        self.revise_beliefs(
            perception.current_world,
            perception.previous_action_succeeded,
            perception.previous_action_message,
        )

        if self.mode == MyAgent.MODE_COMMITTED and self.current_desire:
            if self._is_desire_achieved(self.current_desire, perception.current_world, perception.holding_block):
                self._drop_current_desire("desire achieved")
            elif self._is_desire_impossible(self.current_desire, perception.current_world, perception.holding_block):
                self._drop_current_desire("desire became impossible in current world")

        if self.mode == MyAgent.MODE_IDLE:
            selected_desire = self._select_next_desire(perception.current_world, perception.holding_block)
            if selected_desire is None:
                self.last_failure_reason = "NO_DESIRE_SELECTED"
                self.last_action = NoAction()
                return self.last_action

            self.current_desire = selected_desire
            self.current_intention = []
            self.mode = MyAgent.MODE_COMMITTED

        if self.mode == MyAgent.MODE_COMMITTED and self.current_desire and len(self.current_intention) == 0:
            self.current_intention = self._plan_for_current_desire(perception.current_world, perception.holding_block)
            if len(self.current_intention) == 0:
                self._drop_current_desire("no plan available for committed desire")
                self.last_action = NoAction()
                return self.last_action

        if self.mode == MyAgent.MODE_COMMITTED and self.current_desire and len(self.current_intention) > 0:
            next_action = self.current_intention[0]

            if self._can_apply_action(next_action, perception.current_world, perception.holding_block):
                self.last_action = self.current_intention.pop(0)
                return self.last_action

            self.current_intention = self._plan_for_current_desire(perception.current_world, perception.holding_block)
            if len(self.current_intention) == 0:
                self._drop_current_desire("intention invalidated and replan failed")
                self.last_action = NoAction()
                return self.last_action

            next_action = self.current_intention[0]
            if self._can_apply_action(next_action, perception.current_world, perception.holding_block):
                self.last_action = self.current_intention.pop(0)
                return self.last_action

            self._drop_current_desire("replanned first action still not applicable")

        self.last_action = NoAction()
        return self.last_action


    def _can_apply_action(self, act: BlocksWorldAction, world: BlocksWorld, holding_block: Optional[Block]) -> bool:
        """
        Check if the action can be applied to the current world state.
        """
        ## create a clone of the world
        sim_world = world.clone()

        ## apply the action to the clone, surrpressing any exceptions
        try:
            ## locking can be performed at any time, so check if the action is a lock actio
            if act.get_type() == "lock":
                ## try to lock the block
                sim_world.lock(act.get_argument())
            else:
                if holding_block is None:
                    if act.get_type() == "putdown" or act.get_type() == "stack":
                        ## If we are not holding anything, we cannot putdown or stack a block
                        return False
                    
                    if act.get_type() == "pickup":
                        ## try to pickup the block
                        sim_world.pick_up(act.get_argument())
                    elif act.get_type() == "unstack":
                        ## try to unstack the block
                        sim_world.unstack(act.get_first_arg(), act.get_second_arg())
                else:
                    ## we are holding a block, so we can only putdown or stack
                    if act.get_type() == "pickup" or act.get_type() == "unstack":
                        ## If we are holding a block, we cannot pickup or unstack
                        return False

                    if act.get_type() == "putdown":
                        ## If we want to putdown the block we have to check if it's the same block we are holding
                        if act.get_argument() != holding_block:
                            return False
                        sim_world.put_down(act.get_argument(), sim_world.get_stacks()[-1])

                    if act.get_type() == "stack":
                        ## If we want to stack the block we have to check if it's the same block we are holding
                        if act.get_first_arg() != holding_block:
                            return False
                        ## try to stack the block
                        sim_world.stack(act.get_first_arg(), act.get_second_arg())
        except Exception as e:
            return False
        
        return True


    def _initialize_desire_pool_from_target(self) -> None:
        self.desire_pool = []

        for stack in self.target_state.get_stacks():
            stack_blocks = stack.get_blocks()

            for i, block in enumerate(stack_blocks):
                expected_above = stack_blocks[i + 1:] if i + 1 < len(stack_blocks) else []
                
                if i == 0:
                    self.desire_pool.append(PlaceBlockDesire(
                        block=block, 
                        support=None,
                        expected_above=expected_above
                    ))
                else:
                    support = stack_blocks[i - 1]
                    self.desire_pool.append(PlaceBlockDesire(
                        block=block, 
                        support=support,
                        expected_above=expected_above
                ))


    def _drop_current_desire(self, reason: str) -> None:
        if self.current_desire is not None:
            self.last_failure_reason = f"{reason} | desire={self.current_desire.desire_id} ({self.current_desire.description})"
        else:
            self.last_failure_reason = reason
        self.current_desire = None
        self.current_intention = []
        self.mode = MyAgent.MODE_IDLE


    def _select_next_desire(self, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> Optional[AgentDesire]:
        available_desires = []
    
        for desire in self.desire_pool:
            if self._is_desire_achieved(desire, current_world, holding_block):
                continue

            if self._is_desire_impossible(desire, current_world, holding_block):
                continue
            
            if isinstance(desire, PlaceBlockDesire):
                try:
                    stack = current_world.get_stack(desire.block)
                    
                    # check if the block is in its correct position
                    
                    in_position = False
                    if desire.support is None:
                        in_position = stack.is_on_table(desire.block)
                    else:
                        in_position = stack.is_on(desire.block, desire.support)
                    
                    if in_position and len(desire.expected_above) > 0:
                        blocks = stack.get_blocks()
                        block_idx = blocks.index(desire.block)

                        # the blocks above the block from desire 
                        actual_above = blocks[block_idx + 1:]
                        
                        # current block is correct in its position, just the above it should be completed
                        if len(actual_above) < len(desire.expected_above):
                            continue
                except:
                    pass
            
            available_desires.append(desire)
        
        if not available_desires:
            return None
        
        best_desire = None
        best_priority = 10

        # custom priority given to desires from 0->10
        def get_priority(desire: AgentDesire) -> int:
            if isinstance(desire, PlaceBlockDesire):
                try:
                    stack = current_world.get_stack(desire.block)
                    
                    in_position = False
                    if desire.support is None:
                        in_position = stack.is_on_table(desire.block)
                    else:
                        in_position = stack.is_on(desire.block, desire.support)
                    
                    if in_position and stack.is_clear(desire.block):
                        return 10
                        
                    if desire.support is None:
                        return 0  
                except: 
                    pass
                
                support_of_desire = None
                for _desire in self.desire_pool:
                    if isinstance(_desire, PlaceBlockDesire) and _desire.block == desire.support:
                        support_of_desire = _desire
                        break

                # support is in place
                if support_of_desire and self._is_desire_achieved(support_of_desire, current_world, holding_block):
                    return 1  
                
                return 2 

        for desire in available_desires:
            priority_value = get_priority(desire)

            if priority_value < best_priority:
                best_priority = priority_value
                best_desire = desire

        return best_desire
    

    def _plan_for_current_desire(self, current_world: BlocksWorld, holding_block: Optional[Block]) -> List[BlocksWorldAction]:
        if self.current_desire is None:
            return []
        
        if not isinstance(self.current_desire, PlaceBlockDesire):
            return []
        
        plan = []
        current_block = self.current_desire.block 
        current_support = self.current_desire.support  

        # holding the block 
        if holding_block == current_block:
            # place it down on the table
            if current_support is None:
                return [PutDown(current_block)]
            else:
                # if it must have a support, do that action
                plan.append(Stack(current_block, current_support))
                return plan

        # holding another block than the current desire, put it down (free the hand)
        if holding_block is not None:
            plan.append(PutDown(holding_block))

        # get target stack for the current block
        stack = current_world.get_stack(current_block)
        
        
        if stack.is_locked(current_block):
            return []
        
        # get blocks above and clear them
        blocks = stack.get_blocks()
        block_idx = blocks.index(current_block)
        blocks_above = blocks[block_idx + 1:]

        for above_block in reversed(blocks_above):
            below_block = stack.get_below(above_block)
            plan.append(Unstack(above_block, below_block))
            plan.append(PutDown(above_block))
        
        # pick the current block if it doesnt have anythink above
        if len(blocks_above) == 0 and stack.is_single_block():
            # put the block on the table
            plan.append(PickUp(current_block))
        else:
            # block has something below
            below = stack.get_below(current_block)
            plan.append(Unstack(current_block, below))
        
        
        # place the block (if it does or not have support)
        if current_support is not None:
            plan.append(Stack(current_block, current_support))
        else:
            plan.append(PutDown(current_block))
        
        return plan

    def _is_desire_achieved(self, desire: AgentDesire, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> bool:
        try:
            return desire.is_achieved(current_world, holding_block)
        except NotImplementedError:
            return False


    def _is_desire_impossible(self, desire: AgentDesire, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> bool:
        try:
            return desire.is_impossible(current_world, holding_block)
        except NotImplementedError:
            return True


    def revise_beliefs(self, perceived_world_state: BlocksWorld, previous_action_succeeded: bool,
                      previous_action_message: Optional[str] = None):
        """
        TODO: revise internal agent structured depending on whether what the agent *expects* to be true 
        corresponds to what the agent perceives from the environment.
        :param perceived_world_state: the world state perceived by the agent
        :param previous_action_succeeded: whether the previous action succeeded or not
        """
        self.belief = perceived_world_state.clone()

        if previous_action_succeeded:
            if previous_action_message:
                self.last_failure_reason = f"last outcome: {previous_action_message}"
            return

        if self.mode == MyAgent.MODE_COMMITTED and self.current_desire:
            self.last_failure_reason = (
                f"last action failed while committed to {self.current_desire.desire_id}"
                + (f" | {previous_action_message}" if previous_action_message else "")
            )
        elif previous_action_message:
            self.last_failure_reason = f"last action failed | {previous_action_message}"


    def plan(self) -> Tuple[List[Block], List[BlocksWorldAction]]:
        """
        Deprecated compatibility shim for old lab skeleton.
        Planning must now be done through `_select_next_desire` and `_plan_for_current_desire`.
        """
        selected = self._select_next_desire(self.belief if self.belief is not None else self.target_state, None)
        if selected is None:
            return [], []
        self.current_desire = selected
        return selected.get_desired_blocks(), self._plan_for_current_desire(self.belief if self.belief is not None else self.target_state, None)


    def status_string(self):
        desire_info = "none" if self.current_desire is None else self.current_desire.description
        next_action = "none" if len(self.current_intention) == 0 else str(self.current_intention[0])
        return (
            f"{self} : MODE={self.mode} | DESIRE={desire_info} | "
            f"INTENTION_STEPS={len(self.current_intention)} | NEXT={next_action} | "
            f"LAST={self.last_action} | REASON={self.last_failure_reason}"
        )


class Tester(object):

    STEP_DELAY = 0.2
    VERBOSE = True

    EXT = ".txt"
    SI  = "si"
    SF  = "sf"

    AGENT_NAME = "*A"

    def __init__(self, test_suite, dynamic_prob):
        self._environment = None
        self._agents = []

        
        self.test_suite = test_suite
        self.dynamic_prob = dynamic_prob

        self._initialize_environment(self.test_suite)
        self._initialize_agents(self.test_suite)

    def _initialize_environment(self, test_suite: str) -> None:
        filename = test_suite + Tester.SI + Tester.EXT

        with open(filename) as input_stream:
            self._environment = DynamicEnvironment(
                BlocksWorld(input_stream=input_stream),
                verbose=Tester.VERBOSE,
                dynamics_prob=self.dynamic_prob,
            )


    def _initialize_agents(self, test_suite: str) -> None:
        filename = test_suite + Tester.SF + Tester.EXT

        agent_states = {}

        with open(filename) as input_stream:
            desires = BlocksWorld(input_stream=input_stream)
            agent = MyAgent(Tester.AGENT_NAME, desires)

            agent_states[agent] = desires
            self._agents.append(agent)

            self._environment.add_agent(agent, desires, None)

            if Tester.VERBOSE:
                print("Agent %s desires:" % str(agent))
                print(str(desires))


    def make_steps(self):
        if Tester.VERBOSE:
            print("\n\n================================================= INITIAL STATE:")
            print(str(self._environment))
            print("\n\n=================================================")
        else:
            print("Simulation started (verbose=False)")

        completed = False
        nr_steps = 0

        while not completed:
            completed = self._environment.step()

            time.sleep(Tester.STEP_DELAY)
            if Tester.VERBOSE:
                print(str(self._environment))

                for ag in self._agents:
                    print(ag.status_string())

            nr_steps += 1

            if nr_steps >= 500:
                print("MAX STEP - STOP")
                break

            if Tester.VERBOSE:
                print("\n\n================================================= STEP %i completed." % nr_steps)

        if Tester.VERBOSE:
            print("\n\n================================================= ALL STEPS COMPLETED")
        else:
            print("Simulation completed in %i steps" % nr_steps)


        goal_achieved = self._check_goal_achievement()
    
        if goal_achieved:
            print(f"SUCCESS: Goal achieved in {nr_steps} steps!")
        else:
            print(f"FAILURE: Agent completed but goal not achieved in {nr_steps} steps")
        
        return goal_achieved

    def _check_goal_achievement(self) -> bool:
        """
        Check if all agents achieved their goals.
        Returns True if the current world state contains all desired states.
        """
        current_world = self._environment.worldstate
        
        for agent in self._agents:
            agent_data = self._environment._get_agent_data(agent)
            desired_state = agent_data.target_state
            
            if not current_world.contains_world(desired_state):
                if Tester.VERBOSE:
                    print(f"Agent {agent} did NOT achieve its goal:")
                    print("Current world:")
                    print(str(current_world))
                    print("Desired state:")
                    print(str(desired_state))
                return False
        
        return True



if __name__ == "__main__":
    import sys

    sys.stdout = open("output.txt", "w", encoding="utf-8")

    test_suites = ['tests/0/', 'tests/0a/', 'tests/0c/', 'tests/0d/', 'tests/0d2/', 'tests/0e-large/']
    dynamic_probs = [0.0, 1.0]

    test_failed = 0
    test_success = 0

    test_success_no_dynamic = 0
    test_failed_no_dynamic = 0

    for dynamic_prob in dynamic_probs:
        print(f"-----------DYNAMIC ENV PROB: {dynamic_prob}----------\n")
        for test_suite in test_suites:
            print(f"-----------TEST SUITE: {test_suite}----------\n")
            tester = Tester(test_suite, dynamic_prob)
            achieved = tester.make_steps()

            if achieved:
                test_success += 1

                if dynamic_prob == 0.0:
                    test_success_no_dynamic += 1

            else:
                test_failed += 1

                if dynamic_prob == 0.0:
                    test_failed_no_dynamic += 1

    print(f"SUCCESS: {test_success} / {test_success + test_failed}")
    print(f"SUCCESS FOR NO DYNAMIC: {test_success_no_dynamic} / {test_success_no_dynamic + test_failed_no_dynamic}")
            