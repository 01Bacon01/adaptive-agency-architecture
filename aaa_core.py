from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class SearchMode(Enum):
    EXPLORE = "explore"
    EXPLOIT = "exploit"
    BALANCED = "balanced"
    PIVOT = "pivot"


@dataclass(frozen=True)
class State:
    description: str
    variables: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Constraint:
    name: str
    severity: float
    description: str = ""


@dataclass(frozen=True)
class Values:
    priorities: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Interests:
    domains: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    content: Any
    source: str = "unknown"


@dataclass(frozen=True)
class CandidateTransition:
    description: str
    resulting_state: State
    tags: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class Transition:
    description: str
    resulting_state: State
    tags: set[str]

    attainability: float
    alignment: float
    risk: float
    reversibility: float

    optionality_gain: float
    specialization_gain: float
    pivot_capacity: float
    decay_rate: float


@dataclass
class Memory:
    observations: list[Observation] = field(default_factory=list)
    candidates: list[CandidateTransition] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    outcomes: list[Any] = field(default_factory=list)


@dataclass
class WorldModel:
    beliefs: dict[str, Any] = field(default_factory=dict)

    def update(self, observation: Observation) -> None:
        self.beliefs[observation.source] = observation.content


@dataclass(frozen=True)
class AlignmentPolicy:
    values_weight: float = 0.7
    interests_weight: float = 0.3

    def normalized(self) -> "AlignmentPolicy":
        total = self.values_weight + self.interests_weight
        if total <= 0:
            return AlignmentPolicy()
        return AlignmentPolicy(
            values_weight=self.values_weight / total,
            interests_weight=self.interests_weight / total,
        )


@dataclass(frozen=True)
class TradeoffPolicy:
    optionality_weight: float = 0.4
    specialization_weight: float = 0.3
    pivot_weight: float = 0.3

    def normalized(self) -> "TradeoffPolicy":
        total = self.optionality_weight + self.specialization_weight + self.pivot_weight
        if total <= 0:
            return TradeoffPolicy()
        return TradeoffPolicy(
            optionality_weight=self.optionality_weight / total,
            specialization_weight=self.specialization_weight / total,
            pivot_weight=self.pivot_weight / total,
        )


@dataclass(frozen=True)
class Context:
    resources: dict[str, Any] = field(default_factory=dict)
    rules: dict[str, Any] = field(default_factory=dict)
    risks: dict[str, Any] = field(default_factory=dict)


class ObjectiveFunction:
    def score(self, transition: Transition, tradeoff_policy: TradeoffPolicy) -> float:
        tradeoff = tradeoff_policy.normalized()

        adaptive_optionality = (
            tradeoff.optionality_weight * transition.optionality_gain
            + tradeoff.specialization_weight * transition.specialization_gain
            + tradeoff.pivot_weight * transition.pivot_capacity
        )

        penalty = 1.0 + transition.risk + transition.decay_rate

        return (
            transition.attainability
            * transition.alignment
            * transition.reversibility
            * adaptive_optionality
            / penalty
        )

    def total_score(
        self,
        transitions: list[Transition],
        tradeoff_policy: TradeoffPolicy,
    ) -> float:
        return sum(self.score(t, tradeoff_policy) for t in transitions)

    def select_best(
        self,
        transitions: list[Transition],
        tradeoff_policy: TradeoffPolicy,
    ) -> Transition:
        if not transitions:
            raise ValueError("No transitions available.")
        return max(transitions, key=lambda t: self.score(t, tradeoff_policy))


class TransitionGenerator(Protocol):
    def generate(
        self,
        system: "AdaptiveSystem",
        context: Context,
        mode: SearchMode,
    ) -> list[Transition]:
        ...


class Planner(Protocol):
    def plan(
        self,
        system: "AdaptiveSystem",
        context: Context,
        transition: Transition,
    ) -> Any:
        ...


class Executor(Protocol):
    def execute(self, plan: Any) -> Any:
        ...


@dataclass
class AdaptiveSystem:
    current_state: State
    values: Values
    interests: Interests
    constraints: list[Constraint]

    alignment_policy: AlignmentPolicy = field(default_factory=AlignmentPolicy)
    tradeoff_policy: TradeoffPolicy = field(default_factory=TradeoffPolicy)

    memory: Memory = field(default_factory=Memory)
    world_model: WorldModel = field(default_factory=WorldModel)
    objective: ObjectiveFunction = field(default_factory=ObjectiveFunction)

    transition_generator: TransitionGenerator | None = None
    planner: Planner | None = None
    executor: Executor | None = None


class BasicTransitionGenerator:
    def generate(
        self,
        system: AdaptiveSystem,
        context: Context,
        mode: SearchMode,
    ) -> list[Transition]:
        candidates = self.imagine(system, context, mode)
        system.memory.candidates.extend(candidates)
        return [self.evaluate_candidate(c, system, context) for c in candidates]

    def imagine(
        self,
        system: AdaptiveSystem,
        context: Context,
        mode: SearchMode,
    ) -> list[CandidateTransition]:
        current = system.current_state

        candidates = [
            CandidateTransition(
                description="Preserve current trajectory",
                resulting_state=current,
                tags={"stability", "safety"},
            )
        ]

        if mode in {SearchMode.BALANCED, SearchMode.EXPLORE}:
            candidates.append(
                CandidateTransition(
                    description="Expand optionality through new resources",
                    resulting_state=State(
                        description="Expanded resource access",
                        variables={
                            **current.variables,
                            "resources": current.variables.get("resources", 1.0)
                            + 0.25,
                        },
                    ),
                    tags={"optionality", "growth"},
                )
            )

        if mode == SearchMode.EXPLOIT:
            candidates.append(
                CandidateTransition(
                    description="Deepen existing specialization",
                    resulting_state=State(
                        description="Strengthened existing capability",
                        variables={
                            **current.variables,
                            "capability": current.variables.get("capability", 1.0)
                            + 0.25,
                        },
                    ),
                    tags={"specialization", "agency", "growth"},
                )
            )

        if mode == SearchMode.PIVOT:
            candidates.append(
                CandidateTransition(
                    description="Pivot into adjacent trajectory",
                    resulting_state=State(
                        description="Shifted toward adjacent path",
                        variables={
                            **current.variables,
                            "pivot": current.variables.get("pivot", 0.0) + 0.25,
                        },
                    ),
                    tags={"pivot", "optionality", "growth"},
                )
            )

        if mode == SearchMode.EXPLORE:
            candidates.append(
                CandidateTransition(
                    description="Explore novel high-variance trajectory",
                    resulting_state=State(
                        description="Entered novel possibility space",
                        variables={
                            **current.variables,
                            "novelty": current.variables.get("novelty", 0.0) + 0.50,
                        },
                    ),
                    tags={"novelty", "optionality", "growth"},
                )
            )

        return candidates

    def evaluate_candidate(
        self,
        candidate: CandidateTransition,
        system: AdaptiveSystem,
        context: Context,
    ) -> Transition:
        return Transition(
            description=candidate.description,
            resulting_state=candidate.resulting_state,
            tags=candidate.tags,
            attainability=self.estimate_attainability(candidate, system),
            alignment=self.estimate_alignment(candidate, system),
            risk=self.estimate_risk(candidate),
            reversibility=self.estimate_reversibility(candidate),
            optionality_gain=self.estimate_optionality_gain(candidate),
            specialization_gain=self.estimate_specialization_gain(candidate),
            pivot_capacity=self.estimate_pivot_capacity(candidate),
            decay_rate=self.estimate_decay_rate(candidate),
        )

    def estimate_attainability(
        self,
        candidate: CandidateTransition,
        system: AdaptiveSystem,
    ) -> float:
        constraint_penalty = average([c.severity for c in system.constraints])
        score = 0.8 - 0.25 * constraint_penalty

        if "stability" in candidate.tags:
            score += 0.10
        if "novelty" in candidate.tags:
            score -= 0.25
        if "pivot" in candidate.tags:
            score -= 0.15

        return clamp(score)

    def estimate_alignment(
        self,
        candidate: CandidateTransition,
        system: AdaptiveSystem,
    ) -> float:
        policy = system.alignment_policy.normalized()

        value_score = weighted_tag_match(candidate.tags, system.values.priorities)
        interest_score = weighted_tag_match(candidate.tags, system.interests.domains)

        return clamp(
            policy.values_weight * value_score
            + policy.interests_weight * interest_score
        )

    def estimate_risk(self, candidate: CandidateTransition) -> float:
        score = 0.25

        if "stability" in candidate.tags:
            score -= 0.10
        if "novelty" in candidate.tags:
            score += 0.35
        if "pivot" in candidate.tags:
            score += 0.20
        if "specialization" in candidate.tags:
            score += 0.10

        return clamp(score, minimum=0.01)

    def estimate_reversibility(self, candidate: CandidateTransition) -> float:
        score = 0.75

        if "stability" in candidate.tags:
            score += 0.10
        if "novelty" in candidate.tags:
            score -= 0.25
        if "pivot" in candidate.tags:
            score -= 0.10
        if "specialization" in candidate.tags:
            score -= 0.15

        return clamp(score)

    def estimate_optionality_gain(self, candidate: CandidateTransition) -> float:
        score = 0.10

        if "optionality" in candidate.tags:
            score += 0.40
        if "growth" in candidate.tags:
            score += 0.15
        if "novelty" in candidate.tags:
            score += 0.20
        if "specialization" in candidate.tags:
            score -= 0.15
        if "stability" in candidate.tags:
            score -= 0.05

        return clamp(score)

    def estimate_specialization_gain(self, candidate: CandidateTransition) -> float:
        score = 0.10

        if "specialization" in candidate.tags:
            score += 0.50
        if "agency" in candidate.tags:
            score += 0.15
        if "stability" in candidate.tags:
            score += 0.10
        if "novelty" in candidate.tags:
            score -= 0.10
        if "pivot" in candidate.tags:
            score -= 0.05

        return clamp(score)

    def estimate_pivot_capacity(self, candidate: CandidateTransition) -> float:
        score = 0.75

        if "optionality" in candidate.tags:
            score += 0.15
        if "stability" in candidate.tags:
            score += 0.10
        if "specialization" in candidate.tags:
            score -= 0.25
        if "novelty" in candidate.tags:
            score -= 0.15

        return clamp(score)

    def estimate_decay_rate(self, candidate: CandidateTransition) -> float:
        score = 0.10

        if "specialization" in candidate.tags:
            score += 0.20
        if "novelty" in candidate.tags:
            score += 0.10
        if "pivot" in candidate.tags:
            score += 0.05
        if "optionality" in candidate.tags:
            score -= 0.05
        if "stability" in candidate.tags:
            score -= 0.05

        return clamp(score)


class BasicPlanner:
    def plan(
        self,
        system: AdaptiveSystem,
        context: Context,
        transition: Transition,
    ) -> dict[str, Any]:
        return {
            "goal": transition.description,
            "target_state": transition.resulting_state.description,
            "steps": [],
        }


class BasicExecutor:
    def execute(self, plan: Any) -> dict[str, Any]:
        return {
            "status": "placeholder_executed",
            "plan": plan,
        }


def observe(system: AdaptiveSystem, observation: Observation) -> None:
    system.memory.observations.append(observation)
    system.world_model.update(observation)


def generate_transitions(
    system: AdaptiveSystem,
    context: Context,
    mode: SearchMode,
) -> list[Transition]:
    if system.transition_generator is None:
        raise RuntimeError("No transition generator installed.")
    return system.transition_generator.generate(system, context, mode)


def mode_adjusted_tradeoff_policy(
    base: TradeoffPolicy,
    mode: SearchMode,
) -> TradeoffPolicy:
    if mode == SearchMode.EXPLORE:
        return TradeoffPolicy(0.60, 0.10, 0.30)

    if mode == SearchMode.EXPLOIT:
        return TradeoffPolicy(0.20, 0.60, 0.20)

    if mode == SearchMode.PIVOT:
        return TradeoffPolicy(0.30, 0.10, 0.60)

    return base


def score_system(
    system: AdaptiveSystem,
    context: Context,
    mode: SearchMode,
) -> float:
    transitions = generate_transitions(system, context, mode)
    policy = mode_adjusted_tradeoff_policy(system.tradeoff_policy, mode)
    return system.objective.total_score(transitions, policy)


def select_transition(
    system: AdaptiveSystem,
    context: Context,
    mode: SearchMode,
) -> Transition:
    transitions = generate_transitions(system, context, mode)
    policy = mode_adjusted_tradeoff_policy(system.tradeoff_policy, mode)
    return system.objective.select_best(transitions, policy)


def act(
    system: AdaptiveSystem,
    context: Context,
    mode: SearchMode = SearchMode.BALANCED,
) -> dict[str, Any]:
    if system.planner is None:
        raise RuntimeError("No planner installed.")
    if system.executor is None:
        raise RuntimeError("No executor installed.")

    before = score_system(system, context, mode)

    transition = select_transition(system, context, mode)
    plan = system.planner.plan(system, context, transition)
    result = system.executor.execute(plan)

    system.current_state = transition.resulting_state
    system.memory.transitions.append(transition)
    system.memory.outcomes.append(result)

    after = score_system(system, context, mode)

    return {
        "selected_transition": transition.description,
        "result": result,
        "score_before": before,
        "score_after": after,
        "score_delta": after - before,
    }


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def weighted_tag_match(tags: set[str], weights: dict[str, float]) -> float:
    if not weights:
        return 0.5

    total = sum(weights.values())
    if total <= 0:
        return 0.5

    matched = sum(weight for tag, weight in weights.items() if tag in tags)
    return clamp(matched / total)


def make_test_system() -> AdaptiveSystem:
    return AdaptiveSystem(
        current_state=State(
            description="Test state",
            variables={
                "resources": 1.0,
                "capability": 1.0,
                "novelty": 0.0,
                "pivot": 0.0,
            },
        ),
        values=Values(
            priorities={
                "agency": 1.0,
                "optionality": 1.0,
                "safety": 0.8,
                "growth": 0.7,
                "specialization": 0.5,
                "pivot": 0.5,
            }
        ),
        interests=Interests(
            domains={
                "growth": 1.0,
                "optionality": 0.9,
                "specialization": 0.6,
                "pivot": 0.7,
            }
        ),
        constraints=[
            Constraint(name="limited_information", severity=0.5),
        ],
        transition_generator=BasicTransitionGenerator(),
        planner=BasicPlanner(),
        executor=BasicExecutor(),
    )


def run_basic_tests() -> None:
    system = make_test_system()
    context = Context()

    observe(system, Observation(content="test observation", source="test"))
    assert len(system.memory.observations) == 1
    assert system.world_model.beliefs["test"] == "test observation"

    transitions = generate_transitions(system, context, SearchMode.BALANCED)
    assert len(transitions) > 0
    assert all(isinstance(t, Transition) for t in transitions)

    score = score_system(system, context, SearchMode.BALANCED)
    assert isinstance(score, float)
    assert score >= 0.0

    selected = select_transition(system, context, SearchMode.BALANCED)
    assert isinstance(selected, Transition)

    result = act(system, context, SearchMode.BALANCED)
    assert "selected_transition" in result
    assert "score_before" in result
    assert "score_after" in result
    assert "score_delta" in result

    assert len(system.memory.transitions) >= 1
    assert len(system.memory.outcomes) >= 1

    print("All basic tests passed.")


def run_mode_tests() -> None:
    context = Context()
    results = {}

    print()
    print("Search mode behavior:")
    print("-" * 40)

    for mode in SearchMode:
        system = make_test_system()
        result = act(system, context, mode)
        results[mode] = result["selected_transition"]
        print(f"{mode.value:<10} -> {result['selected_transition']}")

    print()
    print(f"Distinct transitions selected: {len(set(results.values()))} / {len(SearchMode)}")


if __name__ == "__main__":
    run_basic_tests()
    run_mode_tests()
