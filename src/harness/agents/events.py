"""The event bus: wraps payloads in ``Event`` envelopes and fans them out to subscribers."""

import asyncio
from collections import defaultdict

from harness.core.events import EVENT_BY_PAYLOAD, Event, EventPayload
from harness.core.types import AgentId, Clock


class Subscription:
    """One consumer of the event stream; registered from the moment it is created."""

    def __init__(self, bus: "EventBus") -> None:
        self._bus = bus
        self._queue: asyncio.Queue[Event | None] = asyncio.Queue()

    def push(self, event: Event | None) -> None:
        """Deliver an event (``None`` ends the stream)."""
        self._queue.put_nowait(event)

    def __aiter__(self) -> "Subscription":
        return self

    async def __anext__(self) -> Event:
        event = await self._queue.get()
        if event is None:
            self._bus.unsubscribe(self)
            raise StopAsyncIteration
        return event

    def drain(self) -> list[Event]:
        """Return the events delivered so far without waiting."""
        events: list[Event] = []
        while not self._queue.empty():
            event = self._queue.get_nowait()
            if event is not None:
                events.append(event)
        return events


class EventBus:
    """Publishes events with a per-agent sequence number."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._seqs: defaultdict[AgentId | None, int] = defaultdict(int)
        self._subscriptions: list[Subscription] = []

    def publish(self, agent_id: AgentId | None, payload: EventPayload) -> Event:
        """Wrap ``payload`` in an envelope and deliver it to every subscriber."""
        self._seqs[agent_id] += 1
        event = Event(
            name=EVENT_BY_PAYLOAD[type(payload)],
            agent_id=agent_id,
            seq=self._seqs[agent_id],
            ts=self._clock.now(),
            payload=payload,
        )
        for subscription in self._subscriptions:
            subscription.push(event)
        return event

    def subscribe(self) -> Subscription:
        """Open a new stream that receives every event published from now on."""
        subscription = Subscription(self)
        self._subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        """Stop delivering to a subscription."""
        if subscription in self._subscriptions:
            self._subscriptions.remove(subscription)

    def forget(self, agent_id: AgentId) -> None:
        """Drop the sequence counter of a deleted agent."""
        self._seqs.pop(agent_id, None)

    def close(self) -> None:
        """End every open stream."""
        for subscription in list(self._subscriptions):
            subscription.push(None)
