from dynagroup.model2a.basketball.animate import animate_event
from dynagroup.model2a.basketball.data.baller2vec_format import (
    get_event_in_baller2vec_format,
)


event_idxs = [0, 1, 2, 3, 4]
for event_idx in event_idxs:
    event = get_event_in_baller2vec_format(event_idx)
    print(f"Now animating event idx {event_idx}, which has type {event.label}")
    animate_event(event)
