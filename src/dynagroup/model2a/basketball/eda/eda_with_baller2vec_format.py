from dynagroup.model2a.basketball.animate import animate_event
from dynagroup.model2a.basketball.data.baller2vec.moments_and_events import (
    get_event_in_baller2vec_format,
)


PATH_TO_GAME_DATA = (
    "/Users/mwojno01/Repos/dynagroup/data/basketball/baller2vec_format/TOR_vs_CHA/0021500492_X.npy",
)
PATH_TO_EVENT_LABEL_DATA = (
    "/Users/mwojno01/Repos/dynagroup/data/basketball/baller2vec_format/TOR_vs_CHA/0021500492_y.npy",
)
PATH_TO_BALLER2VEC_INFO = (
    "/Users/mwojno01/Repos/dynagroup/data/basketball/baller2vec_format/TOR_vs_CHA/baller2vec_config.pydict",
)

event_idxs = [0, 1, 2, 3, 4]
for event_idx in event_idxs:
    event = get_event_in_baller2vec_format(
        event_idx,
        PATH_TO_GAME_DATA,
        PATH_TO_EVENT_LABEL_DATA,
        PATH_TO_BALLER2VEC_INFO,
        sampling_rate_Hz=5,
    )
    print(f"Now animating event idx {event_idx}, which has type {event.label}")
    animate_event(event)
