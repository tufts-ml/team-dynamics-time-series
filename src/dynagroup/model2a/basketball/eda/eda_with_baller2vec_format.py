from dynagroup.model2a.basketball.animate import animate_event
from dynagroup.model2a.basketball.data.baller2vec_format import (
    get_event_in_baller2vec_format,
)


event = get_event_in_baller2vec_format()
animate_event(event)
