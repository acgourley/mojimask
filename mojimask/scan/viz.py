"""What the "decide" step is doing, as events the scan screen animates (app.html, "Deciding who's who").

Every event is real work as it happens -- nothing is made up for the picture:
  pieces       the tracklets (join.py): n frames, the frame's aspect, and each piece's detections as [t, x, y, size]
               (as fractions of the frame width)
  follow       one tracklet's forward and backward follow (join.py): per frame x, y and how it saw the head (face /
               looks the same / hold), why it stopped and which piece it met
  joins        the mutual joins (join.py): [earlier piece, later piece] pairs
  faces_start  the face measurements begin (faces.py): how many frames and detections
  faces        one frame's face measurements: which detections showed a comparable (frontal) face
  chains       the chains the graph starts from: each chain's pieces (graph.py)
  pairs        how many pairs of chains were compared, how many say "same" and how many "ask"
  merge, refuse, ask   one merge made, one refused (on screen together), one question kept (graph.py)
  people       the groups once fragments are attached and overlaps and blips decided (cover.py)
  cover        one person's cover made: the frames it fills beyond the detections
  same         the same-person questions (same.py)
newalgo.run's progress bar counts follow, faces and cover events against pieces, faces_start and people.

emit() is a no-op unless a sink is set (newalgo.run(..., events=)). One scan at a time sets it (newalgo.run holds a
lock): the sink is global because the join and the face measurements run in thread pools.
"""
import threading

sink = None
_lock = threading.Lock()


def emit(e, **data):
    s = sink
    if s is None:
        return
    with _lock:
        s({"e": e, **data})
