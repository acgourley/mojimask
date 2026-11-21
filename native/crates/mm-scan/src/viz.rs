//! What the "decide" step is doing, as events the scan screen animates (Python `mojimask/scan/viz.py`; the app shows
//! them as "Deciding who's who").
//!
//! Every event is real work as it happens -- nothing is made up for the picture:
//!   pieces       the tracklets (join): n frames, the frame's aspect, and each piece's detections as [t, x, y, size]
//!                (as fractions of the frame width)
//!   follow       one tracklet's forward and backward follow (join): per frame x, y and how it saw the head (face /
//!                looks the same / hold), why it stopped and which piece it met
//!   joins        the mutual joins (join): [earlier piece, later piece] pairs
//!   faces_start  the face measurements begin (faces): how many frames and detections
//!   faces        one frame's face measurements: which detections showed a comparable (frontal) face
//!   chains       the chains the graph starts from: each chain's pieces (graph)
//!   pairs        how many pairs of chains were compared, how many say "same" and how many "ask"
//!   merge, refuse, ask   one merge made, one refused (on screen together), one question kept (graph)
//!   people       the groups once fragments are attached and overlaps and blips decided (cover)
//!   cover        one person's cover made: the frames it fills beyond the detections
//!   same         the same-person questions (same)
//! newalgo's progress counts follow, faces and cover events against pieces, faces_start and people.
//!
//! Python keeps one global sink (its stages run in thread pools); here the log is passed along explicitly. Each event
//! gets "ms", the time since the step started, as Python's sink adds it.
use mm_core::pyjson::{Dict, Py};

pub struct Viz {
    pub events: Vec<Py>,
    t0: std::time::Instant,
    pub sink: Option<Box<dyn FnMut(&Py)>>,
}

impl Default for Viz {
    fn default() -> Self {
        Viz { events: vec![], t0: std::time::Instant::now(), sink: None }
    }
}

impl Viz {
    pub fn emit(&mut self, e: &str, data: Vec<(&str, Py)>) {
        let mut d = Dict::new();
        d.insert("e", Py::Str(e.to_string()));
        for (k, v) in data {
            d.insert(k, v);
        }
        d.insert("ms", Py::Int(self.t0.elapsed().as_millis() as i64));
        let ev = Py::Dict(d);
        if let Some(s) = self.sink.as_mut() {
            s(&ev);
        }
        self.events.push(ev);
    }
}

/// Python's round(x, n) as a JSON float.
pub(crate) fn r(x: f64, n: usize) -> Py {
    Py::Float(mm_core::pyjson::py_round(x, n))
}
