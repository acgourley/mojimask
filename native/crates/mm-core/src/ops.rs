//! The edits document's operations. Port of `_apply_op` and its tables in `mojimask/review_server.py`.
//!
//! The project's edits have ONE owner: the project's process. Pages (Review, the editor, a second window) keep a
//! working copy and send only what they changed, as small operations on paths (edits_sync.js computes them):
//!   {"op": "set", "path": ["not_face", "142"], "value": "you"}     set a value (objects are created on the way)
//!   {"op": "del", "path": ["trims", "79"]}                          remove a key
//!   {"op": "add" / "remove", "path": ["not_real"], "value": "150"}  add to / take from a set-like list
//! so two pages can never overwrite each other's answers with a stale copy (the old whole-file save let the
//! last writer win). Every change is saved at once (write + rename) and is one step on a shared undo history
//! ([`crate::edits`]).
//!
//! "The same value" is Python's test: `json.dumps(x, sort_keys=True)` equal (so `1` and `1.0` differ, key order
//! doesn't matter).
use crate::pyjson::Py;

pub const EDITS_VERSION: i64 = 2;
/// Set-like lists (+ people.<pid>.pieces): add / remove work on these only.
pub const SET_PATHS: [&str; 5] = ["deleted", "not_real", "dismissed", "shown", "not_same"];
/// The edits fields the facts own (facts::derive writes them after every change; pages never do).
pub const DERIVED: [&str; 8] = ["people", "emoji", "not_face", "part_of", "merges", "not_same", "shown", "attach"];

pub fn is_set(path: &[String]) -> bool {
    (path.len() == 1 && SET_PATHS.contains(&path[0].as_str())) || (path.len() == 3 && path[0] == "people" && path[2] == "pieces")
}

/// An op's path as strings (`[str(p) for p in op["path"]]`).
pub fn path_of(op: &Py) -> Result<Vec<String>, String> {
    match op.get("path") {
        Py::List(l) => Ok(l.iter().map(Py::py_str).collect()),
        Py::Str(s) => Ok(s.chars().map(String::from).collect()),
        Py::Dict(d) => Ok(d.keys().map(String::from).collect()),
        Py::Null if !op.as_dict().is_some_and(|d| d.contains("path")) => Err("'path'".into()),
        other => Err(format!("'{}' object is not iterable", type_name(other))),
    }
}

fn type_name(v: &Py) -> &'static str {
    match v {
        Py::Null => "NoneType",
        Py::Bool(_) => "bool",
        Py::Int(_) => "int",
        Py::Float(_) => "float",
        Py::Str(_) => "str",
        Py::List(_) => "list",
        Py::Dict(_) => "dict",
    }
}

fn op_value(op: &Py) -> Result<Py, String> {
    op.as_dict().and_then(|d| d.get("value")).cloned().ok_or_else(|| "'value'".into())
}

/// Apply one op to doc in place; returns the ops that undo it ([] if it changed nothing). Err: a bad op (the
/// caller drops the whole batch).
pub fn apply_op(doc: &mut Py, op: &Py) -> Result<Vec<Py>, String> {
    let d = op.as_dict().ok_or_else(|| format!("'{}' object is not subscriptable", type_name(op)))?;
    let kind = d.get("op").ok_or("'op'")?.clone();
    let path = path_of(op)?;
    let kind = kind.as_str().unwrap_or("");
    if path.is_empty() || !["set", "del", "add", "remove"].contains(&kind) {
        return Err(format!("bad op {}", py_repr(op)));
    }
    let mut parent = doc.as_dict_mut().ok_or("edits: not an object")?;
    for k in &path[..path.len() - 1] {
        if !matches!(parent.get(k), Some(Py::Dict(_))) {
            if kind == "del" || kind == "remove" {
                return Ok(vec![]);
            }
            parent.insert(k.clone(), Py::Dict(Default::default()));
        }
        parent = parent.get_mut(k).unwrap().as_dict_mut().unwrap();
    }
    let k = path.last().unwrap().clone();
    let pv = || Py::List(path.iter().map(Py::from).collect());
    if kind == "add" || kind == "remove" {
        if !is_set(&path) {
            return Err(format!("not a set: {}", py_repr(&pv())));
        }
        let value = op_value(op)?;
        let lst = match parent.get(&k) {
            Some(Py::List(l)) => l.clone(),
            _ => vec![],
        };
        let key = value.dumps_sorted();
        let have = lst.iter().any(|x| x.dumps_sorted() == key);
        if kind == "add" && !have {
            let mut l = lst;
            l.push(value.clone());
            parent.insert(k, Py::List(l));
            return Ok(vec![crate::pydict![("op", "remove"), ("path", pv()), ("value", value)]]);
        }
        if kind == "remove" && have {
            parent.insert(k, Py::List(lst.into_iter().filter(|x| x.dumps_sorted() != key).collect()));
            return Ok(vec![crate::pydict![("op", "add"), ("path", pv()), ("value", value)]]);
        }
        return Ok(vec![]);
    }
    let old = parent.get(&k).cloned();
    if kind == "set" {
        let value = op_value(op)?;
        if let Some(o) = &old {
            if o.dumps_sorted() == value.dumps_sorted() {
                return Ok(vec![]);
            }
        }
        parent.insert(k, value);
    } else {
        if old.is_none() {
            return Ok(vec![]);
        }
        parent.remove(&k);
    }
    Ok(vec![match old {
        Some(o) => crate::pydict![("op", "set"), ("path", pv()), ("value", o)],
        None => crate::pydict![("op", "del"), ("path", pv())],
    }])
}

/// Apply a batch of ops to a copy of doc, all or nothing. Ok((new doc, the ops that take it back)).
pub fn apply_all(doc: &Py, ops: &[Py]) -> Result<(Py, Vec<Py>), String> {
    let mut work = doc.clone();
    let mut inverse: Vec<Py> = vec![];
    for op in ops {
        let mut inv = apply_op(&mut work, op)?;
        inv.extend(inverse);
        inverse = inv;
    }
    Ok((work, inverse))
}

/// Roughly Python's repr of a JSON value (for error messages).
pub fn py_repr(v: &Py) -> String {
    match v {
        Py::Null => "None".into(),
        Py::Bool(b) => if *b { "True".into() } else { "False".into() },
        Py::Str(s) => format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'")),
        Py::List(l) => format!("[{}]", l.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Py::Dict(d) => format!("{{{}}}", d.iter().map(|(k, v)| format!("'{k}': {}", py_repr(v))).collect::<Vec<_>>().join(", ")),
        other => other.py_str(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(s: &str) -> Py {
        Py::parse(s).unwrap()
    }

    #[test]
    fn set_del_add_remove_and_their_inverses() {
        let doc = Py::parse(r#"{"trims": {"79": [1, 2]}, "deleted": []}"#).unwrap();
        let ops = [
            op(r#"{"op": "set", "path": ["light", "strength"], "value": 0.5}"#),
            op(r#"{"op": "del", "path": ["trims", "79"]}"#),
            op(r#"{"op": "add", "path": ["deleted"], "value": "150"}"#),
            op(r#"{"op": "add", "path": ["deleted"], "value": "150"}"#),
        ];
        let (new, inv) = apply_all(&doc, &ops).unwrap();
        assert_eq!(new.dumps(), r#"{"trims": {}, "deleted": ["150"], "light": {"strength": 0.5}}"#);
        assert_eq!(inv.len(), 3);
        let (back, _) = apply_all(&new, &inv).unwrap();
        assert_eq!(back.dumps(), r#"{"trims": {"79": [1, 2]}, "deleted": [], "light": {}}"#); // (created objects stay)
    }

    #[test]
    fn same_value_is_no_change_and_bad_ops_fail() {
        let doc = Py::parse(r#"{"x": {"a": 1, "b": 2}}"#).unwrap();
        assert!(apply_all(&doc, &[op(r#"{"op": "set", "path": ["x"], "value": {"b": 2, "a": 1}}"#)]).unwrap().1.is_empty());
        assert!(!apply_all(&doc, &[op(r#"{"op": "set", "path": ["x"], "value": {"b": 2, "a": 1.5}}"#)]).unwrap().1.is_empty());
        assert!(apply_all(&doc, &[op(r#"{"op": "add", "path": ["x"], "value": 1}"#)]).unwrap_err().starts_with("not a set"));
        assert!(apply_all(&doc, &[op(r#"{"op": "move", "path": ["x"]}"#)]).unwrap_err().starts_with("bad op"));
        assert!(apply_all(&doc, &[op(r#"{"op": "del", "path": ["nope", "deeper"]}"#)]).unwrap().1.is_empty());
        assert!(is_set(&["people".into(), "5".into(), "pieces".into()]));
    }
}
