//! mmctl: drive the running native app (native/app) through its developer hooks, from the command line.
//!
//! The app, started with `--dev`, serves its hooks over HTTP on 127.0.0.1 (the first free port in 9200-9299) and
//! writes that port to `ROOT/out/native/devtools.json` (app/src/devtools.cpp). mmctl reads the file, sends one request
//! per command and prints the answer as JSON; it exits 1 when the app says no (`"ok": false`). `native/tools/mmctl`
//! is a wrapper that builds this binary if needed and runs it, so the commands in the docs work unchanged.
//!
//! Commands (the full list, with their flags, is `HELP` below and `mmctl help`): start, state, call, tree, find,
//! click, drag, key, type, eval, grab, reload, logs, wait, perf, restart, rebuild, quit.
//!
//! ROOT is the repo root: `MM_ROOT` if set (an isolated root of symlinks, AGENTS.md), else the first folder holding
//! `native/ARCHITECTURE.md` above the working directory, else above this executable. The log is
//! `ROOT/out/native/app.log`; the app is `ROOT/native/app/build/Mojimask.app` (`native/build.sh`).
//!
//! This is a port of the Python script it replaced, and prints what it printed: the same JSON, laid out as Python's
//! `json.dumps(obj, indent=1)` (keys in the app's order, non-ASCII escaped, floats as Python writes them). So it
//! carries its own small JSON reader and writer and its own HTTP/1.1 client: std only.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Component, Path, PathBuf};
use std::process::{exit, Command, Stdio};
use std::time::{Duration, Instant};

const HELP: &str = r#"mmctl: drive the native app (native/app) through its developer hooks, from the command line. No dependencies.

The app must run with --dev (`mmctl start` does that); it writes its port to out/native/devtools.json.
Every command prints JSON. Exit 1 when the app says no.

  mmctl start [APP ARGS...]       launch the built app with --dev in the background (log: out/native/app.log), wait for
                                  its hooks; e.g. start --video out/poc/clips/jazz_4k.mp4 --data out/poc/data/jazz_4k
  mmctl state [PATH]              the store's state, or the part at a dotted path (editor.video.width)
  mmctl call CMD [JSON]           run a store command as the UI does: call seek '{"t": 40}'; call help
  mmctl tree [--root NAME] [--depth N]       QML items: type, objectName, text, scene rect, visible, enabled, ...
  mmctl find TEXT | --name NAME   items by visible text (exact first, then contains) or objectName
  mmctl click TEXT | --name NAME | X Y        a real mouse click at the item's centre (or at X, Y)
  mmctl drag (--name NAME | --text T | --from X,Y) (--dx DX --dy DY | --to X,Y | --to-name NAME) [--steps N]
  mmctl key KEY                   a real key press: Space, Right, Ctrl+E, A
  mmctl type TEXT                 type text into the focused item
  mmctl eval JS                   evaluate JavaScript in the root QML context: eval 'store.state.editor.t'
  mmctl grab PNG [--name ITEM]    a picture of the window (or one item). The video is greyed unless the clip is public
                                  domain (test_clips/jazz.mp4, out/poc/clips/): AGENTS.md, Privacy
  mmctl reload                    reload the QML now (it also reloads by itself when a file in native/app/qml changes)
  mmctl logs [--since N]          qDebug, QML warnings and Rust's log, in order; "next" is the N for the next call
  mmctl wait JS [--timeout MS]    wait until a JS expression is true: wait 'store.state.editor.playing'
  mmctl perf                      frame times, composite and decode times, QML reload times
  mmctl restart                   relaunch with the same arguments; the store restores its state (open video, frame,
                                  selection, edits) from out/native/state.json
  mmctl rebuild                   cargo build the store dylib, then restart: a Rust change, live (timed)
  mmctl quit
"#;

// ------------------------------------------------------------------------------------------------------------ JSON

/// A JSON value that keeps its objects' key order and its numbers' text, so what the app sends is printed back as
/// Python printed it.
#[derive(Clone, Debug)]
enum J {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    fn get(&self, k: &str) -> Option<&J> {
        match self {
            J::Obj(v) => v.iter().rev().find(|(key, _)| key == k).map(|(_, x)| x),
            _ => None,
        }
    }
    fn str(s: impl Into<String>) -> J {
        J::Str(s.into())
    }
    fn int(n: i64) -> J {
        J::Num(n.to_string())
    }
    fn float(f: f64) -> J {
        J::Num(py_float(f))
    }
    /// Python's truthiness.
    fn truthy(&self) -> bool {
        match self {
            J::Null => false,
            J::Bool(b) => *b,
            J::Num(n) => n.parse::<f64>().map(|f| f != 0.0).unwrap_or(true),
            J::Str(s) => !s.is_empty(),
            J::Arr(a) => !a.is_empty(),
            J::Obj(o) => !o.is_empty(),
        }
    }
    fn as_i64(&self) -> Option<i64> {
        match self {
            J::Num(n) => n.parse::<i64>().ok().or_else(|| n.parse::<f64>().ok().map(|f| f as i64)),
            _ => None,
        }
    }
    fn as_str(&self) -> Option<&str> {
        match self {
            J::Str(s) => Some(s),
            _ => None,
        }
    }
    fn set(&mut self, k: &str, v: J) {
        if let J::Obj(o) = self {
            match o.iter_mut().find(|(key, _)| key == k) {
                Some(slot) => slot.1 = v,
                None => o.push((k.to_string(), v)),
            }
        }
    }
}

fn obj(pairs: Vec<(&str, J)>) -> J {
    J::Obj(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn parse(s: &str) -> Result<J, String> {
    let b = s.as_bytes();
    let mut p = Parser { b, i: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != b.len() {
        return Err(format!("extra data at char {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn err<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("{what} at char {}", self.i))
    }
    fn lit(&mut self, word: &str, v: J) -> Result<J, String> {
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            self.err("expecting value")
        }
    }
    fn value(&mut self) -> Result<J, String> {
        match self.b.get(self.i) {
            None => self.err("expecting value"),
            Some(b'n') => self.lit("null", J::Null),
            Some(b't') => self.lit("true", J::Bool(true)),
            Some(b'f') => self.lit("false", J::Bool(false)),
            Some(b'"') => self.string().map(J::Str),
            Some(b'[') => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(J::Arr(a));
                }
                loop {
                    self.ws();
                    a.push(self.value()?);
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(J::Arr(a));
                        }
                        _ => return self.err("expecting ',' or ']'"),
                    }
                }
            }
            Some(b'{') => {
                self.i += 1;
                let mut o = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(J::Obj(o));
                }
                loop {
                    self.ws();
                    if self.b.get(self.i) != Some(&b'"') {
                        return self.err("expecting property name");
                    }
                    let k = self.string()?;
                    self.ws();
                    if self.b.get(self.i) != Some(&b':') {
                        return self.err("expecting ':'");
                    }
                    self.i += 1;
                    self.ws();
                    let v = self.value()?;
                    // a repeated key keeps its first place and its last value, as Python's dict does
                    match o.iter_mut().find(|(key, _): &&mut (String, J)| *key == k) {
                        Some(slot) => slot.1 = v,
                        None => o.push((k, v)),
                    }
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(J::Obj(o));
                        }
                        _ => return self.err("expecting ',' or '}'"),
                    }
                }
            }
            Some(c) if *c == b'-' || c.is_ascii_digit() => {
                let start = self.i;
                if self.b[self.i] == b'-' {
                    self.i += 1;
                }
                let digits = |p: &mut Self| {
                    let s = p.i;
                    while p.i < p.b.len() && p.b[p.i].is_ascii_digit() {
                        p.i += 1;
                    }
                    p.i > s
                };
                if !digits(self) {
                    return self.err("expecting value");
                }
                if self.b.get(self.i) == Some(&b'.') {
                    self.i += 1;
                    if !digits(self) {
                        return self.err("expecting digits");
                    }
                }
                if matches!(self.b.get(self.i), Some(b'e' | b'E')) {
                    self.i += 1;
                    if matches!(self.b.get(self.i), Some(b'+' | b'-')) {
                        self.i += 1;
                    }
                    if !digits(self) {
                        return self.err("expecting digits");
                    }
                }
                Ok(J::Num(String::from_utf8_lossy(&self.b[start..self.i]).into_owned()))
            }
            Some(_) => {
                // Python also reads these, which Qt never writes
                for (w, f) in [("NaN", f64::NAN), ("Infinity", f64::INFINITY), ("-Infinity", f64::NEG_INFINITY)] {
                    if self.b[self.i..].starts_with(w.as_bytes()) {
                        self.i += w.len();
                        return Ok(J::float(f));
                    }
                }
                self.err("expecting value")
            }
        }
    }
    fn hex4(&mut self) -> Result<u32, String> {
        let h = self.b.get(self.i..self.i + 4).and_then(|h| std::str::from_utf8(h).ok());
        match h.and_then(|h| u32::from_str_radix(h, 16).ok()) {
            Some(v) => {
                self.i += 4;
                Ok(v)
            }
            None => self.err("invalid \\uXXXX escape"),
        }
    }
    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // the opening quote
        let mut out: Vec<u8> = Vec::new();
        loop {
            let Some(&c) = self.b.get(self.i) else { return self.err("unterminated string") };
            self.i += 1;
            match c {
                b'"' => return Ok(String::from_utf8_lossy(&out).into_owned()),
                b'\\' => {
                    let Some(&e) = self.b.get(self.i) else { return self.err("unterminated string") };
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let mut u = self.hex4()?;
                            if (0xD800..0xDC00).contains(&u) && self.b[self.i..].starts_with(b"\\u") {
                                let save = self.i;
                                self.i += 2;
                                let lo = self.hex4()?;
                                if (0xDC00..0xE000).contains(&lo) {
                                    u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
                                } else {
                                    self.i = save;
                                }
                            }
                            char::from_u32(u).unwrap_or('\u{FFFD}')
                        }
                        _ => return self.err("invalid escape"),
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                _ => out.push(c),
            }
        }
    }
}

/// A float as Python's repr writes it: the shortest digits that read back the same, in exponent form below 1e-4 and
/// from 1e16.
fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    let s = format!("{:e}", f.abs()); // shortest round-trip digits: "1.5e-5"
    let (mant, exp) = s.split_once('e').unwrap();
    let e: i32 = exp.parse().unwrap();
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let sign = if f < 0.0 { "-" } else { "" };
    let body = if !(-4..16).contains(&e) {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.clone() };
        format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
    } else if e >= 0 {
        let point = e as usize + 1;
        if digits.len() <= point {
            format!("{digits}{}.0", "0".repeat(point - digits.len()))
        } else {
            format!("{}.{}", &digits[..point], &digits[point..])
        }
    } else {
        format!("0.{}{digits}", "0".repeat((-e - 1) as usize))
    };
    format!("{sign}{body}")
}

/// A number as Python prints it after reading it: an integer literal stays an integer, anything else is a float.
fn py_num(n: &str) -> String {
    let int = n.strip_prefix('-').unwrap_or(n);
    if !int.is_empty() && int.bytes().all(|c| c.is_ascii_digit()) {
        return if int.bytes().all(|c| c == b'0') { "0".into() } else { n.into() };
    }
    match n.parse::<f64>() {
        Ok(f) => py_float(f),
        Err(_) => n.into(), // NaN, Infinity as written
    }
}

/// A string as Python's json writes it (ensure_ascii): everything outside printable ASCII escaped.
fn py_str(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(ch),
            _ => {
                let mut buf = [0u16; 2];
                for u in ch.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", u));
                }
            }
        }
    }
    out.push('"');
}

/// `json.dumps(v, indent=indent)`; `None` is Python's compact form (", " and ": ").
fn dump(v: &J, indent: Option<usize>, level: usize, out: &mut String) {
    let nl = |out: &mut String, lv: usize| {
        if let Some(n) = indent {
            out.push('\n');
            out.push_str(&" ".repeat(n * lv));
        }
    };
    let sep = if indent.is_some() { "," } else { ", " };
    match v {
        J::Null => out.push_str("null"),
        J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        J::Num(n) => out.push_str(&py_num(n)),
        J::Str(s) => py_str(s, out),
        J::Arr(a) if a.is_empty() => out.push_str("[]"),
        J::Obj(o) if o.is_empty() => out.push_str("{}"),
        J::Arr(a) => {
            out.push('[');
            for (k, x) in a.iter().enumerate() {
                if k > 0 {
                    out.push_str(sep);
                }
                nl(out, level + 1);
                dump(x, indent, level + 1, out);
            }
            nl(out, level);
            out.push(']');
        }
        J::Obj(o) => {
            out.push('{');
            for (k, (key, x)) in o.iter().enumerate() {
                if k > 0 {
                    out.push_str(sep);
                }
                nl(out, level + 1);
                py_str(key, out);
                out.push_str(": ");
                dump(x, indent, level + 1, out);
            }
            nl(out, level);
            out.push('}');
        }
    }
}

fn to_string(v: &J, indent: Option<usize>) -> String {
    let mut s = String::new();
    dump(v, indent, 0, &mut s);
    s
}

// ------------------------------------------------------------------------------------------------------------ paths

fn find_root() -> PathBuf {
    if let Some(r) = std::env::var_os("MM_ROOT") {
        return PathBuf::from(r);
    }
    let exe = std::env::current_exe().ok().and_then(|e| e.canonicalize().ok());
    let starts = [std::env::current_dir().ok(), exe.and_then(|e| e.parent().map(Path::to_path_buf))];
    for s in starts.into_iter().flatten() {
        let mut d = Some(s.as_path());
        while let Some(p) = d {
            if p.join("native/ARCHITECTURE.md").exists() {
                return p.to_path_buf();
            }
            d = p.parent();
        }
    }
    std::env::current_dir().unwrap_or_default()
}

/// Python's os.path.abspath: joined to the working directory and tidied, without resolving links.
fn abspath(p: &str) -> PathBuf {
    let joined = std::env::current_dir().unwrap_or_default().join(p);
    let mut out = PathBuf::new();
    for c in joined.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            c => out.push(c.as_os_str()),
        }
    }
    out
}

struct Ctx {
    root: PathBuf,
    info: PathBuf,
    log: PathBuf,
    app: PathBuf,
}

impl Ctx {
    fn new() -> Ctx {
        let root = find_root();
        let n = root.join("out").join("native");
        Ctx {
            info: n.join("devtools.json"),
            log: n.join("app.log"),
            app: root.join("native/app/build/Mojimask.app/Contents/MacOS/Mojimask"),
            root,
        }
    }
    fn info(&self) -> Option<J> {
        let s = std::fs::read_to_string(&self.info).ok()?;
        parse(&s).ok().filter(|j| matches!(j, J::Obj(_)))
    }
}

// ------------------------------------------------------------------------------------------------------------ process

extern "C" {
    fn kill(pid: i32, sig: i32) -> i32;
    fn setsid() -> i32;
}

fn alive(pid: Option<i64>) -> bool {
    match pid {
        Some(p) if p > 0 && p <= i32::MAX as i64 => unsafe { kill(p as i32, 0) == 0 },
        _ => false,
    }
}

fn pid_of(i: &J) -> Option<i64> {
    i.get("pid").and_then(J::as_i64)
}

// ------------------------------------------------------------------------------------------------------------ HTTP

/// Percent-encoding for a query value: letters, digits and `-._~` stay, everything else (a space too) is %XX.
fn quote(s: &str) -> String {
    let mut o = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
            o.push(b as char);
        } else {
            o.push_str(&format!("%{:02X}", b));
        }
    }
    o
}

fn query_string(q: &[(&str, String)]) -> String {
    q.iter().map(|(k, v)| format!("{}={}", quote(k), quote(v))).collect::<Vec<_>>().join("&")
}

enum HttpErr {
    /// couldn't connect, or the connection broke or timed out: the text after "can't reach the app at URL: "
    Reach(String),
}

/// One HTTP/1.1 request to `url` (http://host:port/path?query): (status, body). A body makes it a POST.
fn http(url: &str, body: Option<&[u8]>, timeout: Duration) -> Result<(u16, Vec<u8>), HttpErr> {
    let rest = url.strip_prefix("http://").ok_or_else(|| HttpErr::Reach(format!("unknown url type: {url}")))?;
    let (hostport, path) = match rest.find('/') {
        Some(k) => (&rest[..k], &rest[k..]),
        None => (rest, "/"),
    };
    let oserr = |e: std::io::Error| -> HttpErr {
        if matches!(e.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock) {
            return HttpErr::Reach("timed out".into());
        }
        // as Python's urllib says it: <urlopen error [Errno 61] Connection refused>
        let msg = e.to_string();
        let msg = msg.split(" (os error").next().unwrap_or(&msg).to_string();
        match e.raw_os_error() {
            Some(n) => HttpErr::Reach(format!("<urlopen error [Errno {n}] {msg}>")),
            None => HttpErr::Reach(format!("<urlopen error {msg}>")),
        }
    };
    let addr = hostport.to_socket_addrs().map_err(oserr)?.next().ok_or_else(|| HttpErr::Reach("no address".into()))?;
    let mut s = TcpStream::connect_timeout(&addr, timeout).map_err(oserr)?;
    s.set_read_timeout(Some(timeout)).ok();
    s.set_write_timeout(Some(timeout)).ok();
    let method = if body.is_some() { "POST" } else { "GET" };
    let b = body.unwrap_or(&[]);
    let mut req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {hostport}\r\nUser-Agent: mmctl\r\nAccept-Encoding: identity\r\n\
         Content-Type: application/json\r\nConnection: close\r\n"
    );
    if body.is_some() {
        req.push_str(&format!("Content-Length: {}\r\n", b.len()));
    }
    req.push_str("\r\n");
    let mut msg = req.into_bytes();
    msg.extend_from_slice(b);
    s.write_all(&msg).map_err(oserr)?;

    // the head
    let mut buf = Vec::new();
    let mut chunk = [0u8; 65536];
    let head_end = loop {
        if let Some(k) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break k;
        }
        let n = s.read(&mut chunk).map_err(oserr)?;
        if n == 0 {
            return Err(HttpErr::Reach("<urlopen error Remote end closed connection without response>".into()));
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
    let mut rest = buf[head_end + 4..].to_vec();
    let mut lines = head.split("\r\n");
    let status: u16 = lines.next().and_then(|l| l.split_whitespace().nth(1)).and_then(|c| c.parse().ok()).unwrap_or(0);
    let mut length = None;
    let mut chunked = false;
    for l in lines {
        if let Some((k, v)) = l.split_once(':') {
            let (k, v) = (k.trim().to_ascii_lowercase(), v.trim());
            if k == "content-length" {
                length = v.parse::<usize>().ok();
            } else if k == "transfer-encoding" && v.to_ascii_lowercase().contains("chunked") {
                chunked = true;
            }
        }
    }
    // the body: by length, chunks, or to the end
    let mut read_more = |rest: &mut Vec<u8>| -> Result<bool, HttpErr> {
        let n = s.read(&mut chunk).map_err(oserr)?;
        rest.extend_from_slice(&chunk[..n]);
        Ok(n > 0)
    };
    if chunked {
        let mut out = Vec::new();
        let mut pos = 0;
        loop {
            let line_end = loop {
                if let Some(k) = rest[pos..].windows(2).position(|w| w == b"\r\n") {
                    break pos + k;
                }
                if !read_more(&mut rest)? {
                    return Ok((status, out));
                }
            };
            let size_text = String::from_utf8_lossy(&rest[pos..line_end]).into_owned();
            let size = usize::from_str_radix(size_text.split(';').next().unwrap_or("").trim(), 16).unwrap_or(0);
            pos = line_end + 2;
            if size == 0 {
                return Ok((status, out));
            }
            while rest.len() < pos + size + 2 {
                if !read_more(&mut rest)? {
                    break;
                }
            }
            let end = (pos + size).min(rest.len());
            out.extend_from_slice(&rest[pos..end]);
            pos = (pos + size + 2).min(rest.len());
        }
    }
    match length {
        Some(n) => {
            while rest.len() < n {
                if !read_more(&mut rest)? {
                    break;
                }
            }
            rest.truncate(n);
        }
        None => while read_more(&mut rest)? {},
    }
    Ok((status, rest))
}

/// Python's repr of the first bytes of a body, for "HTTP 404: b'...'".
fn bytes_repr(b: &[u8]) -> String {
    let quote = if b.contains(&b'\'') && !b.contains(&b'"') { '"' } else { '\'' };
    let mut s = format!("b{quote}");
    for &c in b {
        match c {
            b'\\' => s.push_str("\\\\"),
            b'\n' => s.push_str("\\n"),
            b'\r' => s.push_str("\\r"),
            b'\t' => s.push_str("\\t"),
            c if c as char == quote => {
                s.push('\\');
                s.push(quote);
            }
            0x20..=0x7e => s.push(c as char),
            _ => s.push_str(&format!("\\x{:02x}", c)),
        }
    }
    s.push(quote);
    s
}

enum Resp {
    Json(J),
    Raw(Vec<u8>),
}

// ------------------------------------------------------------------------------------------------------------ output

fn out(v: &J, code: i32) -> ! {
    let indent = match v {
        J::Arr(_) | J::Obj(_) | J::Null => Some(1),
        _ => None,
    };
    println!("{}", to_string(v, indent));
    exit(code)
}

fn fail(msg: impl Into<String>) -> ! {
    out(&obj(vec![("ok", J::Bool(false)), ("error", J::str(msg))]), 1)
}

fn show(r: &J) -> ! {
    let good = r.get("ok").map(J::truthy).unwrap_or(true);
    out(r, if good { 0 } else { 1 })
}

fn round_to(f: f64, places: i32) -> f64 {
    let m = 10f64.powi(places);
    (f * m).round() / m
}

impl Ctx {
    fn request(&self, route: &str, body: Option<&J>, query: &[(&str, String)], timeout: f64, raw: bool) -> Resp {
        let Some(i) = self.info() else {
            fail("the app isn't running with --dev (no out/native/devtools.json): mmctl start")
        };
        let base = i.get("url").and_then(J::as_str).unwrap_or("").to_string();
        let url = format!("{base}{route}{}", if query.is_empty() { String::new() } else { format!("?{}", query_string(query)) });
        let data = body.map(|b| to_string(b, None).into_bytes());
        match http(&url, data.as_deref(), Duration::from_secs_f64(timeout)) {
            Ok((status, b)) if (200..300).contains(&status) => {
                if raw {
                    return Resp::Raw(b);
                }
                match std::str::from_utf8(&b).ok().and_then(|s| parse(s).ok()) {
                    Some(j) => Resp::Json(j),
                    None => Resp::Json(obj(vec![
                        ("ok", J::Bool(false)),
                        ("error", J::str(format!("the app's answer isn't JSON: {}", bytes_repr(&b[..b.len().min(200)])))),
                    ])),
                }
            }
            Ok((status, b)) => match std::str::from_utf8(&b).ok().and_then(|s| parse(s).ok()) {
                Some(j) => Resp::Json(j),
                None => Resp::Json(obj(vec![
                    ("ok", J::Bool(false)),
                    ("error", J::str(format!("HTTP {status}: {}", bytes_repr(&b[..b.len().min(200)])))),
                ])),
            },
            Err(HttpErr::Reach(e)) => Resp::Json(obj(vec![
                ("ok", J::Bool(false)),
                ("error", J::str(format!("can't reach the app at {url}: {e}"))),
            ])),
        }
    }

    fn json(&self, route: &str, body: Option<&J>, query: &[(&str, String)]) -> J {
        self.json_t(route, body, query, 60.0)
    }

    fn json_t(&self, route: &str, body: Option<&J>, query: &[(&str, String)], timeout: f64) -> J {
        match self.request(route, body, query, timeout, false) {
            Resp::Json(j) => j,
            Resp::Raw(_) => unreachable!(),
        }
    }

    /// Wait until devtools.json names a live process (not old_pid) whose hooks answer.
    fn wait_up(&self, old_pid: Option<i64>, timeout: f64) -> (Option<J>, f64) {
        let t0 = Instant::now();
        while t0.elapsed().as_secs_f64() < timeout {
            if let Some(i) = self.info() {
                let pid = pid_of(&i);
                if pid != old_pid && alive(pid) {
                    let url = format!("{}/state?path=app", i.get("url").and_then(J::as_str).unwrap_or(""));
                    if let Ok((_, b)) = http(&url, None, Duration::from_secs(2)) {
                        let ok = std::str::from_utf8(&b).ok().and_then(|s| parse(s).ok());
                        if ok.and_then(|j| j.get("ok").map(J::truthy)).unwrap_or(false) {
                            return (Some(i), t0.elapsed().as_secs_f64());
                        }
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        (None, t0.elapsed().as_secs_f64())
    }
}

/// Pull `--name VALUE` options out of args (the first of each).
fn opts(args: &mut Vec<String>, names: &[&'static str]) -> Vec<(&'static str, String)> {
    let mut got = Vec::new();
    for &n in names {
        if let Some(k) = args.iter().position(|a| a == n) {
            if k + 1 >= args.len() {
                fail(format!("{n} needs a value"));
            }
            got.push((n, args[k + 1].clone()));
            args.drain(k..k + 2);
        }
    }
    got
}

fn opt<'a>(o: &'a [(&str, String)], name: &str) -> Option<&'a str> {
    o.iter().find(|(k, _)| *k == name).map(|(_, v)| v.as_str())
}

fn num(s: &str) -> f64 {
    s.trim().parse::<f64>().unwrap_or_else(|_| fail(format!("could not convert string to float: '{s}'")))
}

fn spec(text: Option<&str>, name: Option<&str>, xy: Option<&str>) -> J {
    if let Some(n) = name {
        return obj(vec![("name", J::str(n))]);
    }
    if let Some(xy) = xy {
        let Some((x, y)) = xy.split_once(',').filter(|(_, y)| !y.contains(',')) else { fail(format!("X,Y: '{xy}'")) };
        return obj(vec![("x", J::float(num(x))), ("y", J::float(num(y)))]);
    }
    obj(vec![("text", text.map(J::str).unwrap_or(J::Null))])
}

/// Python's `x.replace(".", "", 1).lstrip("-").isdigit()`.
fn looks_numeric(x: &str) -> bool {
    let s = x.replacen('.', "", 1);
    let s = s.trim_start_matches('-');
    !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit())
}

fn main() {
    let mut a: Vec<String> = std::env::args().skip(1).collect();
    if a.is_empty() || matches!(a[0].as_str(), "-h" | "--help" | "help") {
        println!("{HELP}");
        return;
    }
    let cmd = a.remove(0);
    let c = Ctx::new();
    let joined = |a: &[String]| a.join(" ");
    match cmd.as_str() {
        "start" => {
            if let Some(i) = c.info() {
                if alive(pid_of(&i)) {
                    let port = i.get("port").map(|p| to_string(p, None)).unwrap_or_default();
                    fail(format!("already running (pid {}, port {port}): mmctl quit first", pid_of(&i).unwrap_or(0)));
                }
            }
            if !c.app.exists() {
                fail(format!("no app at {}: build it (native/README.md)", c.app.display()));
            }
            if let Some(d) = c.log.parent() {
                let _ = std::fs::create_dir_all(d);
            }
            // Paths are the caller's (relative to where mmctl runs), but the app runs in ROOT: make them absolute.
            let args: Vec<String> = a
                .iter()
                .enumerate()
                .map(|(k, x)| {
                    if k > 0 && matches!(a[k - 1].as_str(), "--video" | "--data") && !x.contains("://") {
                        abspath(x).to_string_lossy().into_owned()
                    } else {
                        x.clone()
                    }
                })
                .collect();
            let log = std::fs::OpenOptions::new().create(true).append(true).open(&c.log);
            let log = log.unwrap_or_else(|e| fail(format!("can't open {}: {e}", c.log.display())));
            let err = log.try_clone().unwrap_or_else(|e| fail(format!("can't open {}: {e}", c.log.display())));
            let mut command = Command::new(&c.app);
            command.arg("--dev").args(&args).current_dir(&c.root).stdout(Stdio::from(log)).stderr(Stdio::from(err));
            // its own session, as Python's start_new_session: a Ctrl-C in this terminal doesn't reach the app
            unsafe {
                std::os::unix::process::CommandExt::pre_exec(&mut command, || {
                    setsid();
                    Ok(())
                });
            }
            let child = command.spawn().unwrap_or_else(|e| fail(format!("can't start {}: {e}", c.app.display())));
            let (i, waited) = c.wait_up(None, 30.0);
            let Some(i) = i else {
                fail(format!("the app's hooks didn't come up in 30 s (pid {}); see {}", child.id(), c.log.display()))
            };
            let logs = c.json("/logs", None, &[]);
            let errs: Vec<J> = match logs.get("result").and_then(|r| r.get("lines")) {
                Some(J::Arr(lines)) => lines
                    .iter()
                    .filter(|e| e.get("level").and_then(J::as_str) == Some("error"))
                    .map(|e| e.get("message").cloned().unwrap_or(J::Null))
                    .collect(),
                _ => Vec::new(),
            };
            let mut r = obj(vec![
                ("ok", J::Bool(errs.is_empty())),
                ("pid", i.get("pid").cloned().unwrap_or(J::Null)),
                ("port", i.get("port").cloned().unwrap_or(J::Null)),
                ("seconds", J::float(round_to(waited, 3))),
                ("log", J::str(c.log.to_string_lossy())),
            ]);
            let bad = !errs.is_empty();
            if bad {
                r.set("errors", J::Arr(errs));
            }
            out(&r, if bad { 1 } else { 0 })
        }
        "state" => {
            let q: Vec<(&str, String)> = a.first().map(|p| vec![("path", p.clone())]).unwrap_or_default();
            show(&c.json("/state", None, &q))
        }
        "call" => {
            if a.is_empty() {
                fail("call CMD [JSON]");
            }
            let args = match a.get(1) {
                Some(s) => parse(s).unwrap_or_else(|e| fail(format!("JSON: {e}"))),
                None => J::Obj(Vec::new()),
            };
            show(&c.json("/call", Some(&obj(vec![("cmd", J::str(&a[0])), ("args", args)])), &[]))
        }
        "tree" => {
            let o = opts(&mut a, &["--root", "--depth"]);
            let q: Vec<(&str, String)> = o.iter().map(|(k, v)| (&k[2..], v.clone())).collect();
            show(&c.json("/tree", None, &q))
        }
        "find" => {
            let o = opts(&mut a, &["--name", "--text"]);
            let q = match opt(&o, "--name") {
                Some(n) => obj(vec![("name", J::str(n))]),
                None => obj(vec![("text", J::str(opt(&o, "--text").filter(|t| !t.is_empty()).map(String::from).unwrap_or_else(|| joined(&a))))]),
            };
            show(&c.json("/find", Some(&q), &[]))
        }
        "click" => {
            let o = opts(&mut a, &["--name", "--text"]);
            let b = if let Some(n) = opt(&o, "--name") {
                obj(vec![("name", J::str(n))])
            } else if a.len() == 2 && a.iter().all(|x| looks_numeric(x)) {
                obj(vec![("x", J::float(num(&a[0]))), ("y", J::float(num(&a[1])))])
            } else {
                obj(vec![("text", J::str(opt(&o, "--text").filter(|t| !t.is_empty()).map(String::from).unwrap_or_else(|| joined(&a))))])
            };
            show(&c.json("/click", Some(&b), &[]))
        }
        "drag" => {
            let o = opts(&mut a, &["--name", "--text", "--from", "--to", "--to-name", "--dx", "--dy", "--steps"]);
            let mut b = obj(vec![("from", spec(opt(&o, "--text"), opt(&o, "--name"), opt(&o, "--from")))]);
            if opt(&o, "--to").is_some() || opt(&o, "--to-name").is_some() {
                b.set("to", spec(None, opt(&o, "--to-name"), opt(&o, "--to")));
            } else {
                b.set("dx", J::float(opt(&o, "--dx").map(num).unwrap_or(0.0)));
                b.set("dy", J::float(opt(&o, "--dy").map(num).unwrap_or(0.0)));
            }
            if let Some(s) = opt(&o, "--steps") {
                let n: i64 = s.trim().parse().unwrap_or_else(|_| fail(format!("invalid literal for int(): '{s}'")));
                b.set("steps", J::int(n));
            }
            show(&c.json("/drag", Some(&b), &[]))
        }
        "key" => {
            let Some(k) = a.first() else { fail("key KEY") };
            show(&c.json("/key", Some(&obj(vec![("key", J::str(k))])), &[]))
        }
        "type" => show(&c.json("/type", Some(&obj(vec![("text", J::str(joined(&a)))])), &[])),
        "eval" => show(&c.json("/eval", Some(&obj(vec![("js", J::str(joined(&a)))])), &[])),
        "grab" => {
            let o = opts(&mut a, &["--name"]);
            if a.is_empty() {
                fail("grab PNG [--name ITEM]");
            }
            let q: Vec<(&str, String)> = opt(&o, "--name").map(|n| vec![("name", n.to_string())]).unwrap_or_default();
            let b = match c.request("/grab", None, &q, 60.0, true) {
                Resp::Json(j) => show(&j),
                Resp::Raw(b) => b,
            };
            let path = abspath(&a[0]);
            if let Some(d) = path.parent() {
                let _ = std::fs::create_dir_all(d);
            }
            if let Err(e) = std::fs::write(&path, &b) {
                fail(format!("can't write {}: {e}", path.display()));
            }
            let png = b.len() >= 24 && b.starts_with(b"\x89PNG\r\n\x1a\n");
            let be = |r: &[u8]| u32::from_be_bytes([r[0], r[1], r[2], r[3]]) as i64;
            let w = if png { be(&b[16..20]) } else { 0 };
            let h = if w != 0 { be(&b[20..24]) } else { 0 };
            out(
                &obj(vec![
                    ("ok", J::Bool(true)),
                    ("png", J::str(path.to_string_lossy())),
                    ("width", J::int(w)),
                    ("height", J::int(h)),
                    ("bytes", J::int(b.len() as i64)),
                ]),
                0,
            )
        }
        "reload" => show(&c.json("/reload", Some(&J::Obj(Vec::new())), &[])),
        "logs" => {
            let o = opts(&mut a, &["--since"]);
            show(&c.json("/logs", None, &[("since", opt(&o, "--since").unwrap_or("0").to_string())]))
        }
        "wait" => {
            let o = opts(&mut a, &["--timeout"]);
            let t = opt(&o, "--timeout").unwrap_or("10000");
            let ms: i64 = t.trim().parse().unwrap_or_else(|_| fail(format!("invalid literal for int(): '{t}'")));
            let b = obj(vec![("js", J::str(joined(&a)))]);
            show(&c.json_t("/wait", Some(&b), &[("timeout", ms.to_string())], ms as f64 / 1000.0 + 10.0))
        }
        "perf" => show(&c.json("/perf", None, &[])),
        "restart" | "rebuild" => {
            let mut res = obj(vec![("ok", J::Bool(true))]);
            let t0 = Instant::now();
            if cmd == "rebuild" {
                let r = Command::new("cargo")
                    .args(["build", "-p", "mm-app", "--lib"])
                    .current_dir(c.root.join("native"))
                    .stdin(Stdio::null())
                    .output()
                    .unwrap_or_else(|e| fail(format!("can't run cargo: {e}")));
                res.set("cargo_seconds", J::float(round_to(t0.elapsed().as_secs_f64(), 2)));
                if !r.status.success() {
                    let err = String::from_utf8_lossy(&r.stderr);
                    let n = err.chars().count();
                    let tail: String = err.chars().skip(n.saturating_sub(4000)).collect();
                    out(&obj(vec![("ok", J::Bool(false)), ("error", J::str("cargo build failed")), ("stderr", J::str(tail))]), 1);
                }
            }
            let not_running = |res: &J, msg: String| -> ! {
                let mut r = obj(vec![("ok", J::Bool(false)), ("error", J::str(msg))]);
                if let Some(s) = res.get("cargo_seconds") {
                    r.set("cargo_seconds", s.clone());
                }
                out(&r, 1)
            };
            let i = match c.info() {
                Some(i) if alive(pid_of(&i)) => i,
                _ => not_running(&res, "the app isn't running: mmctl start".into()),
            };
            let t1 = Instant::now();
            c.json("/restart", Some(&J::Obj(Vec::new())), &[]);
            let (j, _) = c.wait_up(pid_of(&i), 60.0);
            let Some(j) = j else { not_running(&res, format!("the app didn't come back in 60 s; see {}", c.log.display())) };
            res.set("pid", j.get("pid").cloned().unwrap_or(J::Null));
            res.set("port", j.get("port").cloned().unwrap_or(J::Null));
            res.set("restart_seconds", J::float(round_to(t1.elapsed().as_secs_f64(), 2)));
            res.set("total_seconds", J::float(round_to(t0.elapsed().as_secs_f64(), 2)));
            out(&res, 0)
        }
        "quit" => {
            let i = c.info();
            let r = c.json("/quit", Some(&J::Obj(Vec::new())), &[]);
            if let Some(i) = i {
                let t0 = Instant::now();
                while alive(pid_of(&i)) && t0.elapsed() < Duration::from_secs(10) {
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
            show(&r)
        }
        _ => fail(format!("no command '{cmd}' (mmctl help)")),
    }
}
