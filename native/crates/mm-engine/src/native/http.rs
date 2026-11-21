//! The native engine's routes over HTTP on 127.0.0.1, for what QML loads by URL (frames, thumbnails, the scan's
//! preview and heads) and for tools that talk to the engine as they did to app_server (decisions_cli's routes). One
//! thread per connection, one request per connection (`Connection: close`); the same [`super::Server::handle`] the app
//! calls directly.
use super::{Reply, Server};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Weak};

/// Bind the first free port in [lo, hi).
pub fn bind(lo: u16, hi: u16) -> Option<(TcpListener, u16)> {
    (lo..hi).find_map(|p| TcpListener::bind(("127.0.0.1", p)).ok().map(|l| (l, p)))
}

/// Serve until the server is dropped (the listener thread holds only a weak reference).
pub fn serve(l: TcpListener, srv: Weak<Server>) {
    std::thread::Builder::new()
        .name("mm-native-http".into())
        .spawn(move || {
            for c in l.incoming() {
                let Ok(c) = c else { continue };
                let Some(s) = srv.upgrade() else { return };
                let _ = std::thread::Builder::new().name("mm-native-req".into()).spawn(move || {
                    let _ = one(c, &s);
                });
            }
        })
        .expect("http thread");
}

fn one(c: TcpStream, srv: &Arc<Server>) -> std::io::Result<()> {
    let mut r = BufReader::new(c.try_clone()?);
    let mut line = String::new();
    r.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let (method, path) = (parts.next().unwrap_or("GET").to_string(), parts.next().unwrap_or("/").to_string());
    let mut len = 0usize;
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 || h.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                len = v.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut body = vec![0u8; len.min(1 << 30)];
    r.read_exact(&mut body)?;
    let rep: Reply = srv.handle(&method, &path, &body);
    let (status, ctype, cache) = (rep.status, rep.ctype, rep.cache);
    let body = rep.into_bytes();
    let mut w = c;
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Error",
    };
    let mut head = format!(
        "HTTP/1.1 {} {reason}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n",
        status,
        ctype,
        body.len()
    );
    if let Some(cc) = &cache {
        head.push_str(&format!("Cache-Control: {cc}\r\n"));
    }
    head.push_str("\r\n");
    w.write_all(head.as_bytes())?;
    w.write_all(&body)?;
    w.flush()
}
