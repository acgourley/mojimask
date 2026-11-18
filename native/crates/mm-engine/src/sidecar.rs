//! The Python backend (native/docs/porting.md: the UI first, over the Python engine): today's engine as a sidecar process, spoken to over HTTP.
//!
//! [`Sidecar::start`] runs `app_server.py` (unchanged) through `python/sidecar.py`, on a port in the ui stream's
//! range (9300-9399, native/README.md), with `MOJIMASK_PROJECTS` set to the app's projects folder. The project
//! servers app_server starts (`mojimask/review_server.py`, one per open project) take ports from the same range; the
//! app reaches them through app_server's `/p/<id>/...` proxy, as `web/app.html` does.
//!
//! Lifetime: the sidecar holds our end of its stdin. Dropping the [`Sidecar`] (the app quitting) closes it; so does
//! the app dying, since the OS closes the pipe. The sidecar then stops its project servers and exits. Drop waits a
//! few seconds for that, then kills it.
use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const SCRIPT: &str = include_str!("python/sidecar.py");

#[derive(Clone, Debug)]
pub struct Options {
    /// the project's Python (its venv: mojimask's dependencies)
    pub python: String,
    /// the repository root (app_server.py, mojimask/)
    pub root: String,
    /// MOJIMASK_PROJECTS
    pub projects: String,
    /// developer mode (MOJIMASK_DEV=1 for app_server and its project servers)
    pub dev: bool,
    /// the port range [lo, hi): the sidecar's own port and its project servers'
    pub ports: (u16, u16),
    /// the sidecar's output
    pub log: String,
    /// extra environment (MOJIMASK_DEPTH, ...)
    pub env: Vec<(String, String)>,
}

pub struct Sidecar {
    child: Mutex<Child>,
    stdin: Mutex<Option<ChildStdin>>,
    pub port: u16,
    pub base: String,
    agent: ureq::Agent,
    slow: ureq::Agent,
}

fn port_free(p: u16) -> bool {
    std::net::TcpListener::bind(("127.0.0.1", p)).is_ok()
}

impl Sidecar {
    pub fn start(o: &Options) -> Result<Sidecar> {
        let (lo, hi) = o.ports;
        let port = (lo..hi).find(|&p| port_free(p)).ok_or_else(|| anyhow!("no free port in {lo}-{hi}"))?;
        std::fs::create_dir_all(&o.projects).with_context(|| o.projects.clone())?;
        if let Some(d) = PathBuf::from(&o.log).parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let log = std::fs::File::create(&o.log).with_context(|| o.log.clone())?;
        let mut cmd = Command::new(&o.python);
        cmd.arg("-u").arg("-c").arg(SCRIPT).arg(&o.root).arg(port.to_string()).arg((port + 1).to_string()).arg(hi.to_string());
        if o.dev {
            cmd.arg("--dev");
        }
        cmd.current_dir(&o.root)
            .env("MOJIMASK_PROJECTS", &o.projects)
            .env("PYTHONPATH", &o.root)
            .env_remove("MOJIMASK_APP")
            .stdin(Stdio::piped())
            .stdout(log.try_clone()?)
            .stderr(log);
        for (k, v) in &o.env {
            cmd.env(k, v);
        }
        let mut child = cmd.spawn().with_context(|| format!("starting {}", o.python))?;
        let stdin = child.stdin.take();
        let cfg = |secs| ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(secs))).http_status_as_error(false).build();
        let sc = Sidecar {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
            port,
            base: format!("http://127.0.0.1:{port}"),
            agent: ureq::Agent::new_with_config(cfg(30)),
            slow: ureq::Agent::new_with_config(cfg(900)),
        };
        // ready when it answers (Python's imports take a second or two)
        let t0 = Instant::now();
        loop {
            if let Ok(Some(st)) = sc.child.lock().unwrap().try_wait() {
                bail!("the sidecar exited ({st}): see {}", o.log);
            }
            if sc.get("/api/app").is_ok() {
                break;
            }
            if t0.elapsed() > Duration::from_secs(60) {
                bail!("the sidecar didn't answer in 60 s: see {}", o.log);
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        Ok(sc)
    }

    pub fn url(&self, path: &str) -> String {
        format!("{}{}", self.base, path)
    }

    fn reply(r: Result<ureq::http::Response<ureq::Body>, ureq::Error>, what: &str) -> Result<Vec<u8>> {
        let mut r = r.map_err(|e| anyhow!("{what}: {e}"))?;
        let status = r.status().as_u16();
        let b = r.body_mut().with_config().limit(1 << 30).read_to_vec().map_err(|e| anyhow!("{what}: {e}"))?;
        if !(200..300).contains(&status) {
            let msg = serde_json::from_slice::<Value>(&b)
                .ok()
                .and_then(|v| v.get("error").and_then(Value::as_str).map(String::from))
                .unwrap_or_else(|| String::from_utf8_lossy(&b).chars().take(200).collect());
            bail!("{what}: {status} {msg}");
        }
        Ok(b)
    }

    pub fn get_bytes(&self, path: &str) -> Result<Vec<u8>> {
        Self::reply(self.agent.get(self.url(path)).call(), path)
    }

    pub fn get(&self, path: &str) -> Result<Value> {
        Ok(serde_json::from_slice(&self.get_bytes(path)?)?)
    }

    /// POST JSON; `slow`: requests that can take minutes (a project server starting, an answer re-resolving a long clip)
    pub fn post(&self, path: &str, body: &Value, slow: bool) -> Result<Value> {
        let a = if slow { &self.slow } else { &self.agent };
        let b = serde_json::to_vec(body)?;
        let r = a.post(self.url(path)).header("Content-Type", "application/json").send(&b[..]);
        Ok(serde_json::from_slice(&Self::reply(r, path)?)?)
    }

    pub fn get_slow(&self, path: &str) -> Result<Value> {
        Ok(serde_json::from_slice(&Self::reply(self.slow.get(self.url(path)).call(), path)?)?)
    }

    pub fn alive(&self) -> bool {
        matches!(self.child.lock().unwrap().try_wait(), Ok(None))
    }

    pub fn pid(&self) -> u32 {
        self.child.lock().unwrap().id()
    }
}

impl Drop for Sidecar {
    fn drop(&mut self) {
        // closing its stdin asks it to stop its project servers and exit
        if let Some(mut s) = self.stdin.lock().unwrap().take() {
            let _ = s.flush();
        }
        let mut c = self.child.lock().unwrap();
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(8) {
            if let Ok(Some(_)) = c.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = c.kill();
        let _ = c.wait();
    }
}
