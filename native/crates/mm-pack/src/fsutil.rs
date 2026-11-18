//! The bits of Python's os.path, os.walk, glob and shutil the packaging steps relied on, with the same semantics
//! where they decide an output: the order files are visited in (os.walk: top-down, each folder's entries in
//! readdir order, symlinked folders listed but not entered), which paths a glob returns and in what order (readdir
//! order, `*` skipping dot files, wildcard folders only matching folders), lexical abspath/relpath, and copy2
//! (contents, mode and times). Paths are Strings throughout, as the scripts' were: every rule here is string work
//! on paths (prefixes, basenames), and the bundle's paths are ASCII.

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

pub fn exists(p: &str) -> bool {
    Path::new(p).exists()
}

/// os.path.lexists: true for a broken symlink too.
pub fn lexists(p: &str) -> bool {
    fs::symlink_metadata(p).is_ok()
}

pub fn is_link(p: &str) -> bool {
    fs::symlink_metadata(p).map(|m| m.file_type().is_symlink()).unwrap_or(false)
}

pub fn is_file(p: &str) -> bool {
    Path::new(p).is_file()
}

pub fn is_dir(p: &str) -> bool {
    Path::new(p).is_dir()
}

/// os.path.realpath for a path that exists (what the scripts called it on); falls back to the lexical path.
pub fn realpath(p: &str) -> String {
    fs::canonicalize(p).map(|x| x.to_string_lossy().into_owned()).unwrap_or_else(|_| normpath(p))
}

pub fn join(a: &str, b: &str) -> String {
    if b.starts_with('/') {
        b.to_string()
    } else if a.is_empty() || a.ends_with('/') {
        format!("{a}{b}")
    } else {
        format!("{a}/{b}")
    }
}

pub fn dirname(p: &str) -> String {
    match p.rfind('/') {
        Some(i) => {
            let head = &p[..=i];
            let t = head.trim_end_matches('/');
            if t.is_empty() { head.to_string() } else { t.to_string() }
        }
        None => String::new(),
    }
}

pub fn basename(p: &str) -> String {
    match p.rfind('/') {
        Some(i) => p[i + 1..].to_string(),
        None => p.to_string(),
    }
}

/// os.path.normpath (POSIX), without the double-slash special case.
pub fn normpath(p: &str) -> String {
    if p.is_empty() {
        return ".".into();
    }
    let abs = p.starts_with('/');
    let mut out: Vec<&str> = Vec::new();
    for c in p.split('/') {
        match c {
            "" | "." => {}
            ".." => {
                if !abs && (out.is_empty() || *out.last().unwrap() == "..") {
                    out.push("..");
                } else if !out.is_empty() {
                    out.pop();
                }
            }
            c => out.push(c),
        }
    }
    let s = out.join("/");
    if abs {
        format!("/{s}")
    } else if s.is_empty() {
        ".".into()
    } else {
        s
    }
}

/// os.path.abspath
pub fn abspath(p: &str) -> String {
    if p.starts_with('/') {
        normpath(p)
    } else {
        let cwd = std::env::current_dir().map(|c| c.to_string_lossy().into_owned()).unwrap_or_default();
        normpath(&join(&cwd, p))
    }
}

/// os.path.relpath(path, start) for two absolute paths.
pub fn relpath(path: &str, start: &str) -> String {
    let a = abspath(path);
    let b = abspath(start);
    let pa: Vec<&str> = a.split('/').filter(|c| !c.is_empty()).collect();
    let pb: Vec<&str> = b.split('/').filter(|c| !c.is_empty()).collect();
    let n = pa.iter().zip(pb.iter()).take_while(|(x, y)| x == y).count();
    let mut parts: Vec<&str> = vec![".."; pb.len() - n];
    parts.extend_from_slice(&pa[n..]);
    if parts.is_empty() { ".".into() } else { parts.join("/") }
}

/// A folder's entry names in readdir order (os.listdir / os.scandir order).
pub fn listdir(d: &str) -> Result<Vec<String>> {
    let mut v = Vec::new();
    for e in fs::read_dir(d).with_context(|| format!("listdir {d}"))? {
        v.push(e?.file_name().to_string_lossy().into_owned());
    }
    Ok(v)
}

/// os.walk(top) (top-down, followlinks=False): (folder, files) in the order Python yields them. "files" are the
/// entries that aren't folders once symlinks are followed, as in os.walk.
pub fn walk(top: &str) -> Vec<(String, Vec<String>)> {
    let mut out = Vec::new();
    let mut stack = vec![top.to_string()];
    while let Some(d) = stack.pop() {
        let Ok(names) = listdir(&d) else { continue };
        let mut dirs = Vec::new();
        let mut files = Vec::new();
        for n in names {
            let p = join(&d, &n);
            if is_dir(&p) { dirs.push(n) } else { files.push(n) }
        }
        out.push((d.clone(), files));
        for n in dirs.iter().rev() {
            let p = join(&d, n);
            if !is_link(&p) {
                stack.push(p);
            }
        }
    }
    out
}

/// fnmatch for the patterns used here: `*` and `?` wildcards, everything else literal.
fn fnmatch(name: &str, pat: &str) -> bool {
    fn m(n: &[u8], p: &[u8]) -> bool {
        match p.first() {
            None => n.is_empty(),
            Some(b'*') => (0..=n.len()).any(|i| m(&n[i..], &p[1..])),
            Some(b'?') => !n.is_empty() && m(&n[1..], &p[1..]),
            Some(c) => n.first() == Some(c) && m(&n[1..], &p[1..]),
        }
    }
    m(name.as_bytes(), pat.as_bytes())
}

fn has_magic(s: &str) -> bool {
    s.contains('*') || s.contains('?') || s.contains('[')
}

/// glob.glob(pattern) for absolute patterns with `*`/`?` (no `[...]`, no `**`), in Python's order.
pub fn glob(pattern: &str) -> Vec<String> {
    glob_inner(pattern, false)
}

fn glob_inner(pattern: &str, dironly: bool) -> Vec<String> {
    let (dir, base) = match pattern.rfind('/') {
        Some(i) => (&pattern[..i.max(1)], &pattern[i + 1..]),
        None => ("", pattern),
    };
    let dir = if pattern.starts_with('/') && pattern.rfind('/') == Some(0) { "/" } else { dir };
    if !has_magic(pattern) {
        // glob of a literal path: [path] if it lexists
        return if lexists(pattern) { vec![pattern.to_string()] } else { vec![] };
    }
    let dirs = if has_magic(dir) { glob_inner(dir, true) } else { vec![dir.to_string()] };
    let mut out = Vec::new();
    for d in dirs {
        if has_magic(base) {
            let Ok(names) = listdir(if d.is_empty() { "." } else { &d }) else { continue };
            for n in names {
                if n.starts_with('.') && !base.starts_with('.') {
                    continue;
                }
                if dironly && !is_dir(&join(&d, &n)) {
                    continue;
                }
                if fnmatch(&n, base) {
                    out.push(join(&d, &n));
                }
            }
        } else if !base.is_empty() && lexists(&join(&d, base)) {
            out.push(join(&d, base));
        } else if base.is_empty() && is_dir(&d) {
            out.push(join(&d, base));
        }
    }
    out
}

/// shutil.copy2: contents and permission bits (fs::copy), then the access and modification times.
pub fn copy2(src: &str, dst: &str) -> Result<()> {
    let dst = if is_dir(dst) { join(dst, &basename(src)) } else { dst.to_string() };
    fs::copy(src, &dst).with_context(|| format!("copy {src} -> {dst}"))?;
    let m = fs::metadata(src)?;
    let f = fs::File::options().write(true).open(&dst);
    match f {
        Ok(f) => {
            let t = fs::FileTimes::new().set_accessed(m.accessed()?).set_modified(m.modified()?);
            let _ = f.set_times(t);
        }
        Err(_) => {
            // a read-only copy: open it writable for a moment to set its times
            use std::os::unix::fs::PermissionsExt;
            let mode = m.permissions().mode();
            fs::set_permissions(&dst, fs::Permissions::from_mode(mode | 0o200))?;
            if let Ok(f) = fs::File::options().write(true).open(&dst) {
                let t = fs::FileTimes::new().set_accessed(m.accessed()?).set_modified(m.modified()?);
                let _ = f.set_times(t);
            }
            fs::set_permissions(&dst, fs::Permissions::from_mode(mode))?;
        }
    }
    Ok(())
}

pub fn chmod(p: &str, mode: u32) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(p, fs::Permissions::from_mode(mode)).with_context(|| format!("chmod {p}"))
}

/// A command's stdout (as subprocess.run(..., capture_output=True, text=True, check=True).stdout).
pub fn run_out(prog: &str, args: &[&str]) -> Result<String> {
    let o = std::process::Command::new(prog).args(args).output().with_context(|| format!("running {prog}"))?;
    if !o.status.success() {
        anyhow::bail!(
            "{prog} {} failed ({}): {}",
            args.join(" "),
            o.status,
            String::from_utf8_lossy(&o.stderr).trim_end()
        );
    }
    Ok(String::from_utf8_lossy(&o.stdout).into_owned())
}

/// The same, without failing on a non-zero exit (subprocess.run without check=True).
pub fn run_out_nocheck(prog: &str, args: &[&str]) -> String {
    std::process::Command::new(prog)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default()
}
