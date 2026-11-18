//! `mm-pack relocate APP [--prefix /opt/homebrew]`: make the .app self-contained. Copies every Homebrew library
//! its Mach-O files load into Contents/Frameworks and points them at @rpath, so nothing under /opt/homebrew is read
//! at run time.
//!
//! Walks every Mach-O file already in the bundle (the executables in MacOS/, the Qt and GStreamer plugins
//! make_app.sh copied into PlugIns/, the QML plugins in Resources/qml/) and, for each load command naming a file
//! under PREFIX (directly, or through @rpath / @loader_path resolved against the file's own LC_RPATHs):
//! - copies that file into Contents/Frameworks once, keyed by its real path (Homebrew names one library several
//!   ways: /opt/homebrew/opt/gstreamer/lib/..., /opt/homebrew/Cellar/gstreamer/1.28.9/lib/..., @rpath/...). A plain
//!   dylib keeps the name the load command uses (libglib-2.0.0.dylib, not the fully versioned real file); a Qt
//!   framework is copied as a minimal framework bundle (Versions/A/<binary> + Resources/Info.plist, no headers);
//! - rewrites the load command to @rpath/<name> (or @rpath/QtX.framework/Versions/A/QtX), sets each copied
//!   library's own id the same way (a plugin's id, when it names Homebrew, becomes @loader_path/<file>);
//! - replaces the file's LC_RPATHs with one @loader_path-relative path to Contents/Frameworks.
//! Then it repeats for the copies until nothing new is found, and checks with `otool -l` that no PREFIX path
//! remains: exit 1 if one does. Signing is make_app.sh's job (install_name_tool breaks the signatures; it re-signs
//! after).
//!
//! Prints a JSON summary (make_app.sh keeps it as dist/relocate.json; `notices` reads its `sources`). The order
//! files are visited in, and so the order of `sources`, is relocate.py's: os.walk order, then last-in-first-out.

use crate::fsutil::*;
use crate::json::{dumps_indent1, Value};
use anyhow::{bail, Result};
use std::collections::{HashMap, HashSet};

const MAGICS: [[u8; 4]; 3] = [[0xcf, 0xfa, 0xed, 0xfe], [0xca, 0xfe, 0xba, 0xbe], [0xfe, 0xed, 0xfa, 0xcf]];

fn is_macho(p: &str) -> bool {
    if is_link(p) || !is_file(p) {
        return false;
    }
    use std::io::Read;
    let mut b = [0u8; 4];
    match std::fs::File::open(p) {
        Ok(mut f) => {
            let mut n = 0;
            while n < 4 {
                match f.read(&mut b[n..]) {
                    Ok(0) | Err(_) => break,
                    Ok(k) => n += k,
                }
            }
            n == 4 && MAGICS.contains(&b)
        }
        Err(_) => false,
    }
}

/// The libraries p loads (otool -L lists a dylib's own id first: left out).
fn otool_deps(p: &str) -> Result<Vec<String>> {
    let out = run_out("otool", &["-L", p])?;
    let mut deps: Vec<String> = out
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().split(" (compatibility").next().unwrap_or("").to_string())
        .collect();
    if let Some(i) = own_id(p)? {
        if deps.first() == Some(&i) {
            deps.remove(0);
        }
    }
    Ok(deps)
}

fn own_id(p: &str) -> Result<Option<String>> {
    let out = run_out("otool", &["-D", p])?;
    Ok(out.lines().nth(1).map(|l| l.trim().to_string()))
}

/// The LC_RPATH paths in `otool -l` text (relocate.py's regex: `cmd LC_RPATH\n\s+cmdsize \d+\n\s+path (.+?) \(offset`).
fn parse_rpaths(txt: &str) -> Vec<String> {
    let lines: Vec<&str> = txt.split('\n').collect();
    let mut v = Vec::new();
    for i in 0..lines.len().saturating_sub(2) {
        if !lines[i].ends_with("cmd LC_RPATH") {
            continue;
        }
        let c = lines[i + 1].trim_start();
        let Some(num) = c.strip_prefix("cmdsize ") else { continue };
        if num.is_empty() || !num.bytes().all(|b| b.is_ascii_digit()) || lines[i + 1].len() == c.len() {
            continue;
        }
        let pl = lines[i + 2];
        let p = pl.trim_start();
        if pl.len() == p.len() {
            continue;
        }
        let Some(rest) = p.strip_prefix("path ") else { continue };
        // non-greedy (.+?): the first " (offset" after at least one character
        if let Some(j) = rest.get(1..).and_then(|r| r.find(" (offset")) {
            v.push(rest[..j + 1].to_string());
        }
    }
    v
}

fn rpaths(p: &str) -> Result<Vec<String>> {
    Ok(parse_rpaths(&run_out("otool", &["-l", p])?))
}

/// The file a load command names, on this machine (None when it can't be found).
fn resolve(dep: &str, src_path: &str, src_rpaths: &[String], exe_dir: &str) -> Option<String> {
    let here = dirname(src_path);
    if let Some(rest) = dep.strip_prefix("@rpath/") {
        for rp in src_rpaths {
            let base = rp.replace("@loader_path", &here).replace("@executable_path", exe_dir);
            let c = join(&base, rest);
            if exists(&c) {
                return Some(realpath(&c));
            }
        }
        return None;
    }
    if let Some(rest) = dep.strip_prefix("@loader_path/") {
        let c = join(&here, rest);
        return exists(&c).then(|| realpath(&c));
    }
    if let Some(rest) = dep.strip_prefix("@executable_path/") {
        let c = join(exe_dir, rest);
        return exists(&c).then(|| realpath(&c));
    }
    exists(dep).then(|| realpath(dep))
}

/// (QtX.framework, "QtX.framework/Versions/A/QtX") for a framework binary, else None.
fn framework_parts(real: &str) -> Option<(String, String)> {
    let c: Vec<&str> = real.split('/').collect();
    let n = c.len();
    if n < 5 {
        return None;
    }
    let (fwn, versions, ver, bin) = (c[n - 4], c[n - 3], c[n - 2], c[n - 1]);
    if versions != "Versions" || ver.is_empty() || bin.is_empty() {
        return None;
    }
    if !(fwn.len() > ".framework".len() && fwn.ends_with(".framework")) {
        return None;
    }
    Some((fwn.to_string(), format!("{fwn}/Versions/{ver}/{bin}")))
}

struct State {
    fw: String,
    /// real source path -> (bundle path of the binary, @rpath name), in insertion order
    copied: Vec<(String, (String, String))>,
    copied_idx: HashMap<String, usize>,
    origin: HashMap<String, String>,
    orig_rpaths: HashMap<String, Vec<String>>,
    todo: Vec<String>,
}

impl State {
    fn insert_copied(&mut self, real: &str, dest: String, name: String) {
        if let Some(&i) = self.copied_idx.get(real) {
            self.copied[i].1 = (dest, name);
        } else {
            self.copied_idx.insert(real.to_string(), self.copied.len());
            self.copied.push((real.to_string(), (dest, name)));
        }
    }

    fn lookup(&self, real: &str) -> Option<String> {
        self.copied_idx.get(real).map(|&i| self.copied[i].1 .1.clone())
    }

    fn bring(&mut self, real: &str) -> Result<String> {
        if let Some(n) = self.lookup(real) {
            return Ok(n);
        }
        let fw = self.fw.clone();
        let (dest, name);
        if let Some((fwname, rel)) = framework_parts(real) {
            dest = join(&fw, &rel);
            std::fs::create_dir_all(dirname(&dest))?;
            copy2(real, &dest)?;
            chmod(&dest, 0o755)?;
            // a framework bundle needs Resources/Info.plist to be signed; nothing else (no headers)
            let src_fw = &real[..real.find(&fwname).unwrap() + fwname.len()];
            let ver = rel.split('/').nth(2).unwrap().to_string();
            let plist = format!("{src_fw}/Versions/{ver}/Resources/Info.plist");
            if exists(&plist) {
                let rd = format!("{fw}/{fwname}/Versions/{ver}/Resources");
                std::fs::create_dir_all(&rd)?;
                copy2(&plist, &format!("{rd}/Info.plist"))?;
            }
            let bin_name = basename(&rel);
            let lp = format!("{fw}/{fwname}/Versions/Current");
            if !lexists(&lp) {
                std::os::unix::fs::symlink(&ver, &lp)?;
            }
            for n in [bin_name.as_str(), "Resources"] {
                let lp = format!("{fw}/{fwname}/{n}");
                if !lexists(&lp) && exists(&format!("{fw}/{fwname}/Versions/Current/{n}")) {
                    std::os::unix::fs::symlink(format!("Versions/Current/{n}"), &lp)?;
                }
            }
            name = format!("@rpath/{rel}");
        } else {
            dest = join(&fw, &basename(real));
            copy2(real, &dest)?;
            chmod(&dest, 0o755)?;
            name = format!("@rpath/{}", basename(real));
        }
        self.insert_copied(real, dest.clone(), name.clone());
        self.origin.insert(dest.clone(), real.to_string());
        let rp = rpaths(real)?;
        self.orig_rpaths.insert(dest.clone(), rp);
        self.todo.push(dest);
        Ok(name)
    }

    /// Prefer the name a load command uses for a plain dylib (libglib-2.0.0.dylib, not the versioned real file).
    fn bring_dep(&mut self, dep: &str, real: &str) -> Result<String> {
        if let Some(n) = self.lookup(real) {
            return Ok(n);
        }
        if framework_parts(real).is_none() && !dep.starts_with('@') {
            let base = basename(dep);
            let dest = join(&self.fw, &base);
            if self.copied.iter().any(|(_, c)| c.0 == dest) {
                bail!("two different libraries named {base}: {real}");
            }
            copy2(real, &dest)?;
            chmod(&dest, 0o755)?;
            let name = format!("@rpath/{base}");
            self.insert_copied(real, dest.clone(), name.clone());
            self.origin.insert(dest.clone(), real.to_string());
            let rp = rpaths(real)?;
            self.orig_rpaths.insert(dest.clone(), rp);
            self.todo.push(dest);
            return Ok(name);
        }
        self.bring(real)
    }
}

pub fn run(args: &[String]) -> Result<i32> {
    let app = abspath(&args[0]);
    let mut prefix = "/opt/homebrew".to_string();
    if let Some(i) = args.iter().position(|a| a == "--prefix") {
        prefix = args.get(i + 1).cloned().ok_or_else(|| anyhow::anyhow!("--prefix needs a value"))?;
    }
    let real_prefix = realpath(&prefix);
    let contents = join(&app, "Contents");
    let fw = join(&contents, "Frameworks");
    let exe_dir = join(&contents, "MacOS");
    std::fs::create_dir_all(&fw)?;

    let mut st = State {
        fw: fw.clone(),
        copied: Vec::new(),
        copied_idx: HashMap::new(),
        origin: HashMap::new(),
        orig_rpaths: HashMap::new(),
        todo: Vec::new(),
    };
    for (d, fs) in walk(&contents) {
        for f in fs {
            let p = join(&d, &f);
            if is_macho(&p) {
                st.todo.push(p);
            }
        }
    }
    // remember where each file came from (its original rpaths resolve its @rpath deps), before rewriting anything
    for p in st.todo.clone() {
        st.origin.insert(p.clone(), p.clone());
        let rp = rpaths(&p)?;
        st.orig_rpaths.insert(p, rp);
    }

    let mut done: HashSet<String> = HashSet::new();
    let mut changes = 0i64;
    while let Some(p) = st.todo.pop() {
        if !done.insert(p.clone()) {
            continue;
        }
        let src = st.origin.get(&p).cloned().unwrap_or_else(|| p.clone());
        let mut targs: Vec<String> = Vec::new();
        let deps = otool_deps(&p)?;
        for dep in &deps {
            if dep.starts_with("/System/") || dep.starts_with("/usr/lib/") {
                continue;
            }
            let orp = st.orig_rpaths.get(&p).cloned().unwrap_or_default();
            let mut real = resolve(dep, &src, &orp, &exe_dir);
            if real.is_none() {
                if let Some(rest) = dep.strip_prefix("@rpath/") {
                    if !exists(&join(&fw, rest)) {
                        // a file copied out of Homebrew (a QML plugin): its rpaths pointed into Homebrew's tree
                        let c = join(&join(&prefix, "lib"), rest);
                        real = exists(&c).then(|| realpath(&c));
                    }
                }
            }
            let Some(real) = real else {
                if dep.starts_with("@rpath/") || dep.starts_with("@loader_path") {
                    // already relative (one of ours, or resolved by a later copy); checked at the end
                    continue;
                }
                bail!("{p}: can't find {dep}");
            };
            if !real.starts_with(&prefix) && !real.starts_with(&real_prefix) {
                continue;
            }
            let new = st.bring_dep(dep, &real)?;
            if &new != dep {
                targs.extend(["-change".into(), dep.clone(), new]);
            }
        }
        // own id
        let i = own_id(&p)?;
        if let Some(i) = &i {
            if p.starts_with(&format!("{fw}/")) && (i.starts_with(&prefix) || i.starts_with("@rpath")) {
                targs.extend(["-id".into(), format!("@rpath/{}", relpath(&p, &fw))]);
            } else if i.starts_with(&prefix) {
                // a plugin (dlopen()ed by path): its id only has to stop naming Homebrew
                targs.extend(["-id".into(), format!("@loader_path/{}", basename(&p))]);
            }
        }
        // rpaths: drop all, add the one to Frameworks (only for files that load something through @rpath)
        let old_rp = rpaths(&p)?;
        let uses_rpath = !targs.is_empty() || otool_deps(&p)?.iter().any(|d| d.starts_with("@rpath/"));
        if !uses_rpath && old_rp.is_empty() {
            continue;
        }
        for rp in &old_rp {
            targs.extend(["-delete_rpath".into(), rp.clone()]);
        }
        let rel = relpath(&fw, &dirname(&p));
        targs.extend(["-add_rpath".into(), if rel != "." { format!("@loader_path/{rel}") } else { "@loader_path".into() }]);
        targs.push(p.clone());
        let o = std::process::Command::new("install_name_tool").args(&targs).output()?;
        if !o.status.success() {
            bail!("install_name_tool {p}: {}", String::from_utf8_lossy(&o.stderr));
        }
        changes += 1;
    }

    // every Mach-O in the bundle: no PREFIX left in its load commands or rpaths
    let mut bad = Vec::new();
    let mut n = 0i64;
    for (d, fs) in walk(&contents) {
        for f in fs {
            let p = join(&d, &f);
            if !is_macho(&p) {
                continue;
            }
            n += 1;
            let txt = run_out_nocheck("otool", &["-l", &p]);
            if txt.contains(&prefix) {
                bad.push(Value::Str(relpath(&p, &contents)));
            }
        }
    }
    // {relpath(dest): real}: a later real with the same dest overwrites the value, keeping the first key's place
    let mut sources: Vec<(String, Value)> = Vec::new();
    for (real, (dest, _)) in &st.copied {
        let k = relpath(dest, &contents);
        if let Some(e) = sources.iter_mut().find(|(kk, _)| *kk == k) {
            e.1 = Value::Str(real.clone());
        } else {
            sources.push((k, Value::Str(real.clone())));
        }
    }
    let ok = bad.is_empty();
    let out = Value::Obj(vec![
        ("machos".into(), Value::Int(n)),
        ("rewritten".into(), Value::Int(changes)),
        ("copied".into(), Value::Int(st.copied.len() as i64)),
        ("sources".into(), Value::Obj(sources)),
        ("prefix_left".into(), Value::Arr(bad)),
        ("ok".into(), Value::Bool(ok)),
    ]);
    println!("{}", dumps_indent1(&out));
    Ok(if ok { 0 } else { 1 })
}
