//! `mm-pack trim-icu APP`: replace the bundle's ICU data library (Homebrew's libicudata, 34 MB) with one holding
//! only what Qt uses (~10 MB).
//!
//! Why ICU is there at all: Homebrew's QtCore links ICU, so it can't be left out without a Qt built with -no-icu.
//! What QtCore calls (nm -u QtCore, Qt 6.12): ucnv_* (QStringConverter's non-UTF encodings), ucol_* (QCollator) and
//! u_strToLower/Upper (QLocale's case mapping); nothing else. Qt keeps its own CLDR tables for QLocale, so ICU's
//! display names (lang/, region/, curr/, unit/, zone/), number spell-out (rbnf/), break iterators (brkitr/),
//! transliterators and the per-locale .res files (all but root and en) are never read.
//!
//! Kept: every converter (*.cnv, cnvalias.icu), every collation tailoring (coll/), normalization (*.nrm, *.spp),
//! root/en and the structural tables (res_index, pool, supplementalData, likelySubtags, metadata, keyTypeData, ...).
//! Checked: ucol_open/strcoll (root, de), ucnv_open (UTF-8, Shift_JIS, windows-1252) and u_strToUpper("tr") give
//! the same results as with the full data (2026-11-17).
//!
//! How: the data is one symbol (_icudtNN_dat) in libicudata's __TEXT,__const; it is cut out, trimmed with
//! `icupkg -r` (Homebrew's icu4c, build time only), and assembled back into a dylib with the same install name and
//! versions (clang). The result is the same file trim_icu.py made, byte for byte.

use crate::fsutil::*;
use crate::json::f1;
use anyhow::{anyhow, bail, Context, Result};
use std::io::{Read, Seek, SeekFrom};
use std::process::Command;

const KEEP_TOP: [&str; 18] = [
    "root.res", "res_index.res", "pool.res", "supplementalData.res", "likelySubtags.res", "metadata.res",
    "keyTypeData.res", "en.res", "en_US.res", "en_US_POSIX.res", "icustd.res", "icuver.res", "plurals.res",
    "grammaticalFeatures.res", "zoneinfo64.res", "timezoneTypes.res", "windowsZones.res", "numberingSystems.res",
];
const DROP_DIRS: [&str; 8] = ["curr", "zone", "region", "lang", "unit", "rbnf", "brkitr", "translit"];

/// The ICU major version in "…libicudata.NN.dylib" (trim_icu.py's `libicudata\.(\d+)\.dylib`, first match).
fn icu_version(lib: &str) -> Option<String> {
    let key = "libicudata.";
    let mut from = 0;
    while let Some(i) = lib[from..].find(key) {
        let s = from + i + key.len();
        let digits: String = lib[s..].chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() && lib[s + digits.len()..].starts_with(".dylib") {
            return Some(digits);
        }
        from = from + i + 1;
    }
    None
}

/// (size, offset) of __TEXT,__const in `otool -l` text (the script's regex over five lines).
fn const_section(txt: &str) -> Option<(u64, u64)> {
    let l: Vec<&str> = txt.split('\n').collect();
    let ws = |s: &str| s.len() > s.trim_start().len();
    for i in 0..l.len().saturating_sub(4) {
        if !l[i].ends_with("sectname __const") {
            continue;
        }
        if !(ws(l[i + 1]) && l[i + 1].trim_start() == "segname __TEXT") {
            continue;
        }
        let a = l[i + 2].trim_start();
        let Some(addr) = a.strip_prefix("addr ") else { continue };
        if !ws(l[i + 2]) || addr.is_empty() || addr.contains(char::is_whitespace) {
            continue;
        }
        let s = l[i + 3].trim_start();
        let Some(hex) = s.strip_prefix("size 0x") else { continue };
        if !ws(l[i + 3]) || hex.is_empty() || !hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
            continue;
        }
        let o = l[i + 4].trim_start();
        let Some(off) = o.strip_prefix("offset ") else { continue };
        // `(\d+)` takes the leading digits
        let d: String = off.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !ws(l[i + 4]) || d.is_empty() {
            continue;
        }
        return Some((u64::from_str_radix(hex, 16).ok()?, d.parse().ok()?));
    }
    None
}

fn check(c: &mut Command, what: &str) -> Result<()> {
    let s = c.status().with_context(|| format!("running {what}"))?;
    if !s.success() {
        bail!("{what} failed ({s})");
    }
    Ok(())
}

pub fn run(app: &str) -> Result<()> {
    let fw = join(&join(app, "Contents"), "Frameworks");
    let libs = glob(&join(&fw, "libicudata.*.dylib"));
    let Some(lib) = libs.first().cloned() else {
        println!("{{\"icu\": \"none in bundle\"}}");
        return Ok(());
    };
    let ver = icu_version(&lib).ok_or_else(|| anyhow!("{lib}: no ICU version in its name"))?;
    let mut icu = glob(&format!("/opt/homebrew/opt/icu4c@{ver}/sbin/icupkg"));
    icu.extend(glob("/opt/homebrew/opt/icu4c/sbin/icupkg"));
    icu.sort();
    let Some(icupkg) = icu.first().cloned() else { bail!("no icupkg for ICU {ver}") };
    let sym = format!("_icudt{ver}_dat");
    let nm = run_out("nm", &["-m", &lib])?;
    if !nm.contains(&sym) {
        bail!("{lib}: no {sym}");
    }
    let sect = run_out("otool", &["-l", &lib])?;
    let (size, off) = const_section(&sect).ok_or_else(|| anyhow!("{lib}: no __TEXT,__const section"))?;
    let before = std::fs::metadata(&lib)?.len();

    // tempfile.TemporaryDirectory(): a fresh folder in $TMPDIR, removed afterwards
    let t = std::env::temp_dir().join(format!(
        "mm-pack-icu-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()
    ));
    std::fs::create_dir_all(&t)?;
    let t = t.to_string_lossy().into_owned();
    let r = (|| -> Result<(usize, usize)> {
        let name = format!("icudt{ver}l");
        let dat = join(&t, &format!("{name}.dat"));
        let mut buf = Vec::new();
        let mut f = std::fs::File::open(&lib)?;
        f.seek(SeekFrom::Start(off))?;
        f.take(size).read_to_end(&mut buf)?;
        std::fs::write(&dat, &buf)?;
        let items_txt = run_out(&icupkg, &["-l", &dat])?;
        let items: Vec<&str> = items_txt.split_whitespace().collect();
        let drop: Vec<&str> = items
            .iter()
            .copied()
            .filter(|i| {
                (i.contains('/') && DROP_DIRS.contains(&i.split('/').next().unwrap()))
                    || (!i.contains('/') && i.ends_with(".res") && !KEEP_TOP.contains(i))
            })
            .collect();
        let drop_txt = join(&t, "drop.txt");
        std::fs::write(&drop_txt, format!("{}\n", drop.join("\n")))?;
        std::fs::create_dir_all(join(&t, "out"))?;
        let out = join(&join(&t, "out"), &format!("{name}.dat"));
        check(Command::new(&icupkg).args(["-r", &drop_txt, &dat, &out]), "icupkg -r")?;
        let asm = join(&t, "data.s");
        std::fs::write(&asm, format!(".section __TEXT,__const\n.globl {sym}\n.p2align 4\n{sym}:\n.incbin \"{out}\"\n"))?;
        let idname = run_out_nocheck("otool", &["-D", &lib])
            .lines()
            .nth(1)
            .map(|l| l.trim().to_string())
            .ok_or_else(|| anyhow!("{lib}: no install name"))?;
        let new = join(&t, "lib.dylib");
        check(
            Command::new("clang").args([
                "-dynamiclib",
                "-arch",
                "arm64",
                "-mmacosx-version-min=15.0",
                "-install_name",
                &idname,
                "-compatibility_version",
                &format!("{ver}.0.0"),
                "-current_version",
                &format!("{ver}.1.0"),
                "-o",
                &new,
                &asm,
            ]),
            "clang",
        )?;
        // os.replace; the temporary folder may be on another volume than the bundle
        if std::fs::rename(&new, &lib).is_err() {
            std::fs::copy(&new, &lib)?;
        }
        Ok((items.len(), drop.len()))
    })();
    let _ = std::fs::remove_dir_all(&t);
    let (n_items, n_drop) = r?;
    let after = std::fs::metadata(&lib)?.len();
    println!(
        "{{\"icu\": \"{}\", \"items_dropped\": {}, \"items_kept\": {}, \"mb_before\": {}, \"mb_after\": {}}}",
        basename(&lib),
        n_drop,
        n_items - n_drop,
        f1(before as f64 / 1e6),
        f1(after as f64 / 1e6)
    );
    Ok(())
}
