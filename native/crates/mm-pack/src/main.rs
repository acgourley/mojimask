//! mm-pack: the steps of native/packaging/make_app.sh that work on the built bundle, as one program.
//!
//!     mm-pack relocate APP [--prefix /opt/homebrew]   copy Homebrew's libraries in, point load commands at @rpath
//!     mm-pack trim-icu APP                            cut ICU's data to what Qt uses
//!     mm-pack notices APP ROOT NATIVE                 THIRD_PARTY_NOTICES.md + licenses/ in Contents/Resources
//!     mm-pack sizes APP                               the bundle's size by part, as JSON
//!
//! They replace relocate.py, trim_icu.py, notices.py and sizes.py (the repo is dropping Python) with the same
//! arguments, the same output byte for byte (JSON written the way Python's json module writes it, files walked in
//! the same order) and the same exit codes: 1 on a failure, with the reason on stderr. Each step's rules are in
//! its module. Only the standard library and serde_json (to read `cargo metadata`); the Mach-O work shells out to
//! the tools the scripts used (otool, install_name_tool, nm, clang, icupkg).

mod fsutil;
mod json;
mod notices;
mod relocate;
mod sizes;
mod trim_icu;

/// A failure that ends the program: its message goes to stderr and the exit code is 1 (what Python's
/// `SystemExit("...")` and an uncaught exception both did).
pub type Res<T> = anyhow::Result<T>;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: mm-pack relocate APP [--prefix P] | trim-icu APP | notices APP ROOT NATIVE | sizes APP";
    let (cmd, rest) = match args.split_first() {
        Some((c, r)) => (c.as_str(), r),
        None => {
            eprintln!("{usage}");
            std::process::exit(2);
        }
    };
    let need = |n: usize| {
        if rest.len() < n {
            eprintln!("{usage}");
            std::process::exit(2);
        }
    };
    let r: Res<i32> = match cmd {
        "relocate" => {
            need(1);
            relocate::run(rest)
        }
        "trim-icu" => {
            need(1);
            trim_icu::run(&rest[0]).map(|_| 0)
        }
        "notices" => {
            need(3);
            notices::run(&rest[0], &rest[1], &rest[2])
        }
        "sizes" => {
            need(1);
            sizes::run(&rest[0]).map(|_| 0)
        }
        _ => {
            eprintln!("{usage}");
            std::process::exit(2);
        }
    };
    match r {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("{e:#}");
            std::process::exit(1);
        }
    }
}
