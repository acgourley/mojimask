//! A Python `set` of ints, with CPython 3.11's iteration order.
//!
//! The scan keeps each person's chains as a set and walks it to build their boxes (`obs.update(info[c]["obs"])`):
//! where two chains share a frame (one head detected twice, merged), the chain walked last wins. So the walk order is
//! part of the result, and it is CPython's hash-table order (setobject.c: open addressing, 9 linear probes then
//! perturbation, a table that grows when 3/5 full). This reproduces that table for non-negative int keys (hash = the
//! int), for the operations the scan uses: `{c}`, `add`, `|=`, `set(other)`, membership and iteration.
#[derive(Clone, Debug)]
pub struct PySet {
    table: Vec<Option<i64>>, // None: empty (the scan never discards, so no dummies)
    fill: usize,
    used: usize,
}

const LINEAR_PROBES: usize = 9;
const PERTURB_SHIFT: u32 = 5;
const MINSIZE: usize = 8;

impl Default for PySet {
    fn default() -> Self {
        PySet::new()
    }
}

impl PySet {
    pub fn new() -> PySet {
        PySet { table: vec![None; MINSIZE], fill: 0, used: 0 }
    }
    pub fn of(items: &[i64]) -> PySet {
        let mut s = PySet::new();
        for &i in items {
            s.add(i);
        }
        s
    }
    fn mask(&self) -> usize {
        self.table.len() - 1
    }
    pub fn len(&self) -> usize {
        self.used
    }
    pub fn is_empty(&self) -> bool {
        self.used == 0
    }
    pub fn contains(&self, k: i64) -> bool {
        let mask = self.mask();
        let hash = k as u64 as usize;
        let mut i = hash & mask;
        let mut perturb = hash;
        loop {
            let probes = if i + LINEAR_PROBES <= mask { LINEAR_PROBES } else { 0 };
            for j in 0..=probes {
                match self.table[i + j] {
                    None => return false,
                    Some(x) if x == k => return true,
                    _ => {}
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb)) & mask;
        }
    }
    fn insert_clean(table: &mut [Option<i64>], k: i64) {
        let mask = table.len() - 1;
        let hash = k as u64 as usize;
        let mut perturb = hash;
        let mut i = hash & mask;
        loop {
            if table[i].is_none() {
                table[i] = Some(k);
                return;
            }
            if i + LINEAR_PROBES <= mask {
                for j in 1..=LINEAR_PROBES {
                    if table[i + j].is_none() {
                        table[i + j] = Some(k);
                        return;
                    }
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb)) & mask;
        }
    }
    fn resize(&mut self, minused: usize) {
        let mut newsize = MINSIZE;
        while newsize <= minused {
            newsize <<= 1;
        }
        let old = std::mem::replace(&mut self.table, vec![None; newsize]);
        self.fill = self.used;
        for k in old.into_iter().flatten() {
            PySet::insert_clean(&mut self.table, k);
        }
    }
    /// `s.add(k)` (set_add_entry).
    pub fn add(&mut self, k: i64) {
        if self.contains(k) {
            return;
        }
        let mask = self.mask();
        let hash = k as u64 as usize;
        let mut i = hash & mask;
        let mut perturb = hash;
        'outer: loop {
            let probes = if i + LINEAR_PROBES <= mask { LINEAR_PROBES } else { 0 };
            for j in 0..=probes {
                if self.table[i + j].is_none() {
                    i += j;
                    break 'outer;
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb)) & mask;
        }
        self.table[i] = Some(k);
        self.fill += 1;
        self.used += 1;
        if self.fill * 5 >= mask * 3 {
            let n = if self.used > 50000 { self.used * 2 } else { self.used * 4 };
            self.resize(n);
        }
    }
    /// `s |= other` (set_merge).
    pub fn update(&mut self, other: &PySet) {
        if other.used == 0 {
            return;
        }
        if (self.fill + other.used) * 5 >= self.mask() * 3 {
            self.resize((self.used + other.used) * 2);
        }
        if self.fill == 0 && self.mask() == other.mask() && other.fill == other.used {
            self.table = other.table.clone();
            self.fill = other.fill;
            self.used = other.used;
            return;
        }
        if self.fill == 0 {
            self.fill = other.used;
            self.used = other.used;
            for k in other.table.iter().flatten() {
                PySet::insert_clean(&mut self.table, *k);
            }
            return;
        }
        for k in other.table.iter().flatten() {
            self.add(*k);
        }
    }
    /// `set(other)`.
    pub fn copy(&self) -> PySet {
        let mut s = PySet::new();
        s.update(self);
        s
    }
    /// Iteration in table order.
    pub fn iter(&self) -> impl Iterator<Item = i64> + '_ {
        self.table.iter().flatten().copied()
    }
    pub fn to_vec(&self) -> Vec<i64> {
        self.iter().collect()
    }
    pub fn sorted(&self) -> Vec<i64> {
        let mut v = self.to_vec();
        v.sort();
        v
    }
}
