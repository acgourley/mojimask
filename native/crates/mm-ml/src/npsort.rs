//! numpy's default sort order where results depend on it: `np.argsort(a)` (kind "quicksort") on float32 runs
//! numpy's introsort (npysort/quicksort.cpp `aquicksort_`: median-of-3 partitions, insertion sort under 16, heapsort
//! past 2*log2(n) levels). On arm64 numpy has no SIMD argsort (x86-simd-sort is x86 only), so this is what the
//! project's numpy does. Ties come out in this algorithm's order, not stably: RetinaFace's `sc.argsort()[::-1]` puts
//! equal scores in an order that then survives NMS's stable sort, so the boxes' order in dets/*.json depends on it.

#[inline]
fn less(a: f32, b: f32) -> bool {
    a < b || (b != b && a == a)
}

fn aheapsort(v: &[f32], tosort: &mut [usize]) {
    let n = tosort.len();
    if n < 2 {
        return;
    }
    // 1-based indexing as numpy's
    let a = |i: usize| i - 1;
    let mut n_ = n;
    let mut l = n_ >> 1;
    while l > 0 {
        let tmp = tosort[a(l)];
        let (mut i, mut j) = (l, l << 1);
        while j <= n_ {
            if j < n_ && less(v[tosort[a(j)]], v[tosort[a(j + 1)]]) {
                j += 1;
            }
            if less(v[tmp], v[tosort[a(j)]]) {
                tosort[a(i)] = tosort[a(j)];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        tosort[a(i)] = tmp;
        l -= 1;
    }
    while n_ > 1 {
        let tmp = tosort[a(n_)];
        tosort[a(n_)] = tosort[a(1)];
        n_ -= 1;
        let (mut i, mut j) = (1, 2);
        while j <= n_ {
            if j < n_ && less(v[tosort[a(j)]], v[tosort[a(j + 1)]]) {
                j += 1;
            }
            if less(v[tmp], v[tosort[a(j)]]) {
                tosort[a(i)] = tosort[a(j)];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        tosort[a(i)] = tmp;
    }
}

/// `np.argsort(v)` for float32 (the default kind).
pub fn argsort_f32(v: &[f32]) -> Vec<usize> {
    let num = v.len();
    let mut t: Vec<usize> = (0..num).collect();
    if num < 2 {
        return t;
    }
    const SMALL: isize = 15;
    let mut stack: Vec<(isize, isize, i32)> = Vec::new();
    let (mut pl, mut pr) = (0isize, num as isize - 1);
    let mut cdepth: i32 = (usize::BITS - 1 - num.leading_zeros()) as i32 * 2;
    let sw = |t: &mut Vec<usize>, a: isize, b: isize| t.swap(a as usize, b as usize);
    loop {
        let mut heaped = false;
        if cdepth < 0 {
            aheapsort(v, &mut t[pl as usize..=pr as usize]);
            heaped = true;
        } else {
            while pr - pl > SMALL {
                let pm = pl + ((pr - pl) >> 1);
                if less(v[t[pm as usize]], v[t[pl as usize]]) {
                    sw(&mut t, pm, pl);
                }
                if less(v[t[pr as usize]], v[t[pm as usize]]) {
                    sw(&mut t, pr, pm);
                }
                if less(v[t[pm as usize]], v[t[pl as usize]]) {
                    sw(&mut t, pm, pl);
                }
                let vp = v[t[pm as usize]];
                let mut pi = pl;
                let mut pj = pr - 1;
                sw(&mut t, pm, pj);
                loop {
                    loop {
                        pi += 1;
                        if !less(v[t[pi as usize]], vp) {
                            break;
                        }
                    }
                    loop {
                        pj -= 1;
                        if !less(vp, v[t[pj as usize]]) {
                            break;
                        }
                    }
                    if pi >= pj {
                        break;
                    }
                    sw(&mut t, pi, pj);
                }
                let pk = pr - 1;
                sw(&mut t, pi, pk);
                cdepth -= 1;
                if pi - pl < pr - pi {
                    stack.push((pi + 1, pr, cdepth));
                    pr = pi - 1;
                } else {
                    stack.push((pl, pi - 1, cdepth));
                    pl = pi + 1;
                }
            }
        }
        if !heaped {
            let mut pi = pl + 1;
            while pi <= pr {
                let vi = t[pi as usize];
                let vpv = v[vi];
                let mut pj = pi;
                let mut pk = pi - 1;
                while pj > pl && less(vpv, v[t[pk as usize]]) {
                    t[pj as usize] = t[pk as usize];
                    pj -= 1;
                    pk -= 1;
                }
                t[pj as usize] = vi;
                pi += 1;
            }
        }
        match stack.pop() {
            None => break,
            Some((l, r, d)) => {
                pl = l;
                pr = r;
                cdepth = d;
            }
        }
    }
    t
}
