//! Evaluation graph: nodes cached by the fingerprint of their inputs.
//!
//! MVP lesson: pure functions without a cache are expensive (the fight was 5× faster with one).
//! Two cache levels:
//!
//! - [`Memo`] — in memory, for one run: e.g. a character's figure at a world
//!   time, asked for many times per frame (camera, squash, trails);
//! - [`Store`] — on disk, across runs: expensive results (a frame) keyed by
//!   the fingerprint of everything they depend on. Editing one key recomputes
//!   only the frames whose inputs changed.
//!
//! An expensive node's fingerprint is taken from its *computed* inputs, not the description
//! ("early cutoff", as in salsa): poses and cameras are cheap and always computed;
//! if they come out the same, the frame is taken from the cache.

use mavian_core::Fingerprint;
use std::collections::HashMap;
use std::hash::Hash;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

/// How many times a node was taken from the cache and how many times computed.
#[derive(Default, Debug)]
pub struct Stats {
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl Stats {
    pub fn hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn hits(&self) -> usize {
        self.hits.load(Ordering::Relaxed)
    }

    pub fn misses(&self) -> usize {
        self.misses.load(Ordering::Relaxed)
    }
}

/// In-memory cache: key → value. Safe to use from several threads.
///
/// The value is computed outside the lock: two threads may compute the same
/// thing, but the function is pure and the result identical — determinism holds.
pub struct Memo<K, V> {
    map: Mutex<HashMap<K, V>>,
    pub stats: Stats,
}

impl<K: Eq + Hash + Clone, V: Clone> Default for Memo<K, V> {
    fn default() -> Self {
        Self { map: Mutex::new(HashMap::new()), stats: Stats::default() }
    }
}

impl<K: Eq + Hash + Clone, V: Clone> Memo<K, V> {
    pub fn get_or(&self, key: K, compute: impl FnOnce() -> V) -> V {
        if let Some(v) = self.map.lock().unwrap_or_else(PoisonError::into_inner).get(&key) {
            self.stats.hit();
            return v.clone();
        }
        self.stats.miss();
        let v = compute();
        self.map.lock().unwrap_or_else(PoisonError::into_inner).insert(key, v.clone());
        v
    }

    pub fn clear(&self) {
        self.map.lock().unwrap_or_else(PoisonError::into_inner).clear();
    }
}

/// On-disk cache: one file per fingerprint.
pub struct Store {
    dir: PathBuf,
    pub stats: Stats,
}

impl Store {
    pub fn open(dir: impl Into<PathBuf>) -> io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir, stats: Stats::default() })
    }

    pub fn path(&self, key: Fingerprint, ext: &str) -> PathBuf {
        self.dir.join(format!("{key}.{ext}"))
    }

    /// Path to a ready result, or `None` if there is none.
    pub fn find(&self, key: Fingerprint, ext: &str) -> Option<PathBuf> {
        let p = self.path(key, ext);
        if p.is_file() {
            self.stats.hit();
            Some(p)
        } else {
            self.stats.miss();
            None
        }
    }

    /// Store a result: write a temporary file first, then rename — an interrupted
    /// write never leaves a broken file under a valid name.
    pub fn put_file(&self, key: Fingerprint, ext: &str, src: &Path) -> io::Result<PathBuf> {
        let dst = self.path(key, ext);
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let tmp = self.dir.join(format!("{key}.{ext}.tmp{}-{n}", std::process::id()));
        std::fs::copy(src, &tmp)?;
        std::fs::rename(&tmp, &dst)?;
        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memo_computes_once() {
        let m: Memo<u32, u32> = Memo::default();
        let mut calls = 0;
        for _ in 0..3 {
            m.get_or(7, || {
                calls += 1;
                49
            });
        }
        assert_eq!(calls, 1);
        assert_eq!((m.stats.hits(), m.stats.misses()), (2, 1));
    }
}
