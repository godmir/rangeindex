mod rangeindex {
    use std::cmp::{Ordering, max, min};
    //use std::ops::Range;

    use wasm_bindgen::prelude::*;
    //use wasm_bindgen::convert::RefFromWasmAbi;

    #[wasm_bindgen]
    #[derive(Debug, Clone, Copy)]
    pub struct Interval { // can't use Range as it needs RefFromWasmAbi,
                          // and I can't add it outside of the Range's crate
        start: usize,
        end: usize
    }
    
    #[wasm_bindgen]
    #[derive(Debug, Clone)]
    pub struct RangeIndex {
        chunks: Vec<Interval>,
        offsets: Vec<usize>
    }

    #[wasm_bindgen]
    impl Interval {

        #[wasm_bindgen(constructor)]
        pub fn new(start: usize, end: usize) -> Interval {
            if start < end {
                Interval{ start, end }
            } else {
                Interval{ start: 0, end: 0 }
            }
        }

        pub fn len(&self) -> usize {
            self.end - self.start
        }

        pub fn is_empty(&self) -> bool {
            self.end == self.start
        }

        pub fn intersect(x: &Interval, y: &Interval) -> Interval {
            Interval::new(max(x.start, y.start), min(x.end, y.end))
        }
    }
    
    #[wasm_bindgen]
    impl RangeIndex {

        #[wasm_bindgen(constructor)]
        pub fn new(start: usize, end: usize) -> RangeIndex {
            let mut ri = RangeIndex{ chunks: Vec::new(), offsets: Vec::new() };
            ri.chunks.push(Interval::new(start, end));
            ri.offsets.push(0);
            ri
        }

        #[cfg(test)]
        pub fn extend(&mut self, r: &Interval, offset: usize) {
            self.chunks.push(r.clone());
            self.offsets.push(offset);
        }
        
        pub fn num_chunks(&self) -> usize {
            self.chunks.len()
        }
        
        pub fn len(&self) -> usize {
            let last = self.offsets.len() - 1;
            self.offsets[last] + self.chunks[last].end - self.chunks[last].start
        }

        fn _find_chunk(&self, idx: usize) -> Option<usize> {
            let mut lo: usize = 0;
            let mut hi: usize = self.offsets.len();
            let mut mid = (hi + lo)/2;
            
            while lo < hi {
                mid = (hi + lo) / 2;
                match self.offsets[mid].cmp(&idx) {
                    Ordering::Equal   => return Some(mid),
                    Ordering::Greater => hi = mid,
                    Ordering::Less    => lo = mid + 1
                }
            }
            if self.offsets[mid] <= idx && idx < self.offsets[mid] + &self.chunks[mid].len() {
                return Some(mid);
            } else if self.offsets[hi] <= idx && idx < self.offsets[hi] + &self.chunks[hi].len() {
                return Some(hi);
            }
            None
        }
        
        pub fn at(&self, idx: usize) -> usize {
            match self._find_chunk(idx) {
                Some(chidx) => return self.chunks[chidx].start + idx - self.offsets[chidx],
                None        => panic!("Index {} is out of bounds for a RangeIndex of size {}", idx, self.len())
            }
        }

        pub fn materialize(&self) -> Vec<usize> {
            let mut res = Vec::with_capacity(self.len());
            for i in 0..self.chunks.len() {
                let sz = res.len() + &self.chunks[i].len();
                let mut k = self.chunks[i].start;
                res.resize_with(sz, || { k += 1; k - 1 })
            }
            res
        }

        #[inline]
        fn advance(ptr: &(usize, usize)) -> (usize, usize) {
            (ptr.0 + ptr.1, 1 - ptr.1)
        }

        #[inline]
        fn point(&self, ptr: &(usize, usize)) -> usize {
            // this is a fast convenience function; it will not check boundaries
            if ptr.1 == 0 {
                self.chunks[ptr.0].start
            } else {
                self.chunks[ptr.0].end
            }
        }
        
        pub fn remove(&mut self, other: &RangeIndex) {
            if self.chunks[self.chunks.len()-1].end <= other.chunks[0].start
                || self.chunks[0].start >= other.chunks[other.chunks.len()-1].end {
                    return;
                }
            let mut i_src = (0_usize, 0_usize);
            let mut i_tgt = (0_usize, 0_usize);
            let mut new_chunks: Vec<Interval> = Vec::new();
            let mut new_offsets: Vec<usize> = Vec::new();
            let n_tgt = self.num_chunks();
            let n_src = other.num_chunks();
            let mut offset = 0;
            let mut pts_tgt = ((0, 1), (self.point(&i_tgt), i_tgt.1));
            let mut pts_src = ((0, 1), (other.point(&i_src), i_src.1));
            if pts_tgt.1.0 == 0 {
                i_tgt = RangeIndex::advance(&i_tgt);
                pts_tgt = (pts_tgt.1, (self.point(&i_tgt), i_tgt.1));
            }
            if pts_src.1.0 == 0 {
                i_src = RangeIndex::advance(&i_src);
                pts_src = (pts_src.1, (other.point(&i_src), i_src.1));
            }
            if pts_tgt.0.1 == 0 && pts_tgt.0.0 < pts_src.0.0 {
                let new = Interval::new(pts_tgt.0.0, min(pts_tgt.1.0, pts_src.0.0));
                new_chunks.push(new);
                new_offsets.push(offset);
                offset += new.len();
                i_tgt = RangeIndex::advance(&i_tgt);
                pts_tgt = (pts_tgt.1, (self.point(&i_tgt), i_tgt.1));
            }
            loop {
                let from_state = (pts_src.0.1<<1) + pts_tgt.0.1;
                let from_point = max(pts_src.0.0, pts_tgt.0.0);
                let mut adv_which = 0;
                let mut to_point: usize = from_point;
                if pts_tgt.1.0 <= pts_src.1.0 {
                    to_point = pts_tgt.1.0;
                    adv_which += 1;
                }
                if pts_tgt.1.0 >= pts_src.1.0 {
                    to_point = pts_src.1.0;
                    adv_which += 2;
                }
                if from_state == 2 {
                    let new = Interval::new(from_point, to_point);
                    new_chunks.push(new);
                    new_offsets.push(offset);
                    offset += new.len();
                }
                if (adv_which & 1) == 1 {
                    i_tgt = RangeIndex::advance(&i_tgt);
                    if i_tgt.0 == n_tgt {
                        break;
                    }
                    pts_tgt = (pts_tgt.1, (self.point(&i_tgt), i_tgt.1));
                }
                if (adv_which & 2) == 2 {
                    i_src = RangeIndex::advance(&i_src);
                    if i_src.0 < n_src {
                        pts_src = (pts_src.1, (other.point(&i_src), i_src.1));
                    } else {
                        pts_src = (pts_src.1, (usize::MAX, 0)); // we may need to add more chunks from tgt
                    }
                }
            }
            if new_chunks.is_empty() {
                self.chunks.truncate(1);
                self.chunks[0] = Interval::new(0, 0);
                self.offsets.truncate(1);
                self.offsets[0] = 0;
            } else {
                self.chunks = new_chunks;
                self.offsets = new_offsets;
            }
        }

        pub fn add(&mut self, other: &RangeIndex) {
            if self.len() == 0 {
                self.chunks = other.chunks.clone();
                self.offsets = other.offsets.clone();
            }
            let n_src = other.num_chunks();
            let n_tgt = self.num_chunks();
            let mut new_chunks: Vec<Interval> = Vec::new();
            let mut new_offsets: Vec<usize> = Vec::new();
            let mut offset = 0_usize;
            let mut i_src = 0_usize;
            let mut i_tgt = 0_usize;
            let mut new_item: Option<Interval> = None;
            while i_src < n_src && i_tgt < n_tgt {
                let ch_src = &other.chunks[i_src];
                let ch_tgt = &self.chunks[i_tgt];
                let mut append = false;
                if new_item.is_none() {
                    if max(ch_tgt.start, ch_src.start) < min(ch_tgt.end, ch_src.end) {
                        new_item = Some(Interval::new(min(ch_tgt.start, ch_src.start),
                                                      max(ch_tgt.end, ch_src.end)));
                        i_src += 1;
                        i_tgt += 1;
                    } else if ch_tgt.start < ch_src.start { // no overlap, "tgt" first
                        new_item = Some(ch_tgt.clone());
                        i_tgt += 1;
                    } else { // no overlap, "src" first
                        new_item = Some(ch_src.clone());
                        i_src += 1;
                    }
                } else if ch_tgt.start < ch_src.start {
                    let mut new_chunk = new_item.unwrap(); // no plain "get" alas
                    if ch_tgt.start <= new_chunk.end {
                        new_chunk.end = max(new_chunk.end, ch_tgt.end);
                        i_tgt += 1;
                    } else {
                        append = true;
                    }
                    new_item = Some(new_chunk);
                } else {
                    let mut new_chunk = new_item.unwrap();
                    if ch_src.start <= new_chunk.end {
                        new_chunk.end = max(new_chunk.end, ch_src.end);
                        i_src += 1;
                    } else {
                        append = true;
                    }
                    if ch_tgt.start <= new_chunk.end {
                        new_chunk.end = max(new_chunk.end, ch_tgt.end);
                        i_tgt += 1;
                    }
                    new_item = Some(new_chunk);
                }
                if append {
                    let new_chunk = new_item.unwrap();
                    new_chunks.push(new_chunk);
                    new_offsets.push(offset);
                    offset += new_chunk.len();
                    new_item = None;
                }
            }
            if new_item.is_some() {
                let mut new_chunk = new_item.unwrap();
                if i_tgt < n_tgt && new_chunk.end >= self.chunks[i_tgt].start {
                    new_chunk.end = max(new_chunk.end, self.chunks[i_tgt].end);
                    i_tgt += 1;
                } else if i_src < n_src && new_chunk.end >= other.chunks[i_src].start {
                    new_chunk.end = max(new_chunk.end, other.chunks[i_src].end);
                    i_src += 1;
                }
                new_chunks.push(new_chunk);
                new_offsets.push(offset);
                offset += new_chunk.len();
            }
            if i_tgt < n_tgt {
                new_chunks.extend_from_slice(&self.chunks[i_tgt..]);
                let shift = offset - self.offsets[i_tgt];
                new_offsets.resize_with(new_offsets.len() + (n_tgt - i_tgt),
                                        || { i_tgt += 1; self.offsets[i_tgt-1] + shift });
            }
            if i_src < n_src {
                new_chunks.extend_from_slice(&other.chunks[i_src..]);
                let shift = offset - other.offsets[i_src];
                new_offsets.resize_with(new_offsets.len() + (n_src - i_src),
                                        || { i_src += 1; other.offsets[i_src-1] + shift });
            }
            // Moving
            self.chunks = new_chunks;
            self.offsets = new_offsets;
        }
    }
}

#[cfg(test)]
mod test_rangeindex {

    use crate::rangeindex::rangeindex::{Interval, RangeIndex};
    
    fn vec_from_ranges(ranges: &[(usize, usize)]) -> Vec<usize> {
        let mut v = Vec::new();
        for rg in ranges {
            let mut i = rg.0;
            v.resize_with(v.len() + rg.1 - rg.0, || { i += 1; i-1 });
        }
        v
    }

    #[test]
    fn test_interval() {
        let lower0 = 10;
        let upper0 = 20;
        let i0 = Interval::new(lower0, upper0);
        assert_eq!(i0.len(), upper0 - lower0);

        let lower1 = 15;
        let upper1 = 18;
        let i1 = Interval::new(lower1, upper1);
        let x1 = Interval::intersect(&i0, &i1);
        assert_eq!(x1.len(), i1.len());

        let lower2 = 5;
        let upper2 = 28;
        let i2 = Interval::new(lower2, upper2);
        let x2 = Interval::intersect(&i0, &i2);
        assert_eq!(x2.len(), i0.len());

        let lower3 = 5;
        let upper3 = 18;
        let i3 = Interval::new(lower3, upper3);
        let x3 = Interval::intersect(&i0, &i3);
        assert_eq!(x3.len(), upper3 - lower0);

        let lower4 = 15;
        let upper4 = 28;
        let i4 = Interval::new(lower4, upper4);
        let x4 = Interval::intersect(&i0, &i4);
        assert_eq!(x4.len(), upper0 - lower4);

        let lower5 = 5;
        let upper5 = 28;
        let i5 = Interval::new(lower5, upper5);
        let x5 = Interval::intersect(&i0, &i5);
        assert_eq!(x5.len(), i0.len());

        let lower6 = 25;
        let upper6 = 28;
        let i6 = Interval::new(lower6, upper6);
        let x6 = Interval::intersect(&i0, &i6);
        assert_eq!(x6.len(), 0);
    }

    #[test]
    pub fn test_rangeindex_basic() {
        let ri = RangeIndex::new(0, 10);
        assert_eq!(ri.at(8), 8);
        assert_eq!(ri.len(), 10);
        assert_eq!(ri.materialize(), vec_from_ranges(&[(0, 10)]));
    }

    #[test]
    pub fn test_remove_simple() {
        let mut ri = RangeIndex::new(0, 10);
        assert_eq!(ri.materialize(), vec_from_ranges(&[(0, 10)]));

        ri.remove(&RangeIndex::new(3, 6));
        assert_eq!(ri.materialize(), vec![0, 1, 2, 6, 7, 8, 9]);
        assert_eq!(ri.len(), 7);
        assert_eq!(ri.num_chunks(), 2);

        ri.remove(&RangeIndex::new(2, 5));
        assert_eq!(ri.materialize(), vec![0, 1, 6, 7, 8, 9]);
        assert_eq!(ri.len(), 6);
        assert_eq!(ri.num_chunks(), 2);

        ri.remove(&RangeIndex::new(8, 12));
        assert_eq!(ri.materialize(), vec![0, 1, 6, 7]);
        assert_eq!(ri.len(), 4);
        assert_eq!(ri.num_chunks(), 2);

        ri.remove(&RangeIndex::new(0, 20));
        assert!(ri.materialize().is_empty());
        assert_eq!(ri.len(), 0);
        assert_eq!(ri.num_chunks(), 1);
    }

    #[test]
    pub fn test_add_simple() {
        let mut ri = RangeIndex::new(0, 0);
        ri.add(&RangeIndex::new(0, 10));
        let mut expected = vec_from_ranges(&[(0, 10)]);
        assert_eq!(ri.materialize(), expected);
        assert_eq!(ri.len(), 10);
        assert_eq!(ri.num_chunks(), 1);
        
        ri.add(&RangeIndex::new(20, 30));
        expected = vec_from_ranges(&[(0, 10), (20, 30)]);
        assert_eq!(ri.materialize(), expected);
        assert_eq!(ri.len(), 20);
        assert_eq!(ri.num_chunks(), 2);

        ri.add(&RangeIndex::new(15, 25));
        expected = vec_from_ranges(&[(0, 10), (15, 30)]);
        assert_eq!(ri.materialize(), expected);
        assert_eq!(ri.len(), 25);
        assert_eq!(ri.num_chunks(), 2);

        ri.add(&RangeIndex::new(13, 15));
        expected = vec_from_ranges(&[(0, 10), (13, 30)]);
        assert_eq!(ri.materialize(), expected);
        assert_eq!(ri.len(), 27);
        assert_eq!(ri.num_chunks(), 2);

        ri.add(&RangeIndex::new(10, 13));
        expected = vec_from_ranges(&[(0, 30)]);
        assert_eq!(ri.materialize(), expected);
        assert_eq!(ri.len(), 30);
        assert_eq!(ri.num_chunks(), 1);
    }

    #[test]
    pub fn test_complex() {
        let mut ri = RangeIndex::new(0, 30);
        ri.add(&RangeIndex::new(30, 100));
        let mut expected = vec_from_ranges(&[(0, 100)]);
        assert_eq!(ri.materialize(), expected);
        assert_eq!(ri.len(), 100);
        assert_eq!(ri.num_chunks(), 1);

        let mut s = RangeIndex::new(10, 20);
        s.add(&RangeIndex::new(40, 50));
        s.add(&RangeIndex::new(80, 90));
        expected = vec_from_ranges(&[(10, 20), (40, 50), (80, 90)]);
        assert_eq!(s.materialize(), expected);
        assert_eq!(s.len(), 30);
        assert_eq!(s.num_chunks(), 3);

        ri.remove(&s);
        expected = vec_from_ranges(&[(0, 10), (20, 40), (50, 80), (90, 100)]);
        assert_eq!(ri.materialize(), expected);
        assert_eq!(ri.len(), 70);
        assert_eq!(ri.num_chunks(), 4);

        let mut u = RangeIndex::new(14, 16);
        u.add(&RangeIndex::new(44, 46));
        u.add(&RangeIndex::new(84, 87));
        expected = vec_from_ranges(&[(14, 16), (44, 46), (84, 87)]);
        assert_eq!(u.materialize(), expected);
        assert_eq!(u.len(), 7);
        assert_eq!(u.num_chunks(), 3);

        ri.remove(&u);
        expected = vec_from_ranges(&[(0, 10), (20, 40), (50, 80), (90, 100)]);
        assert_eq!(ri.materialize(), expected);
        assert_eq!(ri.len(), 70);
        assert_eq!(ri.num_chunks(), 4);

        ri.add(&u);
        expected = vec_from_ranges(&[(0, 10), (14, 16), (20, 40), (44, 46), (50, 80), (84, 87), (90, 100)]);
        assert_eq!(ri.materialize(), expected);
        assert_eq!(ri.len(), 77);
        assert_eq!(ri.num_chunks(), 7);

        s.add(&u);
        expected = vec_from_ranges(&[(10, 20), (40, 50), (80, 90)]);
        assert_eq!(s.materialize(), expected);
        assert_eq!(s.len(), 30);
        assert_eq!(s.num_chunks(), 3);

        assert_eq!(ri.clone().materialize(), ri.materialize());
        
        ri.remove(&u);
        expected = vec_from_ranges(&[(0, 10), (20, 40), (50, 80), (90, 100)]);
        assert_eq!(ri.materialize(), expected);
        assert_eq!(ri.len(), 70);
        assert_eq!(ri.num_chunks(), 4);

        let mut w = RangeIndex::new(15, 55);
        w.add(&RangeIndex::new(75, 95));
        ri.remove(&w);
        expected = vec_from_ranges(&[(0, 10), (55, 75), (95, 100)]);
        assert_eq!(ri.materialize(), expected);
        assert_eq!(ri.len(), 35);
        assert_eq!(ri.num_chunks(), 3);

        let mut z = RangeIndex::new(14, 16);
        z.add(&RangeIndex::new(44, 46));
        z.add(&RangeIndex::new(70, 82));
        z.add(&RangeIndex::new(86, 97));    
        ri.remove(&z);
        expected = vec_from_ranges(&[(0, 10), (55, 70), (97, 100)]);
        assert_eq!(ri.materialize(), expected);
        assert_eq!(ri.len(), 28);
        assert_eq!(ri.num_chunks(), 3);
    }

    #[test]
    pub fn test_performance() {
        
        use std::time::Instant; //, Duration};

        fn gen_random_range(x0: usize, rg: usize, n: usize) -> RangeIndex {
            use rand::{thread_rng, Rng};
            let mut stops: Vec<usize> = Vec::with_capacity(2*n);
            let mut rng = thread_rng();
            stops.resize_with(2*n, || rng.gen_range(x0..x0+rg));
            stops.sort();
            let mut i = 1_usize;
            while i < stops.len()-2 {
                if stops[i+1] == stops[i] + 1 {
                    if stops[i+2] == stops[i+1] + 1 {
                        if stops[i-1] + 1 < stops[i] {
                            stops[i] -= 1;
                        } else {
                            println!("Unable to adjust: {}, {}, {}, {}, {}", i, stops[i-1], stops[i], stops[i+1], stops[i+2]);
                        }
                    } else {
                        stops[i+1] += 1;
                    }
                }
                i += 2;
            }
            let mut ri = RangeIndex::new(stops[0], stops[1]);
            let mut offset = stops[1] - stops[0];
            i = 2;
            while i < 2*n {
                let chunk = Interval::new(stops[i], stops[i+1]);
                ri.extend(&chunk, offset);
                offset += chunk.len();
                i += 2;
            }
            ri
        }

        let mut ri = RangeIndex::new(0, 10_000_000);

        let dt0 = 100_u128; // for release version; for debug it's 5x slower
        let test_data = [(2_000_000,  1_000_000,   1_000, "remove",     dt0),
                         (2_000_000,  1_000_000,   1_000, "add",        dt0),
                         (1_500_000,  1_000_000,   1_000, "remove",     dt0),
                         (1_750_000,  1_000_000,   1_000, "add",        dt0),
                         (0,         10_000_000, 100_000, "remove",  100*dt0),
                         (0,         10_000_000, 100_000, "add",     100*dt0),
                         (0,         10_000_000, 100_000, "remove",  100*dt0),
                         (0,         10_000_000, 100_000, "add",     100*dt0),
                         ];

        for &test_case in &test_data[..] {
            let mut now = Instant::now();
            let s = gen_random_range(test_case.0, test_case.1, test_case.2);
            let dt_gen = now.elapsed().as_micros();
            now = Instant::now();
            if test_case.3 == "remove" {
                ri.remove(&s);
            } else {
                ri.add(&s);
            }
            let dt_run = now.elapsed().as_micros();
            eprintln!("{} generated in {} musec", test_case.2, dt_gen);
            eprintln!("{} {} in {} musec, {} chunks", test_case.3, test_case.2, dt_run, ri.num_chunks());
            assert!(dt_run < test_case.4);
        }
    }
}

