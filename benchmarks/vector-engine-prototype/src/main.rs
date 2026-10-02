use rusqlite::{Connection, OpenFlags};
use serde_json::json;
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{BufWriter, Write},
    time::Instant,
};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const DIM: usize = 384;
fn options() -> IndexOptions {
    IndexOptions {
        dimensions: DIM,
        metric: MetricKind::IP,
        quantization: ScalarKind::F32,
        connectivity: 32,
        expansion_add: 128,
        expansion_search: 128,
        multi: false,
    }
}
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn score(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}
fn commit_map(prefix: &str) -> Result<Vec<u32>> {
    Ok(fs::read(format!("{prefix}.commits"))?
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn ranked(best: HashMap<u32, (u64, f64)>, limit: usize) -> Vec<(u32, u64, f64)> {
    let mut v: Vec<_> = best.into_iter().map(|(c, (k, s))| (c, k, s)).collect();
    v.sort_unstable_by(|a, b| b.2.total_cmp(&a.2).then(a.1.cmp(&b.1)));
    v.truncate(limit);
    v
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let command = args.get(1).ok_or("command required")?;
    let prefix = args.get(2).ok_or("prefix required")?;
    if command == "build" {
        let start = Instant::now();
        let db = Connection::open_with_flags(
            args.get(3).ok_or("DB required")?,
            OpenFlags::SQLITE_OPEN_READ_ONLY,
        )?;
        let generation: i64 =
            db.query_row("SELECT id FROM generations WHERE state='active'", [], |r| {
                r.get(0)
            })?;
        let mut statement=db.prepare("SELECT d.commit_oid,e.vector FROM embeddings e JOIN documents d ON d.id=e.document_id WHERE e.generation_id=? ORDER BY e.document_id,e.chunk_index")?;
        let mut rows = statement.query([generation])?;
        let mut data = Vec::new();
        let mut commits = Vec::new();
        let mut ids = HashMap::new();
        while let Some(r) = rows.next()? {
            let commit: String = r.get(0)?;
            let blob: Vec<u8> = r.get(1)?;
            assert_eq!(blob.len(), DIM * 4);
            let n = ids.len() as u32;
            let id = *ids.entry(commit).or_insert(n);
            commits.push(id);
            data.extend(floats(&blob));
        }
        let read_seconds = start.elapsed().as_secs_f64();
        let index = Index::new(&options())?;
        index.reserve_capacity_and_threads(commits.len(), 4)?;
        let build = Instant::now();
        let n = commits.len();
        let threads = 4usize;
        std::thread::scope(|scope| {
            for t in 0..threads {
                let data = &data;
                let index = &index;
                scope.spawn(move || {
                    for key in (t..n).step_by(threads) {
                        index
                            .add(key as u64, &data[key * DIM..(key + 1) * DIM])
                            .unwrap();
                        if t == 0 && key % 10000 == 0 {
                            eprintln!("indexed approximately {key}/{n}");
                        }
                    }
                });
            }
        });
        let build_seconds = build.elapsed().as_secs_f64();
        index.save(&format!("{prefix}.usearch"))?;
        let mut out = BufWriter::new(fs::File::create(format!("{prefix}.vectors"))?);
        for value in data {
            out.write_all(&value.to_le_bytes())?;
        }
        out.flush()?;
        let mut out = BufWriter::new(fs::File::create(format!("{prefix}.commits"))?);
        for value in commits {
            out.write_all(&value.to_le_bytes())?;
        }
        out.flush()?;
        let manifest = json!({"engine":"usearch","version":"2.26.2","scalar":"f32","dimensions":DIM,"vectors":n,"commits":ids.len(),"read_seconds":read_seconds,"build_seconds":build_seconds,"total_seconds":start.elapsed().as_secs_f64(),"threads":threads,"connectivity":32,"expansion_add":128,"index_bytes":fs::metadata(format!("{prefix}.usearch"))?.len(),"hardware_acceleration":index.hardware_acceleration(),"purpose":"engine-only ANN recall and latency; no end-to-end CLI or retrieval quality claim"});
        fs::write(
            format!("{prefix}.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        println!("{manifest}");
    } else if command == "exact" || command == "query" {
        let start = Instant::now();
        let commits = commit_map(prefix)?;
        let queries = floats(&fs::read(args.get(3).ok_or("query f32 file required")?)?);
        let expansion: usize = args.get(4).ok_or("expansion")?.parse()?;
        let divisor: u32 = args.get(5).ok_or("scope divisor")?.parse()?;
        let index = Index::new(&options())?;
        if command == "query" {
            index.view(&format!("{prefix}.usearch"))?;
            index.change_expansion_search(expansion);
        }
        let vectors = if command == "exact" {
            floats(&fs::read(format!("{prefix}.vectors"))?)
        } else {
            Vec::new()
        };
        let startup = start.elapsed().as_secs_f64();
        let eligible: HashSet<_> = commits
            .iter()
            .copied()
            .filter(|c| c % divisor == 0)
            .collect();
        let limit = 100usize.min(eligible.len());
        let mut results = Vec::new();
        for (qi, query) in queries.chunks_exact(DIM).enumerate() {
            let search = Instant::now();
            let mut best = HashMap::new();
            let mut fetched = 0;
            let mut calls = 0;
            if command == "exact" {
                for (key, vector) in vectors.chunks_exact(DIM).enumerate() {
                    let c = commits[key];
                    if c % divisor != 0 {
                        continue;
                    }
                    let s = score(query, vector);
                    let entry = best.entry(c).or_insert((key as u64, s));
                    if s > entry.1 {
                        *entry = (key as u64, s);
                    }
                }
            } else {
                let mut count = limit * 4;
                loop {
                    calls += 1;
                    let hits = index.filtered_search(query, count, |key| {
                        commits[key as usize] % divisor == 0
                    })?;
                    fetched = hits.keys.len();
                    for key in hits.keys {
                        let c = commits[key as usize];
                        let mut vector = vec![0f32; DIM];
                        assert_eq!(index.get(key, &mut vector)?, 1);
                        let s = score(query, &vector);
                        let entry = best.entry(c).or_insert((key, s));
                        if s > entry.1 || (s == entry.1 && key < entry.0) {
                            *entry = (key, s);
                        }
                    }
                    if best.len() >= limit || fetched < count || count >= commits.len() {
                        break;
                    }
                    count = (count * 2).min(commits.len());
                }
            }
            let hits = ranked(best, limit);
            results.push(json!({"query":qi,"seconds":search.elapsed().as_secs_f64(),"commits":hits.iter().map(|x|x.0).collect::<Vec<_>>(),"vector_keys":hits.iter().map(|x|x.1).collect::<Vec<_>>(),"scores":hits.iter().map(|x|x.2).collect::<Vec<_>>(),"fetched":fetched,"calls":calls}));
        }
        println!(
            "{}",
            json!({"command":command,"startup_seconds":startup,"total_seconds":start.elapsed().as_secs_f64(),"expansion":expansion,"scope_divisor":divisor,"eligible_commits":eligible.len(),"results":results})
        );
    } else {
        return Err("unknown command".into());
    }
    Ok(())
}
