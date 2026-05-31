---
title: PreparedMol zstd timings
---

These timings measure optional zstd compression for serialized
`PreparedMol` payloads. The benchmark compresses the same 1024 prepared
molecules at zstd levels 1 through 19, once without a dictionary and once with
the shipped PreparedMol dictionary.

Use these plots to choose a compression level for bulk storage. Lower
compression ratio is smaller output; lower time is faster. The benchmark is
indicative for this sample and machine, not a universal storage study.

<style>
figure.timing-plot {
  margin: 1.5rem 0;
}
figure.timing-plot img {
  max-width: 100%;
  height: auto;
}
figure.timing-plot figcaption {
  margin-bottom: 0.25rem;
  overflow-x: auto;
  font-size: 0.9rem;
}
</style>

## Benchmark setup

| Field | Value |
| --- | --- |
| Recorded | 2026-05-31 13:58 UTC |
| Runtime | Python 3.12.13, RDKit 2026.03.1, zstandard 0.25.0, zstd 1.5.7 |
| Platform | Linux-6.17.0-23-generic-x86_64-with-glibc2.36 |
| CPU | AMD Ryzen 5 7640U w/ Radeon 760M Graphics; 12 logical CPUs visible |
| Memory limit | 2 GiB |
| Container | `compose/timings-prepared-mol-zstd.yml` `timings-prepared-mol-zstd` service, network disabled |
| Sample | 1024 molecules, `random-parseable-preparable-v1`, seed `20260531` |
| Raw payload bytes | 5,742,205 |
| Dictionary artifact | `20260531_40762836` |
| Dictionary ID | `1421770218` |
| Dictionary SHA-256 | `90acacfeb725...` |

Raw data: [`docs/timings-prepared-mol-zstd.tsv`](timings-prepared-mol-zstd.tsv).

The plotted points are zstd levels. Error bars show standard deviation across
the timing repeats recorded in the TSV.

## Compression time

<figure class="timing-plot">
  <figcaption>Compression ratio versus compression time:</figcaption>
  <img src="timings-prepared-mol-zstd-plots/compression-ratio-vs-compression-time.png" alt="PreparedMol zstd compression ratio versus compression time">
</figure>

With the shipped dictionary, the smallest observed output in this run was
about 4.8% of raw bytes at level 18. Without the dictionary, the smallest
observed output was about 9.5% of raw bytes at level 19.

## Decompression time

<figure class="timing-plot">
  <figcaption>Compression ratio versus decompression time:</figcaption>
  <img src="timings-prepared-mol-zstd-plots/compression-ratio-vs-decompression-time.png" alt="PreparedMol zstd compression ratio versus decompression time">
</figure>

Decompression is much flatter than compression across levels on this sample.
The dictionary mode is also faster to decompress here because it writes fewer
bytes and uses the matching trained dictionary.
