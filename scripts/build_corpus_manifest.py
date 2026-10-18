#!/usr/bin/env python3
"""Build a deterministic manifest for the local SC paper corpus.

The paper PDFs are intentionally ignored by Git.  This script records only
reproducible metadata that is safe to keep in the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import OrderedDict
from pathlib import Path


PAPERS = OrderedDict(
    [
        ("survey", ("综述", "Alaghi_Qian_Hayes_The_Promise_and_Challenge_of_Stochastic_Computing.pdf", "The Promise and Challenge of Stochastic Computing")),
        ("textbook", ("教材", "Stochastic Computing.pdf", "Stochastic Computing: Techniques and Applications")),
        ("iccd2013", ("ICCD2013", "ICCD_2013-2.pdf", "Exploiting Correlation in Stochastic Circuit Design")),
        ("jenson-dsc", ("Jenson-DSC", "A_deterministic_approach_to_stochastic_computation.pdf", "A Deterministic Approach to Stochastic Computation")),
        ("najafi-dsc", ("Najafi-DSC", "Performing_Stochastic_Computation_Deterministically.pdf", "Performing Stochastic Computation Deterministically")),
        ("cang", ("CANG", "Context-Aware_Number_Generator_for_Deterministic_Bit-stream_Computing.pdf", "Context-Aware Number Generator for Deterministic Bit-stream Computing")),
        ("downsample", ("DownSample", "High_Quality_Down-Sampling_for_Deterministic_Approaches_to_Stochastic_Computing.pdf", "High Quality Down-Sampling for Deterministic Approaches to Stochastic Computing")),
        ("sobol", ("Sobol", "Energy_efficient_stochastic_computing_with_Sobol_sequences.pdf", "Energy Efficient Stochastic Computing with Sobol Sequences")),
        ("psobol", ("PSobol", "Toward_Energy-Efficient_Stochastic_Circuits_Using_Parallel_Sobol_Sequences.pdf", "Toward Energy-Efficient Stochastic Circuits Using Parallel Sobol Sequences")),
        ("p2lsg", ("P2LSG", "P2LSG_Powers-of-2_Low-Discrepancy_Sequence_Generator_for_Stochastic_Computing.pdf", "P2LSG: Powers-of-2 Low-Discrepancy Sequence Generator for Stochastic Computing")),
        ("fsm-ldsg", ("FSM-LDSG", "A_Low-Cost_FSM-based_Bit-Stream_Generator_for_Low-Discrepancy_Stochastic_Computing.pdf", "A Low-Cost FSM-based Bit-Stream Generator for Low-Discrepancy Stochastic Computing")),
        ("true-bsg", ("TRUE-BSG", "TRUE-BSG_A_True_Random_Bit-Stream_Generator_for_Fast_and_Efficient_Stochastic_Computing.pdf", "TRUE-BSG: A True Random Bit-Stream Generator for Fast and Efficient Stochastic Computing")),
        ("usd-rns", ("USD-RNS", "Low-Cost_High-Accuracy_Random_Number_Source_Design_for_Stochastic_Computing_via_Exploitation_of_Uniform_Spatial_Distribution.pdf", "Low-Cost High-Accuracy Random Number Source Design for Stochastic Computing via Exploitation of Uniform Spatial Distribution")),
        ("bitcorr", ("BitCorr", "Effect_of_bit-level_correlation_in_stochastic_computing.pdf", "Effect of Bit-Level Correlation in Stochastic Computing")),
        ("corrmat", ("CorrMat", "Analyzing_Multilevel_Stochastic_Circuits_using_Correlation_Matrices.pdf", "Analyzing Multilevel Stochastic Circuits Using Correlation Matrices")),
        ("corrmanip", ("CorrManip", "Correlation_manipulating_circuits_for_stochastic_computing.pdf", "Correlation Manipulating Circuits for Stochastic Computing")),
        ("contingency", ("Contingency", "Utilization_of_Contingency_Tables_in_Stochastic_Computing.pdf", "Utilization of Contingency Tables in Stochastic Computing")),
        ("scc-chapter", ("SCC 章", "SCC_Stochastic_Cross_Correlation.pdf", "SCC: Stochastic Cross Correlation")),
        ("zce", ("ZCE", "Zero_Correlation_Error_A_Metric_for_Finite-Length_Bitstream_Independence_in_Stochastic_Computing.pdf", "Zero Correlation Error: A Metric for Finite-Length Bitstream Independence in Stochastic Computing")),
        ("isolation", ("Isolation", "Isolation-based_decorrelation_of_stochastic_circuits.pdf", "Isolation-Based Decorrelation of Stochastic Circuits")),
        ("mc-fsm", ("MC-FSM", "A_Markov_Chain_Framework_for_Modeling_the_Statistical_Properties_of_Stochastic_Computing_Finite-State_Machines.pdf", "A Markov Chain Framework for Modeling the Statistical Properties of Stochastic Computing Finite-State Machines")),
        ("hypergeo", ("Hypergeo", "The_Hypergeometric_Distribution_as_a_More_Accurate_Model_for_Stochastic_Computing.pdf", "The Hypergeometric Distribution as a More Accurate Model for Stochastic Computing")),
        ("bayesian", ("Bayesian", "Bayesian_Accuracy_Analysis_of_Stochastic_Circuits.pdf", "Bayesian Accuracy Analysis of Stochastic Circuits")),
        ("varprop", ("VarProp", "Understanding_variance_propagation_in_stochastic_computing_systems.pdf", "Understanding Variance Propagation in Stochastic Computing Systems")),
        ("limits", ("Limits", "On_the_Limits_of_Stochastic_Computing.pdf", "On the Limits of Stochastic Computing")),
        ("qian-syn", ("Qian-Syn", "Qian_Riedel_Synthesizing_Logical_Computation_on_Stochastic_Bit_Streams.pdf", "Synthesizing Logical Computation on Stochastic Bit Streams")),
        ("li-fsm", ("Li-FSM", "Li_Qian_Riedel_Bazargan_Lilja_The_Synthesis_of_Linear_Finite_State_Machine_Based_Stochastic_Computational_Elements.pdf", "The Synthesis of Linear Finite State Machine-Based Stochastic Computational Elements")),
        ("cube", ("Cube", "Peng_Qian_Stochastic_Circuit_Synthesis_by_Cube_Assignment.pdf", "Stochastic Circuit Synthesis by Cube Assignment")),
        ("faulttol", ("FaultTol", "Qian_Li_Riedel_Bazargan_Lilja_An_Architecture_for_Fault_Tolerant_Computation_with_Stochastic_Logic.pdf", "An Architecture for Fault-Tolerant Computation with Stochastic Logic")),
        ("maxmin", ("MaxMin", "Stochastic_Computing_Max_amp_Min_Architectures_Using_Markov_Chains_Design_Analysis_and_Implementation.pdf", "Stochastic Computing Max & Min Architectures Using Markov Chains")),
        ("wiremult", ("WireMult", "Accurate_Stochastic_Computing_Using_a_Wire_Exchanging_Unipolar_Multiplier.pdf", "Accurate Stochastic Computing Using a Wire Exchanging Unipolar Multiplier")),
        ("accmult", ("AccMult", "Highly Accurate and Energy Efficient Stochastic Multipliers.pdf", "Highly Accurate and Energy Efficient Stochastic Multipliers")),
        ("bisc", ("BISC", "A_new_stochastic_computing_multiplier_with_application_to_deep_convolutional_neural_networks.pdf", "A New Stochastic Computing Multiplier with Application to Deep Convolutional Neural Networks")),
        ("skippynn", ("SkippyNN", "SkippyNN_An_Embedded_Stochastic-Computing_Accelerator_for_Convolutional_Neural_Networks.pdf", "SkippyNN: An Embedded Stochastic-Computing Accelerator for Convolutional Neural Networks")),
        ("dynamicdnn", ("DynamicDNN", "Dynamic_energy-accuracy_trade-off_using_stochastic_computing_in_deep_neural_networks.pdf", "Dynamic Energy-Accuracy Trade-Off Using Stochastic Computing in Deep Neural Networks")),
        ("detcnn", ("DetCNN", "Energy-Efficient_Convolutional_Neural_Networks_with_Deterministic_Bit-Stream_Processing.pdf", "Energy-Efficient Convolutional Neural Networks with Deterministic Bit-Stream Processing")),
        ("ldpc", ("LDPC", "Hybrid_Stochastic_LDPC_Decoder_With_Fully_Correlated_Stochastic_Computation.pdf", "Hybrid Stochastic LDPC Decoder with Fully Correlated Stochastic Computation")),
        ("dft", ("DFT", "Area-Efficient_Scaling-Free_DFT_FFT_Design_Using_Stochastic_Computing.pdf", "Area-Efficient Scaling-Free DFT/FFT Design Using Stochastic Computing")),
        ("sorting", ("Sorting", "Low-Cost_Sorting_Network_Circuits_Using_Unary_Processing.pdf", "Low-Cost Sorting Network Circuits Using Unary Processing")),
        ("edge", ("Edge", "Robert_edge_detection.pdf", "Stochastic Circuits for Real-Time Image-Processing Applications")),
        ("streaming", ("Streaming", "hsiao-aspdac2022.pdf", "Streaming Accuracy: Characterizing Early Termination in Stochastic Computing")),
        ("pp-cape", ("PP-CAPE", "Achieving_progressive_precision_in_stochastic_computing.pdf", "Achieving Progressive Precision in Stochastic Computing")),
        ("sbong", ("SBONG", "SBONG.pdf", "S-Box-Based Random Number Generation for Stochastic Computing")),
        ("agile", ("Agile", "Agile_Simulation _ TCAD 2023.pdf", "Agile Simulation of Stochastic Computing Image Processing with Contingency Tables")),
        ("bitshuffle", ("BitShuffle", "Bit_Shuffle.pdf", "Design Space Exploration for Hardware-Efficient Stochastic Computing: A Case Study on DCT")),
        ("ecg", ("ECG", "ECG_CZH.pdf", "Miniaturization of Insertable Cardiac Monitor: ECG Signal Processing Based on Stochastic Computing")),
        ("bipolar-lfsr", ("Bipolar-LFSR", "A_Bipolar_Stochastic_Bitstream_Generator_Based_on_Linear_Feedback_Shift_Register.pdf", "A Bipolar Stochastic Bitstream Generator Based on Linear Feedback Shift Register")),
        ("eco", ("ECO", "ECO_Enhanced_In-Stream_Correlation_Manipulation_for_Low-Discrepancy_Stochastic_Computing.pdf", "ECO: Enhanced In-Stream Correlation Manipulation for Low-Discrepancy Stochastic Computing")),
        ("cmpfree", ("CmpFree", "DAC63849.2025.11132702.pdf", "Comparison-Free Bit-Stream Generation for Cost-Efficient Unary Computing")),
        ("area-dft", ("Arrea-DFT", "Arrea_Resilient.pdf", "Area-Efficient Error-Resilient Discrete Fourier Transformation Design Using Stochastic Computing")),
    ]
)

DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


def run_text(*args: str) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True, errors="replace")
    return result.stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pdf_pages(path: Path) -> int:
    for line in run_text("pdfinfo", str(path)).splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError(f"No page count found in {path}")


def pdf_text(path: Path, first_pages_only: bool = False) -> str:
    args = ["pdftotext"]
    if first_pages_only:
        args.extend(["-f", "1", "-l", "2"])
    args.extend([str(path), "-"])
    return run_text(*args)


def extract_doi(text: str) -> str | None:
    match = DOI_PATTERN.search(text)
    if not match:
        return None
    return match.group(0).rstrip(".,;:)").lower()


def build_manifest(corpus_dir: Path) -> list[dict[str, object]]:
    actual = {path.name: path for path in corpus_dir.glob("*.pdf")}
    expected = {filename for _, filename, _ in PAPERS.values()}
    missing = sorted(expected - actual.keys())
    unexpected = sorted(actual.keys() - expected)
    if missing or unexpected:
        raise SystemExit(f"Corpus mapping mismatch. missing={missing}, unexpected={unexpected}")

    records: list[dict[str, object]] = []
    seen_hashes: set[str] = set()
    for paper_id, (alias, filename, title) in PAPERS.items():
        path = actual[filename]
        digest = sha256(path)
        if digest in seen_hashes:
            raise SystemExit(f"Duplicate content remains in cleaned corpus: {filename}")
        seen_hashes.add(digest)
        text = pdf_text(path)
        records.append(
            {
                "paper_id": paper_id,
                "alias": alias,
                "title": title,
                "filename": filename,
                "relative_path": f"data/raw/papers/{filename}",
                "sha256": digest,
                "size_bytes": path.stat().st_size,
                "pages": pdf_pages(path),
                "text_characters": len(text),
                "doi": extract_doi(pdf_text(path, first_pages_only=True)),
                "metadata_status": "corpus_verified",
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/raw/papers"))
    parser.add_argument("--output", type=Path, default=Path("data/metadata/papers.jsonl"))
    args = parser.parse_args()

    records = build_manifest(args.corpus_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(records)} unique paper records to {args.output}")


if __name__ == "__main__":
    main()
