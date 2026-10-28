"""Real-benchmark tooling (RESEARCH_PLAN.md 5.1, 5.4, 6.5, 6.6, 6.8).

schema      metadata.jsonl row dataclasses, validation, JSONL I/O (6.8)
librispeech manifest building and v0/v1 utterance selection (5.1)
splits      speaker/utterance/noise/position disjointness checks (5.4)
playlist    playback blocks, LFM sync chirp, ABBA schedule, budget (6.5, 6.6)
"""

from uwse.bench.librispeech import (Utterance, build_manifest, read_manifest, select_v0_dev,
                                    select_v0_test, select_v1_test, write_manifest)
from uwse.bench.playlist import (BlockLayout, ScheduleEntry, abba_states, block_timing,
                                 lfm_chirp, overhead_fraction, pack_blocks, playback_budget,
                                 render_block, render_blocks, session_schedule)
from uwse.bench.schema import (MetadataRow, SchemaError, make_utt_id, parse_utt_id, read_jsonl,
                               validate, validation_errors, write_jsonl)
from uwse.bench.splits import (SplitLeakError, assert_disjoint, assert_noise_split,
                               assert_positions_disjoint)
