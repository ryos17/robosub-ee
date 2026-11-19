# Research

Research work for the RoboSub EE team — exploratory projects that aren't part
of the production vehicle firmware/hardware yet.

- **[`whisper_ivc/`](whisper_ivc/)** — underwater acoustic intervehicle
  communication: streaming hydrophone audio from the vehicle to the Orin and
  transcribing it live with Whisper, as a channel for one sub to "talk" to
  another. Working.
- **[`uw_speech_enhancement/`](uw_speech_enhancement/)** — sim-to-real speech
  enhancement and Whisper adaptation for hydrophone speech: an underwater
  channel simulator (`uwsim`), fine-tuned front ends and a LoRA-adapted
  Whisper (C8), a no-speech gate against invented words, and the protocol
  and tooling for the first pool recording session (S1). Simulation results
  only so far; see its `README.md` and `FINDINGS.md`.

_Last updated: 2026-11-18_
