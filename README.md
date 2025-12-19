________________________________________
EMMA — Empathetic Memory-Augmented Multi-layer Assistant (Research Prototype)
Empathetic, privacy-aware memory for psychologically informed conversational agents.
Reference implementation of a mobile-friendly memory-augmented assistant inspired by the EMMA architecture (session / episodic / semantic memory, query classification, LlamaIndex retrieval, Gradio demo). This repository contains the code, data processing scripts, retrieval/indexing, the classifier, and evaluation tooling used for the research prototype.
⚠️ Research prototype — not a clinical tool. Use for research, experiments, and controlled simulations only. See Limitations & Safety below.
________________________________________
Table of contents
•	Key features
•	Architecture (high level)
•	Evaluation & metrics
•	Limitations & safety
•	License
•	Citation
________________________________________
Key features
•	Three-tier memory architecture: session (raw transcripts), episodic (session summaries), semantic (long-term traits/values).
•	Dynamic query classifier: routes queries to Episodic, Semantic, Hybrid, or General handlers.
•	On-device / privacy-aware retrieval: LlamaIndex (vector index) for fast local semantic search.
•	Prompt templates for therapy-aligned response generation: blends retrieved memory with therapist-inspired prompt scaffolds.
•	Gradio-based demo UI: lightweight chat interface with panels for memory summaries and session history.
•	Evaluation tooling: scripts for quantitative memory retrieval accuracy and qualitative Likert-based evaluation pipelines.
(Descriptions follow the EMMA paper’s design and evaluation; see Citation).
________________________________________
Architecture (high level)
Pic here
•	Indexing: episodic and semantic items are embedded and stored in vector indexes.
•	Routing: classifier determines which layer(s) to query. Hybrid queries can merge episodic + semantic retrieval.
•	Prompting: retrieved memory is merged into therapy-aware prompt templates before sending to the LLM.
________________________________________

Evaluation & metrics
The original prototype evaluation included:
•	Qualitative: 90 prompts rated on a 5-point Likert scale for Personalization, Continuity, Empathy.
•	Quantitative: Memory retrieval accuracy computed as normalized mean of Likert scores (5-point).
•	Automatic metrics: Automated rubric-based assessment via a stronger LLM evaluator.
You can reproduce evaluation by preparing:
•	A test set of prompts tied to memory entries.
•	Scripts to compare generated responses to ground-truth memory entries and to compute Likert-aligned scores.
________________________________________
Limitations & safety
•	Not a clinical or diagnostic system. The prototype demonstrates research concepts and must not replace professional mental health providers.
•	Automated evaluators and LLM judgments can be noisy or biased. For safety-critical claims, include clinician review and human-in-the-loop validation.
•	The system may occasionally hallucinate memory-grounded facts — always log retrieval traces used by the prompt to help auditing and debugging.
•	See the paper for a discussion of limitations and evaluation methodology.
________________________________________

License
Suggested: MIT License. Replace with your preferred license if required.
________________________________________

