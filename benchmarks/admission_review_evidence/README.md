# Admission review evidence

Existing experiment artifacts for PR #1, kept outside its code diff. See manifest.json for provenance limits, metric definitions and checksums.

summary.json contains the complete load and static-cap comparison. raw/ contains compressed JSON for each selected run, including request timing and status. Both profiles, deployment configurations and original shell commands are included. unset-results contains the GPU/order comparison; admission-overhead.json contains the coordinator microbenchmark.

The original raw fields late_rate and offered_span_s mean late responses divided by all offered requests, and elapsed time including drain, respectively. The corrected benchmark names these late_rate_of_offered and total_span_s, and additionally reports late_rate_of_responses.
