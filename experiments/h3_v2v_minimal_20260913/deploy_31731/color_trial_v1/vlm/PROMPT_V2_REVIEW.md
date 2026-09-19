# Action-timeline prompt v2: development correction, not an accuracy result

On 2026-09-14, actual run `7b7b9babe0ed44c68e1cb54c6a54ef9a` completed full video-conditioned generation on 31731. It returned `change`, confidence 1.0, with order/speed/duration unchanged and events changed. Its reason counted a shirt-color attribute modification as an event, despite the original prompt explicitly excluding appearance-only edits. This is a semantic classification error under the specified action-timeline task; JSON validation correctly preserved the actual answer. The result is retained and is not admitted to the fixed same-frame trial.

The one planned v2 development attempt makes the existing definition prominent and requests concrete action-timeline evidence for each changed effect. It distinguishes static attribute modification from introducing an action, retains mixed appearance-plus-temporal changes as `change`, and allows `uncertain` where sampled frames are insufficient. The parser, video sampling, edit instruction, model, generation settings, ABC algorithms and acceptance rule remain unchanged. It is not a retry-until-preserve loop. A failure, `change` or `uncertain` requires review, not label substitution or another automatic prompt retry.

Predeclared semantic checks for a later held-out real-model evaluation (not executed by CPU tests, not labels fed to this run):

- Recolor clothing with actions unchanged: preserve; recolor clothing and repeat an existing embrace: change.
- Change background with actions unchanged: preserve; change background and make a previously single wave occur twice: change.
- Recolor an already-worn garment: preserve; insert putting on that garment before a later action: change.
- The same instruction "wave then clap" on sources already showing that order vs showing clap then wave: preserve vs change, provided the actions are visible.
- An instruction requiring timing/order evidence absent from the sampled or visible frames: uncertain is permissible; missing evidence must not be forced into preserve.

These development examples and this one color case cannot establish held-out router accuracy or show that video input is necessary. The source-dependent paired-video case and uncertainty cases need separate real evaluation. No reverse-playback generation is requested.
