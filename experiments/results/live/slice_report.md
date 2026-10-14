## Per scenario (each run)
| decider | scenario | runs | outcome | wall s | text_to_hold_s | hand_to_pause_s | leave_to_resume_s | wait_to_pause_s | go_to_resume_s | stop_to_parked_s | first plan ms | replan ms | user-text right_now |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decisions | correction | 3 | done, done, done | 32.1, 34.7, 31.5 | 0.156, 1.115, 0.121 | - | - | - | - | - | 2196, 2408, 2742 | 4209, 2547, 4699, 2804, 3789, 1967 | hold, hold, hold |
| decisions | hand | 3 | done, done, done | 31.1, 27.5, 34.5 | - | 0.150, 0.146, 0.149 | 0.104, 0.144, 0.094 | - | - | - | 2632, 2886, 2981 | 4012, 3579, 3871, 4021, 3277, 3710, 5586, 5212, 2025, 2303 | - |
| decisions | sort | 3 | done, done, done | 25.8, 26.4, 25.8 | - | - | - | - | - | - | 2512, 3259, 2766 | - | - |
| decisions | stop | 3 | stopped, stopped, stopped | 6.1, 7.2, 6.1 | - | - | - | - | - | 2.540, 2.570, 2.450 | 2738, 3863, 2765 | - | - |
| decisions | wait_go | 3 | done, done, done | 33.6, 27.9, 27.3 | - | - | - | 0.213, 0.101, 0.153 | 0.110, 0.103, 0.154 | - | 2314, 2741, 2213 | 5518, 3632, 3915, 6297, 4331, 3421, 5076, 3115 | pause,resume, pause,resume, pause,resume |
| jev | correction | 3 | done, done, done | 36.0, 34.7, 35.2 | none, none, none | - | - | - | - | - | 2901, 2260, 2616 | 4053, 3802, 6315, 3119, 6493, 3247 | carry_on, carry_on, carry_on |
| jev | hand | 3 | done, done, done | 27.2, 27.8, 26.8 | - | 0.097, 0.094, 0.091 | 0.118, 0.096, 0.088 | - | - | - | 2678, 3254, 2312 | 2936 | - |
| jev | sort | 3 | done, done, done | 27.1, 26.0, 25.9 | - | - | - | - | - | - | 4119, 2996, 2862 | - | - |
| jev | stop | 3 | stopped, stopped, stopped | 6.8, 5.9, 6.9 | - | - | - | - | - | 2.470, 2.530, 2.450 | 3403, 2540, 3575 | - | - |
| jev | wait_go | 3 | done, done, done | 27.7, 27.8, 27.6 | - | - | - | 0.141, 0.115, 0.093 | 0.110, 0.160, 0.080 | - | 2719, 2780, 2421 | 4276 | pause,resume, pause,resume, pause,resume |

## Decision latency in the loop (ms, every decision the brain acted on)
| decider | n | p50 | p95 | max | shadow backend | shadow n | shadow p50 | shadow p95 |
|---|---|---|---|---|---|---|---|---|
| decisions | 84 | 129 | 254 | 1097 | jev | 84 | 102 | 141 |
| jev | 77 | 101 | 130 | 156 | decisions | 77 | 126 | 372 |

## The same slice events, both backends (acting + shadow), against the oracle where it judged
| group | n paired | agree | Jev right | Decisions right | n with truth |
|---|---|---|---|---|---|
| right_now | 161 | 96% | 96% | 98% | 123 |
| in_plan_fix | 114 | 99% | 99% | 100% | 84 |
| route | 161 | 94% | 95% | 95% | 123 |

## Disagreements on user text and hands (slice)
- decisions correction 'the user typed: "actually, highest on the left"': right_now jev carry_on / dec hold (truth ['hold']); route jev stay_local / dec ask_user (truth ['fast_llm', 'capable_llm'])
- decisions correction 'the user typed: "actually, highest on the left"': right_now jev carry_on / dec hold (truth ['hold']); route jev stay_local / dec fast_llm (truth ['fast_llm', 'capable_llm'])
- decisions correction 'the user typed: "actually, highest on the left"': right_now jev carry_on / dec hold (truth ['hold']); route jev stay_local / dec ask_user (truth ['fast_llm', 'capable_llm'])
- jev       correction 'the user typed: "actually, highest on the left"': right_now jev carry_on / dec hold (truth None); route jev stay_local / dec ask_user (truth None)
- jev       correction 'the user typed: "actually, highest on the left"': right_now jev carry_on / dec hold (truth None); route jev stay_local / dec fast_llm (truth None)
- jev       correction 'the user typed: "actually, highest on the left"': right_now jev carry_on / dec hold (truth ['hold']); route jev stay_local / dec fast_llm (truth ['fast_llm', 'capable_llm'])
- jev       wait_go  'the user typed: "hang on a second"': right_now jev pause / dec carry_on (truth ['pause']); route jev stay_local / dec stay_local (truth ['stay_local'])
