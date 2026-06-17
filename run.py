"""
Trend Radar CLI.

  python run.py seed                 # generate the sample X dataset
  python run.py digest               # run a digest (Wed/Sun job)
  python run.py feedback "1 yes, 3 no, 4 yes"   # tune preferences
  python run.py reset                # wipe learned preferences
  python run.py webhook              # serve the Twilio inbound-feedback webhook

The scheduler (cron, twice weekly) just calls `python run.py digest`. The
feedback command stands in for the WhatsApp reply webhook locally; `webhook`
runs the real Twilio endpoint that does the same thing from an inbound message.
"""
import sys
import re

from trend_radar.pipeline import run_digest, apply_feedback


def parse_feedback(s: str) -> dict[int, bool]:
    out = {}
    for m in re.finditer(r"(\d+)\s*(yes|no|y|n)", s.lower()):
        out[int(m.group(1))] = m.group(2) in ("yes", "y")
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "digest"

    if cmd == "seed":
        from trend_radar.sample_data import main as gen
        from trend_radar import config
        gen(config.DATA_PATH)
    elif cmd == "digest":
        run_digest()
    elif cmd == "feedback":
        fb = parse_feedback(sys.argv[2] if len(sys.argv) > 2 else "")
        if not fb:
            print("Nothing parsed. Try: feedback \"1 yes, 3 no\"")
            return
        apply_feedback(fb)
    elif cmd == "reset":
        import os
        from trend_radar import config
        for p in (config.PREFERENCE_PATH, config.DIGEST_PATH):
            if os.path.exists(p):
                os.remove(p)
        print("Preferences reset.")
    elif cmd == "webhook":
        from trend_radar.webhook import serve
        serve()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
