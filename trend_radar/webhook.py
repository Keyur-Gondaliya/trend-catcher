"""
Twilio inbound-feedback webhook -- closes the loop in production.

In the prototype, feedback comes via `python run.py feedback "2 yes, 1 no"`.
In production the same reply arrives as a WhatsApp message: Twilio POSTs it to
this endpoint, we parse it with the exact same `parse_feedback`, fold it into
the preference store via `apply_feedback`, and reply with a TwiML confirmation.

Run it:
    python run.py webhook                 # dev server on :5000
    gunicorn 'trend_radar.webhook:app'    # production

Then point your Twilio WhatsApp sandbox's "When a message comes in" webhook at
    https://<your-host>/whatsapp
"""
from trend_radar.pipeline import apply_feedback

# Reuse the CLI parser so the two feedback paths can never drift apart.
from run import parse_feedback


def _build_app():
    """Flask app factory. Flask is an optional dep (whatsapp extra), so it's
    imported lazily -- the rest of the system runs without it."""
    from flask import Flask, request, Response

    app = Flask(__name__)

    @app.post("/whatsapp")
    def whatsapp_reply():
        body = (request.form.get("Body") or "").strip()
        fb = parse_feedback(body)
        if not fb:
            reply = ("Didn't catch that. Reply like `2 yes, 4 no` "
                     "to tune your next digest.")
        else:
            apply_feedback(fb)
            yes = sum(1 for v in fb.values() if v)
            reply = (f"Got it — folded {len(fb)} signal(s) "
                     f"({yes} 👍 / {len(fb) - yes} 👎). "
                     "Your next digest will reflect this.")
        twiml = f"<?xml version='1.0' encoding='UTF-8'?><Response><Message>{reply}</Message></Response>"
        return Response(twiml, mimetype="application/xml")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


# Module-level app so `gunicorn trend_radar.webhook:app` works.
try:
    app = _build_app()
except ImportError:  # Flask not installed (whatsapp extra not selected)
    app = None


def serve(host="0.0.0.0", port=5000):
    if app is None:
        raise SystemExit(
            "Flask is not installed. Install the WhatsApp extra:\n"
            "    pip install '.[whatsapp]'"
        )
    app.run(host=host, port=port)
