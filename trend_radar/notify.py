"""
Delivery layer -- the front end.

Default "console" prints the digest so the whole thing runs with no credentials.
The "whatsapp" backend sends via Twilio's WhatsApp API to your number; the
incoming feedback reply is handled by a webhook (see notes in README). The
interface is identical, so the pipeline doesn't know or care which is active.
"""
from trend_radar import config


def deliver(text: str):
    if config.DELIVERY_BACKEND == "console":
        print("\n" + "=" * 60)
        print(text)
        print("=" * 60 + "\n")
        return

    if config.DELIVERY_BACKEND == "whatsapp":
        import os
        from twilio.rest import Client
        client = Client(os.environ["TWILIO_SID"], os.environ["TWILIO_TOKEN"])
        client.messages.create(
            from_=f"whatsapp:{os.environ['TWILIO_WHATSAPP_FROM']}",
            to=f"whatsapp:{os.environ['MY_WHATSAPP_NUMBER']}",
            body=text,
        )
        print("Digest sent to WhatsApp.")
        return

    raise ValueError(f"Unknown DELIVERY_BACKEND: {config.DELIVERY_BACKEND}")
