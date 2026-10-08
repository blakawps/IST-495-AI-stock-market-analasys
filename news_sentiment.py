import html
import re

from transformers import pipeline


MODEL_NAME = "ProsusAI/finbert"


print("Loading FinBERT sentiment model...")

sentiment_model = pipeline(
    "text-classification",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
)

print("FinBERT loaded.")


LABEL_TO_SCORE = {
    "positive": 1,
    "neutral": 0,
    "negative": -1,
}


LABEL_TO_MARKET = {
    "positive": "Bullish",
    "neutral": "Neutral",
    "negative": "Bearish",
}


def clean_text(text):

    if not text:
        return ""

    text = html.unescape(text)

    # Remove HTML tags from RSS summaries
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Remove excessive whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def analyze_news_sentiment(
    title,
    summary=""
):

    title = clean_text(title)
    summary = clean_text(summary)

    if summary:

        text = (
            f"Headline: {title}\n"
            f"Description: {summary}"
        )

    else:

        text = (
            f"Headline: {title}"
        )

    if not text:
        return {
            "score": 0,
            "label": "Neutral",
            "confidence": 0.0,
            "model": MODEL_NAME,
        }

    result = sentiment_model(
        text,
        truncation=True,
        max_length=512,
    )[0]

    raw_label = (
        result["label"]
        .strip()
        .lower()
    )

    confidence = float(
        result["score"]
    )

    score = LABEL_TO_SCORE.get(
        raw_label,
        0
    )

    market_label = LABEL_TO_MARKET.get(
        raw_label,
        "Neutral"
    )

    return {
        "score": score,
        "label": market_label,
        "confidence": confidence,
        "model": MODEL_NAME,
    }


if __name__ == "__main__":

    tests = [

        (
            "Nvidia beats earnings estimates "
            "and raises guidance",
            ""
        ),

        (
            "Company files for bankruptcy "
            "after sharp revenue decline",
            ""
        ),

        (
            "Microsoft announces quarterly "
            "shareholder meeting",
            ""
        ),
    ]

    for title, summary in tests:

        result = analyze_news_sentiment(
            title,
            summary
        )

        print()
        print(title)
        print(result)