import pandas as pd
import os

from .text_preprocessing import clean_text
from .sentiment_analysis import get_sentiment_score
from .keywords import extract_keywords
from .topic_modeling import run_lda


def calculate_engagement(row):
    likes = row["likes"]
    replies = row["replies"]
    score = (likes * 0.7 + replies * 0.3) / 5000
    return round(score, 3)


def run_nlp_pipeline():

    print("Loading dataset...")

    df = pd.read_csv("data/raw_data/nlp_data/fashion_trend_dataset_2000_nlp_sithin.csv")

    print("Cleaning text...")
    df["clean_text"] = df["text"].astype(str).apply(clean_text)
    df["text_length"] = df["clean_text"].apply(lambda x: len(x.split()))
    df["hashtag_count"] = df["hashtags"].apply(lambda x: len(str(x).split()))

    print("Running sentiment analysis...")
    df["sentiment_score"] = df["clean_text"].apply(get_sentiment_score)

    print("Extracting keywords...")
    keywords = extract_keywords(df["clean_text"].tolist())
    keyword_list = [k[0] for k in keywords]

    df["keyword_list"] = df["clean_text"].apply(
        lambda x: ",".join([w for w in keyword_list if w in x][:5])
    )

    df["keyword_count"] = df["keyword_list"].apply(
        lambda x: len(x.split(",")) if x else 0
    )

    print("Running topic modeling...")
    topics, topic_matrix = run_lda(df["clean_text"].tolist())
    df["topic_id"] = topic_matrix.argmax(axis=1)

    print("Calculating engagement score...")
    df["engagement_score"] = df.apply(calculate_engagement, axis=1)

    print("Calculating trend score...")
    df["trend_score"] = (
        0.35 * df["sentiment_score"]
        + 0.35 * df["engagement_score"]
        + 0.30 * df["keyword_count"]
    )

    print("Saving processed dataset...")
    df.to_csv("data/processed_data/nlp_data/nlp_features.csv", index=False)

    print("NLP pipeline completed.")

    # -------------------------------------------------
    # Generate NLP Trend Summary Report
    # -------------------------------------------------

    report_lines = []

    def log(line=""):
        print(line)
        report_lines.append(line)

    log("\nNLP Trend Summary")
    log("-----------------\n")

    log(f"Total posts analyzed : {len(df)}")
    log(f"Average sentiment score : {df['sentiment_score'].mean():.3f}")
    log(f"Average engagement score: {df['engagement_score'].mean():.3f}")

    log("\nSentiment Score Interpretation")
    log("-1.0 – -0.3 : Negative discussion")
    log("-0.3 – 0.3  : Neutral discussion")
    log("0.3 – 1.0   : Positive discussion")

    log("\nEngagement Score Interpretation")
    log("0.0 – 0.2  : Low engagement")
    log("0.2 – 0.5  : Moderate engagement")
    log("0.5 – 1.0  : High engagement")

    log("\nTop trending keywords")

    top_keywords = (
        df["keyword_list"]
        .str.split(",")
        .explode()
        .value_counts()
        .head(10)
    )

    for word, count in top_keywords.items():
        log(f"{word}: {count}")

    log("\nTop discussion topics")

    top_topics = df["topic_id"].value_counts().head(5)

    for topic, count in top_topics.items():

        try:
            topic_words = topics[topic]

            if isinstance(topic_words, dict):
                topic_words = list(topic_words.keys())

            if isinstance(topic_words, (list, tuple)):
                topic_words = topic_words[:5]
            else:
                topic_words = list(topic_words)[:5]

            topic_label = ", ".join(map(str, topic_words))

        except Exception:
            topic_label = "keywords unavailable"

        log(f"Topic {topic} ({topic_label}): {count} posts")

    output_path = "reports/nlp_trend_summary.txt"

    os.makedirs("reports", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in report_lines:
            f.write(line + "\n")

    log(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    run_nlp_pipeline()