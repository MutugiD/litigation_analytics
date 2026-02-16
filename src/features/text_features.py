"""Text-based feature engineering using sentence-transformers and TF-IDF.

Uses general-purpose sentence-transformers (not Legal-BERT) because:
- Legal-BERT was trained on EU/UK/US legal text, not Kenyan law
- Kenyan legal English mixes common law terms with Kenyan statutory refs
- General models (all-MiniLM-L6-v2) are well-tested and lighter

Features generated:
1. Sentence-transformer embeddings (384-dim) -> PCA-reduced to 30 components
2. TF-IDF on legal keyword vocabulary

Usage:
    from src.features.text_features import TextFeatureBuilder
    builder = TextFeatureBuilder()
    embeddings_df = builder.build_embeddings(texts_dict)
    tfidf_df = builder.build_tfidf(texts_dict)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from configs.settings import settings

logger = logging.getLogger(__name__)

# Key legal terms for TF-IDF feature extraction
# These are terms frequently appearing in Kenyan commercial litigation
LEGAL_VOCABULARY = [
    # Procedural terms
    "injunction", "interlocutory", "restraining order", "stay of execution",
    "leave to appeal", "security for costs", "striking out", "summary judgment",
    "default judgment", "consent order", "adjournment", "joinder",

    # Commercial law
    "breach of contract", "specific performance", "damages", "indemnity",
    "liquidated damages", "quantum meruit", "unjust enrichment", "estoppel",
    "fiduciary duty", "negligence", "misrepresentation", "fraud",

    # Data protection (DPA 2019)
    "data protection", "personal data", "data subject", "data controller",
    "data processor", "consent", "data breach", "privacy",
    "data commissioner", "right to erasure", "data transfer",

    # Kenyan statutory references
    "civil procedure act", "evidence act", "companies act",
    "insolvency act", "arbitration act", "law of contract act",
    "constitution of kenya", "bill of rights", "fair hearing",
    "data protection act",

    # Outcome-related terms
    "prima facie", "balance of convenience", "irreparable harm",
    "triable issue", "arguable case", "status quo",
    "with costs", "without costs", "each party to bear",
]


class TextFeatureBuilder:
    """Builds text-based features from judgment texts."""

    def __init__(
        self,
        model_name: str | None = None,
        pca_components: int | None = None,
    ):
        self.model_name = model_name or settings.model.sentence_transformer_model
        self.pca_components = pca_components or settings.model.text_pca_components
        self._embedder = None
        self._pca = None
        self._tfidf = None

    @property
    def embedder(self):
        """Lazy-load the sentence transformer model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformer model: %s", self.model_name)
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder

    def _truncate_text(self, text: str, max_chars: int = 10_000) -> str:
        """Truncate text to fit within model's context window.

        Keeps the BEGINNING of the judgment (which contains the issues,
        facts, and party descriptions) as these are most predictive.
        """
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def build_embeddings(
        self,
        texts: dict[str, str],
        fit_pca: bool = True,
    ) -> pd.DataFrame:
        """Generate PCA-reduced sentence-transformer embeddings.

        Args:
            texts: Dict mapping case_id -> judgment text
            fit_pca: If True, fit PCA on this data. If False, use existing PCA.

        Returns:
            DataFrame with PCA-reduced embedding columns, indexed by case_id.
        """
        case_ids = list(texts.keys())
        raw_texts = [self._truncate_text(texts[cid]) for cid in case_ids]

        logger.info("Encoding %d texts with %s...", len(raw_texts), self.model_name)
        embeddings = self.embedder.encode(
            raw_texts,
            show_progress_bar=True,
            batch_size=32,
        )
        logger.info("Raw embeddings shape: %s", embeddings.shape)

        # PCA reduction
        if fit_pca or self._pca is None:
            n_components = min(self.pca_components, len(case_ids), embeddings.shape[1])
            self._pca = PCA(n_components=n_components, random_state=42)
            reduced = self._pca.fit_transform(embeddings)
            explained = sum(self._pca.explained_variance_ratio_)
            logger.info(
                "PCA: %d -> %d components (%.1f%% variance explained)",
                embeddings.shape[1], n_components, 100 * explained,
            )
        else:
            reduced = self._pca.transform(embeddings)

        columns = [f"emb_pca_{i}" for i in range(reduced.shape[1])]
        return pd.DataFrame(reduced, index=case_ids, columns=columns)

    def build_tfidf(
        self,
        texts: dict[str, str],
        max_features: int = 100,
        fit: bool = True,
    ) -> pd.DataFrame:
        """Build TF-IDF features using legal vocabulary.

        Args:
            texts: Dict mapping case_id -> judgment text
            max_features: Maximum number of TF-IDF features
            fit: If True, fit vectorizer. If False, use existing.

        Returns:
            DataFrame with TF-IDF features, indexed by case_id.
        """
        case_ids = list(texts.keys())
        raw_texts = [texts[cid] for cid in case_ids]

        if fit or self._tfidf is None:
            self._tfidf = TfidfVectorizer(
                vocabulary=LEGAL_VOCABULARY[:max_features],
                lowercase=True,
                ngram_range=(1, 3),
                sublinear_tf=True,
            )
            matrix = self._tfidf.fit_transform(raw_texts)
        else:
            matrix = self._tfidf.transform(raw_texts)

        feature_names = [f"tfidf_{name.replace(' ', '_')}" for name in self._tfidf.get_feature_names_out()]
        return pd.DataFrame(
            matrix.toarray(),
            index=case_ids,
            columns=feature_names,
        )

    def build_all(
        self,
        texts: dict[str, str],
        include_tfidf: bool = True,
    ) -> pd.DataFrame:
        """Build all text features (embeddings + TF-IDF).

        Returns:
            Combined DataFrame indexed by case_id.
        """
        embeddings_df = self.build_embeddings(texts)

        if include_tfidf:
            tfidf_df = self.build_tfidf(texts)
            return pd.concat([embeddings_df, tfidf_df], axis=1)

        return embeddings_df
